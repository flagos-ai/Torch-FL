# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Iluvatar: keep flag_gems' non-inner reductions off the tiled row-masked kernel.

**This is a correctness workaround, not a feature.** On the Iluvatar backend the
2-D tile load that flag_gems' ``*_non_inner`` kernels use does not honour the
row half of its mask. The kernel loads a ``(TILE_N, TILE_K)`` tile with
``mask = (n < N) & (k < K)``; rows past ``N`` are read anyway. Two consequences,
both silent:

* when ``TILE_N > N`` the reduction covers ``TILE_N`` rows instead of ``N``, so
  the answer is simply wrong, and
* for the last group of rows the read runs past the end of the tensor into
  whatever the allocator last handed out, so identical calls in one process
  return different numbers.

Reduced on a BI-V150 (flag_gems 5.4.0, torch_fl 2.10.0+corex4.5.0, FlagTree
0.6.2a3). A standalone Triton kernel with the same tile-and-mask reproduces it
without flag_gems, which is what makes this a backend defect rather than a
flag_gems one: an ``N=4`` reduction over a ``(4, 128)`` tile masked to 4 of 8
rows returns the sum of all 8 (28 where 6 is correct), with ``other=0.0`` and
with an explicit ``tl.where`` alike, while the same kernel with ``TILE_N == N``,
with a 1-D mask, or with a per-row loop is correct. Driving the same kernel with
vendor CoreX torch on the same box and the same Triton package gives the right
answer, so the difference is in the launch/compile path torch_fl installs.

The reason this matters beyond a single op is the shape. A Qwen3.5-VL vision
tower computes its bilinear position embedding as a weighted sum over the four
neighbouring grid points -- ``aten::sum.dim_IntList`` on a ``(556, 4, 1024)``
tensor with ``dim=1``. Traced operator by operator, that sum is the first and
only operator whose output differs between two identical forwards (its inputs
are bit-identical; 1064 later ops differ only downstream of it). Against a CPU
float64 reference it is off by 4.68 with an input absmax of 3.14. Repeated
forwards of the vision tower on the same fixed ``pixel_values`` produced four
distinct results, while vendor torch produced one, bit-identical across six.
The same tile is used by ``prod``, ``std``, ``logsumexp`` and the non-inner
``softmax``/``log_softmax``, which are wrong for the same reason.

The inner-dim kernels are correct -- their mask is 1-D, and the reduced axis is
the one covered by ``TILE_K`` -- so the workaround moves the reduced axis to the
back and calls the original op there. ``softmax``/``log_softmax`` permute the
result back; the reductions reshape to the shape the caller asked for. Only a
single non-inner ``int`` dim takes this path; everything else calls the original
unchanged, so this costs one transpose only on the calls that were wrong.

Iluvatar only. The miscompile is a property of that backend's masked load; on a
platform whose backend applies the mask this wrapper would add a transpose to
every non-inner reduction and buy nothing.

Remove this once the Iluvatar backend honours the row mask -- the reproduction
in the issue is the regression test.
"""

import sys
from typing import Any


# flag_gems entry point -> how the result is put back: "reduce" reshapes to the
# requested shape, "reduce_kwonly" is std (keyword-only keepdim), "softmax"
# permutes back.
_TARGETS = (
    ("sum_dim", "reduce"),
    ("prod_dim", "reduce"),
    ("std", "reduce_kwonly"),
    ("logsumexp", "reduce"),
    ("special_logsumexp", "reduce"),
    ("softmax", "softmax"),
    ("special_softmax", "softmax"),
    ("log_softmax", "softmax"),
    ("special_log_softmax", "softmax"),
)


def _single_non_inner_dim(x: Any, dim: Any):
    """(axis, passed_as_sequence) for a single non-trailing dim; (None, False) else.

    The aten schemas differ: ``sum.dim_IntList``/``logsumexp`` pass a list,
    ``std.correction`` and ``softmax`` pass an int. The inner call has to repeat
    whichever form the caller used, or flag_gems' own entry point rejects it.
    """
    if not hasattr(x, "dim") or x.dim() == 0:
        return None, False
    if isinstance(dim, int):
        axis, sequence = dim, False
    elif isinstance(dim, (list, tuple)) and len(dim) == 1 and isinstance(dim[0], int):
        axis, sequence = dim[0], True
    else:
        return None, False
    axis %= x.dim()
    if axis == x.dim() - 1:
        return None, False
    return axis, sequence


def _permutations(x: Any, d: int) -> tuple[list[int], list[int]]:
    """(perm, inverse) moving axis ``d`` to the back and back again."""
    perm = [i for i in range(x.dim()) if i != d] + [d]
    inverse = [0] * x.dim()
    for new, old in enumerate(perm):
        inverse[old] = new
    return perm, inverse


def _restore_reduced(y: Any, x: Any, d: int, keepdim: bool) -> Any:
    """Put a reduction result computed on the permuted view back in place."""
    if keepdim:
        shape = list(x.shape)
        shape[d] = 1
        return y.reshape(shape)
    return y


def wrap_reduction(orig, kind: str):
    """Wrap one flag_gems non-inner reduction entry point.

    ``kind`` is "reduce" (sum/prod/logsumexp: ``(inp, dim, keepdim, ...)``),
    "reduce_kwonly" (std, whose ``keepdim`` is keyword-only) or "softmax"
    (``(inp, dim, ...)``, result permuted back). Extra arguments are forwarded
    verbatim; ``keepdim`` is forced off on the inner call and re-applied by the
    reshape.
    """

    def patched(inp, dim=None, *args, **kwargs):
        d, sequence = _single_non_inner_dim(inp, dim)
        if d is None:
            return orig(inp, dim, *args, **kwargs)

        perm, inverse = _permutations(inp, d)
        # flag_gems' softmax/softmax kernels do not materialise the view
        # themselves (sum_dim_comm does), so hand them a contiguous one.
        moved = inp.permute(perm).contiguous()
        axis = [-1] if sequence else -1
        if kind == "softmax":
            return orig(moved, axis, *args, **kwargs).permute(inverse)

        # keepdim may arrive positionally (sum_dim, logsumexp) or by keyword
        # (std); force it off for the inner call, re-apply it in the reshape.
        keepdim = bool(kwargs.get("keepdim", False))
        if args and isinstance(args[0], bool):
            keepdim = args[0]
            args = args[1:]
        kwargs = {k: v for k, v in kwargs.items() if k != "keepdim"}
        if kind == "reduce_kwonly":
            # std's keepdim is keyword-only, so it cannot be passed positionally.
            out = orig(moved, axis, *args, **kwargs)
        else:
            out = orig(moved, axis, False, *args, **kwargs)
        return _restore_reduced(out, inp, d, keepdim)

    return patched


def patch_iluvatar_masked_reductions() -> bool:
    """Install the wrappers. Idempotent; False when there is nothing to patch."""
    try:
        import flag_gems
    except Exception:
        return False

    if getattr(flag_gems, "_flagos_iluvatar_reduction_wrappers", False):
        return False

    installed = False
    for package_attr, kind in _TARGETS:
        orig = getattr(flag_gems, package_attr, None)
        if orig is None:
            continue
        wrapped = wrap_reduction(orig, kind)
        # Bind the wrapper everywhere the original is reachable: the generated
        # kernels resolve the package-level name, flag_gems' own ops call their
        # siblings through the module, and some modules import a sibling by
        # name, so identity is the only reliable way to catch every alias.
        for name, val in list(vars(flag_gems).items()):
            if val is orig:
                setattr(flag_gems, name, wrapped)
        for mod in list(sys.modules.values()):
            if not getattr(mod, "__name__", "").startswith("flag_gems"):
                continue
            try:
                namespace = vars(mod)
            except Exception:
                continue
            for name, val in list(namespace.items()):
                if val is orig:
                    try:
                        setattr(mod, name, wrapped)
                    except Exception:
                        pass
        installed = True

    if installed:
        flag_gems._flagos_iluvatar_reduction_wrappers = True
    return installed
