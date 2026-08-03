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

"""Runtime for the generated TileOPs routes.

TileOPs ops are stateful Python objects whose constructors commit shape/dtype, and
TileLang rejects PrivateUse1 tensors outright. Both are handled here:

  - constructor args are derived from the aten call site per recipe, and instances
    are cached, because a cold constructor costs ~4 s (~54 ms warm) against a
    0.056 ms warm call;
  - flagos tensors are boxed to CUDA views with ``_flagos_to_cuda_view``, which
    only rewrites device metadata -- no copy, ~0.4 us -- and outputs are unboxed
    back. TileOPs allocates through the same caching allocator, so boxed outputs
    are already accounted for.

Calls prefer the compiled kernel over the ``Op`` wrapper, which adds 0.05-0.09 ms
of size-independent Python overhead -- enough to erase the gain on every
memory-bound op. The shortcut is verified per instance rather than assumed (see
``_Entry``), since L2 sometimes reshapes before dispatching. Set
``FLAGOS_TILEOPS_USE_L2=1`` to stay on ``Op.__call__`` when debugging.

See ``docs/tileops_codegen_design.md``.
"""

from __future__ import annotations

import importlib
import os
import warnings
from typing import Dict, Optional, Sequence, Tuple

import torch

from torch_fl.tileops_spec import BINARY, REDUCE, SOFTMAX, UNARY

__all__ = [
    "enable_tileops_for_flagos",
    "is_tileops_available",
    "registered_ops",
    "build_impl",
    "sample_inputs",
    "warmup",
]

# TileOPs only ships Hopper kernels.
_REQUIRED_CAPABILITY = (9, 0)

_instances: Dict[tuple, tuple] = {}
_registered: Dict[str, str] = {}
_library = None
_available: Optional[bool] = None


def _use_l2() -> bool:
    return os.environ.get("FLAGOS_TILEOPS_USE_L2") == "1"


def _disable_frontend_cache() -> bool:
    """Neutralize TileLang's *frontend* disk cache, keeping the kernel cache.

    Ops whose ctor commits a size (the whole UNARY recipe) otherwise reuse the
    first-compiled artifact across shapes and fail with ``violates packed ABI
    constraint; expected: <N>, got: <first N>``.

    The cause is not the kernel disk cache. TileOPs builds these kernels via
    ``functools.lru_cache``-d factories whose inner ``@tilelang.jit`` closure
    captures ``N`` as a free variable, and the frontend cache key
    (``JITImpl._frontend_cache_key_data``) covers only the explicit args -- for
    ReluFwdOp it is ``((('npt_arg', 8), ('threads_arg', 256)), None)`` for every
    N. Instrumenting ``load_frontend_cached`` shows it returning HIT for a
    different N; the per-``JITImpl`` ``_kernel_cache`` objects are distinct, so
    the leak is purely through this on-disk layer. ``GemmOp`` is immune because
    it passes ``(m, n, k)`` explicitly rather than closing over them.

    Upstream tile-ai/tilelang#2358, fixed by #2363 ("Remove frontend disk
    cache") on 2026-06-09 -- one day after 0.1.11 shipped, hence our pin needs
    this. Dropping the same path here is that fix, and it lets the kernel disk
    cache keep working: measured on H800 with three ReluFwdOp shapes, second
    process 0.3 s vs 12.0 s under a blanket ``TILELANG_DISABLE_CACHE=1``, both
    numerically correct. Delete this function when the pin reaches >=0.1.12.
    """
    if os.environ.get("TILELANG_DISABLE_CACHE") == "1":
        return False  # blanket disable already in force; nothing to patch
    patched = False
    _noop_load = lambda *a, **kw: None  # noqa: E731
    for name in ("tilelang.cache", "tilelang.jit"):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for attr in ("load_frontend_cached", "store_frontend_cache"):
            if hasattr(mod, attr):
                setattr(mod, attr, _noop_load)
                patched = True
    if not patched:
        # Future TileLang without these symbols: the bug is fixed upstream, so
        # there is nothing to work around.
        pass
    return patched


# --------------------------------------------------------------------------- #
# availability gate
# --------------------------------------------------------------------------- #
def is_tileops_available() -> bool:
    """True when TileOPs is importable and the host is SM90.

    Never raises: a missing optional dependency must not break ``import torch_fl``.
    """
    global _available
    if _available is not None:
        return _available

    _available = False

    # Escape hatch: kill every TileLang cache. Correct but slow (see
    # _disable_frontend_cache), kept for hosts where the targeted fix does not
    # apply. Must precede the tileops import -- TileLang reads it at import time.
    if os.environ.get("FLAGOS_TILEOPS_DISABLE_ALL_CACHE") == "1":
        os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")

    try:
        importlib.import_module("tileops")
    except Exception:
        return _available

    _disable_frontend_cache()

    try:
        if torch.cuda.get_device_capability() != _REQUIRED_CAPABILITY:
            warnings.warn(
                "TileOPs kernels target SM90 (Hopper) only; skipping registration on "
                f"capability {torch.cuda.get_device_capability()}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _available
    except Exception:
        return _available

    _available = True
    return _available


# --------------------------------------------------------------------------- #
# boxing
# --------------------------------------------------------------------------- #
def _box(t):
    """flagos tensor -> CUDA view sharing the same storage (no copy)."""
    if not isinstance(t, torch.Tensor) or t.device.type != "flagos":
        return t
    from torch_fl import _C

    return _C._flagos_to_cuda_view(t)


def _unbox(t, index: int):
    """CUDA tensor -> flagos view, for values handed back to the caller."""
    from torch_fl import _C

    return _C._cuda_to_flagos_view(t, index)


def _device_index(t: torch.Tensor) -> int:
    return t.device.index or 0


# --------------------------------------------------------------------------- #
# instance cache
# --------------------------------------------------------------------------- #
def _log_dispatch(overload: str, target: str) -> None:
    """Mirror the C++ FLAGOS_LOG_DISPATCH line for python-bound routes.

    Ops bound via torch.library on PrivateUse1 never reach the C++ dispatcher, so
    they are invisible to its logging. Without this, ``FLAGOS_LOG_DISPATCH=1``
    shows only the ops that fell through to cuda, which reads as if TileOPs were
    not routing anything at all.
    """
    if os.environ.get("FLAGOS_LOG_DISPATCH") == "1":
        print(f"[flagos dispatch] {overload} -> {target}", flush=True)


def _get(op_cls, ctor_key: tuple, *args, **kwargs):
    """Return the cached ``_Entry`` for these ctor args, building at most once.

    Keyed on the ctor args themselves, so a new shape or dtype gets its own
    instance while repeat calls stay on the warm path.
    """
    key = (op_cls, ctor_key)
    entry = _instances.get(key)
    if entry is None:
        entry = _instances[key] = _Entry(op_cls(*args, **kwargs))
    return entry


class _Entry:
    """A cached Op, plus its kernel once that shortcut is proven safe.

    Skipping the L2 wrapper is worth 0.05-0.09 ms per call, but it cannot be
    assumed: ``Op.forward`` may reshape or broadcast before handing off, so
    ``op.kernel`` does not necessarily accept the arguments ``Op.__call__`` does.
    Binary ops are exactly that case -- L2 flattens first, and calling the kernel
    with the original 2-D tensors fails an ndim check.

    So the shortcut is verified rather than assumed: on the first call, run L2
    (always correct), then try the kernel on the same inputs and compare. Only an
    exact match enables the fast path; anything else pins this instance to L2 for
    good. One extra invocation is negligible next to the ~4 s constructor.
    """

    __slots__ = ("op", "kernel")

    def __init__(self, op):
        self.op = op
        # None = not probed yet, False = L2 only, callable = verified shortcut.
        self.kernel = None if not _use_l2() else False

    def invoke(self, *args):
        if self.kernel:
            return self.kernel(*args)

        out = self.op(*args)
        if self.kernel is None:
            self.kernel = self._probe(args, out)
        return out

    def _probe(self, args, expected):
        """Return the kernel if it reproduces the L2 result exactly, else False."""
        kernel = getattr(self.op, "kernel", None)
        if kernel is None:
            return False
        try:
            got = kernel(*args)
        except Exception:
            return False
        if isinstance(expected, torch.Tensor) and isinstance(got, torch.Tensor):
            if got.shape == expected.shape and torch.equal(got, expected):
                return kernel
        return False


# --------------------------------------------------------------------------- #
# recipes
# --------------------------------------------------------------------------- #
def _aten_op(overload: str):
    base, _, ov = overload.partition(".")
    packet = getattr(torch.ops.aten, base)
    return getattr(packet, ov) if ov else packet


def _make_fallback(overload: str):
    """Build the not-handled-here path for one overload.

    Calling ``torch.ops.aten.<op>`` on a flagos tensor would re-enter this very
    registration and recurse forever. Instead the args are boxed to CUDA views
    (zero-copy) and aten runs there, which is exactly what the default
    ``backends_cuda.conf`` route does anyway, then the result is unboxed.
    """
    op = _aten_op(overload)

    def fallback(*args, **kwargs):
        _log_dispatch(overload, "cuda (tileops declined)")
        index = next(
            (
                a.device.index or 0
                for a in args
                if isinstance(a, torch.Tensor) and a.device.type == "flagos"
            ),
            None,
        )
        if index is None:
            return op(*args, **kwargs)
        boxed = tuple(_box(a) for a in args)
        boxed_kw = {k: _box(v) for k, v in kwargs.items()}
        out = op(*boxed, **boxed_kw)
        if isinstance(out, torch.Tensor):
            return _unbox(out, index)
        if isinstance(out, (tuple, list)):
            return type(out)(
                _unbox(o, index) if isinstance(o, torch.Tensor) else o for o in out
            )
        return out

    return fallback


def _unary_impl(op_cls, dtypes, extra, overload):
    fallback = _make_fallback(overload)
    ctor_extra = (False,) if extra.get("inplace") else ()

    def impl(x, *rest):
        if x.dtype not in dtypes or rest:
            return fallback(x, *rest)
        index = _device_index(x)
        xb = _box(x).contiguous()
        # These kernels commit N_total in the ctor and take a flat buffer.
        flat = xb.reshape(-1)
        entry = _get(
            op_cls,
            (flat.numel(), x.dtype) + ctor_extra,
            flat.numel(),
            x.dtype,
            *ctor_extra,
        )
        out = entry.invoke(flat)
        return _unbox(out.reshape(xb.shape), index)

    return impl


def _binary_impl(op_cls, dtypes, extra, overload):
    fallback = _make_fallback(overload)
    takes_alpha = extra.get("alpha", False)

    def impl(a, b, *rest, **kw):
        alpha = kw.pop("alpha", 1)
        if (
            not isinstance(a, torch.Tensor)
            or not isinstance(b, torch.Tensor)
            or a.dtype != b.dtype
            or a.dtype not in dtypes
            or rest
            or kw
            or (alpha != 1 and not takes_alpha)
        ):
            return fallback(a, b, *rest, **({"alpha": alpha} if alpha != 1 else {}))
        index = _device_index(a)
        ab, bb = _box(a).contiguous(), _box(b).contiguous()
        shapes = (tuple(ab.shape), tuple(bb.shape))
        ctor_args = shapes + (a.dtype,)
        ctor_kw = {"alpha": alpha} if takes_alpha else {}
        key = ctor_args + ((alpha,) if takes_alpha else ())
        entry = _get(op_cls, key, *ctor_args, **ctor_kw)
        return _unbox(entry.invoke(ab, bb), index)

    return impl


def _reduce_impl(op_cls, dtypes, extra, overload):
    fallback = _make_fallback(overload)
    takes_keepdim = extra.get("keepdim", False)
    takes_correction = extra.get("correction", False)

    def impl(x, dim=None, keepdim=False, *rest, **kw):
        correction = kw.get("correction")
        # Pass the caller's args through untouched on the fallback path -- these
        # ops carry keyword-only correction/dtype that must not be dropped.
        if (
            x.dtype not in dtypes
            or rest
            or "dtype" in kw
            or set(kw) - {"correction"}
            or (correction is not None and not takes_correction)
        ):
            return fallback(x, dim, keepdim, *rest, **kw)
        if isinstance(dim, (list, tuple)):
            if len(dim) != 1:
                return fallback(x, dim, keepdim, **kw)
            dim = dim[0]
        elif dim is None:
            # Full reduction; TileOPs wants an explicit axis.
            return fallback(x, dim, keepdim, **kw)
        index = _device_index(x)
        xb = _box(x).contiguous()

        ctor_args = [x.dtype, dim]
        ctor_kw = {}
        if takes_keepdim:
            ctor_kw["keepdim"] = keepdim
        if takes_correction and correction is not None:
            ctor_kw["correction"] = correction
        key = (x.dtype, dim, tuple(sorted(ctor_kw.items())))
        entry = _get(op_cls, key, *ctor_args, **ctor_kw)
        out = entry.invoke(xb)
        if isinstance(out, (tuple, list)):
            return tuple(_unbox(o, index) for o in out)
        return _unbox(out, index)

    return impl


def _softmax_impl(op_cls, dtypes, extra, overload):
    fallback = _make_fallback(overload)

    def impl(x, dim=-1, half_to_float=False, *rest):
        if x.dtype not in dtypes or half_to_float or rest:
            return fallback(x, dim, half_to_float, *rest)
        index = _device_index(x)
        xb = _box(x).contiguous()
        entry = _get(op_cls, (dim,), dim)
        return _unbox(entry.invoke(xb), index)

    return impl


_BUILDERS = {
    UNARY: _unary_impl,
    BINARY: _binary_impl,
    REDUCE: _reduce_impl,
    SOFTMAX: _softmax_impl,
}


def build_impl(recipe, module, cls_name, dtype_names, extra, overload):
    """Build the callable for one generated route, or None if unavailable."""
    try:
        op_cls = getattr(importlib.import_module(module), cls_name)
    except Exception:
        return None
    dtypes = tuple(getattr(torch, name) for name in dtype_names)
    impl = _BUILDERS[recipe](op_cls, dtypes, extra, overload)
    if impl is None or os.environ.get("FLAGOS_LOG_DISPATCH") != "1":
        return impl

    # Logging wrapper, added only when asked for, so the hot path keeps its shape.
    # The dtype guard inside `impl` may still divert to the fallback, which logs
    # its own line -- so a route can print "tileops" and then "cuda (declined)".
    def logged(*args, **kwargs):
        _log_dispatch(overload, "tileops")
        return impl(*args, **kwargs)

    return logged


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #
def _env_override_backend(overload: str) -> Optional[str]:
    """Backend named by ``FLAGOS_OP_<overload>``, matching the C++ conf parser.

    Dots become double underscores, per csrc/aten/common.cc.
    """
    var = "FLAGOS_OP_" + overload.replace(".", "__")
    val = os.environ.get(var)
    return val.strip().lower() if val else None


def enable_tileops_for_flagos(include=None, exclude=None) -> int:
    """Bind generated TileOPs routes on the PrivateUse1 (flagos) key.

    Returns the number of ops registered; 0 when TileOPs is unavailable or the
    host is not SM90. Idempotent -- repeat calls re-report the same count without
    rebinding.

    ``FLAGOS_OP_<overload>=<backend>`` is honored here, not just in the C++
    dispatcher. It has to be: a torch.library PrivateUse1 binding intercepts
    *before* the dispatcher runs, so an op bound here would never reach the
    dispatcher to see its override, and the documented per-op escape hatch
    would silently do nothing.
    """
    global _library

    if not is_tileops_available():
        return 0
    if _library is not None:
        return len(_registered)

    from torch_fl.generated.tileops_routes import ROUTES

    library = torch.library.Library("aten", "IMPL")
    count = 0
    for overload, recipe, module, cls_name, dtype_names, extra in ROUTES:
        if include is not None and overload not in include:
            continue
        if exclude is not None and overload in exclude:
            continue
        override = _env_override_backend(overload)
        if override is not None and override != "tileops":
            # Routed elsewhere on purpose: leave the key unbound so dispatch
            # falls through to the conf-selected backend.
            continue
        impl = build_impl(recipe, module, cls_name, dtype_names, extra, overload)
        if impl is None:
            continue
        try:
            library.impl(overload, impl, "PrivateUse1")
        except RuntimeError as exc:
            # Another backend already claimed this key; leave it alone rather than
            # fight over the registration.
            warnings.warn(
                f"TileOPs: skipping {overload}: {exc}", RuntimeWarning, stacklevel=2
            )
            continue
        _registered[overload] = cls_name
        count += 1

    _library = library
    return count


def registered_ops() -> Dict[str, str]:
    """aten overload -> TileOPs class name, for everything bound so far."""
    return dict(_registered)


# --------------------------------------------------------------------------- #
# helpers for tests and warmup
# --------------------------------------------------------------------------- #
#: Overloads whose result is only real for positive inputs. Comparing against
#: aten on negative values yields nan on both sides, which says nothing about the
#: kernel, so tests and warmup feed these strictly positive data.
POSITIVE_DOMAIN = frozenset(
    {
        "log",
        "log1p",
        "log2",
        "log10",
        "sqrt",
        "rsqrt",
        "pow.Tensor_Tensor",
        "remainder.Tensor",
    }
)


#: Reduce overloads whose schema declares ``int[1] dim`` rather than a bare int.
#: Passing the wrong one is a hard schema-match error, so it is spelled out.
DIM_IS_LIST = frozenset(
    {
        "sum.dim_IntList",
        "mean.dim",
        "amax",
        "amin",
        "logsumexp",
        "count_nonzero.dim_IntList",
        "var.correction",
        "std.correction",
        "var_mean.correction",
    }
)

#: Reduce overloads that take no ``keepdim`` positionally (or at all).
NO_POSITIONAL_KEEPDIM = frozenset({"count_nonzero.dim_IntList"})

#: Reduce overloads whose ``correction``/``keepdim`` are keyword-only.
KEYWORD_ONLY_REDUCE = frozenset(
    {"var.correction", "std.correction", "var_mean.correction"}
)


def sample_inputs(
    recipe, shape: Sequence[int], dtype, device="flagos", overload=None
) -> Tuple:
    """Positional args matching an overload's schema, for tests and warmup."""
    shape = tuple(shape)

    def make():
        if dtype == torch.bool:
            return torch.randint(0, 2, shape, device=device, dtype=torch.bool)
        if not dtype.is_floating_point:
            # Small positive ints keep bitwise/comparison results exact and stay
            # within range of the narrow int dtypes.
            return torch.randint(1, 9, shape, device=device, dtype=dtype)
        t = torch.randn(shape, device=device, dtype=dtype)
        return t.abs() + 0.5 if overload in POSITIVE_DOMAIN else t

    x = make()
    if recipe == BINARY:
        return (x, make())
    if recipe == SOFTMAX:
        return (x, -1, False)
    if recipe == REDUCE:
        dim = [-1] if overload in DIM_IS_LIST else -1
        if overload in KEYWORD_ONLY_REDUCE or overload in NO_POSITIONAL_KEEPDIM:
            return (x, dim)
        return (x, dim, False)
    return (x,)


def warmup(overloads=None, dtype=torch.float16) -> int:
    """Pre-build kernels on manifest workload shapes.

    Useful for a first run on a cold ``TILELANG_CACHE_DIR``, and mandatory under
    ``FLAGOS_TILEOPS_DISABLE_ALL_CACHE=1``, where every process re-pays the full
    compile per op. With the default path (see ``_disable_frontend_cache``) the
    kernel disk cache survives, so later processes start warm and this is only a
    convenience.
    """
    if not is_tileops_available():
        return 0
    from torch_fl.generated.tileops_routes import ROUTES, WORKLOADS

    want = str(dtype).replace("torch.", "")
    built = 0
    for overload, recipe, module, cls_name, dtype_names, extra in ROUTES:
        if overloads is not None and overload not in overloads:
            continue
        shape = WORKLOADS.get(overload)
        if shape is None or want not in dtype_names:
            continue
        impl = build_impl(recipe, module, cls_name, dtype_names, extra, overload)
        if impl is None:
            continue
        try:
            impl(*sample_inputs(recipe, shape, dtype, overload=overload))
            built += 1
        except Exception:
            continue
    return built
