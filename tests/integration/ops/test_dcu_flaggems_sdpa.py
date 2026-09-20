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

"""DCU coverage for the FlagGems ``scaled_dot_product_attention`` route.

``backends_dcu.conf`` routes ``scaled_dot_product_attention`` to FlagGems, which
means the composite reaches ``flag_gems``' Triton ``_attn_fwd`` instead of the
boxing route's math decomposition. The route exists for one call in particular:
Qwen-Image-2.1's prefill hands every one of its 32 blocks a ``(1, 1, 1, 4122)``
all-true bool mask, and on the boxing route a mask sends the composite to its
math path, which materialises the fp32 score matrix -- 32 x 4096 x 4122 x 4 B,
2.16 GB per call. The gate that lets that call through is the mask clause in
``FlagGemsEligible()`` plus ``RouteMask()``, both in
``csrc/aten/sdp_choice_stub.cc`` and both shared with MetaX: a 4-D bool mask
with one entry per key is admitted, and converted into the fp32 additive the
kernel can actually read.

Two things are asserted, because two things can go wrong:

1. **The gate.** A ``(1,1,1,KV)`` bool mask must reach the Triton kernel; a
   float mask, a materialised ``(B,H,Q,KV)`` bool mask and every shape outside
   the envelope must not. The *form* of the mask the kernel receives is
   asserted too, because the kernel indexes it by
   ``batch_id*stride(0) + head_id*stride(1) + offs_m*stride(2) + offs_n*stride(3)``
   with nothing bounding the first three. What that requires is not that those
   strides are 0 but that no index the kernel can form reaches outside the
   KV-long buffer: ``stride(i) * (size(i) - 1) == 0`` for each of the three.
   ``expand`` gives that for free -- an axis that grows from 1 takes stride 0,
   and an axis that stays at 1 is only ever indexed at 0 -- so the batch axis
   keeps a real stride at batch 1 and the mask still arrives as a bounded
   ``(B,H,Q,KV)`` fp32 additive. Handing the kernel the caller's ``(1,1,1,KV)``
   byte layout instead reads past the buffer: NaN on a small shape, a KERNEL
   VMFault at the model's own.
2. **The answer.** The route must agree with the host, an all-true key-valid row
   must leave the answer where the maskless call leaves it -- to within the bf16
   rounding of the output, as measured below -- and a row with a dropped key
   must move it far outside that.

Measured on Hygon DCU bw1000, DTK 6.3.26113, torch 2.10.0+cpu decoupled,
``flag_gems`` 5.4.0rc2.post1+g437ba3938. The op-level and end-to-end numbers are
in ``docs/reference/operator-support.md``.

Usage:
    pytest tests/integration/ops/test_dcu_flaggems_sdpa.py -v
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

import pytest
import torch_fl


DEVICE = "flagos:0"

pytestmark = pytest.mark.dcu

# The FlagGems SDPA route is the one conf entry the dispatcher does not select
# for itself. `csrc/aten/sdp_choice_stub.cc` registers one PrivateUse1 kernel for
# the composite under every conf, and that kernel reads `GetBackendForOp` to
# decide between the FlagGems callable and the boxing kernel -- so neither the
# conf text nor the dispatch log says which arm a call took (both arms are served
# by the same key; there is no route to print, the kernel *is* the route).
#
# The oracle is therefore the callable the FlagGems arm calls.
# `PythonOpCache::GetFunc` resolves `flag_gems.scaled_dot_product_attention` with
# a package import plus a `getattr`, memoized per qualname for the life of the
# process, so a counter installed on the package before the first call is the
# callable the route calls -- and a fresh interpreter is what makes "before the
# first call" true.
#
# The counter records the mask's shape, dtype and strides as well as how often
# the callable was reached. Those three are the route's promise about the mask:
# the kernel is handed the fp32 additive `RouteMask` builds, not the caller's
# bool row and not `None` -- a routed call carrying either of those is the bug
# this file is here to catch, not a variant of a passing case.
_SDPA_ORACLE = (
    "import flag_gems\n"
    "_sdpa_calls = []\n"
    "_sdpa_real = flag_gems.scaled_dot_product_attention\n"
    "def _mask_repr(m):\n"
    "    if m is None:\n"
    "        return None\n"
    "    return (tuple(m.shape), str(m.dtype).replace('torch.', ''),\n"
    "            tuple(m.stride()))\n"
    "def _sdpa_counting(*args, **kwargs):\n"
    "    _sdpa_calls.append((tuple(tuple(t.shape) for t in args),\n"
    "                        _mask_repr(kwargs.get('attn_mask'))))\n"
    "    return _sdpa_real(*args, **kwargs)\n"
    "flag_gems.scaled_dot_product_attention = _sdpa_counting\n"
    "\n"
    # The host reference moves the arguments back rather than rebuilding them, so
    # both arms of every case see the same values. A bool mask keeps its dtype on
    # the way over -- it is the meaning of the mask, and a float mask would take
    # a different branch of the composite.
    "def host_ref(q, k, v, kw):\n"
    "    moved = {}\n"
    "    for n, t in kw.items():\n"
    "        if torch.is_tensor(t):\n"
    "            moved[n] = t.cpu() if t.dtype == torch.bool else t.cpu().float()\n"
    "        else:\n"
    "            moved[n] = t\n"
    "    return torch.nn.functional.scaled_dot_product_attention(\n"
    "        q.float().cpu(), k.float().cpu(), v.float().cpu(), **moved)\n"
    "\n"
    # `cmp=False` marks the one case whose answer cannot be compared: a nonzero
    # dropout_p makes both arms draw from their own RNG, so neither is the
    # reference for the other. The clause it pins is a routing clause.
    "def run_case(tag, build, kw, cmp=True):\n"
    "    q, k, v = build()\n"
    "    ref = host_ref(q, k, v, kw) if cmp else None\n"
    "    before = len(_sdpa_calls)\n"
    "    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, **kw)\n"
    "    made = _sdpa_calls[before:]\n"
    "    routed = len(made)\n"
    "    masks = [m for _, m in made]\n"
    "    diff = 'skip' if ref is None else \\\n"
    "        f'{(out.float().cpu() - ref).abs().max().item():.6f}'\n"
    "    print(f'RESULT {tag} routed={routed} maxdiff={diff} '\n"
    "          f'device={out.device} masks={masks!r}')\n"
)

# The cases. `joint()` is the shape the route was measured on -- bf16, 4-D,
# head_dim 128, query seq 1024, key and value 1040 so the two lengths differ the
# way the model's 4096 and 4122 do, no mask -- and it is the one shape in the set
# the envelope admits. Every other case differs from it in exactly one respect,
# so a failing assertion names the clause that let a call through.
#
# The masks are built on the host and moved, so `mask_bcast_true` is exactly what
# `build_token_metadata` hands the model: `torch.ones` over the padded joint
# sequence, `(1,1,1,KV)`. `mask_false` puts a single False in an otherwise
# all-true row, which is the smallest change that makes the mask carry
# information -- and it must still route, because this clause converts the row
# rather than testing it. `mask_full_true` is the same information in a
# materialised `(B,H,Q,KV)` shape: the clause refuses it on `numel != KV`, which
# is also what keeps `RouteMask`'s `reshape({1,1,1,-1})` a view.
_SDPA_CASES = (
    # Built on the host and moved. `torch.randn(..., device=DEVICE)` is itself
    # dispatched -- privately, not by any conf entry -- to `flag_gems.randn`, so
    # drawing the operands on the host keeps every case's device work to the one
    # call this file is about, and keeps the values independent of the generator
    # the runtime picks.
    "def bf16(*shapes):\n"
    "    return tuple(torch.randn(*s).to(DEVICE).to(torch.bfloat16)\n"
    "                 for s in shapes)\n"
    "def joint():\n"
    "    return bf16((1, 2, 1024, 128), (1, 2, 1040, 128), (1, 2, 1040, 128))\n"
    "def true_mask():\n"
    "    return torch.ones(1, 1, 1, 1040, dtype=torch.bool)\n"
    "def full_mask():\n"
    "    return torch.ones(1, 2, 1024, 1040, dtype=torch.bool)\n"
    "\n"
    'run_case("eligible", lambda: bf16((1, 2, 1024, 128),\n'
    "                                  (1, 2, 1024, 128), (1, 2, 1024, 128)), {})\n"
    'run_case("joint", joint, {})\n'
    'run_case("mask_bcast_true", joint,\n'
    '         {"attn_mask": true_mask().to(DEVICE)})\n'
    'run_case("mask_full_true", joint,\n'
    '         {"attn_mask": full_mask().to(DEVICE)})\n'
    "def false_mask():\n"
    "    m = true_mask()\n"
    "    m[0, 0, 0, 7] = False\n"
    "    return m.to(DEVICE)\n"
    'run_case("mask_false", joint, {"attn_mask": false_mask()})\n'
    'run_case("mask_float_zeros", joint,\n'
    '         {"attn_mask": torch.zeros(1, 1, 1, 1040).to(DEVICE)})\n'
    'run_case("head_dim64", lambda: bf16((1, 2, 1024, 64),\n'
    "                                    (1, 2, 1040, 64), (1, 2, 1040, 64)), {})\n"
    'run_case("seq512", lambda: bf16((1, 2, 512, 128),\n'
    "                                (1, 2, 512, 128), (1, 2, 512, 128)), {})\n"
    'run_case("float32", lambda: tuple(\n'
    "    torch.randn(1, 2, 1024, 128).to(DEVICE) for _ in range(3)), {})\n"
    'run_case("float16", lambda: tuple(\n'
    "    torch.randn(1, 2, 1024, 128).to(DEVICE).to(torch.float16)\n"
    "    for _ in range(3)), {})\n"
    'run_case("rank2", lambda: bf16((1024, 128), (1040, 128), (1040, 128)), {})\n'
    'run_case("causal", joint, {"is_causal": True})\n'
    'run_case("scale", joint, {"scale": 0.1})\n'
    'run_case("dropout", joint, {"dropout_p": 0.1}, cmp=False)\n'
    'run_case("gqa", joint, {"enable_gqa": True})\n'
)

# Case tag -> (calls that may reach the FlagGems kernel, the mask form those
# calls must carry, the clause that decides both). The count is per case, so a
# run where an ineligible case reaches the kernel fails at that case rather than
# moving a global total.
#
# `mask_form` is "none" when every routed call must have been made without a
# mask, "converted" when every routed call must have carried `RouteMask`'s fp32
# additive, and None when the case must not reach the kernel at all and there is
# therefore no call to describe.
_SDPA_CASES_EXPECTED = {
    "eligible": (1, "none", "bf16, 4-D, head_dim 128, query seq 1024, no mask"),
    "joint": (1, "none", "the model's shape: query 1024 against key/value 1040"),
    "mask_bcast_true": (
        1,
        "converted",
        "the model's mask: an all-true (1,1,1,KV) bool row",
    ),
    "mask_full_true": (
        0,
        None,
        "a materialised (B,H,Q,KV) bool mask, whose numel is not KV",
    ),
    "mask_false": (
        1,
        "converted",
        "one false in the row, which the additive must carry through",
    ),
    "mask_float_zeros": (
        0,
        None,
        "a float mask, which the clause admits as bool only",
    ),
    "head_dim64": (0, None, "head_dim 64, which `_attn_fwd` tiles differently"),
    "seq512": (0, None, "query seq 512, under the 1024 the route requires"),
    "float32": (0, None, "float32, and the route is bf16-only"),
    "float16": (0, None, "float16, and the route is bf16-only"),
    "rank2": (0, None, "2-D input, and the route requires the 4-D call form"),
    "causal": (0, None, "is_causal, which the flag_gems entry point does not take"),
    "scale": (0, None, "an explicit scale, which the entry point does not take"),
    "dropout": (
        0,
        None,
        "a nonzero dropout_p, which the entry point does not take",
    ),
    "gqa": (0, None, "enable_gqa, which the entry point does not take"),
}

# Max |device - host| against a float32 host reference. The bf16 rounding of the
# operands already moves the logits by ~2e-2, and softmax turns that into far
# less on an output of order 1; 5e-2 is the bound the four route cases below
# share, measured well inside it.
_SDPA_DIFF_BOUND = 5e-2

# Max |device - device| between the all-true-row call and the maskless one, and
# the floor the half-dropped-row call has to clear.
#
# The first cannot be 0. The two calls do not run the same kernel: the route
# hands one of them a 1040-wide fp32 additive and the other `attn_mask=None`,
# and FlagGems' entry point takes a different branch on that, so the same
# reduction comes back reassociated. Measured on a DCU bw1000 at
# q(1,2,1024,128) x kv(1,2,1040,128) bf16: max|d| 0.0001220703125 -- 2^-13, one
# bf16 ulp below the 0.287 peak -- over 8 elements of 2.1M, with both arms
# landing on the same 0.000694 error against an fp64 reference. The bound is
# twice the ulp of a bf16 output of this magnitude, and it is 250x under the
# floor below, so the pair still separates "the additive is read" from "the
# additive is built and ignored".
_SDPA_IDENTITY_BOUND = 1e-3
_SDPA_INFORMATIVE_BOUND = 1e-2

# The maskless call, the all-true-row call and a half-dropped-row call, on one
# set of operands. Built once and reused, so both numbers are statements about
# the mask and not about the RNG. The all-true row converts to an additive that
# is zero everywhere, so it must leave the answer where the maskless call left
# it; the third row must move it, or the additive is being built but not read.
_SDPA_IDENTITY = (
    "torch.manual_seed(4242)\n"
    "q = torch.randn(1, 2, 1024, 128).to(DEVICE).to(torch.bfloat16)\n"
    "k = torch.randn(1, 2, 1040, 128).to(DEVICE).to(torch.bfloat16)\n"
    "v = torch.randn(1, 2, 1040, 128).to(DEVICE).to(torch.bfloat16)\n"
    "base = torch.nn.functional.scaled_dot_product_attention(q, k, v)\n"
    "bcast = torch.nn.functional.scaled_dot_product_attention(\n"
    "    q, k, v, attn_mask=true_mask().to(DEVICE))\n"
    "half = true_mask()\n"
    "half[0, 0, 0, :520] = False\n"
    "dropped = torch.nn.functional.scaled_dot_product_attention(\n"
    "    q, k, v, attn_mask=half.to(DEVICE))\n"
    "print('IDENTICAL max|d|='\n"
    "      f'{(bcast.float() - base.float()).abs().max().item():.9f}')\n"
    "print('INFORMATIVE max|d|='\n"
    "      f'{(dropped.float() - base.float()).abs().max().item():.6f}')\n"
    "print(f'ROUTED-MASKS {_sdpa_calls}')\n"
)

_SDPA_RESULT = re.compile(
    r"^RESULT (\S+) routed=(\d+) maxdiff=(\S+) device=(\S+) masks=(\[.*\])$",
    re.MULTILINE,
)
_SDPA_IDENTICAL = re.compile(r"^IDENTICAL max\|d\|=([\d.eE+-]+)$", re.MULTILINE)
_SDPA_INFORMATIVE = re.compile(r"^INFORMATIVE max\|d\|=([\d.eE+-]+)$", re.MULTILINE)
_SDPA_ROUTED_MASKS = re.compile(r"^ROUTED-MASKS (.+)$", re.MULTILINE)


def _active_conf() -> str:
    return torch_fl.backend_config_path()


def _require_dcu_flaggems() -> None:
    """Skip unless this is a DCU boxing build with a usable FlagGems runtime."""
    if not _active_conf().endswith("backends_dcu.conf"):
        pytest.skip(
            "the DCU FlagGems routes are stated in backends_dcu.conf; this run "
            f"selected {_active_conf() or '<no conf>'}"
        )
    if torch_fl.flagos.device_count() < 1:
        pytest.skip("DCU device is unavailable")
    try:
        import triton

        import flag_gems
    except Exception as exc:  # noqa: BLE001 - any import failure means skip
        pytest.skip(f"FlagGems DCU runtime is unavailable: {exc}")
    if "hcu" not in triton.backends.backends:
        pytest.skip("the installed Triton does not provide the Hygon backend")
    print(f"FlagGems: {flag_gems.__version__} ({flag_gems.__file__})")


def _run_sdpa_cases(
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run every case in one fresh interpreter and print a parseable report."""
    code = (
        # `torch_fl` before `torch`, the order every DCU entry point uses. With
        # `torch` first the PrivateUse1 key is registered after the device module
        # has been built and every copy to `flagos:0` raises "Cannot initialize
        # CUDA without ATen_cuda library" instead of moving the tensor.
        "import torch_fl\n"
        "import torch\n"
        f"DEVICE = {DEVICE!r}\n"
        "torch.manual_seed(1729)\n"
        f"{_SDPA_ORACLE}"
        f"{_SDPA_CASES}"
        f"{_SDPA_IDENTITY}"
    )
    env = os.environ.copy()
    # The shipped conf is the arm without an override; an inherited value would
    # silently make every case the other arm.
    env.pop("FLAGOS_OP_scaled_dot_product_attention", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def _sdpa_report(result: subprocess.CompletedProcess):
    """``(per-case rows, all-true max|d|, dropped max|d|, census)``."""
    report = {}
    for tag, routed, diff, device, masks in _SDPA_RESULT.findall(result.stdout):
        report[tag] = (
            int(routed),
            float("nan") if diff == "skip" else float(diff),
            device,
            ast.literal_eval(masks),
        )
    assert set(report) == set(_SDPA_CASES_EXPECTED), (
        "the probe did not report every case: "
        f"{sorted(report)} against {sorted(_SDPA_CASES_EXPECTED)}\n"
        f"{result.stdout}\n{result.stderr}"
    )
    identical = _SDPA_IDENTICAL.findall(result.stdout)
    assert len(identical) == 1, f"expected one IDENTICAL line\n{result.stdout}"
    informative = _SDPA_INFORMATIVE.findall(result.stdout)
    assert len(informative) == 1, f"expected one INFORMATIVE line\n{result.stdout}"
    census = _SDPA_ROUTED_MASKS.findall(result.stdout)
    assert len(census) == 1, f"expected one ROUTED-MASKS line\n{result.stdout}"
    return (
        report,
        float(identical[0]),
        float(informative[0]),
        ast.literal_eval(census[0]),
    )


@pytest.fixture(scope="module")
def sdpa_shipped():
    """The shipped ``backends_dcu.conf``, with no environment override."""
    _require_dcu_flaggems()
    result = _run_sdpa_cases()
    assert result.returncode == 0, f"sdpa probe failed\n{result.stderr}"
    return _sdpa_report(result)


@pytest.fixture(scope="module")
def sdpa_switched_off():
    """The same cases with the documented runtime switch set."""
    _require_dcu_flaggems()
    result = _run_sdpa_cases({"FLAGOS_OP_scaled_dot_product_attention": "cuda"})
    assert result.returncode == 0, f"sdpa probe failed\n{result.stderr}"
    return _sdpa_report(result)


class TestDcuFlaggemsSdpaRoute:
    """``scaled_dot_product_attention`` is routed on the envelope, not the key.

    The route exists because a mask -- any mask -- sent the boxing composite to
    its math decomposition, which materialises the full fp32 score matrix. The
    FlagGems ``_attn_fwd`` kernel does not, which is what the route buys; the
    envelope in ``FlagGemsEligible`` and the conversion in ``RouteMask`` are what
    bound the risk of taking it. The things asserted here are the ones that can
    go wrong: a call the gate admits must take the route, a call it refuses must
    not, the mask the kernel receives must be the converted additive, and the
    answer must move with the mask exactly when the mask carries information.
    """

    @pytest.mark.parametrize("tag", sorted(_SDPA_CASES_EXPECTED))
    def test_route_follows_the_envelope(self, sdpa_shipped, tag):
        report, _, _, _ = sdpa_shipped
        want, _, why = _SDPA_CASES_EXPECTED[tag]
        routed, diff, device, masks = report[tag]
        assert routed == want, (
            f"{tag} ({why}): {routed} call(s) reached the FlagGems kernel, "
            f"expected {want}. Either FlagGemsEligible in "
            "csrc/aten/sdp_choice_stub.cc admits this call, or it admits "
            "nothing and the conf does not route the op"
        )
        assert len(masks) == routed, (
            f"{tag} ({why}): {routed} call(s) but {len(masks)} mask record(s)"
        )
        assert device == DEVICE, f"{tag}: the result is on {device}, not {DEVICE}"
        if diff == diff:  # not the dropout case, which has no reference
            assert diff <= _SDPA_DIFF_BOUND, (
                f"{tag} ({why}): the result differs from the host reference by "
                f"{diff:.6f}, over the {_SDPA_DIFF_BOUND} bound"
            )

    def test_the_kernel_receives_the_converted_mask(self, sdpa_shipped):
        """Every routed call carried the additive ``RouteMask`` builds, or none.

        The census is the whole run, so this is asserted on the calls the kernel
        actually saw rather than on the mask cases alone: a routed call that
        forwarded the caller's ``(1,1,1,KV)`` bool, or a materialised
        ``(B,H,Q,KV)`` one, fails here whatever case it came from.

        The stride check is the load-bearing half. The kernel forms the block
        pointer as ``batch_id*stride(0) + head_id*stride(1) + offs_m*stride(2) +
        offs_n*stride(3)`` with the first three unbounded, and what keeps it
        inside the KV-long buffer is ``stride(i) * (size(i) - 1) == 0`` for each
        of them -- the last index they can reach contributes nothing. It is not
        that all three strides are 0: at batch 1 ``expand`` never touches axis 0,
        so that one keeps the buffer width and is harmless because the only
        index it can take is 0.
        """
        report, _, _, census = sdpa_shipped
        expected = sum(n for n, _, _ in _SDPA_CASES_EXPECTED.values()) + 3
        assert len(census) == expected, (
            f"{len(census)} call(s) reached the FlagGems kernel, expected "
            f"{expected}: {[s for s, _ in census]}"
        )
        converted = [m for _, m in census if m is not None]
        assert len(converted) == 4, (
            "expected the four masked calls in the run (mask_bcast_true, "
            f"mask_false, and the two in the identity block); got {census}"
        )
        for shapes, mask in census:
            if mask is None:
                continue
            shape, dtype, strides = mask
            q_shape, kv_shape = shapes[0], shapes[1]
            assert dtype == "float32", f"the kernel was handed a {dtype} mask"
            assert tuple(shape) == (
                q_shape[0],
                q_shape[1],
                q_shape[2],
                kv_shape[2],
            ), f"the mask shape {shape} does not match the call {shapes}"
            assert strides[3] == 1, (
                f"the kernel was handed a mask with strides {strides}; the key "
                "axis has to be innermost and dense to be read as one row"
            )
            reach = max(strides[i] * (shape[i] - 1) for i in range(3))
            assert reach == 0, (
                f"the kernel was handed a mask with strides {strides} against "
                f"shape {shape}: the block pointer can reach offset {reach} on "
                "the batch/head/query axes, past the KV-long buffer RouteMask "
                "expands"
            )

    def test_an_all_true_row_does_not_change_the_answer(self, sdpa_shipped):
        """An all-true key-valid row leaves the answer where the maskless call did.

        The two calls do not run the same kernel -- one of them is handed a
        1040-wide zero additive and the other ``attn_mask=None``, and FlagGems
        branches on that -- so this is the bf16 rounding of the output, not bit
        equality. It is the whole claim the route makes about the model's
        ``key_valid``: the row the model passes carries no information, and the
        additive it converts to has to carry none either.
        """
        _, identical, _, _ = sdpa_shipped
        assert identical <= _SDPA_IDENTITY_BOUND, (
            "the all-true key-valid row moved the result by "
            f"{identical:.9f}, over the {_SDPA_IDENTITY_BOUND} bound -- a row "
            "with every key valid is not supposed to drop or reweight any"
        )

    def test_a_dropped_key_does_change_the_answer(self, sdpa_shipped):
        """The additive is read, not merely built.

        Half the row is dropped in the identity block, and it lands two orders of
        magnitude above the all-true row's own movement, so this is the other
        half of the claim above: if the route were computing ``RouteMask`` and
        then discarding it, the all-true case would still pass and this one would
        not.
        """
        _, _, informative, _ = sdpa_shipped
        assert informative > _SDPA_INFORMATIVE_BOUND, (
            "dropping half the key-valid row moved the result by only "
            f"{informative:.6f}, under the {_SDPA_INFORMATIVE_BOUND} floor -- "
            "the converted additive is not reaching the kernel"
        )

    def test_off_switch_returns_every_call_to_the_boxing_route(self, sdpa_switched_off):
        """``FLAGOS_OP_scaled_dot_product_attention=cuda`` is the whole switch.

        The same conf is in use in both arms -- only this variable differs -- so
        this is also the check that the eligible cases' route came from the conf
        key and not from the shape alone.
        """
        report, _, _, census = sdpa_switched_off
        assert census == [], f"the FlagGems kernel ran with the switch off: {census}"
        off = {tag: row[0] for tag, row in report.items()}
        assert set(off.values()) == {0}, f"routes with the switch off: {off}"
