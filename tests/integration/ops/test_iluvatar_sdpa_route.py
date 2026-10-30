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

"""The Iluvatar ``scaled_dot_product_attention`` FlagGems route.

``backends_iluvatar.conf`` routes the composite to FlagGems, so the wrapper in
``csrc/aten/sdp_choice_stub.cc`` hands an eligible call to ``flag_gems``' fused
Triton kernel. What it replaces is not a boxing kernel -- this platform has
none -- but ATen's own ``CompositeImplicitAutograd`` math decomposition, which
materialises the whole ``NxN`` score matrix and then walks it twice. That is why
the route is worth the most here of the three platforms that have taken it:
measured on a BI-V150, bf16, the shapes of issue #667's Qwen3.5-VL 2B forward,
the math path takes 165.204 ms at q(1,16,1944,64) where the route takes
0.814 ms. The full table is at ``ILUVATAR_COMPOSITE_FLAGGEMS`` in
``scripts/codegen/gen_vendor_confs.py``.

Which arm a call took is not visible in the conf text or the dispatch log: one
PrivateUse1 kernel serves both arms under every conf, so there is no route to
print -- the kernel *is* the route. The oracle is therefore the callable the
FlagGems arm calls, counted by replacing ``flag_gems.scaled_dot_product_attention``
in a fresh interpreter before the first call (``PythonOpCache::GetFunc``
memoizes the lookup per qualname for the life of the process, so "before the
first call" has to be a new process).

Three things are asserted, because three things can go wrong independently:

1. **The gate.** A call inside ``FlagGemsEligible``'s envelope must reach the
   kernel, and a call outside it must not. The cases differ from the eligible
   one in exactly one respect each, so a failure names the clause.
2. **The mask.** ``flag_gems`` reads a bool mask with the opposite polarity to
   torch (issue #667 §5 measured it: ``True`` means *masked* there) and indexes
   the mask with unbound strides, so a routed masked call must carry the fp32
   additive ``RouteMask`` builds, not the caller's bool row.
3. **The answer.** The route must agree with a host reference, and the mask must
   move the answer exactly when it carries information.

Usage:
    pytest tests/integration/ops/test_iluvatar_sdpa_route.py -v
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

pytestmark = [
    pytest.mark.flaggems,
    pytest.mark.main_ops,
]

# Max |device - host| against a float32 host reference. bf16 operands move the
# logits by ~2e-2 before the kernel sees them and softmax turns that into much
# less on an output of order 1. The measured spread over the five routed shapes
# was 0.0011 to 0.0875 (the masked shape being the most accurate of them), so
# the bound is set just above the widest of those rather than at the tightest.
_DIFF_BOUND = 1e-1

# Max |device - device| between the all-true-row call and the maskless one.
# Cannot be 0: the two calls do not run the same kernel -- one is handed a
# KV-wide fp32 additive and the other `attn_mask=None`, and flag_gems' entry
# point branches on that, so the same reduction comes back reassociated. The
# bound is a few bf16 ulps of an output of this magnitude, and it is two orders
# under the floor below, so the pair still separates "the additive is read" from
# "the additive is built and ignored".
_IDENTITY_BOUND = 5e-3
_INFORMATIVE_BOUND = 1e-2

# The counting wrapper and the host reference, as source for a fresh
# interpreter. The mask's shape, dtype and strides are recorded as well as the
# call, because those three are the route's promise about the mask.
_ORACLE = (
    "import flag_gems\n"
    "_calls = []\n"
    "_real = flag_gems.scaled_dot_product_attention\n"
    "def _mask_repr(m):\n"
    "    if m is None:\n"
    "        return None\n"
    "    return (tuple(m.shape), str(m.dtype).replace('torch.', ''),\n"
    "            tuple(m.stride()))\n"
    "def _counting(*args, **kwargs):\n"
    "    _calls.append((tuple(tuple(t.shape) for t in args),\n"
    "                   _mask_repr(kwargs.get('attn_mask'))))\n"
    "    return _real(*args, **kwargs)\n"
    "flag_gems.scaled_dot_product_attention = _counting\n"
    "\n"
    # The reference moves the arguments rather than rebuilding them, so both
    # arms of a case see the same values. A bool mask keeps its dtype on the way
    # over: it is the meaning of the mask, and a float mask takes a different
    # branch of the composite.
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
    "    before = len(_calls)\n"
    "    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, **kw)\n"
    "    made = _calls[before:]\n"
    "    diff = 'skip' if ref is None else \\\n"
    "        f'{(out.float().cpu() - ref).abs().max().item():.6f}'\n"
    "    print(f'RESULT {tag} routed={len(made)} maxdiff={diff} '\n"
    "          f'device={out.device} masks={[m for _, m in made]!r}')\n"
)

# The cases. `model()` is the shape the route was measured on and the shape
# issue #667's forward issues: bf16, 4-D, head_dim 64, query seq 1944. Every
# other case differs from it in exactly one respect.
#
# The masked cases use a `(1,1,1,KV)` all-true row, which is the only mask shape
# the clause admits -- `numel == KV` together with `size(3) == KV` is what pins
# it, and what keeps `RouteMask`'s `reshape({1,1,1,-1})` a view. `mask_false`
# puts one False in that row, the smallest change that makes the mask carry
# information, and it must still route: the clause converts the row rather than
# testing its contents.
_CASES = (
    # Drawn on the host and moved. `torch.randn(..., device=DEVICE)` is itself
    # dispatched to `flag_gems.randn`, so drawing on the host keeps each case's
    # device work to the one call this file is about.
    "def bf16(*shapes):\n"
    "    return tuple(torch.randn(*s).to(DEVICE).to(torch.bfloat16)\n"
    "                 for s in shapes)\n"
    "def model():\n"
    "    return bf16((1, 16, 1944, 64), (1, 16, 1944, 64), (1, 16, 1944, 64))\n"
    "def true_mask():\n"
    "    return torch.ones(1, 1, 1, 1944, dtype=torch.bool)\n"
    "def full_mask():\n"
    "    return torch.ones(1, 16, 1944, 1944, dtype=torch.bool)\n"
    "\n"
    'run_case("model", model, {})\n'
    'run_case("short", lambda: bf16((1, 16, 336, 64),\n'
    "                               (1, 16, 336, 64), (1, 16, 336, 64)), {})\n"
    'run_case("head_dim128", lambda: bf16((1, 24, 1024, 128),\n'
    "                                     (1, 24, 1024, 128),\n"
    "                                     (1, 24, 1024, 128)), {})\n"
    'run_case("mask_bcast_true", model,\n'
    '         {"attn_mask": true_mask().to(DEVICE)})\n'
    "def false_mask():\n"
    "    m = true_mask()\n"
    "    m[0, 0, 0, 7] = False\n"
    "    return m.to(DEVICE)\n"
    'run_case("mask_false", model, {"attn_mask": false_mask()})\n'
    'run_case("mask_full_true", model,\n'
    '         {"attn_mask": full_mask().to(DEVICE)})\n'
    'run_case("mask_float_zeros", model,\n'
    '         {"attn_mask": torch.zeros(1, 1, 1, 1944).to(DEVICE)})\n'
    'run_case("mask_short_seq", lambda: bf16(\n'
    "    (1, 16, 512, 64), (1, 16, 512, 64), (1, 16, 512, 64)),\n"
    '    {"attn_mask": torch.ones(1, 1, 1, 512, dtype=torch.bool).to(DEVICE)})\n'
    'run_case("float32", lambda: tuple(\n'
    "    torch.randn(1, 16, 1944, 64).to(DEVICE) for _ in range(3)), {})\n"
    'run_case("rank2", lambda: bf16((1944, 64), (1944, 64), (1944, 64)), {})\n'
    'run_case("causal", model, {"is_causal": True})\n'
    'run_case("scale", model, {"scale": 0.1})\n'
    'run_case("dropout", model, {"dropout_p": 0.1}, cmp=False)\n'
    'run_case("gqa", model, {"enable_gqa": True})\n'
)

# Case tag -> (calls that may reach the kernel, the mask form those calls must
# carry, the clause that decides both). The count is per case, so an ineligible
# case that reaches the kernel fails at that case rather than moving a total.
#
# `mask_form` is "none" when every routed call must have been made without a
# mask, "converted" when every routed call must have carried `RouteMask`'s fp32
# additive, and None when the case must not reach the kernel at all.
_EXPECTED = {
    "model": (1, "none", "the measured shape: bf16, 4-D, head_dim 64, no mask"),
    "short": (1, "none", "query seq 336, which the unmasked clause does not floor"),
    "head_dim128": (1, "none", "head_dim 128, the top of the shared bounds"),
    "mask_bcast_true": (
        1,
        "converted",
        "an all-true (1,1,1,KV) bool row above the masked query-seq floor",
    ),
    "mask_false": (
        1,
        "converted",
        "one false in that row, which the additive must carry through",
    ),
    "mask_full_true": (
        0,
        None,
        "a materialised (B,H,Q,KV) bool mask, whose numel is not KV",
    ),
    "mask_float_zeros": (0, None, "a float mask, which the clause admits as bool only"),
    "mask_short_seq": (
        0,
        None,
        "a masked call at query seq 512, under kRoutedMaskedMinQuerySeq",
    ),
    "float32": (0, None, "float32, which the clause admits as bf16/fp16 only"),
    "rank2": (0, None, "2-D input, and the route requires the 4-D call form"),
    "causal": (0, None, "is_causal, which the flag_gems entry point does not take"),
    "scale": (0, None, "an explicit scale, which the entry point does not take"),
    "dropout": (0, None, "a nonzero dropout_p, which the entry point does not take"),
    "gqa": (0, None, "enable_gqa, which the entry point does not take"),
}

# The maskless call, the all-true-row call and a half-dropped-row call on one set
# of operands, so both numbers are statements about the mask and not the RNG. The
# all-true row converts to an additive that is zero everywhere, so it must leave
# the answer where the maskless call left it; the dropped row must move it, or
# the additive is being built and not read.
_IDENTITY = (
    "torch.manual_seed(4242)\n"
    "q = torch.randn(1, 16, 1944, 64).to(DEVICE).to(torch.bfloat16)\n"
    "k = torch.randn(1, 16, 1944, 64).to(DEVICE).to(torch.bfloat16)\n"
    "v = torch.randn(1, 16, 1944, 64).to(DEVICE).to(torch.bfloat16)\n"
    "base = torch.nn.functional.scaled_dot_product_attention(q, k, v)\n"
    "bcast = torch.nn.functional.scaled_dot_product_attention(\n"
    "    q, k, v, attn_mask=true_mask().to(DEVICE))\n"
    "half = true_mask()\n"
    "half[0, 0, 0, :972] = False\n"
    "dropped = torch.nn.functional.scaled_dot_product_attention(\n"
    "    q, k, v, attn_mask=half.to(DEVICE))\n"
    "print('IDENTICAL max|d|='\n"
    "      f'{(bcast.float() - base.float()).abs().max().item():.9f}')\n"
    "print('INFORMATIVE max|d|='\n"
    "      f'{(dropped.float() - base.float()).abs().max().item():.6f}')\n"
    "print(f'ROUTED-MASKS {_calls}')\n"
)

_RESULT = re.compile(
    r"^RESULT (\S+) routed=(\d+) maxdiff=(\S+) device=(\S+) masks=(\[.*\])$",
    re.MULTILINE,
)
_IDENTICAL = re.compile(r"^IDENTICAL max\|d\|=([\d.eE+-]+)$", re.MULTILINE)
_INFORMATIVE = re.compile(r"^INFORMATIVE max\|d\|=([\d.eE+-]+)$", re.MULTILINE)
_ROUTED_MASKS = re.compile(r"^ROUTED-MASKS (.+)$", re.MULTILINE)


def _active_conf() -> str:
    return torch_fl.backend_config_path()


def _require_iluvatar_flaggems() -> None:
    """Skip unless this is an Iluvatar build with a usable FlagGems runtime."""
    if not _active_conf().endswith("backends_iluvatar.conf"):
        pytest.skip(
            "the Iluvatar SDPA route is stated in backends_iluvatar.conf; this "
            f"run selected {_active_conf() or '<no conf>'}"
        )
    if torch_fl.flagos.device_count() < 1:
        pytest.skip("Iluvatar device is unavailable")
    try:
        import triton

        import flag_gems
    except Exception as exc:  # noqa: BLE001 - any import failure means skip
        pytest.skip(f"FlagGems Iluvatar runtime is unavailable: {exc}")
    if "iluvatar" not in triton.backends.backends:
        pytest.skip("the installed Triton does not provide the Iluvatar backend")
    print(f"FlagGems: {flag_gems.__version__} ({flag_gems.__file__})")


def _run_cases(
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run every case in one fresh interpreter and print a parseable report."""
    code = (
        "import torch_fl\n"
        "import torch\n"
        f"DEVICE = {DEVICE!r}\n"
        "torch.manual_seed(1729)\n"
        f"{_ORACLE}"
        f"{_CASES}"
        f"{_IDENTITY}"
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


def _report(result: subprocess.CompletedProcess):
    """``(per-case rows, all-true max|d|, dropped max|d|, census)``."""
    report = {}
    for tag, routed, diff, device, masks in _RESULT.findall(result.stdout):
        report[tag] = (
            int(routed),
            float("nan") if diff == "skip" else float(diff),
            device,
            ast.literal_eval(masks),
        )
    assert set(report) == set(_EXPECTED), (
        "the probe did not report every case: "
        f"{sorted(report)} against {sorted(_EXPECTED)}\n"
        f"{result.stdout}\n{result.stderr}"
    )
    identical = _IDENTICAL.findall(result.stdout)
    assert len(identical) == 1, f"expected one IDENTICAL line\n{result.stdout}"
    informative = _INFORMATIVE.findall(result.stdout)
    assert len(informative) == 1, f"expected one INFORMATIVE line\n{result.stdout}"
    census = _ROUTED_MASKS.findall(result.stdout)
    assert len(census) == 1, f"expected one ROUTED-MASKS line\n{result.stdout}"
    return (
        report,
        float(identical[0]),
        float(informative[0]),
        ast.literal_eval(census[0]),
    )


@pytest.fixture(scope="module")
def sdpa_shipped():
    """The shipped ``backends_iluvatar.conf``, with no environment override."""
    _require_iluvatar_flaggems()
    result = _run_cases()
    assert result.returncode == 0, f"sdpa probe failed\n{result.stderr}"
    return _report(result)


@pytest.fixture(scope="module")
def sdpa_switched_off():
    """The same cases with the documented runtime switch set.

    ``none`` rather than ``cuda``: the switch has to name a backend this build
    has, and an Iluvatar wheel compiles no boxing kernels. At ``none`` the
    wrapper's FlagGems test fails and the tail runs the composite on the flagos
    tensors, which is the math decomposition this route replaced.
    """
    _require_iluvatar_flaggems()
    result = _run_cases({"FLAGOS_OP_scaled_dot_product_attention": "none"})
    assert result.returncode == 0, f"sdpa probe failed\n{result.stderr}"
    return _report(result)


class TestIluvatarSdpaRoute:
    """The composite is routed on the envelope, not on the conf key alone.

    The conf names FlagGems for the op, and ``FlagGemsEligible`` then decides
    per call; what the refused calls fall back to here is ATen's math
    decomposition rather than a vendor kernel, so a gate that is too wide is a
    wrong answer and a gate that is too narrow is only a slow one.
    """

    @pytest.mark.parametrize("tag", sorted(_EXPECTED))
    def test_route_follows_the_envelope(self, sdpa_shipped, tag):
        report, _, _, _ = sdpa_shipped
        want, _, why = _EXPECTED[tag]
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
            assert diff <= _DIFF_BOUND, (
                f"{tag} ({why}): the result differs from the host reference by "
                f"{diff:.6f}, over the {_DIFF_BOUND} bound"
            )

    def test_the_kernel_receives_the_converted_mask(self, sdpa_shipped):
        """Every routed call carried the additive ``RouteMask`` builds, or none.

        Asserted over the whole run's census rather than the mask cases alone: a
        routed call that forwarded the caller's bool row fails here whatever case
        it came from, and forwarding that row is not a near miss -- ``flag_gems``
        reads ``True`` as *masked*, the opposite of torch, so the answer would be
        the complement of the one asked for.

        The stride check is the load-bearing half. The kernel forms its pointer
        as ``batch_id*stride(0) + head_id*stride(1) + offs_m*stride(2) +
        offs_n*stride(3)`` with the first three unbounded, and what keeps it
        inside the KV-long buffer is ``stride(i) * (size(i) - 1) == 0`` for each
        of them: the last index they can reach contributes nothing. Not that all
        three strides are 0 -- at batch 1 ``expand`` never touches axis 0, so
        that one keeps the buffer width and is harmless.
        """
        report, _, _, census = sdpa_shipped
        expected = sum(n for n, _, _ in _EXPECTED.values()) + 3
        assert len(census) == expected, (
            f"{len(census)} call(s) reached the FlagGems kernel, expected "
            f"{expected}: {[s for s, _ in census]}"
        )
        for tag, (want, form, why) in sorted(_EXPECTED.items()):
            if not want:
                continue
            _, _, _, masks = report[tag]
            for mask in masks:
                if form == "none":
                    assert mask is None, (
                        f"{tag} ({why}): routed with {mask!r}, expected no mask"
                    )
                    continue
                assert mask is not None, f"{tag} ({why}): routed with no mask"
                shape, dtype, stride = mask
                assert dtype == "float32", (
                    f"{tag} ({why}): the kernel received a {dtype} mask; "
                    "RouteMask must convert the bool row to an fp32 additive"
                )
                assert len(shape) == 4, f"{tag} ({why}): mask shape {shape}"
                for axis in range(3):
                    assert stride[axis] * (shape[axis] - 1) == 0, (
                        f"{tag} ({why}): mask axis {axis} is size "
                        f"{shape[axis]} stride {stride[axis]}, so the kernel "
                        "can index past the KV-long buffer"
                    )

    def test_an_all_true_mask_leaves_the_answer_and_a_dropped_key_moves_it(
        self, sdpa_shipped
    ):
        """The additive is read, not merely built.

        Two numbers from one set of operands. An all-true row converts to an
        additive of zeros, so it must leave the answer where the maskless call
        left it; dropping half the keys must move it far outside that. A route
        that built the additive and ignored it would pass the first and fail the
        second, and one that read the row with ``flag_gems``' own polarity would
        fail the first.
        """
        _, identical, informative, _ = sdpa_shipped
        assert identical <= _IDENTITY_BOUND, (
            f"an all-true key-valid row moved the answer by {identical:.9f}, "
            f"over the {_IDENTITY_BOUND} bound: the additive is not neutral"
        )
        assert informative >= _INFORMATIVE_BOUND, (
            f"dropping half the keys moved the answer by only "
            f"{informative:.6f}, under the {_INFORMATIVE_BOUND} floor: the "
            "additive is being built and not read"
        )

    def test_the_route_can_be_switched_off(self, sdpa_switched_off):
        """``FLAGOS_OP_scaled_dot_product_attention=none`` takes every call off it.

        The switch is what makes the route reversible in the field without a
        rebuild, and it is the control arm of the measurement: with it set, no
        call reaches the kernel and the composite's math decomposition answers
        them all.
        """
        report, _, _, census = sdpa_switched_off
        assert not census, (
            f"{len(census)} call(s) still reached the FlagGems kernel with the "
            f"route switched off: {[s for s, _ in census]}"
        )
        for tag, (routed, diff, device, _) in sorted(report.items()):
            assert routed == 0, f"{tag}: {routed} call(s) with the route off"
            assert device == DEVICE, f"{tag}: the result is on {device}"
            if diff == diff:
                assert diff <= _DIFF_BOUND, (
                    f"{tag}: the math decomposition differs from the host "
                    f"reference by {diff:.6f}, over the {_DIFF_BOUND} bound"
                )
