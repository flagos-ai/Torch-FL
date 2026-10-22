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

"""The MetaX ``scaled_dot_product_attention`` FlagGems route, in one file.

This is the operator-behaviour half of the MetaX FlagGems coverage that used to
live in ``test_metax_flaggems.py``. The routing, execution and exclusion cases
of that file asserted the FlagGems integration itself and are now expressed once
for every platform by ``tests/integration/flaggems/``; what is left here is
MetaX-specific operator behaviour, because ``scaled_dot_product_attention`` is
routed per shape rather than per conf key.

The FlagGems SDPA route is the one conf entry the dispatcher does not select for
itself. ``csrc/aten/sdp_choice_stub.cc`` registers one PrivateUse1 kernel for the
composite under every conf, and that kernel reads ``GetBackendForOp`` to decide
between the FlagGems callable and the boxing kernel -- so the conf text cannot
say which arm a call took (both arms are served by the same key) and neither can
the dispatch log (there is no route to print; the kernel *is* the route).

The file is collected by the same ``-m "flaggems and main_ops"`` group every
other platform runs and skips itself off a MetaX boxing build, so the case set
is identical everywhere rather than stated per platform.

Usage:
    pytest tests/integration/ops/test_metax_sdpa_route.py -v
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

pytestmark = [pytest.mark.flaggems, pytest.mark.main_ops]


def _active_conf() -> str:
    return torch_fl.backend_config_path()


def _require_metax_flaggems() -> None:
    """Skip unless this is a MetaX boxing build with a usable FlagGems runtime."""
    if not _active_conf().endswith("backends_metax.conf"):
        pytest.skip(
            "the MetaX SDPA route is stated in backends_metax.conf; this "
            f"run selected {_active_conf() or '<no conf>'}"
        )
    if torch_fl.flagos.device_count() < 1:
        pytest.skip("MetaX device is unavailable")
    try:
        import triton

        import flag_gems
    except Exception as exc:  # noqa: BLE001 - any import failure means skip
        pytest.skip(f"FlagGems MetaX runtime is unavailable: {exc}")
    if "metax" not in triton.backends.backends:
        pytest.skip("the installed Triton does not provide the MetaX backend")
    print(f"FlagGems: {flag_gems.__version__} ({flag_gems.__file__})")


# The oracle is the callable the FlagGems arm calls.
# `PythonOpCache::GetFunc` resolves `flag_gems.scaled_dot_product_attention` with
# a package import plus a `getattr`, memoized per qualname for the life of the
# process, so a counter installed on the package before the first call is the
# callable the route calls -- and a fresh interpreter is what makes "before the
# first call" true.
_SDPA_ORACLE = (
    "import flag_gems\n"
    "_sdpa_calls = []\n"
    "_sdpa_real = flag_gems.scaled_dot_product_attention\n"
    "def _sdpa_counting(*args, **kwargs):\n"
    "    _sdpa_calls.append(tuple(tuple(t.shape) for t in args))\n"
    "    return _sdpa_real(*args, **kwargs)\n"
    "flag_gems.scaled_dot_product_attention = _sdpa_counting\n"
    "\n"
    # The host reference moves the arguments back rather than rebuilding them, so
    # both arms of every case see the same values. A float mask is additive and
    # moves as a float; a bool mask is a keep/drop mask and only keeps its meaning
    # if it keeps its dtype, so it moves as a bool (the CPU reference reads one).
    "def host_ref(q, k, v, kw):\n"
    "    def move(t):\n"
    "        if not torch.is_tensor(t):\n"
    "            return t\n"
    "        if t.dtype == torch.bool:\n"
    "            return t.cpu()\n"
    "        return t.cpu().float()\n"
    "    moved = {n: move(t) for n, t in kw.items()}\n"
    "    return torch.nn.functional.scaled_dot_product_attention(\n"
    "        q.float().cpu(), k.float().cpu(), v.float().cpu(), **moved)\n"
    "\n"
    "def run_case(tag, build, kw):\n"
    "    q, k, v = build()\n"
    "    ref = host_ref(q, k, v, kw)\n"
    "    before = len(_sdpa_calls)\n"
    "    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, **kw)\n"
    "    routed = len(_sdpa_calls) - before\n"
    "    diff = (out.float().cpu() - ref).abs().max().item()\n"
    "    print(f'RESULT {tag} routed={routed} maxdiff={diff:.6f} "
    "device={out.device}')\n"
)

# The cases, and what each one is. `joint()` is the shape Qwen-Image-2512's joint
# attention has -- bf16, 4-D, head_dim 128, seq 1024, no mask -- and it is one of
# the five shapes in the set the envelope admits. Every other case differs from
# `joint()` in one respect, so a failing assertion names the clause that let a
# call through; the admitted cases are held at different batches, sequence
# lengths or head dims from each other so the SHAPES line tells them apart.
#
# `joint_masked()` is the reason the mask clause exists: Qwen-Image-2.1's prefill
# processor passes the joint key-valid row -- 4-D bool, one entry per key -- on
# every call it makes (transformer_qwenimage21.py:478-560). It is built at a
# different batch from `joint()` so the two are told apart in the SHAPES line,
# and it drops keys rather than masking nothing, because an all-True row would
# exercise the clause without exercising the polarity the conversion has to get
# right.
#
# The other three admitted cases are the widening for issue #394: a head dim
# other than 128, a sequence shorter than the joint attention's, and fp16. Each
# is in the probe because `FlagGemsEligible` admits it, and each bound the
# widening touched is pinned from both sides -- `head_dim8` and `head_dim256`
# are the first values outside the head-dim bounds, `float32` is the dtype the
# kernel raises on, and `masked_seq512` is the sequence below the one floor the
# widening kept.
#
# The `mask*` cases below it are the other side of the same clause: each supplies
# a mask a *caller* may legitimately pass and that the route must refuse. The
# additive case is 2-D and all-zero, i.e. numerically a no-op, so what it checks
# is that an additive mask at all keeps the call off the route -- a float mask is
# broadcast by the kernel, not converted by this route. `mask_4d_bool` has the
# right dtype and rank but one entry per (query, key) pair rather than per key,
# which is 1 GiB materialised at the shape the route serves; `mask_row_f32` has
# the right shape but is the wrong dtype, and a float row carries no keep/drop
# meaning for the conversion to preserve.
_SDPA_CASES = (
    "def joint():\n"
    "    return tuple(\n"
    "        torch.randn(1, 2, 1024, 128, device=DEVICE).to(torch.bfloat16)\n"
    "        for _ in range(3))\n"
    'run_case("eligible", joint, {})\n'
    "def joint_masked():\n"
    "    return tuple(\n"
    "        torch.randn(2, 2, 1024, 128, device=DEVICE).to(torch.bfloat16)\n"
    "        for _ in range(3))\n"
    "def key_row(n):\n"
    "    row = torch.ones(1, 1, 1, n, device=DEVICE, dtype=torch.bool)\n"
    "    row[..., -128:] = False\n"
    "    return row\n"
    'run_case("masked", joint_masked, {"attn_mask": key_row(1024)})\n'
    'run_case("head_dim64", lambda: tuple(\n'
    "    torch.randn(1, 2, 1024, 64, device=DEVICE).to(torch.bfloat16)\n"
    "    for _ in range(3)), {})\n"
    'run_case("seq512", lambda: tuple(\n'
    "    torch.randn(1, 2, 512, 128, device=DEVICE).to(torch.bfloat16)\n"
    "    for _ in range(3)), {})\n"
    'run_case("float16", lambda: tuple(\n'
    "    torch.randn(3, 2, 1024, 128, device=DEVICE).to(torch.float16)\n"
    "    for _ in range(3)), {})\n"
    'run_case("float32", lambda: tuple(\n'
    "    torch.randn(1, 2, 1024, 128, device=DEVICE) for _ in range(3)), {})\n"
    'run_case("rank2", lambda: tuple(\n'
    "    torch.randn(1024, 128, device=DEVICE).to(torch.bfloat16)\n"
    "    for _ in range(3)), {})\n"
    'run_case("head_dim8", lambda: tuple(\n'
    "    torch.randn(1, 2, 1024, 8, device=DEVICE).to(torch.bfloat16)\n"
    "    for _ in range(3)), {})\n"
    'run_case("head_dim256", lambda: tuple(\n'
    "    torch.randn(1, 2, 1024, 256, device=DEVICE).to(torch.bfloat16)\n"
    "    for _ in range(3)), {})\n"
    'run_case("causal", joint, {"is_causal": True})\n'
    'run_case("masked_seq512", lambda: tuple(\n'
    "    torch.randn(2, 2, 512, 128, device=DEVICE).to(torch.bfloat16)\n"
    "    for _ in range(3)), {'attn_mask': key_row(512)})\n"
    'run_case("mask_additive", joint,\n'
    '         {"attn_mask": torch.zeros(1024, 1024, device=DEVICE,\n'
    "                                   dtype=torch.bfloat16)})\n"
    'run_case("mask_4d_bool", joint,\n'
    '         {"attn_mask": torch.ones(1, 2, 1024, 1024, device=DEVICE,\n'
    "                                   dtype=torch.bool)})\n"
    'run_case("mask_row_f32", joint,\n'
    '         {"attn_mask": torch.zeros(1, 1, 1, 1024, device=DEVICE)})\n'
    "print(f'SHAPES {_sdpa_calls}')\n"
)

# Case tag -> (calls that may reach the FlagGems kernel, the clause that decides
# it). The count is per case, so a run where an ineligible case reaches the
# kernel fails at that case rather than moving a global total.
_SDPA_CASES_EXPECTED = {
    "eligible": (1, "bf16, 4-D, head_dim 128, seq 1024, no mask"),
    "masked": (1, "the same class, with the (1,1,1,KV) bool key row 2.1 passes"),
    "head_dim64": (1, "head_dim 64, inside the route's head_dim bounds"),
    "seq512": (1, "seq 512, which the route no longer floors"),
    "float16": (1, "fp16, the second dtype the route was measured on"),
    "float32": (0, "float32, whose tile the kernel reports OutOfResources on"),
    "rank2": (0, "2-D input, and the route requires the 4-D call form"),
    "head_dim8": (0, "head_dim 8, under the 16 a tile's BLOCK_N bottoms out at"),
    "head_dim256": (0, "head_dim 256, over the 128 whose tile still fits"),
    "causal": (0, "is_causal, which the FlagGems entry point does not take"),
    "masked_seq512": (0, "that key row at seq 512, under the masked route floor"),
    "mask_additive": (0, "an additive mask, which the route does not convert"),
    "mask_4d_bool": (0, "a bool mask with one entry per (query, key) pair"),
    "mask_row_f32": (0, "a per-key row in float32, whose polarity is undefined"),
}

# Max |device - host| for a 16-bit route against a float32 host reference. The
# bf16 rounding of the inputs already moves the logits by ~2e-2; softmax turns
# that into far less on an output of magnitude ~3e-2. The fp16 case sits an order
# below this, at its own rounding floor rather than the bf16 one.
_SDPA_DIFF_BOUND = 2e-2

_SDPA_RESULT = re.compile(
    r"^RESULT (\S+) routed=(\d+) maxdiff=([0-9.eE+-]+) device=(\S+)$",
    re.MULTILINE,
)
_SDPA_SHAPES = re.compile(r"^SHAPES (.+)$", re.MULTILINE)


def _run_sdpa_cases(
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run every case in one fresh interpreter and print a parseable report."""
    code = (
        "import torch\n"
        "import torch_fl\n"
        f"DEVICE = {DEVICE!r}\n"
        "torch.manual_seed(1729)\n"
        f"{_SDPA_ORACLE}"
        f"{_SDPA_CASES}"
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
    """(per-case (routed, maxdiff, device), the shapes the kernels saw)."""
    report = {
        tag: (int(routed), float(diff), device)
        for tag, routed, diff, device in _SDPA_RESULT.findall(result.stdout)
    }
    assert set(report) == set(_SDPA_CASES_EXPECTED), (
        "the probe did not report every case: "
        f"{sorted(report)} against {sorted(_SDPA_CASES_EXPECTED)}\n"
        f"{result.stdout}\n{result.stderr}"
    )
    shapes = _SDPA_SHAPES.findall(result.stdout)
    assert len(shapes) == 1, f"expected one SHAPES line\n{result.stdout}"
    return report, ast.literal_eval(shapes[0])


@pytest.fixture(scope="module")
def sdpa_shipped():
    """The shipped ``backends_metax.conf``, with no environment override."""
    _require_metax_flaggems()
    result = _run_sdpa_cases()
    assert result.returncode == 0, f"sdpa probe failed\n{result.stderr}"
    return _sdpa_report(result)


@pytest.fixture(scope="module")
def sdpa_switched_off():
    """The same cases with the documented runtime switch set."""
    _require_metax_flaggems()
    result = _run_sdpa_cases({"FLAGOS_OP_scaled_dot_product_attention": "cuda"})
    assert result.returncode == 0, f"sdpa probe failed\n{result.stderr}"
    return _sdpa_report(result)


class TestMetaXSdpaRoute:
    """``scaled_dot_product_attention`` is routed on the envelope, not the key.

    The route exists because the MetaX boxing kernel has no memory-efficient
    attention to fall back on (``Torch was not compiled with memory efficient
    attention``), so the composite materialises the full attention matrix. The
    FlagGems ``_attn_fwd`` kernel does not, which is what the route buys; the
    envelope in ``FlagGemsEligible`` is what bounds the risk of taking it. The
    two things asserted here are the two that can go wrong: a call the envelope
    admits must take the route, and a call it refuses must not.
    """

    @pytest.mark.parametrize("tag", sorted(_SDPA_CASES_EXPECTED))
    def test_route_follows_the_envelope(self, sdpa_shipped, tag):
        report, _ = sdpa_shipped
        want, why = _SDPA_CASES_EXPECTED[tag]
        routed, diff, device = report[tag]
        assert routed == want, (
            f"{tag} ({why}): {routed} call(s) reached the FlagGems kernel, "
            f"expected {want}. Either FlagGemsEligible in "
            "csrc/aten/sdp_choice_stub.cc admits this shape, or it admits "
            "nothing and the conf does not route the op"
        )
        assert diff <= _SDPA_DIFF_BOUND, (
            f"{tag} ({why}): the result differs from the host reference by "
            f"{diff:.6f}, over the {_SDPA_DIFF_BOUND} bound"
        )
        assert device == DEVICE, f"{tag}: the result is on {device}, not {DEVICE}"

    def test_only_the_admitted_shapes_reach_the_kernel(self, sdpa_shipped):
        """The shapes the FlagGems kernel saw, in order, across all fourteen cases.

        The count above says how many calls took the route; this says they were
        the calls that were meant to, on the arguments the envelope is written
        around. Five shapes are admitted -- the unmasked joint attention, the
        same class carrying 2.1's key-valid row, and the three the widening for
        issue #394 added -- and no two of them share a shape, so a call that
        reached the kernel on the wrong case is visible here rather than absorbed
        into the count.
        """
        _, shapes = sdpa_shipped
        # One call each, on their own q, k and v: the probe records each call as
        # a three-tuple of argument shapes, so each admitted case contributes a
        # single three-tuple to the list.
        expected = [
            ((1, 2, 1024, 128),) * 3,
            ((2, 2, 1024, 128),) * 3,
            ((1, 2, 1024, 64),) * 3,
            ((1, 2, 512, 128),) * 3,
            ((3, 2, 1024, 128),) * 3,
        ]
        assert shapes == expected, (
            f"the FlagGems SDPA kernel was called with {shapes}; only the five "
            "admitted cases may reach it, and only with their own q, k, v"
        )

    def test_off_switch_returns_every_call_to_the_boxing_route(self, sdpa_switched_off):
        """``FLAGOS_OP_scaled_dot_product_attention=cuda`` is the whole switch.

        The same conf is in use in both arms -- only this variable differs -- so
        this is also the check that the eligible case's route came from the conf
        key and not from the shape alone.
        """
        report, shapes = sdpa_switched_off
        assert shapes == [], f"the FlagGems kernel ran with the switch off: {shapes}"
        off = {tag: row[0] for tag, row in report.items()}
        assert set(off.values()) == {0}, f"routes with the switch off: {off}"
