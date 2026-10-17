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

"""The FlagGems ``scaled_dot_product_attention`` route, on one case set.

This replaces the per-chip files that used to assert this route
(``test_dcu_flaggems_sdpa.py`` and ``test_metax_sdpa_route.py``, both under
``tests/integration/ops/``). Each drove the same probe against the same
envelope, differing only in the chip the probe ran on and in the numeric bounds
it was measured to hold, so the case set is stated once in ``sdpa_caps.py``, the
per-chip numbers are a row there, and this file runs the whole set wherever the
conf routes the operator to a FlagGems backend. A chip that starts routing it
adds a row and gets every case the other chips run.

Three things are asserted, because three things can go wrong:

1. **The gate.** A call the envelope admits has to take the route and a call it
   refuses has to not, per case, so a failure names the clause that let it
   through.
2. **The mask the kernel receives.** ``RouteMask`` does not forward the caller's
   bool row; it expands it into the fp32 additive the kernel can read. The
   kernel forms its block pointer as ``batch_id*stride(0) + head_id*stride(1) +
   offs_m*stride(2) + offs_n*stride(3)`` with the first three unbounded, and
   what keeps it inside the KV-long buffer is ``stride(i) * (size(i) - 1) == 0``
   for each of them -- not that the strides are all zero. Handing it the
   caller's byte layout instead reads past the buffer: NaN on a small shape, a
   KERNEL VMFault at the model's own.
3. **The answer.** The route agrees with the host within the chip's measured
   bound, an all-true key-valid row leaves the answer where the maskless call
   left it, and a row with dropped keys moves it far outside that.

The file also checks that the documented runtime switch
``FLAGOS_OP_scaled_dot_product_attention=cuda`` returns every call to the boxed
route, which is what makes the eligible cases' route a statement about the conf
key rather than about the shape alone.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

import pytest

import torch_fl

from flaggems_caps import EXPECTED_SDPA_ROUTE, accel_label
from sdpa_caps import (
    DEVICE,
    SDPA_CASES,
    SDPA_ROUTE_ROWS,
    expected_converted_masks,
    expected_routed_calls,
    probe_source,
)

_RESULT = re.compile(
    r"^RESULT (\S+) routed=(\d+) maxdiff=(\S+) device=(\S+) masks=(\[.*\])$",
    re.MULTILINE,
)
_IDENTICAL = re.compile(r"^IDENTICAL max\|d\|=([\d.eE+-]+)$", re.MULTILINE)
_INFORMATIVE = re.compile(r"^INFORMATIVE max\|d\|=([\d.eE+-]+)$", re.MULTILINE)
_ROUTED_MASKS = re.compile(r"^ROUTED-MASKS (.+)$", re.MULTILINE)


def _route_row():
    """The row for this chip, or a skip naming the conf's own decision."""
    accel = accel_label()
    if not accel:
        pytest.skip("the active conf is not a named backends_<platform>.conf")
    route = EXPECTED_SDPA_ROUTE.get(accel, "<no row>")
    if accel not in SDPA_ROUTE_ROWS:
        pytest.skip(
            f"backends_{accel}.conf routes scaled_dot_product_attention to "
            f"{route!r}, so there is no FlagGems SDPA route to assert on this "
            f"platform; only {sorted(SDPA_ROUTE_ROWS)} carry one"
        )
    return SDPA_ROUTE_ROWS[accel]


def _require_flaggems_runtime(row) -> None:
    """Skip unless this wheel can actually run the FlagGems SDPA kernel."""
    if torch_fl.flagos.device_count() < 1:
        pytest.skip("no flagos device is available")
    try:
        import triton

        import flag_gems
    except Exception as exc:  # noqa: BLE001 - any import failure means skip
        pytest.skip(f"the FlagGems runtime is not importable: {exc}")
    if row.triton_backend not in triton.backends.backends:
        pytest.skip(
            f"the installed Triton does not provide the {row.triton_backend} backend"
        )
    print(f"FlagGems: {flag_gems.__version__} ({flag_gems.__file__})")


def _run_cases(row, env_extra: dict[str, str] | None = None):
    """Run every case in one fresh interpreter and print a parseable report."""
    env = os.environ.copy()
    # The shipped conf is the arm without an override; an inherited value would
    # silently make every case the other arm.
    env.pop("FLAGOS_OP_scaled_dot_product_attention", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", probe_source(row.imports)],
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
    assert set(report) == {case.tag for case in SDPA_CASES}, (
        "the probe did not report every case: "
        f"{sorted(report)} against {sorted(case.tag for case in SDPA_CASES)}\n"
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
    """The shipped conf, with no environment override."""
    row = _route_row()
    _require_flaggems_runtime(row)
    result = _run_cases(row)
    assert result.returncode == 0, f"sdpa probe failed\n{result.stderr}"
    return _report(result)


@pytest.fixture(scope="module")
def sdpa_switched_off():
    """The same cases with the documented runtime switch set."""
    row = _route_row()
    _require_flaggems_runtime(row)
    result = _run_cases(row, {"FLAGOS_OP_scaled_dot_product_attention": "cuda"})
    assert result.returncode == 0, f"sdpa probe failed\n{result.stderr}"
    return _report(result)


class TestFlagGemsSdpaRoute:
    """``scaled_dot_product_attention`` is routed on the envelope, not the key.

    The route exists because any mask sent the boxed composite to its math
    decomposition, which materialises the full fp32 score matrix -- at the
    model's own shape 32 x 4096 x 4122 x 4 B, 2.16 GB per call. The flag_gems
    ``_attn_fwd`` kernel does not, which is what the route buys; the envelope in
    ``FlagGemsEligible`` and the conversion in ``RouteMask`` are what bound the
    risk of taking it.
    """

    @pytest.mark.parametrize("tag", [case.tag for case in SDPA_CASES])
    def test_route_follows_the_envelope(self, sdpa_shipped, tag):
        row = _route_row()
        case = next(c for c in SDPA_CASES if c.tag == tag)
        report, _, _, _ = sdpa_shipped
        routed, diff, device, masks = report[tag]
        assert routed == case.routed, (
            f"{tag} ({case.clause}): {routed} call(s) reached the FlagGems "
            f"kernel, expected {case.routed}. Either FlagGemsEligible in "
            "csrc/aten/sdp_choice_stub.cc admits this call, or it admits "
            "nothing and the conf does not route the op"
        )
        assert len(masks) == routed, (
            f"{tag} ({case.clause}): {routed} call(s) but {len(masks)} mask record(s)"
        )
        assert device == DEVICE, f"{tag}: the result is on {device}, not {DEVICE}"
        if diff == diff:  # not the dropout case, which has no reference
            assert diff <= row.diff_bound, (
                f"{tag} ({case.clause}): the result differs from the host "
                f"reference by {diff:.6f}, over the {row.diff_bound} bound "
                f"measured on {row.measured_on}"
            )

    def test_the_kernel_receives_the_converted_mask(self, sdpa_shipped):
        """Every routed call carried the additive ``RouteMask`` builds, or none.

        The census is the whole run, so this is asserted on the calls the kernel
        actually saw rather than on the mask cases alone: a routed call that
        forwarded the caller's ``(1,1,1,KV)`` bool, or a materialised
        ``(B,H,Q,KV)`` one, fails here whatever case it came from.

        The stride check is the load-bearing half. It is not that all three
        leading strides are 0 -- at batch 1 ``expand`` never touches axis 0, so
        that one keeps the buffer width and is harmless because the only index it
        can take is 0. What has to hold is that no index the kernel can form
        reaches past the KV-long buffer.
        """
        _, _, _, census = sdpa_shipped
        assert len(census) == expected_routed_calls(), (
            f"{len(census)} call(s) reached the FlagGems kernel, expected "
            f"{expected_routed_calls()}: {[s for s, _ in census]}"
        )
        converted = [m for _, m in census if m is not None]
        assert len(converted) == expected_converted_masks(), (
            f"expected {expected_converted_masks()} masked calls in the run, "
            f"got {census}"
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

        The two calls do not run the same kernel, so this is the chip's measured
        rounding bound and not bit equality. It is the whole claim the route
        makes about the model's ``key_valid``: the row the model passes carries
        no information, and the additive it converts to has to carry none
        either.
        """
        row = _route_row()
        _, identical, _, _ = sdpa_shipped
        assert identical <= row.identity_bound, (
            "the all-true key-valid row moved the result by "
            f"{identical:.9f}, over the {row.identity_bound} bound -- a row "
            "with every key valid is not supposed to drop or reweight any"
        )

    def test_a_dropped_key_does_change_the_answer(self, sdpa_shipped):
        """The additive is read, not merely built.

        Half the row is dropped in the identity block, and it lands two orders
        of magnitude above the all-true row's own movement, so this is the other
        half of the claim above: if the route were computing ``RouteMask`` and
        then discarding it, the all-true case would still pass and this one would
        not.
        """
        row = _route_row()
        _, _, informative, _ = sdpa_shipped
        assert informative > row.informative_bound, (
            "dropping half the key-valid row moved the result by only "
            f"{informative:.6f}, under the {row.informative_bound} floor -- "
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
        off = {tag: entry[0] for tag, entry in report.items()}
        assert set(off.values()) == {0}, f"routes with the switch off: {off}"
