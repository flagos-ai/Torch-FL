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

"""The vendor SDPA lanes, on one case set, run wherever a lane is declared.

Two files used to carry this: ``tests/integration/ops/test_ascend_sdpa_mask_bias.py``
(what ACLNN does with a mask and a bias, asserted by the answer) and
``tests/integration/ops/test_gcu_sdpa_mask.py`` (which of GCU's two lanes a call
takes, asserted by the dispatch census as well as the answer). A third chip
reaching the operator would have written a third file, and the cases the first
two agreed on would have been stated three times. They are stated once now, in
``sdpa_lane_caps.py``; what genuinely differs per chip is a row of that table.

The command is byte-identical on all eight platforms. Where the conf reaches
``scaled_dot_product_attention`` through a vendor lane of its own the row's whole
case set runs; where it does not, the single collected test skips and says so,
and the skip is recorded in ``tests/integration/platform_gates.py`` and
documented in ``docs/reference/platform-capability-matrix.md``. The point of one
command is that a chip cannot quietly run a subset: the case set a platform runs
is the case set every platform carrying the lane runs, and the number is in the
summary line the conftest prints.
"""

from __future__ import annotations

import pytest

from sdpa_lane_caps import SDPA_LANE_ROWS
from vendor_caps import accel_label, run_case

pytestmark = pytest.mark.anyplatform


def _row_for_this_platform():
    """The row for the conf this wheel resolved, or a skip naming its route."""
    accel = accel_label()
    if not accel:
        pytest.skip("the active conf is not a named backends_<platform>.conf")
    row = SDPA_LANE_ROWS.get(accel)
    if row is None:
        pytest.skip(
            f"backends_{accel}.conf reaches scaled_dot_product_attention through "
            f"no vendor lane of its own, so this case set has no row for {accel}; "
            f"it runs on {', '.join(sorted(SDPA_LANE_ROWS))}. Where a conf routes "
            "the operator to a FlagGems backend, that route is asserted by the "
            "flaggems_conformance step instead"
        )
    return row


def _case_tags() -> list[str | None]:
    """This platform's case tags, read at collection so the ids name the cases.

    A platform with no row gets one placeholder, so the run reports a skip that
    states why instead of a command that collected nothing at all.
    """
    row = SDPA_LANE_ROWS.get(accel_label())
    if row is None:
        return [None]
    return [case.tag for case in row.cases]


_CASE_TAGS: list[str | None] = _case_tags()


@pytest.mark.parametrize("tag", _CASE_TAGS, ids=lambda tag: tag or "no-vendor-lane")
def test_the_vendor_lane_holds(tag):
    """Every claim ``sdpa_lane_caps`` states about this platform's lane.

    The Row drives the case: which lane the dispatch has to show, whether the
    call runs under ``no_grad``, what the answer is compared against, and where
    the case is about a mask that must be the identity. The case supplies the
    clause the failure names, so a mask spelling that stops being read fails as
    itself rather than as a number that moved.
    """
    row = _row_for_this_platform()
    case = next(case for case in row.cases if case.tag == tag)
    run_case(row, case)


def test_the_platform_names_the_lane_it_carries():
    """The row this platform declared, in the job's own log.

    Read from the conf the wheel resolved rather than from a platform name, so a
    lane that stops being declared on a chip shows up as this test skipping --
    with the conf's route in the reason -- instead of as a case set that quietly
    shrank on one runner.
    """
    row = _row_for_this_platform()
    print(
        f"vendor SDPA lane on {accel_label()}: lane={row.lane} "
        f"cases={len(row.cases)} measured_on={row.measured_on}"
        f"{'' if row.revalidated else ' (bounds not revalidated)'}"
    )
    tags = [case.tag for case in row.cases]
    assert len(set(tags)) == len(tags), f"two cases share a tag: {sorted(tags)}"
