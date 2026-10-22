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

The command is byte-identical on all ten configured platforms. Every row runs
the same shared cases through its configured arm and adds the calls that are
meaningful only on its lane. A missing row or a shared case that a row fails to
include is a configuration failure; this suite never skips a configured
platform.
"""

from __future__ import annotations

import pytest

from sdpa_lane_caps import SDPA_LANE_ROWS
from shared_sdpa_caps import COMMON_TAGS
from vendor_caps import accel_label, run_case

pytestmark = pytest.mark.anyplatform


def _row_for_this_platform():
    """The row for the conf this wheel resolved, or a configuration failure."""
    accel = accel_label()
    if not accel:
        pytest.fail("the active conf is not a named backends_<platform>.conf")
    row = SDPA_LANE_ROWS.get(accel)
    if row is None:
        pytest.fail(f"the SDPA lane table has no row for {accel}")
    return row


def _case_tags() -> list[str]:
    """This platform's case tags, read at collection so the ids name the cases."""
    row = SDPA_LANE_ROWS.get(accel_label())
    if row is None:
        return ["missing-sdpa-row"]
    return [case.tag for case in row.cases]


_CASE_TAGS = _case_tags()


@pytest.mark.parametrize("tag", _CASE_TAGS)
def test_the_vendor_lane_holds(tag):
    """Every claim ``sdpa_lane_caps`` states about this platform's lane.

    The Row drives the case: which lane the dispatch has to show, whether the
    call runs under ``no_grad``, what the answer is compared against, and where
    the case is about a mask that must be the identity. The case supplies the
    clause the failure names, so a mask spelling that stops being read fails as
    itself rather than as a number that moved.
    """
    row = _row_for_this_platform()
    case = next(
        (case for case in row.cases if case.tag == tag),
        None,
    )
    if case is None:
        pytest.fail(f"the SDPA row has no case {tag}")
    run_case(row, case)


def test_the_platform_names_the_lane_it_carries():
    """The row this platform declared, in the job's own log.

    Read from the conf the wheel resolved rather than from a platform name, so a
    lane that stops being declared on a chip fails here instead of showing up
    as a case set that quietly shrank on one runner.
    """
    row = _row_for_this_platform()
    print(
        f"vendor SDPA lane on {accel_label()}: lane={row.lane} "
        f"cases={len(row.cases)} measured_on={row.measured_on}"
        f"{'' if row.revalidated else ' (bounds not revalidated)'}"
    )
    tags = [case.tag for case in row.cases]
    assert len(set(tags)) == len(tags), f"two cases share a tag: {sorted(tags)}"
    assert row.shared_cases == COMMON_TAGS, (
        f"{accel_label()} does not carry every shared SDPA case"
    )
