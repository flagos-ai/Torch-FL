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

"""Check the shared vendor SDPA coverage table without accelerator hardware."""

import copy
import sys
import pytest

from cpu_device_init import load_on_cpu


with load_on_cpu("vendor_lanes/sdpa_lane_caps.py") as sdpa_lane_caps:
    shared_sdpa_caps = sys.modules["shared_sdpa_caps"]
    COMMON_BY_TAG = shared_sdpa_caps.COMMON_BY_TAG
    COMMON_TAGS = shared_sdpa_caps.COMMON_TAGS
    SHARED_ROUTES = shared_sdpa_caps.SHARED_ROUTES
    common_lane = sys.modules["vendor_caps"].common_lane


def test_every_platform_runs_the_complete_shared_case_set():
    assert set(sdpa_lane_caps.SDPA_LANE_ROWS) == set(SHARED_ROUTES)
    for row in sdpa_lane_caps.SDPA_LANE_ROWS.values():
        assert row.shared_cases == COMMON_TAGS
        assert set(row.common_lanes) == set(COMMON_TAGS)


def test_gcu_declares_the_lane_reached_by_each_shared_format():
    row = sdpa_lane_caps.SDPA_LANE_ROWS["gcu"]
    assert common_lane(row, COMMON_BY_TAG["math"]) == "math"
    for tag in COMMON_TAGS:
        assert common_lane(row, COMMON_BY_TAG[tag]) == "math"


def _replace_rows(monkeypatch, rows):
    monkeypatch.setattr(sdpa_lane_caps, "SDPA_LANE_ROWS", rows)
    sdpa_lane_caps.check_rows()


def test_a_row_must_state_a_lane_for_every_shared_case(monkeypatch):
    rows = copy.deepcopy(sdpa_lane_caps.SDPA_LANE_ROWS)
    del rows["cuda"].common_lanes["causal"]
    with pytest.raises(AssertionError, match="name a lane for every shared SDPA"):
        _replace_rows(monkeypatch, rows)


def test_a_row_must_run_every_shared_case(monkeypatch):
    rows = copy.deepcopy(sdpa_lane_caps.SDPA_LANE_ROWS)
    rows["cuda"] = rows["cuda"]._replace(shared_cases=COMMON_TAGS[:-1])
    with pytest.raises(AssertionError, match="carry every shared SDPA case"):
        _replace_rows(monkeypatch, rows)
