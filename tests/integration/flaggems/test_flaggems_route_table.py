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

"""The route table itself: measured counts and the one exceptional route.

These are the assertions that need no device. They read the active conf as text,
which is the same file the dispatcher reads, so a route that moves is caught
here even when the operator it moved on is not in this suite's call table.

The count guard is the one that has to exist: a per-operator route assertion
cannot see a broad reroute being split into narrower ones while every operator
still reaches some backend, and that is exactly the change that silently reduces
how much of the model runs on FlagGems. The number is a measurement, so moving it
is a deliberate edit that also updates the support report.
"""

import pytest

from flaggems_caps import (
    EXPECTED_FLAGGEMS_ROUTES,
    EXPECTED_SDPA_ROUTE,
    FLAGGEMS_ROUTING_CALLS,
    accel_label,
    conf_path,
    conf_route,
    flaggems_route_count,
    routes_to_flaggems,
)


def _named_accel() -> str:
    accel = accel_label()
    if not accel:
        pytest.fail("the active conf is not a named backends_<platform>.conf")
    return accel


def test_flaggems_route_count_matches_the_measured_one():
    accel = _named_accel()
    assert accel in EXPECTED_FLAGGEMS_ROUTES, (
        f"{accel} has no measured FlagGems route count; add its row to "
        "flaggems_caps.EXPECTED_FLAGGEMS_ROUTES with the conf that was measured"
    )
    measured = flaggems_route_count()
    assert measured == EXPECTED_FLAGGEMS_ROUTES[accel], (
        f"{conf_path()} routes {measured} operators with the `flaggems` key; "
        f"{accel} was measured at {EXPECTED_FLAGGEMS_ROUTES[accel]}. If the route "
        "table really moved, update the count and docs/reference/operator-support.md"
    )


def test_sdpa_route_matches_the_recorded_one():
    """The composite's route is text, and text is all that can be pinned here.

    ``scaled_dot_product_attention`` is served by one PrivateUse1 kernel under
    every conf -- the kernel reads the key itself to decide between the FlagGems
    callable and the boxed CUDA composite -- so the dispatch log cannot say which
    arm a call took. The conf's spelling is the statement, and the numerics of
    each arm are the property of the platform's own operator tests.
    """
    accel = _named_accel()
    assert accel in EXPECTED_SDPA_ROUTE, (
        f"{accel} has no recorded scaled_dot_product_attention route"
    )
    assert conf_route("scaled_dot_product_attention") == EXPECTED_SDPA_ROUTE[accel], (
        f"{conf_path()} routes scaled_dot_product_attention to "
        f"{conf_route('scaled_dot_product_attention')!r}; {accel} was recorded at "
        f"{EXPECTED_SDPA_ROUTE[accel]!r}"
    )


def test_a_chip_that_claims_flaggems_measures_at_least_one_call():
    """The shared table has a FlagGems route wherever this conf has any.

    A chip with no FlagGems route still executes every routing call and checks
    it against that conf. Every configured chip must nevertheless state its
    measurement, so a missing row cannot pass for a measured zero.
    """
    accel = _named_accel()
    assert accel in EXPECTED_FLAGGEMS_ROUTES, (
        f"{accel} has no route-count row; record the number measured from its conf"
    )
    covered = sorted(op for op in FLAGGEMS_ROUTING_CALLS if routes_to_flaggems(op))
    assert bool(covered) == (flaggems_route_count() > 0), (
        f"{accel} has {flaggems_route_count()} FlagGems routes and covers "
        f"{covered}; add a route to the shared table so the measured path runs"
    )


def test_the_route_tables_cover_the_same_platforms():
    """The three tables are one statement per platform, so their keys must agree."""
    assert set(EXPECTED_FLAGGEMS_ROUTES) == set(EXPECTED_SDPA_ROUTE), (
        "every platform with a FlagGems route count needs an SDPA route recorded, "
        "and the other way round"
    )
