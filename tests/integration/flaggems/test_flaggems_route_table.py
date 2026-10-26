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

from pathlib import Path

import pytest

from flaggems_caps import (
    EXPECTED_FLAGGEMS_ROUTES,
    EXPECTED_SDPA_ROUTE,
    FLAGGEMS_ROUTING_CALLS,
    SEEDED_DRAWOPS,
    accel_label,
    conf_path,
    conf_route,
    flaggems_coverage_count,
    routes_to_flaggems,
)


def _named_accel_or_skip() -> str:
    accel = accel_label()
    if not accel:
        pytest.skip("the active conf is not a named backends_<platform>.conf")
    return accel


@pytest.mark.anyplatform
def test_flaggems_route_count_matches_the_measured_one():
    accel = _named_accel_or_skip()
    assert accel in EXPECTED_FLAGGEMS_ROUTES, (
        f"{accel} has no measured FlagGems route count; add its row to "
        "flaggems_caps.EXPECTED_FLAGGEMS_ROUTES with the conf that was measured"
    )
    measured = flaggems_coverage_count()
    assert measured == EXPECTED_FLAGGEMS_ROUTES[accel], (
        f"{conf_path()} routes {measured} operators to FlagGems that this suite "
        f"can call (less flaggems_caps.SEEDED_DRAWOPS); {accel} is recorded at "
        f"{EXPECTED_FLAGGEMS_ROUTES[accel]}. If the route table really moved, "
        "update the count and docs/reference/operator-support.md"
    )


@pytest.mark.anyplatform
def test_the_seeded_draw_ops_are_never_boxed():
    """No day-0 variant sends the seeded-drawout ops: ``boxing`` or ``none``.

    A ``boxing`` row for them is the one route that cannot honour a seed on a
    CPU-torch wheel: the boxing kernel is ATen's ``native_dropout``, which draws
    from a generator ``torch.manual_seed`` cannot write. A ``none`` row is no
    kernel at all. This is the invariant behind ``SEEDED_DRAWOPS``: the count
    above leaves those rows out, so this test is where they are pinned.
    """
    import torch_fl
    from torch_fl._platform import platform_for

    conf_dir = Path(torch_fl.__file__).parent / "configs"
    unrouted = []
    for path in sorted(conf_dir.glob("backends_*_boxing.conf")):
        # The generated variants only: ``platform_for`` names the vendors that
        # have one, and the rest of this directory's ``*_boxing`` files are none.
        if not platform_for(path.stem[len("backends_") : -len("_boxing")]):
            continue
        routes = dict.fromkeys(SEEDED_DRAWOPS)
        with open(path) as f:
            for line in f:
                name, sep, value = line.split("#")[0].partition("=")
                if sep and name.strip() in routes:
                    routes[name.strip()] = value.strip()
        unrouted += [
            (path.name, op, route)
            for op, route in routes.items()
            if route in ("boxing", "none")
        ]
    assert not unrouted, f"the seeded-drawout ops have no seedable route in {unrouted}"


@pytest.mark.anyplatform
def test_sdpa_route_matches_the_recorded_one():
    """The composite's route is text, and text is all that can be pinned here.

    ``scaled_dot_product_attention`` is served by one PrivateUse1 kernel under
    every conf -- the kernel reads the key itself to decide between the FlagGems
    callable and the boxed CUDA composite -- so the dispatch log cannot say which
    arm a call took. The conf's spelling is the statement, and the numerics of
    each arm are the property of the platform's own operator tests.
    """
    accel = _named_accel_or_skip()
    assert accel in EXPECTED_SDPA_ROUTE, (
        f"{accel} has no recorded scaled_dot_product_attention route"
    )
    assert conf_route("scaled_dot_product_attention") == EXPECTED_SDPA_ROUTE[accel], (
        f"{conf_path()} routes scaled_dot_product_attention to "
        f"{conf_route('scaled_dot_product_attention')!r}; {accel} was recorded at "
        f"{EXPECTED_SDPA_ROUTE[accel]!r}"
    )


@pytest.mark.anyplatform
def test_a_chip_that_claims_flaggems_measures_at_least_one_call():
    """A platform with FlagGems routes must route some operator in this table.

    Otherwise the two files that do need a device would skip their way through
    the run and report success for having measured nothing.
    """
    accel = _named_accel_or_skip()
    if flaggems_coverage_count() == 0:
        pytest.skip(f"{accel} routes no operator in this suite to FlagGems")
    covered = sorted(op for op in FLAGGEMS_ROUTING_CALLS if routes_to_flaggems(op))
    assert covered, (
        f"{accel} claims {flaggems_coverage_count()} FlagGems routes but none of "
        "them is in flaggems_caps.FLAGGEMS_ROUTING_CALLS, so this suite measures "
        "nothing on this chip"
    )


@pytest.mark.anyplatform
def test_a_recorded_count_matches_what_this_suite_can_call():
    """A recorded count and the call table agree on whether anything is callable.

    This is what makes the skip in ``test_a_chip_that_claims_flaggems_...``
    above a statement about the platform rather than a hole in the guard: a
    platform whose recorded count is 0 must route no operator in
    ``FLAGGEMS_ROUTING_CALLS``, and one whose count is not 0 must route at
    least one. ``nvidia_boxing`` is the only variant on the 0 side today; it is
    described by this relation, not named here, so a second one shows up as a
    pass rather than a failure.
    """
    accel = _named_accel_or_skip()
    if accel not in EXPECTED_FLAGGEMS_ROUTES:
        pytest.skip(f"{accel} has no measured FlagGems route count")
    covered = sorted(op for op in FLAGGEMS_ROUTING_CALLS if routes_to_flaggems(op))
    if covered:
        assert EXPECTED_FLAGGEMS_ROUTES[accel] > 0, (
            f"{accel} routes {len(covered)} call-table operators to FlagGems "
            "but its recorded count is 0"
        )
    else:
        assert EXPECTED_FLAGGEMS_ROUTES[accel] == 0, (
            f"{accel} is recorded at {EXPECTED_FLAGGEMS_ROUTES[accel]} FlagGems "
            "routes but routes none of flaggems_caps.FLAGGEMS_ROUTING_CALLS"
        )


def test_the_route_tables_cover_the_same_platforms():
    """The three tables are one statement per platform, so their keys must agree."""
    assert set(EXPECTED_FLAGGEMS_ROUTES) == set(EXPECTED_SDPA_ROUTE), (
        "every platform with a FlagGems route count needs an SDPA route recorded, "
        "and the other way round"
    )
