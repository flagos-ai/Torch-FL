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

"""Operators measured to fail on a FlagGems path must stay off it.

The set is per platform and lives in ``flaggems_caps.FLAGGEMS_PINNED_OFF``: each
entry was moved off the FlagGems path after an A/B on that hardware in which the
boxing or native arm matched ATen and the FlagGems arm did not. The test reads
the active conf and asserts none of them is routed to a FlagGems backend, so a
re-measurement that lifts a hold is a deliberate conf edit -- and a hold that is
lost by accident fails here, on the chip that recorded it, instead of silently
resurfacing as a wrong number in a model.

This is a conf-text assertion, so it needs no device and runs wherever a named
``backends_<platform>.conf`` is in use. The hardware reproduction of the shapes
that motivated each hold stays with the operator tests that own those shapes
(``tests/integration/ops/test_dcu_flaggems_pinned_routes.py`` is the model).
"""

import pytest

from flaggems_caps import (
    FLAGGEMS_BACKENDS,
    FLAGGEMS_PINNED_OFF,
    accel_label,
    conf_path,
    load_routes,
)


def test_pinned_operators_are_not_routed_to_flaggems():
    accel = accel_label()
    if not accel:
        pytest.fail("the active conf is not a named backends_<platform>.conf")
    assert accel in FLAGGEMS_PINNED_OFF, (
        f"{accel} has no FlagGems exclusion row; record its measured holds "
        "or an explicit empty tuple"
    )
    pinned = FLAGGEMS_PINNED_OFF[accel]

    routes = load_routes()
    routed_back = []
    missing = []
    for op in pinned:
        if op not in routes:
            missing.append(op)
        elif routes[op] in FLAGGEMS_BACKENDS:
            routed_back.append((op, routes[op]))
    assert not missing, (
        f"{conf_path()} does not list these held operators at all, so the hold "
        f"cannot be read: {missing}"
    )
    assert not routed_back, (
        f"{accel} routes {routed_back} to a FlagGems backend; each of these was "
        "measured to answer wrongly, raise, or refuse a dtype on the FlagGems "
        "path. Lift the hold deliberately -- in flaggems_caps.FLAGGEMS_PINNED_OFF "
        "and in the support report -- or keep the operator off the route"
    )


def test_every_pinned_platform_has_a_recorded_row():
    """A hold with no route-count row would be a measurement with no home."""
    from flaggems_caps import EXPECTED_FLAGGEMS_ROUTES

    unrecorded = sorted(set(FLAGGEMS_PINNED_OFF) - set(EXPECTED_FLAGGEMS_ROUTES))
    assert not unrecorded, (
        f"{unrecorded} carry a pinned-off set but no FlagGems route-count row"
    )
