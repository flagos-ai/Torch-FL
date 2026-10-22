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

"""The routing half of the contract: each operator dispatches where the conf says.

For every operator in the shared routing table, this runs the
call on the device in a fresh child with ``FLAGOS_LOG=dispatch,fallback`` and
asserts the child's
``[flagos dispatch] <op> -> <backend>`` line names exactly the backend the active
conf routes the operator to. A ``none`` route has to show the matching
``[flagos cpu_fallback] <op>`` line instead, and an unlisted route has to reach
FlagGems, the dispatcher's default.

The public wrappers in the execution table run in the same child, but their
dispatch logs can name a different operator. That execution, and a failure
inside it, is checked separately from the route of the named dispatcher row.

Per-operator coverage is allowed to differ, so the case is always driven and
then checked against the conf: an op the conf routes to FlagGems must reach that
backend, and another spelling must reach the backend the conf names. A platform
with no FlagGems arm therefore validates its actual route, while every
configured CI platform still runs the full case set. A missing runtime, device,
conf row or dispatch line is a failure.
"""

import pytest

from flaggems_caps import (
    FLAGGEMS_EXECUTION_CALLS,
    FLAGGEMS_ROUTING_CALLS,
    accel_label,
    conf_route,
    route_reason,
)
from flaggems_probe import probe_routes

# The child runs every call in one interpreter, so the report is computed once
# and shared by the parametrized cases.
_PROBE: dict[str, dict] | None = None


def _probe() -> dict[str, dict]:
    global _PROBE
    if _PROBE is None:
        _PROBE = probe_routes()
    return _PROBE


@pytest.mark.parametrize("op", sorted(FLAGGEMS_ROUTING_CALLS))
def test_operator_dispatches_where_the_conf_routes_it(op: str):
    accel = accel_label()
    if not accel:
        pytest.fail("the active conf is not a named backends_<platform>.conf")

    entry = _probe()[op]
    assert entry["error"] is None, f"{op}: {entry['error']}"
    backend, why = route_reason(op)
    if backend is None:
        assert entry["dispatched_backend"] is None and entry["cpu_fallback"], (
            f"{op} ({why}) in the {accel} conf, but the dispatcher returned "
            f"backend={entry['dispatched_backend']!r}, "
            f"cpu_fallback={entry['cpu_fallback']}"
        )
    else:
        assert entry["dispatched_backend"] == backend, (
            f"{op} ({why}) dispatched to {entry['dispatched_backend']!r}; the "
            f"{accel} conf ({conf_route(op)!r}) resolves it to {backend!r}"
        )
        assert not entry["cpu_fallback"], (
            f"{op} reached {entry['dispatched_backend']!r} and also the CPU "
            f"fallback; the {accel} conf ({conf_route(op)!r}) resolves it to "
            f"{backend!r}"
        )


def test_public_wrapper_calls_execute():
    """Every platform runs the wrappers even when their logs name another op."""
    accel = accel_label()
    if not accel:
        pytest.fail("the active conf is not a named backends_<platform>.conf")
    entries = _probe()
    assert FLAGGEMS_EXECUTION_CALLS, "the shared execution table is empty"
    failures = [
        f"{op}: {entries[op]['error']}"
        for op in FLAGGEMS_EXECUTION_CALLS
        if entries[op]["error"] is not None
    ]
    assert not failures, (
        "the shared execution calls failed on " + accel + ":\n" + ("\n".join(failures))
    )
