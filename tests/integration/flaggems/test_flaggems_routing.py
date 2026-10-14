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

For every operator in ``FLAGGEMS_ROUTING_CALLS`` this runs the call on the device
in a fresh child with ``FLAGOS_LOG=dispatch`` and asserts the child's
``[flagos dispatch] <op> -> <backend>`` line names exactly the backend the active
conf routes the operator to.

Two skips, both data-driven and neither a platform name:

    * the conf does not route the operator to a FlagGems backend on this chip.
      Per-operator coverage is allowed to differ, so this is a skip carrying the
      conf's own route as the reason -- it is the statement that this chip does
      not claim the operator;
    * this wheel cannot run a FlagGems kernel at all (no kernel set in the build
      record, no device, or the FlagGems/Triton packages do not import).

An operator the conf *does* route to FlagGems and that then dispatches elsewhere
or raises is a failure: that is a claimed route the runtime does not take, which
is the one thing this suite refuses to accept.
"""

import pytest

from flaggems_caps import (
    FLAGGEMS_ROUTING_CALLS,
    accel_label,
    conf_route,
    flaggems_runtime_unavailable,
    route_reason,
    routes_to_flaggems,
)
from flaggems_probe import probe_routes

# The child runs every call in one interpreter, so the report is computed once
# and shared by the parametrized cases. A case that skips on its route never
# reaches this, which is what keeps a chip with no FlagGems routes from paying
# for an import of torch.
_PROBE: dict[str, dict] | None = None


def _probe() -> dict[str, dict]:
    global _PROBE
    if _PROBE is None:
        _PROBE = probe_routes()
    return _PROBE


@pytest.mark.anyplatform
@pytest.mark.parametrize("op", sorted(FLAGGEMS_ROUTING_CALLS))
def test_operator_dispatches_where_the_conf_routes_it(op: str):
    accel = accel_label() or "this conf"
    if not routes_to_flaggems(op):
        pytest.skip(
            f"{accel} routes {op} to {conf_route(op) or 'unlisted'}, not to a "
            "FlagGems backend"
        )
    reason = flaggems_runtime_unavailable()
    if reason:
        pytest.skip(reason)

    backend, why = route_reason(op)
    assert why == "routed", f"{op} unexpectedly resolved to {why}"
    entry = _probe()[op]
    assert entry["error"] is None, f"{op}: {entry['error']}"
    assert entry["dispatched_backend"] == backend, (
        f"{op} dispatched to {entry['dispatched_backend']!r}; the {accel} conf "
        f"routes it to {backend!r}"
    )
