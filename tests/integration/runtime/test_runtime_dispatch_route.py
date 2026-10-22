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

"""The routing half of the contract: each op dispatches where the conf says.

For every op in ``RUNTIME_OPS`` this runs the op on the device in a fresh child
with ``FLAGOS_LOG=dispatch``, then asserts two things at once:

    * the child's ``[flagos dispatch] <op> -> <backend>`` line names exactly the
      backend this platform's conf routes the op to (``runtime_caps.route_reason``);
    * the device numerics match a CPU reference (``cpu_reference_ok``).

The only skip is build-driven: an op routed to a FlagGems backend or to the CUDA
dispatch key is marked with the kernel set that route needs, so the package
conftest can skip it on a wheel whose build record lacks that set -- the one
build-driven skip allowed. The CUDA key is on that list because on this project's
CPU-torch wheels it is served by the generated CUDA boxing kernels, not by torch
itself: on the default FlagGems-primary wheel (boxing off) a ``= cuda`` route has
no kernel behind it and the op reaches cpu_fallback, which emits no dispatch line.
An op the conf routes to ``none`` is left to cpu_fallback, which writes no
dispatch. The case skips as an unimplemented kernel, while the suite's summary
records it in ``ops_none`` so the coverage gap stays visible. The conf must
still list the op: an absent entry is a generation gap and fails. No platform
name gates anything here.
"""

import pytest

from runtime_caps import RUNTIME_OPS, effective_route
from runtime_probe import probe_ops


def _parametrized_ops() -> list:
    """One parametrized case per op, tagged with its route and kernel need."""
    params = []
    for op in RUNTIME_OPS:
        marks = [pytest.mark.backend_route(op=op)]
        try:
            backend, _ = effective_route(op)
        except Exception:  # noqa: BLE001 -- a missing conf entry is not marked
            backend = None
        if backend == "flagos":
            marks.append(pytest.mark.flaggems_cpp)
        elif backend == "flagos_python":
            marks.append(pytest.mark.flaggems_python)
        elif backend == "cuda":
            # The CUDA dispatch key is served by the generated boxing kernels on
            # every platform whose torch is CPU-only; see the module docstring.
            marks.append(pytest.mark.boxing)
        params.append(pytest.param(op, marks=marks))
    return params


@pytest.mark.anyplatform
@pytest.mark.parametrize("op", _parametrized_ops())
def test_dispatch_matches_conf_route(op: str):
    backend, reason = effective_route(op)
    if reason == "unimplemented":
        pytest.skip(f"{op} has no accelerator kernel for this platform")
    assert backend is not None
    assert reason == "routed", f"{op} unexpectedly resolved to {reason}"
    report = probe_ops([op])
    assert len(report) == 1
    entry = report[0]
    assert entry["error"] is None, entry["error"]
    assert entry["dispatched_backend"] == backend, (
        f"{op} dispatched to {entry['dispatched_backend']!r}, conf says {backend!r}"
    )
    assert entry["cpu_reference_ok"] is True, (
        f"{op} device result does not match the CPU reference"
    )
