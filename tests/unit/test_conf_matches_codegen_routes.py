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

"""The CUDA conf agrees with the route sets the generator writes it from.

`scripts/codegen/codegen_ops.py` decides each op's backend from three sets --
`cuda_route_exceptions`, `measured_flaggems_rollback` and
`metadata_route_overrides` -- and `torch_fl/configs/backends_cuda.conf` is its
output. The two can only be checked against each other by re-running the
generator, and that is not always possible: the generator discovers the FlagGems
cohort from whatever `flag_gems` is installed, so on a box whose FlagGems is
newer than the one the committed conf was generated against, a full run rewrites
hundreds of unrelated routes and buries the intended change. (Measured on the
A100 development host: the local FlagGems turns 186 more ops into FlagGems
routes, plus every route in the DCU and MetaX confs.)

So this pins the part that is deterministic regardless of which FlagGems is
installed: for every op the sets name, the conf must route it to `cuda`. The
sets are read with `ast`, not imported -- `codegen_ops.py` imports torchgen at
module scope and this file is a fast unit test.

Run: pytest tests/unit/test_conf_matches_codegen_routes.py
"""

import ast
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_CODEGEN = _REPO / "scripts/codegen/codegen_ops.py"
_CONF = _REPO / "torch_fl/configs/backends_cuda.conf"

_ROUTE_SETS = (
    "cuda_route_exceptions",
    "measured_flaggems_rollback",
    "metadata_route_overrides",
)


def _route_set_names():
    """{set name: set of op names} for the three sets, read without importing."""
    tree = ast.parse(_CODEGEN.read_text())
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not targets or targets[0] not in _ROUTE_SETS:
            continue
        assert isinstance(node.value, ast.Set), f"{targets[0]} is not a set literal"
        names = set()
        for element in node.value.elts:
            assert isinstance(element, ast.Constant), (
                f"{targets[0]} has a non-literal entry"
            )
            names.add(element.value)
        found[targets[0]] = names
    return found


def _conf_routes():
    routes = {}
    for raw in _CONF.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        op, backend = (part.strip() for part in line.split("=", 1))
        routes[op] = backend
    return routes


@pytest.fixture(scope="module")
def codegen_sets():
    sets = _route_set_names()
    missing = [name for name in _ROUTE_SETS if name not in sets]
    assert not missing, f"codegen_ops.py no longer defines {missing}"
    return sets


def test_every_exception_set_member_is_routed_to_cuda(codegen_sets):
    """The whole point of the sets is that the conf sends those ops to boxing.

    An entry that is absent from the conf, or present with any other backend,
    means the conf and the generator disagree -- and the next `FLAGOS_CODEGEN_ALL=1`
    run would silently change routing.
    """
    routes = _conf_routes()
    for name, ops in codegen_sets.items():
        for op in sorted(ops):
            assert op in routes, f"{op} ({name}) is not listed in {_CONF.name}"
            assert routes[op] == "cuda", (
                f"{op} ({name}) is routed to {routes[op]!r} in {_CONF.name}, "
                f"but {name} forces it to cuda"
            )


def test_the_metadata_override_set_holds_only_overloads_that_exist(codegen_sets):
    """Guards against a typo turning an override into a no-op.

    `metadata_route_overrides` moves ops off the FlagGems route for cost. An op
    name that is not in the conf at all would be silently inert: the generator
    only consults the set for ops it is about to write.
    """
    routes = _conf_routes()
    unknown = sorted(
        op for op in codegen_sets["metadata_route_overrides"] if op not in routes
    )
    assert not unknown, (
        f"metadata_route_overrides names ops absent from the conf: {unknown}"
    )
