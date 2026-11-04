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

"""A ``FLAGOS_OP_<op>=flaggems_python`` case skips exactly where no slot exists.

Issue #555: five ``@pytest.mark.flaggems_python`` dispatch cases asserted a
``-> flagos_python`` line for ops their platform's conf left unregistered, so on
a boxing-off wheel the assertion could never hold. Three were later guarded by
a conf-route check and one (``cat``) left the class when its conf route moved
to FlagGems. What was left -- ``sum.dim_IntList`` -- plus every future case of
the same shape is decided by ``override_is_logged`` in
``tests/integration/ops/backend_conf.py``, and that is a pure function over
files this test can feed: the conf, the generated claim list, and the build
record's kernel sets.

The one state that matters and cannot be faked is NVIDIA's: the conf routes
``sum.dim_IntList`` to ``none`` and the claim list does not name it, while MUSA's
conf routes it to FlagGems and ``register_flaggems_metax.inc`` names it. The
module is compiled from its source with a stub ``torch_fl`` and the claim-list
directory pointed at the checkout, because importing it would need the built
``torch_fl._C`` a unit host does not carry.
"""

import ast
import re
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tests" / "integration" / "ops" / "backend_conf.py"
CLAIM_DIR = ROOT / "csrc" / "aten" / "generated"


def _load_module():
    """``backend_conf``'s functions, compiled from its source without its imports.

    The module's two imports are why it cannot simply be imported here:
    ``import torch_fl`` claims the PrivateUse1 dispatch key and needs the built
    ``torch_fl._C``, which a host running the unit suite has no reason to carry,
    and it runs before the first case. Every function under test reaches its two
    collaborators (``torch_fl._env``, ``torch_fl._platform``,
    ``route_is_served``) through module globals, so a stub ``torch_fl`` standing
    in for the import is enough to drive them, and the cases install the real
    answers they want.
    """
    if _MODULE is not None:
        return _MODULE
    tree = ast.parse(MODULE_PATH.read_text())
    body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    module = types.ModuleType("backend_conf")
    module.__file__ = str(MODULE_PATH)
    module.torch_fl = types.SimpleNamespace(_env=None, _platform=None)
    module.re = re
    module.Path = Path
    module._claim_list_dir = lambda: CLAIM_DIR
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(MODULE_PATH), "exec"),
        module.__dict__,
    )
    globals()["_MODULE"] = module
    return module


_MODULE = None


@pytest.fixture()
def conf(monkeypatch):
    module = _load_module()

    def as_platform(name: str):
        platform = types.SimpleNamespace(build_accelerator=lambda: name)
        monkeypatch.setattr(module.torch_fl, "_platform", platform)

    def routing(op: str, value: str):
        monkeypatch.setattr(
            module, "_conf_route", lambda name: value if name == op else "none"
        )

    def kernels(*names: str):
        env = types.SimpleNamespace(build_kernels=lambda: frozenset(names))
        monkeypatch.setattr(module.torch_fl, "_env", env)

    def boxing_route_is_served(answer: bool):
        monkeypatch.setattr(module, "route_is_served", lambda op: answer)

    module.as_platform = as_platform
    module.routing = routing
    module.kernels = kernels
    module.boxing_route_is_served = boxing_route_is_served
    return module


def test_flag_override_is_unserved_when_the_conf_leaves_the_op_none(conf):
    """NVIDIA's ``sum.dim_IntList``: the assertion #555 measured as unsatisfiable."""
    conf.as_platform("nvidia")
    conf.routing("sum.dim_IntList", "none")
    conf.kernels("flaggems", "tileops")
    assert conf.override_is_logged("sum.dim_IntList", "flaggems_python") is False


def test_flag_override_is_logged_where_the_conf_and_the_claim_list_agree(conf):
    """MetaX's ``sum.dim_IntList``: ``flaggems_cpp`` and named in its list."""
    conf.as_platform("metax")
    conf.routing("sum.dim_IntList", "flaggems_cpp")
    conf.kernels("flaggems", "flaggems_cpp")
    assert conf.override_is_logged("sum.dim_IntList", "flaggems_python") is True


def test_a_flaggems_route_is_not_enough_without_a_claim(conf, monkeypatch):
    """The class the issue names: a conf row a rollback left behind.

    ``measured_flaggems_rollback`` routes an op ``none`` in the conf, and the
    claim list is generated from that same conf, so the two normally agree. A
    conf that says FlagGems while the list does not name the op is the shape
    every unguarded case had, and it must answer False rather than let the
    case fail on hardware.
    """
    conf.as_platform("nvidia")
    monkeypatch.setattr(conf, "_claim_list_dir", lambda: CLAIM_DIR)
    conf.routing("definitely_not_claimed", "flaggems")
    conf.kernels("flaggems")
    assert conf.override_is_logged("definitely_not_claimed", "flaggems_python") is False


def test_a_claimed_op_on_a_conf_that_routes_it_flaggems_is_logged(conf):
    conf.as_platform("hygon")
    conf.routing("sum.dim_IntList", "flaggems")
    conf.kernels("flaggems")
    assert conf.override_is_logged("sum.dim_IntList", "flaggems_python") is True


def test_no_claim_list_for_the_platform_falls_back_to_the_conf(conf, monkeypatch):
    """A vendor-named platform claims from its own list; the conf is the answer."""
    conf.as_platform("ascend")
    monkeypatch.setattr(conf, "_claim_list_dir", lambda: CLAIM_DIR)
    conf.routing("sum.dim_IntList", "flaggems")
    conf.kernels("flaggems")
    assert conf.override_is_logged("sum.dim_IntList", "flaggems_python") is True


def test_the_wheel_must_compile_flaggems_regardless_of_the_claim(conf):
    conf.as_platform("metax")
    conf.routing("sum.dim_IntList", "flaggems_cpp")
    conf.kernels("boxing")
    assert conf.override_is_logged("sum.dim_IntList", "flaggems_python") is False


def test_boxing_override_asks_the_route_not_the_claim_list(conf):
    conf.as_platform("nvidia")
    conf.routing("sum.dim_IntList", "none")
    conf.kernels("boxing")
    conf.boxing_route_is_served(False)
    assert conf.override_is_logged("sum.dim_IntList", "boxing") is False
