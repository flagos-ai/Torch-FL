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

"""``torch.flagos._utils._get_device_index``, the hook torch.load reads.

``torch.serialization._validate_device`` resolves a ``map_location`` by asking
the registered device module for ``_utils._get_device_index``; its own docstring
names that attribute as the contract a privateuse1 backend must implement. With
no ``_utils`` it falls to an ``else`` branch that calls ``torch.device(location)``
directly, so ``map_location="cuda:3"`` resolves to a **cuda** device rather than
to flagos -- on a CPU-only torch not a device at all, and on a boxing build the
wrong one. Most of the ecosystem checkpoints with ``map_location="cuda"``, so
those spellings are the ones that matter (issue #531).

The checks drive the real ``_validate_device`` against a stand-in device module,
because that is the caller whose behaviour is the point; the function under test
is pulled out of ``torch_fl/flagos/__init__.py`` by AST extraction rather than by
importing torch_fl, which needs the native ``_C`` and claims the PrivateUse1
dispatch key. Same approach as tests/unit/test_kunlunxin_triton_stream.py.

Run: pytest tests/unit/test_flagos_utils_device_index.py -v
"""

import ast
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SOURCE = Path(__file__).resolve().parents[2] / "torch_fl" / "flagos" / "__init__.py"
_UNDER_TEST = ("_device_index", "_utils_get_device_index")

_DEVICE_COUNT = 8
_CURRENT_DEVICE = 0


def _load_functions():
    """Compile just the functions under test into a throwaway module."""
    tree = ast.parse(_SOURCE.read_text())
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _UNDER_TEST
    ]
    assert {node.name for node in wanted} == set(_UNDER_TEST), (
        f"{_SOURCE} no longer defines {_UNDER_TEST}"
    )
    module = types.ModuleType("flagos_utils_under_test")
    module.torch = torch
    # `_device_index` answers "no index given" with the current device.
    module.current_device = lambda: _CURRENT_DEVICE
    exec(
        compile(ast.Module(body=wanted, type_ignores=[]), str(_SOURCE), "exec"),
        module.__dict__,
    )
    return module


@pytest.fixture()
def get_device_index():
    return _load_functions()._utils_get_device_index


@pytest.fixture()
def registered(monkeypatch, get_device_index):
    """A privateuse1 module named flagos, carrying the `_utils` shim.

    The rename and the module registration are process-global in torch and have
    no public teardown, so this asserts the backend is already flagos (it is,
    once anything has named it) or renames it once. The device module itself is
    swapped through torch's own attribute, which monkeypatch can restore.
    """
    if torch._C._get_privateuse1_backend_name() != "flagos":
        torch.utils.rename_privateuse1_backend("flagos")

    module = types.ModuleType("flagos")
    module.is_available = lambda: True
    module.device_count = lambda: _DEVICE_COUNT
    utils = types.ModuleType("flagos._utils")
    utils._get_device_index = get_device_index
    module._utils = utils
    monkeypatch.setattr(torch, "flagos", module, raising=False)
    return module


@pytest.mark.parametrize(
    "location, expected",
    [
        # The spellings the issue is about: a CUDA-named location has to land on
        # flagos, keeping its index.
        ("cuda", "flagos:0"),
        ("cuda:0", "flagos:0"),
        ("cuda:3", "flagos:3"),
        ("cuda:7", "flagos:7"),
        # And the native spellings keep working.
        ("flagos", "flagos:0"),
        ("flagos:0", "flagos:0"),
        ("flagos:5", "flagos:5"),
    ],
)
def test_map_location_resolves_to_flagos(registered, location, expected):
    """Every spelling resolves to a flagos device with the index preserved."""
    import torch.serialization as serialization

    device = serialization._validate_device(location, "flagos")
    assert str(device) == expected


def test_a_torch_device_object_is_accepted(registered):
    """`map_location=torch.device("cuda", 2)` is the form frameworks pass."""
    import torch.serialization as serialization

    device = serialization._validate_device(torch.device("cuda", 2), "flagos")
    assert str(device) == "flagos:2"


def test_an_index_past_the_device_count_is_still_refused(registered):
    """The shim resolves the index; it does not widen what torch accepts."""
    import torch.serialization as serialization

    with pytest.raises(RuntimeError, match="device_count"):
        serialization._validate_device(f"cuda:{_DEVICE_COUNT + 1}", "flagos")


def test_without_the_shim_a_cuda_location_stays_cuda(monkeypatch, registered):
    """The defect this fixes, pinned so the shim cannot be dropped silently.

    `_validate_device`'s `else` branch keeps the location's own device type, so
    the storage is deserialized onto cuda rather than onto flagos.
    """
    import torch.serialization as serialization

    monkeypatch.delattr(torch.flagos, "_utils")
    device = serialization._validate_device("cuda:3", "flagos")
    assert device.type == "cuda", (
        "without _utils torch resolves the location itself; if this now answers "
        "flagos, torch changed and the shim's rationale needs rechecking"
    )


class TestTheCudaSignature:
    """The two flags are `torch.cuda._utils._get_device_index`'s, not ignored."""

    def test_an_int_passes_through(self, get_device_index):
        assert get_device_index(3) == 3

    def test_optional_none_means_the_current_device(self, get_device_index):
        assert get_device_index(None, optional=True) == _CURRENT_DEVICE

    def test_a_non_optional_none_is_an_error(self, get_device_index):
        with pytest.raises(ValueError, match="optional"):
            get_device_index(None)

    @pytest.mark.parametrize("cpu", ["cpu", torch.device("cpu")])
    def test_allow_cpu_answers_minus_one(self, get_device_index, cpu):
        assert get_device_index(cpu, allow_cpu=True) == -1

    def test_a_device_index_is_unaffected_by_allow_cpu(self, get_device_index):
        assert get_device_index("cuda:4", allow_cpu=True) == 4
