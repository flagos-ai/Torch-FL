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

"""The autoload contract that can be checked without a built accelerator wheel."""

import ast
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
INIT = ROOT / "torch_fl" / "__init__.py"


def _function(name):
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(INIT), "exec"), namespace
    )
    return namespace[name]


def test_installed_entry_point_targets_the_device_bootstrap():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["entry-points"]["torch.backends"] == {
        "torch_fl": "torch_fl._autoload:init"
    }


def test_vendor_core_and_cuda_assets_are_selected_before_torch_import():
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    phases = [
        node.value.func.id
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    ]
    assert phases.index("_phase_preload") < phases.index("_phase_claim")
    preload = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_phase_preload"
    )
    calls = [
        node.value.func.id
        for node in preload.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    ]
    assert calls.index("_relink_vendor_libtorch") < calls.index("_preload_cuda_assets")


@pytest.mark.parametrize("current", ["privateuseone", "flagos"])
def test_unclaimed_or_our_privateuse1_name_is_accepted(current):
    check = _function("_check_privateuse1_unclaimed")
    check.__globals__["torch"] = SimpleNamespace(
        _C=SimpleNamespace(_get_privateuse1_backend_name=lambda: current)
    )
    check()


@pytest.mark.parametrize("current", ["musa", "other_backend"])
def test_foreign_privateuse1_claim_fails_before_native_extension(current):
    check = _function("_check_privateuse1_unclaimed")
    check.__globals__["torch"] = SimpleNamespace(
        _C=SimpleNamespace(_get_privateuse1_backend_name=lambda: current)
    )
    with pytest.raises(RuntimeError) as error:
        check()
    message = str(error.value)
    assert f"already claimed by the '{current}' backend" in message
    assert "import torch_fl first" in message
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD=0" in message


@pytest.mark.parametrize("accelerator,expected", [("musa", ["0"]), ("cuda", [])])
def test_only_musa_disables_vendor_entry_points_before_torch_import(
    accelerator, expected
):
    disable = _function("_disable_vendor_backend_autoload")
    values = []
    disable.__globals__.update(
        _build_accelerator=lambda: accelerator,
        _env=SimpleNamespace(set_foreign=lambda key, value: values.append(value)),
    )
    disable()
    assert values == expected
