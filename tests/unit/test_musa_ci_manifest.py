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

"""CPU-only regression checks for MUSA's integration coverage.

The runtime doubles below test the preflight's fail-closed checks, not hardware
execution. The FlagGems and compile suites still require the MUSA runner.

The groups these tests assert on live in the shared manifest rather than in
.github/configs/musa.yml: a platform config carries the data a step needs, and
.github/integration-manifest.yml carries the ordered step list every platform is
composed against. What is checked here is only what the MUSA entries add -- that
they are wired to the right files and ordered against the suites they gate --
while tests/unit/test_integration_manifest.py holds the manifest's own
invariants.
"""

import builtins
import importlib.metadata
import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from manifest_composer import composed_steps


REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = "Check MUSA FlagTree and FlagGems runtime"
COMPILE = "Run torch.compile tests"
PREFLIGHT_TEST = REPO_ROOT / "tests/integration/test_musa_runtime.py"


@pytest.fixture(scope="module", autouse=True)
def register_runtime_marker(pytestconfig):
    # These CPU checks load the hardware test without integration/conftest.py.
    pytestconfig.addinivalue_line("markers", "musa: requires MUSA hardware")


def _steps(platform="musa"):
    return composed_steps(platform)


def _groups(platform="musa"):
    return {step["name"]: step for step in _steps(platform)}


def _preflight_check():
    return runpy.run_path(str(PREFLIGHT_TEST))[
        "test_musa_flagtree_and_flaggems_runtime"
    ]


def test_preflight_runs_the_required_runtime_test():
    assert shlex.split(_groups()[PREFLIGHT]["command"]) == [
        "python",
        "-m",
        "pytest",
        "tests/integration/test_musa_runtime.py",
        "-v",
        "-s",
        "--tb=short",
    ]


def test_runtime_preflight_precedes_the_suites_it_gates():
    # The preflight exists so a broken MThreads stack is reported once, as a
    # missing runtime, rather than as every FlagGems case failing for its own
    # reason -- so its position relative to the suites it gates is the property
    # under test, not an incidental ordering.
    names = [step["name"] for step in _steps()]
    flaggems = names.index("FlagGems native contract (hardware-agnostic)")
    assert names.index(PREFLIGHT) < flaggems
    assert names.index("Check isolated MUSA environment") < names.index(PREFLIGHT)


@pytest.mark.parametrize("name", [PREFLIGHT])
def test_added_commands_have_no_hashes_or_failure_suppression(name):
    command = _groups()[name]["command"]
    assert "#" not in command
    assert "|| true" not in command
    assert "continue-on-error" not in _groups()[name]


def test_all_manifest_commands_are_valid_shell():
    for group in _steps():
        result = subprocess.run(
            ["bash", "-n"], input=group["command"], text=True, capture_output=True
        )
        assert result.returncode == 0, (group["name"], result.stderr)


def test_musa_withholds_torch_compile_until_the_image_ships_the_vendor_stack():
    # The manifest excludes MUSA from the torch.compile group with a recorded
    # image-gap rationale: torch.compile needs the vendor flagtree (Triton with
    # a "mthreads" backend), and a stock PyPI triton cannot compile a MUSA
    # kernel at all. Adding the group to this image would record a pass that
    # measured nothing, so the exclusion is asserted rather than assumed.
    musa = {step["name"] for step in _steps("musa")}
    assert not any("torch.compile" in name for name in musa), sorted(musa)


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    """Install import doubles without loading torch or a vendor runtime."""
    state = {
        "active": True,
        "backend": "mthreads",
        "bound": True,
        "target": ("musa", 31, 32),
        "available": True,
        "kernels": ("vendor", "flaggems"),
        "distribution_root": tmp_path / "triton",
    }
    modules = {
        name: ModuleType(name)
        for name in (
            "torch_fl",
            "torch_fl._env",
            "torch_fl._build_config",
            "torch_fl.compile",
            "torch_fl.compile.flagtree_shim",
            "flag_gems",
            "triton",
            "triton.backends",
        )
    }
    torch_fl = modules["torch_fl"]
    torch_fl._env = modules["torch_fl._env"]
    torch_fl._env.build_kernels = lambda: state["kernels"]
    torch_fl.flagos = SimpleNamespace(is_available=lambda: state["available"])
    modules["torch_fl._build_config"].ACCELERATOR = "musa"
    shim = modules["torch_fl.compile.flagtree_shim"]
    shim.is_flagtree_active = lambda: state["active"]
    shim.flagtree_backend = lambda: state["backend"]
    shim.bind_flagtree_musa_driver = lambda: state["bound"]
    shim.flagtree_musa_driver_target = lambda: state["target"]
    modules["flag_gems"].vendor_name = "mthreads"
    modules["flag_gems"].__version__ = "test"
    modules["triton"].__file__ = str(tmp_path / "triton" / "__init__.py")
    modules["triton"].__version__ = "test"
    modules["triton"].backends = modules["triton.backends"]
    modules["triton.backends"].backends = {"mthreads": object()}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def distribution(name):
        assert name == "flagtree"
        return SimpleNamespace(
            version="test",
            locate_file=lambda path: state["distribution_root"],
        )

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    return state, modules


def test_preflight_imports_torch_fl_before_flaggems(runtime, monkeypatch):
    imports = []
    original_import = builtins.__import__

    def track_import(name, *args, **kwargs):
        if name in ("torch_fl", "flag_gems"):
            imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", track_import)
    _preflight_check()()
    assert imports.index("torch_fl") < imports.index("flag_gems")
