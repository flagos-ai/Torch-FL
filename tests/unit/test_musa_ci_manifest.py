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

"""CPU-only regression checks for MUSA's expanded integration coverage.

The runtime doubles below test the preflight's fail-closed checks, not hardware
execution. The actual FlagGems and compile suites still require the MUSA runner.

The groups these tests assert on live in the shared manifest rather than in
.github/configs/musa.yml: a platform config carries the data a step needs, and
.github/integration-manifest.yml carries the ordered step list every platform is
composed against. What is checked here is only what the MUSA entries add -- that
they are wired to the right files and ordered against the suites they gate --
while tests/unit/test_integration_manifest.py holds the manifest's own
invariants.
"""

import ast
import builtins
import importlib.metadata
import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from manifest_composer import composed_steps


REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = "Check MUSA FlagTree and FlagGems runtime"
HYBRID = "Run MUSA FlagGems hybrid tests"
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


def test_hybrid_suite_selects_the_musa_only_tests():
    tokens = shlex.split(_groups()[HYBRID]["command"])
    assert tokens[:4] == [
        "python",
        "-m",
        "pytest",
        "tests/integration/ops/test_musa_flaggems.py",
    ]
    assert tokens[4:6] == ["-m", "musa"]
    assert "--deselect" not in tokens


def test_runtime_preflight_precedes_the_suites_it_gates():
    # The preflight exists so a broken MThreads stack is reported once, as a
    # missing runtime, rather than as every hybrid case failing for its own
    # reason -- so its position relative to the suites it gates is the property
    # under test, not an incidental ordering.
    names = [step["name"] for step in _steps()]
    assert names.index(PREFLIGHT) < names.index(HYBRID)
    assert names.index("Check isolated MUSA environment") < names.index(PREFLIGHT)


@pytest.mark.parametrize("name", [PREFLIGHT, HYBRID])
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
    ast.parse(PREFLIGHT_TEST.read_text())


def test_musa_withholds_torch_compile_until_the_image_ships_the_vendor_stack():
    # The manifest excludes MUSA from the torch.compile group with a recorded
    # image-gap rationale: torch.compile needs the vendor flagtree (Triton with
    # a "mthreads" backend), and a stock PyPI triton cannot compile a MUSA
    # kernel at all. Adding the group to this image would record a pass that
    # measured nothing, so the exclusion is asserted rather than assumed.
    musa = {step["name"] for step in _steps("musa")}
    assert not any("torch.compile" in name for name in musa), sorted(musa)


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


def _hybrid_runtime_gate():
    """Extract only the dependency gate; importing the suite requires hardware."""
    path = REPO_ROOT / "tests/integration/ops/test_musa_flaggems.py"
    tree = ast.parse(path.read_text())
    gate = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_require_flaggems_mthreads"
    )
    module = ast.Module(body=[gate], type_ignores=[])
    return compile(module, str(path), "exec")


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


def test_hybrid_gate_does_not_register_flaggems_ops(runtime):
    _, modules = runtime
    modules["torch_fl"].flagos.device_count = lambda: 1

    def forbidden_registration(*args, **kwargs):
        raise AssertionError("the dependency gate must not replace torch_fl dispatch")

    modules["flag_gems"].enable = forbidden_registration
    namespace = {"torch_fl": modules["torch_fl"], "pytest": pytest}
    exec(_hybrid_runtime_gate(), namespace)
    namespace["_require_flaggems_mthreads"]()


@pytest.mark.parametrize("missing", ["device", "kernels", "mthreads", "flag_gems"])
def test_hybrid_gate_still_checks_runtime_requirements(runtime, monkeypatch, missing):
    state, modules = runtime
    modules["torch_fl"].flagos.device_count = lambda: 0 if missing == "device" else 1
    if missing == "kernels":
        state["kernels"] = ("vendor",)
    elif missing == "mthreads":
        modules["triton.backends"].backends = {}
    elif missing == "flag_gems":
        original_import = builtins.__import__

        def import_without_flaggems(name, *args, **kwargs):
            if name == "flag_gems":
                raise ModuleNotFoundError("No module named 'flag_gems'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", import_without_flaggems)

    namespace = {"torch_fl": modules["torch_fl"], "pytest": pytest}
    exec(_hybrid_runtime_gate(), namespace)
    with pytest.raises(pytest.skip.Exception):
        namespace["_require_flaggems_mthreads"]()


def test_preflight_accepts_the_musa_stack(runtime, capsys):
    _preflight_check()()
    assert "MUSA compile target: ('musa', 31, 32)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("active", False),
        ("backend", "cuda"),
        ("bound", False),
        ("target", None),
        ("target", ("cuda", 90, 32)),
        ("available", False),
        ("kernels", ("vendor",)),
        ("distribution_root", Path("/wrong/triton")),
    ],
)
def test_preflight_rejects_unusable_runtime(runtime, key, value):
    state, _ = runtime
    state[key] = value
    with pytest.raises(AssertionError):
        _preflight_check()()


@pytest.mark.parametrize("component", ["accelerator", "triton_backend", "gems_vendor"])
def test_preflight_rejects_the_wrong_vendor(runtime, component):
    _, modules = runtime
    if component == "accelerator":
        modules["torch_fl._build_config"].ACCELERATOR = "cuda"
    elif component == "triton_backend":
        modules["triton.backends"].backends = {"nvidia": object()}
    else:
        modules["flag_gems"].vendor_name = "nvidia"
    with pytest.raises(AssertionError):
        _preflight_check()()


@pytest.mark.parametrize("package", ["flag_gems", "triton"])
def test_missing_dependencies_fail_instead_of_skipping(runtime, monkeypatch, package):
    original_import = builtins.__import__

    def import_without_dependency(name, *args, **kwargs):
        if name.split(".", 1)[0] == package:
            raise ModuleNotFoundError(f"No module named {package!r}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_dependency)
    with pytest.raises(ModuleNotFoundError, match=package):
        _preflight_check()()


def test_preflight_requires_the_flagtree_distribution(runtime, monkeypatch):
    def missing_distribution(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing_distribution)
    with pytest.raises(importlib.metadata.PackageNotFoundError, match="flagtree"):
        _preflight_check()()


def test_checks_run_in_the_platform_agnostic_lane():
    # These checks need pytest, PyYAML and bash and nothing else, so they belong
    # on the CPU-only agnostic lane rather than a hardware runner. That lane
    # names its files explicitly, so this file and the env-script one have to be
    # in the list to run at all -- an unlisted file is silently never collected.
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/agnostic-checks.yml").read_text()
    )
    steps = workflow["jobs"]["agnostic-checks"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "tests/unit/test_musa_ci_manifest.py" in commands
    assert "tests/unit/test_musa_env_script.py" in commands
