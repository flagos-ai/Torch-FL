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

"""CPU-only regression checks for MUSA's expanded integration manifest.

The runtime doubles below test the preflight's fail-closed checks, not hardware
execution. The actual FlagGems and compile suites still require the MUSA runner.
"""

import ast
import builtins
import importlib.metadata
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / ".github" / "configs"
PREFLIGHT = "Check MUSA FlagTree and FlagGems runtime"
HYBRID = "Run MUSA FlagGems hybrid tests"
COMPILE = "Run torch.compile tests with FlagTree"


def _manifest(platform="musa"):
    return yaml.safe_load((CONFIGS / f"{platform}.yml").read_text())


def _groups(platform="musa"):
    return {group["name"]: group for group in _manifest(platform)["integration_tests"]}


def _preflight_source():
    command = _groups()[PREFLIGHT]["command"]
    assert command.startswith("python - <<'PY'\n")
    assert command.endswith("\nPY\n")
    return command.split("\n", 1)[1].rsplit("\nPY", 1)[0]


def test_compile_suite_matches_cuda_without_deselects():
    command = _groups()[COMPILE]["command"]
    tokens = shlex.split(command)
    assert tokens == shlex.split(_groups("cuda")[COMPILE]["command"])
    assert tokens[:4] == [
        "FLAGOS_USE_FLAGTREE=1",
        "python",
        "-m",
        "pytest",
    ]
    assert "tests/integration/test_compile.py" in tokens
    assert not any(
        option in tokens[4:] for option in ("--deselect", "--ignore", "-k", "-m")
    )


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


def test_runtime_preflight_precedes_the_new_suites_and_rng_stays_last():
    names = [group["name"] for group in _manifest()["integration_tests"]]
    assert len(names) == len(set(names))
    assert names.index("Check isolated MUSA environment") < names.index(PREFLIGHT)
    assert names.index(PREFLIGHT) < names.index(HYBRID) < names.index(COMPILE)
    assert names[-1] == "Run unified RNG tests"


@pytest.mark.parametrize("name", [PREFLIGHT, HYBRID, COMPILE])
def test_added_commands_have_no_hashes_or_failure_suppression(name):
    command = _groups()[name]["command"]
    assert "#" not in command
    assert "|| true" not in command
    assert "continue-on-error" not in _groups()[name]


def test_all_manifest_commands_are_valid_shell():
    for group in _manifest()["integration_tests"]:
        result = subprocess.run(
            ["bash", "-n"], input=group["command"], text=True, capture_output=True
        )
        assert result.returncode == 0, (group["name"], result.stderr)
    ast.parse(_preflight_source())


def test_preflight_imports_torch_fl_before_flaggems():
    source = _preflight_source()
    assert source.index("import torch_fl") < source.index("import flag_gems")


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
    exec(_preflight_source(), {})
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
        exec(_preflight_source(), {})


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
        exec(_preflight_source(), {})


@pytest.mark.parametrize("package", ["flag_gems", "triton"])
def test_missing_dependencies_fail_instead_of_skipping(runtime, monkeypatch, package):
    original_import = builtins.__import__

    def import_without_dependency(name, *args, **kwargs):
        if name.split(".", 1)[0] == package:
            raise ModuleNotFoundError(f"No module named {package!r}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_dependency)
    with pytest.raises(ModuleNotFoundError, match=package):
        exec(_preflight_source(), {})


def test_manifest_checks_run_in_the_cpu_setup_job():
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/integration-test-musa.yml").read_text()
    )
    steps = workflow["jobs"]["check-setup"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "tests/unit/test_musa_ci_manifest.py" in commands
    assert "PyYAML==6.0.1" in commands
