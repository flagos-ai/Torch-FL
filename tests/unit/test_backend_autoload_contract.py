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
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import setuptools

ROOT = Path(__file__).resolve().parents[2]
INIT = ROOT / "torch_fl" / "__init__.py"


@pytest.fixture
def load_setup(monkeypatch):
    monkeypatch.delenv("FLAGOS_ACCELERATOR", raising=False)

    def load(accelerator):
        monkeypatch.setenv("FLAGOS_ACCELERATOR", accelerator)
        kwargs = {}
        monkeypatch.setattr(setuptools, "setup", lambda **values: kwargs.update(values))
        namespace = runpy.run_path(str(ROOT / "setup.py"))
        monkeypatch.setitem(sys.modules, "setuptools", setuptools)
        return namespace, kwargs

    return load


def _pyproject_entry_points(group: str) -> dict:
    """One ``[project.entry-points."<group>"]`` table of pyproject.toml.

    ``tomllib`` is Python 3.11+, and the accelerator CI interpreters are 3.10:
    DCU and GCU build against the image's cp310 Python, and the DCU unit group
    is the one job that collects this whole directory. Importing it there was a
    collection error, which failed the file and with it the per-file unit group.
    ``tomli`` would be the other fix, but nothing in tests/unit has a parser
    dependency today and this file needs one table of one file, so the table is
    scanned directly instead.
    """
    lines = (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    header = f'[project.entry-points."{group}"]'
    if header not in lines:
        raise AssertionError(f"pyproject.toml has no {header} table")
    entries = {}
    for line in lines[lines.index(header) + 1 :]:
        stripped = line.strip()
        if stripped.startswith("["):  # the next table ends this one
            break
        if not stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition("=")
        entries[key.strip()] = value.strip().strip('"')
    return entries


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


def test_entry_points_are_resolved_by_the_build():
    """The hook is declared in pyproject.toml, not passed from setup().

    setuptools ignores a table that exists only in setup() as soon as [project]
    is present (`_MissingDynamic`), so a setup()-side table reaches no wheel.
    """
    assert _pyproject_entry_points("torch.backends") == {
        "torch_fl": "torch_fl._autoload:init"
    }


def test_entry_points_stay_out_of_the_dynamic_list():
    """A table that is declared dynamic has nothing to be filled from here."""
    lines = (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    dynamic = next(line for line in lines if line.startswith("dynamic ="))
    assert "entry-points" not in dynamic


@pytest.mark.parametrize(
    "accelerator", ["nvidia", "hygon", "mthreads", "iluvatar", "apple"]
)
def test_every_build_declares_only_what_a_wheel_can_carry(load_setup, accelerator):
    """No `scripts=` and no setup()-side table: both are dropped by setuptools.

    The `torch-fl-<platform>` preflight that `scripts=` used to install is a
    console script, which needs a static name in pyproject.toml; it is not
    declared, so a wheel installs the tool from `scripts/tools/` instead.
    """
    _, kwargs = load_setup(accelerator)
    assert "scripts" not in kwargs
    assert "entry_points" not in kwargs


def test_the_packaged_hook_resolves_to_the_autoload_module():
    """The label the wheel carries names a function that exists."""
    hook = _pyproject_entry_points("torch.backends")["torch_fl"]
    module, _, attribute = hook.partition(":")
    loaded = runpy.run_path(str(ROOT.joinpath(*module.split(".")).with_suffix(".py")))
    assert callable(loaded[attribute])


def test_mps_editable_install_rejects_another_accelerators_egg_info(
    load_setup, monkeypatch, tmp_path
):
    """A stale vendor entry point must not leak into the MPS installation."""
    (tmp_path / "torch_fl.egg-info").mkdir()
    namespace, kwargs = load_setup("apple")
    wheel = SimpleNamespace(
        distribution=SimpleNamespace(get_name=lambda: "torch_fl"),
        _find_egg_info_dir=lambda: "vendor-entry-points",
    )
    command_globals = kwargs["cmdclass"]["editable_wheel"].run.__globals__
    monkeypatch.setitem(command_globals, "BASE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="torch.backends entry point"):
        kwargs["cmdclass"]["editable_wheel"]._find_egg_info_dir(wheel)


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


@pytest.mark.parametrize("current", ["mthreads", "other_backend"])
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


@pytest.mark.parametrize("accelerator,expected", [("mthreads", ["0"]), ("nvidia", [])])
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
