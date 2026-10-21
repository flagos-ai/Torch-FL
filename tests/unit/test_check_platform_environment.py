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

"""Check the checkpoint mount assertion used by the model platform jobs."""

import importlib
import json
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest
from manifest_composer import environment_check


@pytest.fixture
def isolated_triton(monkeypatch, tmp_path):
    """Let a test define the whole Triton tree it imports."""
    environment_check._TLE_IMPORT_FAILURES.clear()
    for name in tuple(sys.modules):
        if name == "triton" or name.startswith("triton."):
            parent, _, child = name.rpartition(".")
            if parent and sys.modules.get(name) is not None:
                monkeypatch.delattr(
                    sys.modules[name],
                    child,
                    raising=False,
                )
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "flag_gems", raising=False)
    return monkeypatch


def test_model_mount_requires_a_transformers_configuration(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))

    environment_check.assert_model_dir()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (None, "has no Transformers config.json"),
        ("{truncated", "cannot be read as JSON"),
        ("[]", "must be a JSON object"),
        ({}, "has no non-empty model_type"),
        ({"model_type": ""}, "has no non-empty model_type"),
        ({"model_type": False}, "has no non-empty model_type"),
    ],
)
def test_model_mount_rejects_an_unloadable_checkpoint(
    monkeypatch, tmp_path, config, message
):
    path = tmp_path / "config.json"
    if isinstance(config, str):
        path.write_text(config)
    elif config is not None:
        path.write_text(json.dumps(config))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))

    with pytest.raises(AssertionError, match=re.escape(message)):
        environment_check.assert_model_dir()


def test_model_mount_requires_an_existing_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with pytest.raises(AssertionError, match="MODEL_PATH is empty"):
        environment_check.assert_model_dir()
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing"))
    with pytest.raises(AssertionError, match="is not a directory"):
        environment_check.assert_model_dir()


def _runtime(monkeypatch, kernels, *, triton=True, flag_gems=True):
    modules = {
        name: ModuleType(name) for name in ("torch", "torch_fl", "triton", "flag_gems")
    }
    modules["torch"].__version__ = "2.10.0+cpu"
    modules["torch_fl"].__file__ = "/tmp/torch_fl/__init__.py"
    modules["torch_fl"]._env = SimpleNamespace(build_kernels=lambda: kernels)
    modules["triton"].backends = SimpleNamespace(backends={"backend": object()})
    modules["flag_gems"].__file__ = "/tmp/flag_gems/__init__.py"
    for name, module in modules.items():
        absent = (name == "triton" and not triton) or (
            name == "flag_gems" and not flag_gems
        )
        monkeypatch.setitem(sys.modules, name, None if absent else module)
    return modules


def test_environment_runtime_imports_follow_the_wheel_record(monkeypatch):
    modules = _runtime(monkeypatch, ("flaggems",))
    dependencies = environment_check.import_dependencies(
        {"require_build_kernels": ["flaggems"]}
    )

    assert dependencies.triton is modules["triton"]
    assert dependencies.flag_gems is modules["flag_gems"]
    assert environment_check.assert_build_kernels(
        {"require_build_kernels": ["flaggems"]},
        modules["torch_fl"],
        dependencies,
    ) == ("flaggems",)


@pytest.mark.parametrize("runtime", ["triton", "flag_gems"])
def test_environment_runtime_import_failure_names_the_missing_package(
    monkeypatch, runtime
):
    _runtime(
        monkeypatch,
        ("flaggems",),
        triton=runtime != "triton",
        flag_gems=runtime != "flag_gems",
    )
    with pytest.raises(ModuleNotFoundError):
        environment_check.import_dependencies({"require_build_kernels": ["flaggems"]})


def test_environment_rejects_a_native_only_wheel_when_flaggems_is_required(
    monkeypatch,
):
    modules = _runtime(monkeypatch, ("vendor",))
    with pytest.raises(AssertionError, match="expected the 'flaggems' kernel set"):
        environment_check.assert_build_kernels(
            {"require_build_kernels": ["flaggems"]},
            modules["torch_fl"],
            SimpleNamespace(triton=modules["triton"], flag_gems=modules["flag_gems"]),
        )


def test_environment_runtime_import_is_not_needed_for_a_native_only_wheel(monkeypatch):
    modules = _runtime(monkeypatch, ("vendor",), triton=False, flag_gems=False)
    dependencies = environment_check.import_dependencies({})

    assert dependencies.triton is None
    assert dependencies.flag_gems is None
    assert environment_check.assert_build_kernels(
        {"forbid_build_kernels": ["flaggems"]}, modules["torch_fl"], dependencies
    ) == ("vendor",)


def test_environment_requires_the_tle_submodule_flag_gems_imports(
    isolated_triton, tmp_path
):
    """A parent namespace without ``tle`` is not proof of a usable package."""
    experimental = tmp_path / "triton" / "experimental"
    experimental.mkdir(parents=True)
    (tmp_path / "triton" / "__init__.py").write_text("")
    (experimental / "__init__.py").write_text("")
    isolated_triton.syspath_prepend(str(tmp_path))

    dependencies = SimpleNamespace(flag_gems=object())
    with pytest.raises(ModuleNotFoundError, match=r"triton\.experimental\.tle"):
        environment_check.assert_flaggems_tle("triton.experimental.tle", dependencies)

    (experimental / "tle.py").write_text("")
    importlib.invalidate_caches()
    environment_check._TLE_IMPORT_FAILURES.clear()
    environment_check.assert_flaggems_tle("triton.experimental.tle", dependencies)


def test_environment_does_not_accept_a_partial_tle_import(isolated_triton, tmp_path):
    """A cached parent package cannot hide an incomplete TLE import."""
    experimental = tmp_path / "triton" / "experimental"
    tle = experimental / "tle"
    tle.mkdir(parents=True)
    (tmp_path / "triton" / "__init__.py").write_text("")
    (experimental / "__init__.py").write_text("")
    (tle / "__init__.py").write_text("import triton.experimental.tle.hidden\n")
    isolated_triton.syspath_prepend(str(tmp_path))
    dependencies = SimpleNamespace(flag_gems=object())

    with pytest.raises(ModuleNotFoundError, match="hidden"):
        environment_check.assert_flaggems_tle("triton.experimental.tle", dependencies)

    hidden = tle / "hidden"
    hidden.mkdir()
    (hidden / "__init__.py").write_text("VALUE = 7\n")
    importlib.invalidate_caches()
    # Recording that the first import failed prevents a second attempt from
    # hiding the partial package behind a cached parent module.
    with pytest.raises(SystemExit, match="fresh Python process"):
        environment_check.assert_flaggems_tle("triton.experimental.tle", dependencies)


def test_environment_registers_the_ascend_shmem_placeholder_for_tle(
    isolated_triton, tmp_path
):
    """FlagGems imports TLE without loading torch_fl first."""
    experimental = tmp_path / "triton" / "experimental"
    tle = experimental / "tle"
    tle.mkdir(parents=True)
    (tmp_path / "triton" / "__init__.py").write_text("")
    (experimental / "__init__.py").write_text("")
    (tle / "__init__.py").write_text("import shmem\n")
    isolated_triton.syspath_prepend(str(tmp_path))
    isolated_triton.delitem(sys.modules, "shmem", raising=False)

    environment_check.assert_flaggems_tle(
        "triton.experimental.tle", SimpleNamespace(flag_gems=object())
    )

    assert "shmem" in sys.modules


def test_environment_rejects_a_missing_dsa_module_imported_by_tle(
    isolated_triton, tmp_path
):
    """A TLE release whose package imports DSA must carry that module."""
    triton = tmp_path / "triton"
    experimental = triton / "experimental"
    tle = experimental / "tle"
    language = tle / "language"
    language.mkdir(parents=True)
    (triton / "__init__.py").write_text("")
    (experimental / "__init__.py").write_text("")
    (tle / "__init__.py").write_text("from triton.experimental.tle import language\n")
    (language / "__init__.py").write_text(
        "from triton.experimental.tle.language import dsa\n"
    )
    isolated_triton.syspath_prepend(str(tmp_path))
    with pytest.raises(ImportError, match="dsa"):
        environment_check.assert_flaggems_tle(
            "triton.experimental.tle", SimpleNamespace(flag_gems=object())
        )


def test_environment_loads_a_tle_package_without_initializing_flag_gems(
    isolated_triton, tmp_path
):
    """FlagGems may need a TLE module without importing the device package."""
    experimental = tmp_path / "triton" / "experimental"
    tle = experimental / "tle"
    language = tle / "language"
    language.mkdir(parents=True)
    (tmp_path / "triton" / "__init__.py").write_text("")
    (experimental / "__init__.py").write_text("")
    (tle / "__init__.py").write_text("")
    (language / "__init__.py").write_text("")
    isolated_triton.syspath_prepend(str(tmp_path))
    package = tmp_path / "installed" / "flag_gems"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('FlagGems device detection ran too early')\n"
    )
    environment_check.assert_flaggems_tle(
        "triton.experimental.tle.language",
        SimpleNamespace(
            flag_gems=SimpleNamespace(__file__=str(package / "__init__.py"))
        ),
    )


@pytest.mark.parametrize("value", ["", [""], [None], [1]])
def test_environment_rejects_malformed_tle_module_names(value):
    dependencies = SimpleNamespace(flag_gems=object())
    with pytest.raises(SystemExit, match="require_flaggems_tle_modules"):
        names = value if isinstance(value, list) else [value]
        for name in names:
            environment_check.assert_flaggems_tle(name, dependencies)
