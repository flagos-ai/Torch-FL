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
import io
import json
import re
import struct
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


def test_model_cli_checks_its_explicit_path(monkeypatch, tmp_path, capsys):
    """The model group states its path instead of depending on shell state."""
    model = tmp_path / "Qwen3-0.6B"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    monkeypatch.delenv("MODEL_PATH", raising=False)

    assert environment_check.main(["check-model", "--path", str(model)]) == 0
    assert "MODEL_PATH is empty" not in capsys.readouterr().out
    assert "MODEL_PATH" not in importlib.import_module("os").environ

    with pytest.raises(SystemExit, match="requires --path"):
        environment_check.main(["check-model"])


def test_model_cli_rejects_an_empty_checkpoint(tmp_path):
    with pytest.raises(AssertionError, match="has no Transformers config.json"):
        environment_check.main(["check-model", "--path", str(tmp_path)])


def _safetensors(header):
    return struct.pack("<Q", len(header)) + header


def _write_safetensors_header(model, header):
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    encoded = json.dumps(header).encode()
    (model / "model.safetensors").write_bytes(_safetensors(encoded) + b"\0" * 64)


def test_model_cache_check_copies_files_without_accepting_the_checkpoint(
    tmp_path, monkeypatch
):
    model = tmp_path / "Qwen3-0.6B"
    model.mkdir()
    cache = tmp_path / "cache" / "Qwen3-0.6B"
    cache.mkdir(parents=True)
    (cache / "config.json").write_text("cached")

    def verify(*args, **kwargs):
        raise AssertionError("the copied cache must be verified separately")

    monkeypatch.setattr(environment_check, "assert_model_dir", verify)
    assert environment_check.stage_model("check", str(model), str(cache)) == 0
    assert (model / "config.json").read_text() == "cached"


def test_model_cache_verification_reports_an_incomplete_checkpoint(tmp_path):
    model = tmp_path / "Qwen3-0.6B"
    model.mkdir()
    assert environment_check.stage_model("verify", str(model)) == 1


def test_model_stage_downloads_every_file_of_an_empty_directory(tmp_path, monkeypatch):
    model = tmp_path / "Qwen3-0.6B"
    model.mkdir()
    downloaded = []

    def download(name, destination):
        downloaded.append(name)
        (destination / name).write_bytes(name.encode())

    monkeypatch.setattr(environment_check, "_download_checkpoint_file", download)
    monkeypatch.setattr(environment_check, "assert_model_dir", lambda *a, **k: None)
    assert environment_check.stage_model("stage", str(model)) == 0
    assert downloaded == list(environment_check.QWEN3_FILE_SIZES)


def test_model_download_uses_the_immutable_revision_and_checks_the_digest(
    tmp_path, monkeypatch
):
    name = "config.json"
    content = b"pinned config"
    monkeypatch.setattr(environment_check, "QWEN3_FILE_SIZES", {name: len(content)})
    monkeypatch.setattr(
        environment_check,
        "QWEN3_SHA256",
        {name: environment_check.hashlib.sha256(content).hexdigest()},
    )

    class Response(io.BytesIO):
        headers = {
            "X-Repo-Commit": environment_check.QWEN3_REVISION,
            "X-Linked-Size": str(len(content)),
        }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def urlopen(request, timeout):
        assert environment_check.QWEN3_REVISION in request.full_url
        assert timeout == 120
        return Response(content)

    monkeypatch.setattr(environment_check.urllib.request, "urlopen", urlopen)
    environment_check._download_checkpoint_file(name, tmp_path)
    assert (tmp_path / name).read_bytes() == content

    (tmp_path / name).write_bytes(b"tampered conf")
    environment_check._download_checkpoint_file(name, tmp_path)
    assert (tmp_path / name).read_bytes() == content


def test_weight_validation_reads_a_safetensors_header(tmp_path):
    model = tmp_path / "Qwen3-0.6B"
    _write_safetensors_header(model, {"weight": {}})

    assert (
        environment_check.main(
            ["check-model", "--path", str(model), "--verify-weights"]
        )
        == 0
    )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (None, "has no .safetensors weights"),
        (b"short", "is truncated"),
        (struct.pack("<Q", 8), "header exceeds file size"),
        (struct.pack("<Q", 1_000_000), "header exceeds file size"),
        (_safetensors(b"{bad}") + b"\0" * 16, "cannot be read"),
        (_safetensors(b"[]") + b"\0" * 16, "must be a JSON object"),
        (_safetensors(b"{}") + b"\0" * 16, "header has no tensors"),
    ],
)
def test_weight_validation_rejects_a_truncated_snapshot(tmp_path, weights, message):
    model = tmp_path / "Qwen3-0.6B"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    if weights is not None:
        (model / "model.safetensors").write_bytes(weights)

    with pytest.raises(AssertionError, match=re.escape(message)):
        environment_check.assert_model_dir(str(model), verify_weights=True)


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
