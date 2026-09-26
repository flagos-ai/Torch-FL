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

"""Unit tests for the standalone _flagos_nccl build configuration."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BUILD_PATH = (
    Path(__file__).resolve().parents[2] / "torch_fl" / "comm" / "_nccl_ext" / "build.py"
)
_SPEC = importlib.util.spec_from_file_location("torch_fl_nccl_ext_build", _BUILD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
build = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = build
_SPEC.loader.exec_module(build)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _make_dtk(root):
    _touch(root / "cuda" / "cuda-12" / "include" / "cuda.h")
    _touch(root / "include" / "rccl" / "nccl.h")
    _touch(root / "lib" / "librccl.so")


def _make_dcu_torch_lib(root):
    _touch(root / "libc10_hip.so")
    _touch(root / "libtorch_hip.so")


def test_cuda_configuration_preserves_existing_link_set(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    assets = repo / ".libtorch_cuda_assets"
    torch_fl_lib = repo / "torch_fl" / "lib"
    nccl_root = tmp_path / "nvidia" / "nccl"
    cuda_root = tmp_path / "cuda"
    for directory in (assets, torch_fl_lib, nccl_root / "lib", cuda_root / "include"):
        directory.mkdir(parents=True)

    fake_spec = SimpleNamespace(submodule_search_locations=[str(tmp_path / "nvidia")])
    monkeypatch.setattr(build.importlib.util, "find_spec", lambda name: fake_spec)

    config = build._build_config(
        {"FLAGOS_ACCELERATOR": "cuda", "CUDA_HOME": str(cuda_root)},
        str(repo),
    )

    assert config.accelerator == "cuda"
    assert config.libraries == ("c10_cuda", "torch_cuda", "nccl")
    assert config.library_dirs == (
        str(assets),
        str(torch_fl_lib),
        str(nccl_root / "lib"),
    )
    assert config.include_dirs == (
        str(cuda_root / "include"),
        str(nccl_root / "include"),
    )
    assert config.rpaths == config.library_dirs


def test_dcu_configuration_uses_bundled_libtorch_and_rccl(tmp_path):
    repo = tmp_path / "repo"
    dtk = tmp_path / "dtk"
    bundled = repo / "torch_fl" / "lib_dcu"
    _make_dtk(dtk)
    _make_dcu_torch_lib(bundled)

    config = build._build_config(
        {"FLAGOS_ACCELERATOR": "dcu", "ROCM_PATH": str(dtk)},
        str(repo),
    )

    assert config.accelerator == "dcu"
    assert config.libraries == ("c10_hip", "torch_hip", "rccl")
    assert config.include_dirs == (
        str(dtk / "cuda" / "cuda-12" / "include"),
        str(dtk / "include"),
        str(dtk / "include" / "rccl"),
    )
    assert config.library_dirs == (str(bundled), str(dtk / "lib"))
    assert config.rpaths == (
        "$ORIGIN/../../lib_dcu",
        str(dtk / "lib"),
    )


def test_dcu_configuration_accepts_external_vendor_torch_lib(tmp_path):
    repo = tmp_path / "repo"
    dtk = tmp_path / "dtk"
    vendor_lib = tmp_path / "vendor-torch" / "lib"
    _make_dtk(dtk)
    _make_dcu_torch_lib(vendor_lib)

    config = build._build_config(
        {
            "ACCELERATOR": "dcu",
            "ROCM_PATH": str(dtk),
            "FLAGOS_VENDOR_TORCH_LIB": str(vendor_lib),
        },
        str(repo),
    )

    assert config.library_dirs == (str(vendor_lib), str(dtk / "lib"))
    assert config.rpaths == (str(vendor_lib), str(dtk / "lib"))


def test_dcu_configuration_rejects_missing_dtk_root(tmp_path):
    missing = tmp_path / "missing-dtk"
    with pytest.raises(RuntimeError, match="ROCM_PATH=.*does not name a DTK"):
        build._build_config(
            {"FLAGOS_ACCELERATOR": "dcu", "ROCM_PATH": str(missing)},
            str(tmp_path / "repo"),
        )


def test_dcu_configuration_reports_missing_vendor_libraries(tmp_path):
    repo = tmp_path / "repo"
    dtk = tmp_path / "dtk"
    _make_dtk(dtk)

    with pytest.raises(RuntimeError, match="libc10_hip, libtorch_hip"):
        build._build_config(
            {"FLAGOS_ACCELERATOR": "dcu", "ROCM_PATH": str(dtk)},
            str(repo),
        )


@pytest.mark.parametrize(("version", "expected"), [("11.4.0\n", 11), ("9\n", 9)])
def test_dcu_compiler_accepts_gcc_9_or_newer(monkeypatch, version, expected):
    monkeypatch.setattr(build.shutil, "which", lambda command: "/opt/gcc/bin/g++")
    monkeypatch.setattr(
        build.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=version),
    )

    assert build._validate_dcu_compiler({"CXX": "g++"}) == expected


def test_dcu_compiler_rejects_gcc_8(monkeypatch):
    monkeypatch.setattr(build.shutil, "which", lambda command: "/usr/bin/g++")
    monkeypatch.setattr(
        build.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="8.5.0\n"),
    )

    with pytest.raises(RuntimeError, match="requires GCC 9 or newer"):
        build._validate_dcu_compiler({"CXX": "g++"})
