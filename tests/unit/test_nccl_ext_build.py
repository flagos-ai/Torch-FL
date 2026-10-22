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


def test_wheel_extension_is_dcu_only(tmp_path):
    """The wheel path is scoped to DCU; every other accelerator stays standalone.

    CUDA in particular: enabling it would make the wheel build depend on
    nvidia-nccl-cu12, which is an optional extra of the installed torch rather
    than a build requirement (see build._WHEEL_ACCELERATORS).
    """
    repo = tmp_path / "repo"

    assert (
        build.wheel_extension(
            {"FLAGOS_ACCELERATOR": "cuda", "CUDA_HOME": str(tmp_path)}, str(repo)
        )
        is None
    )
    assert build.wheel_extension({"FLAGOS_ACCELERATOR": "musa"}, str(repo)) is None
    assert build.wheel_extension({}, str(repo)) is None


def test_wheel_extension_targets_the_package_path_with_the_dcu_link_set(tmp_path):
    """DCU resolves to a dotted Extension so the .so lands in the package.

    The dotted name is what makes setuptools' inplace copy put the file in
    torch_fl/comm/_nccl_ext/, the package torch_fl/comm/_nccl_ext/__init__.py
    imports from -- issue #366's completion bar, an installed wheel that loads
    the bridge with no source checkout.
    """
    repo = tmp_path / "repo"
    dtk = tmp_path / "dtk"
    vendor_lib = tmp_path / "vendor-torch" / "lib"
    _make_dtk(dtk)
    _make_dcu_torch_lib(vendor_lib)
    _make_dcu_torch_lib(repo / "torch_fl" / "lib_dcu")

    extension = build.wheel_extension(
        {
            "FLAGOS_ACCELERATOR": "dcu",
            "ROCM_PATH": str(dtk),
            "FLAGOS_VENDOR_TORCH_LIB": str(vendor_lib),
        },
        str(repo),
    )

    assert extension.name == "torch_fl.comm._nccl_ext._flagos_nccl"
    # Same leaf name on disk as the standalone builder produces, so a source
    # checkout can carry either build's output.
    assert extension.name.rsplit(".", 1)[-1] == build._STANDALONE_EXTENSION_NAME
    assert extension.sources == [
        str(Path(build.__file__).resolve().parent / "nccl_backend.cpp")
    ]
    assert extension.libraries == ["c10_hip", "torch_hip", "rccl"]
    assert dict(extension.define_macros) == {
        **dict(build._COMMON_MACROS),
        "TORCH_EXTENSION_NAME": "_flagos_nccl",
    }
    assert extension.extra_compile_args == ["-std=c++17"]
    # The bundled lib_dcu is the only RUNPATH a wheel may carry, even though the
    # vendor tree and DTK are still link search paths.
    assert extension.extra_link_args == ["-Wl,-rpath,$ORIGIN/../../lib_dcu"]
    assert str(vendor_lib) in extension.library_dirs
    assert str(repo / "torch_fl" / "lib_dcu") in extension.library_dirs
    assert str(dtk / "cuda" / "cuda-12" / "include") in extension.include_dirs
    # torch's own include/library paths are deliberately absent: resolving them
    # here would make loading setup.py import torch. setup.py adds them at
    # build_ext time instead.
    assert not any("site-packages" in directory for directory in extension.include_dirs)


def test_every_build_defines_the_module_init_symbol(tmp_path):
    """Regression guard: the wheel .so must export PyInit__flagos_nccl.

    nccl_backend.cpp writes ``PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)``. torch's
    BuildExtension defines that macro for the standalone build, but the wheel goes
    through plain setuptools, so a wheel built without this define exports
    ``PyInit_TORCH_EXTENSION_NAME`` instead and import fails with "dynamic module
    does not define module export function (PyInit__flagos_nccl)" -- observed on a
    real DCU wheel build, not hypothesised. Both entry points therefore have to
    carry it, and both have to agree on the leaf, because the symbol cannot
    contain dots.
    """
    repo = tmp_path / "repo"
    dtk = tmp_path / "dtk"
    vendor_lib = tmp_path / "vendor-torch" / "lib"
    _make_dtk(dtk)
    _make_dcu_torch_lib(vendor_lib)
    _make_dcu_torch_lib(repo / "torch_fl" / "lib_dcu")
    env = {
        "FLAGOS_ACCELERATOR": "dcu",
        "ROCM_PATH": str(dtk),
        "FLAGOS_VENDOR_TORCH_LIB": str(vendor_lib),
    }

    wheel = build.wheel_extension(env, str(repo))
    standalone_name = build._STANDALONE_EXTENSION_NAME
    assert wheel.name.rsplit(".", 1)[-1] == standalone_name

    assert dict(wheel.define_macros)["TORCH_EXTENSION_NAME"] == standalone_name
    # The standalone builder's kwargs default to the same define, which is also
    # what BuildExtension appends -- a duplicate, identical -D on its command line.
    config = build._build_config(env, str(repo))
    assert (
        dict(build._extension_kwargs(config)["define_macros"])["TORCH_EXTENSION_NAME"]
        == standalone_name
    )
    assert build._module_init_name(wheel.name) == standalone_name
    assert build._module_init_name("_flagos_nccl") == standalone_name


def test_wheel_rpaths_keeps_only_relocatable_entries(tmp_path):
    """A released wheel must not name the machine that built it.

    The standalone builder keeps every entry: it runs out of a source checkout,
    where the vendor torch/lib it was linked against has to be reachable at run
    time too.
    """
    repo = tmp_path / "repo"
    dtk = tmp_path / "dtk"
    vendor_lib = tmp_path / "vendor-torch" / "lib"
    _make_dtk(dtk)
    _make_dcu_torch_lib(vendor_lib)
    _make_dcu_torch_lib(repo / "torch_fl" / "lib_dcu")

    config = build._build_config(
        {
            "FLAGOS_ACCELERATOR": "dcu",
            "ROCM_PATH": str(dtk),
            "FLAGOS_VENDOR_TORCH_LIB": str(vendor_lib),
        },
        str(repo),
    )

    assert config.rpaths == (
        str(vendor_lib),
        "$ORIGIN/../../lib_dcu",
        str(dtk / "lib"),
    )
    assert build._wheel_rpaths(config) == ("$ORIGIN/../../lib_dcu",)
    assert build._extension_kwargs(config)["extra_link_args"] == [
        f"-Wl,-rpath,{vendor_lib}",
        "-Wl,-rpath,$ORIGIN/../../lib_dcu",
        f"-Wl,-rpath,{dtk / 'lib'}",
    ]


def test_wheel_extension_raises_when_the_dcu_link_set_is_unresolvable(tmp_path):
    """A DCU wheel without the bridge is the defect, so it must not be skipped."""
    dtk = tmp_path / "dtk"
    _make_dtk(dtk)

    with pytest.raises(RuntimeError, match="libc10_hip, libtorch_hip"):
        build.wheel_extension(
            {"FLAGOS_ACCELERATOR": "dcu", "ROCM_PATH": str(dtk)},
            str(tmp_path / "repo"),
        )
