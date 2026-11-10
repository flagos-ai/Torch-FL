# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""The runtime audit must detect vendor binaries behind CPU Python metadata."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "iluvatar_verify", ROOT / ".github/scripts/verify_iluvatar.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_corex_headers_precede_profiler_cuda_headers(tmp_path):
    """Compile the actual Iluvatar target setup beside a shadow CUDA SDK."""
    if not shutil.which("cmake") or not shutil.which("c++"):
        pytest.skip("CMake and a host C++ compiler are required")
    sdk = tmp_path / "corex"
    cuda = tmp_path / "cuda"
    sdk.mkdir()
    cuda.mkdir()
    (sdk / "cublas_v2.h").write_text("typedef int ixblasDataType_t;\n")
    (cuda / "cublas_v2.h").write_text("typedef int cublasDataType_t;\n")
    (tmp_path / "probe.cc").write_text(
        "#include <cublas_v2.h>\nixblasDataType_t value = 0;\n"
    )
    branch = (
        (ROOT / "csrc/CMakeLists.txt")
        .read_text()
        .split("elseif(_flagos_is_iluvatar)\n", 1)[1]
        .split("elseif(_flagos_is_musa)", 1)[0]
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.19)\nproject(probe LANGUAGES CXX)\n"
        "set(LIBRARY_NAME probe)\nadd_library(probe STATIC probe.cc)\n"
        f'target_include_directories(probe PRIVATE "{cuda}")\n'
        "set(FLAGOS_BUILD_VENDOR ON)\n"
        f'set(FLAGOS_ILUVATAR_VENDOR_INCLUDE_DIR "{sdk}")\n' + branch
    )
    build = tmp_path / "build"
    subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build)],
        capture_output=True,
        text=True,
        check=True,
    )
    done = subprocess.run(
        ["cmake", "--build", str(build)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr


def cpu_mappings(root):
    return "\n".join(
        f"1000-2000 r-xp 00000000 08:01 123 {root}/lib/{name}"
        for name in ("libc10.so", "libtorch_cpu.so", "libtorch_python.so")
    )


def test_official_cpu_core_allows_corex_device_libraries(tmp_path):
    torch = tmp_path / "torch"
    mappings = cpu_mappings(torch)
    mappings += "\n1000-2000 r-xp 0 08:01 1 /usr/local/corex/lib64/libcudart.so"
    verifier.assert_cpu_core(torch, tmp_path / "torch_fl", mappings)


@pytest.mark.parametrize("name", ["libtorch_cuda.so", "libc10_cuda.so"])
def test_vendor_cuda_core_is_rejected(tmp_path, name):
    torch = tmp_path / "torch"
    mappings = cpu_mappings(torch)
    mappings += f"\n1000-2000 r-xp 0 08:01 1 /vendor/torch/lib/{name}"
    with pytest.raises(AssertionError, match="Vendor CUDA libtorch"):
        verifier.assert_cpu_core(torch, tmp_path / "torch_fl", mappings)


def test_vendor_cpu_core_cannot_impersonate_the_official_distribution(tmp_path):
    torch = tmp_path / "torch"
    with pytest.raises(AssertionError, match="official CPU torch"):
        verifier.assert_cpu_core(
            torch, tmp_path / "torch_fl", cpu_mappings(tmp_path / "vendor/torch")
        )


def test_a_second_loaded_cpu_core_is_rejected(tmp_path):
    torch = tmp_path / "torch"
    mappings = cpu_mappings(torch) + "\n" + cpu_mappings(tmp_path / "vendor/torch")
    with pytest.raises(AssertionError, match="second libc10.so"):
        verifier.assert_cpu_core(torch, tmp_path / "torch_fl", mappings)


def test_stale_wheel_bundle_is_rejected(tmp_path):
    package = tmp_path / "torch_fl"
    (package / "lib_iluvatar").mkdir(parents=True)
    with pytest.raises(AssertionError):
        verifier.assert_cpu_core(tmp_path / "torch", package, "")


def test_runtime_venv_root_uses_the_job_environment(tmp_path, monkeypatch):
    root = tmp_path / "job-venv"
    root.mkdir()
    (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    monkeypatch.setenv("VIRTUAL_ENV", str(root))
    monkeypatch.setattr(
        verifier, "sys", SimpleNamespace(prefix=str(root), base_prefix="/usr")
    )
    assert verifier.isolated_venv_root() == root
    verifier.assert_venv_package(root, root / "lib/python3.12/site-packages/torch")
    with pytest.raises(AssertionError, match="outside CI venv"):
        verifier.assert_venv_package(root, tmp_path / "image/torch")


@pytest.mark.parametrize("problem", ["system-site", "wrong-prefix", "base-python"])
def test_runtime_rejects_shared_or_misidentified_python(tmp_path, monkeypatch, problem):
    root = tmp_path / "job-venv"
    root.mkdir()
    (root / "pyvenv.cfg").write_text(
        f"include-system-site-packages = {'true' if problem == 'system-site' else 'false'}\n"
    )
    monkeypatch.setenv(
        "VIRTUAL_ENV", str(root / "other" if problem == "wrong-prefix" else root)
    )
    monkeypatch.setattr(
        verifier,
        "sys",
        SimpleNamespace(
            prefix=str(root),
            base_prefix=str(root) if problem == "base-python" else "/usr",
        ),
    )
    with pytest.raises(AssertionError):
        verifier.isolated_venv_root()


@pytest.mark.parametrize(
    "problem",
    [
        "",
        "vendor-torch",
        "foreign-torch",
        "foreign-triton",
        "foreign-path",
        "cmake4",
        "wrong-tree",
    ],
)
def test_setup_verifier_keeps_cpu_core_and_compiler_contract(
    tmp_path, monkeypatch, problem
):
    root = tmp_path / "venv"
    root.mkdir()
    (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    sdk = tmp_path / "sdk"
    for directory in (sdk / "include", sdk / "lib"):
        directory.mkdir(parents=True)
    for header in ("cuda_runtime.h", "cudnn.h", "cublas_v2.h"):
        (sdk / "include" / header).touch()
    for library in ("libcudart.so", "libcudnn.so", "libcublas.so"):
        (sdk / "lib" / library).touch()
    monkeypatch.setenv("VIRTUAL_ENV", str(root))
    monkeypatch.setenv("COREX_PATH", str(sdk))
    monkeypatch.setenv("TORCH_FL_VERSION_PINS", str(ROOT / ".github/version-pins.env"))
    monkeypatch.delenv("FLAGTREE_VERSION", raising=False)
    monkeypatch.delenv("FLAGGEMS_VERSION", raising=False)
    torch_file = (
        tmp_path / "vendor/torch/__init__.py"
        if problem == "foreign-torch"
        else root / "site-packages/torch/__init__.py"
    )
    triton_file = (
        tmp_path / "vendor/triton/__init__.py"
        if problem == "foreign-triton"
        else root / "site-packages/triton/__init__.py"
    )
    torch = SimpleNamespace(
        __version__="2.10.0+corex" if problem == "vendor-torch" else "2.10.0+cpu",
        __file__=str(torch_file),
        version=SimpleNamespace(cuda=None),
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
    )
    triton = SimpleNamespace(
        __version__="3.6.0",
        __file__=str(triton_file),
        backends=SimpleNamespace(backends={"iluvatar": object()}),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setattr(
        verifier,
        "sys",
        SimpleNamespace(
            prefix=str(root),
            base_prefix="/usr",
            version_info=(3, 12),
            version="3.12.12",
            path=[str(root / "site-packages")]
            + (
                ["/flagos/lib/python3.12/site-packages"]
                if problem == "foreign-path"
                else []
            ),
        ),
    )
    versions = {
        "torch": "2.10.0+cpu",
        "flagtree": "0.7.0+iluvatar3.6"
        if problem == "wrong-tree"
        else "0.7.1a1+iluvatar3.6",
        "flag_gems": "5.4.0",
        "cmake": "4.0" if problem == "cmake4" else "3.31.8",
    }
    monkeypatch.setattr(verifier.importlib.metadata, "version", versions.__getitem__)
    original_spec = verifier.importlib.util.find_spec
    monkeypatch.setattr(
        verifier.importlib.util,
        "find_spec",
        lambda name: None if name == "vllm" else original_spec(name),
    )
    monkeypatch.setattr(verifier.shutil, "which", lambda name: str(root / "bin" / name))
    monkeypatch.setattr(verifier.subprocess, "run", lambda *args, **kwargs: None)
    if problem:
        with pytest.raises(AssertionError):
            verifier.verify()
    else:
        verifier.verify()
