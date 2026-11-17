# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Configure the actual MetaX vendor-library link block without an SDK."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("metax,boxing", [(True, False), (True, True), (False, False)])
def test_forked_cuda_library_is_only_linked_for_metax_boxing(tmp_path, metax, boxing):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is unavailable")
    source = (ROOT / "csrc/CMakeLists.txt").read_text()
    start = source.index("  set(_metax_torch_lib")
    start = source.rfind("\nif(", 0, start) + 1
    end = source.index("\nif(_flagos_is_ascend)", start)
    block = source[start:end]
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    library = vendor / "libtorch_cuda.so"
    library.touch()
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(probe NONE)\n"
        "add_library(probe IMPORTED INTERFACE)\nset(LIBRARY_NAME probe)\n"
        f"set(_flagos_is_metax {'ON' if metax else 'OFF'})\n"
        f"set(FLAGOS_BUILD_BOXING {'ON' if boxing else 'OFF'})\n"
        "set(FLAGOS_BUILD_VENDOR OFF)\n"
        "set(_torch_fl_link_libs torch_cpu_library flagos runtime_cu)\n"
        + block.replace("PRIVATE", "INTERFACE")
        + '\nfile(WRITE "${CMAKE_BINARY_DIR}/libraries.txt" "${_torch_fl_link_libs}")\n'
    )
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        env={**os.environ, "FLAGOS_VENDOR_TORCH_LIB": str(vendor)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    linked = (tmp_path / "build/libraries.txt").read_text().split(";")
    assert linked[:3] == ["torch_cpu_library", "flagos", "runtime_cu"]
    assert (str(library) in linked) == (metax and boxing)
