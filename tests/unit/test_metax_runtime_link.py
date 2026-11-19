# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""The SDK runtime must link even when numerical vendor kernels are disabled."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("vendor_kernels", [False, True])
def test_metax_runtime_symbols_do_not_depend_on_vendor_kernels(
    tmp_path, vendor_kernels
):
    cmake = shutil.which("cmake")
    if cmake is None or shutil.which("cc") is None:
        pytest.skip("link conformance requires CMake and a host C compiler")
    sdk = tmp_path / "maca"
    (sdk / "tools/cu-bridge/include").mkdir(parents=True)
    (sdk / "lib").mkdir()
    source = (ROOT / "CMakeLists.txt").read_text()
    arm = source.split("elseif(_flagos_is_metax)", 1)[1].split(
        "elseif(_flagos_is_cambricon)", 1
    )[0]
    (tmp_path / "runtime.c").write_text("int __wcuda_version_internal__ = 3813;\n")
    (tmp_path / "consumer.c").write_text(
        "extern int __wcuda_version_internal__;\n"
        "int main(void) { return __wcuda_version_internal__ == 3813 ? 0 : 1; }\n"
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.19)\nproject(runtime_link C)\n"
        f'set(MACA_PATH "{sdk}")\n'
        f"set(FLAGOS_BUILD_VENDOR {'ON' if vendor_kernels else 'OFF'})\n"
        "set(FLAGOS_BUILD_BOXING OFF)\n"
        "add_library(fixture_runtime SHARED runtime.c)\n"
        f'set_target_properties(fixture_runtime PROPERTIES OUTPUT_NAME runtime_cu LIBRARY_OUTPUT_DIRECTORY "{sdk / "lib"}")\n'
        + arm
        + "\nadd_executable(consumer consumer.c)\n"
        "add_dependencies(consumer fixture_runtime)\n"
        "target_link_libraries(consumer PRIVATE ${CUDA_RUNTIME_LIB})\n"
    )
    for args in (
        [cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        [cmake, "--build", str(tmp_path / "build")],
    ):
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(tmp_path / "build/consumer")], check=True, timeout=10)
