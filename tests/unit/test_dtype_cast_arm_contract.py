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

"""The non-boxing CUDA dtype-cast arm may only box where libtorch_cuda exists.

``copy_ops.cc::_to_copy`` serves a flagos -> flagos dtype cast on a CUDA-ABI
platform one of two ways once boxing is off: box the tensors to CUDA and run
ATen's native strided copy, or round-trip through host memory. The native copy
is a CUDA DispatchStub kernel, and the default NVIDIA wheel is built against
the CPU-only pip torch, where that stub is empty -- boxing there aborted the
process with "DispatchStub: missing kernel for cuda" (run 40285867035,
``test_compile_dtypes[float16]``). The arm must therefore branch on the CMake
probe that saw libtorch_cuda.so next to the torch it links, in both files.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
COPY_OPS = REPO / "csrc/aten/copy_ops.cc"
CSRC_CMAKE = REPO / "csrc/CMakeLists.txt"


def _arm() -> str:
    src = COPY_OPS.read_text()
    branch = src.find("if (src_is_flagos && dst_is_flagos) {")
    assert branch != -1, "the same-device flagos copy branch is gone"
    start = src.find("// CUDA platform without the boxing kernels", branch)
    assert start != -1, "the non-boxing CUDA cast arm is gone"
    return src[start : start + 2400]


@pytest.mark.anyplatform
def test_cast_arm_boxes_only_when_libtorch_cuda_is_linked():
    arm = _arm()
    guard = arm.find("FLAGOS_TORCH_LINKS_LIBTORCH_CUDA")
    boxing = arm.find("DeviceBoxingGuard")
    host = arm.find("strided_copy_cast_via_cpu(self_contig, result);")
    assert guard != -1, (
        "the non-boxing cast arm no longer branches on the libtorch_cuda probe, "
        "so it boxes into an empty CUDA DispatchStub on a CPU-only torch"
    )
    assert boxing != -1 and host != -1, (
        "the cast arm is missing one of its two routes (boxed copy / host round-trip)"
    )


@pytest.mark.anyplatform
def test_cmake_defines_the_probe_from_libtorch_cuda():
    cmake = CSRC_CMAKE.read_text()
    assert "FLAGOS_TORCH_LINKS_LIBTORCH_CUDA" in cmake
    # The only way the probe may be turned on is a real libtorch_cuda.so beside
    # the torch_cpu_library the link uses -- not an accelerator name or an
    # option, either of which guesses at a property of the torch build.
    on_assignment = [
        line
        for line in cmake.splitlines()
        if re.search(r"set\(FLAGOS_TORCH_LINKS_LIBTORCH_CUDA\s+ON\)", line)
    ]
    assert on_assignment, "nothing ever turns FLAGOS_TORCH_LINKS_LIBTORCH_CUDA on"
    probe_block = cmake[
        cmake.find("set(FLAGOS_TORCH_LINKS_LIBTORCH_CUDA OFF)") : cmake.find(
            on_assignment[0]
        )
    ]
    assert "libtorch_cuda.so" in probe_block, (
        "the probe does not look for libtorch_cuda.so before turning the define on"
    )
    assert "torch_cpu_library" in probe_block, (
        "the probe must resolve the directory of the torch it actually links"
    )
