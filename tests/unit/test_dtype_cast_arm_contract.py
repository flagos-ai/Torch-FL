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

"""Every non-boxing CUDA arm may only box where libtorch_cuda exists.

``copy_ops.cc::_to_copy`` serves a flagos -> flagos dtype cast on a CUDA-ABI
platform one of two ways once boxing is off: box the tensors to CUDA and run
ATen's native strided copy, or round-trip through host memory. The native copy
is a CUDA DispatchStub kernel, and the default NVIDIA wheel is built against
the CPU-only pip torch, where that stub is empty -- boxing there aborted the
process with "DispatchStub: missing kernel for cuda" (run 40285867035,
``test_compile_dtypes[float16]``). The arm must therefore branch on the CMake
probe that saw libtorch_cuda.so next to the torch it links, in both files.

The same abort came back through the other arms that box: the flagos -> flagos
strided copy in ``_copy_from`` and the strided gathers in
``contiguous_ops.cc::contiguous``/``clone``, which serve ``_to_copy``'s
``contiguous()`` and ``clone(memory_format=...)``. Run 40656617119 measured it
on the default PPU and NVIDIA wheels (four cases in ``test_ops.py``'s
``TestAutograd`` and ``test_clone_contiguous_format_works``). Each of those arms
takes the same link-time branch.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
COPY_OPS = REPO / "csrc/aten/copy_ops.cc"
CONTIGUOUS_OPS = REPO / "csrc/aten/contiguous_ops.cc"
CSRC_CMAKE = REPO / "csrc/CMakeLists.txt"


def _read(path: Path) -> str:
    """``path``'s text, or skip when this job has no checkout to read it from.

    These pins read the arm out of the tree rather than measuring it, so they
    answer on a host with no device -- but not on a job that installed the
    wheel and runs from a temporary directory, where ``REPO`` holds no
    ``csrc/``. Reading it there is a ``FileNotFoundError``, which reads as the
    regression the pin looks for and is not one, so it is a skip (the guard
    ``tests/integration/ops/test_dtype_cast_copy.py`` established, #675).
    """
    if not path.is_file():
        pytest.skip(f"no source checkout to read {path} from")
    return path.read_text(encoding="utf-8")


def _arm() -> str:
    src = _read(COPY_OPS)
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
    cmake = _read(CSRC_CMAKE)
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


@pytest.mark.anyplatform
def test_copy_from_arm_boxes_only_when_libtorch_cuda_is_linked():
    """The flagos -> flagos strided copy in ``_copy_from`` takes the same branch.

    It is the arm every same-device ``copy_`` with a dtype change or a strided
    side lands on, including the ones ``_to_copy`` performs after its own
    ``contiguous()``, so an unconditional ``DeviceBoxingGuard`` there aborts the
    same way the dtype cast did.
    """
    src = _read(COPY_OPS)
    branch = src.find("// Fast path: both contiguous, same shape and dtype")
    assert branch != -1, "the same-device flagos copy fast path is gone"
    start = src.find("// CUDA platform without the boxing kernels", branch)
    assert start != -1, "the non-boxing _copy_from arm is gone"
    end = src.find("// Ascend copies on-device via aclnnInplaceCopy", start)
    assert end != -1, "the de-boxed sibling of the _copy_from arm is gone"
    arm = src[start:end]
    assert "FLAGOS_TORCH_LINKS_LIBTORCH_CUDA" in arm, (
        "the non-boxing _copy_from arm boxes unconditionally again, which calls "
        "into an empty CUDA DispatchStub on a CPU-only torch"
    )
    assert "DeviceBoxingGuard guard(self, dst);" in arm, (
        "the arm lost its on-device route, so a build that links libtorch_cuda "
        "pays a host round trip per copy"
    )
    assert "strided_copy_cast_via_cpu(self, dst);" in arm, (
        "the host round-trip the arm takes without libtorch_cuda is missing, or "
        "the unsigned-dtype escape hatch is"
    )


@pytest.mark.anyplatform
def test_contiguous_gather_arms_box_only_when_libtorch_cuda_is_linked():
    """``contiguous()`` and ``clone()`` gather on device only when they can.

    Both arms box the source to CUDA and call ``at::native::copy_``, which is
    the same CUDA DispatchStub. ``clone(memory_format=contiguous_format)``
    reaches the clone arm directly and ``_to_copy`` reaches the contiguous arm
    for every non-contiguous source, so the predicate has to include the
    libtorch_cuda probe next to the boxing build.
    """
    src = _read(CONTIGUOUS_OPS)
    define = src.find("#define FLAGOS_CONTIGUOUS_CUDA_BOXING 1")
    assert define != -1, "the arm predicate is gone"
    predicate = src[src.rfind("#if", 0, define) : define]
    assert "FLAGOS_TORCH_LINKS_LIBTORCH_CUDA" in predicate, (
        "FLAGOS_CONTIGUOUS_CUDA_BOXING no longer asks whether this wheel's torch "
        "ships libtorch_cuda, so it boxes into an empty CUDA DispatchStub"
    )
    assert "FLAGOS_BOXING_BUILD" in predicate, (
        "the predicate dropped the boxing build, which does link libtorch_cuda"
    )

    arms = re.findall(r"#if FLAGOS_CONTIGUOUS_CUDA_BOXING(.*?)#endif", src, re.S)
    assert len(arms) == 2, (
        "expected the contiguous() and clone() strided-gather arms to branch on "
        f"FLAGOS_CONTIGUOUS_CUDA_BOXING, found {len(arms)}"
    )
    for arm in arms:
        boxed, _, host = arm.partition("#else")
        assert "DeviceBoxingGuard" in boxed and "at::native::copy_" in boxed, (
            "an arm lost its on-device gather, so a build that links "
            "libtorch_cuda pays two blocking memcpy per call"
        )
        assert "copy_strided_via_host(" in host, (
            "an arm boxes without a host-gather fallback for the wheel whose "
            "torch has no CUDA copy kernel"
        )
