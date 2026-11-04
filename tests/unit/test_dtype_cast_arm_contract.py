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
import shutil
import subprocess
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


def _probe_text() -> str:
    """The probe's own CMake lines, lifted out of ``csrc/CMakeLists.txt``.

    Taken from the source rather than restated here, so the executable check
    below runs the text the build runs.
    """
    cmake = _read(CSRC_CMAKE)
    start = cmake.find("set(FLAGOS_TORCH_LINKS_LIBTORCH_CUDA OFF)")
    end = cmake.find('message(STATUS "torch_fl: torch links libtorch_cuda.so')
    assert start != -1 and end != -1 and start < end, (
        "could not locate the FLAGOS_TORCH_LINKS_LIBTORCH_CUDA probe in csrc/CMakeLists.txt"
    )
    return cmake[start:end]


def _run_probe(tmp_path, *, ships_libtorch_cuda: bool) -> str:
    """Configure a throwaway project against a fake torch prefix; return ON/OFF.

    The probe's whole job is to answer a question about the torch that
    ``find_package(Torch)`` resolved, which text inspection cannot check: the
    previous version of this test asserted the probe named
    ``torch_cpu_library``, and that is precisely what made it answer OFF
    everywhere -- an INTERFACE IMPORTED target carries no IMPORTED_LOCATION. So
    build two prefixes that differ only in whether ``libtorch_cuda.so`` is
    present and let CMake decide.
    """
    cmake_exe = shutil.which("cmake")
    if cmake_exe is None:
        pytest.skip("cmake is not installed")

    prefix = tmp_path / ("torch_cuda" if ships_libtorch_cuda else "torch_cpu")
    (prefix / "lib").mkdir(parents=True)
    (prefix / "share" / "cmake" / "Torch").mkdir(parents=True)
    (prefix / "lib" / "libtorch_cpu.so").touch()
    if ships_libtorch_cuda:
        (prefix / "lib" / "libtorch_cuda.so").touch()

    # A stand-in for the two things of torch's CMake export the probe can see.
    #
    # TORCH_INSTALL_PREFIX is derived from the config file's own location, as
    # TorchConfig.cmake does it, which is what makes it name the torch that was
    # found rather than a guess.
    #
    # The two library targets are declared with the shapes the real export uses,
    # because the shape is the whole trap: `torch_cpu_library` is INTERFACE
    # IMPORTED and so can never carry IMPORTED_LOCATION, while `torch_cpu` is
    # SHARED IMPORTED and carries its path under IMPORTED_LOCATION_RELEASE. A
    # probe that reads IMPORTED_LOCATION off either one gets nothing. Declaring
    # them here means a probe written the old way configures successfully and
    # answers OFF -- the real failure -- instead of erroring on a missing target.
    (prefix / "share" / "cmake" / "Torch" / "TorchConfig.cmake").write_text(
        "get_filename_component(TORCH_INSTALL_PREFIX\n"
        '    "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)\n'
        "add_library(torch_cpu_library INTERFACE IMPORTED)\n"
        "add_library(torch_python_library INTERFACE IMPORTED)\n"
        "add_library(torch_cpu SHARED IMPORTED)\n"
        "set_target_properties(torch_cpu PROPERTIES\n"
        '    IMPORTED_LOCATION_RELEASE "${TORCH_INSTALL_PREFIX}/lib/libtorch_cpu.so")\n',
        encoding="utf-8",
    )

    project = tmp_path / ("src_on" if ships_libtorch_cuda else "src_off")
    project.mkdir()
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        "project(probe NONE)\n"
        "find_package(Torch REQUIRED)\n"
        f"{_probe_text()}"
        'message(STATUS "FLAGOS_PROBE=${FLAGOS_TORCH_LINKS_LIBTORCH_CUDA}")\n',
        encoding="utf-8",
    )

    build = tmp_path / ("build_on" if ships_libtorch_cuda else "build_off")
    done = subprocess.run(
        [
            cmake_exe,
            "-S",
            str(project),
            "-B",
            str(build),
            f"-DCMAKE_PREFIX_PATH={prefix / 'share' / 'cmake'}",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, f"cmake failed:\n{done.stdout}\n{done.stderr}"
    match = re.search(r"FLAGOS_PROBE=(\w+)", done.stdout)
    assert match, f"probe printed nothing:\n{done.stdout}"
    return match.group(1)


@pytest.mark.anyplatform
def test_the_probe_answers_on_when_torch_ships_libtorch_cuda(tmp_path):
    """A torch that carries the CUDA kernels must turn the define on.

    This is the half that regressed silently: the probe read IMPORTED_LOCATION
    off an INTERFACE IMPORTED target, got -NOTFOUND, and answered OFF even here,
    leaving all four on-device arms dead on every wheel.
    """
    assert _run_probe(tmp_path, ships_libtorch_cuda=True) == "ON"


@pytest.mark.anyplatform
def test_the_probe_answers_off_on_a_cpu_only_torch(tmp_path):
    """The CPU-only pip torch every CI lane pins must keep the host path."""
    assert _run_probe(tmp_path, ships_libtorch_cuda=False) == "OFF"


@pytest.mark.anyplatform
def test_the_probe_keys_on_libtorch_cuda_and_not_on_a_name():
    """Whatever the mechanism, the question asked must be about the library.

    An accelerator name or a build option would guess at a property of the torch
    that was found instead of reading it.
    """
    probe = _probe_text()
    assert "libtorch_cuda.so" in probe, (
        "the probe does not look for libtorch_cuda.so before turning the define on"
    )
    assert re.search(r"set\(FLAGOS_TORCH_LINKS_LIBTORCH_CUDA\s+ON\)", probe), (
        "nothing ever turns FLAGOS_TORCH_LINKS_LIBTORCH_CUDA on"
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
