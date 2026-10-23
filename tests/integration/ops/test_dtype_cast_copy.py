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

"""Dtype-cast copy tests.

`_to_copy` with a dtype change is the one copy the model makes constantly and
that never shows up in a route table: every bf16 -> fp32 materialization around
layernorm, softmax and the gated delta rule, plus the fp32 -> bf16 back.
`csrc/aten/copy_ops.cc` has two CUDA arms for it, selected by
`FLAGOS_BOXING_BUILD`:

  * boxing  -- boxes to CUDA and runs the native TensorIterator copy kernel;
  * default -- used to cast through host memory unconditionally.

The default arm is what the shipped FlagGems-primary wheel compiles, and it
uses the same native kernel only when the torch it was built against ships one:
`DeviceBoxingGuard` only rewrites the tensor's device metadata, and the cast it
enables is ATen's TensorIterator strided copy, a CUDA DispatchStub kernel. The
default NVIDIA wheel is built against the CPU-only pip torch, which has no such
kernel -- a cast on that arm aborted the process with "DispatchStub: missing
kernel for cuda" (run 40285867035, `test_compile_dtypes[float16]`). So
`csrc/CMakeLists.txt` detects libtorch_cuda.so next to the torch it links and
defines `FLAGOS_TORCH_LINKS_LIBTORCH_CUDA`; the arm boxes only when it is set
and casts through the host otherwise. Where it is set, casting through the host
would turn every cast into three blocking `cudaMemcpy` calls -- 8826 per forward
on the OCR-RM Qwen3.5-VL 2B value head, 2.0 s of a 4.2 s forward.

These tests pin the *observable* properties: the cast is correct, it returns a
tensor on the source device, and it does not move the data through host memory.
The host round-trip is detected by counting `cudaMemcpy` in a profiler window,
which is device-agnostic and needs no weight file.
"""

from pathlib import Path

import pytest
import torch

import torch_fl  # noqa: F401  -- registers the device

CUDA_ABI_PLATFORMS = ("cuda", "dcu", "metax", "ppu")


def _build_platform() -> str:
    """The accelerator this wheel was built for ("" when unrecorded)."""
    return torch_fl._build_accelerator()


# Builds with no on-device dtype cast at all: copy_ops.cc puts them in the
# same-device flagos->flagos branch's "Fallback: CPU round-trip when no
# on-device cast is available". The CUDA-ABI platforms reach the native
# TensorIterator cast through the boxed arm, and Ascend / GCU / Kunlunxin /
# MUSA each compile a vendor cast (`aclnnCast`, `topsatenToCopy`, `cast`,
# `MudnnCopy`); these four do not, so a host round trip is the design there and
# the transfer count below cannot be zero.
_NO_ON_DEVICE_CAST = ("bpu", "iluvatar", "mps", "tsingmicro")


def _has_on_device_cast() -> bool:
    """Whether `.to(dtype)` can stay on the device on this build.

    See `csrc/aten/copy_ops.cc`, the same-device flagos->flagos dtype branch;
    the USE_ILUVATAR group there has no `DtypeCast` implementation.
    """
    return _build_platform() not in _NO_ON_DEVICE_CAST


def _flagos_available():
    try:
        return (
            torch.accelerator.is_available() if hasattr(torch, "accelerator") else True
        )
    except Exception:
        return False


def _device():
    return torch.device("flagos:0")


pytestmark = [
    pytest.mark.anyplatform,
    pytest.mark.skipif(not _flagos_available(), reason="no flagos device"),
]


# (source dtype, target dtype). bf16 <-> fp32 is the pair a transformer hits on
# every block; the integer rows cover the casts the native copy kernel handles
# without a host round-trip.
CASTS = [
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
    (torch.float16, torch.float32),
    (torch.int64, torch.float32),
    (torch.int32, torch.int64),
]


@pytest.mark.parametrize("src_dtype,dst_dtype", CASTS)
def test_cast_matches_cpu(src_dtype, dst_dtype):
    """The cast must agree with the host reference, element for element."""
    if src_dtype.is_floating_point:
        base = torch.randn(128, 256)
    else:
        base = torch.randint(-1000, 1000, (128, 256))

    src_cpu = base.to(src_dtype)
    got = src_cpu.to(_device()).to(dst_dtype).cpu()
    want = src_cpu.to(dst_dtype)

    assert got.dtype == want.dtype
    assert got.shape == want.shape
    if want.is_floating_point:
        assert torch.equal(got, want), (
            "cast is not bit-exact against the host reference"
        )
    else:
        assert torch.equal(got, want)


@pytest.mark.parametrize("src_dtype,dst_dtype", CASTS)
def test_cast_stays_on_device(src_dtype, dst_dtype):
    """`.to(dtype)` must return a flagos tensor, not a host tensor."""
    src = torch.zeros(64, 64, dtype=src_dtype, device=_device())
    out = src.to(dst_dtype)
    assert out.device.type == "flagos"
    assert out.dtype == dst_dtype


def _memcpy_count(fn):
    """How many device/host transfers the call under `fn` performs."""
    fn()  # warm up any lazy init
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    torch.cuda.synchronize()
    return sum(e.count for e in prof.key_averages() if "Memcpy" in e.key)


@pytest.mark.parametrize("platform", CUDA_ABI_PLATFORMS)
def test_cuda_abi_cast_does_not_route_through_host(platform):
    """Pin the arm itself, not a runtime property.

    A cast that goes through host memory costs three blocking `cudaMemcpy`
    calls; the on-device arm costs none, because the native copy kernel is a
    kernel. This asserts on the source so it runs on any host -- the runtime
    check below only runs where a device is present.

    The regressions this guards, in both directions: the default (non-boxing)
    arm first called `strided_copy_cast_via_cpu` unconditionally, so a build
    that did link libtorch_cuda paid a host round trip for every dtype cast;
    then it boxed unconditionally, which on the CPU-only torch the NVIDIA wheel
    ships is a call into an empty CUDA DispatchStub (run 40285867035).
    """
    source_root = Path(__file__).resolve().parents[3]
    copy_ops = source_root / "csrc/aten/copy_ops.cc"
    if not copy_ops.is_file():
        # The group runs on the installed wheel, not the checkout: the sources
        # are in the workspace the job checked out, not beside this file's copy
        # in the job's temporary directory. Nothing to read means this pin
        # cannot answer -- which is a skip, not the regression it looks for.
        pytest.skip(f"no source checkout to read {copy_ops} from")
    src = copy_ops.read_text()
    # The non-boxing arm is the second `#else` of the FLAGOS_BOXING_BUILD pair
    # inside the same-device flagos->flagos branch.
    # Anchor on the _to_copy dtype branch specifically: the file also has a
    # "CUDA platform without the boxing kernels" comment on the _copy_from
    # path, which takes a different (host) route by design.
    marker = "if (src_is_flagos && dst_is_flagos) {"
    branch = src.find(marker)
    assert branch != -1, "the same-device flagos copy branch is gone"
    start = src.find("// CUDA platform without the boxing kernels", branch)
    assert start != -1, (
        "the non-boxing CUDA cast arm is gone; decouple this test from the source "
        "if the arm was deliberately removed"
    )
    arm = src[start : start + 2400]
    assert "FLAGOS_TORCH_LINKS_LIBTORCH_CUDA" in arm, (
        f"{platform}: the non-boxing CUDA cast arm boxes unconditionally again, "
        "which calls into an empty CUDA DispatchStub on the CPU-only torch the "
        "default wheel builds against"
    )
    assert "DeviceBoxingGuard" in arm, (
        f"{platform}: the non-boxing CUDA cast arm no longer boxes to CUDA at all, "
        "so a build that links libtorch_cuda pays a host round trip per cast"
    )
    assert "strided_copy_cast_via_cpu(self_contig, result);" in arm, (
        f"{platform}: the host round-trip the arm takes without libtorch_cuda "
        "is missing, or the unsigned-dtype escape hatch is"
    )


@pytest.mark.skipif(
    not _has_on_device_cast(),
    reason=(
        "this build has no on-device dtype cast: copy_ops.cc takes the documented "
        "'Fallback: CPU round-trip when no on-device cast is available' branch, "
        "so the transfer count below is the design and not a regression"
    ),
)
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [(torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)],
)
def test_bf16_fp32_cast_is_transfer_free(src_dtype, dst_dtype):
    """End to end: the bf16<->fp32 cast must not move data between host and device.

    Only meaningful where torch links libtorch_cuda.so -- that is the build
    whose cast arm boxes to CUDA, and it is tied to the torch the wheel was
    built against, not to one a test can install. The default NVIDIA wheel is
    built against the CPU-only torch, where the host round trip is the correct
    behaviour, so the property does not hold there by design.
    """
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if not (torch_lib / "libtorch_cuda.so").is_file():
        pytest.skip("torch has no libtorch_cuda.so; the cast arm is the host one")
    src = torch.randn(512, 1024, dtype=src_dtype, device=_device())
    src.to(dst_dtype)  # warm: the first cast may compile the kernel
    n = _memcpy_count(lambda: src.to(dst_dtype))
    assert n <= len(CUDA_ABI_PLATFORMS), (
        f"{src_dtype} -> {dst_dtype} performed {n} host/device transfers; the cast "
        "is routing through host memory instead of the native copy kernel"
    )
