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


def _copy_ops_source():
    """``csrc/aten/copy_ops.cc``, or skip when this job has no checkout.

    The source pins below read the arm out of the tree rather than measuring it,
    so they answer on a host with no device. What they cannot answer on is a job
    that installed the wheel and runs the suite from a temporary directory: the
    sources are in the workspace that job checked out, not three levels above
    this file's copy. Reading them there is a ``FileNotFoundError``, which reads
    as the regression the pin looks for and is not one -- so it is a skip.
    """
    copy_ops = Path(__file__).resolve().parents[3] / "csrc/aten/copy_ops.cc"
    if not copy_ops.is_file():
        pytest.skip(f"no source checkout to read {copy_ops} from")
    return copy_ops.read_text()


def _flagos_available():
    try:
        return (
            torch.accelerator.is_available() if hasattr(torch, "accelerator") else True
        )
    except Exception:
        return False


def _device():
    return torch.device("flagos:0")


def _cuda_abi_build():
    """Whether this build dispatches flagos copies to CUDA's native copy kernel.

    The on-device arms below are compiled for the CUDA-ABI platforms (cuda, dcu,
    metax, ppu) and only those. Iluvatar is a PrivateUse1 device whose tensor
    operations this repository does not own -- its `copy_` reaches corex through
    the vendor layer and, as measured on the Iluvatar lane, gathers through host
    memory. `CUDA_ABI_PLATFORMS` is also the set the source pins below assert on,
    so the two cannot drift apart silently.
    """
    try:
        import torch_fl  # noqa: F401  -- already imported at module scope
        from torch_fl import _build_config

        accelerator = getattr(_build_config, "ACCELERATOR", None)
    except Exception:
        return False
    if accelerator is None:
        # Builds predating _build_config: the device name is the only signal.
        accelerator = (
            torch.accelerator.current_accelerator()
            if hasattr(torch, "accelerator")
            else None
        )
        accelerator = str(accelerator) if accelerator is not None else ""
    return accelerator in CUDA_ABI_PLATFORMS


_pytest_marks = [
    pytest.mark.anyplatform,
    pytest.mark.skipif(not _flagos_available(), reason="no flagos device"),
]

pytestmark = _pytest_marks

# A transfer count is only a statement about *this* repository's copy arms. On a
# build that executes copies elsewhere the count says nothing about the change
# these tests guard, and asserting zero there reports the vendor's behaviour as
# our regression.
requires_on_device_copy_arms = pytest.mark.skipif(
    not _cuda_abi_build(),
    reason="copy_ is not served by this repository's CUDA-ABI arms on this build",
)


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
    src = _copy_ops_source()
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


@requires_on_device_copy_arms
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


@requires_on_device_copy_arms
@pytest.mark.parametrize("src_dtype,dst_dtype", CASTS)
def test_copy_into_strided_destination_is_transfer_free(src_dtype, dst_dtype):
    """`copy_` into a non-contiguous destination must stay on device.

    The same-device fast path in `_copy_from` only covers a contiguous source
    into a contiguous destination of equal shape and dtype. Everything else --
    a dtype change, a strided destination, a transposed source -- used to fall
    through to `strided_copy_cast_via_cpu` on the non-boxing build, which is two
    blocking `cudaMemcpy` per call. This is the arm `_to_copy` lands on after its
    own `contiguous()`, so the cost showed up under `_to_copy` and not here.
    """
    src = torch.randn(64, 96, dtype=src_dtype, device=_device())
    dst = torch.empty(96, 64, dtype=dst_dtype, device=_device()).t()

    def run():
        dst.copy_(src)

    n = _memcpy_count(run)
    assert n == 0, (
        f"copy_({src_dtype} -> {dst_dtype}) into a strided destination performed "
        f"{n} host/device transfers; the copy is going through host memory"
    )


@requires_on_device_copy_arms
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [(torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)],
)
def test_contiguous_of_transposed_source_is_transfer_free(src_dtype, dst_dtype):
    """`contiguous()` on a strided source must gather on device.

    `_to_copy` calls it before its cast, so a host gather here costs the round
    trip twice: once to make the source contiguous, once for the cast.
    """
    src = torch.randn(64, 96, dtype=src_dtype, device=_device()).t()
    assert not src.is_contiguous()

    out = src.contiguous()

    assert out.is_contiguous()
    assert out.device.type == "flagos"
    n = _memcpy_count(lambda: src.contiguous())
    assert n == 0, (
        f"contiguous() of a transposed {src_dtype} source performed {n} "
        "host/device transfers; the gather is going through host memory"
    )


def test_contiguous_matches_cpu_after_the_on_device_gather():
    """The on-device gather must produce the same values as ATen's."""
    torch.manual_seed(0)
    base = torch.randn(48, 72, dtype=torch.float32)
    src = base.to(_device()).t()
    expected = base.t().contiguous()
    torch.cuda.synchronize()
    got = src.contiguous().cpu()
    assert torch.equal(got, expected)


@pytest.mark.parametrize("platform", CUDA_ABI_PLATFORMS)
def test_copy_arm_boxes_instead_of_going_host(platform):
    """Pin the `copy_` arm in the source, so this runs without a device.

    The regression: the non-boxing arm called `strided_copy_cast_via_cpu`
    unconditionally, on the premise that the wheel has no native CUDA copy
    kernel. It has one -- `DeviceBoxingGuard` reaches it, and the boxing
    kernels are not involved.
    """
    src = _copy_ops_source()
    # The flagos -> flagos branch of _copy_from, told apart from its
    # device <-> host siblings by the fast-path comment above it.
    marker = "// Fast path: both contiguous, same shape and dtype"
    branch = src.find(marker)
    assert branch != -1, "the same-device copy fast path is gone"
    arm = src[branch : branch + 2200]
    fast = arm.find("CrossDeviceMemcpy")
    assert fast != -1, "the contiguous fast path is gone"
    tail = arm[fast:]
    assert "DeviceBoxingGuard guard(self, dst);" in tail, (
        f"{platform}: the non-boxing copy_ arm no longer boxes to CUDA, so a "
        "strided or dtype-changing copy is a host round trip again"
    )
    assert "strided_copy_cast_via_cpu(self, dst);" in tail, (
        f"{platform}: the unsigned-dtype escape hatch is missing from the arm"
    )
