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

"""CUDA casts, strided copies and gathers must avoid host transfers.

This is a performance contract, separate from cross-platform cast correctness.
The profiler owns the device through an exclusive CUDA-only nightly manifest group.
Other CUDA-ABI vendors need their own measurements before joining this contract.
CPU-only PyTorch intentionally casts through the host and has no CUDA profiler.
"""

from pathlib import Path

import pytest
import torch

from torch_fl._platform import detect_platform

if detect_platform() != "cuda":
    pytest.skip("CUDA dtype-cast performance regression", allow_module_level=True)

pytestmark = pytest.mark.profiler


def _device():
    return torch.device("flagos:0")


def _memcpy_count(fn):
    """How many device/host transfers the call under `fn` performs."""
    fn()  # warm up any lazy init
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    torch.cuda.synchronize()
    return sum(e.count for e in prof.key_averages() if "Memcpy" in e.key)


@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [(torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)],
)
def test_bf16_fp32_cast_is_transfer_free(src_dtype, dst_dtype):
    """End to end: the bf16<->fp32 cast must not move data between host and device."""
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if not (torch_lib / "libtorch_cuda.so").is_file():
        pytest.skip("torch has no libtorch_cuda.so; the cast arm is the host one")
    src = torch.randn(512, 1024, dtype=src_dtype, device=_device())
    n = _memcpy_count(lambda: src.to(dst_dtype))
    assert n == 0, (
        f"{src_dtype} -> {dst_dtype} performed {n} host/device transfers; the cast "
        "is routing through host memory instead of the native copy kernel"
    )


@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.float16, torch.float32),
        (torch.int64, torch.float32),
        (torch.int32, torch.int64),
    ],
)
def test_copy_into_strided_destination_is_transfer_free(src_dtype, dst_dtype):
    """`copy_` into a non-contiguous destination must stay on device.

    The same-device fast path in `_copy_from` only covers a contiguous source
    into a contiguous destination of equal shape and dtype. Everything else --
    a dtype change, a strided destination, a transposed source -- used to fall
    through to `strided_copy_cast_via_cpu` on the non-boxing build, which is two
    blocking `cudaMemcpy` per call. This is the arm `_to_copy` lands on after its
    own `contiguous()`, so the cost showed up under `_to_copy` and not here.
    """
    if src_dtype.is_floating_point:
        src_cpu = torch.randn(64, 96, dtype=src_dtype)
    else:
        src_cpu = torch.randint(-1000, 1000, (64, 96), dtype=src_dtype)
    src = src_cpu.to(_device())
    dst = torch.empty(96, 64, dtype=dst_dtype, device=_device()).t()

    def run():
        dst.copy_(src)

    n = _memcpy_count(run)
    assert n == 0, (
        f"copy_({src_dtype} -> {dst_dtype}) into a strided destination performed "
        f"{n} host/device transfers; the copy is going through host memory"
    )


@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float32])
def test_contiguous_of_transposed_source_is_transfer_free(src_dtype):
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
