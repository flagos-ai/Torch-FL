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

"""CUDA bf16/fp32 casts must avoid host transfers on the measured CUDA stack.

This is a performance contract, separate from cross-platform cast correctness.
The profiler owns the device through an exclusive CUDA-only L3 manifest group.
Other CUDA-ABI vendors need their own measurements before joining this contract.
"""

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
    src = torch.randn(512, 1024, dtype=src_dtype, device=_device())
    n = _memcpy_count(lambda: src.to(dst_dtype))
    assert n == 0, (
        f"{src_dtype} -> {dst_dtype} performed {n} host/device transfers; the cast "
        "is routing through host memory instead of the native copy kernel"
    )
