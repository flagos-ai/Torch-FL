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

"""Cross-platform dtype-cast correctness and result-device checks.

Source contracts run in L1; CUDA transfer profiling runs in its own L3 group.
"""

import pytest
import torch

import torch_fl  # noqa: F401  -- registers the device


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
