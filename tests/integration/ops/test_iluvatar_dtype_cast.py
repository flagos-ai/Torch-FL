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

"""``_to_copy``'s on-device dtype cast on Iluvatar.

``_to_copy`` is a hand-written composite, so the conf cannot route it: its
same-device dtype branch calls a per-platform ``DtypeCast`` hook
(``csrc/aten/backends/iluvatar/dtype_cast.h``). That hook takes the FlagGems
``to_copy`` kernel for the (source, target) pairs it was measured exact for and
declines the rest to the CPU round trip the branch already implements.

Both halves need pinning, and for different reasons:

* the accepted pairs, because a wrong answer here is silent -- the tensor comes
  back with the right shape, the right dtype and the wrong values;
* the declined pairs, because "declined" must mean *slower*, not *wrong*. The
  pairs below were measured wrong in the FlagGems kernel (float64 source reads
  the low 32 bits; float source to uint16/uint32/uint64; float source to bool
  truncates towards zero first), so they must still return the ATen answer
  through the fallback.
"""

import pytest
import torch
import torch_fl  # noqa: F401  (registers the flagos backend)

DEVICE = "flagos:0"

pytestmark = [pytest.mark.anyplatform]

#: Pairs the device path takes. The three float types in every direction is the
#: hot path -- HF casts bf16 <-> fp32 around every RMSNorm -- and the integer
#: conversions are what an index or mask computation asks for.
DEVICE_CAST_PAIRS = [
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
    (torch.float16, torch.float32),
    (torch.float32, torch.float16),
    (torch.bfloat16, torch.float16),
    (torch.float16, torch.bfloat16),
    (torch.int64, torch.float32),
    (torch.int32, torch.bfloat16),
    (torch.int8, torch.float16),
    (torch.uint8, torch.float32),
    (torch.bool, torch.float32),
    (torch.float32, torch.int64),
    (torch.float32, torch.int32),
    (torch.float32, torch.int8),
    (torch.float32, torch.uint8),
]

#: Pairs the hook declines: FlagGems' kernel was measured wrong for them, so
#: they must fall back to the CPU round trip and still be exact.
FALLBACK_CAST_PAIRS = [
    (torch.float64, torch.float32),
    (torch.float64, torch.int64),
    (torch.float32, torch.uint16),
    (torch.float32, torch.uint32),
    (torch.float32, torch.uint64),
    (torch.float32, torch.bool),
    (torch.bfloat16, torch.bool),
    (torch.bool, torch.int64),
]


def _make(dtype):
    if dtype is torch.bool:
        return torch.randint(0, 2, (8, 16), device=DEVICE, dtype=torch.int8).to(
            torch.bool
        )
    if dtype.is_floating_point:
        # A mix of magnitudes so a conversion that truncates towards zero (the
        # float -> bool defect) cannot pass by accident.
        return (torch.randn(8, 16, device=DEVICE, dtype=dtype) * 10).to(dtype)
    return torch.randint(0, 100, (8, 16), device=DEVICE, dtype=dtype)


@pytest.mark.parametrize("src,dst", DEVICE_CAST_PAIRS)
def test_device_cast_matches_cpu(src, dst):
    torch.manual_seed(7)
    tensor = _make(src)
    actual = tensor.to(dst)
    torch.accelerator.synchronize()
    assert actual.dtype == dst
    assert actual.device.type == tensor.device.type
    torch.testing.assert_close(actual.cpu(), tensor.cpu().to(dst))


@pytest.mark.parametrize("src,dst", FALLBACK_CAST_PAIRS)
def test_declined_cast_still_matches_cpu(src, dst):
    torch.manual_seed(8)
    tensor = _make(src)
    actual = tensor.to(dst)
    torch.accelerator.synchronize()
    assert actual.dtype == dst
    torch.testing.assert_close(actual.cpu(), tensor.cpu().to(dst))


def test_cast_of_a_strided_tensor_matches_cpu():
    """A non-contiguous source is copied first; the answer must not change."""
    torch.manual_seed(9)
    base = torch.randn(16, 8, device=DEVICE, dtype=torch.bfloat16)
    strided = base.transpose(0, 1)
    assert not strided.is_contiguous()
    torch.testing.assert_close(
        strided.to(torch.float32).cpu(), strided.cpu().to(torch.float32)
    )


def test_same_dtype_cast_returns_self():
    """``.to(dtype)`` with the same dtype is a no-op, as it is on CPU.

    The device path returns early on a same-dtype request rather than handing it
    to the kernel, and this is the case that would show it did not.
    """
    torch.manual_seed(10)
    tensor = torch.randn(4, 4, device=DEVICE, dtype=torch.bfloat16)
    same = tensor.to(torch.bfloat16)
    assert same.data_ptr() == tensor.data_ptr()


def test_empty_tensor_cast_matches_cpu():
    tensor = torch.empty(0, 4, device=DEVICE, dtype=torch.bfloat16)
    actual = tensor.to(torch.float32)
    assert actual.shape == tensor.shape
    assert actual.numel() == 0
