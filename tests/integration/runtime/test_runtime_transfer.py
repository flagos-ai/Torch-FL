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

"""Device dtype casts for platforms that serve the route."""

import pytest
import torch

import torch_fl  # noqa: F401 -- importing claims the flagos device

DEVICE = "flagos:0"


def _dtype_cast_is_served() -> bool:
    """Whether a device dtype cast reaches a kernel on this wheel.

    A cast on a device tensor dispatches to ``_to_copy``, and a conf that leaves
    that op on ``none`` hands the call to ATen's own dispatcher -- which has no
    CUDA kernel in a CPU-only torch, so it raises
    ``DispatchStub: missing kernel for cuda``. That is the conf's decision, not
    a broken copy engine: ``backends_thead.conf`` and ``backends_metax.conf``
    both leave ``_to_copy.out`` on ``none`` while ``backends_nvidia.conf``
    routes it (PPU lane, run 40188039682). Asked by making the call, on the
    same dtype the case casts to, so the probe cannot answer for a cast the
    case does not make.
    """
    try:
        out = torch.ones(1, device=DEVICE).to(torch.float16)
    except (NotImplementedError, RuntimeError):
        return False
    return out.device.type == "flagos" and out.dtype == torch.float16


@pytest.mark.anyplatform
@pytest.mark.skipif(
    not _dtype_cast_is_served(),
    reason="no route serves a device dtype cast on this wheel's conf, so the "
    "call reaches ATen's CUDA dispatch stub, which a CPU-only torch has no "
    "kernel for",
)
def test_dtype_cast_round_trip():
    torch.manual_seed(3)
    cpu = torch.randn(4, 4)
    device = cpu.to(DEVICE)
    half = device.to(torch.float16)
    assert half.dtype == torch.float16
    assert half.device.type == "flagos"
    torch.testing.assert_close(half.float().cpu(), cpu, rtol=1e-3, atol=1e-3)
