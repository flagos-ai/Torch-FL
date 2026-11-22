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

"""Copy flagos tensors and compare them with the same CPU inputs."""

import pytest
import torch

import torch_fl  # noqa: F401 -- importing claims the flagos device

DEVICE = "flagos:0"


def _cpu_tensor(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(8, 8)


@pytest.mark.anyplatform
def test_host_to_device_round_trip():
    cpu = _cpu_tensor(0)
    device = cpu.to(DEVICE)
    assert device.device.type == "flagos"
    torch.testing.assert_close(device.cpu(), cpu)


@pytest.mark.anyplatform
def test_device_to_device_copy():
    cpu = _cpu_tensor(1)
    device = cpu.to(DEVICE)
    copied = device.clone()
    assert copied.device.type == "flagos"
    torch.testing.assert_close(copied.cpu(), cpu)


@pytest.mark.anyplatform
def test_non_contiguous_source_round_trip():
    cpu = _cpu_tensor(2)
    transposed_cpu = cpu.t()
    assert not transposed_cpu.is_contiguous()
    device = transposed_cpu.to(DEVICE)
    torch.testing.assert_close(device.cpu(), transposed_cpu)
