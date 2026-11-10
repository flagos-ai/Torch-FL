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

"""CPU fallback for a TileOPs shim that cannot serve the current call."""

import pytest
import torch

import torch_fl  # noqa: F401
from torch_fl.tileops import runtime

pytestmark = pytest.mark.anyplatform


def test_declined_shim_runs_on_cpu_and_restores_flagos_device():
    """The fallback must not redispatch into the same PrivateUse1 wrapper."""
    x_cpu = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    y_cpu = torch.arange(4, dtype=torch.float32)
    x = x_cpu.to("flagos")
    y = y_cpu.to("flagos")

    result = runtime._make_fallback("add.Tensor")(x, y, alpha=2)

    assert result.device.type == "flagos"
    torch.testing.assert_close(
        result.cpu(), torch.ops.aten.add.Tensor(x_cpu, y_cpu, alpha=2)
    )
