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

"""MetaX complex tensor routing and device-residency regression tests."""

import pytest
import torch_fl
import torch


pytestmark = pytest.mark.main_ops


def _require_metax() -> None:
    if not torch_fl.backend_config_path().endswith("backends_metax.conf"):
        pytest.skip("requires a MetaX wheel")
    if torch_fl.flagos.device_count() < 1:
        pytest.skip("MetaX device is unavailable")


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_complex_zero_and_fill_stay_on_flagos(dtype):
    _require_metax()
    tensor = torch.empty(8, dtype=dtype, device="flagos:0")

    returned = tensor.zero_()
    assert returned is tensor
    assert tensor.device.type == "flagos"
    torch.testing.assert_close(tensor.cpu(), torch.zeros(8, dtype=dtype))

    returned = tensor.fill_(2 + 3j)
    assert returned is tensor
    assert tensor.device.type == "flagos"
    torch.testing.assert_close(tensor.cpu(), torch.full((8,), 2 + 3j, dtype=dtype))
