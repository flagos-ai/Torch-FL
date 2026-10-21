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

"""`argmax` on Apple Silicon's MPSGraph backend."""

import pytest
from device_init import torch

from backend_conf import dispatched_to_mps
from platform_support import detect_platform


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]


@pytest.mark.anyplatform
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_float_widths_follow_aten(dtype):
    host = torch.tensor([[1, 3, 3], [3, 1, 1]], dtype=dtype)
    actual = torch.argmax(host.to(DEVICE))
    expected = torch.argmax(host)
    assert actual.dtype == torch.int64
    assert actual.item() == expected.item()


@pytest.mark.anyplatform
def test_dim_stays_on_the_unvalidated_reference():
    # The GPU reduction currently diverges from aten when a per-row maximum
    # contains a NaN or duplicate finite maxima. Leave this overload on the
    # checked reference until the graph follows its first/missing-NaN rules.
    host = torch.tensor([[1.0, 3.0, 3.0], [3.0, 1.0, 1.0]])
    actual = torch.argmax(host.to(DEVICE), dim=1)
    expected = torch.argmax(host, dim=1)
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.anyplatform
def test_each_float_width_dispatches_to_mps():
    source = """
import torch, torch_fl
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    x = torch.randn(2, 5, device='flagos:0', dtype=dtype)
    torch.argmax(x)
torch_fl._C._synchronize()
"""
    assert dispatched_to_mps(source) == 3
