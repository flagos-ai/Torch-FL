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

"""`all` on Apple Silicon's MPSGraph backend."""

import pytest
import torch
import torch_fl  # noqa: F401

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
    host = torch.tensor([[1, 3], [4, 5]], dtype=dtype)
    actual = torch.all(host.to(DEVICE))
    assert actual.item() is torch.all(host).item()
    assert torch.all((host > 0).to(DEVICE)).item() is True


@pytest.mark.anyplatform
@pytest.mark.parametrize("dim", [0, -1, (0, 1)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_dim_and_keepdim_follow_aten(dim, keepdim):
    host = torch.tensor([[1, 3], [4, 5]], dtype=torch.float32)
    actual = torch.all(host.to(DEVICE), dim=dim, keepdim=keepdim)
    expected = torch.all(host, dim=dim, keepdim=keepdim)
    assert actual.shape == expected.shape
    assert actual.dtype == torch.bool
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.anyplatform
def test_each_spelling_uses_mps_and_empty_results_keep_their_shape():
    source = """
import torch, torch_fl
x = torch.tensor([[1, 0], [3, 4]], dtype=torch.int32, device='flagos:0')
x = x.to(torch.float32)
torch.all(x)
torch.all(x, dim=0)
torch.all(x, dim=(0, 1), keepdim=True)
torch_fl._C._synchronize()
"""
    assert dispatched_to_mps(source) == 3
    empty = torch.empty((0, 3), dtype=torch.int32)
    expected = torch.all(empty)
    actual = torch.all(empty.to(DEVICE))
    assert actual.item() is expected.item()
