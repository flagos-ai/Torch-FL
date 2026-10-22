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

"""The MPS graph's advanced index gather."""

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


def _source():
    return torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)


def _index(values, dtype=torch.long):
    return torch.tensor(values, dtype=dtype, device=DEVICE)


def _assert_index(selection):
    source = _source()
    expected = source[selection(source.device)]
    actual = source.to(DEVICE)[selection(DEVICE)]
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.anyplatform
def test_a_scalar_and_negative_value_select_one_source_axis():
    source = _source()
    expected = source[_index(1)]
    actual = source.to(DEVICE)[_index(-1)]
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.anyplatform
def test_adjacent_and_separated_indices_keep_aten_layout():
    def adjacent(device):
        return (
            torch.tensor([0, 1, 0], device=device),
            torch.tensor([2, 0, 1], device=device),
        )

    def separated(device):
        return (
            torch.tensor([0, 1], device=device),
            slice(None),
            torch.tensor([2, 0], device=device),
            slice(None),
        )

    _assert_index(adjacent)
    _assert_index(separated)


@pytest.mark.anyplatform
def test_broadcast_and_negative_indices_keep_their_shared_shape():
    source = _source().reshape(2, 4, 3, 5)
    first = torch.tensor([[-1], [0]])
    second = torch.tensor([[1, 2, 0]])
    expected = source[first, :, second]
    actual = source.to(DEVICE)[first.to(DEVICE), :, second.to(DEVICE)]
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.anyplatform
def test_int32_indices_are_cast_without_changing_the_source():
    source = _source()
    first = torch.tensor([0, 1], dtype=torch.int32)
    second = torch.tensor([2, 0], dtype=torch.int32)
    expected = source[first, second]
    device_source = source.to(DEVICE).clone()
    actual = device_source[first.to(DEVICE), second.to(DEVICE)]
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(device_source.cpu(), source)


@pytest.mark.anyplatform
def test_empty_indices_keep_the_result_shape():
    source = _source()
    empty = torch.tensor([], dtype=torch.long)
    actual = source.to(DEVICE)[empty.to(DEVICE)]
    assert actual.shape == source[empty].shape
    assert actual.numel() == 0


@pytest.mark.anyplatform
def test_indexing_preserves_the_masked_permutation():
    source = _source()
    first = torch.tensor([0, 1])
    second = torch.tensor([2, 0])
    actual = source.to(DEVICE)[first.to(DEVICE), :, second.to(DEVICE)]
    torch.testing.assert_close(actual.cpu(), source[first, :, second])


@pytest.mark.anyplatform
def test_repeated_getitem_shapes_reuse_the_address_metadata():
    source = _source()
    device_source = source.to(DEVICE)
    first = _index([0, 1, 0])
    second = _index([2, 0, 1])
    for _ in range(3):
        torch.testing.assert_close(
            device_source[first, second].cpu(), source[[0, 1, 0], [2, 0, 1]]
        )


@pytest.mark.anyplatform
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex64])
def test_unsupported_source_widths_keep_the_reference_result(dtype):
    source = _source().to(dtype)
    first = torch.tensor([0, 1])
    second = torch.tensor([2, 0])
    actual = source.to(DEVICE)[first.to(DEVICE), second.to(DEVICE)]
    expected = source[first, second]
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.anyplatform
def test_boolean_masks_and_out_of_range_coordinates_keep_aten_errors():
    source = _source().to(DEVICE)
    with pytest.raises(IndexError, match="out of bounds"):
        source[_index(2)]
    mask = torch.tensor([True, False])
    expected = source.cpu()[mask]
    actual = source[mask.to(DEVICE)]
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.anyplatform
def test_mps_dispatch_reaches_the_graph():
    source = """
import torch, torch_fl
x = torch.arange(120., device='flagos:0').reshape(2, 3, 4, 5)
a = torch.tensor([0, 1], device='flagos:0')
b = torch.tensor([2, 0], device='flagos:0')
out = x[:, a, b]
out.sum()
torch_fl._C._synchronize()
"""
    assert dispatched_to_mps(source) == 2


@pytest.mark.anyplatform
def test_tensor_getitem_uses_the_index_route():
    source = """
import torch, torch_fl
x = torch.arange(120., device='flagos:0').reshape(2, 4, 3, 5)
a = torch.tensor([[-1], [0]], device='flagos:0')
b = torch.tensor([[1, 2, 0]], device='flagos:0')
x[a, :, b]
torch_fl._C._synchronize()
"""
    assert dispatched_to_mps(source) == 1
