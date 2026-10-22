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

"""Metadata-only Qwen3.5 views on the Apple MPS route."""

import pytest
import torch

from platform_support import detect_platform

import torch_fl  # noqa: F401


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]


def _to_device(value):
    return value.clone().to(DEVICE)


def _assert_view(actual, expected, storage=None):
    assert actual.device.type == "flagos"
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if storage is None:
        assert actual.storage_offset() == expected.storage_offset()
        assert actual.stride() == expected.stride()
    else:
        assert (
            actual.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
        )
    assert torch.equal(actual.cpu(), expected)


def test_qwen35_attention_view_chain_keeps_the_device_storage():
    matrix_host = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
    matrix = _to_device(matrix_host.reshape(3, 20))
    _assert_view(matrix.t(), matrix_host.reshape(3, 20).t())
    _assert_view(
        torch.ops.aten._unsafe_view(matrix, [12, 5]),
        torch.ops.aten._unsafe_view(matrix_host.reshape(3, 20), [12, 5]),
    )

    host = torch.arange(120, dtype=torch.float32).reshape(3, 4, 10)
    source = _to_device(host)
    actual = source.transpose(1, 2).unsqueeze(1)[1:, 0, 1:4, 2:].squeeze(0)
    expected = host.transpose(1, 2).unsqueeze(1)[1:, 0, 1:4, 2:].squeeze(0)
    _assert_view(actual, expected)
    assert actual.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()


def test_views_preserve_offsets_and_reference_errors():
    host = torch.arange(1, 37, dtype=torch.float32).reshape(2, 3, 6)[:, :, 1:5]
    source = _to_device(host)
    for operation in (
        lambda t: t.select(0, 1),
        lambda t: t.select(-2, -1),
        lambda t: t.transpose(-2, -1),
        lambda t: t.unsqueeze(-2),
        lambda t: t.unsqueeze(-2).squeeze(-2),
    ):
        _assert_view(operation(source), operation(host), storage=source)

    with pytest.raises(IndexError):
        source.select(1, 8)


@pytest.mark.parametrize("operation", ["squeeze", "squeeze_dim", "t", "unsqueeze"])
def test_all_claimed_views_preserve_their_storage(operation):
    shape = (3, 4) if operation == "t" else (1, 3, 4)
    host = torch.arange(1, 13, dtype=torch.float32).reshape(shape)
    source = _to_device(host)
    operations = {
        "squeeze": lambda t: t.squeeze(),
        "squeeze_dim": lambda t: t.squeeze(0),
        "t": lambda t: t.t(),
        "unsqueeze": lambda t: t.unsqueeze(0),
    }
    result = operations[operation](source)
    _assert_view(result, operations[operation](host), storage=source)
