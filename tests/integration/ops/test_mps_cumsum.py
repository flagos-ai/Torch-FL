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

"""Cumulative sum on Apple Silicon's MPSGraph backend."""

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import dispatched_to_mps
from platform_support import detect_platform


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps", reason="Apple Silicon (Metal) backend only"
    ),
]


def _device(value):
    return value.detach().clone().to(DEVICE)


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float32, 2e-6), (torch.float16, 2e-2), (torch.bfloat16, 1e-2)],
)
@pytest.mark.parametrize("dim", [0, 1, 2, 3, -1])
def test_cumsum_follows_the_selected_axis(dtype, tolerance, dim):
    source = torch.linspace(-2.0, 3.0, 120).reshape(2, 3, 4, 5).to(dtype)
    expected = torch.cumsum(source, dim)
    actual = torch.cumsum(_device(source), dim)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.anyplatform
@pytest.mark.parametrize("result_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cumsum_honours_a_float_accumulator_dtype(result_dtype):
    source = torch.linspace(-3.0, 2.0, 12, dtype=torch.float16).reshape(3, 4)
    expected = torch.cumsum(source, 1, dtype=result_dtype)
    actual = torch.cumsum(_device(source), 1, dtype=result_dtype)
    assert actual.dtype == result_dtype
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.anyplatform
def test_qwen35_chunk_shape_and_strided_operand():
    source = torch.linspace(-2.0, 1.0, 16 * 64).reshape(1, 1, 16, 64)
    expected = source.cumsum(3)
    actual = _device(source).cumsum(3)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-6)
    noncontiguous = source.transpose(2, 3)
    expected = noncontiguous.cumsum(3)
    actual = _device(noncontiguous).cumsum(3)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-6)


@pytest.mark.anyplatform
def test_empty_and_unsupported_widths_keep_the_reference_result():
    empty = torch.empty((3, 0), dtype=torch.float32)
    assert torch.cumsum(_device(empty), 1).numel() == 0
    source = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    torch.testing.assert_close(torch.cumsum(_device(source), 1).cpu(), source.cumsum(1))
    small = torch.tensor([1.0, -0.25, 2.0])
    torch.testing.assert_close(
        torch.cumsum(_device(small), -1).cpu(), torch.cumsum(small, -1)
    )


@pytest.mark.anyplatform
def test_cumsum_dispatch_reaches_mps():
    source = """
import torch, torch_fl
x = torch.arange(64., device='flagos:0').reshape(1, 16, 1, 4)
x.cumsum(3)
torch_fl._C._synchronize()
"""
    # `arange` creates the operand natively on MPS, then cumsum does the work.
    assert dispatched_to_mps(source) == 2
