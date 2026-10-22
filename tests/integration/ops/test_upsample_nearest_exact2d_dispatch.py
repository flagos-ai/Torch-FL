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

import pytest
import torch

import torch_fl  # noqa: F401

DEVICE = "flagos:0"


@pytest.mark.ascend
@pytest.mark.parametrize(
    ("input_size", "output_size"),
    (
        ((3, 5), (6, 10)),
        ((3, 5), (4, 7)),
        ((3, 5), (7, 2)),
    ),
)
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_upsample_nearest_exact2d_matches_cpu(input_size, output_size, dtype):
    input_cpu = torch.arange(
        2 * 3 * input_size[0] * input_size[1], dtype=dtype
    ).reshape(2, 3, *input_size)
    input_device = input_cpu.to(DEVICE)

    actual = torch.ops.aten._upsample_nearest_exact2d.default(input_device, output_size)
    expected = torch.ops.aten._upsample_nearest_exact2d.default(input_cpu, output_size)

    assert actual.device.type == "flagos"
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.ascend
def test_upsample_nearest_exact2d_noncontiguous_input():
    input_cpu = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    input_cpu = input_cpu.transpose(2, 3)
    input_device = input_cpu.to(DEVICE)

    actual = torch.ops.aten._upsample_nearest_exact2d.default(input_device, (7, 6))
    expected = torch.ops.aten._upsample_nearest_exact2d.default(input_cpu, (7, 6))

    assert actual.device.type == "flagos"
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.ascend
def test_upsample_nearest_exact2d_explicit_scales():
    input_cpu = torch.arange(15, dtype=torch.float32).reshape(1, 1, 3, 5)
    input_device = input_cpu.to(DEVICE)

    actual = torch.ops.aten._upsample_nearest_exact2d.default(
        input_device, (6, 10), 2.0, 2.0
    )
    expected = torch.ops.aten._upsample_nearest_exact2d.default(
        input_cpu, (6, 10), 2.0, 2.0
    )

    assert actual.device.type == "flagos"
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
