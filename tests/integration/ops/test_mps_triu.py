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

"""``triu`` on Apple Silicon's MPSGraph backend.

Qwen3.5 builds the pairwise causal mask with ``triu(ones, 1)``. MPSGraph
coordinates are row-major, so a row coordinate used without the matching axis
expansion selects the wrong triangle. These cases pin all diagonal offsets on
square, non-square and batched operands, where an off-by-one in either
coordinate cannot pass by coincidence.
"""

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

SERVED_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.bool,
]


def _to_device(value):
    return value.clone().to(DEVICE)


def _assert_exact(metal, cpu):
    assert metal.device.type == "flagos"
    assert metal.dtype == cpu.dtype
    assert metal.shape == cpu.shape
    assert torch.equal(metal.cpu(), cpu), "result differs from the CPU reference"


@pytest.mark.anyplatform
@pytest.mark.parametrize("dtype", SERVED_DTYPES)
@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
def test_each_served_width_keeps_the_upper_triangle(dtype, diagonal):
    """The diagonal and both masking directions move together."""
    source = torch.ones((1, 4, 5), dtype=dtype)
    metal = torch.triu(_to_device(source), diagonal)
    _assert_exact(metal, torch.triu(source, diagonal))


@pytest.mark.anyplatform
@pytest.mark.parametrize("dtype", SERVED_DTYPES)
def test_values_keep_their_width_and_below_diagonal_is_zero(dtype):
    """No masked value survives, and the result is exactly zero below the line."""
    source = torch.arange(28).reshape(4, 7) + 3
    source = source.to(dtype)
    if dtype == torch.bool:
        source = source % 3 == 0
    metal = torch.triu(_to_device(source), 1)
    _assert_exact(metal, torch.triu(source, 1))


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    ("shape", "diagonal"),
    [
        ((3, 5), -3),
        ((3, 5), 3),
        ((5, 3), -3),
        ((5, 3), 3),
        ((3, 1), 1),
        ((1, 3), -1),
        ((2, 3, 5), 1),
        ((2, 3, 5), -2),
        ((0, 3), 1),
        ((2, 0, 4), -1),
    ],
)
def test_shapes_and_out_of_range_diagonals(shape, diagonal):
    """Empty operands and fully kept or fully masked matrices follow aten."""
    source = torch.rand(shape)
    metal = torch.triu(_to_device(source), diagonal)
    _assert_exact(metal, torch.triu(source, diagonal))


@pytest.mark.anyplatform
def test_a_strided_input_reads_in_its_own_order():
    source = torch.arange(35).reshape(5, 7).to(torch.float32).t()
    assert not source.is_contiguous()
    metal = torch.triu(_to_device(source), -1)
    _assert_exact(metal, torch.triu(source, -1))


@pytest.mark.anyplatform
def test_qwen35_causal_mask_uses_the_graph():
    """The checkpoint's (64, 64) bool mask reaches MPS and is exact."""
    for diagonal in (-1, 0, 1):
        served = dispatched_to_mps(
            f"""
import torch, torch_fl
mask = torch.ones(64, 64, dtype=torch.bool).cpu().clone().to('flagos:0')
out = torch.triu(mask, {diagonal})
torch_fl._C._synchronize()
"""
        )
        assert served == 1, "the triu graph did not dispatch to MPS"
    source = torch.ones((64, 64), dtype=torch.bool)
    metal = torch.triu(_to_device(source), 1)
    _assert_exact(metal, torch.triu(source, 1))


@pytest.mark.anyplatform
def test_a_uint8_input_round_trips_without_overwriting_its_source():
    source = torch.tensor([[0, 129, 255], [1, 2, 3]], dtype=torch.uint8)
    metal = torch.triu(_to_device(source), 0)
    _assert_exact(metal, torch.triu(source, 0))
    assert torch.equal(
        source, torch.tensor([[0, 129, 255], [1, 2, 3]], dtype=torch.uint8)
    )
