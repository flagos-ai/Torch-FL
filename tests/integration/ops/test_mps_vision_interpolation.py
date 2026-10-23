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

"""Qwen3.5 vision interpolation on Apple Silicon's MPSGraph backend.

The bilinear-weight case holds the expression from Transformers 5.10.4's
``vision_utils.get_vision_bilinear_indices_and_weights``. The CI environment
installs transformers>=4.51, whose module layout differs and whose older
releases do not expose that helper, so importing it here would test the pin
rather than the MPS route.
"""

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


@pytest.mark.anyplatform
@pytest.mark.parametrize("steps", [1, 2, 6, 32, 52])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_linspace_keeps_the_requested_device(steps, dtype):
    expected = torch.linspace(0, 47, steps, dtype=dtype)
    actual = torch.linspace(0, 47, steps, dtype=dtype, device=DEVICE)
    assert actual.device == torch.device(DEVICE)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-3, atol=1e-3)


@pytest.mark.anyplatform
@pytest.mark.parametrize("args", [(0, 32, 2), (0, 32), (32,)])
def test_arange_preserves_values_and_device(args):
    expected = torch.arange(*args)
    actual = torch.arange(*args, device=DEVICE)
    assert actual.device == torch.device(DEVICE)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.anyplatform
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_clamp_runs_on_the_tensor_device(dtype):
    source = torch.arange(-3, 8, dtype=dtype)
    expected = torch.clamp(source, min=0, max=5)
    actual = torch.clamp(source.to(DEVICE), min=0, max=5)
    assert actual.device == torch.device(DEVICE)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.anyplatform
def test_arange_stop_form_dispatches_to_mps():
    expected = torch.arange(32)
    actual = torch.arange(32, device=DEVICE)
    assert actual.device == torch.device(DEVICE)
    assert torch.equal(actual.cpu(), expected)
    assert (
        dispatched_to_mps(
            "import torch, torch_fl\n"
            "torch.arange(32, device='flagos:0')\n"
            "torch_fl._C._synchronize()\n"
        )
        == 1
    )


@pytest.mark.anyplatform
def test_qwen35_bilinear_weights_share_the_coordinate_device():
    side = 48

    def bilinear(device):
        h_grid = torch.linspace(0, side - 1, 6, device=device)
        w_grid = torch.linspace(0, side - 1, 52, device=device)
        h_floor = h_grid.int()
        w_floor = w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac = h_grid - h_floor
        w_frac = w_grid - w_floor
        indices = torch.stack(
            [
                (h_floor[:, None] * side + w_floor[None, :]).flatten(),
                (h_floor[:, None] * side + w_ceil[None, :]).flatten(),
                (h_ceil[:, None] * side + w_floor[None, :]).flatten(),
                (h_ceil[:, None] * side + w_ceil[None, :]).flatten(),
            ]
        )
        weights = torch.stack(
            [
                ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
                ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
                (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
                (h_frac[:, None] * w_frac[None, :]).flatten(),
            ]
        )
        return indices, weights

    indices, weights = bilinear(DEVICE)
    expected_indices, expected_weights = bilinear("cpu")
    assert indices.device == torch.device(DEVICE)
    assert weights.device == torch.device(DEVICE)
    assert torch.equal(indices.cpu(), expected_indices)
    torch.testing.assert_close(weights.cpu(), expected_weights, rtol=1e-6, atol=1e-6)


@pytest.mark.anyplatform
def test_vision_factories_dispatch_to_mps():
    source = """
import torch, torch_fl
torch.linspace(0, 47, 32, device='flagos:0')
torch.arange(0, 32, 2, device='flagos:0')
torch.clamp(torch.arange(32, device='flagos:0'), max=10)
torch.ops.aten.rsub.Scalar(torch.arange(32, device='flagos:0'), 1.0)
torch_fl._C._synchronize()
"""
    # `arange` runs twice: once as the clamp operand and once for rsub.
    assert dispatched_to_mps(source) == 6


@pytest.mark.anyplatform
@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gelu_uses_the_native_mps_graph(dtype, approximate):
    source = torch.linspace(-8, 8, 257, dtype=dtype)
    actual = torch.nn.functional.gelu(source.to(DEVICE), approximate=approximate)
    expected = torch.nn.functional.gelu(source, approximate=approximate)
    assert actual.device == torch.device(DEVICE)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=2e-3)
    source_stmt = f"torch.nn.functional.gelu(x, approximate={approximate!r})"
    trace = (
        "import torch, torch_fl\n"
        f"x = torch.randn(4, 4, dtype={dtype!r}, device='flagos:0')\n"
        f"{source_stmt}\n"
        "torch_fl._C._synchronize()\n"
    )
    assert dispatched_to_mps(trace) == 1
