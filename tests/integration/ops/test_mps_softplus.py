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

"""Softplus on Apple Silicon's MPSGraph backend."""

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import dispatched_to_mps
import platform_support


DEVICE = "flagos:0"
pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        platform_support.detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float32, 2e-6), (torch.float16, 2e-2), (torch.bfloat16, 1e-2)],
)
@pytest.mark.parametrize(
    ("beta", "threshold"),
    [(1.0, 20.0), (0.5, 3.0), (0.5, 20.0)],
)
def test_beta_threshold_and_both_branches_match_cpu(dtype, tolerance, beta, threshold):
    host = torch.tensor(
        [-25.0, -3.0, -0.5, 0.0, 0.5, 2.0, 10.0, 25.0], dtype=dtype
    ).double()
    actual = _device_softplus(host, beta, threshold)
    expected = _host_softplus(host, beta, threshold)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.anyplatform
def test_a_different_beta_and_threshold_builds_a_different_plan():
    host = torch.tensor([-3.0, 0.0, 25.0]).double()
    first = _device_softplus(host, 1.0, 20.0)
    second = _device_softplus(host, 0.5, 3.0)
    default_reference = _host_softplus(host, 1.0, 20.0)
    configured_reference = _host_softplus(host, 0.5, 3.0)
    assert not torch.equal(default_reference, configured_reference)
    torch.testing.assert_close(first.cpu(), default_reference, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(second.cpu(), configured_reference, rtol=2e-6, atol=2e-6)


def _host_softplus(host, beta=1.0, threshold=20.0):
    """Evaluate ATen's CPU kernel without entering torch_fl's CUDA alias mode."""
    return torch.ops.aten.softplus.default(host, beta, threshold)


def _device_softplus(host, beta=1.0, threshold=20.0):
    source = host.clone().to(DEVICE)
    if beta >= 0:
        return torch.ops.aten.softplus.default(source, beta, threshold)
    return torch.ops.aten.softplus.out(
        source, beta, threshold, out=torch.empty_like(source)
    )


@pytest.mark.anyplatform
def test_qwen35_decay_shape_and_strided_operand():
    host = torch.linspace(-40.0, 4.0, 16 * 64).reshape(1, 1, 16, 64).double()
    expected = _host_softplus(host)
    actual = _device_softplus(host)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-6)
    viewed = host.transpose(2, 3)
    expected = _host_softplus(viewed)
    actual = _device_softplus(viewed)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-6)


@pytest.mark.anyplatform
def test_empty_and_unsupported_widths_keep_the_reference_result():
    empty = torch.empty((3, 0), dtype=torch.float32)
    assert _device_softplus(empty).numel() == 0
    source = torch.tensor([[-2, 0, 3]], dtype=torch.int64)
    for beta, threshold in ((1.0, 20.0), (0.5, 3.0)):
        with pytest.raises(NotImplementedError, match="softplus_cpu"):
            _device_softplus(source, beta, threshold)
    double = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
    torch.testing.assert_close(
        _device_softplus(double).cpu(),
        _host_softplus(double),
    )
    for beta in (0.0, -1.0):
        torch.testing.assert_close(
            _device_softplus(double, beta, -1.0).cpu(),
            _host_softplus(double, beta, -1.0),
        )


@pytest.mark.anyplatform
def test_softplus_dispatch_reaches_mps():
    source = """
import torch, torch_fl
x = torch.tensor([[-3., 0., 2., 25.]], device='flagos:0')
torch.nn.functional.softplus(x)
torch_fl._C._synchronize()
"""
    assert dispatched_to_mps(source) == 1
