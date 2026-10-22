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

"""Qwen3.5 rotary trigonometry and predicate routing on MPSGraph.

The rotary embedding computes ``sin`` and ``cos`` for every position, and the
causal mask reaches the scalar and tensor comparisons below. These tests pin
the served widths, the reference's edge values and the private-use dispatch.
They also check the CPU reference boundary for unsupported widths.
"""

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import dispatched_to_mps, routed_backend_or_none
from platform_support import detect_platform


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _to_device(value):
    return value.clone().to(DEVICE)


def _positions(dtype):
    values = torch.tensor([-100.0, -3.0, -1.0, -0.0, 0.0, 1.0, 3.0, 100.0])
    return values.to(dtype).reshape(1, 2, 4)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rotary_sin_cos_follow_the_reference(dtype):
    host = _positions(dtype)
    device = _to_device(host)
    for op in (torch.sin, torch.cos):
        actual = op(device)
        expected = op(host)
        assert actual.device.type == "flagos"
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=2e-5, atol=2e-5
        )


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rotary_cos_of_sin_stays_on_device(dtype):
    host = _positions(dtype)
    device = _to_device(host)
    actual = torch.cos(torch.sin(device))
    expected = torch.cos(torch.sin(host))
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=2e-5, atol=2e-5
    )


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_causal_mask_comparisons_follow_the_reference(dtype):
    host = _positions(dtype)
    device = _to_device(host)
    predicate = torch.le(device, torch.sin(device))
    for actual, expected in (
        (torch.eq(device, 0), torch.eq(host, 0)),
        (predicate, torch.le(host, torch.sin(host))),
        (torch.ne(device, 0), torch.ne(host, 0)),
    ):
        assert actual.device.type == "flagos"
        assert actual.dtype == torch.bool
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual.cpu(), expected)


def test_causal_mask_broadcast_and_scalar_values():
    host = torch.tensor([[1.0, -0.0, 3.0], [4.0, float("-inf"), float("nan")]])
    mask = torch.tensor([[True], [False]])
    device = _to_device(host)
    device_mask = _to_device(mask)
    for actual, expected in (
        (torch.le(device, 1.5), torch.le(host, 1.5)),
        (torch.eq(device, -0.0), torch.eq(host, -0.0)),
        (torch.ne(device, float("nan")), torch.ne(host, float("nan"))),
        (torch.le(device, device_mask), torch.le(host, mask)),
    ):
        torch.testing.assert_close(actual.cpu(), expected)


def test_compare_accepts_bool_and_uint8_operands():
    for dtype in (torch.bool, torch.uint8):
        host = torch.tensor([[True, False], [False, True]], dtype=dtype)
        device = _to_device(host)
        torch.testing.assert_close(
            torch.le(device, _to_device(host.flip(0))).cpu(),
            torch.le(host, host.flip(0)),
        )
        torch.testing.assert_close(torch.eq(device, 0).cpu(), torch.eq(host, 0))
        torch.testing.assert_close(torch.ne(device, 0).cpu(), torch.ne(host, 0))


def test_integral_trigonometry_uses_the_reference_promotion():
    host = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.int32)
    device = _to_device(host)
    for op in (torch.sin, torch.cos):
        actual = op(device)
        expected = op(host)
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual.cpu(), expected)


def test_compare_uses_the_reference_for_mixed_widths():
    device = _to_device(torch.tensor([-1.0, 0.0, 1.0]))
    other = _to_device(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    actual = torch.le(device, other)
    expected = torch.le(
        torch.tensor([-1.0, 0.0, 1.0]),
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(actual.cpu(), expected)


def test_rotary_and_mask_reach_mps():
    source = """
import torch, torch_fl
x = torch.tensor([-1.0, 0.0, 1.0], device='flagos:0')
y = torch.sin(x)
torch.cos(y)
torch.eq(x, 0)
torch.le(x, y)
torch.ne(x, 0)
torch_fl._C._synchronize()
"""
    assert dispatched_to_mps(source) == 5
    for op in ("sin", "cos", "eq.Scalar", "le.Tensor", "ne.Scalar"):
        assert routed_backend_or_none(op) == "mps"


def test_fallback_to_the_cached_graph_reuses_parameters():
    host = torch.tensor([-0.5, 0.0, 0.5])
    device = _to_device(host)
    for value in (0, 1, 0):
        torch.testing.assert_close(torch.eq(device, value).cpu(), torch.eq(host, value))
    for value in (0.0, -0.0):
        torch.testing.assert_close(torch.ne(device, value).cpu(), torch.ne(host, value))


@pytest.mark.parametrize("op", ["sin.out", "cos.out", "eq.Scalar_out", "le.Tensor_out"])
def test_out_spellings_stay_on_the_reference(op):
    assert routed_backend_or_none(op) is None
