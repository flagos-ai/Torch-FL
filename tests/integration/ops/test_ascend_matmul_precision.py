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

"""Ascend must honor PyTorch's full-precision FP32 matmul policy."""

import pytest
import torch_fl  # noqa: F401
import torch


@pytest.mark.ascend
@pytest.mark.parametrize(
    "left_shape,right_shape",
    [((3, 4), (4, 2)), ((2, 3, 4), (2, 4, 2)), ((2, 1, 3, 4), (2, 5, 4, 2))],
)
def test_highest_precision_preserves_forward_and_backward(
    left_shape: tuple[int, ...], right_shape: tuple[int, ...]
) -> None:
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        generator = torch.Generator(device="cpu").manual_seed(7)
        left = torch.randn(left_shape, generator=generator, requires_grad=True)
        right = torch.randn(right_shape, generator=generator, requires_grad=True)
        device_left = left.detach().to("flagos:0").requires_grad_()
        device_right = right.detach().to("flagos:0").requires_grad_()
        expected = left @ right
        actual = device_left @ device_right
        gradient = torch.randn(expected.shape, generator=generator)
        expected.backward(gradient)
        actual.backward(gradient.to("flagos:0"))
        assert actual.device.type == "flagos"
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-6)
        torch.testing.assert_close(
            device_left.grad.cpu(), left.grad, atol=2e-5, rtol=2e-6
        )
        torch.testing.assert_close(
            device_right.grad.cpu(), right.grad, atol=2e-5, rtol=2e-6
        )
    finally:
        torch.set_float32_matmul_precision(previous_precision)


@pytest.mark.ascend
def test_real_abs_backward_stays_on_flagos() -> None:
    values = torch.tensor([-0.7, -0.1, 0.0, 0.3, 0.8])
    device_values = values.to("flagos:0").requires_grad_()
    device_values.abs().sum().backward()
    assert device_values.grad.device.type == "flagos"
    torch.testing.assert_close(device_values.grad.cpu(), values.sign(), atol=0, rtol=0)
