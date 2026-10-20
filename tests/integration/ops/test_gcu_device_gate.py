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

"""A CPU-resident operand must not reach topsaten as device memory (issue #526).

`csrc/aten/backends/gcu/generated/gcu_kernels.cc` wraps operand `data_ptr()`s
into `gcu::TopsatenTensorWrapper`s. ATen's `_convolution` Overrideable arm is the
only conv backend that does not call `check_input_same_type_as_parameters`, so a
CPU `weight`/`bias` reaches `ConvolutionOverrideableKernelGcu` unmoved; the
generated `native_layer_norm` had the same shape, where `weight.value()` was
`.contiguous()`'d (device-preserving) but never `.to(input.device())`.

Both gates tested dtype and shape but never device, so before the fix the host
pointer was handed to topsaten as a device address -- an illegal access that
poisons the accelerator context for every later op, not a clean error.

These cases assert the *result* (correct value, returned on the device), not
merely that no error is raised: a CPU pointer reaching the vendor op does not
raise, it silently corrupts. They run on the default route, so no
`FLAGOS_FORCE_BACKEND` is needed.
"""

import pytest
import torch

import torch_fl  # noqa: F401  -- registers the flagos backends

DEVICE = torch.device("flagos:0")


@pytest.fixture(autouse=True)
def _gcu_only():
    from torch_fl._build_config import ACCELERATOR

    if ACCELERATOR != "gcu":
        pytest.skip("GCU build required")


class TestGcuConvMixedDeviceOperands:
    """The generated `ConvolutionOverrideableKernelGcu` device gates."""

    @pytest.mark.gcu
    def test_conv2d_cpu_weight(self):
        x_cpu = torch.randn(2, 3, 8, 8)
        w_cpu = torch.randn(4, 3, 3, 3)
        out = torch.nn.functional.conv2d(x_cpu.to(DEVICE), w_cpu)
        assert out.device.type == "flagos"
        torch.testing.assert_close(
            out.cpu(),
            torch.nn.functional.conv2d(x_cpu, w_cpu),
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.gcu
    def test_conv2d_cpu_bias(self):
        """The second gate: input/weight on the device pass the dtype test, so
        only the bias's device term keeps the host pointer out of topsaten."""
        x_cpu = torch.randn(2, 3, 8, 8)
        w_cpu = torch.randn(4, 3, 3, 3)
        b_cpu = torch.randn(4)
        out = torch.nn.functional.conv2d(x_cpu.to(DEVICE), w_cpu.to(DEVICE), b_cpu)
        assert out.device.type == "flagos"
        torch.testing.assert_close(
            out.cpu(),
            torch.nn.functional.conv2d(x_cpu, w_cpu, b_cpu),
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.gcu
    def test_conv1d_cpu_weight(self):
        """A 1-D conv takes the same gate with a different spatial rank."""
        x_cpu = torch.randn(2, 3, 16)
        w_cpu = torch.randn(4, 3, 3)
        out = torch.nn.functional.conv1d(x_cpu.to(DEVICE), w_cpu)
        assert out.device.type == "flagos"
        torch.testing.assert_close(
            out.cpu(),
            torch.nn.functional.conv1d(x_cpu, w_cpu),
            rtol=1e-3,
            atol=1e-3,
        )


class TestGcuLayerNormMixedDeviceOperands:
    """The generated `native_layer_norm` device gates.

    The weight/bias are moved onto `input`'s device rather than declining the
    call, so the result must be identical to the CPU reference and on-device.
    """

    @pytest.mark.gcu
    def test_layer_norm_cpu_weight_and_bias(self):
        x_cpu = torch.randn(4, 8, 16)
        w_cpu = torch.randn(16)
        b_cpu = torch.randn(16)
        out, mean, rstd = torch.native_layer_norm(
            x_cpu.to(DEVICE), [16], w_cpu, b_cpu, 1e-5
        )
        assert out.device.type == "flagos"
        expected, _, _ = torch.native_layer_norm(x_cpu, [16], w_cpu, b_cpu, 1e-5)
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.gcu
    def test_layer_norm_cpu_bias_only(self):
        """Weight on the device, bias on the CPU: the bias term alone must keep
        the call off the vendor path."""
        x_cpu = torch.randn(4, 8, 16)
        w_cpu = torch.randn(16)
        b_cpu = torch.randn(16)
        out, _, _ = torch.native_layer_norm(
            x_cpu.to(DEVICE), [16], w_cpu.to(DEVICE), b_cpu, 1e-5
        )
        assert out.device.type == "flagos"
        expected, _, _ = torch.native_layer_norm(x_cpu, [16], w_cpu, b_cpu, 1e-5)
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-3, atol=1e-3)
