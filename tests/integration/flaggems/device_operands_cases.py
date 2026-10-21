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

"""A CPU-resident operand must produce the same result on every accelerator.

ATen permits a CPU-resident weight or bias with a device input. A generated
backend kernel must move that operand or decline the call; handing its host
pointer to a vendor API can corrupt the process silently. These cases assert the
value and its device, and every platform executes them through its configured
default route.
"""

from device_init import torch


DEVICE = torch.device("flagos:0")


class TestConvMixedDeviceOperands:
    """The generated convolution device gates on every accelerator."""

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

    def test_conv2d_cpu_bias(self):
        """The second gate: only the bias remains on the CPU."""
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


class TestLayerNormMixedDeviceOperands:
    """The generated `native_layer_norm` device gates on every accelerator.

    The weight/bias are moved onto `input`'s device rather than declining the
    call, so the result must be identical to the CPU reference and on-device.
    """

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

    def test_layer_norm_cpu_bias_only(self):
        """The second gate: only the bias remains on the CPU."""
        x_cpu = torch.randn(4, 8, 16)
        w_cpu = torch.randn(16)
        b_cpu = torch.randn(16)
        out, _, _ = torch.native_layer_norm(
            x_cpu.to(DEVICE), [16], w_cpu.to(DEVICE), b_cpu, 1e-5
        )
        assert out.device.type == "flagos"
        expected, _, _ = torch.native_layer_norm(x_cpu, [16], w_cpu, b_cpu, 1e-5)
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-3, atol=1e-3)
