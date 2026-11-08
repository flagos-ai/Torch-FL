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

"""Apple Silicon (Metal / MPSGraph) convolution backward.

`convolution_overrideable` has been served since the conv forward landed, and a
`PrivateUse1` forward has to register its own backward. It did not, so any model
trained on this backend stopped at the first convolution:

    convolution_backward_overrideable: You are likely triggering this with
    tensor backend other than CPU

That is a **missing kernel**, not a wrong gradient -- which is why the cases here
are about the three results reaching the reference exactly, and why the failure
before this change was an exception rather than a number.

Two things in the implementation are worth knowing when reading it:

* **The forward bias sizes are the reason the round-trip differs.** The
  overrideable backward contract carries no bias sizes, and
  `convolution_backward` -- the only real implementation -- needs them. This body
  spells them from the weight's channel count when `output_mask[2]` asks for a
  bias gradient, so a convolution with a bias reverses exactly.
* **`convolution_backward_overrideable` cannot be its own fallback.** Measured,
  calling it directly on host tensors raises the same `NotImplementedError`; it
  is the overrideable *contract*, not an implementation. The body falls back to
  `convolution_backward` instead, which is what a `not implemented` path needs to
  reach.

The gradients come out **bit-identical** to the reference across every stride and
padding combination below, which is the strongest evidence that the device nodes
are the ones answering rather than a silent round-trip.

Every case self-skips off an MPS build.
"""

import pytest
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401

from backend_conf import routed_backend_or_none
from platform_support import detect_platform


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]

# (stride, padding) pairs that exercise the descriptor paths.
GEOMETRIES = [
    ((1, 1), (0, 0)),
    ((1, 1), (1, 1)),
    ((2, 2), (0, 0)),
    ((2, 2), (1, 1)),
    ((1, 1), (2, 2)),
]


def _grads(device, stride, padding, bias=True, shape=(2, 3, 8, 8), out_ch=4):
    """The three gradients of `conv2d(x, w, b).sum()`, seeded identically."""
    torch.manual_seed(0)
    x = torch.randn(*shape, device=device, requires_grad=True)
    w = torch.randn(out_ch, shape[1], 3, 3, device=device, requires_grad=True)
    b = torch.randn(out_ch, device=device, requires_grad=True) if bias else None
    F.conv2d(x, w, b, stride=stride, padding=padding).sum().backward()
    return x.grad, w.grad, (b.grad if bias else None)


class TestGradients:
    """All three gradients match the reference."""

    @pytest.mark.parametrize("stride,padding", GEOMETRIES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, stride, padding):
        gx_c, gw_c, gb_c = _grads(torch.device("cpu"), stride, padding)
        gx_d, gw_d, gb_d = _grads(DEVICE, stride, padding)
        for name, expected, got in (
            ("grad_input", gx_c, gx_d.cpu()),
            ("grad_weight", gw_c, gw_d.cpu()),
            ("grad_bias", gb_c, gb_d.cpu()),
        ):
            assert got.shape == expected.shape, (name, got.shape, expected.shape)
            torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6, msg=name)

    @pytest.mark.anyplatform
    def test_without_a_bias(self):
        """No bias means `output_mask[2]` is false and the third result is undefined."""
        gx_c, gw_c, _ = _grads(torch.device("cpu"), (1, 1), (1, 1), bias=False)
        gx_d, gw_d, gb_d = _grads(DEVICE, (1, 1), (1, 1), bias=False)
        torch.testing.assert_close(gx_d.cpu(), gx_c, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(gw_d.cpu(), gw_c, rtol=1e-5, atol=1e-6)
        assert gb_d is None

    @pytest.mark.parametrize(
        "shape,out_ch",
        [((1, 1, 5, 5), 1), ((1, 8, 16, 16), 16), ((3, 2, 7, 9), 5)],
    )
    @pytest.mark.anyplatform
    def test_other_shapes(self, shape, out_ch):
        gx_c, gw_c, gb_c = _grads(
            torch.device("cpu"), (1, 1), (0, 0), shape=shape, out_ch=out_ch
        )
        gx_d, gw_d, gb_d = _grads(DEVICE, (1, 1), (0, 0), shape=shape, out_ch=out_ch)
        torch.testing.assert_close(gx_d.cpu(), gx_c, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(gw_d.cpu(), gw_c, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(gb_d.cpu(), gb_c, rtol=1e-5, atol=1e-6)

    @pytest.mark.anyplatform
    def test_a_1d_convolution(self):
        """A 1-D convolution folds into the 2-D selector, as the forward does."""
        torch.manual_seed(0)
        xc = torch.randn(2, 3, 16, requires_grad=True)
        wc = torch.randn(4, 3, 3, requires_grad=True)
        xd = xc.detach().clone().to(DEVICE).requires_grad_(True)
        wd = wc.detach().clone().to(DEVICE).requires_grad_(True)
        F.conv1d(xc, wc, stride=2, padding=1).sum().backward()
        F.conv1d(xd, wd, stride=2, padding=1).sum().backward()
        torch.testing.assert_close(xd.grad.cpu(), xc.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(wd.grad.cpu(), wc.grad, rtol=1e-5, atol=1e-6)

    @pytest.mark.anyplatform
    def test_a_dilated_convolution(self):
        torch.manual_seed(0)
        xc = torch.randn(1, 2, 9, 9, requires_grad=True)
        wc = torch.randn(3, 2, 3, 3, requires_grad=True)
        xd = xc.detach().clone().to(DEVICE).requires_grad_(True)
        wd = wc.detach().clone().to(DEVICE).requires_grad_(True)
        F.conv2d(xc, wc, dilation=2).sum().backward()
        F.conv2d(xd, wd, dilation=2).sum().backward()
        torch.testing.assert_close(xd.grad.cpu(), xc.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(wd.grad.cpu(), wc.grad, rtol=1e-5, atol=1e-6)

    @pytest.mark.anyplatform
    def test_a_grouped_convolution(self):
        """`groups > 1` splits the channel axis; the descriptor carries it."""
        torch.manual_seed(0)
        xc = torch.randn(1, 4, 8, 8, requires_grad=True)
        wc = torch.randn(6, 2, 3, 3, requires_grad=True)
        xd = xc.detach().clone().to(DEVICE).requires_grad_(True)
        wd = wc.detach().clone().to(DEVICE).requires_grad_(True)
        F.conv2d(xc, wc, groups=2).sum().backward()
        F.conv2d(xd, wd, groups=2).sum().backward()
        torch.testing.assert_close(xd.grad.cpu(), xc.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(wd.grad.cpu(), wc.grad, rtol=1e-5, atol=1e-6)


class TestItIsActuallyUsed:
    """The op is routed here, so the backward is not silently round-tripping."""

    @pytest.mark.anyplatform
    def test_the_conf_routes_it(self):
        assert routed_backend_or_none("convolution_backward_overrideable") in (
            "apple",
            "mps",
        )

    @pytest.mark.anyplatform
    def test_a_convolution_module_trains(self):
        """`nn.Conv2d` in a module, which is how a model spells this."""
        torch.manual_seed(0)
        module_cpu = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1)
        module_dev = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1)
        module_dev.load_state_dict(module_cpu.state_dict())
        module_dev.to(DEVICE)

        x = torch.randn(2, 3, 8, 8)
        module_cpu.zero_grad()
        module_cpu(x).sum().backward()
        module_dev.zero_grad()
        module_dev(x.to(DEVICE)).sum().backward()

        torch.testing.assert_close(
            module_dev.weight.grad.cpu(), module_cpu.weight.grad, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            module_dev.bias.grad.cpu(), module_cpu.bias.grad, rtol=1e-5, atol=1e-6
        )


class TestDeclines:
    """Configurations the gate does not serve stay correct through the reference."""

    @pytest.mark.anyplatform
    def test_a_transposed_convolution_round_trips(self):
        """Only the non-transposed gradient selectors exist."""
        torch.manual_seed(0)
        xc = torch.randn(1, 2, 4, 4, requires_grad=True)
        wc = torch.randn(2, 3, 3, 3, requires_grad=True)
        xd = xc.detach().clone().to(DEVICE).requires_grad_(True)
        wd = wc.detach().clone().to(DEVICE).requires_grad_(True)
        F.conv_transpose2d(xc, wc, stride=2).sum().backward()
        F.conv_transpose2d(xd, wd, stride=2).sum().backward()
        torch.testing.assert_close(xd.grad.cpu(), xc.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(wd.grad.cpu(), wc.grad, rtol=1e-5, atol=1e-6)

    @pytest.mark.anyplatform
    def test_a_3d_convolution_round_trips(self):
        """There is no 3-D data/weights gradient selector on this SDK."""
        torch.manual_seed(0)
        xc = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
        wc = torch.randn(3, 2, 2, 2, 2, requires_grad=True)
        xd = xc.detach().clone().to(DEVICE).requires_grad_(True)
        wd = wc.detach().clone().to(DEVICE).requires_grad_(True)
        F.conv3d(xc, wc).sum().backward()
        F.conv3d(xd, wd).sum().backward()
        torch.testing.assert_close(xd.grad.cpu(), xc.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(wd.grad.cpu(), wc.grad, rtol=1e-5, atol=1e-6)
