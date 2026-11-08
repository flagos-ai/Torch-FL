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

"""Apple Silicon (Metal / MPSGraph) clamp, with the integral abort this found.

``clamp`` was served before this file existed and its float cases all passed, so
nothing here would have been written from reading the conf. It came out of
running a real vision tower: a Qwen3.5-2B checkpoint whose vision side clamps an
**integer** grid, which did not return a wrong number -- it aborted the process
inside MPSGraph:

    mps.maximum op requires the same element type for all operands and results
    see current operation: %6 = "mps.maximum"(%2, %5)
        : (tensor<2x2xsi64>, tensor<4xsi64>) -> tensor<*xf32>
    MPSGraphExecutable.mm: failed assertion `original module failed verification'

Three things in that message are the whole bug:

* The two operands are **2x2** and **4**. The bound was broadcast to the
  operand's *element count* rather than its *shape*, so the pair was not
  conformable and `mps.maximum` had nothing to infer a result from.
* The element types are both `si64`, so the dtype gate was right to let the call
  through -- this was never a dtype problem.
* `mps.maximum` requires shape-conformable operands with one element type; the
  float path only ever worked because a 1-D `[numel]` broadcast against a flat
  placeholder happened to line up for the shapes it was tested at.

The fix broadcasts each bound to `args[0].shape`. The cases below are what holds
it: every integral width (the abort), the float widths that already worked (a
regression guard), and the shapes where a flat broadcast and a shaped broadcast
differ -- the 2-D and 3-D cases, and the rank-0 operand where `numel() == 1`.

Every case self-skips off an MPS build.
"""

import pytest
import torch
import torch_fl  # noqa: F401

from platform_support import detect_platform


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]

INTEGRAL_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# A shape set in which a `[numel]` broadcast and a shaped broadcast disagree, or
# in which the two coincide and the case is a regression guard.
SHAPES = [(2, 2), (3, 4, 5), (6,), (4, 1), (1, 4), ()]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.clone().to(DEVICE)


class TestIntegralClamp:
    """The case that aborted: an integral operand with both bounds."""

    @pytest.mark.parametrize("dtype", INTEGRAL_DTYPES)
    @pytest.mark.anyplatform
    def test_both_bounds(self, dtype):
        """`clamp(int_tensor, 0, 5)` used to abort the process, not raise."""
        host = torch.tensor([[1, 9], [3, 4]], dtype=dtype)
        out = torch.clamp(_to_device(host), 0, 5)
        assert out.dtype == dtype
        torch.testing.assert_close(out.cpu(), torch.clamp(host, 0, 5))

    @pytest.mark.parametrize("dtype", INTEGRAL_DTYPES)
    @pytest.mark.anyplatform
    def test_one_sided(self, dtype):
        """`clamp_min`/`clamp_max` build a different node and already worked.

        They are here so that a fix to the two-bound path cannot silently move
        the one-sided path.
        """
        host = torch.tensor([[1, 9], [3, 4]], dtype=dtype)
        device_host = _to_device(host)
        torch.testing.assert_close(
            torch.clamp_min(device_host, 5).cpu(), torch.clamp_min(host, 5)
        )
        torch.testing.assert_close(
            torch.clamp_max(device_host, 3).cpu(), torch.clamp_max(host, 3)
        )

    @pytest.mark.parametrize("dtype", INTEGRAL_DTYPES)
    @pytest.mark.anyplatform
    def test_negative_bounds(self, dtype):
        """A signed operand with a negative pair, where the wrong bound is visible."""
        host = torch.tensor([[-9, -5, 0, 5, 9]], dtype=dtype)
        out = torch.clamp(_to_device(host), -5, 5).cpu()
        torch.testing.assert_close(out, torch.clamp(host, -5, 5))

    @pytest.mark.anyplatform
    def test_a_grid_shaped_operand(self):
        """The shape the real vision tower clamped, reduced to its essentials.

        A `(n, 2)` integer grid against scalar bounds: the first shape in this
        file's collection where broadcasting to `numel()` is not merely
        redundant but non-conformable.
        """
        host = torch.tensor([[1, 0], [0, 1], [2, 0]], dtype=torch.int64)
        out = torch.clamp(_to_device(host), 0, 1).cpu()
        assert out.dtype == torch.int64
        torch.testing.assert_close(out, torch.clamp(host, 0, 1))


class TestFloatClampRegression:
    """The widths that already worked, so a fix to the integral path cannot break them."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.anyplatform
    def test_both_bounds(self, dtype, shape):
        host = (
            torch.linspace(
                -5.0, 5.0, int(torch.tensor(shape).prod()), dtype=dtype
            ).reshape(shape)
            if shape
            else torch.tensor(4.0, dtype=dtype)
        )
        out = torch.clamp(_to_device(host), -2.0, 2.0)
        assert out.dtype == dtype
        assert out.shape == host.shape
        torch.testing.assert_close(
            out.cpu().float(), torch.clamp(host, -2.0, 2.0).float()
        )

    @pytest.mark.anyplatform
    def test_nan_stays_nan(self):
        host = torch.tensor([float("nan"), -3.0, 3.0])
        out = torch.clamp(_to_device(host), -1.0, 1.0).cpu()
        expected = torch.clamp(host, -1.0, 1.0)
        assert torch.isnan(out).equal(torch.isnan(expected))
        torch.testing.assert_close(out.nan_to_num(), expected.nan_to_num())

    @pytest.mark.anyplatform
    def test_a_non_contiguous_operand(self):
        host = torch.linspace(-8.0, 8.0, 32).reshape(4, 8)
        strided = _to_device(host)[:, ::2]
        assert not strided.is_contiguous()
        torch.testing.assert_close(
            torch.clamp(strided, -2.0, 2.0).cpu(), torch.clamp(host[:, ::2], -2.0, 2.0)
        )


class TestTheBoundsAreInThePlanKey:
    """The defect the shape cases accidentally found.

    ``T_CLAMP`` passed its bounds to ``MpsConstant`` but the graph key was the
    literal ``"clamp"``, so a second call with different bounds reused the first
    call's compiled graph. Measured, clamping ``[[1,9],[2,3]]`` to ``(-2, 2)``
    and then to ``(0, 5)`` answered ``[[1,2],[2,2]]`` both times.

    A single-bound test cannot see this -- it only shows when two different
    bounds are asked for *in the same process*, which is what the interleaving
    below reproduces.
    """

    @pytest.mark.anyplatform
    def test_interleaved_bounds_do_not_alias(self):
        host = torch.tensor([[1.0, 9.0], [2.0, 3.0]])
        device_host = _to_device(host)
        for low, high in [(-2.0, 2.0), (0.0, 5.0), (-2.0, 2.0), (1.0, 1.0), (0.0, 5.0)]:
            torch.testing.assert_close(
                torch.clamp(device_host, low, high).cpu(),
                torch.clamp(host, low, high),
                msg=f"bounds=({low}, {high})",
            )

    @pytest.mark.anyplatform
    def test_an_absent_bound_differs_from_a_present_one(self):
        """`clamp(x, 0, none)` and `clamp(x, 0, 5)` are different graphs.

        The key folds presence and value separately, so a call that supplies
        only the lower bound cannot be answered by one that supplied both.
        """
        host = torch.tensor([[1.0, 9.0], [2.0, 3.0]])
        device_host = _to_device(host)
        torch.testing.assert_close(
            torch.clamp_min(device_host, 5.0).cpu(), torch.clamp_min(host, 5.0)
        )
        torch.testing.assert_close(
            torch.clamp(device_host, 0.0, 5.0).cpu(), torch.clamp(host, 0.0, 5.0)
        )
        torch.testing.assert_close(
            torch.clamp_max(device_host, 2.0).cpu(), torch.clamp_max(host, 2.0)
        )


class TestTheBoundaryTheBugCrossed:
    """A shape whose element count equals its first dimension, spelled out.

    With a `(2, 2)` operand, `numel()` is 4 and the flat placeholder is 4 wide,
    so a `[4]` bound *almost* lines up -- which is why the defect read as a dtype
    problem in the abort message rather than a shape one. The tightening below is
    what distinguishes the two: a `(2, 2)` operand clipped differently per row
    catches a bound applied along the wrong axis.
    """

    @pytest.mark.anyplatform
    def test_the_bound_is_not_applied_along_an_axis(self):
        host = torch.tensor([[1.0, 9.0], [2.0, 3.0]])
        out = torch.clamp(_to_device(host), 0.0, 5.0).cpu()
        torch.testing.assert_close(out, torch.clamp(host, 0.0, 5.0))
        assert out[0, 1].item() == 5.0
