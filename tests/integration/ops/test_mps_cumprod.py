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

"""Apple Silicon (Metal / MPSGraph) cumprod.

`cumsum` has been served since the reduction family landed, through
`cumulativeSumWithTensor:axis:exclusive:reverse:name:`. The product had no route
even though the sibling node exists:

    cumulativeProductWithTensor:axis:name:

That node takes **no** `exclusive:`/`reverse:` pair -- measured, the sum node has
one and the product node does not -- which is why the body is a separate template
rather than the sum's with a parameter.

The cases below concentrate on what a running product can get wrong that a
running sum cannot:

* **Saturation.** A product of values above one grows without bound and one below
  one decays to zero, so a long scan reaches `inf` and `0.0` where a sum would
  stay finite. Both are the reference's answers and both are asserted.
* **A zero anywhere in the scan** pins every later entry to zero, including after
  a sign change.
* **The sign**, which a product tracks and a sum does not: a single negative among
  positives makes every later entry negative.

Measured on an M5, warmed up, against the round-trip:

    (1024, 1024) dim=0       4.5x
    (1024, 1024) dim=1       1.3x
    (256, 4096)  dim=0      17.0x
    (1,16,512,512) dim=1     2.9x

so this is a positive return at every shape measured, unlike the cheap elementwise
ops earlier in this series.

Every case self-skips off an MPS build.
"""

import pytest
import torch
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

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.clone().to(DEVICE)


def _tolerance(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"rtol": 1e-5, "atol": 1e-6}
    return {"rtol": 5e-3, "atol": 5e-3}


class TestCorrectness:
    """Every scan matches the CPU reference."""

    @pytest.mark.parametrize(
        "shape,dim",
        [
            ((4, 5), 0),
            ((4, 5), 1),
            ((4, 5), -1),
            ((2, 3, 4), 0),
            ((2, 3, 4), 2),
            ((2, 3, 4), -2),
            ((63,), 0),
            ((3, 1, 7, 2), 2),
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu(self, shape, dim):
        # Values near one keep a 63-long product inside float32.
        host = torch.rand(shape) * 0.5 + 0.99
        out = torch.cumprod(_to_device(host), dim)
        assert out.shape == host.shape
        assert out.device.type == "flagos"
        torch.testing.assert_close(
            out.cpu(), torch.cumprod(host, dim), **_tolerance(host.dtype)
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_every_served_width(self, dtype):
        host = (torch.rand(32) * 0.01 + 0.999).to(dtype)
        out = torch.cumprod(_to_device(host), 0)
        assert out.dtype == dtype
        torch.testing.assert_close(
            out.cpu().float(), torch.cumprod(host, 0).float(), **_tolerance(dtype)
        )

    @pytest.mark.anyplatform
    def test_a_zero_pins_everything_after_it(self):
        host = torch.tensor([2.0, 0.5, 0.0, 3.0, -2.0])
        out = torch.cumprod(_to_device(host), 0).cpu()
        expected = torch.cumprod(host, 0)
        torch.testing.assert_close(out, expected)
        assert (out[2:] == 0).all(), out

    @pytest.mark.anyplatform
    def test_the_sign_follows_a_negative(self):
        """A sum would not carry this; a product does."""
        host = torch.tensor([2.0, -0.5, 0.5, 2.0])
        out = torch.cumprod(_to_device(host), 0).cpu()
        expected = torch.cumprod(host, 0)
        torch.testing.assert_close(out, expected)
        assert (out[1:] < 0).all(), out

    @pytest.mark.anyplatform
    def test_saturation_is_the_references_answer(self):
        """A long scan past one overflows to inf and one below it decays to zero."""
        # 2.0^128 is past float32's 3.4e38 and 0.5^128 is past its smallest
        # subnormal, so both ends of the scan land where the reference lands.
        big = torch.full((128,), 2.0)
        small = torch.full((128,), 0.5)
        out_big = torch.cumprod(_to_device(big), 0).cpu()
        out_small = torch.cumprod(_to_device(small), 0).cpu()
        assert torch.isinf(out_big[-1]).item()
        assert out_small[-1].item() == 0.0
        torch.testing.assert_close(out_big, torch.cumprod(big, 0), equal_nan=True)
        torch.testing.assert_close(out_small, torch.cumprod(small, 0))

    @pytest.mark.anyplatform
    def test_a_nan_operand_propagates(self):
        host = torch.tensor([1.0, float("nan"), 2.0])
        out = torch.cumprod(_to_device(host), 0).cpu()
        assert torch.isnan(out[1:]).all(), out

    @pytest.mark.anyplatform
    def test_row_pitch_and_strided_operands(self):
        base = _to_device(torch.rand(8, 5) * 0.1 + 0.95)
        strided = base[::2]
        assert not strided.is_contiguous()
        expected = torch.cumprod(base.cpu()[::2], 1)
        torch.testing.assert_close(
            torch.cumprod(strided, 1).cpu(), expected, **_tolerance(torch.float32)
        )

    @pytest.mark.anyplatform
    def test_a_dtype_argument_converts_away_from_the_operand(self):
        """`cumprod(x, dtype=float32)` on a half operand comes back float32."""
        host = (torch.rand(16) * 0.01 + 0.999).to(torch.float16)
        out = torch.cumprod(_to_device(host), 0, dtype=torch.float32)
        assert out.dtype == torch.float32
        expected = torch.cumprod(host, 0, dtype=torch.float32)
        torch.testing.assert_close(out.cpu(), expected, **_tolerance(torch.float32))

    @pytest.mark.anyplatform
    def test_an_empty_operand(self):
        out = torch.cumprod(_to_device(torch.empty(0)), 0)
        assert out.numel() == 0


class TestDeclines:
    """The widths the gate does not serve stay correct through the reference."""

    @pytest.mark.anyplatform
    def test_float64_round_trips(self):
        host = torch.tensor([1.5, 0.5, 3.0], dtype=torch.float64)
        out = torch.cumprod(_to_device(host), 0)
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.cpu(), torch.cumprod(host, 0))

    @pytest.mark.anyplatform
    def test_an_integral_operand_stays_integral(self):
        host = torch.tensor([2, 3, 4], dtype=torch.int64)
        out = torch.cumprod(_to_device(host), 0)
        expected = torch.cumprod(host, 0)
        assert out.dtype == expected.dtype
        torch.testing.assert_close(out.cpu(), expected)


class TestRouting:
    """`cumprod` is routed here, so the cases above are not read as round-trips."""

    @pytest.mark.anyplatform
    def test_the_conf_routes_it(self):
        assert routed_backend_or_none("cumprod") in ("apple", "mps")

    @pytest.mark.anyplatform
    def test_cumsum_still_routes_after_the_split(self):
        """The sum keeps its own body, and this change must not disturb it."""
        assert routed_backend_or_none("cumsum") in ("apple", "mps")
        host = torch.rand(4, 5) * 2.0
        torch.testing.assert_close(
            torch.cumsum(_to_device(host), 1).cpu(),
            torch.cumsum(host, 1),
            **_tolerance(torch.float32),
        )
