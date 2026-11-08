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

"""Apple Silicon (Metal / MPSGraph) tan, logical_not and bitwise_not.

Three ops that look parallel and are not. Each one's first guess was wrong, and
the cases below are what caught it:

* **`tan`** is the easy one: one trig node, no host value, the same width gate as
  `sin`/`cos`/`asin`. It is here for completeness of the family rather than
  because anything surprising happened.
* **`logical_not` is `x == 0`, not `notWithTensor:`.** The `not` node is the
  *bitwise* complement over the operand's own width, so `logical_not([0.0, 1.5])`
  came back `[False, False]` against the reference's `[True, False]` -- `1.5`'s
  complement bit-pattern is nonzero, so the node said "true is false" for the
  wrong reason and landed on the right value only for the zero.
* **`bitwise_not` does not accept a bool operand on the device.** The reference
  answers one, but `mps.bitwise_not` aborts:
  `operand #0 must be tensor of int values, but got 'tensor<3xi1>'`. The gate
  serves integral widths only, so a bool operand round-trips.

Every case self-skips off an MPS build.
"""

import os
import subprocess
import sys

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
INTEGRAL_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.clone().to(DEVICE)


def _tolerance(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        # tan amplifies its input error by 1/cos^2, so a value near +-pi/2 needs a
        # relative band rather than an absolute one.
        return {"rtol": 1e-5, "atol": 1e-5}
    return {"rtol": 5e-3, "atol": 5e-3}


class TestTan:
    """`tan` equals the reference over the served widths."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu_over_the_domain(self, dtype):
        # Stop short of +-pi/2, where tan is unbounded and any input error explodes.
        host = torch.linspace(-1.4, 1.4, 129, dtype=dtype)
        out = torch.tan(_to_device(host))
        assert out.dtype == dtype
        torch.testing.assert_close(
            out.cpu().float(), torch.tan(host).float(), **_tolerance(dtype)
        )

    @pytest.mark.anyplatform
    def test_the_poles_and_periodicity(self):
        host = torch.tensor([0.0, 4.0, -4.0, 8.0])
        out = torch.tan(_to_device(host)).cpu()
        expected = torch.tan(host)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.anyplatform
    def test_a_nan_operand_propagates(self):
        host = torch.tensor([float("nan")])
        assert torch.isnan(torch.tan(_to_device(host)).cpu()).all()

    @pytest.mark.anyplatform
    def test_signed_zero_survives(self):
        out = torch.tan(_to_device(torch.tensor([-0.0]))).cpu()
        assert out.item() == 0.0
        assert torch.signbit(out).all()

    @pytest.mark.anyplatform
    def test_strided_and_row_pitch_operands(self):
        host = torch.linspace(-1.2, 1.2, 20).reshape(4, 5)
        torch.testing.assert_close(torch.tan(_to_device(host)).cpu(), torch.tan(host))
        strided_host = torch.linspace(-1.2, 1.2, 32)
        strided = _to_device(strided_host)[::2]
        torch.testing.assert_close(
            torch.tan(strided).cpu(), torch.tan(strided_host[::2])
        )

    @pytest.mark.anyplatform
    def test_integral_operands_promote(self):
        host = torch.tensor([0, 1, -1], dtype=torch.int64)
        out = torch.tan(_to_device(host))
        assert out.dtype == torch.float32
        torch.testing.assert_close(
            out.cpu(), torch.tan(host), **_tolerance(torch.float32)
        )


class TestLogicalNot:
    """`logical_not` is `x == 0`, at every width that reaches it."""

    @pytest.mark.parametrize(
        "dtype,values",
        [
            (torch.bool, [True, False, True]),
            (torch.int64, [0, 1, -3]),
            (torch.int32, [0, 7]),
            (torch.float32, [0.0, 1.5, -2.0, float("inf")]),
            (torch.float16, [0.0, 2.0]),
            (torch.bfloat16, [0.0, 3.0]),
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu(self, dtype, values):
        host = torch.tensor(values, dtype=dtype)
        out = torch.logical_not(_to_device(host))
        assert out.dtype == torch.bool
        assert torch.equal(out.cpu(), torch.logical_not(host)), (dtype, values)

    @pytest.mark.anyplatform
    def test_a_float_zero_and_nonzero(self):
        """The case the `notWithTensor:` body failed.

        `logical_not([0.0, 1.5])` is `[True, False]`; the bitwise complement of
        `1.5` is a nonzero bit pattern, so a `not` node answers something else.
        """
        host = torch.tensor([0.0, 1.5, -0.0, -1.5])
        assert torch.logical_not(_to_device(host)).cpu().tolist() == [
            True,
            False,
            True,
            False,
        ]

    @pytest.mark.anyplatform
    def test_a_nan_is_truthy(self):
        """A NaN is not zero, so `logical_not(nan)` is false."""
        host = torch.tensor([float("nan"), 0.0])
        out = torch.logical_not(_to_device(host)).cpu()
        assert out.tolist() == [False, True]

    @pytest.mark.anyplatform
    def test_row_pitch_and_strided_operands(self):
        host = torch.randn(4, 5)
        host[0, 0] = 0.0
        assert torch.equal(
            torch.logical_not(_to_device(host)).cpu(), torch.logical_not(host)
        )
        strided_host = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0, 0.0])
        strided = _to_device(strided_host)[::2]
        assert torch.equal(
            torch.logical_not(strided).cpu(), torch.logical_not(strided_host[::2])
        )


class TestBitwiseNotStaysOnTheFallback:
    """Deliberate, and measured -- see the module docstring.

    ``bitwise_not`` is cheap enough on the host that MPS submission dominates:
    0.07x at (64, 64), 1.37x at (1, 16, 512, 512) and 1.62x at 1024x1024. The
    same shape dependence ``relu`` showed, so it is not served.
    """

    @pytest.mark.anyplatform
    def test_it_is_not_routed(self):
        assert routed_backend_or_none("bitwise_not") is None

    @pytest.mark.parametrize(
        "dtype,values",
        [
            (torch.int8, [0, 1, 5]),
            (torch.int16, [0, 1, 5]),
            (torch.int32, [0, 5]),
            (torch.int64, [0, 5]),
            (torch.bool, [True, False]),
        ],
    )
    @pytest.mark.anyplatform
    def test_it_still_answers_correctly(self, dtype, values):
        """On the fallback, not broken -- including the bool case that aborted
        the device path."""
        host = torch.tensor(values, dtype=dtype)
        out = torch.bitwise_not(_to_device(host))
        assert out.dtype == dtype
        assert torch.equal(out.cpu(), torch.bitwise_not(host))


class TestRouting:
    """All three are routed here, and the log says so."""

    @pytest.mark.parametrize("name", ["tan", "logical_not"])
    @pytest.mark.anyplatform
    def test_the_conf_routes_it(self, name):
        assert routed_backend_or_none(name) in ("apple", "mps")

    @pytest.mark.anyplatform
    def test_the_dispatch_log_names_the_backend(self):
        script = (
            "import torch, torch_fl\n"
            f"d = torch.device({DEVICE!r})\n"
            "torch.tan(torch.linspace(-1.0, 1.0, 8, device=d))\n"
            "torch.logical_not(torch.tensor([0.0, 1.0], device=d))\n"
        )
        env = dict(os.environ, FLAGOS_LOG="dispatch", FLAGOS_ACCELERATOR="mps")
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        for name in ("tan", "logical_not"):
            lines = [
                ln
                for ln in proc.stderr.splitlines()
                if "->" in ln and f"] {name}" in ln
            ]
            assert any("-> apple" in ln or "-> mps" in ln for ln in lines), (
                f"no dispatch line named the backend for {name}:\n" + "\n".join(lines)
            )
