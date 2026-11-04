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

"""Apple Silicon (Metal / MPSGraph) leaky_relu and elu.

Two activations, one selector each, and one op that was measured and then
deliberately **not** served:

    leaky_relu(x, negative_slope)   -> leakyReLUWithTensor:alpha:
    elu(x, alpha, scale, input_scale) -> eluWithTensor:alpha:  (scaled around it)
    relu                            -> stays on cpu_fallback, see below

`relu` is the interesting omission. It looks like the easiest of the three and
is the one a transformer hits most, but it measured *slower* than the round-trip
at every shape a model produces:

    (64, 64)            0.09x
    (1, 16, 512, 512)   0.51x
    1024 x 1024         1.63x

and the cause is not the NaN guard it would need -- the already-served,
single-node `clamp_min` measures 0.47x at the same large shape. A ReLU is cheap
enough on the host that MPS's per-call submission cost dominates until the
operand is very large, so serving it would trade a cheap host kernel for a GPU
dispatch. `test_relu_stays_on_the_fallback` pins that decision so it is a
deliberate omission rather than a gap someone fills in later without measuring.

The signed zero is a note rather than a case here. `relu(-0.0)` answers `-0.0`
for a one- or two-element tensor and `+0.0` from eight elements up, because the
CPU kernel switches to its vectorised path; pinning either in a kernel would pin
a torch implementation detail, and the values compare equal.

Every case self-skips off an MPS build.
"""

import os
import subprocess
import sys

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

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# negative_slope values, including the default and the two degenerate ends.
SLOPES = [0.0, 0.01, 0.1, 0.5, 1.0]

# elu's alpha values.
ALPHAS = [0.5, 1.0, 2.0]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.clone().to(DEVICE)


def _tolerance(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"rtol": 1e-6, "atol": 1e-6}
    return {"rtol": 5e-3, "atol": 5e-3}


class TestLeakyRelu:
    """`leaky_relu` equals the reference at every slope."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu_over_the_domain(self, dtype):
        host = torch.linspace(-6.0, 6.0, 129, dtype=dtype)
        for slope in SLOPES:
            out = F.leaky_relu(_to_device(host), slope)
            expected = F.leaky_relu(host, slope)
            assert out.dtype == dtype
            torch.testing.assert_close(
                out.cpu().float(), expected.float(), **_tolerance(dtype)
            )

    @pytest.mark.anyplatform
    def test_the_default_slope(self):
        host = torch.linspace(-4.0, 4.0, 65)
        torch.testing.assert_close(
            F.leaky_relu(_to_device(host)).cpu(), F.leaky_relu(host)
        )

    @pytest.mark.anyplatform
    def test_the_slope_is_part_of_the_plan_key(self):
        """Two slopes over one operand describe two graphs.

        The slope is a `float` argument of the body rather than a tensor, so it
        has to be in the key -- otherwise the second call is answered by the
        first call's compiled graph and silently uses the wrong slope.
        """
        host = torch.tensor([-4.0, -2.0, -1.0, 1.0, 2.0, 4.0])
        device_host = _to_device(host)
        for slope in (0.0, 0.1, 0.5, 0.1, 0.0):
            torch.testing.assert_close(
                F.leaky_relu(device_host, slope).cpu(),
                F.leaky_relu(host, slope),
                msg=f"slope={slope}",
            )

    @pytest.mark.anyplatform
    def test_a_zero_slope_matches_a_relu_up_to_the_signed_zero(self):
        """`leaky_relu(x, 0)` is `relu(x)` on the values.

        Only the values: the sign of a zero operand differs between torch's
        scalar and vectorised relu paths, so the sign bit is not asserted here.
        """
        host = torch.linspace(-8.0, 8.0, 101)
        torch.testing.assert_close(
            F.leaky_relu(_to_device(host), 0.0).cpu(), torch.relu(host)
        )

    @pytest.mark.anyplatform
    def test_nan_and_infinities(self):
        host = torch.tensor([float("nan"), float("-inf"), float("inf"), 0.0])
        out = F.leaky_relu(_to_device(host), 0.1).cpu()
        expected = F.leaky_relu(host, 0.1)
        assert torch.isnan(out).equal(torch.isnan(expected))
        torch.testing.assert_close(out.nan_to_num(), expected.nan_to_num())

    @pytest.mark.anyplatform
    def test_row_pitch_and_strided_operands(self):
        host = torch.randn(4, 5)  # 20-byte float32 rows
        torch.testing.assert_close(
            F.leaky_relu(_to_device(host), 0.1).cpu(), F.leaky_relu(host, 0.1)
        )
        strided_host = torch.linspace(-4.0, 4.0, 32)
        strided = _to_device(strided_host)[::2]
        assert not strided.is_contiguous()
        torch.testing.assert_close(
            F.leaky_relu(strided, 0.1).cpu(), F.leaky_relu(strided_host[::2], 0.1)
        )

    @pytest.mark.anyplatform
    def test_rank_zero_and_empty(self):
        scalar = torch.tensor(-2.5)
        out = F.leaky_relu(_to_device(scalar), 0.1)
        assert out.shape == ()
        assert out.item() == F.leaky_relu(scalar, 0.1).item()
        assert F.leaky_relu(_to_device(torch.empty(0)), 0.1).numel() == 0


class TestElu:
    """`elu` equals the reference at every alpha."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu_over_the_domain(self, dtype):
        host = torch.linspace(-6.0, 6.0, 129, dtype=dtype)
        for alpha in ALPHAS:
            out = F.elu(_to_device(host), alpha)
            expected = F.elu(host, alpha)
            assert out.dtype == dtype
            torch.testing.assert_close(
                out.cpu().float(), expected.float(), **_tolerance(dtype)
            )

    @pytest.mark.anyplatform
    def test_at_zero_the_two_branches_meet(self):
        """`elu(0)` is 0 from both sides, and the positive branch is the identity."""
        host = torch.tensor([0.0, 1.0, 2.0])
        torch.testing.assert_close(F.elu(_to_device(host), 1.0).cpu(), F.elu(host, 1.0))

    @pytest.mark.anyplatform
    def test_the_alpha_is_part_of_the_plan_key(self):
        host = torch.tensor([-3.0, -1.0, -0.5])
        device_host = _to_device(host)
        for alpha in (0.5, 1.0, 2.0, 1.0):
            torch.testing.assert_close(
                F.elu(device_host, alpha).cpu(),
                F.elu(host, alpha),
                msg=f"alpha={alpha}",
            )

    @pytest.mark.anyplatform
    def test_the_scale_and_input_scale_compose_around_the_node(self):
        """The node takes only `alpha`; `scale`/`input_scale` wrap it.

        MPSGraph's `eluWithTensor:alpha:name:` has no notion of a scale, so the
        body applies them as two multiplications. `elu(x, a, scale=s,
        input_scale=i)` is `s * elu(i * x, a)`, and a body that dropped either
        would be off by a constant factor that a default-argument test cannot
        see.
        """
        host = torch.linspace(-4.0, 4.0, 65)
        out = torch.ops.aten.elu(_to_device(host), 1.0, 2.0, 0.5)
        expected = torch.ops.aten.elu(host, 1.0, 2.0, 0.5)
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.anyplatform
    def test_nan_and_infinities(self):
        host = torch.tensor([float("nan"), float("-inf"), float("inf")])
        out = F.elu(_to_device(host), 1.0).cpu()
        expected = F.elu(host, 1.0)
        assert torch.isnan(out).equal(torch.isnan(expected))
        torch.testing.assert_close(out.nan_to_num(), expected.nan_to_num())

    @pytest.mark.anyplatform
    def test_row_pitch_and_strided_operands(self):
        host = torch.randn(4, 5)
        torch.testing.assert_close(F.elu(_to_device(host), 1.0).cpu(), F.elu(host, 1.0))
        strided_host = torch.linspace(-4.0, 4.0, 32)
        strided = _to_device(strided_host)[::2]
        torch.testing.assert_close(
            F.elu(strided, 1.0).cpu(), F.elu(strided_host[::2], 1.0)
        )


class TestDeclines:
    """The widths and the op the coverage does not serve stay correct."""

    @pytest.mark.parametrize(
        "name,fn",
        [
            ("leaky_relu", lambda t: F.leaky_relu(t, 0.1)),
            ("elu", lambda t: F.elu(t, 1.0)),
        ],
    )
    @pytest.mark.anyplatform
    def test_float64_round_trips(self, name, fn):
        host = torch.linspace(-4.0, 4.0, 33, dtype=torch.float64)
        out = fn(_to_device(host))
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.cpu(), fn(host))

    @pytest.mark.anyplatform
    def test_relu_stays_on_the_fallback(self):
        """Deliberate, and measured -- see the module docstring.

        Serving it would trade a cheap host kernel for a GPU dispatch: 0.09x at
        (64, 64) and 0.51x at (1, 16, 512, 512), with the already-served
        single-node `clamp_min` at 0.47x on the same large shape.
        """
        assert routed_backend_or_none("relu") is None

    @pytest.mark.anyplatform
    def test_relu_still_answers_correctly(self):
        """It is on the fallback, not broken."""
        host = torch.tensor([-2.0, -0.0, 0.0, 1.0, float("inf"), float("nan")])
        out = torch.relu(_to_device(host)).cpu()
        expected = torch.relu(host)
        assert torch.isnan(out).equal(torch.isnan(expected))
        torch.testing.assert_close(out.nan_to_num(), expected.nan_to_num())


class TestRouting:
    """The two served activations are routed here, and the log says so."""

    @pytest.mark.anyplatform
    def test_both_are_routed(self):
        assert routed_backend_or_none("leaky_relu") in ("apple", "mps")
        assert routed_backend_or_none("elu") in ("apple", "mps")

    @pytest.mark.anyplatform
    def test_the_dispatch_log_names_the_backend(self):
        script = (
            "import torch, torch_fl\n"
            "import torch.nn.functional as F\n"
            f"d = torch.device({DEVICE!r})\n"
            "x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], device=d)\n"
            "F.leaky_relu(x, 0.1)\n"
            "F.elu(x, 1.0)\n"
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
        for name in ("leaky_relu", "elu"):
            lines = [
                ln
                for ln in proc.stderr.splitlines()
                if "->" in ln and f"] {name}" in ln
            ]
            assert any("-> apple" in ln or "-> mps" in ln for ln in lines), (
                f"no dispatch line named the backend for {name}:\n" + "\n".join(lines)
            )
