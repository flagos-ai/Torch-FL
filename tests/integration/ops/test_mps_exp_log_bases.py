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

"""Apple Silicon (Metal / MPSGraph) exp2, log2 and log10.

Three base-specific spellings that sit beside the served `exp` and `log`. What
made them worth an increment is that MPSGraph names each base directly:

    exp2   exponentBase2WithTensor:
    log2   logarithmBase2WithTensor:
    log10  logarithmBase10WithTensor:

so none of them needs the constant multiply that `log(x) / log(2)` would, which
is what the generic `logarithmWithTensor:` would have forced. That is also why
they are not all the same speed -- see the module's closing note.

The cases below are about the two things a log/exp can get wrong beyond rounding:

* **The domain.** `log2`/`log10` of a negative answers `nan`, of zero answers
  `-inf`, and of `+inf` answers `+inf`. Each is the reference's value rather than
  an error, so each is asserted.
* **The widths.** float32, float16 and bfloat16 are served, with the same gate
  `exp`/`log` use. float64 has no MPSGraph element type and an integral operand
  is promoted by the reference, so both take the round-trip.

On the measurements: `log2` and `log10` are the wins (2.5-2.7x at
(1,16,512,512) on a first pass, 2.9-4.2x warmed up across two shapes), while
`exp2` is the weak one (1.10x at the same shape, 1.62x at 1024x1024). It is
served anyway because the gain is positive and the operand stays on the device,
but the number is reported as measured rather than folded in with the other two.

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

BASE_OPS = {
    "exp2": torch.exp2,
    "log2": torch.log2,
    "log10": torch.log10,
}

LOG_OPS = ["log2", "log10"]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.clone().to(DEVICE)


def _tolerance(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"rtol": 1e-6, "atol": 1e-6}
    return {"rtol": 5e-3, "atol": 5e-3}


class TestCorrectness:
    """Each spelling equals the CPU reference over its domain."""

    @pytest.mark.parametrize("name", sorted(BASE_OPS))
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, name, dtype):
        fn = BASE_OPS[name]
        # A positive operand for the logs, a signed one for exp2, so each op is
        # asked about the input it actually accepts.
        host = (
            torch.logspace(-3.0, 3.0, 129, dtype=dtype)
            if name in LOG_OPS
            else torch.linspace(-8.0, 8.0, 129, dtype=dtype)
        )
        out = fn(_to_device(host))
        assert out.dtype == dtype
        assert out.device.type == "flagos"
        torch.testing.assert_close(
            out.cpu().float(), fn(host).float(), **_tolerance(dtype)
        )

    @pytest.mark.parametrize("name", LOG_OPS)
    @pytest.mark.anyplatform
    def test_the_log_domain(self, name):
        """A negative is nan, a zero is -inf, and +inf is +inf."""
        fn = BASE_OPS[name]
        host = torch.tensor([-2.0, -0.5, 0.0, float("inf"), 1.0, 1024.0])
        out = fn(_to_device(host)).cpu()
        expected = fn(host)
        assert torch.isnan(out[:2]).all(), out
        assert out[2].item() == float("-inf")
        assert out[3].item() == float("inf")
        assert out[4].item() == 0.0, (name, out)
        torch.testing.assert_close(out[5:], expected[5:], **_tolerance(torch.float32))

    @pytest.mark.parametrize("name", sorted(BASE_OPS))
    @pytest.mark.anyplatform
    def test_a_nan_operand_propagates(self, name):
        fn = BASE_OPS[name]
        assert torch.isnan(fn(_to_device(torch.tensor([float("nan")]))).cpu()).all()

    @pytest.mark.anyplatform
    def test_exp2_infinities(self):
        host = torch.tensor([float("-inf"), float("inf")])
        out = torch.exp2(_to_device(host)).cpu()
        assert out[0].item() == 0.0
        assert out[1].item() == float("inf")

    @pytest.mark.parametrize("name", sorted(BASE_OPS))
    @pytest.mark.anyplatform
    def test_the_bases_are_the_bases(self, name):
        """`log2(8)` is 3 and `log10(1000)` is 3, not `ln`.

        A body that accidentally used the natural-log node would be off by a
        constant factor that a tolerance against the *same wrong* reference could
        hide -- this pins the values that only the right base produces.
        """
        fn = BASE_OPS[name]
        cases = {"log2": (8.0, 3.0), "log10": (1000.0, 3.0), "exp2": (3.0, 8.0)}
        arg, want = cases[name]
        out = fn(_to_device(torch.tensor([arg]))).cpu()
        torch.testing.assert_close(out, torch.tensor([want]))

    @pytest.mark.parametrize("name", sorted(BASE_OPS))
    @pytest.mark.anyplatform
    def test_row_pitch_and_strided_operands(self, name):
        fn = BASE_OPS[name]
        host = (
            torch.logspace(-2.0, 2.0, 20).reshape(4, 5)
            if name in LOG_OPS
            else torch.linspace(-4.0, 4.0, 20).reshape(4, 5)
        )
        torch.testing.assert_close(fn(_to_device(host)).cpu(), fn(host))

        strided_host = torch.logspace(-2.0, 2.0, 32)
        strided = _to_device(strided_host)[::2]
        assert not strided.is_contiguous()
        torch.testing.assert_close(fn(strided).cpu(), fn(strided_host[::2]))

    @pytest.mark.parametrize("name", sorted(BASE_OPS))
    @pytest.mark.anyplatform
    def test_rank_zero_and_empty(self, name):
        fn = BASE_OPS[name]
        scalar = torch.tensor(2.0)
        out = fn(_to_device(scalar))
        assert out.shape == ()
        torch.testing.assert_close(out.cpu(), fn(scalar))
        assert fn(_to_device(torch.empty(0))).numel() == 0


class TestDeclines:
    """The widths the gate does not serve stay correct through the reference."""

    @pytest.mark.parametrize("name", sorted(BASE_OPS))
    @pytest.mark.anyplatform
    def test_float64_round_trips(self, name):
        fn = BASE_OPS[name]
        host = torch.tensor([0.5, 2.0, 8.0], dtype=torch.float64)
        out = fn(_to_device(host))
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.cpu(), fn(host))

    @pytest.mark.parametrize("name", sorted(BASE_OPS))
    @pytest.mark.anyplatform
    def test_integral_operands_promote(self, name):
        """The reference promotes; MPSGraph would keep the integral width."""
        fn = BASE_OPS[name]
        host = torch.tensor([1, 2, 4], dtype=torch.int64)
        out = fn(_to_device(host))
        expected = fn(host)
        assert out.dtype == expected.dtype == torch.float32
        torch.testing.assert_close(out.cpu(), expected, **_tolerance(torch.float32))


class TestRouting:
    """All three are routed here, and the log says so."""

    @pytest.mark.parametrize("name", sorted(BASE_OPS))
    @pytest.mark.anyplatform
    def test_the_conf_routes_it(self, name):
        assert routed_backend_or_none(name) in ("apple", "mps")

    @pytest.mark.anyplatform
    def test_the_dispatch_log_names_the_backend(self):
        script = (
            "import torch, torch_fl\n"
            f"d = torch.device({DEVICE!r})\n"
            "x = torch.rand(8, device=d) + 0.5\n"
            "for fn in (torch.exp2, torch.log2, torch.log10):\n"
            "    fn(x)\n"
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
        for name in sorted(BASE_OPS):
            lines = [
                ln
                for ln in proc.stderr.splitlines()
                if "->" in ln and f"] {name}" in ln
            ]
            assert any("-> apple" in ln or "-> mps" in ln for ln in lines), (
                f"no dispatch line named the backend for {name}:\n" + "\n".join(lines)
            )
