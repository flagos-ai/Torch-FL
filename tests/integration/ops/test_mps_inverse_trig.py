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

"""Apple Silicon (Metal / MPSGraph) inverse and hyperbolic trigonometry.

``sin`` and ``cos`` have been served since the ``trig`` category landed; the
eight spellings here are the rest of that family, and each is one MPSGraph
selector with no composition and no host value:

    asin   acos   atan   asinh   acosh   atanh   sinh   cosh

They were on the fallback until this increment, which meant that any model
reaching for one -- an angle, an attention scale written through `atanh`, a
softplus alternative built from `sinh` and `cosh` -- paid a device-to-host copy,
a host kernel and a copy back for a single arithmetic node.

Three things decide the cases below.

* **The domain, not the magnitude, is where a wrong answer hides.** `asin` and
  `acos` are undefined outside `[-1, 1]` and answer `nan` there; `acosh` answers
  `nan` below 1; `atanh` answers `+-inf` at `+-1` and `nan` beyond it. Those are
  the reference's values, not errors, so each is asserted rather than avoided.
* **The widths are the ``trig`` category's.** float32, float16 and bfloat16 are
  served; float64 has no MPSGraph element type and an integral operand is
  promoted to float32 by the reference, so both keep the CPU round-trip. The
  gate is the one ``sin`` and ``cos`` already use.
* **The signed zero survives.** ``asin(-0.0) = -0.0``, ``atan(-0.0) = -0.0``,
  ``sinh(-0.0) = -0.0`` and ``atanh(-0.0) = -0.0``, and ``acosh``/``cosh`` are
  even. A node that dropped the sign would pass a tolerance-based case and fail
  a model, so the sign bit is pinned.

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

# op name -> (representative domain, a point-wise sample count)
DOMAINS = {
    "asin": (-0.99, 0.99),
    "acos": (-0.99, 0.99),
    "atan": (-40.0, 40.0),
    "asinh": (-20.0, 20.0),
    "acosh": (1.01, 40.0),
    "atanh": (-0.99, 0.99),
    "sinh": (-12.0, 12.0),
    "cosh": (-12.0, 12.0),
}

# The same ops as callables, so a parametrised case can name its own function.
OPS = {name: getattr(torch, name) for name in DOMAINS}


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.clone().to(DEVICE)


def _tolerance(dtype: torch.dtype, name: str) -> dict:
    """Per-width tolerance, widened where the function amplifies input error.

    ``sinh`` and ``cosh`` grow like ``exp(x)``, so at ``x = 12`` a float32 input
    that is one ulp off moves the result by more than an absolute 1e-5. The
    relative tolerance covers that; the absolute one is what a value near zero
    needs. ``atanh`` near +-1 has an unbounded derivative, so its sample stops
    at 0.99 and the float32 band there still needs the wider pair.
    """
    if dtype == torch.float32:
        if name in ("sinh", "cosh", "atanh"):
            return {"rtol": 1e-5, "atol": 1e-5}
        return {"rtol": 1e-6, "atol": 1e-6}
    return {"rtol": 5e-3, "atol": 5e-3}


class TestCorrectness:
    """Each spelling equals the CPU reference over its domain."""

    @pytest.mark.parametrize("name", sorted(DOMAINS))
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu_over_the_domain(self, name, dtype):
        lo, hi = DOMAINS[name]
        host = torch.linspace(lo, hi, 257, dtype=dtype)
        out = OPS[name](_to_device(host))
        assert out.dtype == dtype
        assert out.device.type == "flagos"
        torch.testing.assert_close(
            out.cpu().float(),
            OPS[name](host).float(),
            **_tolerance(dtype, name),
        )

    @pytest.mark.parametrize("name", sorted(DOMAINS))
    @pytest.mark.anyplatform
    def test_matches_cpu_at_the_row_pitch_edges(self, name):
        """A (4, 5) float32 operand has 20-byte rows.

        The marshalling note in ``mps_common.h`` records that an N-D MPSNDArray
        reads with its own row pitch, so a 20-byte row width is where a
        descriptor built from the aten shape reads the wrong bytes. The shape
        cases here are the ones the unary suite uses for the same reason.
        """
        lo, hi = DOMAINS[name]
        host = torch.linspace(lo, hi, 4 * 5, dtype=torch.float32).reshape(4, 5)
        out = OPS[name](_to_device(host))
        torch.testing.assert_close(
            out.cpu(), OPS[name](host), **_tolerance(torch.float32, name)
        )

    @pytest.mark.parametrize("name", sorted(DOMAINS))
    @pytest.mark.anyplatform
    def test_rank_zero_and_empty_operands(self, name):
        lo, hi = DOMAINS[name]
        mid = (lo + hi) / 2.0
        scalar = torch.tensor(mid)
        out = OPS[name](_to_device(scalar))
        assert out.shape == ()
        torch.testing.assert_close(
            out.cpu(), OPS[name](scalar), **_tolerance(torch.float32, name)
        )
        empty = torch.empty(0)
        assert OPS[name](_to_device(empty)).numel() == 0

    @pytest.mark.parametrize("name", sorted(DOMAINS))
    @pytest.mark.anyplatform
    def test_strided_operand(self, name):
        lo, hi = DOMAINS[name]
        host = torch.linspace(lo, hi, 64, dtype=torch.float32)
        strided = _to_device(host)[::2]
        assert not strided.is_contiguous()
        torch.testing.assert_close(
            OPS[name](strided).cpu(),
            OPS[name](host[::2]),
            **_tolerance(torch.float32, name),
        )


class TestDomainEdges:
    """Where each function is undefined, the reference's answer is pinned."""

    @pytest.mark.parametrize("name", ["asin", "acos"])
    @pytest.mark.anyplatform
    def test_outside_the_unit_interval_is_nan(self, name):
        host = torch.tensor([-2.0, -1.0001, 1.0001, 2.0, float("inf"), float("-inf")])
        out = OPS[name](_to_device(host)).cpu()
        assert torch.isnan(out).all(), out

    @pytest.mark.parametrize("name", ["asin", "acos", "atan", "asinh", "atanh", "sinh"])
    @pytest.mark.anyplatform
    def test_a_nan_operand_propagates(self, name):
        """A NaN in is a NaN out, at every served width.

        ``tanh`` needed an explicit ``isNaNWithTensor:`` guard because its node
        saturates a NaN to +-1. These do not, and this case is what would catch
        a node that started to.
        """
        for dtype in FLOAT_DTYPES:
            host = torch.tensor([float("nan")], dtype=dtype)
            out = OPS[name](_to_device(host)).cpu()
            assert torch.isnan(out).all(), (name, dtype, out)

    @pytest.mark.anyplatform
    def test_acosh_below_one_is_nan(self):
        host = torch.tensor([0.5, 0.999, -1.0, float("-inf")])
        out = torch.acosh(_to_device(host)).cpu()
        assert torch.isnan(out).all(), out

    @pytest.mark.anyplatform
    def test_acosh_at_one_is_zero(self):
        host = torch.tensor([1.0])
        out = torch.acosh(_to_device(host)).cpu()
        assert out.item() == 0.0

    @pytest.mark.anyplatform
    def test_atanh_at_the_poles(self):
        """``atanh(+-1)`` is ``+-inf`` and beyond it is ``nan``."""
        host = torch.tensor([1.0, -1.0])
        out = torch.atanh(_to_device(host)).cpu()
        assert out[0].item() == float("inf")
        assert out[1].item() == float("-inf")
        beyond = torch.tensor([1.5, -1.5])
        assert torch.isnan(torch.atanh(_to_device(beyond)).cpu()).all()

    @pytest.mark.parametrize("name", ["asin", "atan", "asinh", "atanh", "sinh"])
    @pytest.mark.anyplatform
    def test_signed_zero_survives(self, name):
        """An odd function keeps the sign of a zero operand.

        Pinned by sign bit rather than by equality, because ``-0.0 == 0.0`` and
        a signed-zero check that compares values cannot see the difference.
        """
        for dtype in FLOAT_DTYPES:
            host = torch.tensor([-0.0], dtype=dtype)
            out = OPS[name](_to_device(host)).cpu()
            assert out.item() == 0.0, (name, dtype, out)
            assert torch.signbit(out).all(), (name, dtype, out)

    @pytest.mark.parametrize("name", ["acos", "acosh", "cosh"])
    @pytest.mark.anyplatform
    def test_even_functions_are_non_negative(self, name):
        lo, hi = DOMAINS[name]
        host = torch.linspace(lo, hi, 65)
        out = OPS[name](_to_device(host)).cpu()
        assert (out >= 0).all(), (name, out.min())


class TestDeclines:
    """The widths the graph does not serve stay correct through the reference."""

    @pytest.mark.parametrize("name", sorted(DOMAINS))
    @pytest.mark.anyplatform
    def test_float64_round_trips(self, name):
        lo, hi = DOMAINS[name]
        host = torch.linspace(lo, hi, 32, dtype=torch.float64)
        out = OPS[name](_to_device(host))
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.cpu(), OPS[name](host))

    @pytest.mark.parametrize("name", sorted(DOMAINS))
    @pytest.mark.anyplatform
    def test_integral_operands_promote_to_float32(self, name):
        """The reference promotes an integral operand; MPSGraph would truncate.

        The gate declines the integral widths so the round-trip performs the
        promotion, and the result is a float32 tensor -- which is the part a
        graph running at the operand's own width would get wrong.
        """
        host = (
            torch.tensor([1, 2, 3], dtype=torch.int64)
            if name in ("acosh",)
            else torch.tensor([-1, 0, 1], dtype=torch.int64)
        )
        out = OPS[name](_to_device(host))
        expected = OPS[name](host)
        assert out.dtype == expected.dtype == torch.float32
        torch.testing.assert_close(
            out.cpu(), expected, **_tolerance(torch.float32, name)
        )


class TestRouting:
    """Each spelling is routed to this backend and the log says so."""

    @pytest.mark.parametrize("name", sorted(DOMAINS))
    @pytest.mark.anyplatform
    def test_the_conf_routes_it_here(self, name):
        assert routed_backend_or_none(name) in ("apple", "mps")

    @pytest.mark.anyplatform
    def test_the_dispatch_log_names_the_backend(self):
        """Pinned in a subprocess, because the log is read as the process starts."""
        names = sorted(DOMAINS)
        script = (
            "import torch, torch_fl\n"
            f"d = torch.device({DEVICE!r})\n"
            "x = torch.linspace(0.1, 0.9, 8, device=d)\n"
            "y = torch.linspace(1.5, 9.0, 8, device=d)\n"
            + "".join(f"torch.{n}(y if {n!r} == 'acosh' else x)\n" for n in names)
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
        for name in names:
            lines = [ln for ln in proc.stderr.splitlines() if f"{name} ->" in ln]
            assert any("-> apple" in ln or "-> mps" in ln for ln in lines), (
                f"no dispatch line named the backend for {name}:\n" + "\n".join(lines)
            )


class TestPlanReuse:
    """The compiled plan is reused across dtypes and shapes, not rebuilt."""

    @pytest.mark.parametrize("name", sorted(DOMAINS))
    @pytest.mark.anyplatform
    def test_interleaved_signatures_do_not_alias(self, name):
        """Four different operands through one op give four different answers.

        A plan is keyed on the operand's dtype and shape, so these four share
        nothing -- and if any of them did, the second call would be answered by
        the first call's buffers rather than its own.
        """
        lo, hi = DOMAINS[name]
        results = []
        for dtype, scale in (
            (torch.float32, 0.5),
            (torch.float16, 0.5),
            (torch.float32, 0.8),
        ):
            host = torch.full((3, 4), lo + (hi - lo) * scale, dtype=dtype)
            got = OPS[name](_to_device(host)).cpu()
            expected = OPS[name](host)
            torch.testing.assert_close(
                got.float(), expected.float(), **_tolerance(dtype, name)
            )
            results.append(got.float())
        assert not torch.equal(results[0], results[2]), (
            "two different operands were answered by the same plan"
        )
