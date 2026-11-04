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

"""Stage-4 Apple Silicon (Metal / MPSGraph) ``addmm``, the ``nn.Linear`` body.

``test_mps_gemm.py`` serves ``mm`` and ``bmm``; ``addmm`` is the spelling a model
actually reaches. On a 3-D input ``nn.Linear`` lowers to ``t()`` plus
``aten.addmm``, which shows in the dispatch log, and until this increment that
``addmm`` was routed to ``none`` -- so every ``nn.Linear`` in every model paid a
device-to-host copy of the weight, the CPU product and a copy back. Measured on
an M5 over a (2048, 2048) float32 product, ``torch.addmm(a, a, a)`` cost 15.1 ms
against 7.9 ms for ``torch.mm`` over the same operands; served, it costs 5.4 ms.

Four properties are specific to this op rather than to ``mm``.

* **``alpha`` and ``beta`` are host values it bakes in.** They are not visible
  in any operand's dtype or shape, so both are named in the plan key; two calls
  that differ only in ``alpha`` must not be answered by one another's compiled
  graph. The ``TestScaling`` cases hold that, and the interleaved-scaling
  reuse case is the one that fails if the key drops them.
* **The bias broadcasts onto the product.** aten's ``infer_size`` decides the
  result shape and rejects a bias that does not reach it, so the body asks the
  reference rather than restating the rule. A 1-D bias, a ``(1, N)`` one, a
  rank-0 scalar and a bias as wide as the product are all pinned, and a bias
  that does not broadcast is pinned as the reference's own error.
* **``beta == 0`` does not read the bias at all.** The reference's answer does
  not depend on it, so a bias of a declined width is still a valid call -- it is
  asserted here rather than assumed.
* **The transposed operand is read in place.** ``F.linear`` hands over
  ``weight.t()``, the layout ``MpsGemmOperand`` carries into the graph instead of
  copying, and the peak allocation is what tells the two readings apart.

Every case self-skips off an MPS build, like the stage-1 to stage-3 suites.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401
import torch_fl.flagos as flagos

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

LOW_PRECISION_TOLERANCE = {"rtol": 1e-3, "atol": 1e-3}


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _tolerance(dtype: torch.dtype) -> dict:
    return {} if dtype == torch.float32 else LOW_PRECISION_TOLERANCE


def _assert_matches_cpu(out: torch.Tensor, expected: torch.Tensor, dtype) -> None:
    assert out.shape == expected.shape
    assert out.dtype == expected.dtype
    assert out.device.type == "flagos"
    torch.testing.assert_close(out.cpu().float(), expected.float(), **_tolerance(dtype))


class TestAddmmCorrectness:
    """``addmm`` over flagos tensors equals the CPU reference."""

    @pytest.mark.parametrize(
        "bias_shape,m,k,n",
        [
            ((8, 4), 8, 16, 4),
            ((), 8, 16, 4),  # a rank-0 bias
            ((4,), 8, 16, 4),  # the nn.Linear bias: (N,)
            ((1, 4), 8, 16, 4),
            ((1, 1), 8, 16, 4),
            ((3, 5), 3, 7, 5),  # 28-byte rows: the row-pitch cases
            ((5,), 4, 9, 5),
            ((2, 3), 2, 6, 3),  # a small exact bias, the 1x1-batch reading
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_for_bias_shapes(self, bias_shape, m, k, n):
        """Every bias the reference broadcasts onto the product.

        The product is (M, K) x (K, N) = (M, N) and the bias is broadcast onto
        it, so the shapes here are the ones a reader would expect -- the exact
        shape, the ``(N,)`` an ``nn.Linear`` carries, the one-sided forms --
        plus the rank-0 scalar. A bias that *cannot* broadcast is a different
        case and lives with the declines.
        """
        bias = torch.zeros(bias_shape) if bias_shape == () else torch.randn(*bias_shape)
        a = torch.randn(m, k)
        b = torch.randn(k, n)
        out = torch.addmm(_to_device(bias), _to_device(a), _to_device(b))
        _assert_matches_cpu(out, torch.addmm(bias, a, b), torch.float32)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu_for_every_supported_dtype(self, dtype):
        """Every dtype ``MpsSupportsGemmDtype`` claims, at that width's tolerance."""
        bias = torch.randn(16, 8, dtype=dtype)
        a = torch.randn(16, 24, dtype=dtype) * 2.0
        b = torch.randn(24, 8, dtype=dtype) * 2.0
        out = torch.addmm(_to_device(bias), _to_device(a), _to_device(b))
        _assert_matches_cpu(out, torch.addmm(bias, a, b), dtype)

    @pytest.mark.anyplatform
    def test_transposed_second_operand_is_read_in_place(self):
        """``weight.t()`` -- the ``F.linear`` case -- is not copied.

        The peak allocation is what separates the two readings: a (1024, 1024)
        weight copied on the device adds 4 MB to it, the (64, 1024) result adds
        256 KB.
        """
        bias = torch.randn(1024)
        a = torch.randn(64, 1024)
        b = torch.randn(1024, 1024).t()
        assert not b.is_contiguous()
        device_a, device_b = _to_device(a), _to_device(b)
        device_bias = _to_device(bias)
        flagos.synchronize()
        flagos.reset_peak_memory_stats()
        before = flagos.memory_stats()["peak_allocated_bytes"]
        out = torch.addmm(device_bias, device_a, device_b)
        flagos.synchronize()
        delta = flagos.memory_stats()["peak_allocated_bytes"] - before
        # The contracting dimension is 1024 here, so the graph's accumulation
        # order and the reference's round differently by a few ulp of the sum --
        # the same reason test_mps_gemm.py holds its own 1024-wide cases loosely.
        expected = torch.addmm(bias, a, b)
        assert out.shape == expected.shape and out.dtype == expected.dtype
        assert out.device.type == "flagos"
        torch.testing.assert_close(
            out.cpu().float(), expected.float(), rtol=1e-3, atol=1e-3
        )
        allowed = 2 * out.numel() * out.element_size()
        assert delta <= allowed, (
            f"addmm allocated {delta} bytes over the result's {allowed}: the "
            "transposed operand was materialised rather than read in place"
        )

    @pytest.mark.anyplatform
    def test_linear_forward_matches_cpu(self):
        """``nn.Linear``, on both the rank-2 and the rank-3 input.

        The rank-3 input is the ``transformers`` shape and takes ``t()`` plus
        ``addmm``; the rank-2 input takes ``addmm`` directly. Both are pinned so
        a change to either lowering shows here.
        """
        torch.manual_seed(0)
        host = torch.nn.Linear(16, 5)
        device = torch.nn.Linear(16, 5)
        device.load_state_dict(host.state_dict())
        device = device.to(DEVICE)
        for shape in ((2, 7, 16), (7, 16)):
            inp = torch.randn(*shape)
            _assert_matches_cpu(
                device(inp.to(DEVICE)),
                host(inp),
                torch.float32,
            )

    @pytest.mark.anyplatform
    def test_repeated_calls_with_the_same_shapes_do_not_alias(self):
        """The cached plan is re-fed per call, not reused as a result."""
        bias = torch.randn(8, 4)
        a = torch.randn(8, 16)
        b = torch.randn(16, 4)
        first = torch.addmm(_to_device(bias), _to_device(a), _to_device(b))
        bias2 = torch.randn(8, 4)
        a2 = torch.randn(8, 16)
        b2 = torch.randn(16, 4)
        second = torch.addmm(_to_device(bias2), _to_device(a2), _to_device(b2))
        _assert_matches_cpu(first, torch.addmm(bias, a, b), torch.float32)
        _assert_matches_cpu(second, torch.addmm(bias2, a2, b2), torch.float32)
        assert not torch.equal(first.cpu(), second.cpu())


class TestScaling:
    """``alpha`` and ``beta`` are part of the graph, and part of its key."""

    @pytest.mark.parametrize(
        "alpha,beta",
        [
            (1, 1),  # the ordinary nn.Linear path: no node for either
            (2, 1),
            (1, 2),
            (0.5, 0.5),
            (0.25, 3.0),
            (2.0, 2.0),
            (1, 0),  # beta == 0: the bias is not read at all
            (0, 1),
            (0, 0),
            (1.0, -1.0),
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_for_alpha_and_beta(self, alpha, beta):
        bias = torch.randn(8, 4)
        a = torch.randn(8, 16)
        b = torch.randn(16, 4)
        out = torch.addmm(
            _to_device(bias), _to_device(a), _to_device(b), beta=beta, alpha=alpha
        )
        _assert_matches_cpu(
            out, torch.addmm(bias, a, b, beta=beta, alpha=alpha), torch.float32
        )

    @pytest.mark.anyplatform
    def test_interleaved_scalings_do_not_share_a_plan(self):
        """The one case that fails if a scaling leaves the plan key.

        The four calls describe four identical buffers at one dtype, so
        MpsRunGraph's own dtype-and-shape signature folds them into one key; the
        body supplies ``alpha`` and ``beta`` itself. If either were dropped from
        that key the second call would be answered by the first call's compiled
        graph and return the wrong value -- which is a wrong answer rather than a
        slow path, so it is pinned here.
        """
        bias = torch.randn(8, 4)
        a = torch.randn(8, 16)
        b = torch.randn(16, 4)
        d_bias, d_a, d_b = _to_device(bias), _to_device(a), _to_device(b)
        for alpha, beta in ((1, 1), (2.0, 1.0), (1, 0.5), (0.5, 2.0), (1, 1)):
            got = torch.addmm(d_bias, d_a, d_b, beta=beta, alpha=alpha)
            _assert_matches_cpu(
                got, torch.addmm(bias, a, b, beta=beta, alpha=alpha), torch.float32
            )

    @pytest.mark.anyplatform
    def test_beta_zero_does_not_read_the_bias(self):
        """``beta == 0`` is a valid call whatever the bias is.

        The reference's answer does not depend on the bias when ``beta`` is
        zero, so the body does not build it into the graph. Measured against a
        deliberately non-finite bias: if the node were built, the ``0 * nan``
        would reach the sum and poison it.
        """
        bias = torch.full((8, 4), float("nan"))
        a = torch.randn(8, 16)
        b = torch.randn(16, 4)
        out = torch.addmm(_to_device(bias), _to_device(a), _to_device(b), beta=0)
        expected = torch.addmm(bias, a, b, beta=0)
        _assert_matches_cpu(out, expected, torch.float32)
        assert torch.isfinite(out.cpu()).all()


class TestDeclinedCases:
    """Everything the graph does not serve stays correct through the reference."""

    @pytest.mark.anyplatform
    def test_float64_round_trips(self):
        """float64 has no MPSGraph element type; the value is still right."""
        bias = torch.randn(4, 3, dtype=torch.float64)
        a = torch.randn(4, 5, dtype=torch.float64)
        b = torch.randn(5, 3, dtype=torch.float64)
        out = torch.addmm(_to_device(bias), _to_device(a), _to_device(b))
        _assert_matches_cpu(out, torch.addmm(bias, a, b), torch.float64)

    @pytest.mark.anyplatform
    def test_integral_round_trips(self):
        """aten accumulates an integral ``addmm`` in its own width."""
        bias = torch.randint(-4, 5, (4, 3), dtype=torch.int64)
        a = torch.randint(-4, 5, (4, 5), dtype=torch.int64)
        b = torch.randint(-4, 5, (5, 3), dtype=torch.int64)
        out = torch.addmm(_to_device(bias), _to_device(a), _to_device(b))
        _assert_matches_cpu(out, torch.addmm(bias, a, b), torch.int64)

    @pytest.mark.anyplatform
    def test_a_half_mat_pair_over_a_float_bias_is_the_reference_error(self):
        """aten's ``addmm`` promotes nothing, and the reference is what says so.

        The body holds the bias to the ``mat`` operands' width and declines
        otherwise, so the call reaches the reference's own
        ``self and mat2 must have the same dtype`` rather than a widened graph
        quietly producing a float32 answer.
        """
        bias = torch.randn(8, 4, dtype=torch.float32)
        a = torch.randn(8, 16, dtype=torch.float16)
        b = torch.randn(16, 4, dtype=torch.float16)
        with pytest.raises(RuntimeError, match="same dtype"):
            torch.addmm(_to_device(bias), _to_device(a), _to_device(b))

    @pytest.mark.anyplatform
    def test_a_half_bias_over_float_mat_operands_round_trips(self):
        """A bias the graph declines is answered by the reference, not moved.

        ``(N,)`` is the ``nn.Linear`` shape, so this is the spelling a caller
        meets; the result is the reference's. A bias of *another* width than the
        ``mat`` pair is a reference error rather than a result, which is the case
        beside this one.
        """
        bias = torch.randn(4)
        a = torch.randn(8, 16)
        b = torch.randn(16, 4)
        out = torch.addmm(_to_device(bias), _to_device(a), _to_device(b))
        _assert_matches_cpu(out, torch.addmm(bias, a, b), torch.float32)

    @pytest.mark.anyplatform
    def test_a_bias_that_does_not_broadcast_reaches_the_reference_error(self):
        with pytest.raises(RuntimeError):
            torch.addmm(
                _to_device(torch.randn(8, 5)),
                _to_device(torch.randn(8, 16)),
                _to_device(torch.randn(16, 4)),
            )

    @pytest.mark.anyplatform
    def test_a_non_2d_mat_operand_reaches_the_reference_error(self):
        with pytest.raises(RuntimeError):
            torch.addmm(
                _to_device(torch.randn(4)),
                _to_device(torch.randn(2, 8, 16)),
                _to_device(torch.randn(16, 4)),
            )

    @pytest.mark.anyplatform
    def test_mismatched_contracting_dimensions_reach_the_reference_error(self):
        with pytest.raises(RuntimeError):
            torch.addmm(
                _to_device(torch.randn(8, 4)),
                _to_device(torch.randn(8, 16)),
                _to_device(torch.randn(17, 4)),
            )


class TestRouting:
    """The conf routes ``addmm`` here, and the dispatch log says so."""

    @pytest.mark.anyplatform
    def test_the_conf_routes_addmm_to_this_backend(self):
        assert routed_backend_or_none("addmm") in ("apple", "mps")

    @pytest.mark.anyplatform
    def test_the_dispatch_log_names_the_backend(self):
        """``torch.addmm`` and ``nn.Linear`` both reach the kernel.

        Pinned in a subprocess because the dispatch log is read from the
        process's stderr as it starts.
        """
        script = (
            "import torch, torch_fl\n"
            f"d = torch.device({DEVICE!r})\n"
            "bias = torch.randn(8, 4, device=d)\n"
            "a = torch.randn(8, 16, device=d)\n"
            "b = torch.randn(16, 4, device=d)\n"
            "torch.addmm(bias, a, b)\n"
            "lin = torch.nn.Linear(16, 4).to(d)\n"
            "lin(torch.randn(3, 16, device=d))\n"
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
        lines = [ln for ln in proc.stderr.splitlines() if "addmm" in ln]
        assert any("-> apple" in ln or "-> mps" in ln for ln in lines), (
            "no addmm dispatch line named the backend:\n" + "\n".join(lines)
        )
