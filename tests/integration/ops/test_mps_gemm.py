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

"""Stage-3 Apple Silicon (Metal / MPSGraph) matrix products: ``mm`` and ``bmm``.

Stage 3 of the MPSGraph operator layer (see
docs/superpowers/specs/2026-10-14-apple-mps-native-backend-design.md section 7)
opens with the ``gemm`` category, and it is the stage the whole backend was
motivated by: on a single Qwen3-0.6B layer the eight ``mm`` and two ``bmm``
calls are 71-73% of device dispatch time, and 84% at decode where the layer is
0.09 GFLOP, so what is being measured here is the host round trip around a
resident weight rather than arithmetic throughput. The cases below pin the
numbers the kernel has to be right about; the timings live in the stage's
measurement notes, not in CI.

Three properties are specific to this category.

* MPSGraph exposes exactly one relevant selector,
  ``matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:``, which
  broadcasts its operands. One graph node therefore serves both ops, and the
  kernel -- not MPSGraph -- is what has to enforce aten's shapes: an operand
  that is not 2-D for ``mm`` or 3-D for ``bmm``, a contracting dimension that
  does not match, or a pair of batch dimensions that differ is declined to the
  CPU round-trip so the reference raises its own error rather than the graph
  quietly broadcasting one.
* ``nn.Linear`` passes ``weight.t()``, and that transpose is carried into the
  graph instead of being copied: a transpose of the last two axes of a dense
  buffer is the one strided layout the kernel recognises (``MpsGemmOperand``),
  and MPSGraph's transpose node is free -- measured on an M5, a
  (512, 1024) x (1024, 32768) f32 product cost 11.9 ms with the transpose in
  the graph against 13.1 ms with the operand already dense -- while that operand
  read down its columns cost 70 ms. The layout is a host value the kernel reads,
  so it is part of the plan key as well; the transposed and strided cases below
  hold both halves of that to account.
* The integral overloads take the round-trip. aten accumulates an integral
  ``mm`` in the operand's own width and MPSGraph has no matmul for it, so a
  graph of the operands' element type would be a guess rather than a
  measurement.

Every case self-skips off an MPS build, like the stage-1 and stage-2 suites.
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


def _peak_delta_bytes(run) -> tuple[int, torch.Tensor]:
    """The flagos allocator's peak growth across one call, and its result.

    What a call allocates is the shape of what it copies: an operand
    materialised on the device shows up here as its own bytes, one the graph
    reads in place does not. Only the result is left over, so any operand-sized
    allocation past it is a copy.
    """
    flagos.synchronize()
    flagos.reset_peak_memory_stats()
    before = flagos.memory_stats()["peak_allocated_bytes"]
    out = run()
    flagos.synchronize()
    return flagos.memory_stats()["peak_allocated_bytes"] - before, out


def _tolerance(dtype: torch.dtype) -> dict:
    return {} if dtype == torch.float32 else LOW_PRECISION_TOLERANCE


def _assert_matches_cpu(out: torch.Tensor, expected: torch.Tensor, dtype) -> None:
    assert out.shape == expected.shape
    assert out.dtype == expected.dtype
    assert out.device.type == "flagos"
    torch.testing.assert_close(out.cpu().float(), expected.float(), **_tolerance(dtype))


class TestMmCorrectness:
    """``mm`` over two flagos tensors equals the CPU reference."""

    @pytest.mark.parametrize(
        "shape_a,shape_b",
        [
            ((8, 16), (16, 4)),
            ((1, 1), (1, 1)),
            ((4, 5), (5, 3)),  # 20-byte rows: the row-pitch cases
            ((3, 7), (7, 5)),
            ((128, 3), (3, 8)),
            ((64, 64), (64, 64)),
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_for_shape_pairs(self, shape_a, shape_b):
        """Both operands and the result go through the 1-D-then-reshape path.

        See the row-pitch note in ``mps_common.h``: an MPSNDArray over N-D
        memory reads with its own row pitch, so a (4, 5) float32 operand's
        20-byte rows would be read as 32 and every row after the first would
        come from the wrong offset. The first two shapes are the ones a
        plain-shaped test would miss -- a non-square result and a 1x1 product,
        where a transposed or mis-strided read still lands inside the buffer.
        """
        a = torch.randn(*shape_a)
        b = torch.randn(*shape_b)
        out = torch.mm(_to_device(a), _to_device(b))
        _assert_matches_cpu(out, torch.mm(a, b), torch.float32)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu_for_every_supported_dtype(self, dtype):
        """Every dtype ``MpsSupportsGemmDtype`` claims, at the same tolerance.

        The half widths accumulate over the contracting dimension in their own
        precision, so the tolerances are the ones the stage-1 and stage-2 suites
        use for them rather than the float32 ones.
        """
        a = torch.randn(16, 24, dtype=dtype) * 2.0
        b = torch.randn(24, 8, dtype=dtype) * 2.0
        out = torch.mm(_to_device(a), _to_device(b))
        _assert_matches_cpu(out, torch.mm(a, b), dtype)

    @pytest.mark.anyplatform
    def test_transposed_second_operand_is_read_in_place(self):
        """``weight.t()`` -- the ``nn.Linear`` case -- is not copied.

        A transposed second operand is the one that would otherwise be read as
        though the buffer were dense, and the case the whole layout path exists
        for. The peak allocation is what tells the two apart: a (1024, 1024)
        weight copied on the device adds 4 MB to it, the (64, 1024) result adds
        256 KB.
        """
        a = torch.randn(64, 1024)
        b = torch.randn(1024, 1024).t()
        assert not b.is_contiguous()
        # The operands are resident before the peak is read: what is being
        # measured is what the call itself allocates, and _to_device copies.
        device_a, device_b = _to_device(a), _to_device(b)
        delta, out = _peak_delta_bytes(lambda: torch.mm(device_a, device_b))
        _assert_matches_cpu(out, torch.mm(a, b), torch.float32)
        allowed = 2 * out.numel() * out.element_size()
        assert delta <= allowed, (
            f"mm allocated {delta} bytes over the result's {allowed}: the "
            "transposed operand was materialised rather than read in place"
        )

    @pytest.mark.anyplatform
    def test_transposed_and_dense_operands_do_not_share_a_plan(self):
        """The operand's layout is in the plan key, not only its shape.

        Both calls pass a (32, 32) operand to ``mm`` at one dtype, so their
        tensors describe two identical buffers -- one holding a matrix, the
        other its transpose -- and MpsRunGraph folds nothing but dtype and shape
        into the key. The kernel names the two layouts itself; without that, the
        second call would be answered by the first call's compiled graph.
        """
        a = torch.randn(8, 32)
        dense = torch.randn(32, 32)
        transposed = torch.randn(32, 32).t()
        device_a = _to_device(a)
        _assert_matches_cpu(
            torch.mm(device_a, _to_device(transposed)),
            torch.mm(a, transposed),
            torch.float32,
        )
        _assert_matches_cpu(
            torch.mm(device_a, _to_device(dense)),
            torch.mm(a, dense),
            torch.float32,
        )

    @pytest.mark.anyplatform
    def test_sliced_operands_are_materialised(self):
        """A slice is not a transpose, so both operands still take the copy.

        The layout path covers one stride pattern -- the last two axes, swapped,
        over dense memory. Every other strided operand is read as the dense
        buffer it is not unless it is copied first.
        """
        host_a = torch.randn(8, 32)
        host_b = torch.randn(32, 8)
        a = _to_device(host_a)[:, ::2]
        b = _to_device(host_b)[::2]
        assert not a.is_contiguous() and not b.is_contiguous()
        _assert_matches_cpu(
            torch.mm(a, b), torch.mm(host_a[:, ::2], host_b[::2]), torch.float32
        )

    @pytest.mark.anyplatform
    def test_repeated_calls_with_the_same_shapes_do_not_alias(self):
        """The cached plan is re-fed per call, not reused as a result.

        The plan is keyed on the operand signature, so the second call with the
        same shapes finds the compiled graph; the risk being checked is a plan
        that also cached the *buffers*, which would return the first pair's
        product for the second pair.
        """
        a = torch.randn(8, 16)
        b = torch.randn(16, 4)
        first = torch.mm(_to_device(a), _to_device(b))
        c = torch.randn(8, 16)
        d = torch.randn(16, 4)
        second = torch.mm(_to_device(c), _to_device(d))
        _assert_matches_cpu(first, torch.mm(a, b), torch.float32)
        _assert_matches_cpu(second, torch.mm(c, d), torch.float32)
        assert not torch.equal(first.cpu(), second.cpu())

    @pytest.mark.anyplatform
    def test_linear_forward_matches_cpu(self):
        """``nn.Linear`` -- the op the category exists for.

        A 3-D input keeps the bias add on the same path ``transformers`` uses,
        and the weight is a real ``nn.Linear`` weight so the transposed operand
        is the one the model hands over rather than a test-shaped stand-in.
        """
        torch.manual_seed(0)
        host = torch.nn.Linear(16, 5)
        inp = torch.randn(2, 7, 16)
        expected = torch.nn.functional.linear(
            inp, host.weight.detach(), host.bias.detach()
        )
        device = torch.nn.Linear(16, 5)
        device.load_state_dict(host.state_dict())
        device = device.to(DEVICE)
        _assert_matches_cpu(device(inp.to(DEVICE)), expected, torch.float32)

    @pytest.mark.anyplatform
    def test_empty_contracting_dimension_is_zeros_like_cpu(self):
        """``(m, 0) @ (0, n)`` is zeros, and nothing is handed to MPSGraph.

        The result is non-empty while neither operand is, which is the one
        shape where the operand check is what decides rather than the result
        check; MPSGraph has no descriptor for a zero-element tensor either way.
        """
        host_a = torch.zeros(4, 0)
        host_b = torch.zeros(0, 3)
        out = torch.mm(_to_device(host_a), _to_device(host_b))
        _assert_matches_cpu(out, torch.mm(host_a, host_b), torch.float32)
        assert float(out.abs().sum().cpu()) == 0.0

    @pytest.mark.anyplatform
    def test_zero_width_result_is_empty_like_cpu(self):
        """``(m, k) @ (k, 0)`` has no elements to hand back."""
        host_a = torch.zeros(4, 3)
        host_b = torch.zeros(3, 0)
        assert torch.mm(host_a, host_b).numel() == 0
        _assert_matches_cpu(
            torch.mm(_to_device(host_a), _to_device(host_b)),
            torch.mm(host_a, host_b),
            torch.float32,
        )

    @pytest.mark.parametrize(
        "dtype",
        [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8],
    )
    @pytest.mark.anyplatform
    def test_integral_operands_round_trip_exactly(self, dtype):
        """aten accumulates an integral ``mm`` in the operand's own width.

        MPSGraph has no integral matmul at all, so the result has to come from
        the reference and the comparison is exact. The operands are picked to
        overflow a 1-byte accumulator (``1 * 11 + 7 * 1 + 3 * 7 + 11 * 9`` is
        138) so a graph that accumulated in a wider type would be visible as a
        different answer; the values stay inside int8's construction range so
        the same table can be built for every dtype here.
        """
        host_a = torch.tensor([[0, 1, 3, 11], [1, 7, 3, 11]], dtype=dtype)
        host_b = torch.tensor([[3, 11], [7, 1], [11, 7], [3, 9]], dtype=dtype)
        out = torch.mm(_to_device(host_a), _to_device(host_b))
        _assert_matches_cpu(out, torch.mm(host_a, host_b), dtype)

    @pytest.mark.anyplatform
    def test_rank_mismatch_raises_exactly_like_cpu(self):
        """A 1-D or 3-D operand is declined so the reference reports it."""
        a = torch.randn(4)
        b = torch.randn(4)
        with pytest.raises(RuntimeError, match="must be a matrix"):
            torch.mm(_to_device(a), _to_device(b))
        with pytest.raises(RuntimeError, match="must be a matrix"):
            torch.mm(_to_device(torch.randn(2, 3, 4)), _to_device(torch.randn(4, 2)))

    @pytest.mark.anyplatform
    def test_contracting_dimension_mismatch_raises_exactly_like_cpu(self):
        """The inner dimensions are checked here, not left to the graph."""
        a = torch.randn(3, 4)
        b = torch.randn(5, 2)
        with pytest.raises(RuntimeError, match="cannot be multiplied"):
            torch.mm(_to_device(a), _to_device(b))

    @pytest.mark.anyplatform
    def test_mixed_dtype_pair_raises_exactly_like_cpu(self):
        """aten promotes nothing here; a mismatched pair is an error, not a cast."""
        a = torch.randn(3, 4)
        b = torch.randn(4, 2, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="same dtype"):
            torch.mm(_to_device(a), _to_device(b))

    @pytest.mark.anyplatform
    def test_float64_operands_round_trip(self):
        """float64 has no MPSGraph element type; the value is still right."""
        host_a = torch.randn(4, 5, dtype=torch.float64)
        host_b = torch.randn(5, 3, dtype=torch.float64)
        out = torch.mm(_to_device(host_a), _to_device(host_b))
        _assert_matches_cpu(out, torch.mm(host_a, host_b), torch.float64)


class TestBmmCorrectness:
    """``bmm`` -- one graph node, with aten's batch rule enforced by the kernel."""

    @pytest.mark.parametrize(
        "shape_a,shape_b",
        [
            ((4, 8, 16), (4, 16, 4)),
            ((1, 1, 1), (1, 1, 1)),
            ((2, 4, 5), (2, 5, 3)),  # 20-byte rows again, now batched
            ((1, 64, 16), (1, 16, 32)),
            ((8, 3, 7), (8, 7, 5)),
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_for_batched_shape_pairs(self, shape_a, shape_b):
        """The batch dimensions must line up, and MPSGraph is not asked to check.

        The selector broadcasts, so a pair with differing batch dimensions would
        produce a result where aten raises; the kernel's own equality check is
        what makes the reference's rule the one that holds, and the cases here
        are the shapes where the graph would otherwise broadcast a size-1 batch.
        """
        a = torch.randn(*shape_a)
        b = torch.randn(*shape_b)
        out = torch.bmm(_to_device(a), _to_device(b))
        _assert_matches_cpu(out, torch.bmm(a, b), torch.float32)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu_for_every_supported_dtype(self, dtype):
        a = torch.randn(3, 16, 24, dtype=dtype) * 2.0
        b = torch.randn(3, 24, 8, dtype=dtype) * 2.0
        out = torch.bmm(_to_device(a), _to_device(b))
        _assert_matches_cpu(out, torch.bmm(a, b), dtype)

    @pytest.mark.anyplatform
    def test_each_batch_entry_is_its_own_product(self):
        """A batched plan has to index the batch, not reuse one matrix.

        The operands are built so that every entry differs; a plan that ignored
        the batch dimension, or one that cached the first entry's result, would
        still produce a plausible-looking tensor of the right shape.
        """
        a = torch.zeros(3, 2, 2)
        b = torch.zeros(3, 2, 2)
        for i in range(3):
            a[i] = torch.eye(2) * (i + 1)
            b[i] = torch.ones(2, 2) * (i + 1)
        _assert_matches_cpu(
            torch.bmm(_to_device(a), _to_device(b)), torch.bmm(a, b), torch.float32
        )

    @pytest.mark.anyplatform
    def test_sliced_operands_are_materialised(self):
        """Both operands strided, the case the copy still covers."""
        host_a = torch.randn(2, 4, 16)
        host_b = torch.randn(2, 16, 8)
        a = _to_device(host_a)[:, :, ::2]
        b = _to_device(host_b)[:, ::2]
        assert not a.is_contiguous() and not b.is_contiguous()
        _assert_matches_cpu(
            torch.bmm(a, b),
            torch.bmm(host_a[:, :, ::2], host_b[:, ::2]),
            torch.float32,
        )

    @pytest.mark.anyplatform
    def test_transposed_attention_operand_is_read_in_place(self):
        """``bmm(q, k.transpose(1, 2))``: the same layout, one axis further left.

        The min(3) operand's buffer holds ``(b, s, d)`` and presents
        ``(b, d, s)``, which is the stride pattern the layout path recognises --
        the batch axis is simply carried along -- so the transpose stays in the
        graph. The (8, 64, 256) buffer would add 512 KB to the peak if it were
        copied; the (8, 64, 64) result adds 128 KB.
        """
        b, s, d = 8, 64, 256
        q = torch.randn(b, s, d)
        k = torch.randn(b, s, d)
        kt = k.transpose(1, 2)
        assert not kt.is_contiguous()
        device_q, device_kt = _to_device(q), _to_device(kt)
        delta, out = _peak_delta_bytes(lambda: torch.bmm(device_q, device_kt))
        _assert_matches_cpu(out, torch.bmm(q, kt), torch.float32)
        allowed = 2 * out.numel() * out.element_size()
        assert delta <= allowed, (
            f"bmm allocated {delta} bytes over the result's {allowed}: the "
            "transposed operand was materialised rather than read in place"
        )

    @pytest.mark.anyplatform
    def test_transposed_and_dense_batches_do_not_share_a_plan(self):
        """The batched form of the plan-key guard in ``TestMmCorrectness``."""
        q = torch.randn(2, 8, 16)
        dense = torch.randn(2, 16, 16)
        transposed = torch.randn(2, 16, 16).transpose(1, 2)
        device_q = _to_device(q)
        _assert_matches_cpu(
            torch.bmm(device_q, _to_device(transposed)),
            torch.bmm(q, transposed),
            torch.float32,
        )
        _assert_matches_cpu(
            torch.bmm(device_q, _to_device(dense)),
            torch.bmm(q, dense),
            torch.float32,
        )

    @pytest.mark.anyplatform
    def test_rank_mismatch_raises_exactly_like_cpu(self):
        """``bmm`` takes 3-D operands only; the reference reports the 2-D case."""
        with pytest.raises(RuntimeError, match="must be a 3D tensor"):
            torch.bmm(_to_device(torch.randn(3, 4)), _to_device(torch.randn(4, 2)))

    @pytest.mark.anyplatform
    def test_batch_mismatch_raises_exactly_like_cpu(self):
        """Unequal batch dimensions are declined, so aten's message surfaces."""
        a = torch.randn(2, 3, 4)
        b = torch.randn(5, 4, 2)
        with pytest.raises(
            RuntimeError, match="Expected size for first two dimensions"
        ):
            torch.bmm(_to_device(a), _to_device(b))

    @pytest.mark.anyplatform
    def test_contracting_dimension_mismatch_raises_exactly_like_cpu(self):
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 5, 2)
        with pytest.raises(
            RuntimeError, match="Expected size for first two dimensions"
        ):
            torch.bmm(_to_device(a), _to_device(b))

    @pytest.mark.anyplatform
    def test_empty_operand_round_trips(self):
        """A zero-element operand has no descriptor; the reference still computes."""
        host_a = torch.zeros(2, 3, 0)
        host_b = torch.zeros(2, 0, 4)
        _assert_matches_cpu(
            torch.bmm(_to_device(host_a), _to_device(host_b)),
            torch.bmm(host_a, host_b),
            torch.float32,
        )

    @pytest.mark.anyplatform
    def test_float64_operands_round_trip(self):
        host_a = torch.randn(2, 4, 5, dtype=torch.float64)
        host_b = torch.randn(2, 5, 3, dtype=torch.float64)
        _assert_matches_cpu(
            torch.bmm(_to_device(host_a), _to_device(host_b)),
            torch.bmm(host_a, host_b),
            torch.float64,
        )


class TestGemmDispatch:
    """``mm`` and ``bmm`` are claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.parametrize("op", ["mm", "bmm"])
    @pytest.mark.main_ops
    def test_conf_routes_op_to_mps(self, op):
        assert routed_backend_or_none(op) == "mps"

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """The dispatch log is the proof the kernels ran, not a fallback.

        A numeric-only check passes whether MPSGraph ran or the op reached
        ``cpu_fallback``, because both compute the same answer. The log line is
        emitted by the C++ dispatcher only on the ``mps`` slot. Run in a
        subprocess because ``LogEnabled`` latches the environment on first use.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "a = torch.randn(8, 16, device='flagos:0'); "
                "b = torch.randn(16, 4, device='flagos:0'); "
                "c = torch.randn(2, 8, 16, device='flagos:0'); "
                "d = torch.randn(2, 16, 4, device='flagos:0'); "
                "torch.mm(a, b); torch.bmm(c, d)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos dispatch] mm -> apple" in result.stderr
        assert "[flagos dispatch] bmm -> apple" in result.stderr
