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

"""Stage-3 Apple Silicon (Metal / MPSGraph) softmax.

``softmax`` is the op a small model calls between the matmuls -- one per
attention block, over the (batch, heads, seq, seq) scores -- so it is the third
stage-3 lane after ``gemm``. MPSGraph has the whole operation behind one
selector, ``softMaxWithTensor:axis:``, and the interesting part of the kernel is
the contract around that selector rather than the arithmetic. Four properties
are pinned here.

* **Which spelling is registered.** On this PyTorch build ``softmax.int`` is
  ``CompositeImplicitAutograd``: the dispatcher decomposes it above the backend
  key, so a kernel for it would never be called and there is in fact no wrapper
  to register. What reaches the ``PrivateUse1`` key -- what ``torch.softmax``,
  ``F.softmax``, SDPA and ``torch.softmax(dtype=...)`` all lower to -- is
  ``aten._softmax(self, dim, half_to_float)``. The dispatch-log cases below are
  what hold that, because a numeric-only test passes on the fallback too.
* **``half_to_float``.** The flag is the CUDA spelling of "accumulate in float
  and hand back float", and the reference *errors* on it
  ("softmax with half to float conversion is not supported on CPU"), so the
  expectation here is a float32 result equal to a float32 softmax of the widened
  operand rather than a call to ``torch._softmax`` on the host.
* **The widths.** float32 measured within 29 ulp and float16 bit-identical on an
  M5; bfloat16 came back as denormals from the same graph, so the kernel declines
  it and the CPU round-trip has to make it correct anyway. float64 and the
  integral types are the reference's own errors.
* **The axis.** aten resolves a negative ``dim`` against the rank and rejects one
  out of range; MPSGraph takes the axis as given, so the kernel resolves the
  position on the host and declines anything outside the rank.

Every case self-skips off an MPS build, like the stage-1 and stage-2 suites.
"""

import os
import subprocess
import sys

import pytest
from device_init import torch

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

FLOAT_DTYPES = [torch.float32, torch.float16]

TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}

# float16 is compared at its own width rather than through a float32 widening:
# the operand only carries ~3 decimal digits, so a reference that rounds once and
# a graph that rounds once can differ by an ulp or two of half, which is 5e-4
# relative -- well past the float32 tolerance above and not a defect.
HALF_TOLERANCE = {"rtol": 1e-3, "atol": 1e-3}


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _softmax_reference(host: torch.Tensor, dim: int, half_to_float: bool):
    """The reference ``_softmax(host, dim, half_to_float)`` is written against.

    The host CPU kernel rejects ``half_to_float`` outright, so the widened
    ``softmax`` it stands for is what the flag means: compute in float, hand
    back float. A call with the flag clear is the plain public spelling.
    """
    if half_to_float:
        return torch.softmax(host.float(), dim)
    return torch.softmax(host, dim)


class TestSoftmaxCorrectness:
    """``softmax`` on the flagos device equals the CPU reference."""

    @pytest.mark.parametrize("shape", [(4, 8), (2, 3, 5), (1, 16, 8, 8)])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu_over_the_last_axis(self, shape, dtype):
        host = torch.randn(*shape, dtype=dtype)
        out = torch.softmax(_to_device(host), dim=-1)
        assert out.dtype == dtype
        torch.testing.assert_close(
            out.cpu().float(),
            torch.softmax(host, dim=-1).float(),
            **TOLERANCE,
        )

    @pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
    @pytest.mark.anyplatform
    def test_matches_cpu_for_every_axis(self, dim):
        """The axis the graph is handed, including the negative spellings.

        aten resolves a negative ``dim`` against the rank and MPSGraph does not,
        so every negative position here is resolved on the host before the
        selector sees it. Reading the same tensor down every axis is what makes a
        mis-resolved position a wrong number rather than a shape error.
        """
        host = torch.randn(3, 4, 5)
        out = torch.softmax(_to_device(host), dim=dim)
        torch.testing.assert_close(out.cpu(), torch.softmax(host, dim=dim), **TOLERANCE)

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 5),  # 20-byte rows
            (3, 7),
            (5, 4),  # rows already a multiple of 16 bytes
            (4, 8, 16),
        ],
    )
    @pytest.mark.anyplatform
    def test_rows_narrower_than_a_16_byte_multiple(self, shape):
        """The shapes MPSNDArray's row pitch would silently corrupt.

        MPSNDArray rounds the stride between the rows of an N-D array up to 16
        bytes, so a torch-contiguous buffer described with its aten shape is
        misread whenever the innermost row is not already that wide. The kernel
        describes the buffers as 1-D and reshapes inside the graph; the aligned
        shapes in the list show the padding was the variable rather than the
        shape.
        """
        host = torch.randn(*shape)
        out = torch.softmax(_to_device(host), dim=-1)
        torch.testing.assert_close(out.cpu(), torch.softmax(host, dim=-1), **TOLERANCE)

    @pytest.mark.anyplatform
    def test_qwen_shaped_attention_scores(self):
        """The shape the op is on the hot path for: (batch, heads, seq, seq).

        A 0.6B-class model at seq 512, which is where ``softmax`` sits between
        the two matmuls. The reference is the public spelling, so this is what a
        caller gets rather than what ``_softmax`` promises in isolation.
        """
        host = torch.randn(1, 16, 512, 512) * 2.0
        out = torch.softmax(_to_device(host), dim=-1)
        torch.testing.assert_close(out.cpu(), torch.softmax(host, dim=-1), **TOLERANCE)

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        """A non-contiguous operand takes the same answer as the CPU.

        MPSGraph reads dense memory, so the kernel materialises the operand
        first. The narrowing leaves 5 columns after the stride, so the
        materialised operand has the unaligned rows too.
        """
        host = torch.randn(8, 10)
        a = _to_device(host)[:, ::2]
        assert not a.is_contiguous()
        out = torch.softmax(a, dim=-1)
        torch.testing.assert_close(
            out.cpu(), torch.softmax(host[:, ::2], dim=-1), **TOLERANCE
        )

    @pytest.mark.anyplatform
    def test_empty_tensor(self):
        a = _to_device(torch.empty(0, 4))
        out = torch.softmax(a, dim=-1)
        assert out.shape == (0, 4)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_zero_length_axis(self):
        a = _to_device(torch.empty(2, 0))
        out = torch.softmax(a, dim=-1)
        assert out.shape == (2, 0)

    @pytest.mark.anyplatform
    def test_rank_zero_is_one(self):
        """A scalar softmax is the whole mass on the one element: ``1.0``.

        The kernel declines the rank-0 case (no ``dim`` is in range for it) and
        the round-trip produces the reference's value, which the test holds.
        """
        out = torch.softmax(_to_device(torch.tensor(3.0)), dim=0)
        assert out.item() == 1.0


class TestSoftmaxHalfToFloat:
    """``_softmax(self, dim, half_to_float=True)`` widens and returns float."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_flag_widens_the_result(self, dtype):
        host = torch.randn(4, 8, dtype=dtype)
        out = torch._softmax(_to_device(host), -1, True)
        assert out.dtype == torch.float32
        torch.testing.assert_close(
            out.cpu(), _softmax_reference(host, -1, True), **TOLERANCE
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_flag_clear_keeps_the_operand_dtype(self, dtype):
        host = torch.randn(4, 8, dtype=dtype)
        out = torch._softmax(_to_device(host), -1, False)
        assert out.dtype == dtype
        torch.testing.assert_close(
            out.cpu().float(), _softmax_reference(host, -1, False).float(), **TOLERANCE
        )


class TestSoftmaxDeclinedDtypes:
    """The widths the graph does not serve stay correct through the round-trip."""

    @pytest.mark.anyplatform
    def test_bfloat16_is_correct_despite_being_declined(self):
        """The gate declines bfloat16; the answer still has to be right.

        A bfloat16 graph returned a denormal for every element on the measured
        SDK, so the kernel hands the operand to the CPU instead. The value is
        what matters: a declined width is a slow path, not a wrong one.
        """
        host = torch.randn(4, 8, dtype=torch.bfloat16)
        out = torch.softmax(_to_device(host), dim=-1)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out.cpu(), torch.softmax(host, dim=-1))

    @pytest.mark.anyplatform
    def test_float64_is_correct(self):
        host = torch.randn(4, 8, dtype=torch.float64)
        out = torch.softmax(_to_device(host), dim=-1)
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.cpu(), torch.softmax(host, dim=-1), **TOLERANCE)

    @pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.bool])
    @pytest.mark.anyplatform
    def test_integral_and_bool_raise_exactly_like_cpu(self, dtype):
        """No integer softmax on any backend; the round-trip raises the same way."""
        host = torch.tensor([1, 2, 3], dtype=dtype)
        with pytest.raises(NotImplementedError):
            torch.softmax(_to_device(host), dim=0)

    @pytest.mark.anyplatform
    def test_dim_out_of_range_raises_exactly_like_cpu(self):
        host = torch.randn(2, 3)
        with pytest.raises(IndexError):
            torch.softmax(_to_device(host), dim=5)


class TestSoftmaxScheduleAndPlanReuse:
    """The marshalling contract every MPS kernel shares (mps_common.mm)."""

    @pytest.mark.anyplatform
    def test_interleaved_axis_and_flag_never_share_a_plan(self):
        """Each axis and each ``half_to_float`` gets its own compiled graph.

        A plan key that named only the op would let the first graph built serve
        every later call: an axis-0 graph answering an axis--1 request is a wrong
        number, not a slow path. Rotating through the signatures instead of
        grouping them puts a different axis and dtype on a warm cache every step.
        """
        host = torch.randn(4, 6)
        half = host.half()
        cases = [
            (half, -1, False),
            (half, -1, True),
            (host, 0, False),
            (host, -1, False),
            (half, 0, False),
            (host, 0, False),
            (half, -1, False),
        ]
        on_device = [_to_device(operand) for operand, _, _ in cases]
        wanted = [
            _softmax_reference(operand, dim, flag) for operand, dim, flag in cases
        ]

        for round_ in range(3):
            for offset in range(len(cases)):
                index = (offset + round_) % len(cases)
                operand, dim, flag = cases[index]
                out = torch._softmax(on_device[index], dim, flag)
                assert out.dtype == wanted[index].dtype
                tolerance = TOLERANCE if out.dtype == torch.float32 else HALF_TOLERANCE
                torch.testing.assert_close(out.cpu(), wanted[index], **tolerance)

    @pytest.mark.anyplatform
    def test_results_survive_block_reuse_and_immediate_readback(self):
        """Nothing waits for the GPU, so both ends of the call are checked here.

        An operand's block goes back to the pool as soon as the caller drops it,
        while the graph reading it has only been scheduled; and the value the
        caller reads back is produced by work the runtime has not awaited. Both
        rest on the sentinel command buffer the runtime commits behind the graph
        (see mps_common.h), so a call that returned before its work was visible
        to StreamQuery would show up here as a wrong result.
        """
        held = []
        for _ in range(32):
            src = torch.randn(1 << 12, device=DEVICE)
            wanted = torch.softmax(src.cpu(), dim=-1)
            out = torch.softmax(src, dim=-1)
            del src  # recycled while the graph above may still be reading it
            for _ in range(4):  # blocks the allocator is free to hand back out
                torch.softmax(torch.randn(1 << 12, device=DEVICE), dim=-1)
            held.append((out.cpu(), wanted))  # read back without a synchronize

        for index, (got, wanted) in enumerate(held):
            torch.testing.assert_close(got, wanted, **TOLERANCE)


class TestSoftmaxDispatch:
    """``_softmax`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_softmax_to_mps(self):
        assert routed_backend_or_none("_softmax") == "mps"

    @pytest.mark.main_ops
    def test_conf_leaves_the_out_spelling_on_the_fallback(self):
        """The ``out=`` form is deliberately unregistered, like every `.out`."""
        assert routed_backend_or_none("_softmax.out") is None

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps_for_the_public_spelling(self):
        """``torch.softmax`` reaches the kernel through ``aten._softmax``.

        This is the case that shows why ``softmax.int`` is not registered: the
        public spelling decomposes above the backend key, so the log line for the
        op that actually runs is the one to hold. Run in a subprocess because
        ``LogEnabled`` latches the environment on first use.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from device_init import torch, torch_fl; "
                "a = torch.randn(4, 8, device='flagos:0'); torch.softmax(a, dim=-1)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos dispatch] _softmax -> mps" in result.stderr
