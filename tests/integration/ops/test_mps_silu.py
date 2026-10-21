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

"""Stage-4 Apple Silicon (Metal / MPSGraph) SiLU.

``silu`` is what a SwiGLU MLP multiplies its gate projection by, so it runs once
per layer over the ``(batch, seq, intermediate)`` tensor -- a larger operand than
the softmax a layer also calls. It is the first op on this backend MPSGraph has
no selector for, so its kernel is a *composition* rather than one selector, and
that is what most of this file is about. Four properties are pinned here.

* **The composition is a measurement, not an identity.** ``silu`` is defined as
  ``x * sigmoid(x)``, but aten's CPU ``silu`` is its own kernel rather than that
  expression: evaluated on the host the two readings differ by 7.8e-03 on
  float16 and 3.1e-02 on bfloat16 -- a wrong answer, not a rounding difference.
  So the case that matters is not "does the graph agree with the formula" but
  "does the graph agree with ``at::silu``", and the answer was measured on an M5
  rather than assumed: bit-identical on float16 and bfloat16, and 2.7e-07
  relative on float32, where the graph and the reference each sit 1.2353e-06 from
  an exact float64 evaluation. The tests below compare against the reference for
  that reason, including the saturated tail where a ``sigmoid`` difference is
  most visible.
* **Every width is served.** This gate is wider than the softmax one, which
  declines bfloat16 (its graph returned denormals there): the measurement is per
  op here, not per dtype, so a case that would silently follow the softmax gate
  is pinned below.
* **The plan key has no host parts.** ``silu`` takes no host argument at all, so
  its graph key is the op name and nothing else -- ``MpsRunGraph`` folds the
  operand's dtype and shape in itself. The interleaved case below is what holds
  that a dtype or shape on a warm cache still gets its own plan.
* **Which spelling is registered.** ``silu.out``, ``silu_`` and
  ``silu_backward.grad_input`` write into a tensor the caller owns, so all three
  stay on the fallback. The functional ``silu_backward`` is registered on the
  same device for equal float32 and float16 pairs, so the backward pass and an
  explicit aten call stay on MPS. bfloat16 is declined because its graph
  differed from the reference by up to 3.1e-02 around x=3. The dispatch-log
  cases hold that the calls reach their kernels, because a numeric-only test
  passes on the fallback too.

Every case self-skips off an MPS build, like the stage-1 to stage-3 suites.
"""

import os
import subprocess
import sys

import pytest
from device_init import torch

from backend_conf import routed_backend_or_none
from platform_support import detect_platform
from test_mps_runtime import _child_env


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}

# float16 and bfloat16 carry ~3 decimal digits, so a graph that rounds once and a
# reference that rounds once can differ by an ulp or two of the narrow type.
LOW_PRECISION_TOLERANCE = {"rtol": 1e-3, "atol": 1e-3}
NATIVE_BACKWARD_DTYPES = [torch.float32, torch.float16]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _reference(host: torch.Tensor) -> torch.Tensor:
    """``at::silu`` on the host, in the operand's own dtype.

    The reference the kernel is held to. Not ``host * torch.sigmoid(host)``:
    that expression is what the graph builds, and it does not reproduce
    ``at::silu`` on the narrow widths -- which is exactly what the header notes
    say, so a test written against the formula would pass while the kernel was
    wrong.
    """
    return torch.ops.aten.silu(host)


def _backward_reference(grad_output: torch.Tensor, host: torch.Tensor) -> torch.Tensor:
    """The functional derivative, promoted exactly as the reference promotes."""
    return torch.ops.aten.silu_backward(grad_output, host)


def _tolerance(dtype: torch.dtype) -> dict:
    return TOLERANCE if dtype == torch.float32 else LOW_PRECISION_TOLERANCE


def _assert_matches_cpu(out: torch.Tensor, expected: torch.Tensor) -> None:
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    torch.testing.assert_close(
        out.cpu().float(), expected.float(), **_tolerance(expected.dtype)
    )


class TestSiluCorrectness:
    """``F.silu`` on the flagos device equals the CPU reference."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, dtype):
        host = (torch.randn(4, 64) * 3.0).to(dtype)
        _assert_matches_cpu(
            torch.nn.functional.silu(_to_device(host)), _reference(host)
        )

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 5),  # 20-byte rows
            (3, 7),
            (5, 4),  # rows already a multiple of 16 bytes
            (2, 3, 5, 7),
            (4096,),
            (1, 1),
        ],
    )
    @pytest.mark.anyplatform
    def test_shape_independence(self, shape):
        host = torch.randn(*shape)
        _assert_matches_cpu(
            torch.nn.functional.silu(_to_device(host)), _reference(host)
        )

    @pytest.mark.anyplatform
    def test_swiglu_shaped_operand(self):
        """The shape the op is on the hot path for.

        A 0.6B-class SwiGLU MLP projects to an intermediate width of 3072, so a
        layer's activation is (batch, seq, 3072) -- three times the hidden size
        the matmuls around it work in.
        """
        host = torch.randn(1, 512, 3072)
        _assert_matches_cpu(
            torch.nn.functional.silu(_to_device(host)), _reference(host)
        )

    @pytest.mark.anyplatform
    def test_saturated_tail_is_not_approximated(self):
        """The region where a ``sigmoid`` difference would show up as a number.

        For a strongly negative x the activation is the product of a large
        magnitude and a ``sigmoid`` that has underflowed, so the result is a tiny
        denormal-ish value whose last bits are decided by how the sigmoid rounded.
        A composition that used a less accurate sigmoid reads as a large relative
        error here while agreeing on the near-zero region, so both ends are held.
        """
        host = torch.cat(
            [
                torch.linspace(-40.0, -10.0, 512),
                torch.linspace(10.0, 40.0, 512),
                torch.linspace(-1.0, 1.0, 512),
            ]
        )
        _assert_matches_cpu(
            torch.nn.functional.silu(_to_device(host)), _reference(host)
        )

    @pytest.mark.anyplatform
    def test_negative_tail_keeps_its_sign_bit(self):
        """A negative operand in the saturated tail stays negative.

        The sign bit is the whole difference between the reference and a
        composition that cancelled a negative factor, so it is checked directly
        rather than through ``assert_close``, which does not distinguish ``-0.0``
        from ``0.0``. The magnitude is a separate matter: at -40 the tail has only
        underflowed as far as -1.7e-16, and the deeper values are the ones whose
        ``sigmoid`` is fully zero, so the two regions are held separately rather
        than by one "is it zero" assertion that would have been wrong for the
        first element.
        """
        host = torch.tensor([-40.0, -100.0, -1e30])
        got = torch.nn.functional.silu(_to_device(host)).cpu()
        expected = _reference(host)
        assert got.signbit().tolist() == expected.signbit().tolist()
        assert got[0].item() < 0  # small and negative, not a zero
        assert (got[1:] == 0).all() and (expected[1:] == 0).all()

    @pytest.mark.anyplatform
    def test_infinities_and_their_nan(self):
        """``silu(-inf)`` is ``nan`` on both sides, and a finite operand is not.

        ``sigmoid(-inf)`` is 0 and ``-inf * 0`` is nan, so the reference produces
        nan here rather than the 0 an "obvious" activation would. The kernel
        declines nothing about it, so this is the graph's own arithmetic.
        """
        host = torch.tensor([float("inf"), float("-inf"), 0.0])
        out = torch.nn.functional.silu(_to_device(host)).cpu()
        expected = _reference(host)
        assert torch.isnan(out[1]) and torch.isnan(expected[1])
        assert out[0].item() == float("inf")
        assert out[2].item() == 0.0

    @pytest.mark.anyplatform
    def test_zero_and_tiny_operands(self):
        host = torch.cat(
            [
                torch.zeros(4),
                torch.full((4,), 1e-30),
                torch.full((4,), -1e-30),
                torch.full((4,), 1e-8),
                torch.full((4,), -1e-8),
            ]
        )
        _assert_matches_cpu(
            torch.nn.functional.silu(_to_device(host)), _reference(host)
        )

    @pytest.mark.anyplatform
    def test_rank_zero(self):
        host = torch.tensor(0.5)
        out = torch.nn.functional.silu(_to_device(host))
        assert out.shape == ()
        _assert_matches_cpu(out, _reference(host))

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
        _assert_matches_cpu(torch.nn.functional.silu(a), _reference(host[:, ::2]))

    @pytest.mark.anyplatform
    def test_empty_tensor(self):
        a = _to_device(torch.empty(0, 4))
        out = torch.nn.functional.silu(a)
        assert out.shape == (0, 4)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_aten_spelling_matches(self):
        """``torch.ops.aten.silu`` is the same kernel, not a second path."""
        host = torch.randn(4, 16)
        _assert_matches_cpu(torch.ops.aten.silu(_to_device(host)), _reference(host))


class TestSiluCompositionIsNotTheFormula:
    """The gate was set from the reference, and this pins why that mattered."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_the_formula_alone_would_not_have_been_enough(self, dtype):
        """``x * sigmoid(x)`` on the host is *not* the reference on these widths.

        If it were, comparing the kernel against the formula would be as good a
        test as comparing it against ``at::silu``. It is not: the two readings
        differ by more than the narrow type's own last digit, which is the reason
        the kernel's dtype gate is a measurement of the graph against
        ``at::silu`` rather than a statement that silu equals its definition.
        This case guards the *reference* rather than the kernel -- if a future
        PyTorch made ``silu`` the composition, the note in ``mps_common.h`` would
        be stale and this would say so.
        """
        host = (torch.randn(4096) * 6.0).to(dtype)
        formula = host * torch.sigmoid(host)
        reference = _reference(host)
        assert not torch.equal(formula, reference), (
            "silu now equals x * sigmoid(x) on this build; the measurement the "
            "MPS silu gate records in mps_common.h should be re-taken"
        )


class TestSiluBackwardCorrectness:
    """The derivative stays on MPS and equals the reference."""

    @pytest.mark.parametrize("dtype", NATIVE_BACKWARD_DTYPES)
    @pytest.mark.anyplatform
    def test_autograd_matches_cpu(self, dtype):
        """A forward and backward pass through ``F.silu`` use the native pair."""
        host = (torch.randn(4, 64) * 3.0).to(dtype)
        expected_input = host.detach().clone().requires_grad_()
        (expected,) = torch.autograd.grad(
            torch.nn.functional.silu(expected_input).sum(),
            expected_input,
        )

        operand = _to_device(host).requires_grad_()
        (got,) = torch.autograd.grad(torch.nn.functional.silu(operand).sum(), operand)
        assert operand.grad is None
        _assert_matches_cpu(got, expected)

    @pytest.mark.parametrize("dtype", NATIVE_BACKWARD_DTYPES)
    @pytest.mark.anyplatform
    def test_direct_aten_call_matches_cpu(self, dtype):
        grad_host = (torch.randn(3, 7) * 2.0).to(dtype)
        host = (torch.randn(3, 7) * 4.0).to(dtype)
        _assert_matches_cpu(
            torch.ops.aten.silu_backward(
                _to_device(grad_host),
                _to_device(host),
            ),
            _backward_reference(grad_host, host),
        )

    @pytest.mark.anyplatform
    def test_broadcasting_and_rank_zero(self):
        """The node broadcasts both operands into the reference result shape."""
        grad_host = torch.randn(4, 1, 5)
        host = torch.randn(3, 1)
        _assert_matches_cpu(
            torch.ops.aten.silu_backward(
                _to_device(grad_host),
                _to_device(host),
            ),
            _backward_reference(grad_host, host),
        )
        scalar_grad = torch.tensor(0.5)
        scalar_host = torch.tensor(2.0)
        out = torch.ops.aten.silu_backward(
            _to_device(scalar_grad),
            _to_device(scalar_host),
        )
        assert out.shape == ()
        _assert_matches_cpu(out, _backward_reference(scalar_grad, scalar_host))

    @pytest.mark.anyplatform
    def test_large_positive_operand_does_not_overflow(self):
        """The derivative settles at one; an ``exp``-based graph could overflow."""
        host = torch.tensor([20.0, 40.0, 80.0, 100.0])
        grad_host = torch.tensor([1.0, 0.0, -1.0, 2.0])
        got = torch.ops.aten.silu_backward(
            _to_device(grad_host),
            _to_device(host),
        )
        expected = _backward_reference(grad_host, host)
        _assert_matches_cpu(got, expected)
        assert torch.isfinite(got.cpu()).all()
        assert expected[1:].tolist() == [0.0, -1.0, 2.0]

    @pytest.mark.anyplatform
    def test_edges_follow_the_reference(self):
        """Zeros, infinities and NaN are compared by their reference semantics."""
        host = torch.tensor(
            [0.0, -0.0, float("inf"), float("-inf"), float("nan"), -1e30]
        )
        grad_host = torch.ones_like(host)
        got = torch.ops.aten.silu_backward(
            _to_device(grad_host),
            _to_device(host),
        ).cpu()
        expected = _backward_reference(grad_host, host)
        finite = torch.isfinite(expected)
        assert got.shape == expected.shape
        assert got.dtype == expected.dtype
        assert torch.equal(torch.isnan(got), torch.isnan(expected))
        torch.testing.assert_close(got[finite], expected[finite], **TOLERANCE)

    @pytest.mark.anyplatform
    def test_strided_operands_are_materialised(self):
        host = torch.randn(6, 8)
        grad_host = torch.randn(3, 8)
        got = torch.ops.aten.silu_backward(
            _to_device(grad_host)[:, ::2],
            _to_device(host)[::2, ::2],
        )
        _assert_matches_cpu(
            got,
            _backward_reference(grad_host[:, ::2], host[::2, ::2]),
        )

    @pytest.mark.anyplatform
    def test_empty_operands(self):
        host = torch.empty(0, 4)
        grad_host = torch.empty(1, 4)
        out = torch.ops.aten.silu_backward(
            _to_device(grad_host),
            _to_device(host),
        )
        assert out.shape == (0, 4)
        assert out.device.type == "flagos"


class TestSiluBackwardDeclinedInputs:
    """The reference owns the answers for cases the graph cannot share a dtype."""

    @pytest.mark.parametrize(
        ("grad_dtype", "self_dtype"),
        [
            (torch.float64, torch.float64),
            (torch.float16, torch.float32),
            (torch.bfloat16, torch.bfloat16),
            (torch.float32, torch.bfloat16),
            (torch.int64, torch.int64),
            (torch.bool, torch.bool),
        ],
    )
    @pytest.mark.anyplatform
    def test_declined_pairs_follow_the_reference(self, grad_dtype, self_dtype):
        grad_host = torch.tensor([1, 2, 3], dtype=grad_dtype)
        host = torch.tensor([-1, 0, 1], dtype=self_dtype)
        expected_error = None
        try:
            expected = _backward_reference(grad_host, host)
        except Exception as error:
            expected_error = error

        if expected_error is not None:
            with pytest.raises(type(expected_error)):
                torch.ops.aten.silu_backward(
                    _to_device(grad_host),
                    _to_device(host),
                )
            return
        out = torch.ops.aten.silu_backward(
            _to_device(grad_host),
            _to_device(host),
        )
        assert out.dtype == expected.dtype
        torch.testing.assert_close(
            out.cpu().float(),
            expected.float(),
            **_tolerance(self_dtype),
        )

    @pytest.mark.anyplatform
    def test_non_broadcastable_pair_raises_like_cpu(self):
        grad_host = torch.randn(3)
        host = torch.randn(4)
        try:
            _backward_reference(grad_host, host)
        except RuntimeError:
            pass
        else:
            pytest.fail("the CPU reference accepted incompatible shapes")
        with pytest.raises(RuntimeError):
            torch.ops.aten.silu_backward(
                _to_device(grad_host),
                _to_device(host),
            )


class TestSiluScheduleAndPlanReuse:
    """The marshalling contract every MPS kernel shares (mps_common.mm)."""

    @pytest.mark.anyplatform
    def test_interleaved_dtypes_and_shapes_never_share_a_plan(self):
        """A dtype or shape change on a warm cache still gets its own graph.

        ``silu`` has no host constant, so its plan key is the op name alone and
        the operand signature is folded in by ``MpsRunGraph``. A key that was
        *only* the op name -- which is what the code reads like before that fold
        is accounted for -- would let a float32 (4, 6) graph answer a bfloat16
        (6, 4) request. Rotating the cases puts a different dtype and a different
        shape on a warm cache every step.
        """
        cases = [
            torch.randn(4, 6),
            torch.randn(4, 6).half(),
            torch.randn(6, 4),
            torch.randn(4, 6).bfloat16(),
            torch.randn(2, 3, 5),
            torch.randn(4, 6),
            torch.randn(6, 4).half(),
        ]
        on_device = [_to_device(case) for case in cases]
        wanted = [_reference(case) for case in cases]

        for round_ in range(3):
            for offset in range(len(cases)):
                index = (offset + round_) % len(cases)
                out = torch.nn.functional.silu(on_device[index])
                assert out.dtype == wanted[index].dtype
                assert out.shape == wanted[index].shape
                torch.testing.assert_close(
                    out.cpu().float(), wanted[index].float(), **_tolerance(out.dtype)
                )

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
            wanted = _reference(src.cpu())
            out = torch.nn.functional.silu(src)
            del src  # recycled while the graph above may still be reading it
            for _ in range(4):  # blocks the allocator is free to hand back out
                torch.nn.functional.silu(torch.randn(1 << 12, device=DEVICE))
            held.append((out.cpu(), wanted))  # read back without a synchronize

        for got, wanted in held:
            torch.testing.assert_close(got, wanted, **TOLERANCE)

    @pytest.mark.anyplatform
    def test_repeated_calls_on_one_operand_are_stable(self):
        """The compiled plan is reused across calls and keeps answering."""
        src = torch.randn(64, 64, device=DEVICE)
        wanted = _reference(src.cpu())
        first = torch.nn.functional.silu(src).cpu()
        for _ in range(16):
            torch.testing.assert_close(
                torch.nn.functional.silu(src).cpu(), first, **TOLERANCE
            )
        torch.testing.assert_close(first, wanted, **TOLERANCE)


class TestSiluDeclinedDtypes:
    """The widths the graph does not serve stay correct through the round-trip."""

    @pytest.mark.anyplatform
    def test_float64_is_correct(self):
        host = torch.randn(4, 8, dtype=torch.float64)
        out = torch.nn.functional.silu(_to_device(host))
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.cpu(), _reference(host), **TOLERANCE)

    @pytest.mark.parametrize(
        "dtype", [torch.int64, torch.int32, torch.int16, torch.bool]
    )
    @pytest.mark.anyplatform
    def test_integral_and_bool_raise_exactly_like_cpu(self, dtype):
        """No integer ``silu`` on any backend; the round-trip raises the same way."""
        host = torch.tensor([1, 2, 3], dtype=dtype)
        with pytest.raises(RuntimeError):
            torch.nn.functional.silu(_to_device(host))


class TestSiluDispatch:
    """``silu`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_silu_to_mps(self):
        assert routed_backend_or_none("silu") == "mps"

    @pytest.mark.main_ops
    def test_conf_leaves_the_out_spelling_on_the_fallback(self):
        """The ``out=`` form is deliberately unregistered, like every `.out`."""
        assert routed_backend_or_none("silu.out") is None

    @pytest.mark.main_ops
    def test_conf_leaves_the_inplace_spelling_on_the_fallback(self):
        assert routed_backend_or_none("silu_") is None

    @pytest.mark.main_ops
    def test_conf_routes_the_functional_backward_to_mps(self):
        assert routed_backend_or_none("silu_backward") == "mps"

    @pytest.mark.main_ops
    def test_conf_leaves_the_backward_out_spelling_on_the_fallback(self):
        assert routed_backend_or_none("silu_backward.grad_input") is None

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps_for_both_spellings(self):
        """``F.silu`` and ``aten.silu`` both reach the kernel.

        The public spelling is the one that matters to a caller, and the aten
        one is what a decomposition would lower to; both are held so a future
        change that made the public form composite would show up here. Run in a
        subprocess because ``LogEnabled`` latches the environment on first use.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from device_init import torch, torch_fl; "
                "a = torch.randn(4, 8, device='flagos:0'); "
                "torch.nn.functional.silu(a); torch.ops.aten.silu(a)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr.count("[flagos dispatch] silu -> mps") == 2

    @pytest.mark.main_ops
    def test_autograd_and_aten_backward_keep_their_reference_answer(self, tmp_path):
        """Both the autograd path and an explicit call reach ``silu_backward``."""
        if not torch.backends.mps.is_available():
            pytest.skip("torch.backends.mps is unavailable in this torch build")
        # Once torch_fl claims PrivateUse1, PyTorch 2.10's autograd engine asserts
        # in the native MPS process. Use the installed backend without exposing
        # the checkout here; the process boundary is pinned in the runtime suite.
        env = _child_env(tmp_path)
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from device_init import torch, torch_fl; "
                "a = torch.randn(4, 8, device='flagos:0', requires_grad=True); "
                "torch.nn.functional.silu(a).sum().backward(); "
                "torch.ops.aten.silu_backward("
                "torch.ones(4, 8, device='flagos:0'), a.detach())",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr.count("[flagos dispatch] silu_backward -> mps") == 2

    @pytest.mark.main_ops
    def test_declined_dtype_does_not_reach_the_kernel(self):
        """A float64 call logs no dispatch: the kernel declines before the graph.

        The decline is what keeps the reference's own dtype and error, so a case
        that ran the graph anyway would be a wrong answer rather than a slow one.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from device_init import torch, torch_fl; "
                "a = torch.randn(4, 8, dtype=torch.float64, device='flagos:0'); "
                "print(torch.nn.functional.silu(a).dtype)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "torch.float64" in result.stdout
        # `silu` still logs once -- the kernel is entered and declines -- but the
        # answer comes back in float64 rather than being computed by the graph.
        assert "[flagos dispatch] silu -> mps" in result.stderr
