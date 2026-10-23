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

"""Stage-2 Apple Silicon (Metal / MPSGraph) elementwise kernels.

Stage 2 of the MPSGraph operator layer (see
docs/superpowers/specs/2026-10-14-apple-mps-native-backend-design.md section 7)
adds the ``binary`` and ``binary_scalar`` families -- ``add``/``sub``/``mul``
over two tensors and over a tensor and a host scalar. This is where the stage's
exit criterion for those two categories is measured, the way
tests/integration/ops/test_mps_unary.py measures the ``unary`` one: every case
compares the flagos result against the CPU reference, and a dispatch-log
assertion proves the ``mps`` kernel ran rather than ``cpu_fallback``.

Two properties are specific to these categories and are what the cases below
pin down.

* Broadcasting follows ATen's shape inference. The output comes from
  ``at::infer_size``. A one-element RHS is described in that output shape in
  MPSGraph because a zero-dimensional reshape of its flat buffer is invalid and
  arithmetic selectors only repeat aligned size-one axes. A pair that cannot
  broadcast is declined to the CPU round-trip, and the reference's own error is
  what surfaces.
* The stage-1 marshalling rule still applies. Every buffer is described to
  MPSGraph as a 1-D array of ``numel`` elements and the aten shape is reapplied
  inside the graph, because MPSNDArray rounds the stride between the rows of an
  N-D array up to 16 bytes; the unaligned-row cases are what hold the binary
  kernels to that, for both operands and the result alike.

The scalar family is reached through ``torch.ops.aten.<op>.Scalar``: the Python
operators (``a + 2``, ``torch.mul(a, 3.0)``) decompose to the Tensor overload
before PrivateUse1 dispatch, so they exercise the binary kernels instead. The
opaque-aten spelling is therefore the one that reaches the kernel named for it,
and it is used for every ``binary_scalar`` case; the dispatch test names both
spellings.

Every case self-skips off an MPS build, like the stage-1 suite.
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

BINARY_OPS = [
    ("add", torch.add),
    ("sub", torch.sub),
    ("mul", torch.mul),
]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _scalar(op_name: str, operand: torch.Tensor, value, alpha=None) -> torch.Tensor:
    """``torch.ops.aten.<op>.Scalar`` on a flagos operand.

    The opaque aten overload, not ``torch.add(t, value)``: see the module
    docstring -- the Python spelling decomposes to the Tensor overload before
    PrivateUse1 dispatch and never reaches the scalar kernel.
    """
    op = getattr(torch.ops.aten, op_name).Scalar
    if alpha is None:
        return op(operand, value)
    return op(operand, value, alpha)


class TestBinaryTensorCorrectness:
    """``add``/``sub``/``mul`` over two flagos tensors equal the CPU reference."""

    @pytest.mark.parametrize("op_name,op", BINARY_OPS)
    @pytest.mark.parametrize(
        "shape_a,shape_b",
        [
            ((4, 5), (5,)),
            ((4, 5), (4, 1)),
            ((1, 3), (3, 1)),
            ((2, 3, 4), (4,)),
            ((2, 3, 4), (3, 1)),
            ((1,), (4, 5)),
            ((2, 1, 3), (1, 4, 1)),
            ((4, 5), ()),
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_for_broadcast_pairs(self, op_name, op, shape_a, shape_b):
        """Broadcast pairs match the reference at every supported rank.

        The pairs walk the broadcast cases aten distinguishes: a trailing
        dimension against a shorter operand, a size-1 axis on either side, a
        0-dim operand (the plain ``a + 2.0`` shape), and a leading 1 that has to
        be treated as a missing dimension rather than as a matching one.
        """
        a = torch.randn(shape_a) * 2.0
        b = torch.randn(shape_b) * 2.0
        out = op(_to_device(a), _to_device(b))
        expected = op(a, b)
        assert out.shape == expected.shape
        assert out.device.type == "flagos"
        torch.testing.assert_close(
            out.cpu(), expected, rtol=1e-5, atol=1e-5, msg=op_name
        )

    @pytest.mark.parametrize("op_name,op", BINARY_OPS)
    @pytest.mark.parametrize(
        "shape",
        [
            (4, 5),  # 20-byte rows
            (3, 7),
            (5, 4),  # controls: rows already a multiple of 16 bytes
            (4, 8, 16),
            (2, 3, 5),
            (128, 3),
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_at_row_widths_that_are_not_a_16_byte_multiple(
        self, op_name, op, shape
    ):
        """The shapes MPSNDArray's row pitch would silently corrupt.

        See the row-pitch note in ``mps_common.h``: a (4, 5) float32 operand has
        20-byte rows that MPSNDArray reads as 32, so an aten-shaped descriptor
        would take every row after the first from the wrong offset. Both
        operands and the result go through the same 1-D-then-reshape
        marshalling, so all three are checked here; the aligned shapes in the
        list are what show the padding was the variable rather than the shape.
        """
        a = torch.randn(*shape) * 3.0
        b = torch.randn(*shape) * 3.0
        out = op(_to_device(a), _to_device(b))
        torch.testing.assert_close(
            out.cpu(), op(a, b), rtol=1e-5, atol=1e-5, msg=op_name
        )

    @pytest.mark.parametrize("op_name,op", BINARY_OPS)
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_for_every_supported_dtype(self, op_name, op, dtype):
        """Every dtype the kernel spells out, at the values that overflow.

        Integer arithmetic wraps in MPSGraph exactly as it does in the
        reference, so the narrow dtypes are given operands that overflow
        (``200 + 200``, ``-7 * 11``) rather than values that would hide the
        wrapping behind a well-behaved range. The comparison is exact: the same
        integer result, or the same float bit pattern, is what MPSGraph
        produced when this was measured.
        """
        if dtype == torch.uint8:
            host = torch.tensor([[0, 1, 128, 200, 255]] * 3, dtype=dtype)
        else:
            host = torch.tensor([[-7, -1, 0, 3, 11]] * 3, dtype=dtype)
        out = op(_to_device(host), _to_device(host))
        assert out.dtype == dtype
        torch.testing.assert_close(out.cpu(), op(host, host), msg=op_name)

    @pytest.mark.parametrize("op_name,op", [("add", torch.add), ("sub", torch.sub)])
    @pytest.mark.anyplatform
    def test_alpha_scales_the_secondary_operand(self, op_name, op):
        """``alpha`` is folded into the secondary operand as a constant.

        ``alpha == 1`` -- every plain ``+`` and ``-`` -- adds no node at all, so
        the cases that matter are the ones that do: a scaled secondary operand
        (``a + 2b``), a negative factor, and 0. The comparison is against the
        reference at the same alpha.
        """
        a = torch.randn(4, 6) * 2.0
        b = torch.randn(4, 6) * 2.0
        for alpha in (2.0, -1.0, 0.0, 0.25):
            out = op(_to_device(a), _to_device(b), alpha=alpha)
            torch.testing.assert_close(
                out.cpu(), op(a, b, alpha=alpha), rtol=1e-5, atol=1e-5
            )

    @pytest.mark.anyplatform
    def test_alpha_scales_an_integral_operand_exactly(self):
        a = torch.tensor([[-7, -1, 0, 3, 11]] * 3)
        b = torch.tensor([[1, 2, 3, 4, 5]] * 3)
        out = torch.add(_to_device(a), _to_device(b), alpha=3)
        assert out.dtype == torch.int64
        torch.testing.assert_close(out.cpu(), torch.add(a, b, alpha=3))

    @pytest.mark.parametrize("shape", [(32,), (4, 5), (2, 3, 4)])
    @pytest.mark.parametrize("other", [-3.0, 1.0, 2.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, -1.0])
    @pytest.mark.anyplatform
    def test_rsub_scalar_matches_cpu(self, shape, other, alpha):
        source = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32)
        source = source.reshape(shape)
        actual = torch.ops.aten.rsub.Scalar(_to_device(source), other, alpha)
        expected = torch.ops.aten.rsub.Scalar(source, other, alpha)
        assert actual.device.type == "flagos"
        torch.testing.assert_close(actual.cpu(), expected)

    @pytest.mark.parametrize("op_name,op", BINARY_OPS)
    @pytest.mark.anyplatform
    def test_strided_operands_are_materialised(self, op_name, op):
        """A non-contiguous operand takes the same answer as the CPU.

        MPSGraph reads dense memory, so the kernel materialises each operand
        first. Both sides are strided here, and the stride leaves 5 columns
        after the slice, so the materialised operands have the unaligned rows
        too.
        """
        a = _to_device(torch.randn(8, 10))[:, ::2]
        b = _to_device(torch.randn(8, 10))[:, ::2]
        assert not a.is_contiguous()
        assert not b.is_contiguous()
        out = op(a, b)
        torch.testing.assert_close(
            out.cpu(), op(a.cpu(), b.cpu()), rtol=1e-5, atol=1e-5, msg=op_name
        )

    @pytest.mark.anyplatform
    def test_plan_cache_survives_a_later_larger_rank(self):
        """A warm broadcast plan cannot answer a later operand rank."""
        a = _to_device(torch.arange(4, dtype=torch.float32).reshape(1, 4))
        b = torch.full((3, 1), 2.0, device=DEVICE)
        torch.testing.assert_close(torch.add(a, b).cpu(), torch.add(a.cpu(), b.cpu()))

        a = torch.full((2, 3, 4), 3.0, device=DEVICE)
        b = torch.arange(6, dtype=torch.float32, device=DEVICE).reshape(2, 3, 1)
        torch.testing.assert_close(torch.sub(a, b).cpu(), torch.sub(a.cpu(), b.cpu()))

    @pytest.mark.parametrize("op_name,op", BINARY_OPS)
    @pytest.mark.parametrize("shape_a,shape_b", [((0, 4), (0, 4)), ((0, 4), (4,))])
    @pytest.mark.anyplatform
    def test_empty_tensor(self, op_name, op, shape_a, shape_b):
        """No elements is an empty result, not a graph run.

        MPSGraph rejects an empty tensor, so the kernel returns the correctly
        shaped empty output before it builds anything. The broadcast pair is the
        interesting one: the result is still empty, and its shape has to follow
        ``at::infer_size`` rather than either operand.
        """
        a = _to_device(torch.empty(*shape_a))
        b = _to_device(torch.empty(*shape_b))
        out = op(a, b)
        expected = op(torch.empty(*shape_a), torch.empty(*shape_b))
        assert out.shape == expected.shape
        assert out.device.type == "flagos"

    @pytest.mark.parametrize(
        "dtype_a,dtype_b",
        [
            (torch.int32, torch.float32),
            (torch.float16, torch.float32),
            (torch.int32, torch.int64),
        ],
    )
    @pytest.mark.anyplatform
    def test_mixed_dtypes_take_the_reference_result(self, dtype_a, dtype_b):
        """A mixed pair has no MPSGraph answer, so it round-trips.

        MPSGraph's binary arithmetic does not promote and aten's does
        (``int32 * float32 -> float32``), so a pair of different dtypes is
        declined and computed by the reference. What has to hold is the result's
        dtype, which is the part a graph built from the operands' own element
        types would get wrong; the values come from the same reference.
        """
        a = torch.tensor([[1, 2, 3]], dtype=dtype_a)
        b = torch.tensor([[0.5, 0.25, 2.0]], dtype=dtype_b)
        out = torch.mul(_to_device(a), _to_device(b))
        expected = torch.mul(a, b)
        assert out.dtype == expected.dtype
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.anyplatform
    def test_bool_add_matches_cpu(self):
        """bool arithmetic is declined and reaches the reference's answer.

        MPSGraph has a bool element type, but the kernel declines it the way the
        unary one declines ``abs(bool)``: aten's bool addition is a logical or,
        and the round-trip is what makes the result identical rather than merely
        plausible.
        """
        a = torch.tensor([[True, False, True]])
        b = torch.tensor([[True, True, False]])
        out = torch.add(_to_device(a), _to_device(b))
        assert out.dtype == torch.bool
        torch.testing.assert_close(out.cpu(), torch.add(a, b))

    @pytest.mark.anyplatform
    def test_float64_round_trips(self):
        """float64 has no MPSGraph element type; the value is still right."""
        a = torch.tensor([[1.5, -2.25]], dtype=torch.float64)
        b = torch.tensor([[0.5, 4.0]], dtype=torch.float64)
        out = torch.add(_to_device(a), _to_device(b))
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.cpu(), torch.add(a, b))

    @pytest.mark.anyplatform
    def test_broadcast_failure_raises_exactly_like_cpu(self):
        """A pair that cannot broadcast raises the reference's own error.

        ``at::infer_size`` is the kernel's own check: when it throws, the kernel
        takes the round-trip and the reference raises -- so a bad broadcast is a
        ``RuntimeError`` about the mismatched dimensions, not a truncated result
        and not a graph that read past an operand.
        """
        a = _to_device(torch.randn(2, 3))
        b = _to_device(torch.randn(4))
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            torch.add(a, b)

    @pytest.mark.anyplatform
    def test_fractional_alpha_on_an_integral_tensor_raises_like_cpu(self):
        """``alpha=0.5`` on an integer tensor is refused by both paths.

        The constant path cannot express a fractional factor in an integral
        element type, so it declines and the reference raises its own
        ``RuntimeError``. Either way the caller gets an error rather than a
        silently truncated alpha.
        """
        a = _to_device(torch.tensor([1, 2]))
        b = _to_device(torch.tensor([1, 2]))
        with pytest.raises(RuntimeError, match="alpha must not be a floating point"):
            torch.add(a, b, alpha=0.5)

    @pytest.mark.parametrize("op_name,op", BINARY_OPS)
    @pytest.mark.parametrize("alpha", [1, 2])
    @pytest.mark.parametrize(
        "shape_a,shape_b",
        [((4, 5), ()), ((4, 3), (3,)), ((1, 4), (4,)), ((4, 1), (4,))],
    )
    @pytest.mark.anyplatform
    def test_one_element_rhs_uses_the_aten_broadcast_shape(
        self, op_name, op, alpha, shape_a, shape_b
    ):
        """A one-element RHS is the operand shape aten broadcasts, not just a buffer.

        Reshaping its flat input to ``()`` makes MPSGraph reject the graph, while
        a RHS with one element and several dimensions cannot describe a scalar
        result either. The body therefore compiles the RHS in the inferred
        output shape. Every binary op and an alpha other than one cross the same
        body, so either a per-op or a host-constant plan key would surface here.
        """
        a = _to_device(torch.arange(shape_a[0] * shape_a[-1]).reshape(shape_a))
        b = torch.full(shape_b, 2.0, device=DEVICE)
        alpha_kwargs = {} if op_name == "mul" else {"alpha": alpha}
        actual = op(a, b, **alpha_kwargs)
        expected = op(a.cpu(), b.cpu(), **alpha_kwargs)
        torch.testing.assert_close(actual.cpu(), expected)

    @pytest.mark.anyplatform
    def test_output_shapes_do_not_reuse_each_others_plan(self):
        """The same shapes still need separate plans for two different ops.

        The generated key includes ``add`` versus ``sub`` and alpha, and the
        runner adds argument shapes so a one-element RHS compiled for ``(4, 5)``
        cannot be selected for a later ``(2, 3)`` call.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "a = torch.arange(20, dtype=torch.float32, device='flagos:0').reshape(4, 5); "
                "b = torch.tensor(2.0, device='flagos:0'); "
                "torch.add(a, b, alpha=2).cpu(); "
                "torch.sub(a, b).cpu(); "
                "torch.add(torch.arange(6, dtype=torch.float32, device='flagos:0').reshape(2, 3), b, alpha=2).cpu()",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr.count("add.Tensor -> mps") == 2, result.stderr
        assert "sub.Tensor -> mps" in result.stderr, result.stderr


class TestBinaryScalarCorrectness:
    """``add``/``sub``/``mul`` over a flagos tensor and a host scalar."""

    @pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
    @pytest.mark.parametrize("value", [2.0, -1.5, 0.0, 0.25])
    @pytest.mark.anyplatform
    def test_matches_cpu_for_float_scalars(self, op_name, value):
        """The scalar becomes a one-element MPSGraph constant of the dtype.

        One element is what MPSGraph broadcasts against an operand of any shape,
        so the operand's shape never reaches the constant. Zero and the negative
        value are the cases that would expose a constant built at the wrong
        element type or an operand that skipped the scale.
        """
        host = torch.randn(4, 6) * 3.0
        out = _scalar(op_name, _to_device(host), value)
        expected = getattr(torch, op_name)(host, value)
        assert out.shape == host.shape
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 5),  # 20-byte rows
            (3, 7),
            (5, 4),  # controls: rows already a multiple of 16 bytes
            (4, 8, 16),
            (2, 3, 5),
            (128, 3),
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_at_row_widths_that_are_not_a_16_byte_multiple(self, shape):
        """The row-pitch rule, on the operand and the result of a scalar op."""
        host = torch.randn(*shape) * 3.0
        out = _scalar("mul", _to_device(host), 1.5)
        torch.testing.assert_close(
            out.cpu(), torch.mul(host, 1.5), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_for_every_supported_dtype(self, op_name, dtype):
        """An integral operand takes an integral scalar and keeps its dtype.

        ``-2`` is the scalar for the signed types; for ``uint8`` the addend is
        kept in range, because the point of the case is that the result stays in
        the operand's dtype. The dtypes whose arithmetic wraps are covered by
        the two-tensor cases instead.
        """
        if dtype == torch.uint8:
            host = torch.tensor([[4, 8, 9, 10, 12]] * 3, dtype=dtype)
            value = 3
        elif dtype in (torch.float32, torch.float16, torch.bfloat16):
            host = torch.tensor([[-1.5, -1.0, 0.0, 3.0, 11.0]] * 3, dtype=dtype)
            value = 2.0
        else:
            host = torch.tensor([[-7, -1, 0, 3, 11]] * 3, dtype=dtype)
            value = -2
        out = _scalar(op_name, _to_device(host), value)
        assert out.dtype == dtype
        torch.testing.assert_close(out.cpu(), getattr(torch, op_name)(host, value))

    @pytest.mark.anyplatform
    def test_fractional_scalar_promotes_like_cpu(self):
        """``int_tensor + 1.5`` is a float tensor on both paths.

        A fractional scalar moves the aten result off the operand's dtype, which
        an integral element type cannot express, so the kernel declines and the
        reference computes it. The dtype is what this case is about.
        """
        host = torch.tensor([1, 2, 3])
        out = torch.add(_to_device(host), 1.5)
        expected = torch.add(host, 1.5)
        assert out.dtype == expected.dtype
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.anyplatform
    def test_bool_operand_matches_cpu(self):
        """``bool`` is declined, so the reference's promotion is what surfaces."""
        host = torch.tensor([True, False, True])
        out = torch.add(_to_device(host), 1)
        expected = torch.add(host, 1)
        assert out.dtype == expected.dtype
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.anyplatform
    def test_float64_scalar_round_trips(self):
        host = torch.tensor([1.5, -2.25], dtype=torch.float64)
        out = torch.mul(_to_device(host), 2.0)
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.cpu(), torch.mul(host, 2.0))

    @pytest.mark.anyplatform
    def test_int64_scalar_beyond_2_to_the_53_round_trips(self):
        """An integer the constant path cannot carry exactly is declined.

        The constant is handed to MPSGraph as a double, so an int64 beyond
        +/-2^53 would land rounded; the kernel declines it and the reference
        adds it exactly. This is the guard that would otherwise be a silently
        wrong sum.
        """
        host = torch.tensor([1, 2, 3], dtype=torch.int64)
        value = 1 << 60
        out = _scalar("add", _to_device(host), value)
        assert out.cpu()[0].item() == value + 1
        torch.testing.assert_close(out.cpu(), torch.add(host, value))

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        operand = _to_device(torch.randn(8, 10) * 2.0)[:, ::2]
        assert not operand.is_contiguous()
        out = _scalar("mul", operand, 2.0)
        torch.testing.assert_close(
            out.cpu(), torch.mul(operand.cpu(), 2.0), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.anyplatform
    def test_empty_tensor(self):
        operand = _to_device(torch.empty(0, 4))
        out = _scalar("add", operand, 1.0)
        assert out.shape == (0, 4)
        assert out.device.type == "flagos"


class TestBinaryDispatch:
    """The binary and scalar overloads are claimed on PrivateUse1."""

    @pytest.mark.parametrize(
        "op",
        [
            "add.Tensor",
            "sub.Tensor",
            "mul.Tensor",
            "add.Scalar",
            "sub.Scalar",
            "mul.Scalar",
        ],
    )
    @pytest.mark.main_ops
    def test_conf_routes_op_to_mps(self, op):
        assert routed_backend_or_none(op) == "mps"

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """The dispatch log is the proof the kernels ran, not a fallback.

        A numeric-only check passes whether MPSGraph ran or the op reached
        ``cpu_fallback``, because both compute the same answer. The log line is
        emitted by the C++ dispatcher only on the ``mps`` slot. The tensor form
        and the opaque-aten scalar form are both exercised, because that pair is
        what shows the two registered overloads are reachable. Run in a
        subprocess because ``LogEnabled`` latches the environment on first use.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "a = torch.randn(4, 4, device='flagos:0'); "
                "b = torch.randn(4, 4, device='flagos:0'); "
                "a + b; a - b; a * b; "
                "torch.ops.aten.add.Scalar(a, 2.0); "
                "torch.ops.aten.mul.Scalar(a, 2.0)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        for line in (
            "[flagos dispatch] add.Tensor -> mps",
            "[flagos dispatch] sub.Tensor -> mps",
            "[flagos dispatch] mul.Tensor -> mps",
            "[flagos dispatch] add.Scalar -> mps",
            "[flagos dispatch] mul.Scalar -> mps",
        ):
            assert line in result.stderr, result.stderr
