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

"""
matmul forward + backward shape/value tests.

On Ascend, aten::matmul is claimed as a fused aclnnMatmul kernel instead of
decomposing into mm/bmm/view. That means autograd binds the op's real
derivative, aten::matmul_backward, which the backend implements by hand -- and
that kernel has to reproduce every shape rule aten::matmul applies in the
forward pass: 1-D promotion, batch broadcasting, and the 2-D/N-D fold.

Those rules are exactly where a hand-written backward goes wrong silently: the
gradient still has a plausible shape, so only a value comparison catches it.
(op-plugin's reference kernel, which this one started from, mixes batches for
2-D x N-D and returns a wrong-shaped gradient for interior broadcasts. Both are
covered below.)

Everything is compared against a float64 CPU reference and scored relatively,
so fp32/hf32 accumulation over a large K is not mistaken for a shape bug.

Usage:
    pytest tests/integration/ops/test_matmul_backward_dispatch.py -v
"""

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"

# Relative to the float64 reference. The Ascend kernel runs with
# ALLOW_FP32_DOWN_PRECISION (hf32) cube math, which lands around 1e-4.
TOL = 2e-3

# (a_shape, b_shape, id) -- one per shape family the kernel branches on.
SHAPE_CASES = [
    ((5,), (5,), "1d_x_1d_dot"),
    ((3, 4), (4,), "2d_x_1d"),
    ((4,), (4, 6), "1d_x_2d"),
    ((3, 4), (4, 6), "2d_x_2d_mm"),
    ((2, 3, 4), (2, 4, 6), "3d_x_3d_bmm"),
    ((2, 5, 3, 4), (2, 5, 4, 6), "4d_x_4d"),
    ((2, 3, 4), (4,), "3d_x_1d"),
    ((4,), (2, 4, 6), "1d_x_3d"),
    ((2, 3, 4, 5), (5,), "4d_x_1d"),
    ((4,), (2, 3, 4, 6), "1d_x_4d"),
    # The 2-D/N-D fold: matmul collapses batch dims into the contraction, and
    # the backward has to reproduce that pairing rather than a plain reshape.
    ((2, 3, 4), (4, 6), "3d_x_2d_fold"),
    ((3, 4), (2, 4, 6), "2d_x_3d_fold"),
    ((1, 128, 1024), (1024, 512), "qwen_like_fold"),
    # Batch broadcasting: the raw gradient is shaped like the *broadcast*
    # operand and must be summed back down. Leading, interior and trailing
    # singletons all have to work, not just a leading prefix.
    ((1, 3, 4), (2, 4, 6), "broadcast_leading_a"),
    ((1, 1, 3, 4), (3, 4, 6), "broadcast_leading_singletons"),
    ((2, 1, 3, 4), (2, 5, 4, 6), "broadcast_interior_a"),
    ((2, 5, 3, 4), (2, 1, 4, 6), "broadcast_interior_b"),
    ((2, 3, 4), (1, 4, 6), "broadcast_leading_b"),
    ((3, 4), (1, 4, 6), "broadcast_2d_x_singleton_batch"),
    ((5, 3, 4), (1, 4, 6), "broadcast_b_batch"),
    ((2, 1, 1, 3, 4), (7, 5, 4, 6), "broadcast_5d_multi"),
    ((7, 5, 3, 4), (1, 1, 4, 6), "broadcast_trailing_singleton"),
]


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Max abs error relative to the reference's magnitude."""
    return (got.double() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-12)


def _run(a_shape, b_shape, requires_grad=(True, True)):
    """matmul fwd+bwd on device and on a float64 CPU reference."""
    torch.manual_seed(0)
    a_ref = torch.randn(a_shape, dtype=torch.float64, requires_grad=requires_grad[0])
    b_ref = torch.randn(b_shape, dtype=torch.float64, requires_grad=requires_grad[1])
    a_dev = a_ref.detach().float().to(DEVICE).requires_grad_(requires_grad[0])
    b_dev = b_ref.detach().float().to(DEVICE).requires_grad_(requires_grad[1])

    out_ref = torch.matmul(a_ref, b_ref)
    out_dev = torch.matmul(a_dev, b_dev)

    torch.manual_seed(1)
    grad = torch.randn(out_ref.shape, dtype=torch.float64)
    out_ref.backward(grad)
    out_dev.backward(grad.float().to(DEVICE))

    return (a_ref, b_ref, out_ref), (a_dev, b_dev, out_dev)


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [(a, b) for a, b, _ in SHAPE_CASES],
    ids=[i for _, _, i in SHAPE_CASES],
)
def test_matmul_forward_backward_matches_cpu(a_shape, b_shape):
    ref, dev = _run(a_shape, b_shape)
    (a_ref, b_ref, out_ref), (a_dev, b_dev, out_dev) = ref, dev

    assert out_dev.shape == out_ref.shape
    # Shape first: a wrong-shaped gradient is a different (and more severe) bug
    # than a wrong-valued one, and asserting it separately says which occurred.
    assert a_dev.grad.shape == a_ref.grad.shape, "grad_self shape mismatch"
    assert b_dev.grad.shape == b_ref.grad.shape, "grad_other shape mismatch"

    assert _rel_err(out_dev.cpu(), out_ref.detach()) < TOL, "forward value mismatch"
    assert _rel_err(a_dev.grad.cpu(), a_ref.grad) < TOL, "grad_self value mismatch"
    assert _rel_err(b_dev.grad.cpu(), b_ref.grad) < TOL, "grad_other value mismatch"


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [((2, 3, 4), (4, 6)), ((3, 4), (2, 4, 6)), ((1, 3, 4), (5, 4, 6))],
)
@pytest.mark.parametrize("side", ["self", "other"])
def test_matmul_backward_grad_input_mask(a_shape, b_shape, side):
    """Only the requested side gets a gradient (exercises grad_input_mask)."""
    mask = (side == "self", side == "other")
    (a_ref, b_ref, _), (a_dev, b_dev, _) = _run(a_shape, b_shape, requires_grad=mask)

    if side == "self":
        assert b_dev.grad is None, "other must not get a gradient"
        assert _rel_err(a_dev.grad.cpu(), a_ref.grad) < TOL
    else:
        assert a_dev.grad is None, "self must not get a gradient"
        assert _rel_err(b_dev.grad.cpu(), b_ref.grad) < TOL


@pytest.mark.anyplatform
def test_matmul_records_autograd_graph():
    """The fused kernel must still build a real autograd node, not detach.

    Claiming a CompositeImplicitAutograd op on PrivateUse1 silently drops the
    graph unless an AutogradPrivateUse1 kernel re-creates it, which is what
    csrc/aten/generated/variable_type.cc exists to do.
    """
    a = torch.randn(2, 3, 4, device=DEVICE, requires_grad=True)
    b = torch.randn(4, 6, device=DEVICE, requires_grad=True)
    out = torch.matmul(a, b)
    assert out.grad_fn is not None, "matmul produced no grad_fn"
    out.sum().backward()
    assert a.grad is not None and b.grad is not None


@pytest.mark.anyplatform
def test_matmul_backward_through_chain():
    """Gradients flow through a matmul that is not the last op in the graph."""
    torch.manual_seed(0)
    w_ref = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    x_ref = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    w_dev = w_ref.detach().float().to(DEVICE).requires_grad_(True)
    x_dev = x_ref.detach().float().to(DEVICE).requires_grad_(True)

    (torch.matmul(x_ref, w_ref) * 2.0).sum().backward()
    (torch.matmul(x_dev, w_dev) * 2.0).sum().backward()

    assert _rel_err(x_dev.grad.cpu(), x_ref.grad) < TOL
    assert _rel_err(w_dev.grad.cpu(), w_ref.grad) < TOL
