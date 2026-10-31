"""Logit domain, clamping and backward match PyTorch without host fallback."""

import pytest
import torch_fl  # noqa: F401
import torch

pytestmark = pytest.mark.ascend


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("eps", [None, -0.1, 0.0, 0.1, 0.6])
def test_logit_forward_out_inplace_and_gradient(dtype, eps):
    reference = torch.tensor(
        [-0.2, 0.0, 0.2, 0.5, 0.9, 1.0, 1.2, float("nan")],
        dtype=dtype,
        requires_grad=True,
    )
    tensor = reference.detach().to("flagos:0").requires_grad_()
    expected = torch.logit(reference, eps=eps)
    actual = torch.logit(tensor, eps=eps)
    tolerance = 3e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(
        actual.cpu(), expected, atol=tolerance, rtol=tolerance, equal_nan=True
    )
    expected.backward(torch.ones_like(expected))
    actual.backward(torch.ones_like(actual))
    torch.testing.assert_close(
        tensor.grad.cpu(),
        reference.grad,
        atol=tolerance,
        rtol=tolerance,
        equal_nan=True,
    )
    output = torch.empty(8, dtype=dtype, device=tensor.device)
    torch.logit(tensor.detach(), eps=eps, out=output)
    torch.testing.assert_close(
        output.cpu(), expected, atol=tolerance, rtol=tolerance, equal_nan=True
    )
    inplace = tensor.detach().clone()
    inplace.logit_(eps=eps)
    torch.testing.assert_close(
        inplace.cpu(), expected, atol=tolerance, rtol=tolerance, equal_nan=True
    )
    grad_output = torch.ones_like(tensor)
    torch.ops.aten.logit_backward.grad_input(
        grad_output, tensor.detach(), eps, grad_input=output
    )
    torch.testing.assert_close(
        output.cpu(), reference.grad, atol=tolerance, rtol=tolerance, equal_nan=True
    )


def test_logit_integer_input_promotes_and_inplace_rejects():
    tensor = torch.tensor([0, 1], dtype=torch.int32, device="flagos:0")
    torch.testing.assert_close(torch.logit(tensor).cpu(), torch.logit(tensor.cpu()))
    with pytest.raises(RuntimeError):
        tensor.logit_()
