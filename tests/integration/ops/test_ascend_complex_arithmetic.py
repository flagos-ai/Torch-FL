"""Compare complex forward and conjugate-Wirtinger gradients with PyTorch CPU."""

import pytest
import torch_fl  # noqa: F401
import torch


pytestmark = pytest.mark.ascend


def _operation(name, tensor):
    if name == "mul":
        return tensor * (tensor + 0.2j)
    if name == "div":
        return tensor / (tensor + 2 + 0.3j)
    if name == "sum":
        return tensor.sum(dim=-1)
    if name == "cat":
        return torch.cat([tensor, tensor], dim=1)
    if name == "stack":
        return torch.stack([tensor, tensor], dim=1)
    if name == "matmul":
        return tensor @ tensor.mH
    if name == "flip":
        return tensor.flip(-1)
    return getattr(tensor, name)()


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("conjugated", [False, True])
@pytest.mark.parametrize(
    "operation",
    ["mul", "div", "sum", "cat", "stack", "matmul", "flip", "abs", "sin", "cos", "exp"],
)
def test_complex_forward_and_backward(dtype, conjugated, operation):
    generator = torch.Generator().manual_seed(19)
    reference = (
        torch.randn((2, 3), dtype=dtype, generator=generator) * 0.3
    ).requires_grad_()
    tensor = reference.detach().to("flagos:0").requires_grad_()
    expected = _operation(operation, reference.conj() if conjugated else reference)
    actual = _operation(operation, tensor.conj() if conjugated else tensor)
    assert actual.device == tensor.device
    tolerance = 2e-5 if dtype == torch.complex64 else 1e-11
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    expected_loss = (
        (expected.real.square() + expected.imag.square()).sum()
        if expected.is_complex()
        else expected.square().sum()
    )
    actual_loss = (
        (actual.real.square() + actual.imag.square()).sum()
        if actual.is_complex()
        else actual.square().sum()
    )
    expected_loss.backward()
    actual_loss.backward()
    assert tensor.grad.device == tensor.device
    torch.testing.assert_close(
        tensor.grad.cpu(), reference.grad, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "shape",
    [
        ((), ()),
        ((3,), (3,)),
        ((2, 3), (3, 4)),
        ((2, 1, 3, 4), (1, 5, 4, 2)),
        ((2, 0), (0, 3)),
    ],
)
def test_complex_matmul_shapes(dtype, shape):
    left_shape, right_shape = shape
    if not left_shape:
        with pytest.raises(RuntimeError):
            torch.ones((), dtype=dtype, device="flagos:0") @ torch.ones(
                (), dtype=dtype, device="flagos:0"
            )
        return
    generator = torch.Generator().manual_seed(7)
    left = torch.randn(left_shape, dtype=dtype, generator=generator)
    right = torch.randn(right_shape, dtype=dtype, generator=generator)
    expected = left @ right
    actual = left.to("flagos:0") @ right.to("flagos:0")
    tolerance = 2e-5 if dtype == torch.complex64 else 1e-11
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)


def test_bmm_does_not_broadcast_batch_dimensions():
    left = torch.ones((2, 3, 4), dtype=torch.complex64, device="flagos:0")
    right = torch.ones((1, 4, 5), dtype=torch.complex64, device="flagos:0")
    with pytest.raises(RuntimeError):
        torch.bmm(left, right)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "left_shape,right_shape",
    [
        ((3,), (3,)),
        ((3,), (3, 2)),
        ((2, 3), (3,)),
        ((2, 3), (4, 3, 2)),
        ((4, 2, 3), (3, 2)),
        ((2, 1, 3, 4), (1, 5, 4, 2)),
    ],
)
def test_complex_matmul_broadcast_and_vector_gradients(dtype, left_shape, right_shape):
    generator = torch.Generator().manual_seed(41)
    left = torch.randn(left_shape, dtype=dtype, generator=generator, requires_grad=True)
    right = torch.randn(
        right_shape, dtype=dtype, generator=generator, requires_grad=True
    )
    device_left = left.detach().to("flagos:0").requires_grad_()
    device_right = right.detach().to("flagos:0").requires_grad_()
    expected = left @ right
    actual = device_left @ device_right
    upstream = torch.randn(expected.shape, dtype=dtype, generator=generator)
    expected.backward(upstream)
    actual.backward(upstream.to("flagos:0"))
    tolerance = 2e-5 if dtype == torch.complex64 else 1e-11
    torch.testing.assert_close(
        device_left.grad.cpu(), left.grad, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        device_right.grad.cpu(), right.grad, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complex_constructor_and_real_complex_cast_gradients(dtype):
    reference = torch.tensor([0.25, -0.5], dtype=dtype, requires_grad=True)
    tensor = reference.detach().to("flagos:0").requires_grad_()
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    expected = torch.complex(reference, reference.sin()) + reference.to(complex_dtype)
    actual = torch.complex(tensor, tensor.sin()) + tensor.to(complex_dtype)
    expected.real.sum().backward()
    actual.real.sum().backward()
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(tensor.grad.cpu(), reference.grad)
