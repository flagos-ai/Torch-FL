"""Complex and FP64 norms retain magnitude, precision and gradient semantics."""

import pytest
import torch_fl  # noqa: F401
import torch


pytestmark = pytest.mark.ascend


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("order", [0, 1, 2, float("inf"), -2, -float("inf")])
@pytest.mark.parametrize("keepdim", [False, True])
def test_norm_matches_cpu(dtype, order, keepdim):
    generator = torch.Generator().manual_seed(43)
    reference = torch.randn((2, 3), dtype=dtype, generator=generator)
    actual = torch.linalg.vector_norm(
        reference.to("flagos:0"), ord=order, dim=-1, keepdim=keepdim
    )
    expected = torch.linalg.vector_norm(reference, ord=order, dim=-1, keepdim=keepdim)
    assert actual.device.type == "flagos"
    assert actual.dtype == expected.dtype
    tolerance = 2e-5 if dtype == torch.complex64 else 1e-11
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex64, torch.complex128])
def test_norm_gradient_matches_cpu(dtype):
    generator = torch.Generator().manual_seed(53)
    reference = torch.randn(
        (2, 3), dtype=dtype, generator=generator, requires_grad=True
    )
    tensor = reference.detach().to("flagos:0").requires_grad_()
    torch.linalg.vector_norm(reference, dim=-1).sum().backward()
    torch.linalg.vector_norm(tensor, dim=-1).sum().backward()
    tolerance = 2e-5 if dtype == torch.complex64 else 1e-11
    torch.testing.assert_close(
        tensor.grad.cpu(), reference.grad, atol=tolerance, rtol=tolerance
    )


def test_norm_does_not_discard_imaginary_or_demote_fp64_components():
    reference = torch.tensor([3 + 4j, 5 - 12j], dtype=torch.complex128)
    actual = torch.linalg.vector_norm(reference.to("flagos:0"))
    torch.testing.assert_close(
        actual.cpu(), torch.linalg.vector_norm(reference), atol=1e-11, rtol=1e-11
    )


def test_norm_refuses_dtype_narrowing():
    tensor = torch.ones(3, device="flagos:0", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        torch.linalg.vector_norm(tensor, dtype=torch.float32)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_empty_norm_matches_cpu(shape):
    tensor = torch.empty(shape, device="flagos:0", dtype=torch.complex128)
    expected = torch.linalg.vector_norm(
        torch.empty(shape, dtype=torch.complex128), dim=-1
    )
    torch.testing.assert_close(torch.linalg.vector_norm(tensor, dim=-1).cpu(), expected)
