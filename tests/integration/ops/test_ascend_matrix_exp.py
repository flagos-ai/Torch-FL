"""Matrix exponentials and their Frechet derivatives remain on Ascend."""

import pytest
import torch_fl  # noqa: F401
import torch


pytestmark = pytest.mark.ascend
DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(2, 2), (2, 3, 3), (0, 3, 3), (2, 0, 0)])
@pytest.mark.parametrize("scale", [0.01, 1.0, 8.0])
def test_matrix_exp_matches_cpu(dtype, shape, scale):
    reference = torch.randn(
        shape, dtype=dtype, generator=torch.Generator().manual_seed(79)
    )
    # Anti-Hermitian inputs also test unitary evolution without exponential overflow.
    reference = (reference - reference.mH) * scale
    tensor = reference.to("flagos:0").transpose(-1, -2)
    actual = torch.linalg.matrix_exp(tensor)
    assert actual.device == tensor.device and actual.dtype == dtype
    tolerance = 8e-5 if dtype in {torch.float32, torch.complex64} else 2e-11
    torch.testing.assert_close(
        actual.cpu(),
        torch.linalg.matrix_exp(reference.transpose(-1, -2)),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_exp_gradient_matches_cpu(dtype):
    reference = (
        torch.randn((2, 2), dtype=dtype, generator=torch.Generator().manual_seed(83))
        * 0.2
    ).requires_grad_()
    tensor = reference.detach().to("flagos:0").requires_grad_()
    torch.linalg.matrix_exp(reference).abs().square().sum().backward()
    torch.linalg.matrix_exp(tensor).abs().square().sum().backward()
    tolerance = 6e-5 if dtype in {torch.float32, torch.complex64} else 2e-11
    torch.testing.assert_close(
        tensor.grad.cpu(), reference.grad, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_exp_strided_out(dtype):
    reference = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=dtype)
    tensor = reference.to("flagos:0")
    storage = torch.empty((2, 4), device=tensor.device, dtype=dtype)
    output = storage[:, ::2]
    result = torch.ops.aten.linalg_matrix_exp.out(tensor, out=output)
    assert result is output
    tolerance = 6e-5 if dtype in {torch.float32, torch.complex64} else 2e-11
    torch.testing.assert_close(
        output.cpu(), torch.linalg.matrix_exp(reference), atol=tolerance, rtol=tolerance
    )
