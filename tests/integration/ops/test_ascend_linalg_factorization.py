"""Device QR/SVD preserve reconstruction, bases, singular values and gradients."""

import pytest
import torch_fl  # noqa: F401
import torch


pytestmark = pytest.mark.ascend
DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(4, 3), (3, 4), (2, 4, 3), (4, 0), (0, 3)])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("operation", ["qr", "svd"])
def test_factorization_contract(dtype, shape, full, operation):
    generator = torch.Generator().manual_seed(61)
    reference = torch.randn(shape, dtype=dtype, generator=generator)
    tensor = reference.to("flagos:0")
    if operation == "qr":
        q, r = torch.linalg.qr(tensor, mode="complete" if full else "reduced")
        reconstructed = q @ r
    else:
        q, singular, vh = torch.linalg.svd(tensor, full_matrices=full)
        k = min(shape[-2:])
        reconstructed = (q[..., :k] * singular.unsqueeze(-2)) @ vh[..., :k, :]
        tolerance = 2e-5 if dtype in {torch.float32, torch.complex64} else 1e-11
        torch.testing.assert_close(
            singular.cpu(),
            torch.linalg.svdvals(reference),
            atol=tolerance,
            rtol=tolerance,
        )
    assert q.device == reconstructed.device == tensor.device
    tolerance = 2e-5 if dtype in {torch.float32, torch.complex64} else 1e-11
    torch.testing.assert_close(
        reconstructed.cpu(), reference, atol=tolerance, rtol=tolerance
    )
    basis = q.cpu()
    expected_identity = torch.eye(basis.shape[-1], dtype=dtype).expand(
        *shape[:-2], basis.shape[-1], basis.shape[-1]
    )
    torch.testing.assert_close(
        basis.mH @ basis, expected_identity, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("operation", ["qr", "svd"])
def test_factorization_gradient_matches_analytic_norm(dtype, operation):
    reference = torch.randn(
        (4, 3), dtype=dtype, generator=torch.Generator().manual_seed(67)
    )
    tensor = reference.to("flagos:0").requires_grad_()
    if operation == "qr":
        q, r = torch.linalg.qr(tensor, mode="reduced")
        reconstructed = q @ r
    else:
        u, singular, vh = torch.linalg.svd(tensor, full_matrices=False)
        reconstructed = (u * singular.unsqueeze(-2)) @ vh
    reconstructed.abs().square().sum().backward()
    assert tensor.grad.device == tensor.device
    tolerance = 4e-5 if dtype in {torch.float32, torch.complex64} else 1e-11
    torch.testing.assert_close(
        tensor.grad.cpu(), 2 * reference, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("left", [False, True])
@pytest.mark.parametrize("unitriangular", [False, True])
def test_triangular_solve_matches_cpu(dtype, upper, left, unitriangular):
    generator = torch.Generator().manual_seed(79)
    matrix = torch.randn((2, 3, 3), dtype=dtype, generator=generator) + 4 * torch.eye(
        3, dtype=dtype
    )
    rhs = torch.randn((3, 2) if left else (2, 3), dtype=dtype, generator=generator)
    expected = torch.linalg.solve_triangular(
        matrix, rhs, upper=upper, left=left, unitriangular=unitriangular
    )
    actual = torch.linalg.solve_triangular(
        matrix.to("flagos:0").conj(),
        rhs.conj().to("flagos:0"),
        upper=upper,
        left=left,
        unitriangular=unitriangular,
    )
    tolerance = 4e-5 if dtype in {torch.float32, torch.complex64} else 1e-11
    torch.testing.assert_close(
        actual.cpu(), expected.conj(), atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", DTYPES + [torch.int64, torch.bool])
@pytest.mark.parametrize("shape", [(3, 3), (3, 5), (0, 4)])
def test_identity_factory_matches_cpu(dtype, shape):
    actual = torch.eye(*shape, dtype=dtype, device="flagos:0")
    torch.testing.assert_close(actual.cpu(), torch.eye(*shape, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("diagonal", [-1, 0, 2])
def test_complex_triangle_and_advanced_index(dtype, diagonal):
    reference = torch.randn(
        (2, 3, 4), dtype=dtype, generator=torch.Generator().manual_seed(83)
    )
    actual = reference.to("flagos:0").conj()
    for operation in [torch.triu, torch.tril]:
        torch.testing.assert_close(
            operation(actual, diagonal).cpu(), operation(reference.conj(), diagonal)
        )
    rows = torch.tensor([0, 1], device="flagos:0")
    columns = torch.tensor([2, 0], device="flagos:0")
    torch.testing.assert_close(
        actual[rows, columns, :].cpu(), reference.conj()[rows.cpu(), columns.cpu(), :]
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_diagonal_fill_preserves_off_diagonal_values(dtype):
    reference = torch.arange(16, dtype=torch.float64).reshape(4, 4).to(dtype)
    actual = reference.to("flagos:0")
    actual.diagonal().fill_(1)
    reference.diagonal().fill_(1)
    torch.testing.assert_close(actual.cpu(), reference)
