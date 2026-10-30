"""Device spectral decomposition and seeded categorical draws preserve ATen semantics."""

import pytest
import torch_fl  # noqa: F401
import torch

pytestmark = pytest.mark.ascend


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
@pytest.mark.parametrize("uplo", ["L", "U"])
@pytest.mark.parametrize("shape", [(4, 4), (2, 4, 4), (0, 0)])
def test_hermitian_spectrum_and_basis(dtype, uplo, shape):
    source = torch.randn(
        shape, dtype=dtype, generator=torch.Generator().manual_seed(97)
    )
    reference = source + source.mH
    actual = reference.to("flagos:0")
    values, vectors = torch.linalg.eigh(actual, UPLO=uplo)
    tolerance = 5e-5 if dtype in {torch.float32, torch.complex64} else 1e-11
    torch.testing.assert_close(
        values.cpu(),
        torch.linalg.eigvalsh(reference, UPLO=uplo),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        ((vectors * values.unsqueeze(-2)) @ vectors.mH).cpu(),
        reference,
        atol=tolerance,
        rtol=tolerance,
    )
    identity = torch.eye(shape[-1], dtype=dtype).expand(shape)
    torch.testing.assert_close(
        (vectors.mH @ vectors).cpu(), identity, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_spectral_gradient_matches_analytic_norm(dtype):
    source = torch.randn(
        (4, 4), dtype=dtype, generator=torch.Generator().manual_seed(101)
    )
    reference = source + source.mH
    actual = reference.to("flagos:0").requires_grad_()
    torch.linalg.eigvalsh(actual).square().sum().backward()
    tolerance = 5e-5 if dtype in {torch.float32, torch.complex64} else 1e-11
    torch.testing.assert_close(
        actual.grad.cpu(), 2 * reference, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("replacement", [False, True])
def test_multinomial_consumes_explicit_generator(dtype, replacement):
    probabilities = torch.tensor([[0.1, 0.2, 0.0, 0.7]], dtype=dtype, device="flagos:0")
    generator = torch.Generator(device="flagos:0").manual_seed(13)
    state = generator.get_state()
    count = 256 if replacement else 3
    first = torch.multinomial(probabilities, count, replacement, generator=generator)
    generator.set_state(state)
    repeated = torch.multinomial(probabilities, count, replacement, generator=generator)
    torch.testing.assert_close(first, repeated, atol=0, rtol=0)
    assert first.dtype == torch.int64 and first.device == probabilities.device
    assert not bool((first == 2).any())
    if not replacement:
        assert first.cpu().unique().numel() == count


@pytest.mark.parametrize(
    "weights", [[-1.0, 2.0], [0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0]]
)
def test_multinomial_rejects_invalid_weights(weights):
    probabilities = torch.tensor(weights, device="flagos:0")
    with pytest.raises(RuntimeError, match="multinomial"):
        torch.multinomial(probabilities, 4, True)
