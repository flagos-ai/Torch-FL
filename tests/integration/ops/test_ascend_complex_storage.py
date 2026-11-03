"""Complex storage views and copies retain device residency and aliasing."""

import pytest
import torch_fl  # noqa: F401
import torch


pytestmark = pytest.mark.ascend


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_real_view_aliases_complex_storage(dtype):
    reference = torch.tensor([1 + 2j, -3 + 4j], dtype=dtype)
    tensor = reference.to("flagos:0")
    parts = torch.view_as_real(tensor)
    assert parts.device == tensor.device
    assert parts.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
    parts[:, 1].add_(2)
    expected = reference + 2j
    torch.testing.assert_close(tensor.cpu(), expected, atol=0, rtol=0)
    restored = torch.view_as_complex(parts)
    assert restored.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
    assert restored.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("layout", ["transpose", "offset", "broadcast", "empty"])
def test_strided_complex_copy_matches_reference(dtype, layout):
    reference = torch.complex(
        torch.arange(24, dtype=torch.float64).reshape(4, 6),
        -torch.arange(24, dtype=torch.float64).reshape(4, 6) - 0.5,
    ).to(dtype)
    tensor = reference.to("flagos:0")
    if layout == "transpose":
        source, expected = tensor.T, reference.T
    elif layout == "offset":
        source, expected = tensor[1:, 1::2], reference[1:, 1::2]
    elif layout == "broadcast":
        source, expected = tensor[:1, :].expand(5, -1), reference[:1, :].expand(5, -1)
    else:
        source, expected = tensor[:0, :], reference[:0, :]
    result = source.contiguous()
    assert result.device == tensor.device
    assert result.dtype == dtype
    torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
    # Also exercise a strided destination, rather than only contiguous output.
    destination = torch.empty(
        (result.shape[1], result.shape[0]), dtype=dtype, device=tensor.device
    ).T
    destination.copy_(source)
    torch.testing.assert_close(destination.cpu(), expected, atol=0, rtol=0)


def test_complex_view_rejects_invalid_last_stride():
    tensor = torch.empty((4, 2), device="flagos:0")[:, ::2]
    with pytest.raises(RuntimeError):
        torch.view_as_complex(tensor)


def test_conjugate_is_a_lazy_storage_alias():
    tensor = torch.tensor([1 + 2j], device="flagos:0")
    conjugate = tensor.conj()
    assert conjugate.is_conj()
    assert conjugate.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_negation_of_offset_real_view(dtype):
    reference = torch.tensor([[1 + 2j, 3 - 1j], [-0.5 + 0.7j, 0.1 - 0.2j]], dtype=dtype)
    tensor = reference.to("flagos:0")
    assert tensor.imag.storage_offset() == 1
    actual = -tensor.imag
    torch.testing.assert_close(actual.cpu(), -reference.imag, atol=0, rtol=0)
    torch.testing.assert_close(tensor.cpu(), reference, atol=0, rtol=0)


def test_power_one_preserves_two_dimensional_component_view():
    reference = torch.tensor([[1 + 2j, 3 - 1j], [-0.5 + 0.7j, 0.1 - 0.2j]])
    tensor = reference.to("flagos:0")
    torch.testing.assert_close(tensor.real.pow(1).cpu(), reference.real)
    torch.testing.assert_close(tensor.imag.pow(1).cpu(), reference.imag)


def test_real_imag_absolute_value_loss_gradients():
    reference = torch.tensor(
        [[1 + 2j, 3 - 1j], [-0.5 + 0.7j, 0.1 - 0.2j]], requires_grad=True
    )
    tensor = reference.detach().to("flagos:0").requires_grad_()
    expected_loss = (
        reference.real.abs().square() + reference.imag.abs().square()
    ).sum()
    actual_loss = (tensor.real.abs().square() + tensor.imag.abs().square()).sum()
    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(tensor.grad.cpu(), reference.grad)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
@pytest.mark.parametrize(
    "permutation",
    [(0, 8, 3, 1, 2, 4, 5, 6, 7, 9, 10, 11), (0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11)],
)
def test_high_rank_permuted_copy_matches_cpu(dtype, permutation):
    reference = torch.arange(2**12, dtype=torch.float64).reshape((2,) * 12).to(dtype)
    if dtype.is_complex:
        reference = reference + 0.25j * reference
    expected = reference.permute(permutation).contiguous()
    actual = reference.to("flagos:0").permute(permutation).contiguous()
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("dim", [0, 4, -1])
def test_high_rank_complex_stack_preserves_values_and_gradient(dtype, dim):
    reference = torch.randn(
        (2,) * 8, dtype=dtype, generator=torch.Generator().manual_seed(103)
    )
    tensor = reference.to("flagos:0").requires_grad_()
    actual = torch.stack((tensor, tensor.conj()), dim=dim)
    expected = torch.stack((reference, reference.conj()), dim=dim)
    torch.testing.assert_close(actual.cpu(), expected)
    actual.abs().square().sum().backward()
    torch.testing.assert_close(tensor.grad.cpu(), 4 * reference)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_high_rank_strided_fill_and_zero(dtype):
    reference = torch.arange(2**10, dtype=torch.float64).reshape((2,) * 10).to(dtype)
    actual = reference.to("flagos:0").transpose(0, 8)
    reference = reference.transpose(0, 8)
    value = 2.5 + 1.5j if dtype.is_complex else 2.5
    actual.fill_(value)
    reference.fill_(value)
    torch.testing.assert_close(actual.cpu(), reference, atol=0, rtol=0)
    actual.zero_()
    reference.zero_()
    torch.testing.assert_close(actual.cpu(), reference, atol=0, rtol=0)
