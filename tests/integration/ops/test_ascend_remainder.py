"""Parity arithmetic keeps signs, broadcasting and scalar promotion on device."""

import pytest
import torch_fl  # noqa: F401
import torch

pytestmark = pytest.mark.ascend


@pytest.mark.parametrize(
    "dtype",
    [
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float32,
        torch.float64,
    ],
)
def test_remainder_scalar_and_broadcast_tensor(dtype):
    reference = torch.tensor([-7, -2, -1, 0, 1, 4]).to(dtype).reshape(3, 2)
    tensor = reference.to("flagos:0")
    scalar = tensor.remainder(2)
    torch.testing.assert_close(scalar.cpu(), reference.remainder(2), atol=0, rtol=0)
    divisor = torch.tensor([2, 3], dtype=dtype)
    result = torch.remainder(tensor, divisor.to(tensor.device))
    torch.testing.assert_close(
        result.cpu(), torch.remainder(reference, divisor), atol=0, rtol=0
    )


def test_remainder_fractional_scalar_promotes_integer_input():
    reference = torch.tensor([-7, -2, -1, 0, 1, 4], dtype=torch.int32)
    actual = reference.to("flagos:0").remainder(1.5)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), reference.remainder(1.5), atol=0, rtol=0)
