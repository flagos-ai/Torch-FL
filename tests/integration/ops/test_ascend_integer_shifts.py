"""Circuit index bit operations execute on device with ATen semantics."""

import pytest
import torch_fl  # noqa: F401
import torch


pytestmark = pytest.mark.ascend


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize("direction", ["left", "right"])
def test_shift_tensor_scalar_and_python_aliases(dtype, direction):
    values = torch.tensor([[-8, -1, 0, 1, 7], [4, 3, -2, 2, 6]], dtype=dtype)
    amounts = torch.tensor([0, 1, 2, 3, 4], dtype=dtype)
    tensor, shifts = values.to("flagos:0"), amounts.to("flagos:0")
    function = (
        torch.bitwise_left_shift if direction == "left" else torch.bitwise_right_shift
    )
    for other, reference_other in [(shifts, amounts), (1, 1)]:
        expected = function(values, reference_other)
        actual = function(tensor, other)
        assert actual.device == tensor.device
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
        alias = tensor << other if direction == "left" else tensor >> other
        torch.testing.assert_close(alias.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("direction", ["left", "right"])
def test_shift_counts_outside_element_width(direction):
    values = torch.tensor([-8, -1, 0, 1, 7, 8], dtype=torch.int64)
    amounts = torch.tensor([-1, 0, 1, 63, 64, 128], dtype=torch.int64)
    function = (
        torch.bitwise_left_shift if direction == "left" else torch.bitwise_right_shift
    )
    actual = function(values.to("flagos:0"), amounts.to("flagos:0"))
    torch.testing.assert_close(actual.cpu(), function(values, amounts), atol=0, rtol=0)
