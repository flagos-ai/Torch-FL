# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare NVIDIA FP32 division with the CUDA PyTorch reference."""

import pytest
import torch

import torch_fl  # noqa: F401

DEVICE = "flagos:0"


@pytest.mark.parametrize("operand", [127.5, -127.5, 7.0])
@pytest.mark.parametrize("overload", ["Scalar", "Tensor"])
def test_fp32_host_scalar_matches_cuda_bits(operand, overload):
    if not torch.version.cuda or not torch.cuda.is_available():
        pytest.skip("This parity test requires CUDA PyTorch")
    if overload == "Tensor":
        operand = torch.tensor(operand, dtype=torch.float64)
    cpu = torch.arange(-255, 257, dtype=torch.float32).reshape(16, 32).T
    divide = getattr(torch.ops.aten.div, overload)
    divide_inplace = getattr(torch.ops.aten.div_, overload)
    cuda = cpu.to("cuda:0")
    flagos = cpu.to(DEVICE)
    expected = divide(cuda, operand).cpu()
    actual = divide(flagos, operand).cpu()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert divide_inplace(flagos, operand) is flagos
    divide_inplace(cuda, operand)
    assert torch.equal(flagos.cpu().view(torch.int32), cuda.cpu().view(torch.int32))
