"""Regression coverage for CNNL mul with host scalars and strided operands."""

import os

import pytest
import torch
import torch_fl  # noqa: F401

pytestmark = [
    pytest.mark.anyplatform,
    pytest.mark.main_ops,
    pytest.mark.skipif(
        os.environ.get("FLAGOS_ACCELERATOR") != "cambricon",
        reason="Requires the Cambricon native kernel",
    ),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "case",
    ["scalar", "cpu_first", "broadcast", "strided", "zero_dim", "empty", "mixed"],
)
def test_device_only_mul(dtype, case):
    torch.manual_seed(41)
    a = torch.randn(2, 4, 8).to(dtype)
    b = torch.randn(2, 4, 8).to(dtype)
    if case in ("scalar", "cpu_first"):
        b = torch.tensor(2.5, dtype=torch.float64)
    elif case == "broadcast":
        b = b[:, :1, :]
    elif case == "strided":
        a = a.transpose(0, 1)
        b = b.transpose(0, 1)
    elif case == "zero_dim":
        a, b = a[0, 0, 0], b[0, 0, 0]
    elif case == "empty":
        a, b = a[:0], b[:0]
    elif case == "mixed":
        b = b.float()
    ad = a.to("flagos")
    bd = b if case in ("scalar", "cpu_first") else b.to("flagos")
    if case == "cpu_first":
        got, ref = torch.mul(bd, ad), torch.mul(b, a)
    else:
        got, ref = torch.mul(ad, bd), torch.mul(a, b)
    assert got.device.type == "flagos"
    assert got.dtype == ref.dtype
    torch.testing.assert_close(got.cpu(), ref, rtol=1e-3, atol=1e-3)
    # Exercise a following Triton operator on the actual result, not an
    # unrelated tensor: asynchronous invalid-pointer faults surfaced here.
    torch.testing.assert_close((got + got).cpu(), ref + ref, rtol=1e-3, atol=1e-3)


def test_int32_mul_is_exact():
    a = torch.tensor([16777217, 536870911, -16777217, 0], dtype=torch.int32)
    b = torch.tensor([3, 3, 3, -7], dtype=torch.int32)
    got = torch.mul(a.to("flagos"), b.to("flagos"))
    torch.testing.assert_close(got.cpu(), a * b, rtol=0, atol=0)


def test_broadcast_backward():
    torch.manual_seed(42)
    a = torch.randn(2, 4, 8, requires_grad=True)
    b = torch.randn(2, 1, 8, requires_grad=True)
    ad = a.detach().to("flagos").requires_grad_(True)
    bd = b.detach().to("flagos").requires_grad_(True)
    (ad * bd).sum().backward()
    (a * b).sum().backward()
    torch.testing.assert_close(ad.grad.cpu(), a.grad)
    torch.testing.assert_close(bd.grad.cpu(), b.grad)
