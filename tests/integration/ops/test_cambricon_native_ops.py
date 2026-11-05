"""Regression coverage for the Cambricon CNNL operators a training step uses.

`mul.Tensor`, its `out=`/in-place spellings and `embedding_dense_backward`: the
three ops a Qwen3 causal-LM training step reached instead of the device before
these kernels existed.
"""

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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["same", "broadcast", "cpu_scalar", "empty_out"])
def test_mul_out(dtype, case):
    """The out= spelling, which is what an optimizer step reaches directly."""
    torch.manual_seed(43)
    a = torch.randn(2, 4, 8).to(dtype)
    b = torch.randn(2, 4, 8).to(dtype)
    if case == "broadcast":
        b = b[:, :1, :]
    elif case == "cpu_scalar":
        b = torch.tensor(2.5, dtype=torch.float64)
    ref = torch.mul(a, b)
    # An empty out= is the shape a caller passes to be filled.
    size = 0 if case == "empty_out" else ref.shape
    out = torch.empty(size, dtype=dtype, device="flagos")
    got = torch.ops.aten.mul.out(a.to("flagos"), b.to("flagos"), out=out)
    assert got.device.type == "flagos"
    assert got.shape == ref.shape
    torch.testing.assert_close(got.cpu(), ref, rtol=1e-3, atol=1e-3)


def test_mul_out_serves_inplace():
    """`mul_` lowers to `mul.out`, so it must not fall back either."""
    torch.manual_seed(44)
    a = torch.randn(8, 8)
    b = torch.randn(8, 8)
    ref = a * b
    dev = a.to("flagos")
    dev.mul_(b.to("flagos"))
    torch.testing.assert_close(dev.cpu(), ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_type", [torch.int32, torch.int64])
@pytest.mark.parametrize("padding_idx", [-1, 0])
@pytest.mark.parametrize("scale_by_freq", [False, True])
def test_embedding_dense_backward(dtype, index_type, padding_idx, scale_by_freq):
    torch.manual_seed(45)
    grad = torch.randn(2, 6, 16).to(dtype)
    indices = torch.randint(0, 20, (2, 6), dtype=index_type)
    indices[0, 0] = padding_idx if 0 <= padding_idx < 20 else 3
    indices[0, 1] = indices[0, 2]  # a duplicate, so frequency scaling matters
    ref = torch.ops.aten.embedding_dense_backward(
        grad, indices, 20, padding_idx, scale_by_freq
    )
    got = torch.ops.aten.embedding_dense_backward(
        grad.to("flagos"), indices.to("flagos"), 20, padding_idx, scale_by_freq
    )
    assert got.device.type == "flagos"
    assert got.shape == ref.shape
    # The tolerance is the dtype's own precision, not a slack budget: measured
    # against a float64 reference the device result is as close as aten's own
    # (bf16 3.9e-03 for both, fp16 2.4e-04 vs 3.3e-04).
    tol = 4 * torch.finfo(dtype).eps
    torch.testing.assert_close(got.cpu(), ref, rtol=tol, atol=tol)
    # A following elementwise op consumes the result, which is where an
    # asynchronous fault in the vendor call would surface.
    torch.testing.assert_close((got + got).cpu(), ref + ref, rtol=tol, atol=tol)


def test_embedding_dense_backward_ignores_out_of_range_indices():
    """aten ignores these; narrowing int64 to int32 must not alias 2**32+1 to 1."""
    torch.manual_seed(46)
    grad = torch.randn(5, 8)
    indices = torch.tensor([1, -2, 6, 2**32 + 1, 0], dtype=torch.int64)
    ref = torch.ops.aten.embedding_dense_backward(grad, indices, 6, -1, False)
    got = torch.ops.aten.embedding_dense_backward(
        grad.to("flagos"), indices.to("flagos"), 6, -1, False
    )
    torch.testing.assert_close(got.cpu(), ref)


def test_embedding_dense_backward_empty_indices():
    grad = torch.zeros(0, 8)
    indices = torch.zeros(0, dtype=torch.int64)
    got = torch.ops.aten.embedding_dense_backward(
        grad.to("flagos"), indices.to("flagos"), 4, -1, False
    )
    assert got.shape == (4, 8)
    assert torch.count_nonzero(got.cpu()) == 0


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
