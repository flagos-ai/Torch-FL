"""Regression coverage for the Cambricon CNNL operators a training step uses.

The set a Qwen3 causal-LM training step reaches, in the two groups it was added
in. `mul.Tensor`, its `out=` spelling and `embedding_dense_backward` were the ops
FlagGems could not serve at all. The rest -- elementwise, reductions, the two
matmuls -- FlagGems does serve, but through its Python/Triton path, where one
launch costs hundreds of microseconds whatever the tensor holds; for those the
assertions are about correctness, and the timing evidence lives in the PR.
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


# ---------------------------------------------------------------------------
# The ops that move a Qwen3 forward pass. Each is checked against the CPU
# reference, including the shapes and views the model actually produces rather
# than only the contiguous case.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fn", [torch.abs, torch.rsqrt, torch.neg])
def test_unary_ops(dtype, fn):
    torch.manual_seed(71)
    # rsqrt's domain is positive; abs and neg accept anything.
    x = torch.randn(4, 8, 16)
    if fn is torch.rsqrt:
        x = x.abs() + 0.5
    # The reference is taken on the same dtype the kernel sees: the kernel works
    # in float32 internally and narrows once, so comparing a half answer against
    # a float32 reference would be checking the wrong pair.
    xd = x.to(dtype)
    ref = fn(xd)
    got = fn(xd.to("flagos"))
    assert got.dtype == ref.dtype
    torch.testing.assert_close(got.cpu(), ref, rtol=0, atol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar", [2.5, -1.0, 0.0])
def test_scalar_arithmetic(dtype, scalar):
    """`x * 2.0` wraps the number in a rank-0 tensor; CNNL cannot describe rank
    zero, so this is the shape that used to fail before the kernel reshaped it."""
    torch.manual_seed(72)
    x = torch.randn(2, 4, 8)
    xd = x.to(dtype).to("flagos")
    torch.testing.assert_close(
        (xd * scalar).cpu(), (x * scalar).to(dtype), rtol=0, atol=5e-2
    )
    torch.testing.assert_close(
        (xd + scalar).cpu(), (x + scalar).to(dtype), rtol=0, atol=5e-2
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "dim,keepdim", [(-1, False), (0, True), (None, False), ((0, 2), False)]
)
@pytest.mark.parametrize("fn", [torch.sum, torch.mean])
def test_reductions(dtype, dim, keepdim, fn):
    torch.manual_seed(73)
    x = torch.randn(2, 4, 16)
    xd = x.to(dtype).to("flagos")
    if dim is None:
        got, ref = fn(xd), fn(x)
    else:
        got, ref = fn(xd, dim=dim, keepdim=keepdim), fn(x, dim=dim, keepdim=keepdim)
    # The reference is computed on the half-precision tensor and compared as
    # PyTorch returns it: mean and sum promote half inputs to float32, so
    # narrowing the answer to `dtype` here would compare the wrong two things.
    assert got.dtype == ref.dtype
    torch.testing.assert_close(got.cpu(), ref, rtol=0, atol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_matmul_against_a_transposed_weight(dtype):
    """`x @ w.t()` is how every Linear layer spells its product.

    The transposed operand is described by its storage plus the Ex descriptor's
    transpose flag; an earlier version copied the weight on every call, which
    was correct but 122 times slower than the same product on a contiguous
    operand. Both paths are compared so the fast one cannot silently disagree.
    """
    torch.manual_seed(74)
    x = torch.randn(16, 32)
    w = torch.randn(24, 32)
    xd, wd = x.to(dtype).to("flagos"), w.to(dtype).to("flagos")
    torch.testing.assert_close(
        torch.mm(xd, wd.t()).cpu(), torch.mm(x, w.t()).to(dtype), rtol=0, atol=5e-2
    )
    torch.testing.assert_close(
        torch.nn.functional.linear(xd, wd).cpu(),
        torch.nn.functional.linear(x, w).to(dtype),
        rtol=0,
        atol=5e-2,
    )


def test_bmm_matches_matmul_per_batch():
    torch.manual_seed(75)
    a = torch.randn(4, 8, 16)
    b = torch.randn(4, 16, 8)
    got = torch.bmm(a.to("flagos"), b.to("flagos"))
    torch.testing.assert_close(got.cpu(), torch.bmm(a, b), rtol=0, atol=1e-5)


def test_broadcast_add_is_not_materialised():
    """cnnlOpTensor broadcasts from the operand's own rank, so [2,512,1] against
    [2,512,1024] must answer without the result-shaped copy the first version
    made. The assertion is on the answer; the cost is what the PR measures."""
    torch.manual_seed(76)
    a = torch.randn(2, 8, 16)
    b = torch.randn(2, 8, 1)
    torch.testing.assert_close(
        torch.add(a.to("flagos"), b.to("flagos")).cpu(), a + b, rtol=0, atol=1e-6
    )


def test_ones_like():
    x = torch.randn(3, 5)
    got = torch.ones_like(x.to("flagos"))
    assert torch.equal(got.cpu(), torch.ones_like(x))


def test_neg_does_not_square():
    """The first implementation passed the operand twice to cnnlOpTensor, which
    computes -x*x rather than -x. Pinned because the wrong answer is plausible
    enough to survive a loose tolerance."""
    x = torch.tensor([1.0, -2.0, 3.0, -4.0])
    got = torch.neg(x.to("flagos"))
    assert got.cpu().tolist() == [-1.0, 2.0, -3.0, 4.0]
