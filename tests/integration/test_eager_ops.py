# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare eager operators on the installed wheel with ordinary PyTorch."""

import math
from functools import cache

import pytest
import torch
import torch.nn.functional as F

import torch_fl
from tests.integration.flaggems.flaggems_caps import load_routes

pytestmark = pytest.mark.anyplatform
DEVICE = torch.device("flagos:0")
BF16 = torch.bfloat16
FP32 = torch.float32


@pytest.fixture(scope="module")
def routes():
    return cache(load_routes)


@pytest.fixture
def require_ops(routes):
    def require(dtype, *ops):
        active = routes(dtype)
        missing = [op for op in ops if active.get(op, "none") == "none"]
        if missing:
            pytest.skip(f"No {dtype} device route for {', '.join(missing)}")

    return require


@pytest.fixture(autouse=True)
def strict_eager():
    native = torch_fl._platform.build_info().native_hosts
    before = torch_fl.native_host.enqueue_origins()["python"] if native else None
    torch.ops.flagos.strict_host_routes(True)
    try:
        with torch_fl.native_host.execution_scope(DEVICE.index):
            yield
        if native:
            assert torch_fl.native_host.enqueue_origins()["python"] == before
    finally:
        torch.ops.flagos.strict_host_routes(False)


def values(shape, dtype=FP32):
    return (torch.arange(math.prod(shape)).reshape(shape) % 11 - 5).to(dtype) / 8


def assert_exact(actual, expected):
    assert actual.device == DEVICE
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [128**-0.5, -0.18731])
def test_bf16_multiply_with_python_scalar(require_ops, scale):
    require_ops("BFloat16", "mul.Tensor")
    source = torch.arange(-512, 512, dtype=FP32).div(8).to(BF16)
    assert_exact(source.to(DEVICE) * scale, source * scale)


@pytest.mark.parametrize("scalar_first", [False, True])
def test_bf16_multiply_with_host_scalar_tensor(require_ops, scalar_first):
    require_ops("BFloat16", "mul.Tensor")
    cpu = torch.arange(120, dtype=BF16).reshape(12, 10)
    host = torch.tensor(0.1875, dtype=FP32)
    source = cpu.to(DEVICE).T
    mul = torch.ops.aten.mul.Tensor
    actual = mul(host, source) if scalar_first else mul(source, host)
    expected = mul(host, cpu.T) if scalar_first else mul(cpu.T, host)
    assert_exact(actual, expected)


@pytest.mark.parametrize("alpha", [1, -0.5])
def test_bf16_add_handles_transposed_input(require_ops, alpha):
    require_ops("BFloat16", "add.Tensor")
    first, second = values((17, 33), BF16), values((33, 17), BF16)
    actual = torch.add(first.to(DEVICE).T, second.to(DEVICE), alpha=alpha)
    assert_exact(actual, torch.add(first.T, second, alpha=alpha))


def test_bf16_broadcast_multiply_preserves_layout(require_ops):
    require_ops("BFloat16", "mul.Tensor")
    lhs = torch.arange(16, dtype=BF16).reshape(1, 1, 1, 16)
    rhs = torch.arange(8, dtype=BF16).reshape(1, 8, 1, 1)
    actual = lhs.to(DEVICE) * rhs.to(DEVICE)
    expected = lhs * rhs
    assert actual.stride() == expected.stride()
    assert_exact(actual.transpose(1, 3) * 0.25, expected.transpose(1, 3) * 0.25)


@pytest.mark.parametrize(
    "dtype,aten_dtype", [(FP32, "Float"), (BF16, "BFloat16"), (torch.int64, "Long")]
)
def test_reflected_subtraction_preserves_operand_order(require_ops, dtype, aten_dtype):
    source = torch.tensor([1, -2, 0, 9], dtype=dtype)
    other = torch.tensor([5, 2, -8, 1], dtype=dtype)
    require_ops(aten_dtype, "rsub.Scalar", "rsub.Tensor")
    op = torch.ops.aten.rsub
    assert_exact(op.Scalar(source.to(DEVICE), 3, 2), op.Scalar(source, 3, 2))
    assert_exact(
        op.Tensor(source.to(DEVICE), other.to(DEVICE), alpha=2),
        op.Tensor(source, other, alpha=2),
    )


def test_scalar_subtraction_writes_the_output(require_ops):
    require_ops("Long", "sub.Tensor", "sub.out")
    cpu = torch.arange(37, dtype=torch.int64).reshape(1, -1)
    source = cpu.to(DEVICE)
    out = torch.empty(0, dtype=cpu.dtype, device=DEVICE)
    assert_exact(source - 1, cpu - 1)
    assert torch.sub(source, 1, out=out) is out
    assert_exact(out, cpu - 1)


@pytest.mark.parametrize("exponent", [1.0, 3])
def test_pow_preserves_values_and_allocates_output(require_ops, exponent):
    require_ops("Float", "pow.Tensor_Scalar")
    cpu = values((17, 33))
    if exponent == 1.0:
        # Vendor implementations may round other bases by one FP32 ULP.
        cpu = cpu.sign()
    source = cpu.to(DEVICE)
    output = source.pow(exponent)
    assert output.data_ptr() != source.data_ptr()
    assert_exact(output, cpu.pow(exponent))


def test_bf16_silu_rounding(require_ops):
    require_ops("BFloat16", "silu")
    cpu = torch.tensor([-5.9375, -0.25, 0.0, 0.25, 5.9375], dtype=BF16)
    assert_exact(F.silu(cpu.to(DEVICE)), F.silu(cpu))


@pytest.mark.parametrize("repeated", [False, True])
def test_cat_preserves_narrow_views_and_layout(require_ops, repeated):
    require_ops("BFloat16", "cat")
    cpu = values((2, 3, 5, 16), BF16)

    def inputs(value):
        return (value,) * 3 if repeated else value.transpose(1, 2).split(8, dim=-1)

    actual = torch.cat(inputs(cpu.to(DEVICE)), dim=-1)
    expected = torch.cat(inputs(cpu), dim=-1)
    assert actual.stride() == expected.stride()
    assert_exact(actual, expected)


@pytest.mark.parametrize("shape", [(2, 37, 64, 32), (3, 17, 33, 65)])
def test_bf16_bmm_with_transposed_rhs(require_ops, shape):
    require_ops("BFloat16", "bmm")
    batch, m, n, k = shape
    rng = torch.Generator().manual_seed(79)
    lhs = torch.randn(batch, m, k, dtype=BF16, generator=rng)
    rhs = torch.randn(batch, n, k, dtype=BF16, generator=rng)
    result = torch.bmm(lhs.to(DEVICE), rhs.to(DEVICE).transpose(1, 2))
    assert result.device == DEVICE
    torch.testing.assert_close(
        result.cpu(), torch.bmm(lhs, rhs.transpose(1, 2)), rtol=0.015, atol=0.035
    )


@pytest.mark.parametrize(
    "op,left,right",
    [
        ("mm", (17, 0), (0, 16)),
        ("mm", (0, 16), (16, 8)),
        ("bmm", (4, 17, 0), (4, 0, 16)),
        ("bmm", (0, 4, 16), (0, 16, 8)),
    ],
)
def test_bf16_matmul_with_empty_dimensions(require_ops, op, left, right):
    require_ops("BFloat16", op)
    apply = getattr(torch, op)
    lhs, rhs = torch.empty(left, dtype=BF16), torch.empty(right, dtype=BF16)
    assert_exact(apply(lhs.to(DEVICE), rhs.to(DEVICE)), apply(lhs, rhs))


def test_bf16_matmul_is_repeatable_in_deterministic_mode(require_ops):
    require_ops("BFloat16", "mm")
    rng = torch.Generator().manual_seed(42)
    left = torch.randn(17, 4097, dtype=BF16, generator=rng).to(DEVICE)
    right = torch.randn(4097, 33, dtype=BF16, generator=rng).to(DEVICE)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        expected = torch.mm(left, right).cpu()
        for _ in range(3):
            assert_exact(torch.mm(left, right), expected)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_bf16_linear_forward_and_gradients(require_ops):
    require_ops("BFloat16", "mm")
    # Keep the dot products away from cancellation at zero for bit-exact
    # BF16 comparisons across different FP32 accumulation implementations.
    cpu = values((2, 3, 8), BF16)
    weights = values((7, 8), BF16) + 0.125
    incoming = values((2, 3, 7), BF16) + 0.25

    def apply(device):
        x = cpu.to(device).detach().requires_grad_()
        w = weights.to(device).detach().requires_grad_()
        output = F.linear(x, w)
        return output.detach(), *torch.autograd.grad(
            output, (x, w), incoming.to(device)
        )

    for actual, expected in zip(apply(DEVICE), apply("cpu"), strict=True):
        assert_exact(actual, expected)


@pytest.mark.parametrize("shape", [(64,), (7, 256), (23, 2048)])
def test_fp32_mean_preserves_values_and_output_rank(require_ops, shape):
    require_ops("Float", "mean.dim")
    cpu = values(shape)
    assert_exact(cpu.to(DEVICE).mean(-1), cpu.mean(-1))


def test_rms_scale_preserves_fp32_accumulation(require_ops):
    require_ops("Float", "pow.Tensor_Scalar", "mean.dim", "add.Tensor", "rsqrt")
    cpu = values((3, 7, 257), BF16).float()

    def scale(value):
        return torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + 1e-6)

    torch.testing.assert_close(scale(cpu.to(DEVICE)).cpu(), scale(cpu))


@pytest.mark.parametrize("dims", [None, [], [-1], [0]])
def test_mean_handles_dtype_and_dimensions_on_strided_input(require_ops, dims):
    require_ops("BFloat16", "mean.dim")
    cpu = values((2, 3, 4), BF16)
    options = dict(dim=dims, keepdim=True, dtype=FP32)
    actual = cpu.to(DEVICE).transpose(0, 1).mean(**options)
    torch.testing.assert_close(actual.cpu(), cpu.transpose(0, 1).mean(**options))


@pytest.mark.parametrize(
    "dims,error,match",
    [
        ([4], IndexError, "Dimension out of range"),
        ([-1, -1], RuntimeError, "appears multiple times"),
    ],
)
def test_mean_rejects_invalid_dimensions(require_ops, dims, error, match):
    require_ops("Float", "mean.dim")
    with pytest.raises(error, match=match):
        values((2, 3, 4)).to(DEVICE).mean(dim=dims)


@pytest.mark.parametrize("dtype,aten_dtype", [(FP32, "Float"), (BF16, "BFloat16")])
@pytest.mark.parametrize("keepdim", [False, True])
def test_sum_handles_strides_and_empty_inputs(require_ops, dtype, aten_dtype, keepdim):
    require_ops(aten_dtype, "sum.dim_IntList")
    cpu = values((2, 3, 7), dtype)
    source = cpu.to(DEVICE).transpose(0, 1)
    reference = cpu.transpose(0, 1)
    assert_exact(source.sum((0, 1), keepdim), reference.sum((0, 1), keepdim))
    assert_exact(source[..., :0].sum(-1, keepdim), reference[..., :0].sum(-1, keepdim))


@pytest.mark.parametrize(
    "name", ["_softmax_backward_data", "_log_softmax_backward_data"]
)
@pytest.mark.parametrize("layout", ["functional", "strided_out", "empty_out"])
def test_softmax_backward_preserves_values_and_output_storage(
    require_ops, name, layout
):
    require_ops("Float", name if layout == "functional" else f"{name}.out")
    rng = torch.Generator().manual_seed(42)
    logits = torch.randn(2, 3, 257, generator=rng)
    grad = torch.randn(2, 3, 257, generator=rng)
    logarithmic = name.startswith("_log")
    output = logits.log_softmax(-1) if logarithmic else logits.softmax(-1)
    op = getattr(torch.ops.aten, name)
    expected = op.default(grad, output, -1, FP32)
    args = (grad.to(DEVICE), output.to(DEVICE), -1, FP32)
    if layout == "functional":
        actual = op.default(*args)
    else:
        out = torch.empty(0, device=DEVICE)
        if layout == "strided_out":
            out = torch.empty(2, 3, 514, device=DEVICE)[..., ::2]
        pointer, stride = out.data_ptr(), out.stride()
        output_keyword = "out" if logarithmic else "grad_input"
        actual = op.out(*args, **{output_keyword: out})
        assert actual is out
        if layout == "strided_out":
            assert (actual.data_ptr(), actual.stride()) == (pointer, stride)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=3e-8)


def test_log_softmax_backward_preserves_small_probabilities(require_ops):
    require_ops("Float", "_log_softmax_backward_data")
    log_probabilities = torch.tensor([[-0.1, -70.0, -2.5]])
    gradient = torch.tensor([[-0.25, 0.0, 0.0]])
    op = torch.ops.aten._log_softmax_backward_data.default
    expected = op(gradient, log_probabilities, -1, FP32)
    actual = op(gradient.to(DEVICE), log_probabilities.to(DEVICE), -1, FP32)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=0)


@pytest.mark.parametrize("all_ignored", [False, True])
def test_cross_entropy_with_optional_weight_and_ignored_labels(
    require_ops, all_ignored
):
    require_ops(
        "Float",
        "_log_softmax",
        "_log_softmax_backward_data",
        "nll_loss_forward",
        "nll_loss_backward",
    )
    labels = torch.tensor([0, 2, -100, 9, 7])
    if all_ignored:
        labels.fill_(-100)
    host = values((5, 11)).requires_grad_()
    value = host.detach().to(DEVICE).requires_grad_()
    expected = F.cross_entropy(host, labels, weight=None)
    actual = F.cross_entropy(value, labels.to(DEVICE), weight=None)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)
    torch.testing.assert_close(value.grad.cpu(), host.grad, atol=2e-7, rtol=3e-6)


@pytest.mark.parametrize(
    "name,overload,args",
    [
        ("_foreach_mul_", "Scalar", (0.5,)),
        ("_foreach_lerp_", "Scalar", ("tensors", 0.1)),
        ("_foreach_addcmul_", "Scalar", ("tensors", "tensors", 0.25)),
        ("_foreach_addcdiv_", "ScalarList", ("tensors", "tensors", [0.25, 0.5])),
        ("_foreach_div_", "ScalarList", ([0.5, 1.0],)),
        ("_foreach_add_", "Scalar", (0.5,)),
        ("_foreach_sqrt", "default", ()),
    ],
)
def test_bf16_foreach_preserves_outputs_and_versions(require_ops, name, overload, args):
    require_ops("BFloat16", name if overload == "default" else f"{name}.{overload}")
    op = getattr(getattr(torch.ops.aten, name), overload)
    source = [values((3, 8), BF16) + 1, values((129,), BF16) + 2]
    other = [torch.full_like(item, 2) for item in source]

    def apply(device):
        tensors = [item.to(device, copy=True) for item in source]
        operands = [item.to(device) for item in other]
        arguments = [operands if arg == "tensors" else arg for arg in args]
        before = [item._version for item in tensors]
        result = op(tensors, *arguments)
        if name.endswith("_"):
            assert [item._version for item in tensors] == [v + 1 for v in before]
            return tensors
        assert [item._version for item in tensors] == before
        return result

    for actual, expected in zip(apply(DEVICE), apply("cpu"), strict=True):
        assert_exact(actual, expected)


def test_bf16_inplace_lerp_preserves_scalar_weight(require_ops):
    require_ops("BFloat16", "lerp_.Scalar")
    incoming = values((5, 17), BF16)
    moment = torch.zeros_like(incoming)
    result = moment.to(DEVICE)
    assert result.lerp_(incoming.to(DEVICE), 0.1) is result
    assert_exact(result, moment.lerp_(incoming, 0.1))


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("dtype,aten_dtype", [(FP32, "Float"), (BF16, "BFloat16")])
@pytest.mark.parametrize(
    "name,overload,options",
    [
        ("div", "out", {}),
        ("addcdiv", "out", {"value": -0.25}),
        ("addcmul", "out", {"value": 0.125}),
        ("lerp", "Scalar_out", {"weight": 0.125}),
    ],
)
def test_out_preserves_aliasing(
    require_ops, name, overload, options, alias, dtype, aten_dtype
):
    require_ops(aten_dtype, f"{name}.{overload}")
    cpu = [values((5, 7), dtype) + 1, values((5, 7), dtype) + 2]
    if name == "div":
        # Use exact quotients so this case isolates the output storage contract.
        cpu[1] = torch.where(cpu[1] > 2, 0.5, -2.0).to(dtype)
    elif name in ("addcdiv", "addcmul"):
        cpu.append(torch.full_like(cpu[0], 2))
    op = getattr(torch, name)

    def apply(device):
        args = [value.to(device, copy=True) for value in cpu]
        out = args[0] if alias else torch.empty_like(args[0])
        pointer = out.data_ptr()
        assert op(*args, **options, out=out) is out
        assert out.data_ptr() == pointer
        return out

    assert_exact(apply(DEVICE), apply("cpu"))


@pytest.mark.parametrize("start", [0, 8])
def test_slice_backward_preserves_strided_gradient(require_ops, start):
    require_ops("BFloat16", "slice_backward")
    cpu = values((2, 7, 5, 16), BF16)
    args = ((2, 5, 7, 16), -1, start, start + 8, 1)
    op = torch.ops.aten.slice_backward.default
    result = op(cpu.to(DEVICE).transpose(1, 2)[..., 8:], *args)
    assert_exact(result, op(cpu.transpose(1, 2)[..., 8:], *args))


@pytest.mark.parametrize(
    "dtype,aten_dtype,padding_idx,scale_by_frequency",
    [(BF16, "BFloat16", -1, False), (FP32, "Float", 7, True)],
)
def test_embedding_gradient_accumulates_repeated_indices(
    require_ops, dtype, aten_dtype, padding_idx, scale_by_frequency
):
    require_ops(aten_dtype, "embedding_dense_backward")
    index = torch.tensor([[0, 1, 1, 7, 13]])
    grad = values((1, 5, 129), dtype)
    options = (17, padding_idx, scale_by_frequency)
    op = torch.ops.aten.embedding_dense_backward.default
    result = op(grad.to(DEVICE), index.to(DEVICE), *options)
    assert_exact(result, op(grad, index, *options))


@pytest.mark.parametrize("shape", [(1, 37), (2, 3, 7)])
def test_constant_padding_keeps_integer_labels(require_ops, shape):
    require_ops("Long", "constant_pad_nd")
    cpu = torch.arange(math.prod(shape)).reshape(shape)
    assert_exact(
        F.pad(cpu.to(DEVICE), (0, 1), value=-100), F.pad(cpu, (0, 1), value=-100)
    )


def test_pointwise_autograd_preserves_gradients(require_ops):
    require_ops("Float", "cos", "sin", "neg", "mul.Tensor")
    cpu = values((37,)).requires_grad_()
    source = cpu.detach().to(DEVICE).requires_grad_()
    result = source.cos()
    reference = cpu.cos()
    seed = torch.ones_like(cpu)
    (actual_grad,) = torch.autograd.grad(result, source, seed.to(DEVICE))
    (expected_grad,) = torch.autograd.grad(reference, cpu, seed)
    assert result.grad_fn is not None
    torch.testing.assert_close(result.cpu(), reference)
    torch.testing.assert_close(actual_grad.cpu(), expected_grad)


def test_scalar_factory_writes_a_rank_zero_tensor(require_ops):
    require_ops("Float", "scalar_tensor")
    actual = torch.scalar_tensor(3.125, dtype=FP32, device=DEVICE)
    assert_exact(actual, torch.scalar_tensor(3.125, dtype=FP32))


def test_argmax_preserves_first_tie_across_blocks(require_ops):
    require_ops("Float", "argmax")
    cpu = values((2, 2053))
    cpu[0, [17, 1537]] = 9
    cpu[1, -1] = 9
    assert_exact(cpu.to(DEVICE).argmax(-1), cpu.argmax(-1))


def test_where_scalar_branch_preserves_gradient(require_ops):
    require_ops("BFloat16", "where.self", "fill_.Scalar", "sum.dim_IntList")
    cpu_mask = torch.arange(37) % 2 == 0
    host = torch.tensor(0, dtype=BF16).requires_grad_()
    value = host.detach().to(DEVICE).requires_grad_()
    fill = torch.finfo(BF16).min
    actual = torch.where(cpu_mask.to(DEVICE), value, fill)
    expected = torch.where(cpu_mask, host, fill)
    assert_exact(actual, expected)
    (gradient,) = torch.autograd.grad(
        actual, value, torch.ones(37, dtype=BF16).to(DEVICE)
    )
    assert gradient.item() == cpu_mask.sum().item()


@pytest.mark.parametrize("separated", [False, True])
def test_advanced_indexing_broadcasts_adjacent_and_separated_indices(
    require_ops, separated
):
    require_ops("Float", "index.Tensor")
    cpu = values((4, 5, 6))
    row, col = torch.tensor([[1], [3]]), torch.tensor([0, -1, 2])

    def apply(value, indices):
        i, j = indices
        if separated:
            return value.transpose(0, 1)[i, :, j]
        return value[:, i, j]

    assert_exact(
        apply(cpu.to(DEVICE), (row.to(DEVICE), col.to(DEVICE))), apply(cpu, (row, col))
    )


def test_fp32_to_bf16_preserves_subnormals_and_rounds_ties():
    bits = torch.tensor(
        [0x00008000, 0x00018000, 0x3F808000, 0x3F818000, -(1 << 31), 0x7F800000],
        dtype=torch.int32,
    )
    cpu = bits.view(FP32)
    actual = cpu.to(DEVICE).to(BF16).cpu()
    assert torch.equal(actual.view(torch.int16), cpu.to(BF16).view(torch.int16))


def test_bf16_to_fp32_preserves_finite_bit_patterns():
    bits = torch.arange(1 << 16).to(torch.int16)
    cpu = bits[(bits & 0x7F80) != 0x7F80].view(BF16)
    actual = cpu.to(DEVICE).float().cpu()
    assert torch.equal(actual.view(torch.int32), cpu.float().view(torch.int32))


@pytest.mark.parametrize("layout,dtype", [("offset", torch.int64), ("expanded", BF16)])
def test_contiguous_clone_keeps_values_and_owns_storage(layout, dtype):
    cpu = torch.arange(37, dtype=dtype).reshape(1, -1)

    def view(value):
        return value[:, 5:] if layout == "offset" else value.expand(4, -1)

    source = view(cpu.to(DEVICE))
    actual = source.clone(memory_format=torch.contiguous_format)
    assert actual.is_contiguous()
    assert actual.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert_exact(actual, view(cpu).clone(memory_format=torch.contiguous_format))


def test_eager_events_and_blocking_scalar_read_order_work(require_ops):
    require_ops("BFloat16", "mul.Tensor", "sum.dim_IntList")
    producer = torch.Stream(device=DEVICE)
    consumer = torch.Stream(device=DEVICE)
    cpu = torch.arange(32, dtype=BF16)
    with producer:
        result = cpu.to(DEVICE)
        for _ in range(513):
            result = result * -1
        event = producer.record_event()
    with consumer:
        consumer.wait_event(event)
        found = result * -1
        assert found.sum(dim=0).item() == cpu.sum(dim=0).item()
    assert_exact(found, cpu)


@pytest.mark.parametrize("use_side_stream", [False, True])
def test_queued_ops_keep_view_storage_alive(require_ops, use_side_stream):
    require_ops("Float", "add.Tensor", "neg")
    cpu = values((16, 8))
    stream = (
        torch.Stream(device=DEVICE)
        if use_side_stream
        else torch.accelerator.current_stream(DEVICE)
    )
    results = []
    with stream:
        source = cpu.to(DEVICE)
        for index in range(12):
            temp = source + index
            view = torch.ops.aten._unsafe_view.default(temp, [8, 16])
            results.append(view.neg())
            del temp, view
            torch.empty_like(source).copy_(source)
    stream.synchronize()
    for index, output in enumerate(results):
        assert_exact(output, -(cpu + index).reshape(8, 16))
