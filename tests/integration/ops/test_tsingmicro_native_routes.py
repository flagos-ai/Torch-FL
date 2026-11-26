# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from torch_fl import native_host

pytestmark = [
    pytest.mark.tsingmicro,
    pytest.mark.flaggems,
    pytest.mark.main_ops,
    pytest.mark.native_source_host,
]


@pytest.mark.parametrize("stride, count", [(37, 128), (1215488, 8)])
def test_strided_bfloat16_sample_keeps_values_and_offsets(stride, count):
    values = torch.arange(1, count + 1).to(torch.bfloat16)
    storage = torch.zeros(2 + (count - 1) * stride, dtype=torch.bfloat16)
    storage[1::stride].copy_(values)
    view = storage.to("flagos")[1::stride]
    untouched = torch.full((count + 2,), -1, dtype=torch.bfloat16)
    output = untouched.to("flagos")
    before = native_host.enqueue_origins()["python"]

    sample = view.cpu()
    output[1:-1].copy_(view)

    untouched[1:-1].copy_(values)
    torch.testing.assert_close(sample, values, atol=0, rtol=0)
    torch.testing.assert_close(output.cpu(), untouched, atol=0, rtol=0)
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.parametrize("offset", [-1, 0, 1, 3])
@pytest.mark.tsingmicro_txdnn
def test_native_nll_autograd_preserves_wide_class_offsets(offset):
    rows, classes = 128, 151936
    labels = (torch.arange(rows) * 1009 + 31) % classes
    row, column = divmod((1 << 24) + offset, classes)
    labels[row] = column
    labels[-2] = classes - 1
    labels[-1] = -100
    active = labels != -100
    values = torch.zeros(rows, classes)
    values[active, labels[active]] = -2
    source = values.to("flagos").requires_grad_()
    target = torch.nn.functional.pad(labels, (1, 0)).to("flagos")[1:]
    before = native_host.enqueue_origins()

    torch.nn.functional.nll_loss(source, target, weight=None).backward()
    torch.nn.functional.nll_loss(
        values.requires_grad_(), labels, weight=None
    ).backward()

    torch.testing.assert_close(source.grad.cpu(), values.grad, atol=0, rtol=0)
    after = native_host.enqueue_origins()
    assert after["python"] == before["python"]
    name = "nll_loss_backward_kernel"
    assert after["native"][name] == before["native"].get(name, 0) + 1


@pytest.mark.tsingmicro_txdnn
def test_txdnn_long_cumsum_preserves_large_signed_prefixes():
    values = torch.arange(128, dtype=torch.long).view(1, -1) % 5 - 2
    values[:, ::7] = -(1 << 34)
    expected = values.cumsum(-1)
    device_values = values.to("flagos")
    output = torch.empty_like(device_values)
    before = native_host.enqueue_origins()["python"]

    result = device_values.cumsum(-1)
    torch.cumsum(device_values, -1, out=output)
    torch.flagos.synchronize()

    torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.parametrize("invalid", [-1, 7, (1 << 32) + 2])
@pytest.mark.tsingmicro_txdnn
def test_txdnn_embedding_rejects_invalid_long_indices(invalid):
    weights = torch.arange(7 * 16).reshape(7, 16).to(torch.bfloat16)
    indices = torch.tensor([[1, 1, 6]])
    expected = torch.embedding(weights, indices)
    device_weights = weights.to("flagos")
    before = native_host.enqueue_origins()["python"]

    result = torch.embedding(device_weights, indices.to("flagos"))
    with pytest.raises(IndexError, match="index out of range"):
        torch.embedding(device_weights, torch.tensor([invalid]).to("flagos"))
    torch.flagos.synchronize()

    torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.tsingmicro_txdnn
def test_txdnn_long_positions_preserve_the_eager_integer_operations():
    values = torch.tensor([[0, 1, (1 << 59) + 127, -(1 << 59)]])
    before = native_host.enqueue_origins()["python"]

    positions = torch.arange(128, device="flagos") + 0
    shifted = values.to("flagos") - 1
    torch.flagos.synchronize()

    torch.testing.assert_close(positions.cpu(), torch.arange(128), atol=0, rtol=0)
    torch.testing.assert_close(shifted.cpu(), values - 1, atol=0, rtol=0)
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.tsingmicro_txdnn
def test_native_long_arithmetic_preserves_integer_lanes():
    offsets = torch.arange(129)
    values = torch.where(offsets % 2 == 0, (1 << 60) + offsets, -(1 << 60) + offsets)
    scalar = (1 << 59) + 3
    source = values.to("flagos")
    before = native_host.enqueue_origins()

    with native_host.execution_scope(source.device.index):
        added = source[:-1] + scalar
        subtracted = source[1:] - source[:-1]

    assert torch.equal(added.cpu(), values[:-1] + scalar)
    assert torch.equal(subtracted.cpu(), values[1:] - values[:-1])
    assert added.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    after = native_host.enqueue_origins()
    for kernel in ("add_scalar_lanes", "sub_scalar_lanes"):
        assert after["native"][kernel] == before["native"].get(kernel, 0) + 1
    assert after["python"] == before["python"]


@pytest.mark.tsingmicro_txdnn
def test_txdnn_packed_positions_match_the_eager_mask_operations():
    positions = torch.tensor([[0, 1, 0, (1 << 59) + 127, (1 << 59) + 128]])

    def packed_mask(value):
        earlier = value[:, :1] - 1
        diff = torch.diff(value, prepend=earlier, dim=-1)
        return (diff != 1).cumsum(-1), value.masked_fill(value == 0, -1)

    before = native_host.enqueue_origins()["python"]
    actual = packed_mask(positions.to("flagos"))
    torch.flagos.synchronize()

    for output, expected in zip(actual, packed_mask(positions)):
        torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.parametrize("values", [[], [True], [False], [True] * 127 + [False]])
@pytest.mark.tsingmicro_txdnn
def test_txdnn_bool_all_is_visible_to_an_immediate_eager_scalar_read(values):
    tensor = torch.tensor(values, dtype=torch.bool).to("flagos")
    before = native_host.enqueue_origins()["python"]

    assert tensor.all().item() is all(values)
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.parametrize("queries, keys", [(128, 128), (1, 129)])
@pytest.mark.tsingmicro_txdnn
def test_txdnn_causal_broadcast_preserves_large_long_comparisons(queries, keys):
    key_positions = ((1 << 58) + torch.arange(keys)).view(1, 1, 1, keys)
    query_positions = ((1 << 58) + torch.arange(queries)).view(1, 1, queries, 1)
    before = native_host.enqueue_origins()["python"]

    actual = key_positions.to("flagos") <= query_positions.to("flagos")

    torch.testing.assert_close(actual.cpu(), key_positions <= query_positions)
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.tsingmicro_txdnn
def test_txdnn_long_scalar_fill_keeps_adjacent_values():
    initial = torch.arange(130, dtype=torch.long)
    output = initial.to("flagos")
    before = native_host.enqueue_origins()["python"]

    output[1:-1].fill_(-(1 << 24))
    output[2:3].fill_(-1.5)
    unfinished_sequences = torch.ones(1, dtype=torch.long, device="flagos")

    initial[1:-1].fill_(-(1 << 24))
    initial[2:3].fill_(-1.5)
    torch.testing.assert_close(output.cpu(), initial, rtol=0, atol=0)
    assert unfinished_sequences.item() == 1
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.tsingmicro_txdnn
def test_txdnn_scalar_tensor_preserves_signed_zero(dtype):
    before = native_host.enqueue_origins()["python"]

    zero = torch.scalar_tensor(-0.0, dtype=dtype, device="flagos")

    assert zero.dtype == dtype and zero.shape == ()
    assert torch.signbit(zero.cpu()).item()
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.tsingmicro_txdnn
def test_txdnn_bfloat16_conversion_preserves_vendor_payloads():
    patterns = torch.tensor([0x00000000, 0x80000000, 0x3F808000, 0xFFC01234])
    source = patterns.to(torch.int32).view(torch.float32).to("flagos")
    expected_bfloat16 = torch.tensor([0, 0x8000, 0x3F80, 0xFFC0]).to(torch.int16)
    before = native_host.enqueue_origins()["python"]

    narrow = source.to(torch.bfloat16)
    wide = narrow.to(torch.float32)

    torch.testing.assert_close(narrow.cpu().view(torch.int16), expected_bfloat16)
    torch.testing.assert_close(
        wide.cpu().view(torch.int32), expected_bfloat16.to(torch.int32) << 16
    )
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.tsingmicro_txdnn
def test_txdnn_bfloat16_conversion_preserves_dense_transpose():
    bits = torch.arange(8 * 128 * 128, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16).reshape(1, 8, 128, 128)
    source = values.to("flagos").transpose(1, 2)
    before = native_host.enqueue_origins()["python"]

    result = source.to(torch.float32, memory_format=torch.preserve_format)

    assert result.shape == source.shape and result.stride() == source.stride()
    flat = result.as_strided((result.numel(),), (1,), result.storage_offset()).cpu()
    torch.testing.assert_close(flat.view(torch.int32), bits.to(torch.int32) << 16)
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.tsingmicro_txdnn
def test_native_expanded_attention_division_preserves_gradient_rows():
    rows = torch.linspace(-5, 5, 128 * 8).reshape(1, 128, 8, 1)
    source = rows.to("flagos").expand(1, 128, 8, 128)
    before = native_host.enqueue_origins()

    result = torch.div(source, 128)

    expected = rows.expand_as(result) / 128
    assert result.stride() == expected.stride()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    after = native_host.enqueue_origins()
    name = "true_div_func_tensor_scalar_kernel_rank_4"
    assert after["native"][name] == before["native"].get(name, 0) + 1
    assert after["python"] == before["python"]


def test_native_silu_backward_preserves_bfloat16_intermediate_rounding():
    x = torch.tensor([0.953125, 0.474609375, -1.1484375], dtype=torch.bfloat16)
    dy = torch.tensor(
        [0.00274658203125, -0.00057220458984375, -0.0047607421875],
        dtype=torch.bfloat16,
    )
    x = x.repeat(131072).view(1, 128, 3072)
    dy = dy.repeat(131072).view_as(x)
    sigmoid = torch.sigmoid(x.float()).bfloat16()
    derivative = sigmoid * (1 + x * (1 - sigmoid))
    expected = dy * derivative
    device_x, device_dy = x.to("flagos"), dy.to("flagos")
    before = native_host.enqueue_origins()

    actual = torch.ops.aten.silu_backward(device_dy, device_x)
    with torch.inference_mode():
        inference = torch.ops.aten.silu_backward(device_dy, device_x)
        expected_inference = torch.ops.aten.silu_backward(dy, x)

    assert torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))
    assert torch.equal(
        inference.cpu().view(torch.int16), expected_inference.view(torch.int16)
    )
    after = native_host.enqueue_origins()
    name = "silu_backward_kernel_kernel_rank_1"
    assert after["native"][name] == before["native"].get(name, 0) + 2
    assert after["python"] == before["python"]


@pytest.mark.tsingmicro_txdnn
def test_txdnn_foreach_add_rounds_wrapped_scalar_to_bfloat16():
    values = torch.full((1024,), 0x2E01, dtype=torch.int16).view(torch.bfloat16)
    epsilon = 1e-8
    expected = values + torch.scalar_tensor(epsilon, dtype=torch.bfloat16)
    denominator = values.to("flagos")
    before = native_host.enqueue_origins()

    torch._foreach_add_([denominator], epsilon)

    assert torch.equal(denominator.cpu().view(torch.int16), expected.view(torch.int16))
    after = native_host.enqueue_origins()
    kernel = "add_func_tensor_scalar_kernel_rank_1"
    assert after["native"][kernel] == before["native"].get(kernel, 0) + 1
    assert after["python"] == before["python"]


@pytest.mark.tsingmicro_txdnn
def test_txdnn_foreach_addcdiv_matches_bfloat16_cancellation():
    # Two first-step projection updates and two signed infinite denominators.
    inputs = [
        torch.tensor(values, dtype=torch.bfloat16).repeat(32)
        for values in (
            (9.965896606445312e-05, -0.00010013580322265625, 0.0, -0.0),
            (6.246566772460938e-05, -0.0002689361572265625, -1.0, -1.0),
            (0.0006256103515625, 0.002685546875, float("inf"), float("-inf")),
        )
    ]
    coefficient = -1e-4 / (1 - 0.9)
    expected = torch.addcdiv(*inputs, value=coefficient)
    weights, moment, divisor = [value.to("flagos") for value in inputs]
    versions = tuple(value._version for value in (weights, moment, divisor))
    before = native_host.enqueue_origins()

    torch._foreach_addcdiv_([weights], [moment], [divisor], [coefficient])

    assert weights._version == versions[0] + 1
    assert (moment._version, divisor._version) == versions[1:]
    assert torch.equal(weights.cpu().view(torch.int16), expected.view(torch.int16))
    assert torch.equal(moment.cpu().view(torch.int16), inputs[1].view(torch.int16))
    assert torch.equal(divisor.cpu().view(torch.int16), inputs[2].view(torch.int16))
    after = native_host.enqueue_origins()
    kernel = "addcdiv_kernel_kernel_rank_1"
    assert after["native"][kernel] == before["native"].get(kernel, 0) + 1
    assert after["python"] == before["python"]


@pytest.mark.parametrize("shared_operand", [True, False])
@pytest.mark.tsingmicro_txdnn
def test_txdnn_foreach_addcmul_preserves_gradient_and_output(shared_operand):
    variance = torch.linspace(0, 1e-5, 1024).to(torch.bfloat16)
    gradient = torch.linspace(-0.03, 0.03, 1024).to(torch.bfloat16)
    second = gradient if shared_operand else gradient.roll(5)
    coefficient = 1 - 0.999
    expected = torch.addcmul(variance, gradient, second, value=coefficient)
    device_variance, device_gradient = variance.to("flagos"), gradient.to("flagos")
    device_second = device_gradient if shared_operand else second.to("flagos")
    versions = (
        device_variance._version,
        device_gradient._version,
        device_second._version,
    )
    storage = device_variance.untyped_storage().data_ptr()
    before = native_host.enqueue_origins()["python"]

    torch._foreach_addcmul_(
        [device_variance], [device_gradient], [device_second], value=coefficient
    )

    assert device_variance._version == versions[0] + 1
    assert (device_gradient._version, device_second._version) == versions[1:]
    assert device_variance.untyped_storage().data_ptr() == storage
    assert torch.equal(
        device_variance.cpu().view(torch.int16), expected.view(torch.int16)
    )
    assert torch.equal(
        device_gradient.cpu().view(torch.int16), gradient.view(torch.int16)
    )
    assert torch.equal(device_second.cpu().view(torch.int16), second.view(torch.int16))
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.parametrize(
    "heads, transposed", [(None, False), (8, False), (8, True), (16, False), (16, True)]
)
def test_native_bfloat16_leading_sum_matches_stock_accumulation_order(
    heads, transposed
):
    generator = torch.Generator().manual_seed(7223 if heads is None else 363928)
    shape = (1, 128, 1024) if heads is None else (1, 128, heads, 128)
    dims = (0, 1) if heads is None else (0, 1, 2)
    values = torch.ldexp(
        torch.randn(shape, generator=generator),
        torch.randint(-24, 24, shape, generator=generator),
    ).bfloat16()
    values[:, 64:] = -values[:, :64].roll(1, dims=1)
    values[:, 13] *= 0.99609375
    if transposed:
        packed = values.transpose(1, 2).contiguous()
        values = packed.transpose(1, 2)
        source = packed.to("flagos").transpose(1, 2)
    else:
        source = values.to("flagos")
    expected = values.sum(dims, keepdim=True)
    before = native_host.enqueue_origins()

    result = source.sum(dims, keepdim=True)

    assert result.stride() == expected.stride()
    assert torch.equal(result.cpu().view(torch.int16), expected.view(torch.int16))
    after = native_host.enqueue_origins()
    kernel = "sum_dim_kernel_outer_ordered"
    assert after["native"][kernel] == before["native"].get(kernel, 0) + 1
    assert after["python"] == before["python"]


@pytest.mark.parametrize("length, heads", [(128, 16), (1, 8)])
@pytest.mark.tsingmicro_txdnn
def test_txdnn_bfloat16_negation_keeps_zero_signs_and_nan_payloads(length, heads):
    encodings = torch.tensor(
        [0, 0x8000, 0x7FC1, 0xFFC7, 0xFFFF, 0x7F80, 0xFF80, 1, 0x8001],
        dtype=torch.int32,
    ).to(torch.int16)
    indices = torch.arange(length * heads * 128) % encodings.numel()
    values = encodings[indices].reshape(1, length, heads, 128).view(torch.bfloat16)
    expected = -values.transpose(1, 2)[..., 64:]
    source = values.to("flagos").transpose(1, 2)[..., 64:]
    before = native_host.enqueue_origins()["python"]

    output = -source

    assert output.dtype == torch.bfloat16
    assert output.stride() == expected.stride()
    flat = output.as_strided((output.numel(),), (1,), output.storage_offset()).cpu()
    actual = flat.as_strided(output.shape, output.stride())
    torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16))
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.parametrize("length, transposed", [(128, True), (129, False)])
@pytest.mark.tsingmicro_txdnn
def test_txdnn_bfloat16_expanded_view_reshape_preserves_storage_bits(
    length, transposed
):
    shape = (1, length, 8, 128) if transposed else (1, 8, length, 128)
    values = torch.arange(8 * length * 128, dtype=torch.int32).to(torch.int16)
    values = values.view(torch.bfloat16).reshape(shape)
    device_values = values.to("flagos")

    def repeat(tensor):
        if transposed:
            tensor = tensor.transpose(1, 2)
        expanded = tensor[:, :, None, :, :].expand(1, 8, 2, length, 128)
        return expanded.reshape(1, 16, length, 128)

    before = native_host.enqueue_origins()
    result = repeat(device_values)
    expected = repeat(values)

    assert result.stride() == expected.stride()
    torch.testing.assert_close(
        result.cpu().view(torch.int16), expected.view(torch.int16)
    )
    after = native_host.enqueue_origins()
    name = "copy_func_kernel_rank_5"
    assert after["native"][name] == before["native"].get(name, 0) + 1
    assert after["python"] == before["python"]


@pytest.mark.parametrize("length, axis", [(130, -2), (191, 2)])
@pytest.mark.tsingmicro_txdnn
def test_txdnn_bfloat16_concat_preserves_each_prefix_and_append(length, axis):
    count = 8 * length * 128
    prefix_storage = torch.arange(count + 1, dtype=torch.int32).to(torch.int16)
    append_storage = -torch.arange(1, 8 * 128 + 2, dtype=torch.int16)
    prefix_storage = prefix_storage.view(torch.bfloat16)
    append_storage = append_storage.view(torch.bfloat16)
    prefix = prefix_storage[1:].view(1, 8, length, 128)
    append = append_storage[1:].view(1, 8, 1, 128)
    device_prefix = prefix_storage.to("flagos")[1:].view(prefix.shape)
    device_append = append_storage.to("flagos")[1:].view(append.shape)
    before = native_host.enqueue_origins()["python"]

    result = torch.cat((device_prefix, device_append), dim=axis)

    expected = torch.cat((prefix, append), dim=axis)
    assert result.shape == expected.shape and result.stride() == expected.stride()
    assert torch.equal(result.cpu().view(torch.int16), expected.view(torch.int16))
    assert torch.equal(device_prefix.cpu().view(torch.int16), prefix.view(torch.int16))
    assert torch.equal(device_append.cpu().view(torch.int16), append.view(torch.int16))
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.tsingmicro_txdnn
def test_txdnn_decode_projection_uses_native_mm_in_inference_mode():
    values = (torch.arange(1024).reshape(1, 1024) % 7 - 3).to(torch.bfloat16)
    weights = (torch.arange(2048 * 1024).reshape(2048, 1024) % 11 - 5).to(
        torch.bfloat16
    )
    left, right = values.to("flagos"), weights.to("flagos").T
    expected = values @ weights.T
    before = native_host.enqueue_origins()

    with native_host.execution_scope(left.device.index):
        with torch.no_grad():
            ordinary = torch.mm(left, right)
        with torch.inference_mode():
            decode = torch.mm(left, right)

    assert torch.equal(ordinary.cpu(), expected)
    assert torch.equal(decode.cpu(), expected)
    after = native_host.enqueue_origins()
    assert after["native"]["mm_kernel"] == before["native"].get("mm_kernel", 0) + 1
    assert after["python"] == before["python"]


@pytest.mark.parametrize(
    "heads, permutation, dtype, bits_type",
    [
        (16, (0, 2, 1, 3), torch.bfloat16, torch.int16),
        (8, (3, 0, 1, 2), torch.bfloat16, torch.int16),
        (8, (0, 2, 1, 3), torch.float32, torch.int32),
    ],
)
@pytest.mark.tsingmicro_txdnn
def test_txdnn_attention_transpose_contiguous_preserves_storage_bits(
    heads, permutation, dtype, bits_type
):
    values = torch.arange(heads * 128 * 128, dtype=torch.int32).to(bits_type)
    values = values.view(dtype).reshape(1, heads, 128, 128)
    source = values.to("flagos").permute(permutation)

    before = native_host.enqueue_origins()
    output = source.contiguous()
    expected = values.permute(permutation).contiguous()

    assert output.stride() == expected.stride()
    torch.testing.assert_close(output.cpu().view(bits_type), expected.view(bits_type))
    after = native_host.enqueue_origins()
    name = "copy_func_kernel_rank_4"
    assert after["native"][name] == before["native"].get(name, 0) + 1
    assert after["python"] == before["python"]


@pytest.mark.parametrize("offset", [0, 64])
@pytest.mark.tsingmicro_txdnn
def test_txdnn_bfloat16_row_window_contiguous_preserves_storage_bits(offset):
    bits = torch.arange(8 * 128 * 128, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16).reshape(1, 8, 128, 128)
    window = slice(offset, offset + 64)
    source = values.to("flagos")[..., window]

    before = native_host.enqueue_origins()
    result = source.contiguous()
    expected = values[..., window].contiguous()

    assert result.stride() == expected.stride()
    torch.testing.assert_close(
        result.cpu().view(torch.int16), expected.view(torch.int16)
    )
    after = native_host.enqueue_origins()
    name = "copy_func_kernel_rank_4"
    assert after["native"][name] == before["native"].get(name, 0) + 1
    assert after["python"] == before["python"]


@pytest.mark.tsingmicro_txdnn
def test_txdnn_bfloat16_attention_gradient_add_out_keeps_output_alias():
    shape = (1, 8, 128, 128)
    values = torch.arange(8 * 128 * 128).reshape(shape)
    left = ((values % 257 - 128) / 256).to(torch.bfloat16)
    right = ((values % 521 - 260) / 512).to(torch.bfloat16)
    source, other = left.to("flagos"), right.to("flagos")
    source_version, other_version = source._version, other._version
    before = native_host.enqueue_origins()

    result = torch.add(source, other, out=source)

    assert result is source
    assert result._version == source_version + 1 and other._version == other_version
    torch.testing.assert_close(
        result.cpu().view(torch.int16), (left + right).view(torch.int16)
    )
    after = native_host.enqueue_origins()
    name = "add_func_kernel_rank_1"
    assert after["native"][name] == before["native"].get(name, 0) + 1
    assert after["python"] == before["python"]


@pytest.mark.tsingmicro_txdnn
def test_txdnn_fp32_attention_gradient_add_out_keeps_transposed_output():
    values = torch.arange(8 * 128 * 128, dtype=torch.float32).reshape(1, 8, 128, 128)
    other = ((values % 13) / 16).reshape(1, 128, 8, 128)
    left = values.to("flagos").transpose(1, 2)
    right = other.to("flagos")
    versions = (left._version, right._version)
    before = native_host.enqueue_origins()

    result = torch.add(left, right, out=left)

    assert result is left and result.stride() == (131072, 128, 16384, 1)
    assert (left._version, right._version) == (versions[0] + 1, versions[1])
    packed = result.as_strided((result.numel(),), (1,), result.storage_offset()).cpu()
    expected = (values.transpose(1, 2) + other).transpose(1, 2).flatten()
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    after = native_host.enqueue_origins()
    name = "add_func_kernel_rank_4"
    assert after["native"][name] == before["native"].get(name, 0) + 1
    assert after["python"] == before["python"]


@pytest.mark.tsingmicro_txdnn
def test_txdnn_pad_long_label_row_preserves_full_width_values():
    labels = ((1 << 54) + torch.arange(128)).reshape(1, 128)
    source = labels.to("flagos")
    before = native_host.enqueue_origins()["python"]

    actual = torch.nn.functional.pad(source, (0, 1), value=-100)

    reference = torch.nn.functional.pad(labels, (0, 1), value=-100)
    assert actual.stride() == reference.stride()
    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
    assert native_host.enqueue_origins()["python"] == before


@pytest.mark.parametrize("value", [False, True])
def test_native_bool_full_uses_the_published_fill_kernel(value):
    before = native_host.enqueue_origins()

    actual = torch.full((1,), value, device="flagos", dtype=torch.bool)

    assert actual.item() is value
    after = native_host.enqueue_origins()
    name = "fill_scalar_kernel"
    assert after["native"][name] == before["native"].get(name, 0) + 1
    assert after["python"] == before["python"]
