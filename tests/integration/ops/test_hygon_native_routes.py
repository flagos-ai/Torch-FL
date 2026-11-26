# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Measured HCU routes, output contracts and device-resident conversions."""

import pytest
import torch
import torch_fl  # noqa: F401

pytestmark = [pytest.mark.dcu, pytest.mark.native_source_host, pytest.mark.flaggems]


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.float64]
)
@pytest.mark.parametrize("op", ["add", "mul", "cat"])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "resize"])
def test_hcu_out_routes_preserve_storage_and_values(dtype, op, layout):
    cpu = torch.arange(21, dtype=dtype).reshape(3, 7)
    other = torch.arange(7, dtype=dtype).reshape(1, 7)
    a, b = cpu.to("flagos"), other.to("flagos")
    if op == "cat":
        shape = (3, 14)
        expected = torch.cat([cpu, cpu], dim=-1)
    else:
        shape = cpu.shape
        expected = torch.add(cpu, other, alpha=2) if op == "add" else cpu * other
    out = (
        torch.empty(0, dtype=dtype, device="flagos")
        if layout == "resize"
        else torch.empty(tuple(reversed(shape)), dtype=dtype, device="flagos").t()
        if layout == "transposed"
        else torch.empty(shape, dtype=dtype, device="flagos")
    )
    pointer = out.data_ptr()
    if op == "add":
        result = torch.ops.aten.add.out(a, b, alpha=2, out=out)
    elif op == "mul":
        result = torch.ops.aten.mul.out(a, b, out=out)
    else:
        result = torch.ops.aten.cat.out([a, a], -1, out=out)
    assert result is out
    if layout != "resize":
        assert out.data_ptr() == pointer
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 7), (2, 0), (0, 7)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_hcu_mean_out_including_empty_reductions(dtype, shape, keepdim):
    cpu = torch.arange(shape[0] * shape[1], dtype=dtype).reshape(shape)
    value = cpu.to("flagos")
    out = torch.empty(0, device="flagos", dtype=dtype)
    result = torch.ops.aten.mean.out(value, [-1], keepdim, out=out)
    assert result is out
    torch.testing.assert_close(out.cpu(), cpu.mean(-1, keepdim), equal_nan=True)


@pytest.mark.parametrize("op", ["add", "mul"])
def test_hcu_pointwise_out_allows_full_alias_and_rejects_partial_overlap(op):
    cpu = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    a = cpu.to("flagos")
    b = torch.ones_like(a)
    call = getattr(torch.ops.aten, op).out
    assert call(a, b, out=a) is a
    expected = cpu + 1 if op == "add" else cpu
    torch.testing.assert_close(a.cpu(), expected)
    storage = torch.arange(22, dtype=torch.float32).to("flagos")
    with pytest.raises(RuntimeError, match="single memory location|overlap"):
        call(storage[:-1], b.reshape(-1), out=storage[1:])


def test_hcu_cat_out_rejects_input_alias():
    value = torch.ones(3, 7, device="flagos")
    with pytest.raises(RuntimeError, match="single memory location|overlap"):
        torch.ops.aten.cat.out([value], 0, out=value)


@pytest.mark.parametrize(
    "source_dtype,target_dtype",
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.int64, torch.float32),
    ],
)
def test_hcu_copy_cast_broadcast_and_stream_order(source_dtype, target_dtype):
    cpu = torch.arange(21, dtype=source_dtype).reshape(3, 7).t()
    stream = torch.Stream(device="flagos")
    with stream:
        source = cpu.t().contiguous().to("flagos").t()
        converted = source.to(target_dtype)
        assert converted.stride() == cpu.to(target_dtype).stride()
        target = torch.empty(2, 7, 3, device="flagos", dtype=target_dtype)
        target.copy_(converted)
    stream.synchronize()
    torch.testing.assert_close(target.cpu(), cpu.to(target_dtype).expand(2, 7, 3))


def test_hcu_strided_float64_copy_preserves_bits():
    bits = torch.tensor(
        [[0x3FF0000000000001, -(1 << 63)], [0x7FF8000000000042, 1]],
        dtype=torch.int64,
    )
    expected = torch.zeros(2, 3, dtype=torch.int64)
    expected[:, 1:].copy_(bits.t())
    stream = torch.Stream(device="flagos")
    with stream:
        source = bits.view(torch.float64).to("flagos")
        storage = torch.zeros(2, 3, dtype=torch.float64).to("flagos")
        storage[:, 1:].t().copy_(source)
    stream.synchronize()
    torch.testing.assert_close(
        storage.cpu().view(torch.int64), expected, rtol=0, atol=0
    )


def test_hcu_precision_entries_are_installed_without_boxing():
    from torch_fl import _build_config
    from torch_fl.native_host import vendor

    assert _build_config.NATIVE_VENDOR
    assert "boxing" not in _build_config.KERNELS
    record = vendor.audit()
    assert record["vendor"] == "hygon"
    assert {
        "bmm",
        "mm",
        "cat",
        "_softmax",
        "mean.dim",
        "rsqrt",
    } <= {row["op"] for row in record["catalog"]}
    assert all(row["vendor"] == "hygon" for row in record["catalog"])


@pytest.mark.parametrize("op", ["add", "mul", "cat", "mean"])
def test_hcu_out_dtype_promotion_matches_aten(op):
    cpu = torch.tensor(
        [[1.001, 0.002, 3.003], [0.004, 5.005, 0.006]], dtype=torch.bfloat16
    )
    a = cpu.to("flagos")
    out = torch.empty(0, device="flagos", dtype=torch.float32)
    expected = torch.empty(0, dtype=torch.float32)
    if op == "mean":
        torch.ops.aten.mean.out(a, [-1], False, out=out)
        torch.ops.aten.mean.out(cpu, [-1], False, out=expected)
    elif op == "cat":
        torch.ops.aten.cat.out([a, a], -1, out=out)
        torch.ops.aten.cat.out([cpu, cpu], -1, out=expected)
    else:
        getattr(torch.ops.aten, op).out(a, a, out=out)
        getattr(torch.ops.aten, op).out(cpu, cpu, out=expected)
    torch.testing.assert_close(out.cpu(), expected)


def test_hcu_reenabled_functional_routes_keep_aten_validation_and_aliasing():
    value = torch.arange(7, dtype=torch.int64).to("flagos")
    with pytest.raises(RuntimeError, match="alpha"):
        torch.add(value, value, alpha=0.5)
    with pytest.raises(RuntimeError, match="alpha"):
        torch.add(value, value, alpha=True)
    floating = value.float()
    with pytest.raises(RuntimeError, match="alpha"):
        torch.add(floating, floating, alpha=1j)
    with pytest.raises(RuntimeError, match="mean.*dtype|dtype.*mean"):
        torch.mean(value, dim=-1)
    result = torch.cat([value])
    assert result.data_ptr() != value.data_ptr()
    torch.testing.assert_close(result.cpu(), value.cpu())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "memory_format",
    [torch.preserve_format, torch.contiguous_format, torch.channels_last],
)
def test_hcu_cast_retains_requested_memory_format(dtype, memory_format):
    cpu = torch.arange(120, dtype=dtype).reshape(2, 3, 4, 5).transpose(1, 2)
    target = torch.bfloat16 if dtype == torch.float32 else torch.float32
    source = cpu.transpose(1, 2).contiguous().to("flagos").transpose(1, 2)
    result = source.to(target, memory_format=memory_format)
    expected = cpu.to(target, memory_format=memory_format)
    assert result.data_ptr() != source.data_ptr()
    assert result.stride() == expected.stride()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scalar", [0.08838834764831845, 0.1, -1.75, 2.0])
def test_hcu_published_source_admits_wrapped_scalar_values(dtype, scalar):
    from torch_fl import native_host

    cpu = torch.randn(1, 16, 1, 24, dtype=dtype)
    value = cpu.to("flagos")
    before = native_host.audit()["source_variant_hits"]["published_mul"]
    with torch.no_grad():
        actual = value * scalar
    torch.testing.assert_close(actual.cpu(), cpu * scalar, rtol=0, atol=0)
    assert native_host.audit()["source_variant_hits"]["published_mul"] == before + 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("exponent", [2, 2.0])
def test_hcu_floating_square_uses_the_compiled_mul_route(dtype, exponent):
    from torch_fl import native_host

    cpu = torch.randn(3, 7, dtype=dtype)
    value = cpu.to("flagos")
    before = native_host.audit()["source_variant_hits"]["published_mul"]
    with torch.no_grad():
        actual = value.pow(exponent)
    torch.testing.assert_close(actual.cpu(), cpu.pow(exponent), rtol=0, atol=0)
    assert native_host.audit()["source_variant_hits"]["published_mul"] == before + 1


@pytest.mark.min_devices(2)
@pytest.mark.parametrize("other_device", [False, True])
@pytest.mark.parametrize("new_stream", [False, True])
def test_hcu_native_precision_restores_device_and_stream(other_device, new_stream):
    from torch_fl.native_host import vendor

    initial = torch_fl.flagos.current_device()
    target = (initial + int(other_device)) % torch_fl.flagos.device_count()
    stream = (
        torch.Stream(device=f"flagos:{target}")
        if new_stream
        else torch.accelerator.current_stream(target)
    )
    before = vendor.audit()["hits"]
    a = torch.arange(24, dtype=torch.float32).reshape(3, 8) * 0.125
    b = torch.arange(40, dtype=torch.float32).reshape(8, 5) * 0.0625
    retained = []
    try:
        with torch.no_grad(), stream:
            fp = (a + 1).to(f"flagos:{target}")
            left, right = a.bfloat16(), b.bfloat16()
            dl, dr = left.to(f"flagos:{target}"), right.to(f"flagos:{target}")
            ba, bb = (
                left.unsqueeze(0).repeat(2, 1, 1),
                right.unsqueeze(0).repeat(2, 1, 1),
            )
            dba, dbb = ba.to(f"flagos:{target}"), bb.to(f"flagos:{target}")
            calls = (
                (lambda: dl @ dr, (left.float() @ right.float()).bfloat16()),
                (
                    lambda: torch.bmm(dba, dbb),
                    torch.bmm(ba.float(), bb.float()).bfloat16(),
                ),
                (lambda: fp.mean(-1), (a + 1).mean(-1)),
                (lambda: fp.rsqrt(), (a + 1).rsqrt()),
                (lambda: fp.softmax(-1), (a + 1).softmax(-1)),
                (lambda: torch.cat([fp.T, fp.T], -1), torch.cat([(a + 1).T] * 2, -1)),
            )
            for call, expected in calls:
                torch_fl.flagos.set_device(initial)
                result = call()
                assert result.device.index == target
                assert torch_fl.flagos.current_device() == initial
                assert (
                    torch.accelerator.current_stream(target).stream_id
                    == stream.stream_id
                )
                retained.append((result, expected))
        stream.synchronize()
        for result, expected in retained:
            torch.testing.assert_close(result.cpu(), expected, rtol=2e-5, atol=1e-6)
        after = vendor.audit()["hits"]
        for name in ("mm", "bmm", "cat", "mean.dim", "rsqrt", "_softmax"):
            assert after[name] == before[name] + 1
    finally:
        torch_fl.flagos.set_device(initial)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "empty"])
@pytest.mark.parametrize("op", ["unsafe_view", "transpose", "unsqueeze", "slice"])
def test_hcu_metadata_adapters_preserve_aten_aliases(dtype, layout, op):
    cpu = torch.arange(0 if layout == "empty" else 21, dtype=dtype).reshape(-1, 7)
    value = cpu.to("flagos")
    if layout == "transposed":
        cpu, value = cpu.t(), value.t()
    calls = {
        "unsafe_view": lambda x: torch.ops.aten._unsafe_view.default(x, [-1]),
        "transpose": lambda x: x.transpose(-1, -2),
        "unsqueeze": lambda x: x.unsqueeze(-1),
        "slice": lambda x: torch.ops.aten.slice.Tensor(x, -1, 1, None, 2),
    }
    if op == "unsafe_view" and layout == "transposed":
        with pytest.raises(RuntimeError, match="view size is not compatible"):
            calls[op](value)
        return
    expected = calls[op](cpu)
    actual = calls[op](value)
    assert actual.untyped_storage()._cdata == value.untyped_storage()._cdata
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("fail", [False, True])
def test_hcu_execution_scope_restores_nested_stream_state(fail):
    from torch_fl.native_host import vendor

    device = torch_fl.flagos.current_device()
    original = vendor._module._stream_state(device)
    scope = torch_fl.native_host.execution_scope(device)
    stream = torch.Stream(device=f"flagos:{device}")
    assert stream.stream_id != 0
    assert stream.stream_id != torch.accelerator.current_stream(device).stream_id

    def run():
        with scope, torch.no_grad():
            outer = vendor._module._stream_state(device)
            assert outer[1] == torch.accelerator.current_stream(device).stream_id
            with stream, torch_fl.native_host.execution_scope(device):
                assert vendor._module._stream_state(device)[1] == stream.stream_id
                cpu = torch.arange(8, dtype=torch.float32) + 1
                value = cpu.to(f"flagos:{device}")
                result = value.rsqrt()
                stream.synchronize()
                torch.testing.assert_close(result.cpu(), cpu.rsqrt())
            assert vendor._module._stream_state(device) == outer
            if fail:
                raise ValueError("scope exit")

    if fail:
        with pytest.raises(ValueError, match="scope exit"):
            run()
    else:
        run()
    assert vendor._module._stream_state(device) == original
    with scope:
        assert vendor._module._stream_state(device)[1] == (
            torch.accelerator.current_stream(device).stream_id
        )
    assert vendor._module._stream_state(device) == original


@pytest.mark.min_devices(2)
@pytest.mark.parametrize("other_device", [False, True])
def test_hcu_stream_events_and_recorded_storage_lifetime(other_device):
    initial = torch_fl.flagos.current_device()
    device = (initial + 1) % torch_fl.flagos.device_count() if other_device else initial
    producer = torch.Stream(device=f"flagos:{device}")
    consumer = torch.Stream(device=f"flagos:{device}")
    assert producer.stream_id != consumer.stream_id
    cpu = torch.arange(1 << 20, dtype=torch.float32) / (1 << 20)
    source = cpu.to(f"flagos:{device}")
    retained = []
    with torch.no_grad():
        for _ in range(8):
            with producer:
                value = source * 2
                ready = producer.record_event()
            with consumer:
                consumer.wait_event(ready)
                result = value.cos()
                retained.append(result)
            value.record_stream(consumer)
            assert torch_fl.flagos.current_device() == initial
            del value
            with producer:
                replacement = source * 3
            del replacement
    producer.synchronize()
    consumer.synchronize()
    assert producer.query() and consumer.query()
    expected = (cpu * 2).cos()
    for result in retained:
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-6)
