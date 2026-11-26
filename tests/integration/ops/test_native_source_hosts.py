# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Runtime contracts of the precompiled source-host wheel."""

import sys
import importlib
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest
import torch
import torch_fl

pytestmark = [pytest.mark.nvidia, pytest.mark.native_source_host, pytest.mark.flaggems]


def _foreign_alias_roundtrip(
    value: torch.Tensor, slot: int, packed: bool
) -> Tuple[List[int], List[int], List[int], bool]:
    shared = [16]
    if packed:
        fields: List[Any] = [
            "__flagos_descriptor_v2__",
            value,
            [16],
            [1],
            shared,
            "zero",
        ]
        descriptor: Any = fields
    else:
        mapping: Dict[str, Any] = {
            "__flagos_tensor_descriptor_v1__": True,
            "base": value,
            "shape": [16],
            "strides": [1],
            "block_shape": shared,
            "padding": "zero",
        }
        descriptor = mapping
    nested: Dict[str, Any] = {"prior": shared}
    args: List[Any] = [descriptor, shared, nested]
    kwargs: Dict[str, Any] = {}
    result = torch.ops.flagos_source_native.foreign_value(slot, args, kwargs)
    assert torch.jit.isinstance(result, Tuple[Dict[str, Any], List[int]])
    returned, replacement = result
    returned["after"] = True
    replacement.append(3)
    latest = nested["latest"]
    after = nested["after"]
    assert torch.jit.isinstance(latest, List[int])
    assert torch.jit.isinstance(after, bool)
    if torch.jit.isinstance(descriptor, Dict[str, Any]):
        block = descriptor["block_shape"]
        assert torch.jit.isinstance(block, List[int])
        return shared, block, latest, after
    else:
        assert torch.jit.isinstance(descriptor, List[Any])
        block = descriptor[4]
        assert torch.jit.isinstance(block, List[int])
        return shared, block, latest, after


@pytest.mark.parametrize("packed", [False, True])
def test_foreign_helper_preserves_mutable_inputs_and_return_aliases(packed):
    launcher = importlib.import_module(
        torch_fl.native_host._hosts.package + "._launcher"
    )

    def metadata(descriptor, shared, nested):
        assert descriptor.block_shape is shared is nested["prior"]
        shared.append(4)
        replacement = [8]
        descriptor.block_shape = replacement
        nested["latest"] = replacement
        return nested, replacement

    slot = launcher.register_foreign(metadata, ())
    function = torch.jit.script(_foreign_alias_roundtrip)
    with torch.jit.optimized_execution(False):
        result = function(torch.empty(16, device="flagos"), slot, packed)
    assert result == ([16, 4], [8, 3], [8, 3], True)


@pytest.mark.parametrize("shape", [(), (0,), (137,), (2, 11, 37)])
def test_unseen_shapes_keep_host_artifacts(shape):
    before = torch_fl.native_host.audit()["host_artifacts"]
    x = torch.randn(shape).to("flagos")
    with torch.no_grad():
        actual = torch.cos(x)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(actual.cpu(), x.cpu().cos(), rtol=1e-5, atol=1e-6)
    assert torch_fl.native_host.audit()["host_artifacts"] == before


def test_broadcast_strides_and_retained_outputs():
    a = torch.arange(231, dtype=torch.float32).reshape(11, 21).t().to("flagos")
    b = torch.arange(11, dtype=torch.float32).to("flagos")
    with torch.no_grad():
        first = a + b
        second = a + b * 2
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(first.cpu(), a.cpu() + b.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(second.cpu(), a.cpu() + b.cpu() * 2, rtol=0, atol=0)
    assert first.data_ptr() != second.data_ptr()


def test_new_pointwise_specialization_preserves_noncontiguous_layout():
    cpu = torch.randn(3, 5, 7, 11, dtype=torch.bfloat16).transpose(1, 2)
    value = cpu.to("flagos")
    before = torch_fl.native_host.audit()["host_artifacts"]
    actual = value.pow(2)
    torch_fl.flagos.synchronize()
    assert actual.stride() == value.stride()
    torch.testing.assert_close(actual.cpu(), cpu.pow(2), rtol=0, atol=0)
    assert torch_fl.native_host.audit()["host_artifacts"] == before


def test_pointwise_promotion_matches_source_across_default_dtypes():
    import flag_gems

    original = torch.get_default_dtype()
    integers = torch.tensor([1, 2, 4], dtype=torch.int32, device="flagos")
    hosts = torch_fl.native_host._hosts
    promotion = hosts.originals["flag_gems.utils.pointwise_dynamic"].type_promotion
    owner = hosts.original_ops["cos"].__globals__["cos_func"]
    method = owner.fx._promotion_methods[0][-1]
    try:
        for dtype in (torch.float32, torch.float64, torch.float32):
            torch.set_default_dtype(dtype)
            actual = flag_gems.cos(integers)
            _, output_dtype = promotion(integers.cpu(), type_promotion=method)
            expected = integers.cpu().to(output_dtype).cos()
            torch_fl.flagos.synchronize()
            assert actual.dtype == expected.dtype
            torch.testing.assert_close(actual.cpu(), expected)
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "layout",
    ["dense", "transpose", "channels_last", "slice", "expand", "scalar", "empty"],
)
def test_pointwise_output_preserves_layout_and_ownership(dtype, layout):
    def view(value):
        if layout == "transpose":
            return value.transpose(1, 2)
        if layout == "channels_last":
            return value.contiguous(memory_format=torch.channels_last)
        if layout == "slice":
            return value[:, :, ::2, ::2]
        if layout == "expand":
            return value[:1, :1].expand(3, 5, 7, 11)
        if layout == "scalar":
            return value[0, 0, 0, 0]
        if layout == "empty":
            return value[:, :0]
        return value

    cpu = torch.arange(3 * 5 * 7 * 11).reshape(3, 5, 7, 11).to(dtype)
    value = view(cpu.to("flagos"))
    cpu = view(cpu)
    with torch.no_grad():
        actual = torch.add(value, 0.5)
        retained = torch.add(value, 1.0)
    torch_fl.flagos.synchronize()
    assert actual.stride() == torch.empty_like(cpu).stride()
    torch.testing.assert_close(actual.cpu(), cpu + 0.5, rtol=0, atol=0)
    if actual.numel():
        assert actual.data_ptr() != retained.data_ptr()


def test_pointwise_promotion_distinguishes_scalar_rank():
    value = torch.arange(7, dtype=torch.float16, device="flagos")
    with torch.no_grad():
        for shape in ((), (1,), ()):
            other = torch.ones(shape, dtype=torch.float32, device="flagos")
            actual = value + other
            expected = value.cpu() + other.cpu()
            assert actual.dtype == expected.dtype
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_pointwise_keeps_preallocated_output_dtype():
    hosts = torch_fl.native_host._hosts
    launcher = importlib.import_module(hosts.package + "._launcher")
    owner = hosts.resolve(hosts.original_ops["frexp"].__globals__["_frexp_func"])
    slot = launcher.register_pointwise(owner)
    value = torch.arange(7, dtype=torch.float32, device="flagos")
    exponent = torch.empty_like(value, dtype=torch.int32)
    actual = torch.ops.flagos_source_native.pointwise_apply(
        slot, [value], {"out1": exponent}
    )
    expected = torch.frexp(value.cpu())
    assert actual[1].data_ptr() == exponent.data_ptr()
    for result, reference in zip(actual, expected):
        assert result.dtype == reference.dtype
        torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)


def _published_source_module():
    hosts = torch_fl.native_host._hosts
    assert hosts.manifest.get("source_variants"), "missing published source policy"
    return importlib.import_module(hosts.package + ".typed._source_hosts")


@pytest.mark.native_vendor_precision
@pytest.mark.parametrize("left_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("right_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "layout", ["dense", "transpose", "slice", "expand", "broadcast", "scalar", "empty"]
)
def test_published_mul_keeps_values_dtypes_and_storage(left_dtype, right_dtype, layout):
    module = _published_source_module()
    left = (torch.arange(35).reshape(5, 7) / 8 - 2).to(left_dtype)
    right = (torch.arange(35).reshape(5, 7) / 16 + 0.5).to(right_dtype)

    def views(left, right):
        if layout == "transpose":
            return left.t(), right.t()
        if layout == "slice":
            return left[:, ::2], right[:, ::2]
        if layout == "expand":
            return left[:1].expand(5, 7), right
        if layout == "broadcast":
            return left[:, :1], right[:1]
        if layout == "scalar":
            return left, right[0, 0]
        if layout == "empty":
            return left[:0], right[:0]
        return left, right

    a, b = views(left.to("flagos"), right.to("flagos"))
    left, right = views(left, right)
    identity = [(v.data_ptr(), v.device, v.stride(), v._version) for v in (a, b)]
    before = module.source_variant_hits()["published_mul"]
    with torch.no_grad():
        actual = a * b
        retained = a * b
    assert module.source_variant_hits()["published_mul"] == before + 2
    assert identity == [
        (v.data_ptr(), v.device, v.stride(), v._version) for v in (a, b)
    ]
    torch.testing.assert_close(actual.cpu(), left * right, atol=0, rtol=0)
    torch.testing.assert_close(retained.cpu(), actual.cpu(), atol=0, rtol=0)
    if actual.numel():
        assert actual.data_ptr() != retained.data_ptr()


@pytest.mark.native_vendor_precision
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
def test_published_mul_keeps_unmeasured_dtypes_on_original_source(dtype):
    module = _published_source_module()
    a = torch.arange(9, dtype=dtype, device="flagos")
    before = module.source_variant_hits()["published_mul"]
    with torch.no_grad():
        actual = a * a
    assert module.source_variant_hits()["published_mul"] == before
    torch.testing.assert_close(actual.cpu(), a.cpu() * a.cpu(), atol=0, rtol=0)


@pytest.mark.native_vendor_precision
def test_published_mul_does_not_replace_scalar_or_grad_enabled_routes():
    module = _published_source_module()
    a = torch.ones(9, dtype=torch.float32, device="flagos")
    before = module.source_variant_hits()["published_mul"]
    with torch.no_grad():
        scalar = a * 2.0
        mixed = a * torch.ones(9, dtype=torch.int32, device="flagos")
    with torch.enable_grad():
        grad_enabled = a * a
    assert module.source_variant_hits()["published_mul"] == before
    for actual, value in ((scalar, 2.0), (mixed, 1.0), (grad_enabled, 1.0)):
        torch.testing.assert_close(
            actual.cpu(), torch.full((9,), value), atol=0, rtol=0
        )


@pytest.mark.min_devices(2)
@pytest.mark.native_vendor_precision
def test_published_mul_preserves_ambient_device_and_current_stream():
    module = _published_source_module()
    initial = torch_fl.flagos.current_device()
    other = (initial + 1) % torch_fl.flagos.device_count()
    device = torch.device(f"flagos:{other}")
    stream = torch.Stream(device=device)
    before = module.source_variant_hits()["published_mul"]
    with stream, torch.no_grad():
        a = torch.full((5, 7), 2.0, dtype=torch.bfloat16, device=device)
        b = torch.full((5, 7), 3.0, dtype=torch.float32, device=device)
        torch_fl.flagos.set_device(initial)
        actual = a * b
        assert torch_fl.flagos.current_device() == initial
        assert torch.accelerator.current_stream(device).stream_id == stream.stream_id
    assert actual.device == device
    assert module.source_variant_hits()["published_mul"] == before + 1
    torch.testing.assert_close(actual.cpu(), torch.full((5, 7), 6.0), atol=0, rtol=0)


@pytest.mark.min_devices(2)
def test_empty_strided_preserves_current_device():
    initial = torch_fl.flagos.current_device()
    other = (initial + 1) % torch_fl.flagos.device_count()
    with torch_fl.flagos.device(other):
        explicit = torch.empty_strided((7, 3), (1, 7), device=f"flagos:{initial}")
        assert explicit.device.index == initial
        assert torch_fl.flagos.current_device() == other
        implicit = torch.empty_strided((7, 3), (1, 7), device="flagos")
        assert implicit.device.index == other
        assert torch_fl.flagos.current_device() == other
    assert torch_fl.flagos.current_device() == initial


def test_alias_and_slice_preserve_storage():
    x = torch.arange(33, dtype=torch.float32).to("flagos")
    alias = torch.ops.aten.alias(x)
    view = x[1:25:3]
    assert alias._cdata != x._cdata
    assert alias.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert view.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    torch.testing.assert_close(view.cpu(), x.cpu()[1:25:3], rtol=0, atol=0)


def test_mutation_uses_current_stream():
    x = torch.zeros(257, device="flagos")
    source = torch.arange(257, dtype=torch.float32).to("flagos")
    torch_fl.flagos.synchronize()
    default = torch.accelerator.current_stream(x.device)
    stream = torch.Stream(device=x.device)
    assert stream.stream_id != default.stream_id
    launcher = importlib.import_module(
        torch_fl.native_host._hosts.package + "._launcher"
    )
    with torch.no_grad():
        torch.cos(x)
    with stream, torch.no_grad():
        assert launcher.current_stream(x.device.index) == stream.stream_id
        assert torch.cuda.current_stream(x.device.index).cuda_stream == stream.stream_id
        result = x.copy_(source)
        output = torch.cos(x)
    stream.synchronize()
    assert result is x
    torch.testing.assert_close(x.cpu(), source.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(output.cpu(), source.cpu().cos(), rtol=1e-5, atol=1e-6)


def test_concurrent_calls_use_current_addresses_and_streams():
    inputs = [torch.randn(131 + index).to("flagos") for index in range(4)]
    torch_fl.flagos.synchronize()
    before = torch_fl.native_host.audit()["host_artifacts"]

    def execute(value):
        stream = torch.Stream(device=value.device)
        with stream, torch.no_grad():
            results = [torch.cos(value) for _ in range(3)]
        stream.synchronize()
        assert len({result.data_ptr() for result in results}) == len(results)
        return [result.cpu() for result in results]

    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = list(executor.map(execute, inputs))
    for value, results in zip(inputs, outputs):
        for result in results:
            torch.testing.assert_close(result, value.cpu().cos(), rtol=1e-5, atol=1e-6)
    assert torch_fl.native_host.audit()["host_artifacts"] == before


def test_model_host_accepts_unseen_matmul_dimensions_without_host_compilation():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return torch.cos(a @ b)

    model = Model().eval()
    assert torch_fl.native_host.compile(model) is model
    before = torch_fl.native_host.audit()["host_artifacts"]
    outputs = []
    with torch.no_grad():
        for m, k, n in ((7, 13, 17), (11, 19, 17), (23, 31, 17)):
            a = torch.randn(m, k, dtype=torch.bfloat16).to("flagos")
            b = torch.randn(k, n, dtype=torch.bfloat16).to("flagos")
            expected = torch.cos(a @ b)
            actual = model(a, b)
            torch_fl.flagos.synchronize()
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
            outputs.append(actual)
    assert len({output.data_ptr() for output in outputs}) == len(outputs)
    stream = torch.Stream(device=a.device)
    with stream, torch.no_grad():
        on_stream = model(a, b)
    stream.synchronize()
    torch.testing.assert_close(on_stream.cpu(), expected.cpu(), rtol=0, atol=0)
    assert len(model.forward._flagos_host_backend.graphs) == 1
    assert torch_fl.native_host.audit()["host_artifacts"] == before
    with torch.enable_grad(), pytest.raises(
        RuntimeError, match="require torch.no_grad"
    ):
        model(a, b)


@pytest.mark.cpython_dict_watchers
def test_pointwise_config_change_during_instantiation_is_not_cached(monkeypatch):
    hosts = torch_fl.native_host._hosts
    owner = hosts.resolve(hosts.original_ops["cos"].__globals__["cos_func"])
    value = torch.randn(137).to("flagos")
    with torch.no_grad():
        torch.cos(value)
    torch_fl.flagos.synchronize()
    original = owner.instantiate
    entered, resume = Event(), Event()
    calls = []

    def instantiate(rank):
        result = original(rank)
        calls.append(rank)
        if len(calls) == 1:
            entered.set()
            if not resume.wait(30):
                raise RuntimeError("pointwise invalidation test timed out")
        return result

    def execute():
        with torch.no_grad():
            return torch.cos(value)

    monkeypatch.setattr(owner, "instantiate", instantiate)
    owner.overloads.clear()
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(execute)
        try:
            assert entered.wait(30)
            monkeypatch.setattr(
                owner.config, "balance_grid", not owner.config.balance_grid
            )
        finally:
            resume.set()
        first = pending.result(timeout=30)
    second = execute()
    assert calls == [1, 1]
    third = execute()
    assert calls == [1, 1]
    torch_fl.flagos.synchronize()
    assert len({output.data_ptr() for output in (first, second, third)}) == 3
    for output in (first, second, third):
        torch.testing.assert_close(
            output.cpu(), value.cpu().cos(), rtol=1e-5, atol=1e-6
        )


def test_missing_route_never_reaches_cpu():
    library = torch.library.Library("native_source_host_test", "DEF")
    library.define("missing(Tensor value) -> Tensor")
    library.impl("missing", lambda value: value.clone(), "CPU")
    x = torch.empty(1, device="flagos")
    with pytest.raises(RuntimeError, match="do not permit CPU fallback"):
        torch.ops.native_source_host_test.missing(x)


def test_warm_kernel_does_not_execute_flaggems_python():
    x = torch.randn(263).to("flagos")
    with torch.no_grad():
        torch.cos(x)
        torch.cos(x)
        frames = []

        def record(frame, event, arg):
            if event == "call" and any(
                part in frame.f_code.co_filename for part in ("/flag_gems/", "/triton/")
            ):
                frames.append(frame.f_code.co_filename + ":" + frame.f_code.co_name)

        sys.setprofile(record)
        try:
            torch.cos(x)
        finally:
            sys.setprofile(None)
    assert not frames, frames
    assert torch_fl.native_host.audit()["launched_kernels"]


def _typed_hits(op):
    return torch.ops.flagos.native_source_typed_hits("aten::" + op)


@pytest.mark.parametrize(
    "source_dtype,target_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.int32, torch.float32),
        (torch.bool, torch.int64),
    ],
)
def test_dtype_cast_uses_generated_source_bridge(source_dtype, target_dtype):
    value = torch.arange(257).to(source_dtype).to("flagos")
    before = torch_fl.native_host.audit()["host_artifacts"]
    with torch.no_grad():
        hits = _typed_hits("copy_")
        legacy = torch.ops.flagos.native_source_stats()[1]
        first = value.to(target_dtype)
        second = value.to(target_dtype)
        assert _typed_hits("copy_") == hits + 2
        assert torch.ops.flagos.native_source_stats()[1] == legacy
    torch_fl.flagos.synchronize()
    expected = value.cpu().to(target_dtype)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(second.cpu(), expected, rtol=0, atol=0)
    assert torch_fl.native_host.audit()["host_artifacts"] == before


def test_generated_copy_bridge_preserves_strides_alias_and_version():
    value = torch.arange(35, dtype=torch.float32).to("flagos").reshape(5, 7).t()
    assert not value.is_contiguous()
    target = torch.empty(value.shape, device="flagos", dtype=torch.float32)
    address, version = target.data_ptr(), target._version
    with torch.no_grad():
        hits = _typed_hits("copy_")
        legacy = torch.ops.flagos.native_source_stats()[1]
        returned = target.copy_(value, non_blocking=True)
        assert _typed_hits("copy_") == hits + 1
        assert torch.ops.flagos.native_source_stats()[1] == legacy
    torch_fl.flagos.synchronize()
    assert returned is target and target.data_ptr() == address
    assert target._version > version
    torch.testing.assert_close(target.cpu(), value.cpu(), rtol=0, atol=0)
    with torch.no_grad():
        hits = _typed_hits("copy_")
        assert target.copy_(target) is target
        assert _typed_hits("copy_") == hits


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_extra_python_default_preserves_round_and_inplace(dtype):
    cpu = torch.tensor(
        [
            float("-inf"),
            -3.5,
            -2.5,
            -0.5,
            -0.0,
            0.0,
            0.125,
            0.5,
            2.5,
            3.5,
            float("inf"),
            float("nan"),
        ],
        dtype=dtype,
    )
    value = cpu.to("flagos")
    originals = torch_fl.native_host._hosts.original_ops
    artifacts = torch_fl.native_host.audit()["host_artifacts"]
    with torch.no_grad():
        expected = originals["round"](value)
        hits = _typed_hits("round")
        actual = torch.round(value)
        assert _typed_hits("round") == hits + 1
        inplace = value.clone()
        address, version = inplace.data_ptr(), inplace._version
        hits = _typed_hits("round_")
        returned = inplace.round_()
        assert _typed_hits("round_") == hits + 1
    torch_fl.flagos.synchronize()
    assert returned is inplace and inplace.data_ptr() == address
    assert inplace._version > version
    for got, reference in ((actual, expected), (inplace, expected)):
        torch.testing.assert_close(
            got.cpu(), reference.cpu(), rtol=0, atol=0, equal_nan=True
        )
    torch.testing.assert_close(
        actual.cpu(), cpu.round(), rtol=0, atol=0, equal_nan=True
    )
    assert torch_fl.native_host.audit()["host_artifacts"] == artifacts


@pytest.mark.parametrize("alpha", [2, 0.25])
def test_typed_scalar_keyword_avoids_generic_entry(alpha):
    value = torch.arange(21, dtype=torch.float32).reshape(3, 7).to("flagos")
    other = torch.ones_like(value)
    with torch.no_grad():
        torch.add(value, other, alpha=alpha)
        before = _typed_hits("add.Tensor")
        legacy = torch.ops.flagos.native_source_stats()[1]
        actual = torch.add(value, other, alpha=alpha)
        assert _typed_hits("add.Tensor") == before + 1
        assert torch.ops.flagos.native_source_stats()[1] == legacy
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(actual.cpu(), value.cpu() + other.cpu() * alpha)


@pytest.mark.parametrize("dim", [None, [0], [0, 1]])
@pytest.mark.parametrize("correction", [0.0, 1.0, None])
def test_typed_tuple_optional_list_and_scalar(dim, correction):
    cpu = torch.arange(21, dtype=torch.float32).reshape(3, 7) / 16
    value = cpu.to("flagos")
    before = torch_fl.native_host.audit()["host_artifacts"]
    with torch.no_grad():
        expected = torch_fl.native_host._hosts.original_ops["var_mean"](
            value, dim=dim, correction=correction, keepdim=True
        )
        hits = _typed_hits("var_mean.correction")
        actual = torch.var_mean(value, dim=dim, correction=correction, keepdim=True)
        assert _typed_hits("var_mean.correction") == hits + 1
    torch_fl.flagos.synchronize()
    for got, original, reference in zip(
        actual,
        expected,
        torch.var_mean(cpu, dim=dim, correction=correction, keepdim=True),
    ):
        torch.testing.assert_close(got.cpu(), original.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(got.cpu(), reference)
    assert actual[0].data_ptr() != actual[1].data_ptr()
    assert torch_fl.native_host.audit()["host_artifacts"] == before


@pytest.mark.parametrize("correction", [0, 1, False, True])
@pytest.mark.parametrize(
    "op,host", [("var_mean", "var_mean"), ("var", "var_correction")]
)
def test_typed_reduction_preserves_exact_scalar_variant(correction, op, host):
    value = torch.arange(21, dtype=torch.float32).reshape(3, 7).to("flagos")
    hosts = torch_fl.native_host._hosts
    with torch.no_grad():
        expected = hosts.original_ops[host](value, dim=[0], correction=correction)
        before = _typed_hits(op + ".correction")
        actual = getattr(torch, op)(value, dim=[0], correction=correction)
        assert _typed_hits(op + ".correction") == before + 1
    torch_fl.flagos.synchronize()
    if isinstance(actual, tuple):
        for got, original in zip(actual, expected):
            torch.testing.assert_close(got.cpu(), original.cpu(), rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("end,dtype", [(7.5, None), (7.5, torch.int64), (0.0, None)])
def test_factory_retains_original_helper_dtype(end, dtype):
    # arange_start needs object semantics for float-to-int conversion. A typed
    # facade must not pass a raw dtype integer into that untyped helper.
    with torch.no_grad():
        expected = torch_fl.native_host._hosts.original_ops["arange"](
            end,
            dtype=dtype or torch.get_default_dtype(),
            device=torch.device("flagos:0"),
        )
        hits = _typed_hits("arange")
        actual = torch.arange(end, dtype=dtype, device="flagos:0")
        assert _typed_hits("arange") == hits
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [None, torch.float64])
@pytest.mark.parametrize("op", ["nansum", "sum.IntList_out"])
def test_object_reduction_helper_preserves_optional_dtype(op, dtype):
    cpu = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]])
    value = cpu.to("flagos")
    hosts = torch_fl.native_host._hosts
    with torch.no_grad():
        if op == "nansum":
            expected = hosts.original_ops["nansum"](value, dim=[0], dtype=dtype)
            hits = _typed_hits(op)
            actual = torch.nansum(value, dim=[0], dtype=dtype)
            reference = torch.nansum(cpu, dim=[0], dtype=dtype)
        else:
            out = torch.empty(3, device="flagos", dtype=dtype or cpu.dtype)
            expected = hosts.original_ops["sum_dim_out"](
                value, dim=[0], dtype=dtype, out=torch.empty_like(out)
            )
            hits = _typed_hits(op)
            # This schema is not enabled in the default conf. Validate its
            # compiled original host without claiming a new ATen route.
            assert not any(
                r["op"] == "aten::" + op for r in hosts.typed["typed_entries"]
            )
            actual = hosts.op("sum_dim_out")(value, dim=[0], dtype=dtype, out=out)
            assert actual is out
            reference = torch.sum(cpu, dim=[0], dtype=dtype)
        assert _typed_hits(op) == hits
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=0, atol=0, equal_nan=True
    )
    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "value,dtype",
    [
        (False, torch.bool),
        (True, None),
        (0, torch.bool),
        (2**53 + 3, torch.int64),
        (-(2**63), torch.int64),
        (0.0, torch.bool),
        (-0.0, torch.bool),
        (0.25, torch.bool),
        (float("inf"), torch.bool),
        (float("nan"), torch.bool),
        (0.25, None),
        (float("nan"), torch.float32),
        (float("inf"), torch.float64),
    ],
)
def test_typed_scalar_factory_keeps_values_dtypes_and_streams(value, dtype):
    before = torch_fl.native_host.audit()["host_artifacts"]
    stream = torch.Stream(device="flagos:0")
    with torch.no_grad(), stream:
        expected = torch_fl.native_host._hosts.original_ops["scalar_tensor"](
            value, dtype=dtype, device=torch.device("flagos:0")
        )
        hits = _typed_hits("scalar_tensor")
        actual = torch.scalar_tensor(value, dtype=dtype, device="flagos:0")
        assert _typed_hits("scalar_tensor") == hits + 1
    stream.synchronize()
    assert actual.device == expected.device == torch.device("flagos:0")
    assert actual.shape == expected.shape == ()
    assert actual.dtype == expected.dtype
    assert actual.data_ptr() != expected.data_ptr()
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=0, atol=0, equal_nan=True
    )
    torch.testing.assert_close(
        actual.cpu(),
        torch.scalar_tensor(value, dtype=dtype),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert torch_fl.native_host.audit()["host_artifacts"] == before


def test_typed_scalar_factory_reads_current_default_dtype():
    original = torch.get_default_dtype()
    try:
        with torch.no_grad():
            for dtype in (torch.float32, torch.float64, torch.float32):
                torch.set_default_dtype(dtype)
                hits = _typed_hits("scalar_tensor")
                actual = torch.scalar_tensor(0.25, device="flagos:0")
                assert _typed_hits("scalar_tensor") == hits + 1
                assert actual.dtype == dtype
                torch.testing.assert_close(actual.cpu(), torch.tensor(0.25))
    finally:
        torch.set_default_dtype(original)


def test_typed_tensorlist_keeps_borrowed_inputs_and_live_results():
    values = [
        torch.arange(15, dtype=torch.float32).reshape(3, 5).t().to("flagos")
        for _ in range(2)
    ]
    with torch.no_grad():
        hits = _typed_hits("cat")
        first = torch.cat(values, dim=1)
        second = torch.cat(values[::-1], dim=0)
        assert _typed_hits("cat") == hits + 2
    torch_fl.flagos.synchronize()
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first.cpu(), torch.cat([v.cpu() for v in values], dim=1))
    torch.testing.assert_close(
        second.cpu(), torch.cat([v.cpu() for v in values[::-1]], dim=0)
    )


def test_typed_out_and_inplace_preserve_aliases_and_versions():
    value = (torch.arange(31, dtype=torch.float32) / 16).to("flagos")
    out = torch.empty_like(value)
    address, version = out.data_ptr(), out._version
    with torch.no_grad():
        hits = _typed_hits("exp.out")
        result = torch.exp(value, out=out)
        assert _typed_hits("exp.out") == hits + 1
        assert result is out and result.data_ptr() == address
        assert out._version > version
        prior = out.clone()
        version = out._version
        hits = _typed_hits("add_.Tensor")
        mutated = out.add_(value, alpha=2)
        assert _typed_hits("add_.Tensor") == hits + 1
    torch_fl.flagos.synchronize()
    assert mutated is out and out.data_ptr() == address
    assert out._version > version
    torch.testing.assert_close(prior.cpu(), value.cpu().exp())
    torch.testing.assert_close(out.cpu(), prior.cpu() + value.cpu() * 2)


def test_typed_entry_preserves_grad_mode_fallback():
    value = torch.randn(37).to("flagos")
    with torch.enable_grad():
        before = _typed_hits("cos")
        actual = torch.cos(value)
        assert _typed_hits("cos") == before
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(actual.cpu(), value.cpu().cos())


def test_typed_entry_registration_rejects_mismatched_abi():
    hosts = torch_fl.native_host._hosts
    module = importlib.import_module(hosts.package + ".typed._source_hosts")
    entry = next(
        r for r in hosts.typed["typed_entries"] if r["op"] == "aten::mul.Tensor"
    )
    address = module.typed_callbacks[entry["op"]]
    before = torch.ops.flagos.native_source_typed_stats()
    register = torch.ops.flagos.native_source_register_typed
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        register(entry["op"], entry["host"], "wrong-abi", address)
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        register(entry["op"], "flag_gems.wrong_host", entry["abi"], address)
    with pytest.raises(RuntimeError, match="already registered"):
        register(entry["op"], entry["host"], entry["abi"], 1)
    with pytest.raises(RuntimeError, match="unknown typed source"):
        register(
            "aten::__missing_typed_source_test", entry["host"], entry["abi"], address
        )
    register(entry["op"], entry["host"], entry["abi"], address)
    assert torch.ops.flagos.native_source_typed_stats() == before


def test_direct_jit_warmup_does_not_launch_and_keeps_return_type():
    module = importlib.import_module("flag_gems.ops.scalar_tensor")
    entry = torch_fl.native_host._hosts.resolve(module.scalar_tensor_kernel)
    output = torch.zeros((), device="flagos")
    first = entry.warmup(output, 1.25, grid=(1,))
    second = entry.run(output, 1.25, grid=(1,), warmup=True)
    torch_fl.flagos.synchronize()
    assert first is second
    assert hasattr(first, "metadata")
    assert output.cpu().item() == 0
    launched = entry[(1,)](output, 1.25)
    assert launched is first
    assert entry[(1,)](output, 1.25) is first
    torch_fl.flagos.synchronize()
    assert output.cpu().item() == 1.25


def test_nested_abi_validation_without_a_device_launch():
    from torch_fl.native_host._generated._launcher import Launcher

    metadata = SimpleNamespace(
        num_warps=1,
        shared=0,
        global_scratch_size=0,
        global_scratch_align=1,
        profile_scratch_size=0,
        profile_scratch_align=1,
    )
    launcher = Launcher(
        SimpleNamespace(signature={"values": (("i32", "i32"),)}), metadata
    )
    # A zero grid exercises ABI packing without invoking any device kernel.
    launcher(0, 1, 1, 0, 0, (), None, None, None, ([3, 4],))
    with pytest.raises(IndexError):
        launcher(0, 1, 1, 0, 0, (), None, None, None, ((),))
    with pytest.raises(ValueError, match="tuple or list"):
        launcher(0, 1, 1, 0, 0, (), None, None, None, (3,))
