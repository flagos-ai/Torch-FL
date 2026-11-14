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
