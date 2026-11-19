# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Installed HCU host launch ABI, eager correctness and resource lifetime."""

import importlib
import gc
import struct
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
import torch
import torch_fl

pytestmark = [pytest.mark.dcu, pytest.mark.native_source_host, pytest.mark.flaggems]


def launcher():
    return importlib.import_module(torch_fl.native_host._hosts.package + "._launcher")


def metadata(**kwargs):
    return SimpleNamespace(
        **{
            "num_warps": 1,
            "warp_size": 64,
            "num_ctas": 1,
            "shared": 0,
            "profile_scratch_size": 0,
            "profile_scratch_align": 1,
            **kwargs,
        }
    )


def test_hcu_contract_and_nonboxing_routes():
    from torch_fl import _build_config
    from torch_fl.native_host.backend import COMPILED_FIELDS

    assert _build_config.NATIVE_HOSTS
    assert "boxing" not in _build_config.KERNELS
    record = torch_fl.native_host._hosts.manifest["backend"]
    assert record["vendor"] == "hygon" and record["compiler_backend"] == "hip"
    assert record["abi_version"] == 2 and record["capabilities"] == 187
    assert launcher().backend_contract() == {
        name: record[name] for name in COMPILED_FIELDS
    }


@pytest.mark.parametrize(
    "kind,value",
    [
        ("i1", True),
        ("i8", -128),
        ("u8", 255),
        ("i16", -32768),
        ("u16", 65535),
        ("i32", -(2**31)),
        ("u32", 2**32 - 1),
        ("i64", -(2**63)),
        ("u64", -1),
        ("fp32", -1.5),
        ("fp64", 2**53 + 0.0),
        ("fp64", float("nan")),
        ("fp64", float("inf")),
    ],
)
def test_hcu_native_compact_arguments_match_original_keys(kind, value):
    module = launcher()
    tensor = torch.empty(1, device="flagos")
    args, kwargs = (tensor, value), {"warmup": False}
    cache = module.KernelCache(("tensor", "value"), {}, False)
    key = cache.run(args, kwargs, (0,), 0)
    call = module.Launcher(
        SimpleNamespace(signature={"tensor": "*fp32", "value": kind}), metadata()
    )
    cache.prepare(key, args, kwargs, (None, call, 0, None, {}, {}, {}, ()))

    def miss(*args, **kwargs):
        raise RuntimeError("unexpected native key miss")

    slot = module.register_script_kernel(SimpleNamespace(native_cache=cache, run=miss))
    # A zero grid exercises C++ argument storage and scratch-pointer binding.
    torch.ops.flagos_source_native.launch(slot, list(args), {}, [0])


def cache_probe():
    module = launcher()
    cache = module.KernelCache(("tensor", "value"), {}, False)
    call = module.Launcher(
        SimpleNamespace(signature={"tensor": "*fp32", "value": "constexpr"}),
        metadata(),
    )

    def miss(*args, **kwargs):
        raise RuntimeError("specialization metadata changed")

    slot = module.register_script_kernel(SimpleNamespace(native_cache=cache, run=miss))

    def seed(tensor, value, options=None):
        kwargs = dict(options or {})
        kwargs.setdefault("warmup", False)
        key = cache.run((tensor, value), kwargs, (0,), tensor.device.index)
        assert isinstance(key, bytes)
        cache.prepare(
            key, (tensor, value), kwargs, (None, call, 0, None, {}, {}, {}, ())
        )

    def invoke(tensor, value, options=None):
        torch.ops.flagos_source_native.launch(slot, [tensor, value], options or {}, [0])

    return seed, invoke


@pytest.mark.parametrize("change", ["shape", "rank", "stride", "dtype", "alignment"])
def test_hcu_specialization_rechecks_current_tensor_metadata(change):
    seed, invoke = cache_probe()
    tensor = torch.empty(4, 4, device="flagos")
    seed(tensor, 1)
    invoke(torch.empty_like(tensor), 1)
    changed = {
        "shape": lambda: tensor.view(2, 8),
        "rank": lambda: tensor.view(16),
        "stride": tensor.t,
        "dtype": lambda: tensor.to(torch.bfloat16),
        "alignment": lambda: torch.empty(17, device="flagos")[1:].view(4, 4),
    }[change]()
    with pytest.raises(RuntimeError, match="specialization metadata changed"):
        invoke(changed, 1)
    invoke(tensor, 1)


def scalar_bits(bits):
    return struct.unpack("d", struct.pack("Q", bits))[0]


@pytest.mark.parametrize(
    "initial,changed",
    [
        (1, True),
        (True, 1),
        (1, 1.0),
        (False, 0),
        (None, False),
        (0.0, -0.0),
        (-0.0, 0.0),
        (1.5, 1.5000000000000002),
        ("a\0b", "a\0c"),
        (scalar_bits(0x7FF8000000000001), scalar_bits(0x7FF8000000000002)),
        (float("inf"), -float("inf")),
    ],
)
def test_hcu_specialization_preserves_scalar_types_and_bits(initial, changed):
    seed, invoke = cache_probe()
    tensor = torch.empty(4, device="flagos")
    seed(tensor, initial)
    invoke(tensor, initial)
    with pytest.raises(RuntimeError, match="specialization metadata changed"):
        invoke(tensor, changed)


def test_hcu_specialization_preserves_keyword_order_and_warmup_semantics():
    seed, invoke = cache_probe()
    tensor = torch.empty(4, device="flagos")
    seed(tensor, 1, {"tile": 32, "enabled": True})
    invoke(tensor, 1, {"tile": 32, "enabled": True})
    invoke(tensor, 1, {"enabled": True, "tile": 32, "warmup": False})
    for options in (
        {"tile": 64, "enabled": True},
        {"tile": 32, "enabled": 1},
        {"tile": 32, "enabled": True, "warmup": True},
    ):
        with pytest.raises(RuntimeError, match="specialization metadata changed"):
            invoke(tensor, 1, options)


def test_hcu_specialization_retains_metadata_without_retaining_input_tensors():
    seed, invoke = cache_probe()
    tensor = torch.empty(4, device="flagos")
    reference = weakref.ref(tensor)
    seed(tensor, 1)
    del tensor
    gc.collect()
    assert reference() is None
    invoke(torch.empty(4, device="flagos"), 1)


def test_hcu_specialization_handles_many_colliding_scalar_variants():
    seed, invoke = cache_probe()
    tensor = torch.empty(4, device="flagos")
    for value in range(16):
        seed(tensor, value)
    for value in reversed(range(16)):
        invoke(tensor, value)


def test_hcu_specialization_lookup_is_safe_during_concurrent_preparation():
    seed, invoke = cache_probe()
    tensor = torch.empty(4, device="flagos")
    seed(tensor, 0)
    started = Event()

    def prepare():
        started.set()
        for value in range(1, 32):
            seed(tensor, value)

    def launch():
        assert started.wait(5)
        for _ in range(100):
            invoke(tensor, 0)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(prepare), pool.submit(launch), pool.submit(launch)]
        for future in futures:
            future.result(timeout=30)


@pytest.mark.parametrize(
    "kind,value",
    [
        ("i1", True),
        ("i8", -128),
        ("u8", 255),
        ("i16", -32768),
        ("u16", 65535),
        ("i32", -(2**31)),
        ("u32", 2**32 - 1),
        ("i64", -(2**63)),
        ("u64", 2**64 - 1),
        ("fp16", 1.5),
        ("bf16", 1.5),
        ("fp32", -1.5),
        ("fp64", 1.25),
    ],
)
def test_hcu_scalar_abi_on_zero_grid(kind, value):
    call = launcher().Launcher(SimpleNamespace(signature={"value": kind}), metadata())
    call(0, 1, 1, 0, 0, None, None, None, None, value)


@pytest.mark.parametrize(
    "fields",
    [
        {"num_warps": 0},
        {"num_warps": 3},
        {"warp_size": 16},
        {"num_ctas": 0},
        {"global_scratch_size": 1},
    ],
)
def test_hcu_rejects_invalid_geometry_and_unsupported_scratch(fields):
    with pytest.raises(ValueError, match="geometry|global scratch"):
        launcher().Launcher(SimpleNamespace(signature={}), metadata(**fields))


def test_hcu_profile_scratch_alignment_rejection_leaks_no_memory():
    call = launcher().Launcher(
        SimpleNamespace(signature={}),
        metadata(profile_scratch_size=64, profile_scratch_align=512),
    )
    allocated = torch_fl.flagos.memory_allocated()
    for _ in range(3):
        with pytest.raises(ValueError, match="scratch alignment"):
            call(1, 1, 1, 0, 0, None, None, None, None)
        assert torch_fl.flagos.memory_allocated() == allocated


@pytest.mark.parametrize("hardware", [False, True])
def test_hcu_descriptors_and_invalid_rank_before_submission(hardware):
    from triton.tools.tensor_descriptor import TensorDescriptor

    value = torch.empty(8, 32, device="flagos")
    descriptor = TensorDescriptor(value, [8, 32], [32, 1], [4, 16])
    meta = (
        {"elem_bits": 32, "block_size": [4, 16], "interval_padding_pairs": []}
        if hardware
        else None
    )
    call = launcher().Launcher(
        SimpleNamespace(signature={"descriptor": "tensordesc<fp32[2]>"}),
        metadata(tensordesc_meta=[meta]),
    )
    call(0, 1, 1, 0, 0, None, None, None, None, descriptor)
    if hardware:
        nan = TensorDescriptor(value, [8, 32], [32, 1], [4, 16], padding="nan")
        with pytest.raises(ValueError, match="zero padding"):
            call(0, 1, 1, 0, 0, None, None, None, None, nan)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(), (0,), (137,), (3, 5, 7), (19, 31)])
def test_hcu_eager_cos_uses_current_values_and_retains_outputs(dtype, shape):
    cpu = torch.randn(shape, dtype=dtype)
    value = cpu.to("flagos")
    artifacts = torch_fl.native_host.audit()["host_artifacts"]
    hits = torch.ops.flagos.native_source_typed_hits("aten::cos")
    with torch.no_grad():
        results = [torch.cos(value) for _ in range(3)]
    torch_fl.flagos.synchronize()
    for result in results:
        torch.testing.assert_close(
            result.cpu(),
            cpu.cos(),
            rtol=1e-5 if dtype == torch.float32 else 1e-2,
            atol=1e-6 if dtype == torch.float32 else 1e-2,
        )
    if cpu.numel():
        assert len({result.data_ptr() for result in results}) == 3
    assert torch.ops.flagos.native_source_typed_hits("aten::cos") == hits + 3
    assert torch_fl.native_host.audit()["host_artifacts"] == artifacts


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hcu_transposed_mm_on_nondefault_stream(dtype):
    stream = torch.Stream(device="flagos:0")
    retained = []
    hits = torch.ops.flagos.native_source_typed_hits("aten::mm")
    with torch.no_grad(), stream:
        for m, k, n in ((17, 128, 64), (3, 64, 32), (17, 128, 64)):
            a = torch.randn(m, k, dtype=dtype)
            b = torch.randn(n, k, dtype=dtype)
            actual = torch.mm(a.to("flagos"), b.to("flagos").t())
            retained.append((actual, a.float() @ b.float().t()))
    stream.synchronize()
    for actual, expected in retained:
        torch.testing.assert_close(
            actual.cpu().float(),
            expected,
            rtol=1e-4 if dtype == torch.float32 else 3e-2,
            atol=1e-4 if dtype == torch.float32 else 5e-2,
        )
    assert len({actual.data_ptr() for actual, _ in retained}) == len(retained)
    hosts = torch_fl.native_host._hosts
    typed = any(entry["op"] == "aten::mm" for entry in hosts.typed["typed_entries"])
    assert torch.ops.flagos.native_source_typed_hits("aten::mm") == hits + (
        3 if typed else 0
    )
    # A source signature outside typed lowering still uses its prebuilt Cython
    # host and the same HCU launcher; report it separately from a typed hit.
    assert hosts.op("mm").__module__.startswith(hosts.package)
    if dtype != torch.bfloat16 or not torch_fl._build_config.NATIVE_VENDOR:
        assert any("mm_kernel" in name for name in hosts.audit()["launched_kernels"])


@pytest.mark.min_devices(2)
@pytest.mark.parametrize("op", ["cos", "mul"])
def test_hcu_launch_restores_device_and_uses_the_target_stream(op):
    initial = torch_fl.flagos.current_device()
    target = (initial + 1) % torch_fl.flagos.device_count()
    stream = torch.Stream(device=f"flagos:{target}")
    retained = []
    try:
        with stream, torch.no_grad():
            for shape in ((17,), (3, 5, 7), (19, 31)):
                cpu = torch.randn(shape)
                value = cpu.to(f"flagos:{target}")
                torch_fl.flagos.set_device(initial)
                result = torch.cos(value) if op == "cos" else torch.mul(value, value)
                expected = cpu.cos() if op == "cos" else cpu * cpu
                retained.append((result, expected))
                assert result.device.index == target
                assert torch_fl.flagos.current_device() == initial
                assert (
                    torch.accelerator.current_stream(target).stream_id
                    == stream.stream_id
                )
        stream.synchronize()
        for actual, expected in retained:
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
    finally:
        torch_fl.flagos.set_device(initial)


def test_hcu_pointwise_rank_cache_validates_live_dictionaries(monkeypatch):
    hosts = torch_fl.native_host._hosts
    owner = hosts.resolve(hosts.original_ops["cos"].__globals__["cos_func"])
    original = owner.instantiate
    calls = []

    def instantiate(rank):
        calls.append(rank)
        return original(rank)

    monkeypatch.setattr(owner, "instantiate", instantiate)
    owner.overloads.clear()
    cpu = torch.randn(137)
    value = cpu.to("flagos")
    with torch.no_grad():
        for _ in range(3):
            torch.cos(value)
        warmed = len(calls)
        retained = [torch.cos((cpu + i).to("flagos")) for i in range(5)]
        assert len(calls) == warmed
        monkeypatch.setattr(owner.config, "balance_grid", not owner.config.balance_grid)
        changed = torch.cos(value)
        assert len(calls) > warmed
    for i, result in enumerate(retained):
        torch.testing.assert_close(result.cpu(), (cpu + i).cos(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(changed.cpu(), cpu.cos(), rtol=1e-5, atol=1e-6)
    assert len({result.data_ptr() for result in retained}) == 5


def test_hcu_concurrent_pointwise_change_is_not_cached(monkeypatch):
    hosts = torch_fl.native_host._hosts
    owner = hosts.resolve(hosts.original_ops["cos"].__globals__["cos_func"])
    value = torch.randn(137).to("flagos")
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
    third = execute()
    assert calls == [1, 1]
    for result in (first, second, third):
        torch.testing.assert_close(
            result.cpu(), value.cpu().cos(), rtol=1e-5, atol=1e-6
        )
    assert len({result.data_ptr() for result in (first, second, third)}) == 3
