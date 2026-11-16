# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Current-call kernel keys and ABI packing, plus original FlagGems execution."""

import importlib
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
import torch_fl

pytestmark = [pytest.mark.nvidia, pytest.mark.native_source_host, pytest.mark.flaggems]


def _metadata():
    return SimpleNamespace(
        num_warps=1,
        shared=0,
        global_scratch_size=0,
        global_scratch_align=1,
        profile_scratch_size=0,
        profile_scratch_align=1,
    )


def test_installed_backend_contract_matches_the_built_adapter():
    from torch_fl.native_host.backend import COMPILED_FIELDS

    hosts = torch_fl.native_host._hosts
    launcher = importlib.import_module(hosts.package + "._launcher")
    record = hosts.manifest["backend"]
    assert launcher.backend_contract() == {
        name: record[name] for name in COMPILED_FIELDS
    }
    assert record["vendor"] == "nvidia" and record["compiler_backend"] == "cuda"
    assert torch_fl.native_host.audit()["backend"] == record


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
def test_backend_packs_scalar_widths_without_submitting_a_zero_grid(kind, value):
    launcher = importlib.import_module(
        torch_fl.native_host._hosts.package + "._launcher"
    )
    call = launcher.Launcher(SimpleNamespace(signature={"value": kind}), _metadata())
    call(0, 1, 1, 0, 0, None, None, None, None, value)


def test_backend_refuses_invalid_workspace_before_any_kernel_submission():
    launcher = importlib.import_module(
        torch_fl.native_host._hosts.package + "._launcher"
    )
    metadata = _metadata()
    metadata.global_scratch_size = 1
    metadata.global_scratch_align = 512
    call = launcher.Launcher(SimpleNamespace(signature={}), metadata)
    with pytest.raises(ValueError, match="scratch alignment"):
        call(1, 1, 1, 0, 0, None, None, None, None)
    metadata = _metadata()
    metadata.num_warps = 0
    with pytest.raises(ValueError, match="launch geometry"):
        launcher.Launcher(SimpleNamespace(signature={}), metadata)


def test_backend_releases_global_workspace_when_profile_workspace_is_rejected():
    launcher = importlib.import_module(
        torch_fl.native_host._hosts.package + "._launcher"
    )
    metadata = _metadata()
    metadata.global_scratch_size = 64
    metadata.global_scratch_align = 256
    metadata.profile_scratch_size = 64
    metadata.profile_scratch_align = 512
    call = launcher.Launcher(SimpleNamespace(signature={}), metadata)
    allocated = torch_fl.flagos.memory_allocated()
    for _ in range(3):
        with pytest.raises(ValueError, match="scratch alignment"):
            call(1, 1, 1, 0, 0, None, None, None, None)
        assert torch_fl.flagos.memory_allocated() == allocated


@pytest.mark.parametrize("padding", ["zero", "nan"])
def test_backend_encodes_hardware_descriptors_before_zero_grid_submission(padding):
    from triton.tools.tensor_descriptor import TensorDescriptor

    launcher = importlib.import_module(
        torch_fl.native_host._hosts.package + "._launcher"
    )
    value = torch.empty(3, 16, device="flagos")
    descriptor = TensorDescriptor(value, [3, 16], [16, 1], [1, 16], padding=padding)
    metadata = _metadata()
    metadata.tensordesc_meta = [
        {
            "block_size": [1, 16],
            "elem_size": 4,
            "elem_type": 7,  # Published backend's float32 tensor-map type.
            "swizzle": 0,
            "fp4_padded": False,
        }
    ]
    call = launcher.Launcher(
        SimpleNamespace(signature={"descriptor": "tensordesc<fp32[2]>"}), metadata
    )
    call(0, 1, 1, 0, 0, None, None, None, None, descriptor)
    invalid = dict(metadata.tensordesc_meta[0], block_size=[1, 1, 16])
    metadata.tensordesc_meta = [invalid]
    bad_call = launcher.Launcher(
        SimpleNamespace(signature={"descriptor": "tensordesc<fp32[2]>"}), metadata
    )
    with pytest.raises(ValueError, match="block rank mismatch"):
        bad_call(0, 1, 1, 0, 0, None, None, None, None, descriptor)
    call(0, 1, 1, 0, 0, None, None, None, None, descriptor)


@pytest.mark.min_devices(2)
def test_backend_uses_the_tensor_device_and_current_nondefault_stream():
    hosts = torch_fl.native_host._hosts
    artifacts = hosts.audit()["host_artifacts"]
    initial = torch_fl.flagos.current_device()
    target = (initial + 1) % torch_fl.flagos.device_count()
    stream = torch.Stream(device=f"flagos:{target}")
    retained = []
    hits = torch.ops.flagos.native_source_typed_hits("aten::cos")
    try:
        with stream, torch.no_grad():
            torch_fl.flagos.set_device(initial)
            for shape in ((17,), (3, 5, 7), (19, 31)):
                cpu = torch.randn(shape)
                value = cpu.to(f"flagos:{target}")
                expected = torch.cos(cpu)
                actual = torch.cos(value)
                retained.append((actual, expected))
                assert torch_fl.flagos.current_device() == initial
                assert (
                    torch.accelerator.current_stream(target).stream_id
                    == stream.stream_id
                )
        stream.synchronize()
        for actual, expected in retained:
            torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-6)
        assert len({value.data_ptr() for value, _ in retained}) == len(retained)
        assert torch.ops.flagos.native_source_typed_hits("aten::cos") == hits + 3
        assert hosts.audit()["host_artifacts"] == artifacts
    finally:
        torch_fl.flagos.set_device(initial)


def _cache(signature, args, kwargs):
    launcher = importlib.import_module(
        torch_fl.native_host._hosts.package + "._launcher"
    )
    # The original source-call path supplies its Python warmup default before
    # consulting this cache. Native callers may omit that default keyword.
    kwargs = {"warmup": False, **kwargs}
    cache = launcher.KernelCache(tuple(signature), {}, False)
    key = cache.run(args, kwargs, (0,), 0)
    assert isinstance(key, bytes)
    abi = launcher.Launcher(SimpleNamespace(signature=signature), _metadata())
    # A zero grid validates packing and the native key against the Python key.
    # No synthetic numerical kernel is compiled or launched.
    cache.prepare(key, args, kwargs, (None, abi, 0, None, {}, {}, {}, ()))

    def miss(*args, **kwargs):
        raise RuntimeError("unexpected native key miss")

    return launcher.register_script_kernel(
        SimpleNamespace(native_cache=cache, run=miss)
    )


@pytest.mark.parametrize("value", [0.0, -0.0, float("nan"), float("inf")])
@pytest.mark.parametrize("explicit_warmup", [False, True])
def test_zero_grid_native_key_matches_python_for_nested_and_keyword_values(
    value, explicit_warmup
):
    tensor = torch.empty_strided((3, 5), (1, 3), device="flagos")
    signature = {"tensor": "*fp32", "values": (("i8", "u16"), "i64")}
    signature.update({f"v{i}": "fp64" for i in range(10)})
    args = (tensor, ((-3, 65535), 2**53 + 3))
    kwargs = {f"v{i}": value if i == 0 else float(i) for i in reversed(range(10))}
    if explicit_warmup:
        kwargs["warmup"] = False
    slot = _cache(signature, args, kwargs)
    torch.ops.flagos_source_native.launch(slot, list(args), kwargs, [0])
    # Keyword insertion order is not part of the device-specialization key.
    torch.ops.flagos_source_native.launch(
        slot, list(args), dict(reversed(list(kwargs.items()))), [0]
    )
    changed = dict(kwargs, v9=99.0)
    with pytest.raises(RuntimeError, match="native key miss"):
        torch.ops.flagos_source_native.launch(slot, list(args), changed, [0])


def _descriptor_launch(slot: int, value: torch.Tensor, packed: bool, width: int):
    shape = [3, width]
    strides = [16, 1]
    block = [1, 1]
    if packed:
        fields: List[Any] = [
            "__flagos_descriptor_v2__",
            value,
            shape,
            strides,
            block,
            "zero",
        ]
        descriptor: Any = fields
    else:
        mapping: Dict[str, Any] = {
            "__flagos_tensor_descriptor_v1__": True,
            "base": value,
            "shape": shape,
            "strides": strides,
            "block_shape": block,
            "padding": "zero",
        }
        descriptor = mapping
    args: List[Any] = [value, descriptor]
    kwargs: Dict[str, Any] = {"warmup": False}
    torch.ops.flagos_source_native.launch(slot, args, kwargs, [0])


@pytest.mark.parametrize("packed", [False, True])
def test_zero_grid_descriptor_representations_match_the_original_key(packed):
    from triton.tools.tensor_descriptor import TensorDescriptor

    value = torch.empty(3, 16, device="flagos")
    descriptor = TensorDescriptor(value, [3, 16], [16, 1], [1, 1])
    slot = _cache(
        {"value": "*fp32", "descriptor": "tensordesc<fp32[2]>"},
        (value, descriptor),
        {"warmup": False},
    )
    launch = torch.jit.script(_descriptor_launch)
    launch(slot, value, packed, 16)
    with pytest.raises((RuntimeError, torch.jit.Error), match="native key miss"):
        launch(slot, value, packed, 8)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mm_uses_current_shapes_addresses_and_streams(dtype):
    hosts = torch_fl.native_host._hosts
    before = torch_fl.native_host.audit()["host_artifacts"]
    stream = torch.Stream(device="flagos:0")
    retained = []
    with torch.no_grad(), stream:
        for m, k, n in ((7, 128, 64), (37, 128, 256), (1, 1024, 3072), (7, 128, 64)):
            a = torch.randn(m, k, dtype=dtype).to("flagos")
            b = torch.randn(n, k, dtype=dtype).to("flagos").t()
            expected = hosts.original_ops["mm"](a, b)
            hits = torch.ops.flagos.native_source_typed_hits("aten::mm")
            actual = torch.mm(a, b)
            assert torch.ops.flagos.native_source_typed_hits("aten::mm") == hits + 1
            retained.append((actual, expected))
    stream.synchronize()
    for actual, expected in retained:
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    assert len({actual.data_ptr() for actual, _ in retained}) == len(retained)
    assert torch_fl.native_host.audit()["host_artifacts"] == before
