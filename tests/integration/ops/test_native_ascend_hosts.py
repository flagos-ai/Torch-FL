# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Installed Ascend hosts using original published FlagGems kernels."""

import importlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch
import torch_fl
from tests.integration.common._native_host import (
    LAYOUT_VIEWS,
    check_add_layout,
    check_launch_on_other_device,
)


pytestmark = [pytest.mark.ascend, pytest.mark.native_source_host, pytest.mark.flaggems]


def _launcher():
    return importlib.import_module(torch_fl.native_host._hosts.package + "._launcher")


def test_installed_ascend_contract_without_boxing_or_torch_npu():
    import sys
    from torch_fl.native_host.backend import COMPILED_FIELDS

    assert "boxing" not in torch_fl._build_config.KERNELS
    assert torch.version.cuda is None
    if "torch_npu" in sys.modules:
        assert getattr(sys.modules["torch_npu"], "__torch_fl_shim__", False)
    record = torch_fl.native_host.audit()["backend"]
    assert record["vendor"] == "ascend" and record["compiler_backend"] == "npu"
    assert record["capabilities"] == 132
    assert _launcher().backend_contract() == {
        name: record[name] for name in COMPILED_FIELDS
    }
    assert "backends/ascend/compiler.py" in record["driver_sources"]
    assert "backends/ascend/npu_utils.cpp" in record["driver_sources"]


@pytest.mark.parametrize(
    "kind,value",
    [
        ("i8", -128),
        ("u8", 255),
        ("i16", -32768),
        ("u16", 65535),
        ("i32", -(2**31)),
        ("u32", 2**32 - 1),
        ("i64", -(2**63)),
        ("u64", 2**64 - 1),
        ("i1", True),
        ("fp16", 1.0001),
        ("bf16", -1.0001),
        ("fp32", 1.25),
        ("fp64", -1.25),
    ],
)
def test_ascend_scalar_wire_abi_accepts_zero_grid_without_submission(kind, value):
    from torch_fl.native_host.backends.ascend import prepare
    from triton.backends.ascend import utils
    from triton.runtime import driver

    source, metadata = prepare(
        SimpleNamespace(fn=object(), signature={"value": kind}),
        SimpleNamespace(
            force_simt_only=False,
            parallel_mode="vector",
            compile_on_910_95=False,
            mix_mode="aiv",
            enable_auto_blockify=False,
        ),
        utils,
        driver.active.utils,
    )
    call = _launcher().Launcher(source, metadata)
    call(0, 1, 1, 0, 0, None, None, None, None, value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", LAYOUT_VIEWS)
def test_ascend_flaggems_hosts_keep_layout_values_and_distinct_live_outputs(
    dtype, layout
):
    check_add_layout(torch.add, dtype, layout)


@pytest.mark.min_devices(2)
def test_ascend_native_launch_uses_tensor_device_and_current_acl_stream():
    check_launch_on_other_device()


def test_ascend_first_allocation_binds_each_worker_thread():
    ready = Barrier(2, timeout=30)

    def allocate_and_launch(offset):
        # Neither the allocator nor the launcher may assume another thread's
        # ACL context binds this thread, even when both request device zero.
        ready.wait()
        cpu = torch.arange(129, dtype=torch.float32) / 128 + offset
        value = cpu.to("flagos:0")
        stream = torch.Stream(device="flagos:0")
        with stream, torch.no_grad():
            result = torch.cos(value)
        stream.synchronize()
        assert torch_fl.flagos.current_device() == 0
        torch.testing.assert_close(result.cpu(), torch.cos(cpu), atol=1e-6, rtol=1e-6)
        return result

    with ThreadPoolExecutor(max_workers=2) as workers:
        first, second = list(workers.map(allocate_and_launch, (0.0, 0.25)))
    assert first.data_ptr() != second.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["dense", "transpose", "slice", "expand"])
@pytest.mark.parametrize("dims", [[2], [1, 2], [0, 2], [2, 0]])
def test_ascend_mean_metadata_lowering_matches_original_host(dtype, layout, dims):
    value = (
        torch.arange(3 * 5 * 7, dtype=torch.float32)
        .reshape(3, 5, 7)
        .to(dtype)
        .to("flagos:0")
    )
    if layout == "transpose":
        value = value.transpose(0, 2)
    elif layout == "slice":
        value = value[1:, :, :]
    elif layout == "expand":
        value = value[:, :1, :].expand(3, 5, 7)
    hosts = torch_fl.native_host._hosts
    original = hosts.resolve(hosts.original_ops["mean_dim"])
    with torch.no_grad():
        expected = original(value, dims, True)
        hits = torch.ops.flagos.native_source_typed_hits("aten::mean.dim")
        first = torch.mean(value, dim=dims, keepdim=True)
        second = torch.mean(value, dim=dims, keepdim=True)
    torch_fl.flagos.synchronize()
    assert torch.ops.flagos.native_source_typed_hits("aten::mean.dim") == hits + 2
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first.cpu(), expected.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(second.cpu(), expected.cpu(), rtol=0, atol=0)


def _published_concat_hits():
    return torch_fl.native_host.audit()["source_variant_hits"]["published_concat"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "case",
    [
        "rope_q",
        "rope_k",
        "rope_decode",
        "kv_empty",
        "kv_short",
        "kv127",
        "kv511",
        "kv601",
        "transpose",
        "slice",
        "outer_slice",
        "outer_slice_transpose",
        "many_inputs",
        "expand",
        "empty",
    ],
)
def test_ascend_published_concat_keeps_values_and_retained_outputs(dtype, case):
    if case.startswith("rope"):
        heads, length = {"rope_q": (16, 60), "rope_k": (8, 17), "rope_decode": (8, 1)}[
            case
        ]
        cpu = (
            torch.arange(length * heads * 128, dtype=torch.float32).reshape(
                1, length, heads, 128
            )
            / 128
        )
        value = cpu.to(dtype).to("flagos:0").transpose(1, 2)
        inputs, dim = [-value[..., 64:], value[..., :64]], -1
    elif case.startswith("kv"):
        length = {
            "kv_empty": 0,
            "kv_short": 60,
            "kv127": 127,
            "kv511": 511,
            "kv601": 601,
        }[case]
        first = (
            torch.arange(8 * length * 128, dtype=torch.float32).reshape(
                1, 8, length, 128
            )
            / 128
        )
        last = torch.arange(8 * 128, dtype=torch.float32).reshape(1, 1, 8, 128) / 128
        inputs = [
            first.to(dtype).to("flagos:0"),
            last.to(dtype).to("flagos:0").transpose(1, 2),
        ]
        dim = 2
    else:
        value = (
            torch.arange(30, dtype=torch.float32).reshape(5, 6).to(dtype).to("flagos:0")
        )
        if case == "transpose":
            inputs, dim = [value.t(), value.t()], -1
        elif case == "slice":
            inputs, dim = [value[:, ::2], value[:, 1::2]], 1
        elif case == "outer_slice":
            inputs, dim = [value[::2], value[1::2]], 0
        elif case == "outer_slice_transpose":
            inputs, dim = [value[::2].t(), value[1::2].t()], 1
        elif case == "many_inputs":
            inputs, dim = [value[:1], value[1:3], value[3:]], 0
        elif case == "expand":
            inputs, dim = [value[:1].expand(3, 6), value[:1].expand(2, 6)], 0
        else:
            inputs, dim = [value[:, :0], value[:, :0]], 1
    expected = torch.cat([value.cpu() for value in inputs], dim)
    before = _published_concat_hits()
    with torch.no_grad():
        first, second = torch.cat(inputs, dim), torch.cat(inputs, dim)
    torch_fl.flagos.synchronize()
    # The unchanged published alternative returns zeros for stepped inputs
    # without a unit-stride axis; those retain the original Ascend host.
    assert _published_concat_hits() == before + (0 if case == "slice" else 2)
    torch.testing.assert_close(first.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(second.cpu(), expected, atol=0, rtol=0)
    if expected.numel():
        assert first.data_ptr() != second.data_ptr()


@pytest.mark.parametrize("case", ["single", "mixed_dtype", "mixed_rank"])
def test_ascend_published_concat_constraints_retain_the_base_host(case):
    first = torch.arange(6, dtype=torch.float32).reshape(2, 3).to("flagos:0")
    inputs = [first]
    if case == "mixed_dtype":
        inputs.append(first.to(torch.bfloat16))
    elif case == "mixed_rank":
        inputs.insert(0, torch.empty(0, device="flagos:0"))
    before = _published_concat_hits()
    with torch.no_grad():
        actual = torch.cat(inputs, dim=1)
    expected = torch.cat([value.cpu() for value in inputs], dim=1)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    assert _published_concat_hits() == before


@pytest.mark.min_devices(2)
def test_ascend_published_concat_uses_its_device_and_current_stream():
    initial = torch_fl.flagos.current_device()
    target = (initial + 1) % torch_fl.flagos.device_count()
    stream = torch.Stream(device=f"flagos:{target}")
    cpu = torch.arange(8 * 17 * 128, dtype=torch.float32).reshape(1, 8, 17, 128) / 128
    before = _published_concat_hits()
    try:
        with stream, torch.no_grad():
            value = cpu.to(torch.bfloat16).to(f"flagos:{target}")
            torch_fl.flagos.set_device(initial)
            actual = torch.cat([value[..., 64:], value[..., :64]], dim=-1)
            assert torch_fl.flagos.current_device() == initial
        stream.synchronize()
        expected = torch.cat([cpu[..., 64:], cpu[..., :64]], dim=-1).bfloat16()
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
        assert _published_concat_hits() == before + 1
    finally:
        torch_fl.flagos.set_device(initial)
