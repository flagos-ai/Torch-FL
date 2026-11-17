# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Installed Ascend hosts using original published FlagGems kernels."""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch_fl


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
@pytest.mark.parametrize(
    "layout", ["dense", "transpose", "slice", "expand", "scalar", "empty"]
)
def test_ascend_flaggems_hosts_keep_layout_values_and_distinct_live_outputs(
    dtype, layout
):
    cpu = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(3, 5, 7).to(dtype) / 16
    if layout == "transpose":
        cpu = cpu.transpose(0, 2)
    elif layout == "slice":
        cpu = cpu[:, :, ::2]
    elif layout == "expand":
        cpu = cpu[:, :1, :].expand(3, 5, 7)
    elif layout == "scalar":
        cpu = cpu[0, 0, 0]
    elif layout == "empty":
        cpu = cpu[:, :0, :]
    value = cpu.to("flagos")
    artifacts = torch_fl.native_host.audit()["host_artifacts"]
    with torch.no_grad():
        first = torch.add(value, value)
        second = torch.add(first, value)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(first.cpu(), torch.add(cpu, cpu), atol=0, rtol=0)
    torch.testing.assert_close(second.cpu(), torch.add(cpu + cpu, cpu), atol=0, rtol=0)
    if cpu.numel():
        assert first.data_ptr() != second.data_ptr()
    assert torch_fl.native_host.audit()["host_artifacts"] == artifacts


@pytest.mark.min_devices(2)
def test_ascend_native_launch_uses_tensor_device_and_current_acl_stream():
    initial = torch_fl.flagos.current_device()
    target = (initial + 1) % torch_fl.flagos.device_count()
    stream = torch.Stream(device=f"flagos:{target}")
    hits = torch.ops.flagos.native_source_typed_hits("aten::cos")
    outputs = []
    try:
        with stream, torch.no_grad():
            torch_fl.flagos.set_device(initial)
            for shape in ((129,), (3, 5, 7), (17, 31)):
                cpu = torch.randn(shape)
                actual = torch.cos(cpu.to(f"flagos:{target}"))
                outputs.append((actual, torch.cos(cpu)))
                assert torch_fl.flagos.current_device() == initial
                assert (
                    torch.accelerator.current_stream(target).stream_id
                    == stream.stream_id
                )
        stream.synchronize()
        for actual, expected in outputs:
            torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-6)
        assert len({value.data_ptr() for value, _ in outputs}) == 3
        assert torch.ops.flagos.native_source_typed_hits("aten::cos") == hits + 3
    finally:
        torch_fl.flagos.set_device(initial)
