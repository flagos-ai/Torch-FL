# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Installed MACA launch conformance with original published FlagGems kernels."""

import importlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch_fl


pytestmark = [pytest.mark.metax, pytest.mark.native_source_host, pytest.mark.flaggems]


@pytest.mark.parametrize("import_order", ["torch_first", "torch_fl_first"])
def test_metax_import_order_preserves_the_installed_cpu_core(import_order):
    imports = (
        "import torch; import torch_fl"
        if import_order == "torch_first"
        else "import torch_fl; import torch"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            imports + "\nfrom pathlib import Path\n"
            "import importlib.metadata, json\n"
            "distribution = importlib.metadata.distribution('torch')\n"
            "mapped = {Path(line.split()[-1]).resolve() "
            "for line in Path('/proc/self/maps').read_text().splitlines() "
            "if '/' in line}\n"
            "for name in ('libc10.so', 'libtorch_cpu.so', 'libtorch_python.so'):\n"
            "    expected = Path(distribution.locate_file('torch/lib/' + name)).resolve()\n"
            "    actual = {path for path in mapped if path.name == name}\n"
            "    assert actual == {expected}, (name, actual, expected)\n"
            "assert torch_fl._build_config.NATIVE_HOSTS\n"
            "assert 'boxing' not in torch_fl._build_config.KERNELS\n"
            "print(json.dumps({'torch': torch.__version__, 'devices': torch_fl.flagos.device_count()}))\n",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])["torch"] == "2.10.0+cpu"


def _launcher():
    return importlib.import_module(torch_fl.native_host._hosts.package + "._launcher")


def _metadata(**changes):
    return SimpleNamespace(
        **{
            "num_warps": 1,
            "shared": 0,
            "global_scratch_size": 0,
            "global_scratch_align": 1,
            "profile_scratch_size": 0,
            "profile_scratch_align": 1,
            **changes,
        }
    )


def test_installed_metax_contract_and_routes_exclude_boxing():
    import flag_gems

    from torch_fl.native_host.backend import COMPILED_FIELDS

    assert "boxing" not in torch_fl._build_config.KERNELS
    assert torch.cuda.device_count() == torch_fl.flagos.device_count()
    assert flag_gems.runtime.device.device_count == torch_fl.flagos.device_count()
    record = torch_fl.native_host.audit()["backend"]
    assert record["vendor"] == "metax" and record["compiler_backend"] == "maca"
    assert record["capabilities"] == 141
    assert _launcher().backend_contract() == {
        name: record[name] for name in COMPILED_FIELDS
    }
    assert "backends/metax/compiler.py" in record["driver_sources"]


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
        ("fp16", 1.5),
        ("bf16", -1.5),
        ("fp32", 1.25),
        ("fp64", -1.25),
    ],
)
def test_metax_scalar_arguments_pack_without_submitting_a_zero_grid(kind, value):
    call = _launcher().Launcher(SimpleNamespace(signature={"value": kind}), _metadata())
    call(0, 1, 1, 0, 0, None, None, None, None, value)


@pytest.mark.parametrize(
    "changes",
    [
        {"num_ctas": 2},
        {"num_warps": 0},
        {"cluster_dims": (2, 1, 1)},
        {"launch_pdl": True},
        {"launch_cooperative_grid": True},
    ],
)
def test_metax_rejects_unsupported_launch_capabilities(changes):
    with pytest.raises(ValueError, match="unsupported MetaX launch"):
        _launcher().Launcher(SimpleNamespace(signature={}), _metadata(**changes))


def test_metax_hardware_descriptor_is_rejected_without_cuda_emulation():
    with pytest.raises(ValueError, match="hardware tensor descriptors are unsupported"):
        _launcher().Launcher(
            SimpleNamespace(signature={"value": "tensordesc<fp32[2]>"}),
            _metadata(tensordesc_meta=[{}]),
        )


def test_metax_software_descriptor_and_workspace_failure_cleanup():
    from triton.tools.tensor_descriptor import TensorDescriptor

    value = torch.empty(3, 16, device="flagos")
    descriptor = TensorDescriptor(value, [3, 16], [16, 1], [1, 16])
    call = _launcher().Launcher(
        SimpleNamespace(signature={"value": "tensordesc<fp32[2]>"}), _metadata()
    )
    call(0, 1, 1, 0, 0, None, None, None, None, descriptor)
    bad = _launcher().Launcher(
        SimpleNamespace(signature={}),
        _metadata(
            global_scratch_size=64,
            global_scratch_align=256,
            profile_scratch_size=64,
            profile_scratch_align=512,
        ),
    )
    allocated = torch_fl.flagos.memory_allocated()
    for _ in range(3):
        with pytest.raises(ValueError, match="scratch alignment"):
            bad(1, 1, 1, 0, 0, None, None, None, None)
        assert torch_fl.flagos.memory_allocated() == allocated


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "layout", ["dense", "transpose", "slice", "expand", "scalar", "empty"]
)
def test_metax_source_hosts_preserve_layout_values_and_live_outputs(dtype, layout):
    import flag_gems

    hosts = torch_fl.native_host._hosts
    assert flag_gems.add is hosts.op("add")
    assert flag_gems.add is not hosts.original_ops["add"]
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
    # The ATen add route is gapped for MetaX's wrapped-scalar f64 -> bf16
    # compilation defect. Test the compiled original host with tensor operands.
    with torch.no_grad():
        first = flag_gems.add(value, value)
        second = flag_gems.add(first, value)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(first.cpu(), cpu + cpu, atol=0, rtol=0)
    torch.testing.assert_close(second.cpu(), (cpu + cpu) + cpu, atol=0, rtol=0)
    if cpu.numel():
        assert first.data_ptr() != second.data_ptr()
    assert torch_fl.native_host.audit()["host_artifacts"] == artifacts


@pytest.mark.min_devices(2)
def test_metax_native_launch_uses_tensor_device_and_current_stream():
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
