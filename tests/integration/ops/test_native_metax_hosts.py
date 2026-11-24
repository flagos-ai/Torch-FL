# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Installed MACA launch conformance with original published FlagGems kernels."""

import importlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch_fl
from tests.integration.common._native_host import (
    LAYOUT_VIEWS,
    check_add_layout,
    check_launch_on_other_device,
)


pytestmark = [pytest.mark.metax, pytest.mark.native_source_host, pytest.mark.flaggems]


@pytest.mark.parametrize("import_order", ["torch_first", "torch_fl_first"])
@pytest.mark.parametrize("profile", ["full", "minimal"])
def test_metax_import_order_preserves_the_installed_cpu_core(import_order, profile):
    imports = (
        "import torch, sys\nassert 'torch_fl' not in sys.modules\nimport torch_fl"
        if import_order == "torch_first"
        else "import torch_fl; import torch"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            imports + "\nfrom pathlib import Path\n"
            "import importlib.metadata, json, os\n"
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
            "status = torch_fl.optional_integration_status()\n"
            "if os.environ['FLAGOS_STARTUP_PROFILE'] == 'minimal':\n"
            "    assert all(value == 'inactive' for value in status.values()), status\n"
            "else:\n"
            "    assert status['ddp'] == 'active' or status['ddp'].startswith('failed:'), status\n"
            "assert torch.flagos is torch_fl.flagos\n"
            "torch.flagos.init()\n"
            "assert torch.empty(1, device='flagos').device.type == 'flagos'\n"
            "print(json.dumps({'torch': torch.__version__, 'devices': torch_fl.flagos.device_count()}))\n",
        ],
        env={
            **os.environ,
            "FLAGOS_STARTUP_PROFILE": profile,
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
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
@pytest.mark.parametrize("layout", LAYOUT_VIEWS)
def test_metax_source_hosts_preserve_layout_values_and_live_outputs(dtype, layout):
    import flag_gems

    hosts = torch_fl.native_host._hosts
    assert flag_gems.add is hosts.op("add")
    assert flag_gems.add is not hosts.original_ops["add"]
    # The ATen add route is gapped for MetaX's wrapped-scalar f64 -> bf16
    # compilation defect. Test the compiled original host with tensor operands.
    check_add_layout(flag_gems.add, dtype, layout)


@pytest.mark.min_devices(2)
def test_metax_native_launch_uses_tensor_device_and_current_stream():
    check_launch_on_other_device()
