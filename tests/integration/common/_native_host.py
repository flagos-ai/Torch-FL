# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared checks for the installed native host runtime."""

import json
import os
from pathlib import Path
import subprocess
import sys

import torch

import torch_fl


LAYOUT_VIEWS = {
    "dense": lambda x: x,
    "transpose": lambda x: x.transpose(0, 2),
    "slice": lambda x: x[:, :, ::2],
    "expand": lambda x: x[:, :1, :].expand_as(x),
    "scalar": lambda x: x[0, 0, 0],
    "empty": lambda x: x[:, :0, :],
}


def check_add_layout(add, dtype, layout):
    cpu = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(3, 5, 7).to(dtype) / 16
    cpu = LAYOUT_VIEWS[layout](cpu)
    value = cpu.to("flagos")
    artifacts = torch_fl.native_host.audit()["host_artifacts"]
    with torch.no_grad():
        first = add(value, value)
        second = add(first, value)
    torch_fl.flagos.synchronize()
    expected = torch.add(cpu, cpu)
    torch.testing.assert_close(first.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(second.cpu(), torch.add(expected, cpu), atol=0, rtol=0)
    if cpu.numel():
        assert first.data_ptr() != second.data_ptr()
    assert torch_fl.native_host.audit()["host_artifacts"] == artifacts


def check_launch_on_other_device():
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


def run_native_host_case(case, pytestconfig):
    package = Path(torch_fl.__file__).resolve().parent
    if runner_temp := os.environ.get("RUNNER_TEMP"):
        record = Path(runner_temp) / "torch-fl-native-host-evidence/wheel.json"
        if record.is_file() or os.environ.get("GITHUB_ACTIONS") == "true":
            assert record.is_file(), "the lane did not verify its native host wheel"
            evidence = json.loads(record.read_text())
            # FlagCICD consumes the plan environment when it configures
            # pytest. The integration runner records it alongside the reports.
            report = pytestconfig.getoption("json_report_file", default=None)
            groups = Path(report).with_name("groups.json") if report else None
            if groups and groups.is_file():
                plan = json.loads(groups.read_text())["context"]
            else:
                plan = json.loads(os.environ.get("CI_PLAN_CONTEXT") or "{}")
            assert evidence["plan"] == plan, (
                "the native host wheel belongs to a different plan"
            )
            package = Path(evidence["package"])
        elif torch_fl.native_host._model_runtime() is None:
            package = Path(runner_temp) / "torch-fl-native-host-install/torch_fl"

    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, runpy, sys\n"
            "from contextlib import nullcontext\n"
            "from pathlib import Path\n"
            "package = Path(sys.argv[1]).resolve()\n"
            "sys.path.insert(0, str(package.parent))\n"
            "measurement = nullcontext()\n"
            "if os.environ.get('FLAGCICD_NATIVE_HOST_COVERAGE_DIR'):\n"
            "    from flagcicd_coverage import native_host_coverage\n"
            "    measurement = native_host_coverage(package)\n"
            "with measurement:\n"
            "    import torch_fl\n"
            "    assert Path(torch_fl.__file__).resolve().parent == package\n"
            "    assert torch_fl.native_host._model_runtime() is not None\n"
            "    runpy.run_path(sys.argv[2])[sys.argv[3]]()\n",
            str(package),
            case.__code__.co_filename,
            case.__name__,
        ],
        env={
            **os.environ,
            "FLAGOS_LOG": "fallback",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "[flagos cpu_fallback]" not in process.stderr, process.stderr
