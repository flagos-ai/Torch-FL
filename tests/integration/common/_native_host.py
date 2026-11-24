# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run shared native host cases against the wheel validated by the lane."""

import json
import os
from pathlib import Path
import subprocess
import sys

import torch_fl


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
            "import runpy, sys\n"
            "from pathlib import Path\n"
            "package = Path(sys.argv[1]).resolve()\n"
            "sys.path.insert(0, str(package.parent))\n"
            "import torch_fl\n"
            "assert Path(torch_fl.__file__).resolve().parent == package\n"
            "assert torch_fl.native_host._model_runtime() is not None\n"
            "runpy.run_path(sys.argv[2])[sys.argv[3]]()\n",
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
