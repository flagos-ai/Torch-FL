# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prove representative TileOPs routes ran on a vendor through the dispatcher."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import torch_fl  # noqa: F401 -- claims the flagos PrivateUse1 device


DEVICE = "flagos:0"
ROUTES = (
    ("relu", lambda x, y: torch.relu(x), 1e-3, 1e-3),
    ("add.Tensor", lambda x, y: torch.add(x, y), 1e-3, 1e-3),
    (
        "sum.dim_IntList",
        lambda x, y: torch.sum(x, dim=-1, keepdim=True),
        5e-3,
        5e-3,
    ),
    (
        "_softmax",
        lambda x, y: torch.softmax(x, dim=-1),
        5e-3,
        5e-3,
    ),
)

DISPATCH_CALLS = {
    "relu": "torch.relu(x)",
    "add.Tensor": "torch.add(x,y)",
    "sum.dim_IntList": "torch.sum(x,dim=-1,keepdim=True)",
    "_softmax": "torch.softmax(x,dim=-1)",
}


def _cpu_reference(call, x, y):
    if call is torch.relu:
        return call(x.cpu())
    return call(x.cpu(), y.cpu())


def _run_logged(code: str, output: Path) -> str:
    env = dict(os.environ, FLAGOS_LOG="dispatch")
    done = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        cwd=Path(os.environ.get("RUNNER_TEMP", "/tmp")),
    )
    log = done.stdout + done.stderr
    output.write_text(log)
    assert done.returncode == 0, f"{log}\nreturncode={done.returncode}"
    return log


def _log_path(name: str, tmp_path: Path) -> Path:
    """Put collection logs where the workflow uploads them."""
    output_dir = Path(os.environ.get("TILEOPS_LOG_DIR", tmp_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{name}.dispatch.log"


@pytest.mark.parametrize("operation,call,rtol,atol", ROUTES)
def test_route_runs_on_tileops_and_matches_the_cpu_reference(
    operation, call, rtol, atol, tmp_path
):
    """Each route executes twice, crosses back to flagos, and is numerically checked."""
    x = torch.randn(64, 128, dtype=torch.float16, device=DEVICE)
    y = torch.randn(64, 128, dtype=torch.float16, device=DEVICE)
    expected = _cpu_reference(call, x, y)
    first, second = call(x, y), call(x, y)
    output = _run_logged(
        "import torch, torch_fl; "
        "x=torch.randn(64,128,dtype=torch.float16,device='flagos:0'); "
        "y=torch.randn_like(x); "
        f"{DISPATCH_CALLS[operation]}",
        _log_path(operation, tmp_path),
    )
    assert f"[flagos dispatch] {operation} -> tileops" in output, output
    assert first.device.type == "flagos", first.device
    assert second.device.type == "flagos", second.device
    torch.testing.assert_close(first.cpu(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(second.cpu(), expected, rtol=rtol, atol=atol)


def test_unrouted_op_reaches_cpu_fallback(tmp_path):
    """A route absent from the TileOPs config must not disappear into boxing."""
    x_cpu = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    output = torch.ops.aten.transpose.int(x_cpu.to(DEVICE), 0, 1)
    log = _run_logged(
        "import torch, torch_fl; "
        "x=torch.arange(12,dtype=torch.float16).reshape(3,4); "
        "y=torch.ops.aten.transpose.int(x.to('flagos:0'),0,1); "
        "assert y.device.type == 'flagos'",
        _log_path("transpose", tmp_path),
    )
    assert output.device.type == "flagos", output.device
    torch.testing.assert_close(output.cpu(), x_cpu.T)
    assert "[flagos dispatch] transpose.int -> none" in log, log
    assert "boxing" not in log, log
