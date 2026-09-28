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

"""Real import-order and activation checks for built accelerator environments."""

import importlib.util
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


def _run(code, profile):
    env = os.environ.copy()
    env["FLAGOS_STARTUP_PROFILE"] = profile
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _require_torch():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch and a built torch_fl are needed for subprocess tests")


def test_minimal_import_and_explicit_repeated_activation():
    _require_torch()
    _run(
        """
import torch_fl
import torch

assert all(value == 'inactive' for value in
           torch_fl.optional_integration_status().values())
torch.empty(1, device='flagos')
torch_fl.activate_optional_integrations('ddp')
first = torch.nn.parallel.DistributedDataParallel.__init__
torch_fl.activate_optional_integrations('ddp')
assert torch.nn.parallel.DistributedDataParallel.__init__ is first
assert torch_fl.optional_integration_status()['ddp'] == 'active'
""",
        "minimal",
    )


def test_full_profile_activates_framework_hooks_during_import():
    _require_torch()
    _run(
        """
import torch_fl
import torch

status = torch_fl.optional_integration_status()
assert status['ddp'] == 'active', status
assert status['parallel_comm'] == 'active', status
torch.empty(1, device='flagos')
""",
        "full",
    )


def test_torch_first_import_on_cuda():
    _require_torch()
    detector = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "torch_fl" / "_platform.py")
    )
    if detector["build_accelerator"]() != "cuda":
        pytest.skip("Vendor libtorch overlays require torch_fl before torch")
    _run(
        """
import torch
import torch_fl

assert torch.device('flagos').type == 'flagos'
assert torch_fl.optional_integration_status()['ddp'] == 'inactive'
""",
        "minimal",
    )
