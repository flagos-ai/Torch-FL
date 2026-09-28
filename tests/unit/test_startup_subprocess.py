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


def _run(code, profile, *, autoload=None):
    env = os.environ.copy()
    env["FLAGOS_STARTUP_PROFILE"] = profile
    if autoload is not None:
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = autoload
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


def _require_torch_first_supported():
    _require_torch()
    spec = importlib.util.find_spec("torch_fl")
    assert spec is not None and spec.origin, "torch_fl package is not installed"
    detector = runpy.run_path(str(Path(spec.origin).resolve().parent / "_platform.py"))
    if detector["build_accelerator"]() != "cuda":
        pytest.skip("Vendor libtorch overlays require torch_fl before torch")


def test_minimal_import_and_explicit_repeated_activation():
    _require_torch()
    _run(
        """
import torch_fl
import torch
from importlib.metadata import entry_points

assert all(value == 'inactive' for value in
           torch_fl.optional_integration_status().values())
assert torch._C._get_privateuse1_backend_name() == 'flagos'
hooks = [ep for ep in entry_points(group='torch.backends')
         if ep.name == 'torch_fl']
assert len(hooks) == 1, hooks
hook = hooks[0].load()
hook()
hook()
assert torch._C._get_privateuse1_backend_name() == 'flagos'
torch.empty(1, device='flagos')
first_status = torch_fl.activate_optional_integrations('ddp', strict=False)['ddp']
first = torch.nn.parallel.DistributedDataParallel.__init__
second_status = torch_fl.activate_optional_integrations('ddp', strict=False)['ddp']
assert torch.nn.parallel.DistributedDataParallel.__init__ is first
assert second_status == first_status
assert first_status == 'active' or first_status.startswith('failed:'), first_status
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
for name in ('ddp', 'parallel_comm'):
    assert status[name] == 'active' or status[name].startswith('failed:'), status
torch.empty(1, device='flagos')
""",
        "full",
    )


def test_torch_first_import_on_cuda():
    _require_torch_first_supported()
    _run(
        """
import torch
import sys
assert 'torch_fl' not in sys.modules
import torch_fl

assert torch.device('flagos').type == 'flagos'
assert torch_fl.optional_integration_status()['ddp'] == 'inactive'
""",
        "minimal",
        autoload="0",
    )


def test_bare_torch_autoloads_in_fresh_compile_worker_on_cuda():
    _require_torch_first_supported()
    if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD", "1") != "1":
        pytest.skip("This CI image disables all torch.backends entry points")
    _run(
        """
import sys
import torch
import triton

assert 'torch_fl' in sys.modules
assert torch._C._get_privateuse1_backend_name() == 'flagos'
assert torch.device('flagos').type == 'flagos'
assert hasattr(torch, 'flagos')
assert torch.flagos.is_available()
from torch_fl._autoload import init
init()
init()
assert torch._C._get_privateuse1_backend_name() == 'flagos'
""",
        "full",
        autoload="1",
    )


def test_foreign_privateuse1_plugin_has_actionable_diagnostic_on_cuda():
    _require_torch_first_supported()
    _run(
        """
import sys
import torch

torch.utils.rename_privateuse1_backend('foreign')
try:
    import torch_fl
except RuntimeError as error:
    message = str(error)
    assert "PrivateUse1 is already claimed by the 'foreign' backend" in message
    assert 'TORCH_DEVICE_BACKEND_AUTOLOAD=0' in message
    assert 'torch_fl._C' not in sys.modules
else:
    raise AssertionError('torch_fl claimed a foreign PrivateUse1 backend')
""",
        "minimal",
        autoload="0",
    )
