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
    env["PYTHONPATH"] = _selected_package_environment()
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _selected_package_environment():
    """Keep inherited package entries while removing pytest's source override.

    ``conftest.py`` puts the checkout first so these files can import from it,
    but a fresh interpreter cannot load an extension built only in the wheel
    from that checkout. Subprocesses inherit the installed parent supplied by
    the test runner; that entry must stay ahead of the checkout for the same
    reason as the parent process.
    """
    entries = [
        entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry
    ]
    checkout = str(Path(__file__).resolve().parents[2])
    checkout_path = Path(checkout)
    return os.pathsep.join(
        entry for entry in entries if Path(entry).resolve() != checkout_path
    )


def _require_torch():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch and a built torch_fl are needed for subprocess tests")


def _build_accelerator():
    _require_torch()
    spec = importlib.util.find_spec("torch_fl")
    assert spec is not None and spec.origin, "torch_fl package is not installed"
    detector = runpy.run_path(str(Path(spec.origin).resolve().parent / "_platform.py"))
    return detector["build_accelerator"]()


def _require_torch_first_supported():
    if _build_accelerator() != "cuda":
        pytest.skip("Vendor libtorch overlays require torch_fl before torch")


def _require_autoload():
    if _build_accelerator() == "mps":
        pytest.skip("MPS deliberately leaves bare import torch on the MPS backend")


def test_minimal_import_and_explicit_repeated_activation():
    _require_torch()
    _require_autoload()
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
    _require_autoload()
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


def _require_mps():
    if _build_accelerator() != "mps":
        pytest.skip("This check needs an MPS build")


def test_mps_minimal_add_runs_clean():
    """The smallest native MPS op there is: one add, no compiled matmul.

    A fresh process has to reach the device, build a queue and encode one graph
    without dying. That is the ground the whole backend stands on, and it is
    the one part of this file that is a contract rather than a probe: it runs
    on every MPS lane so a bring-up regression fails here, next to the device
    and queue code, instead of inside an operator suite.
    """
    _require_mps()
    env = os.environ.copy()
    env["PYTHONPATH"] = _selected_package_environment()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, torch_fl\n"
            "torch_fl.flagos.init()\n"
            "dev = torch.device('flagos', 0)\n"
            "a = torch.randn(4, device=dev)\n"
            "b = torch.randn(4, device=dev)\n"
            "assert (a + b).shape == (4,)\n"
            "torch_fl._C._synchronize()\n",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mps_allocator_shutdown_survives_persistent_test_tensors():
    """Leaving test tensors alive must not turn a clean suite into exit 134."""
    _require_mps()
    env = os.environ.copy()
    env["PYTHONPATH"] = _selected_package_environment()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, torch_fl\n"
            "a = torch.arange(12, dtype=torch.float32, device='mps')\n"
            "b = torch.empty(3, 4, device='flagos')\n"
            "b.copy_(a.reshape(3, 4))\n"
            "assert torch.equal(b.cpu(), a.reshape(3, 4).cpu())\n"
            "raise SystemExit(0)\n",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
