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


def _build_accelerator():
    return _installed_platform_module()["build_accelerator"]()


def _require_torch_first_supported():
    if _build_accelerator() != "cuda":
        pytest.skip("Vendor libtorch overlays require torch_fl before torch")


def _entry_point_hooks():
    """Whether the wheel registers a ``torch.backends`` entry point for torch_fl.

    A fresh interpreter in this process's environment, because a package
    installed after this one started is invisible to the parent's
    ``importlib.metadata`` cache. ``resolve()`` rather than ``load()``: the
    test is about whether the hook is *registered*, not whether calling it (and
    so importing the backend) works.
    """
    probe = (
        "import sys\n"
        "from importlib.metadata import entry_points\n"
        "sys.exit(0 if [ep.resolve() for ep in entry_points(group='torch.backends')"
        " if ep.name == 'torch_fl'] else 1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    return result.returncode == 0


#: Accelerators whose wheel deliberately omits the autoload entry point, and
#: why, from setup.py's ``kwargs["entry_points"] = {}`` branch. The record has
#: to live here because it cannot be read from the install: the contract group
#: runs on the wheel's *installed* files, which hold no setup.py, and a source
#: path relative to this file points into a temporary copy of the tree.
#: tests/unit/test_autoload_omission.py keeps this list equal to setup.py's.
_NO_AUTOLOAD = {
    "mps": "setup.py omits it for MPS so a bare import torch stays on the MPS backend",
}


def _installed_platform_module():
    """The installed package's ``_platform.py``, loaded without importing it.

    ``import torch_fl`` runs backend selection, which this test must observe
    rather than pre-empt. A path relative to this file is not an option: the
    contract group runs on the wheel's installed files, so ``parents[2]`` is a
    temporary copy of the tree, not the package that is installed.
    """
    _require_torch()
    spec = importlib.util.find_spec("torch_fl")
    assert spec is not None and spec.origin, "torch_fl package is not installed"
    return runpy.run_path(str(Path(spec.origin).resolve().parent / "_platform.py"))


def _resolved(name):
    """``name`` in the platform table's vocabulary, or "" when it names nothing.

    Every accelerator reader here answers in its own spelling -- the build
    record carries the vendor setup.py chose (``apple``), ``FLAGOS_ACCELERATOR``
    carries what the build was told, and either may be a table id already --
    while ``_NO_AUTOLOAD`` is keyed by table id.
    """
    if not name:
        return ""
    detector = _installed_platform_module()
    return detector["platform_for"](name)


def _accelerator_recorded_by_the_install():
    """The accelerator this install was built for, or "".

    The wheel's own record first. A tree that was never built for one has no
    record, and what setup.py was told (``FLAGOS_ACCELERATOR``) is what its
    next build will be.
    """
    _require_torch()
    return _resolved(_build_accelerator()) or _resolved(
        os.environ.get("FLAGOS_ACCELERATOR", "")
    )


def _require_declared_entry_point():
    """Skip a wheel that declares no autoload hook, and say whether it should.

    pyproject.toml keeps the entry-points table dynamic so the MPS wheel can
    omit it. The omission is a property of that wheel, not of the platform --
    but a wheel built for any other accelerator that lacks the hook never
    exercises the autoload path at all, which is what makes this a check and
    not a skip.
    """
    if _entry_point_hooks():
        return
    accelerator = _accelerator_recorded_by_the_install()
    if accelerator in _NO_AUTOLOAD:
        pytest.skip(
            f"the {accelerator} wheel has no autoload hook: {_NO_AUTOLOAD[accelerator]}"
        )
    if not accelerator:
        # Nothing says what this install is: a source checkout that was never
        # built for an accelerator. The test is about that wheel, not this tree.
        pytest.skip("this install records no accelerator")
    pytest.fail(
        f"the {accelerator} wheel declares no torch.backends entry point, and "
        f"only {sorted(_NO_AUTOLOAD)} omit it by design"
    )


def test_minimal_import_and_explicit_repeated_activation():
    _require_torch()
    _require_declared_entry_point()
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
    _require_declared_entry_point()
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
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
