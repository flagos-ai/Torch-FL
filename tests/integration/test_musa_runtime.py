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

"""Require the installed MUSA runtime before hybrid and compile coverage."""

import importlib.metadata
from pathlib import Path

import pytest


pytestmark = pytest.mark.musa


def test_musa_flagtree_and_flaggems_runtime():
    # Import torch_fl first: it installs the MUSA compatibility surface used by
    # FlagGems. Missing dependencies must fail this required CI preflight.
    import torch_fl
    from torch_fl import _env
    from torch_fl._build_config import ACCELERATOR
    from torch_fl.compile.flagtree_shim import (
        bind_flagtree_musa_driver,
        flagtree_backend,
        flagtree_musa_driver_target,
        is_flagtree_active,
    )

    import flag_gems
    import triton.backends

    assert ACCELERATOR == "musa", ACCELERATOR
    assert "flaggems" in _env.build_kernels(), _env.build_kernels()
    assert is_flagtree_active(), triton.__file__
    assert flagtree_backend() == "mthreads", flagtree_backend()
    assert "mthreads" in triton.backends.backends, sorted(triton.backends.backends)

    distribution = importlib.metadata.distribution("flagtree")
    triton_root = Path(triton.__file__).resolve().parent
    assert Path(distribution.locate_file("triton")).resolve() == triton_root, (
        triton_root
    )
    assert flag_gems.vendor_name == "mthreads", flag_gems.vendor_name
    assert torch_fl.flagos.is_available(), "MUSA device is unavailable"
    assert bind_flagtree_musa_driver(), "FlagTree cannot bind to torch_fl"
    target = flagtree_musa_driver_target()
    assert target is not None and target[0] == "musa", target

    print(f"FlagTree: {distribution.version}")
    print(f"Triton: {triton.__version__} ({triton_root})")
    print(f"FlagGems: {flag_gems.__version__} (vendor: {flag_gems.vendor_name})")
    print(f"MUSA compile target: {target}")
