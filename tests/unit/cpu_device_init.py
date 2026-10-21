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

"""Load shared hardware test tables with CPU torch for unit tests."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import torch

INTEGRATION = Path(__file__).resolve().parents[1] / "integration"


@contextmanager
def load_on_cpu(relative_path: str):
    """Load one table without importing torch_fl or using accelerator imports."""
    if relative_path.startswith("vendor_lanes/"):
        helpers = (
            ("shared_sdpa_caps", INTEGRATION / "shared_sdpa_caps.py"),
            ("vendor_caps", INTEGRATION / "vendor_lanes/vendor_caps.py"),
        )
    else:
        helpers = (("shared_sdpa_caps", INTEGRATION / "shared_sdpa_caps.py"),)
    names = ("device_init", *(name for name, _ in helpers))
    previous = {name: sys.modules.get(name) for name in names}
    saved_paths = list(sys.path)
    try:
        device_init = ModuleType("device_init")
        device_init.torch = torch
        device_init.torch_fl = _cpu_torch_fl()
        sys.modules["device_init"] = device_init
        for name, path in helpers:
            if path.is_file():
                sys.modules[name] = _load_module(name, path)
        target = INTEGRATION / relative_path
        sys.path.insert(0, str(target.parent))
        yield _load_module("cpu_test_table", target)
    finally:
        sys.path[:] = saved_paths
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _cpu_torch_fl():
    """Expose only the backend config lookup needed to import the tables."""
    torch_fl = ModuleType("torch_fl")
    torch_fl.backend_config_path = lambda: "backends_test.conf"
    return torch_fl


def _load_module(name: str, path: Path):
    """Execute one module from a path with its own module namespace."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    sys.modules[name] = module
    sys.path.append(str(path.parent))
    spec.loader.exec_module(module)
    return module
