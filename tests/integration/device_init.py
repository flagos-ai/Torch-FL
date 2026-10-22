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

"""Select the device, then load torch and torch_fl.

CUDA-compatible vendors preload their libtorch when ``torch_fl`` is imported.
Importing torch first bypasses that selection, so test modules should use
``from device_init import torch, torch_fl``. Tests that do not use the
``torch_fl`` API may import only ``torch``; importing it still loads
``torch_fl`` first.

Test modules outside the test directory resolve this file through
``device_init_path``. Child Python processes receive the same path through
``PYTHONPATH``; its parent test session loads the op tables into that path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch_fl  # noqa: F401 -- select the device before loading torch
import torch  # noqa: F401 -- re-export the selected torch build

__all__ = ("torch", "torch_fl")


def device_init_path() -> str:
    """The test root that makes this module importable by a child process."""
    return str(Path(__file__).resolve().parent)


def child_source(source: str) -> str:
    """Make a ``python -c`` program use this initializer before torch loads."""
    return f"from device_init import torch, torch_fl\n{source}"


def prepare_child_imports() -> list[str]:
    """Expose test helpers and return the shared child import path prefix."""
    test_root = Path(__file__).resolve().parent
    candidates = (test_root, test_root / "ops", test_root / "runtime")
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    modules = (
        ("backend_conf", test_root / "ops" / "backend_conf.py"),
        ("platform_support", test_root / "platform_support.py"),
    )
    for name, path in modules:
        if name in sys.modules or not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    missing = [
        name for name, path in modules if path.is_file() and name not in sys.modules
    ]
    if missing:
        raise ImportError(f"child test helpers are unavailable: {missing!r}")
    return [str(path) for path in candidates]
