# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Import native-host modules without starting the torch_fl runtime."""

import importlib
import importlib.util
from pathlib import Path
import sys


def import_native_host(name):
    package = "_native_host_unit_test"
    if package not in sys.modules:
        root = Path(__file__).resolve().parents[2] / "torch_fl/native_host"
        spec = importlib.util.spec_from_file_location(package, root / "__init__.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{package}.{name}")
