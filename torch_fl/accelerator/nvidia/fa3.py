# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Publish the bundled FA3 Python API on a CPU PyTorch installation."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch


def install() -> None:
    name = "flash_attn_3_nv"
    if torch.version.cuda is not None or name in sys.modules:
        return
    if not hasattr(torch.ops.torch_fl_fa3, "fwd"):
        return

    package = Path(__file__).resolve().parents[2] / "lib" / name
    spec = spec_from_file_location(name, package / "__init__.py")
    if spec is None or spec.loader is None:
        raise ImportError("The Torch-FL FA3 build is missing its Python package")
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
