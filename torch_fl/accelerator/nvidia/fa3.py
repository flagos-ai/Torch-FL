# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Publish the bundled FA3 Python API on a CPU PyTorch installation."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch

from ._cuda_compat import cuda_runtime_version


def install() -> None:
    name = "flash_attn_3_nv"
    if torch._C._has_cuda or name in sys.modules:
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

    # Some CUDA consumers use the version string as a runtime capability probe.
    # ATen's compiled CUDA flag continues to describe the installed Torch wheel.
    if torch.version.cuda is None:
        torch.version.cuda = cuda_runtime_version()
