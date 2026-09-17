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

"""Hardware platform detection shared by the cross-backend test contracts.

Contract modules (profiler, AMP) select capabilities from one platform name so
that a new backend is described in a single place. Nothing here may import
torch at module scope: the integration conftest loads the support modules as
pytest plugins before torch_fl preloads its device assets, and importing torch
first breaks the required library initialization order.
"""

import importlib.machinery
import importlib.util
import os
from pathlib import Path


def _build_accelerator() -> str:
    """The accelerator the installed wheel was built for ("" if unknown).

    From the build record setup.py writes (torch_fl/_build_config.py), which is
    the source torch_fl itself reads -- the environment no longer overrides it,
    so a stale FLAGOS_ACCELERATOR exported for some other build must not steer
    these contracts toward a platform the wheel was not built for either.

    find_spec() locates the package without executing it. This module is loaded
    as a pytest plugin before torch_fl preloads its device assets, so it must
    not import torch_fl here; the record itself imports nothing.
    """
    try:
        spec = importlib.util.find_spec("torch_fl")
    except (ImportError, ValueError):
        return ""
    if spec is None or not spec.origin:
        return ""
    path = Path(spec.origin).resolve().parent / "_build_config.py"
    try:
        loader = importlib.machinery.SourceFileLoader("_flagos_build_record", str(path))
        module = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(loader.name, loader)
        )
        loader.exec_module(module)
    except (OSError, ImportError, SyntaxError):
        # A source checkout with no build yet; the marker below is what
        # identifies such an install, exactly as it does for a native kernel
        # vendor whose conf is not named after it.
        return ""
    return str(getattr(module, "ACCELERATOR", "")).strip().lower()


def detect_platform() -> str:
    """Return the active hardware platform using the integration conventions."""
    accelerator = _build_accelerator()
    if accelerator in {"ascend", "dcu", "metax", "maca", "musa", "gcu", "ppu"}:
        return "metax" if accelerator == "maca" else accelerator
    if os.environ.get("PPU_SDK"):
        return "ppu"
    if Path("/usr/local/PPU_SDK").is_dir():
        return "ppu"

    try:
        import torch_fl

        marker = Path(torch_fl.__file__).resolve().parent / "lib" / "flagos_platform"
        platform = marker.read_text().strip().lower()
        if platform:
            return platform
    except (ImportError, OSError):
        config = os.environ.get("FLAGOS_BACKEND_CONFIG", "").lower()
    else:
        # Asked of torch_fl rather than read from the environment, which torch_fl
        # no longer writes: the variable holds only what a user set, and the
        # wheel's own choice lives in backend_config_path().
        config = torch_fl.backend_config_path().lower()

    for platform in ("ascend", "metax", "musa", "gcu", "cuda"):
        if platform in config:
            return platform
    return "cuda"
