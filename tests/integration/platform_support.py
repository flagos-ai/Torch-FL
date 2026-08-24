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

import os
from pathlib import Path


def detect_platform() -> str:
    """Return the active hardware platform using the integration conventions."""
    accelerator = os.environ.get("ACCELERATOR", "").lower()
    if accelerator in {"ascend", "metax", "maca", "musa", "gcu"}:
        return "metax" if accelerator == "maca" else accelerator
    if os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME"):
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
        pass

    config = os.environ.get("FLAGOS_BACKEND_CONFIG", "").lower()
    for platform in ("ascend", "metax", "musa", "gcu", "cuda"):
        if platform in config:
            return platform
    return "cuda"
