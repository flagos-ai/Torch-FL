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
that a new backend is described in a single place. The detector itself lives in
``torch_fl/_platform.py`` -- the package and both test trees read that one
module -- and is re-exported here under the names the contracts import.

Nothing here may import torch at module scope: the integration conftest loads
the support modules as pytest plugins before torch_fl preloads its device
assets, and importing torch first breaks the required library initialization
order. ``torch_fl/_platform.py`` is stdlib-only and is loaded by path for the
same reason -- importing the package would execute ``torch_fl/__init__.py``.
"""

import importlib.machinery
import importlib.util
from pathlib import Path


def _platform_path() -> Path:
    """Locate torch_fl/_platform.py without importing the package.

    From wherever torch_fl resolves: the installed wheel in CI's wheel-only test
    workspace (which has no source tree), or the repo copy in a source checkout.
    A unit-test stub torch_fl carries no _platform.py, so fall back to this
    file's repo copy.
    """
    spec = importlib.util.find_spec("torch_fl")
    if spec is not None and spec.origin:
        candidate = Path(spec.origin).resolve().parent / "_platform.py"
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parents[2] / "torch_fl" / "_platform.py"


def _load_platform():
    """Load torch_fl/_platform.py by path, without importing the package."""
    loader = importlib.machinery.SourceFileLoader(
        "torch_fl._platform_under_test", str(_platform_path())
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_platform = _load_platform()

#: Re-exported under the names the integration contracts import.
detect_platform = _platform.detect_platform
build_accelerator = _platform.build_accelerator
