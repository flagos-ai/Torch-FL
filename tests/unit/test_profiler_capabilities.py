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

"""Unit coverage for the profiler capability table's `memset` row.

`tests/integration/profiler_support.py::capabilities_for_platform` decides
whether the cross-backend profiler suite asserts on `gpu_memset`. The row is
a platform set intersected with the build record carrying boxing kernels, so
it is pure data: it can be checked without an accelerator and without the
integration conftest.

The DCU entry is covered here because it is otherwise only observable as a
green PCIe-free integration lane. The `hygon-boxing` job failed
`test_profiler_parity.py::test_category_coverage` with "missing categories
torch-cuda emits: ['gpu_memset']" precisely because the row answered False on
DCU, which also meant `append_boxing_path_probe()` never appended the
`linalg_lu_factor` call that makes rocBLAS zero its workspace. Dropping the
platform back out of the set would re-open the same failure silently, and the
memset cases would report as skipped rather than failing.

Run: pytest tests/unit/test_profiler_capabilities.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INTEGRATION_DIR = _REPO_ROOT / "tests" / "integration"
_PROFILER_SUPPORT_PATH = _INTEGRATION_DIR / "profiler_support.py"


def _load_profiler_support():
    """Import profiler_support.py without the integration pytest plugin.

    The module imports `platform_support` by bare name, which the integration
    conftest puts on `sys.path`; do the same here instead of loading the whole
    plugin machinery.
    """
    integration_dir = str(_INTEGRATION_DIR)
    if integration_dir not in sys.path:
        sys.path.insert(0, integration_dir)
    spec = importlib.util.spec_from_file_location(
        "profiler_support_under_test", _PROFILER_SUPPORT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profiler_support = _load_profiler_support()

# Platforms the row may claim, and platforms it must never claim. Kept as two
# lists rather than one parametrization so a new platform has to be placed
# deliberately instead of inheriting whichever list is closer.
_MEMSET_PLATFORMS = ["cuda", "metax", "dcu"]
_NEVER_MEMSET_PLATFORMS = [
    "ascend",
    "gcu",
    "iluvatar",
    "kunlunxin",
    "mps",
    "musa",
    "ppu",
    "tsingmicro",
]


def _set_boxing_built(monkeypatch, built):
    monkeypatch.setattr(profiler_support, "_boxing_built", lambda: built)


@pytest.mark.parametrize("platform", _MEMSET_PLATFORMS)
def test_memset_row_follows_the_boxing_build_record(monkeypatch, platform):
    """A boxing wheel claims memset; the same platform without boxing does not.

    The memset the row asserts is the vendor blas zeroing its gemm workspace,
    and that library is reachable only through the generated boxing kernels.
    """
    _set_boxing_built(monkeypatch, True)
    assert profiler_support.capabilities_for_platform(platform).memset is True

    _set_boxing_built(monkeypatch, False)
    assert profiler_support.capabilities_for_platform(platform).memset is False


def test_dcu_boxing_wheel_claims_memset(monkeypatch):
    """The DCU row added for the hygon-boxing lane, stated on its own.

    Tested separately from the parametrization above so a failure names the
    platform this row was actually added for.
    """
    _set_boxing_built(monkeypatch, True)
    capabilities = profiler_support.capabilities_for_platform("dcu")
    assert capabilities.platform == "dcu"
    assert capabilities.memset is True


def test_dcu_vendor_wheel_leaves_memset_off(monkeypatch):
    """`backends_hygon.conf` routes linalg_lu_factor_ex `none`, so no fill.

    The default DCU wheel reaches cpu_fallback for the probe op and makes no
    rocBLAS call, so claiming the capability there would turn a skip into a
    failure on a record the wheel cannot produce.
    """
    _set_boxing_built(monkeypatch, False)
    assert profiler_support.capabilities_for_platform("dcu").memset is False


@pytest.mark.parametrize("platform", _NEVER_MEMSET_PLATFORMS)
def test_memset_row_stays_off_where_the_tracer_emits_no_fill(monkeypatch, platform):
    """Boxing kernels alone are not the signal; the vendor blas call is."""
    _set_boxing_built(monkeypatch, True)
    assert profiler_support.capabilities_for_platform(platform).memset is False


def test_boxing_path_probe_is_a_noop_without_the_capability(monkeypatch):
    """The probe is gated on the same row, so it costs nothing when off.

    `append_boxing_path_probe` imports torch only after the gate, which is what
    lets this check run on a machine with no torch installed.
    """
    _set_boxing_built(monkeypatch, False)
    assert profiler_support.append_boxing_path_probe(platform="dcu") is None
    assert profiler_support.append_boxing_path_probe(platform="cuda") is None
