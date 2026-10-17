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

"""The profiler capability table has to be answerable for all eight platforms.

`tests/integration/profiler_support.py` is consulted on hardware only -- the
contract in `tests/integration/test_profiler_contract.py` runs inside the
profiler step of each platform's job -- so a table entry that cannot be
evaluated is not noticed until a runner is already busy. That is how this file
came to exist: a row was added that read a variable the return statement builds
rather than one it is given, and the table raised `NameError` for every platform
while every off-hardware check stayed green. Ruff's F821 was what turned it red.

So this file imports the table the way the profiler step does, by module path,
and asks it for every platform name in `tests/integration/platform_gates.py`,
off hardware and without a device. What each row *means* is held by the contract
that measures it; what is held here is that the answer exists and is the same
kind of answer on all eight.

No torch: the table is about tracer behavior as configured, so it must not need
an imported backend to say what a platform's tracer emits.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_INTEGRATION = _REPO / "tests" / "integration"


def _load(name, path):
    """Load one of the integration helpers without the integration conftest."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# `profiler_support` imports its sibling `platform_support` by module name, so
# that one is loaded under its own name first. Loading both by path keeps this
# file from putting tests/integration on sys.path, where the two suites share
# basenames (`conftest.py`, `test_apex_compat.py`).
_load("platform_support", _INTEGRATION / "platform_support.py")
profiler_support = _load(
    "profiler_support_under_test", _INTEGRATION / "profiler_support.py"
)

# The gate table carries the platform list this has to agree with, and the
# composer loads it by path for the same reason.
_GATES_PATH = _INTEGRATION / "platform_gates.py"
ALL_PLATFORMS = _load("platform_gates", _GATES_PATH).ALL_PLATFORMS

_BOOL_ROWS = (
    "device",
    "kernel",
    "demangled",
    "runtime",
    "cbid",
    "memcpy",
    "memset",
    "flow",
    "linkage",
    "metadata",
)


@pytest.mark.parametrize("platform", ALL_PLATFORMS)
def test_every_platform_has_a_row(platform):
    caps = profiler_support.capabilities_for_platform(platform)
    assert caps.platform == platform
    for row in _BOOL_ROWS:
        assert isinstance(getattr(caps, row), bool), row


@pytest.mark.parametrize("platform", ALL_PLATFORMS)
def test_device_rows_follow_the_device_row(platform):
    """A row about device records is off exactly where no device record is emitted.

    Kunlunxin is the one platform whose build collects nothing (the vendor shim
    instruments a dispatch key the de-boxed backend does not use), so the rows
    that read device records have nothing to read there and say so rather than
    claiming an activity that never arrives. The two rows that are off for a
    narrower measured reason are stated separately below.
    """
    caps = profiler_support.capabilities_for_platform(platform)
    for row in ("kernel", "runtime", "flow", "metadata"):
        assert getattr(caps, row) == caps.device, row
    if not caps.device:
        assert not caps.memcpy and not caps.memset
        assert not caps.linkage and not caps.demangled


@pytest.mark.parametrize(
    "platform,demangled,linkage",
    [
        # GCU delivers TOPSPTI activity, so its device rows are open, but it
        # names kernels with the raw topsaten symbols and its records carry no
        # link back to the launching operator (`self_device_time_total == 0.0`
        # for both aten::matmul and aten::mm). Measured in the ci/unify-coverage
        # run; the narrower reason is written at the row in profiler_support.py.
        ("gcu", False, False),
        # Every other platform with device records reads both.
        ("ascend", True, True),
        ("cuda", True, True),
        ("dcu", True, True),
        ("metax", True, True),
        ("musa", True, True),
        ("ppu", True, True),
    ],
)
def test_narrow_rows_are_recorded(platform, demangled, linkage):
    caps = profiler_support.capabilities_for_platform(platform)
    assert caps.device
    assert caps.demangled == demangled
    assert caps.linkage == linkage
