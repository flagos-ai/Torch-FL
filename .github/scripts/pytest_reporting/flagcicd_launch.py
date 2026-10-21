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

"""Read a FlagCICD pytest launch once, before any plugin changes it."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

GROUP = "group"
SAMPLE = "sample"


@dataclass(frozen=True)
class Launch:
    """The runtime activation inherited by this Python process."""

    report_dir: str
    coverage_dir: str
    launcher: str
    group: str


def capture() -> Launch:
    return Launch(
        report_dir=os.environ.get("FLAGCICD_REPORT_DIR", ""),
        coverage_dir=os.environ.get("FLAGCICD_COVERAGE_DIR", ""),
        launcher=os.environ.get("FLAGCICD_LAUNCHER", ""),
        group=os.environ.get("FLAGCICD_TEST_GROUP", ""),
    )


_LAUNCH = capture()


def current() -> Launch:
    """The launch captured while this process imported its plugins."""
    return _LAUNCH


def strip_child_activation() -> None:
    """Keep nested processes from joining the parent's CI sessions."""
    for name in (
        "FLAGCICD_REPORT_DIR",
        "FLAGCICD_COVERAGE_DIR",
        "FLAGCICD_TEST_GROUP",
    ):
        os.environ.pop(name, None)


def set_reporting_plugin_path(workspace: str) -> None:
    """Put only reporting plugins ahead of the installed packages."""
    import sysconfig

    purelib = sysconfig.get_path("purelib")
    if purelib is None:
        raise RuntimeError("Python has no purelib directory")
    entries = [entry for entry in sys.path if entry and entry != purelib]
    sys.path[:] = [workspace, *entries]


def set_sample_environment() -> None:
    """Mark this process and its descendants as deliberate samples."""
    os.environ["FLAGCICD_LAUNCHER"] = SAMPLE
    strip_child_activation()
