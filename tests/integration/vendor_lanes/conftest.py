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

"""Package-level wiring for the per-platform vendor-lane conformance suite.

This package is where the cases that used to be chip-specific files live. Every
domain in it is one manifest step that every platform runs, one data table keyed
by the accelerator the conf names, and one shared test that reads the row for
the platform it is running on. A platform the domain does not reach skips with
the conf's own route (or its own measured absence) as the reason, and the skip
is recorded in ``tests/integration/platform_gates.py`` and documented in
``docs/reference/platform-capability-matrix.md``.

Two jobs here, both about keeping the suite comparable across platforms:

    * put this directory on ``sys.path``, so the test modules resolve their data
      modules by their bare names the way ``tests/integration/flaggems`` and
      ``tests/integration/runtime`` do;
    * print one summary line per domain from ``pytest_terminal_summary``, naming
      the accelerator and the rows this platform carries, so a reviewer reading
      a hardware job's log sees which lanes that platform declared rather than
      only how many cases passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PACKAGE_DIR = str(Path(__file__).resolve().parent)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

_PACKAGE_PATH = Path(__file__).resolve().parent

#: Node IDs of this package's tests that survived selection in this session.
#: ``pytest_configure`` registers the report plugin process-wide and a conftest
#: is loaded whenever its directory is *collected*, so a broad
#: ``pytest tests/integration`` run registers it too. Recording the items that
#: actually ran keeps the summary counting only this suite's outcomes, and keeps
#: it silent when none of them ran.
_VENDOR_NODEIDS: set[str] = set()


def _is_vendor_item(item: pytest.Item) -> bool:
    """Whether an item belongs to this package's directory."""
    path = getattr(item, "path", None)
    if path is None:  # pytest < 7 fell back to fspath
        path = Path(str(item.fspath))
    try:
        return _PACKAGE_PATH in path.resolve().parents
    except OSError:  # a path that will not resolve is not ours
        return False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "anyplatform: runs on every supported hardware platform"
    )
    config.pluginmanager.register(_VendorReport(), "vendor-lane-conformance-report")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Record this suite's selected tests, after pytest's ``-m`` deselection."""
    _VENDOR_NODEIDS.clear()
    for item in items:
        if _is_vendor_item(item):
            _VENDOR_NODEIDS.add(item.nodeid)


def _count_outcomes(stats: dict, nodeids: set[str]) -> tuple[int, int, int, int]:
    """(passed, failed, skipped, errors) among ``nodeids`` in a run's stats."""

    def count(outcome: str) -> int:
        return sum(
            1
            for report in stats.get(outcome, ())
            if getattr(report, "nodeid", "") in nodeids
        )

    return count("passed"), count("failed"), count("skipped"), count("error")


class _VendorReport:
    """Emit one comparable summary line naming this platform's rows."""

    def pytest_terminal_summary(
        self,
        terminalreporter: object,
        exitstatus: int,
        config: pytest.Config,
    ) -> None:
        stats = getattr(terminalreporter, "stats", {})
        passed, failed, skipped, errors = _count_outcomes(stats, _VENDOR_NODEIDS)
        if not (passed or failed or skipped or errors):
            # None of this suite's tests ran in this session.
            return
        try:
            from vendor_caps import accel_label, declared_rows

            accel = accel_label() or "unknown"
            rows = declared_rows()
        except Exception:  # noqa: BLE001 -- report the shape, not a crash
            accel, rows = "unknown", []
        line = (
            f"VENDOR-LANES accel={accel} "
            f"passed={passed} failed={failed} skipped={skipped} errors={errors}"
        )
        write_line = getattr(terminalreporter, "write_line", None)
        if write_line is not None:
            write_line(line)
        for entry in rows:
            if write_line is not None:
                write_line(f"  {entry}")
