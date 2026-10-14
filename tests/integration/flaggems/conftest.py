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

"""Package-level wiring for the hardware-agnostic FlagGems conformance suite.

Four jobs:

    * register this package's markers (``anyplatform``, ``flaggems_python``,
      ``flaggems_route``);
    * skip a test that needs the FlagGems kernel set this wheel did not compile
      -- the one build-driven skip the mandate allows, keyed off the build
      record and never a platform name;
    * print exactly one comparable summary line (and the JSON doc it names) from
      ``pytest_terminal_summary``, and only when this suite's own tests ran.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The package carries __init__.py, so pytest imports these modules as
# ``flaggems.<name>`` while the test modules do a bare ``from flaggems_caps
# import ...``. Put this directory on sys.path up front so both resolve.
_PACKAGE_DIR = str(Path(__file__).resolve().parent)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

_PACKAGE_PATH = Path(__file__).resolve().parent

#: Node IDs of this suite's tests that survived selection in this session.
#: ``pytest_configure`` registers the report plugin process-wide and a conftest
#: is loaded whenever its directory is *collected*, so a broad
#: ``pytest tests/integration`` run registers it too. Recording the items that
#: actually ran keeps the summary counting only this suite's outcomes, and keeps
#: it silent when none of them ran.
_FLAGGEMS_NODEIDS: set[str] = set()

#: The JSON doc's directory. Overridable so a CI job can route it into its
#: artifacts; defaults to the run's rootdir.
_REPORT_DIR_ENV = "FLAGGEMS_CONFORMANCE_REPORT_DIR"


def _is_flaggems_item(item: pytest.Item) -> bool:
    """Whether an item belongs to this package's directory."""
    path = getattr(item, "path", None)
    if path is None:  # pytest < 7 fell back to fspath
        path = Path(str(item.fspath))
    try:
        return _PACKAGE_PATH in path.resolve().parents
    except OSError:  # a path that will not resolve is not ours
        return False


def _build_kernels() -> frozenset[str]:
    """Kernel sets from the build record, or empty when it cannot be read."""
    try:
        from flaggems_caps import build_kernels

        return build_kernels()
    except Exception:  # noqa: BLE001 -- a missing record means "nothing compiled"
        return frozenset()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "anyplatform: runs on every supported hardware platform"
    )
    config.addinivalue_line(
        "markers",
        "flaggems_python: requires the FlagGems Python kernel set "
        "(built with FLAGOS_BUILD_FLAGGEMS=ON)",
    )
    config.addinivalue_line(
        "markers",
        "flaggems_route(op): asserts the conf-declared FlagGems route for a "
        "single operator",
    )
    config.pluginmanager.register(_ConformanceReport(), "flaggems-conformance-report")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Record this suite's selected tests, and skip the one build-driven gap.

    ``trylast`` matters: pytest's ``-m`` deselection is another
    ``pytest_collection_modifyitems`` implementation, and running after it means
    the recorded node IDs are the tests this session will actually report on.
    """
    _FLAGGEMS_NODEIDS.clear()
    kernels: frozenset[str] | None = None
    for item in items:
        if not _is_flaggems_item(item):
            continue
        _FLAGGEMS_NODEIDS.add(item.nodeid)
        if item.get_closest_marker("flaggems_python") is None:
            continue
        if kernels is None:
            kernels = _build_kernels()
        if "flaggems" not in kernels:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "the FlagGems kernel set is not compiled into this wheel "
                        "(build record)"
                    )
                )
            )


def _count_outcomes(stats: dict, nodeids: set[str]) -> tuple[int, int, int, int]:
    """(passed, failed, skipped, errors) among ``nodeids`` in a run's stats."""

    def count(outcome: str) -> int:
        return sum(
            1
            for report in stats.get(outcome, ())
            if getattr(report, "nodeid", "") in nodeids
        )

    return count("passed"), count("failed"), count("skipped"), count("error")


class _ConformanceReport:
    """Emit one comparable summary line plus the JSON doc it names."""

    def pytest_terminal_summary(
        self,
        terminalreporter: object,
        exitstatus: int,
        config: pytest.Config,
    ) -> None:
        stats = getattr(terminalreporter, "stats", {})
        passed, failed, skipped, errors = _count_outcomes(stats, _FLAGGEMS_NODEIDS)
        if not (passed or failed or skipped or errors):
            # None of this suite's tests ran in this session.
            return
        accel, kernels, total, routed, none = _conformance_facts()
        line = (
            f"FLAGGEMS-CONFORMANCE accel={accel} kernels={kernels} "
            f"ops_total={total} ops_routed={routed} ops_none={none} "
            f"passed={passed} failed={failed} skipped={skipped} errors={errors}"
        )
        write_line = getattr(terminalreporter, "write_line", None)
        if write_line is not None:
            write_line(line)
        _write_doc(
            config,
            {
                "accel": accel,
                "kernels": kernels,
                "ops_total": total,
                "ops_routed": routed,
                "ops_none": none,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "errors": errors,
                "exitstatus": int(exitstatus),
            },
        )


def _conformance_facts() -> tuple[str, str, int, int, int]:
    """(accel, kernels, ops_total, ops_routed, ops_none) for the summary line."""
    try:
        from flaggems_caps import (
            FLAGGEMS_ROUTING_CALLS,
            accel_label,
            build_kernels,
            routes_to_flaggems,
        )

        accel = accel_label() or "unknown"
        kernels_set = build_kernels()
    except Exception:  # noqa: BLE001 -- report the shape, not a crash
        return "unknown", "none", 0, 0, 0
    total = len(FLAGGEMS_ROUTING_CALLS)
    routed = sum(1 for op in FLAGGEMS_ROUTING_CALLS if routes_to_flaggems(op))
    kernels = ",".join(sorted(kernels_set)) if kernels_set else "none"
    return accel, kernels, total, routed, total - routed


def _write_doc(config: pytest.Config, payload: dict) -> None:
    """Write the JSON doc, named from the accelerator the conf names."""
    try:
        import os

        directory = os.environ.get(_REPORT_DIR_ENV)
        root = Path(directory) if directory else Path(str(config.rootpath))
        name = f"flaggems-conformance-{payload['accel']}.json"
        (root / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except Exception:  # noqa: BLE001 -- a report is never worth failing a run
        pass
