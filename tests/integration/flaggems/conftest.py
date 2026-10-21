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

"""Select one complete pass of the shared FlagGems conformance suite.

``run.py`` asks this directory for either the ``native`` or ``flaggems``
pass. Both passes contain the same ordered cases, including cases marked with
``flaggems_python``: that marker describes the kernel path a case exercises,
not permission to omit it from one pass. This conftest records the complete
inventory for each selection and reports it by name for the CI job's own
inventory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_PACKAGE_DIR = str(Path(__file__).resolve().parent)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

_PACKAGE_PATH = Path(__file__).resolve().parent
_SUITES = frozenset({"native", "flaggems"})
_FLAGGEMS_NODEIDS: dict[str, str] = {}
_SUITE: str | None = None
_EXPECTED: list[str] | None = None
_DUPLICATES = False
_FLAGCICD_NAME = {
    "tests/integration/flaggems/test_flaggems_device_identity.py::"
    "test_device_properties_accepts_the_registered_name[flagos]": (
        "tests/integration/flaggems/test_flaggems_device_identity.py::"
        "test_device_properties_accepts_the_registered_name[device]"
    ),
    "tests/integration/flaggems/test_flaggems_device_identity.py::"
    "test_device_properties_accepts_the_registered_name[flagos:0]": (
        "tests/integration/flaggems/test_flaggems_device_identity.py::"
        "test_device_properties_accepts_the_registered_name[device-index]"
    ),
}


def _is_flaggems_item(item: pytest.Item) -> bool:
    path = getattr(item, "path", None)
    if path is None:
        path = Path(str(item.fspath))
    try:
        return _PACKAGE_PATH in path.resolve().parents
    except OSError:
        return False


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("FlagGems conformance")
    group.addoption(
        "--flaggems-conformance-suite",
        action="store",
        choices=sorted(_SUITES),
        default=None,
        help="select the native or FlagGems kernel inventory",
    )


def _suite(config: pytest.Config) -> str | None:
    suite = config.getoption("--flaggems-conformance-suite")
    if suite is None:
        try:
            suite = config.getini("flaggems_conformance_suite") or None
        except ValueError:
            suite = None
    if suite not in _SUITES:
        if not _selects_this_suite(config):
            return None
        raise pytest.UsageError(
            "the FlagGems conformance suite needs --flaggems-conformance-suite="
            "native or --flaggems-conformance-suite=flaggems; run "
            "tests/integration/flaggems/run.py"
        )
    return suite


def _selects_this_suite(config: pytest.Config) -> bool:
    """Tell whether this pytest session targets the shared FlagGems suite.

    Pytest loads conftest modules from directories it discovers, even when a
    marker selects tests outside them. Those outer sessions do not own this
    suite and must not fail merely because its runner option is absent. A
    runner constructed with ``pytest.main`` keeps its directory argument in
    ``invocation_params`` while its inherited process arguments remain
    unchanged, so inspect the directory pytest was configured to collect.
    """
    directory = Path(config.invocation_params.dir).resolve()
    root = config.rootpath.resolve()
    for value in config.args:
        path = Path(value)
        if not path.is_absolute():
            path = directory / path
        path = path.resolve()
        if not path.is_relative_to(root):
            continue
        relative = path.relative_to(root)
        if relative.parts[-2:] == ("integration", "flaggems"):
            return True
    return False


def pytest_configure(config: pytest.Config) -> None:
    global _SUITE
    _SUITE = _suite(config)
    config.addinivalue_line(
        "markers", "anyplatform: runs on every supported hardware platform"
    )
    config.addinivalue_line(
        "markers",
        "flaggems_python: exercises the FlagGems Python kernel set",
    )
    config.addinivalue_line(
        "markers",
        "flaggems_route(op): asserts the conf-declared route for an operator",
    )
    if _SUITE is not None:
        config.pluginmanager.register(
            _ConformanceReport(), "flaggems-conformance-report"
        )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Require the complete ordered shared suite and record it for reporting."""
    global _EXPECTED, _DUPLICATES
    _FLAGGEMS_NODEIDS.clear()
    _EXPECTED = None
    _DUPLICATES = False
    if _SUITE is None:
        return
    selected = [item for item in items if _is_flaggems_item(item)]
    # A runner invocation is the only owner of this selection. Returning fewer
    # items than it collected would allow a marker, plugin or CLI filter to
    # silently shrink the shared inventory.
    if len(selected) != len(items):
        unexpected = [item.nodeid for item in items if item not in selected]
        raise pytest.UsageError(
            "the FlagGems conformance runner must collect only its shared suite; "
            f"unexpected items: {unexpected[:5]}"
        )
    # Keep the selection untouched: filtering by marker would give the passes
    # different executions while still reporting full pytest counts. A CI
    # inventory is loaded without importing a hardware backend.
    nodeids = [item.nodeid for item in selected]
    duplicates = len(nodeids) != len(set(nodeids))
    _DUPLICATES = duplicates
    for nodeid in nodeids:
        _FLAGGEMS_NODEIDS[_FLAGCICD_NAME.get(nodeid, nodeid)] = nodeid
    plan = os.environ.get("FLAGCICD_EXPECTED_CASES")
    if plan:
        expected = json.loads(plan).get("flaggems_conformance") or []
        if not expected:
            raise pytest.UsageError("the CI FlagGems plan is empty")
        # Some explicit inventory names cover two equivalent pytest IDs.
        # Check both counts and the source case set; a test that pytest does
        # not collect leaves an explicit name without a matching item.
        missing = [name for name in expected if name not in _FLAGGEMS_NODEIDS]
        if (
            missing
            or duplicates
            or len(nodeids) != len(_FLAGGEMS_NODEIDS)
            or len(nodeids) != len(expected)
        ):
            raise pytest.UsageError(
                "the FlagGems conformance selection differs from the CI "
                f"inventory: missing={missing[:5]}, duplicate_nodeids={duplicates}"
            )
        _EXPECTED = expected


def _count_outcomes(stats: dict, nodeids: set[str]) -> tuple[int, int, int, int]:
    """(passed, failed, skipped, errors) among ``nodeids`` in a run's stats."""

    def count(outcome: str) -> int:
        return sum(
            1
            for report in stats.get(outcome, ())
            if getattr(report, "nodeid", "") in nodeids
        )

    return count("passed"), count("failed"), count("skipped"), count("error")


def _require_complete_selection(terminalreporter: object) -> None:
    """Reject a session that configured but executed no shared cases."""
    config = terminalreporter.config
    if (
        getattr(config, "_flaggems_selection_checked", False)
        or _SUITE is None
        or set(config.args) != set(_FLAGGEMS_NODEIDS.values())
    ):
        return
    setattr(config, "_flaggems_selection_checked", True)
    stats = getattr(terminalreporter, "stats", {})
    expected = set(_FLAGGEMS_NODEIDS.values())
    passed, failed, skipped, errors = _count_outcomes(stats, expected)
    if not expected:
        raise pytest.UsageError("the FlagGems conformance selection is empty")
    if not (passed or failed or skipped or errors):
        raise pytest.UsageError(
            "the FlagGems conformance selection completed without executing any case"
        )


class _ConformanceReport:
    """Emit facts that name this selection for the merged CI report."""

    @pytest.hookimpl(tryfirst=True)
    def pytest_unconfigure(self, config: pytest.Config) -> None:
        if getattr(config, "_flaggems_selection_checked", False):
            return
        # Configuration can fail before the terminal summary is available.
        config.exitstatus = pytest.ExitCode.TESTS_FAILED

    @pytest.hookimpl(trylast=True)
    def pytest_terminal_summary(
        self,
        terminalreporter: object,
        exitstatus: int,
        config: pytest.Config,
    ) -> None:
        _require_complete_selection(terminalreporter)
        stats = getattr(terminalreporter, "stats", {})
        accel, kernels, total, routed, none = _conformance_facts()
        suite = _SUITE
        node_ids = set(_FLAGGEMS_NODEIDS.values())
        passed, failed, skipped, errors = _count_outcomes(stats, node_ids)
        line = (
            f"FLAGGEMS-CONFORMANCE accel={accel} suite={suite} kernels={kernels} "
            f"ops_total={total} ops_routed={routed} ops_none={none} "
            f"selected={len(node_ids)} passed={passed} failed={failed} "
            f"skipped={skipped} errors={errors}"
        )
        if _EXPECTED is not None:
            line += f" expected={len(_EXPECTED)}"
        if _DUPLICATES:
            line += " duplicate_nodeids=true"
        write_line = getattr(terminalreporter, "write_line", None)
        if write_line is not None:
            write_line(line)
        if not _FLAGGEMS_NODEIDS:
            if write_line is not None:
                write_line(f"FLAGGEMS-CONFORMANCE-FAILURE empty suite={suite}")
            return
        if (
            exitstatus == pytest.ExitCode.OK
            and passed == len(node_ids)
            and not (failed or skipped or errors)
            and (_EXPECTED is None or len(_EXPECTED) == len(node_ids))
        ):
            if write_line is not None:
                write_line(f"FLAGGEMS-CONFORMANCE-COMPLETE suite={suite}")
        elif write_line is not None:
            write_line(
                f"FLAGGEMS-CONFORMANCE-FAILURE suite={suite} selected={len(node_ids)}"
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
