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

"""Package-level wiring for the hardware-agnostic runtime-conformance suite.

Three jobs:

    * register the package's markers (``backend_route``, ``anyplatform``,
      ``flaggems_cpp``, ``flaggems_python``, ``boxing``);
    * skip a test that needs a kernel set this wheel did not compile -- the one
      build-driven skip the mandate allows, keyed off the build record, not a
      platform name;
    * print exactly one comparable summary line (and the JSON doc it names) from
      ``pytest_terminal_summary`` -- and only when this suite's own tests ran, so
      a broader session that merely collected this directory does not print a
      line carrying that other selection's numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The package carries __init__.py, so pytest imports these modules as
# ``runtime.<name>`` and a bare ``from runtime_caps import ...`` in a test would
# miss. Put this directory on sys.path up front so the helpers resolve by their
# bare names, the way the ops/ tree's helpers do.
_PACKAGE_DIR = str(Path(__file__).resolve().parent)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

_PACKAGE_PATH = Path(__file__).resolve().parent

#: Node IDs of this package's tests that survived selection in this session.
#: ``pytest_configure`` registers the report plugin process-wide, and a conftest
#: is loaded whenever its directory is *collected* -- so a broad
#: ``pytest tests/integration`` run registers it too. Recording the items that
#: actually ran lets the summary count only this suite's outcomes and stay
#: silent when none of them ran, instead of printing a RUNTIME-CONFORMANCE line
#: whose numbers belong to some unrelated selection.
_RUNTIME_NODEIDS: set[str] = set()


def _is_runtime_item(item: pytest.Item) -> bool:
    """Whether an item belongs to this package's directory."""
    path = getattr(item, "path", None)
    if path is None:  # pytest < 7 fell back to fspath
        path = Path(str(item.fspath))
    try:
        return _PACKAGE_PATH in path.resolve().parents
    except OSError:  # a path that will not resolve is not ours
        return False


#: capability marker -> the kernel set the build record must list for it to run.
#:
#: ``boxing`` is here for the ``cuda`` conf route: on this project's CPU-torch
#: wheels that dispatch key is provided by the generated CUDA boxing kernels, so
#: a ``= cuda`` route with no boxing in the build record has no kernel to reach
#: and the op lands on cpu_fallback instead. Same build-driven skip, same source
#: of truth as the two FlagGems entries.
_CAPABILITY_KERNEL = {
    "flaggems_cpp": "flaggems_cpp",
    "flaggems_python": "flaggems",
    "boxing": "boxing",
}

#: The JSON doc's directory. Overridable so a CI job can route it into its
#: artifacts; defaults to the run's rootdir.
_REPORT_DIR_ENV = "RUNTIME_CONFORMANCE_REPORT_DIR"


def _build_kernels() -> frozenset[str]:
    """Kernel sets from the build record, or empty when it cannot be read."""
    try:
        from runtime_caps import build_kernels

        return build_kernels()
    except Exception:  # noqa: BLE001 -- a missing record means "nothing compiled"
        return frozenset()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "anyplatform: runs on every supported hardware platform"
    )
    config.addinivalue_line(
        "markers",
        "backend_route(op): asserts the conf-declared dispatch route for a "
        "single operator",
    )
    config.addinivalue_line(
        "markers",
        "flaggems_cpp: requires the FlagGems C++ kernel set "
        "(built with FLAGOS_BUILD_FLAGGEMS_CPP=ON)",
    )
    config.addinivalue_line(
        "markers", "flaggems_python: requires the FlagGems Python kernel set"
    )
    config.addinivalue_line(
        "markers",
        "boxing: requires the generated CUDA boxing kernel set "
        "(built with FLAGOS_BUILD_BOXING=ON)",
    )
    config.pluginmanager.register(_ConformanceReport(), "runtime-conformance-report")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Record this suite's selected tests, and skip capability-marked gaps.

    ``trylast`` matters: pytest's ``-m`` deselection is another
    ``pytest_collection_modifyitems`` implementation, and running after it means
    the recorded node IDs are the tests this session will actually report on.
    """
    _RUNTIME_NODEIDS.clear()
    kernels: frozenset[str] | None = None
    for item in items:
        if not _is_runtime_item(item):
            continue
        _RUNTIME_NODEIDS.add(item.nodeid)
        for marker_name, kernel in _CAPABILITY_KERNEL.items():
            if item.get_closest_marker(marker_name) is None:
                continue
            if kernels is None:
                kernels = _build_kernels()
            if kernel not in kernels:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"kernel set {kernel!r} is not compiled into this "
                            "wheel (build record)"
                        )
                    )
                )
            break


def _count_outcomes(stats: dict, nodeids: set[str]) -> tuple[int, int, int, int]:
    """(passed, failed, skipped, errors) among ``nodeids`` in a run's stats.

    The stats belong to the whole session, so a session that ran other tests
    alongside this suite still reports this suite's numbers.
    """

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
        passed, failed, skipped, errors = _count_outcomes(stats, _RUNTIME_NODEIDS)
        if not (passed or failed or skipped or errors):
            # None of this suite's tests ran in this session: a collection-only
            # run, or a session that collected the directory but selected
            # something else entirely. Printing a line here would file another
            # selection's numbers under this suite's name.
            return
        accel, kernels, total, routed, none = _conformance_facts()
        line = (
            f"RUNTIME-CONFORMANCE accel={accel} kernels={kernels} "
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


def _accel_label() -> str:
    """Accelerator label, from the build record -- never detect_platform().

    The conf basename (``backends_<platform>.conf``) is the wheel's own
    statement of what it was built for; where the conf is unnamed (a user
    ``FLAGOS_BACKEND_CONFIG``), fall back to the recorded accelerator.
    """
    try:
        from runtime_caps import conf_path

        name = conf_path().name
        if name.startswith("backends_") and name.endswith(".conf"):
            return name[len("backends_") : -len(".conf")]
    except Exception:  # noqa: BLE001 -- fall through to the recorded accelerator
        pass
    try:
        from torch_fl import _env

        return _env.build_accelerator() or "unknown"
    except Exception:  # noqa: BLE001 -- an unreadable record is still a label
        return "unknown"


def _conformance_facts() -> tuple[str, str, int, int, int]:
    """(accel, kernels, ops_total, ops_routed, ops_none) for the summary line."""
    accel = _accel_label()
    try:
        from runtime_caps import RUNTIME_OPS, build_kernels, load_routes

        kernels_set = build_kernels()
        routes = load_routes()
    except Exception:  # noqa: BLE001 -- report the shape, not a crash
        return accel, "none", len(_runtime_ops()), 0, 0
    total = len(RUNTIME_OPS)
    none = sum(1 for op in RUNTIME_OPS if routes.get(op) == "none")
    routed = sum(1 for op in RUNTIME_OPS if routes.get(op) not in (None, "none"))
    kernels = ",".join(sorted(kernels_set)) if kernels_set else "none"
    return accel, kernels, total, routed, none


def _runtime_ops() -> list[str]:
    """RUNTIME_OPS without importing runtime_caps (last-resort fallback)."""
    try:
        from runtime_caps import RUNTIME_OPS

        return list(RUNTIME_OPS)
    except Exception:  # noqa: BLE001
        return []


def _write_doc(config: pytest.Config, payload: dict) -> None:
    """Write the JSON doc, named from the build record's accelerator."""
    try:
        import os

        directory = os.environ.get(_REPORT_DIR_ENV)
        root = Path(directory) if directory else Path(str(config.rootpath))
        name = f"runtime-conformance-{payload['accel']}.json"
        (root / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except Exception:  # noqa: BLE001 -- a report is never worth failing a run
        pass
