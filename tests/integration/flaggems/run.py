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

"""Run the shared FlagGems conformance contract for one wheel.

The native contract runs on every wheel. A wheel built with the FlagGems kernel
set must also pass the kernel contract. A wheel without that set refuses the
kernel pass instead of reporting a skip or borrowing the native result. Each
pass starts with fresh pytest state.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SUITES = ("native", "flaggems")
_ROOT = Path(__file__).resolve().parents[2]
_KERNEL_SET = "flaggems"
_CI_GROUPS = {
    "native": "flaggems_conformance",
    "flaggems": "flaggems_kernel_conformance",
}
_ROOT_ENV = "FLAGCICD_PYTHON_ROOT"
_OUTPUT_LIMIT = 64 * 1024


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SUITES)
    return parser.parse_args(argv)


def _torch_fl(root=None):
    """Read the installed wheel's build record once, after its device loads."""

    root = Path(root or _ROOT).resolve()
    # The integration runner selects this checkout in sitecustomize. A later
    # nested pytest session is free to add the test tree to its own path.
    wanted = Path(os.environ.get("TORCH_FL_INSTALLED_PACKAGE_ROOT", root))
    wanted = (wanted / "torch_fl/__init__.py").resolve()
    package = sys.modules.get("torch_fl")
    if package is None or Path(package.__file__).resolve() != wanted:
        spec = importlib.util.spec_from_file_location("torch_fl", wanted)
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load torch_fl from {wanted}")
        wheel = importlib.util.module_from_spec(spec)
        sys.modules["torch_fl"] = wheel
        spec.loader.exec_module(wheel)
        package = wheel
    return package


def wheel_kernels(root=None):
    """Read the installed wheel's build record once, after its device loads."""
    return _torch_fl(root)._env.build_kernels()


def pytest_arguments(suite):
    return [
        str(Path(__file__).resolve().parent),
        f"--flaggems-conformance-suite={suite}",
        "-v",
        "-s",
        "--tb=short",
    ]


def launch(suite, *, owner=False):
    """Run the selected pass in either a local or a CI-owned session."""
    if owner:
        return _launch_ci_group(suite)
    # A local pass gets its own pytest state. Inherited argv and CI activation
    # must not join the caller's report. The fresh process runs pytest
    # directly, so it cannot launch another runner.
    launcher = _physical_python(os.environ.get(_ROOT_ENV))
    process = subprocess.Popen(
        [launcher, "-m", "pytest", *pytest_arguments(suite)],
        cwd=_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_local_child_environment(),
    )
    tail = _Tail(collections.deque())
    try:
        while data := process.stdout.read1(8192):
            chunk = data.decode(errors="replace")
            tail.write(chunk)
            print(chunk, end="", flush=True)
    finally:
        process.stdout.close()
    return process.wait()


def _launch_ci_group(suite):
    """Run the group from the pytest session that owns its CI report.

    The hardware runner starts this script inside a pytest group process. The
    outer process captured the report activation before its plugins could
    strip it. Start the only selected pass in that process, with no nested
    report, so its pytest JSON report and conformance inventory share the CI
    session this script started.
    """
    os.environ["FLAGCICD_TEST_GROUP"] = _CI_GROUPS[suite]
    os.environ["FLAGCICD_LAUNCHER"] = "group"
    return int(pytest.main(pytest_arguments(suite)))


def _local_child_environment():
    """Keep a local invocation from joining its caller's CI report."""
    environment = os.environ.copy()
    environment["TORCH_FL_SOURCE_ROOT"] = str(_ROOT)
    for name in (
        "FLAGCICD_REPORT_DIR",
        "FLAGCICD_COVERAGE_DIR",
        "FLAGCICD_TEST_GROUP",
        "FLAGCICD_LAUNCHER",
        "FLAGCICD_FLAGGEMS_GROUP",
        "FLAGCICD_PYTHON_NO_LAUNCHER",
    ):
        environment.pop(name, None)
    return environment


def _physical_python(launcher):
    """Find the interpreter behind a CI shell wrapper."""
    if not launcher:
        return sys.executable
    path = Path(launcher)
    try:
        first = path.read_bytes()[:4096]
    except (OSError, ValueError):
        return launcher
    if not path.is_absolute():
        return launcher
    if first.startswith(b"#!"):
        line = first.splitlines()[0].decode(errors="replace")[2:].strip()
    else:
        try:
            wrapper = path.read_text(errors="replace")
        except (OSError, ValueError):
            return launcher
        match = re.search(r'^exec "([^"]+)" "\$@"$', wrapper, re.MULTILINE)
        if match is None:
            return launcher
        line = match.group(1)
    words = line.split()
    if not words or not Path(words[-1]).is_absolute():
        return launcher
    return words[-1]


def main(argv=None):
    args = parse_args(argv)
    plan = os.environ.get("FLAGCICD_EXPECTED_CASES")
    owned_group = os.environ.get("FLAGCICD_TEST_GROUP")
    if owned_group and owned_group != _CI_GROUPS[args.suite]:
        raise SystemExit(
            f"the FlagGems {args.suite} pass runs in CI group {owned_group!r}, "
            f"not {_CI_GROUPS[args.suite]!r}"
        )
    if plan:
        try:
            expected = json.loads(plan).get("flaggems_conformance")
        except (AttributeError, json.JSONDecodeError) as error:
            raise SystemExit(f"the CI FlagGems plan is invalid: {error}") from error
        if not expected:
            raise SystemExit("the CI FlagGems plan has no conformance cases")
        native_plan = os.environ.get("FLAGCICD_FLAGGEMS_PLAN")
        if args.suite == "flaggems" and owned_group and native_plan != plan:
            raise SystemExit(
                "the FlagGems kernel pass requires the native CI pass to run "
                "under the same validated inventory"
            )
    wheel = _torch_fl()
    kernels = wheel._env.build_kernels()
    has_flaggems = "flaggems" in kernels
    print(
        f"FlagGems conformance: suite={args.suite} "
        f"kernel_sets={','.join(sorted(kernels)) or 'none'} "
        f"wheel_flaggems={str(has_flaggems).lower()}",
        flush=True,
    )
    if args.suite == "flaggems" and not has_flaggems:
        raise SystemExit(
            "the FlagGems conformance suite requires a wheel built with the "
            f"{_KERNEL_SET!r} kernel set, found {sorted(kernels)}"
        )
    if plan and args.suite == "native":
        # The local runner starts a fresh interpreter; CI's native group is
        # the one that records this validated inventory for the kernel pass.
        os.environ["FLAGCICD_FLAGGEMS_PLAN"] = plan
    status = launch(args.suite, owner=owned_group == _CI_GROUPS[args.suite])
    if status != 0:
        print(
            f"the FlagGems conformance suite ({args.suite}) exited with status {status}",
            file=sys.stderr,
        )
        return status
    return 0


class _Tail:
    """Forward output while retaining only the report tail used for checks."""

    def __init__(self, output):
        self.output = output

    def write(self, text):
        self.output.append(text)
        while sum(map(len, self.output)) > _OUTPUT_LIMIT:
            self.output.popleft()
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
