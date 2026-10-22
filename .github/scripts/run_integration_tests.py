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

import argparse
import importlib.machinery
import importlib.util
import json
import os
import re
import signal
import shlex
import subprocess
import sys
import uuid
from pathlib import Path


INVENTORY_GROUPS = {
    "unit_tests_per_file": "unit",
    "sdpa_lanes": "sdpa",
    "flaggems_conformance": "flaggems",
    "flaggems_kernel_conformance": "flaggems_kernels",
}
INVENTORY_SOURCES = {
    "unit_tests_per_file": "tests/unit",
    "sdpa_lanes": "tests/integration/vendor_lanes",
    "flaggems_conformance": "tests/integration/flaggems",
    "flaggems_kernel_conformance": "tests/integration/flaggems",
}

# A group command can die with a fatal signal *after* pytest has finished:
# some vendor containers tear their driver stack down during exit() (an ELF
# destructor faults), so the process dies with SIGSEGV although every test ran
# and reported. INTEGRATION_ALLOW_TEARDOWN_CRASH (set per group through that
# group's `environment` mapping) reads that case -- a complete, clean pytest
# summary followed by a fatal signal -- as a pass and prints it as a warning.
# Without the variable the crash is a failure, which is the right reading for
# a signal that arrives before the summary: nothing proved the tests
# themselves were fine. This mirrors UNIT_TESTS_ALLOW_TEARDOWN_CRASH in
# run_unit_tests.py; the two scripts are standalone files with no shared
# package, so the small matching logic below is duplicated on purpose -- keep
# the pair in sync by hand.
#
# The output below is both streamed live (a multi-minute hardware group must
# stay observable, and a killed runner must leave its log behind) and
# accumulated for the summary check.
_SUMMARY_RE = re.compile(
    r"^={0,8}\s*(?:\d+ (?:passed|skipped)(?:, )?)*"
    r"\s*in \d+(?:\.\d+)?s\s*=*$"
)
_SUMMARY_FAILURE_RE = re.compile(r"\b(?:failed|errors?|xfailed|xpassed|no tests ran)\b")

# Signals that mean the process died rather than exited: SIGSEGV from a bad
# address in a teardown destructor is the observed case; the rest are the same
# class.
_FATAL_SIGNALS = {signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS, signal.SIGILL}

_TEARDOWN_CRASH_ENV = "INTEGRATION_ALLOW_TEARDOWN_CRASH"
_SOURCE_ROOT_ENV = "INTEGRATION_SOURCE_ROOT"
_REPORT_ACTIVATION = (
    "FLAGCICD_REPORT_DIR",
    "FLAGCICD_TEST_GROUP",
    "FLAGCICD_LAUNCHER",
    "FLAGCICD_FLAGGEMS_GROUP",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument(
        "--inventory",
        action="append",
        default=[],
        metavar="GROUP=SOURCE",
        help=(
            "read GROUP from SOURCE/google-test.json; a repository-relative "
            "source directory derives the cases from its test declarations"
        ),
    )
    return parser.parse_args()


def load_configuration(allow_empty):
    try:
        tests = json.loads(os.environ.get("INTEGRATION_TESTS", "[]"))
        environment = json.loads(os.environ.get("INTEGRATION_ENVIRONMENT", "{}"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid integration configuration JSON: {error}") from error

    if not isinstance(tests, list):
        raise SystemExit("integration_tests must be an array")
    if not allow_empty and not tests:
        raise SystemExit("integration_tests must not be empty")
    for index, test in enumerate(tests, start=1):
        if not isinstance(test, dict):
            raise SystemExit(f"integration_tests[{index}] must be an object")
        name = test.get("name")
        command = test.get("command")
        test_environment = test.get("environment", {})
        if test.get("id") is not None:
            step_id = test["id"]
            if not isinstance(step_id, str) or not re.fullmatch(
                r"[a-z][a-z0-9_]*", step_id
            ):
                raise SystemExit(
                    f"integration_tests[{index}].id must be a lowercase identifier"
                )
        if (
            not isinstance(name, str)
            or not name.strip()
            or "\n" in name
            or "\r" in name
        ):
            raise SystemExit(
                f"integration_tests[{index}].name must be a non-empty single-line string"
            )
        if not isinstance(command, str) or not command.strip() or "\0" in command:
            raise SystemExit(
                f"integration_tests[{index}].command must be a non-empty string"
            )
        if test.get("requires") is not None:
            raise SystemExit(
                f"integration_tests[{index}].requires states a fixed count; "
                "use --inventory to detect missing, extra and skipped cases"
            )
        validate_environment(
            test_environment,
            f"integration_tests[{index}].environment",
        )

    validate_environment(environment, "integration_environment")

    return tests, environment


def inventory_directory(value):
    """A report directory that pytest-json-report can read as a Google report."""
    if os.path.isabs(value):
        return Path(value)
    # Node IDs have the repository root as their base. Keep the argument stable
    # between the CPU-only loader and a harness that runs in a test workspace.
    return Path(os.environ.get("INTEGRATION_INVENTORY_ROOT", os.getcwd())) / value


def load_inventory(assignments):
    """Read the exact cases for the groups that declare a shared inventory.

    A repository-relative source directory (``tests/integration/flaggems``)
    names the test declarations; the case set is derived from them. An
    absolute path or directory written by the loader names a materialized
    inventory, with its cases in ``google-test.json``. An absolute source path
    ending in a separator is accepted for callers that build such a path.
    """
    expected = {}
    for assignment in assignments:
        group, separator, name = assignment.partition("=")
        if not separator or not group or not name:
            raise SystemExit(
                f"--inventory must be GROUP=SOURCE, received {assignment!r}"
            )
        if group in expected:
            raise SystemExit(f"inventory for {group!r} is configured twice")
        if is_source_inventory(group, name):
            document = source_inventory(group)
        else:
            directory = inventory_directory(name).resolve()
            path = directory if directory.is_file() else directory / "google-test.json"
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise SystemExit(
                    f"cannot read inventory for {group}: {error}"
                ) from error
        expected[group] = validate_inventory(group, document)
    return expected


def is_source_inventory(group, value):
    """Whether this argument identifies source declarations rather than reports."""
    if os.path.isabs(value):
        return value.endswith(("/", "\\"))
    expected = (INVENTORY_SOURCES.get(group) or "").rstrip("/")
    normal = Path(value).as_posix().rstrip("/")
    return bool(expected) and normal == expected


def validate_inventory(group, document):
    """Validate an inventory according to what its cases name."""
    kind = INVENTORY_GROUPS.get(group)
    if kind is None:
        raise SystemExit(f"inventory for {group!r} has no declared kind")
    if kind == "unit":
        return validate_unit_inventory(document)
    return _validate_inventory_document(group, document)


def validate_unit_inventory(document):
    """Unit cases are files: every one has to be a test file on disk.

    ``run_unit_tests.py`` runs each unit file in its own interpreter, so the
    inventory names files rather than pytest node IDs. The hardware runner
    checks each file's report contains a completed pass; here the source
    tree is checked before any hardware is scheduled.
    """
    if not isinstance(document, dict) or set(document) != {"cases", "aliases", "group"}:
        raise SystemExit(
            "unit inventory must contain exactly 'cases', 'aliases' and 'group'"
        )
    cases, aliases = document["cases"], document["aliases"]
    if document["group"] != "unit_tests_per_file":
        raise SystemExit("unit inventory must name its group 'unit_tests_per_file'")
    if (
        not isinstance(cases, list)
        or not cases
        or not all(isinstance(case, str) and case for case in cases)
        or len(cases) != len(set(cases))
    ):
        raise SystemExit("unit inventory must be a non-empty list of unique files")
    if aliases:
        raise SystemExit("unit inventory cannot alias node IDs; its cases are files")
    root = Path(os.environ.get("INTEGRATION_SOURCE_ROOT", os.getcwd()))
    missing = [case for case in cases if not (root / case).is_file()]
    if missing:
        raise SystemExit(
            "unit inventory names missing test file(s): " + ", ".join(missing)
        )
    return document


def source_directory(require_workspace=False):
    """The checkout whose sources this runner describes.

    The loader and the hardware runner start from different directories. Use
    the checkout recorded by the loader when present, then the workflow
    workspace, and finally the tree containing this script. The last fallback
    keeps a direct CLI invocation independent of its caller's directory.
    """
    for name in (_SOURCE_ROOT_ENV, "INTEGRATION_INVENTORY_ROOT"):
        value = os.environ.get(name)
        if value and Path(value).is_dir():
            return Path(value).resolve()
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace:
        if Path(workspace).is_dir():
            return Path(workspace).resolve()
        if require_workspace:
            raise SystemExit(f"GITHUB_WORKSPACE does not exist: {workspace}")
    return Path(__file__).resolve().parents[2]


def source_inventory(group):
    """Derive one group's source cases through the shared inventory module."""
    path = Path(__file__).with_name("flaggems_cases.py")
    spec = importlib.util.spec_from_file_location("flaggems_cases", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the shared inventory module: {path}")
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    original_root = cases.ROOT
    cases.ROOT = source_directory()
    try:
        kinds = getattr(cases, "INVENTORY_KINDS", {})
        if kinds.get(group) != INVENTORY_GROUPS[group]:
            raise SystemExit(
                f"inventory source module disagrees that {group!r} names "
                f"{INVENTORY_GROUPS[group]!r}"
            )
        try:
            document = cases.inventory_document(INVENTORY_GROUPS[group])
        except (cases.InventoryError, OSError, UnicodeError) as error:
            raise SystemExit(f"cannot read inventory for {group}: {error}") from error
    finally:
        cases.ROOT = original_root
    return validate_inventory(group, document)


def flaggems_expected_cases(inventory_by_group):
    """Pass one explicit case inventory to both FlagGems passes."""
    document = inventory_by_group.get("flaggems_conformance")
    if document is None:
        return ""
    return json.dumps(
        {"flaggems_conformance": document["cases"]}, separators=(",", ":")
    )


def inventory_environment(test_id, inventory_by_group):
    """Environment that binds a run command to the same cases CI validates."""
    if test_id not in ("flaggems_conformance", "flaggems_kernel_conformance"):
        return {}
    expected = flaggems_expected_cases(inventory_by_group)
    if not expected:
        raise SystemExit(
            f"Integration test {test_id!r} requires the shared FlagGems "
            "conformance inventory"
        )
    environment = {"FLAGCICD_EXPECTED_CASES": expected}
    # The native contract runner records its pass, and a local native runner
    # hands the same inventory to its child. A kernel pass in this same CI
    # environment requires that record, so invoking only the kernel command
    # cannot report a complete two-pass contract.
    return environment


def runner_environment(test_environment, report_directory):
    """Start the pytest owner as the one group inside its checkout.

    Integration groups normally run in a wheel-only directory. A FlagGems
    command starts pytest itself, so it needs the checkout's test tree while
    preserving the reporting activation from the hardware step.
    """
    environment = test_environment.copy()
    if (
        test_environment.get("FLAGCICD_LAUNCHER") == "group"
        and test_environment.get("FLAGCICD_REPORT_DIR") == report_directory
    ):
        activation = {
            name: test_environment.get(name, "") for name in _REPORT_ACTIVATION
        }
        environment.update(activation)
    return environment


def _validate_inventory_document(group, document):
    """Validate a legacy list or a case document with native pytest aliases."""
    if isinstance(document, list):
        cases, aliases = document, {}
    elif isinstance(document, dict):
        if set(document) != {"cases", "aliases"}:
            raise SystemExit(
                f"inventory for {group} must contain exactly 'cases' and 'aliases'"
            )
        cases, aliases = document["cases"], document["aliases"]
    else:
        raise SystemExit(
            f"inventory for {group} must be a list or an object with cases and aliases"
        )
    if (
        not isinstance(cases, list)
        or not cases
        or not all(isinstance(case, str) and case for case in cases)
        or len(cases) != len(set(cases))
    ):
        raise SystemExit(
            f"inventory for {group} must be a non-empty list of unique node IDs"
        )
    if not isinstance(aliases, dict):
        raise SystemExit(f"aliases for {group} must be an object")
    wanted = set(cases)
    native_ids = set()
    for case, values in aliases.items():
        if case not in wanted:
            raise SystemExit(
                f"inventory alias for {group} names undeclared case {case!r}"
            )
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
            or len(values) != len(set(values))
        ):
            raise SystemExit(
                f"aliases for {group} case {case!r} must be a non-empty "
                "list of unique node IDs"
            )
        native_ids.update(values)
    collision = sorted(wanted & native_ids)
    if collision:
        raise SystemExit(
            f"inventory alias for {group} collides with case(s): "
            + ", ".join(collision)
        )
    if len(native_ids) != sum(len(values) for values in aliases.values()):
        raise SystemExit(f"inventory alias for {group} is assigned to multiple cases")
    return {"cases": cases, "aliases": aliases}


def _report_nodeid(test):
    """Read a pytest-json-report node ID in its import-independent form."""
    nodeid = test.get("nodeid")
    if not isinstance(nodeid, str) or not nodeid:
        return ""
    return nodeid.split("/", 1)[1] if "/" in nodeid else nodeid


def _report_outcome(test):
    """A top-level passed outcome still has to pass its setup and teardown."""
    outcome = test.get("outcome")
    if outcome != "passed":
        return str(outcome or "unknown")
    for phase in ("setup", "call", "teardown"):
        phase_outcome = test.get(phase, {}).get("outcome")
        if phase_outcome == "skipped":
            return "skipped"
        if phase_outcome not in (None, "passed"):
            return str(phase_outcome)
    return "passed"


def _unit_report_file(source, report_path, tests, expected):
    """Resolve a per-file unit run to exactly one source-relative test file."""
    inventory = expected.get("cases", []) if isinstance(expected, dict) else []
    if not isinstance(inventory, list):
        raise SystemExit("unit inventory has no list of source files")
    marker = {}
    try:
        marker = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    recorded = marker.get("unit_file", "") if isinstance(marker, dict) else ""
    recorded_path = Path(recorded) if isinstance(recorded, str) and recorded else None
    report_name = report_path.with_suffix("").name.removeprefix("unit-")
    requested = []
    vague_names = []
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        args = report.get("args", [])
    except (OSError, ValueError, AttributeError):
        args = []
        report = {}
    supplied_args = [value for value in args if isinstance(value, str)]
    for value in args:
        if not isinstance(value, str) or value.startswith("-"):
            continue
        if not value.endswith(".py"):
            continue
        candidate = Path(value).as_posix()
        if candidate in inventory:
            requested.append(candidate)
        elif (
            recorded_path is not None
            and Path(value).resolve() == recorded_path.resolve()
        ):
            requested.extend(
                case for case in inventory if case.endswith("/" + recorded_path.name)
            )
        elif Path(value).suffix == ".py":
            matches = [
                case for case in inventory if case.endswith("/" + Path(value).name)
            ]
            if len(matches) == 1:
                requested.extend(matches)
            else:
                vague_names.append(Path(value).name)
    # pytest-json-report omits its own argv when the JSON path is absolute.
    # The file runner's marker says which invocation produced this report.
    marker_path = Path(recorded).resolve() if recorded_path is not None else None
    marker_proves_file = marker_path is not None and marker_path.name == report_name
    tests = report.get("tests", []) if isinstance(report, dict) else []
    if not isinstance(tests, list):
        tests = []
    files = set()
    report_files = {Path(_report_nodeid(test).split("::", 1)[0]).name for test in tests}
    for case in inventory:
        file = Path(case).as_posix()
        for test in tests:
            tested = _report_nodeid(test).split("::", 1)[0]
            if (
                tested == file
                or (tested.endswith(file) and tested[-len(file) - 1] in ("/", "\\"))
                or Path(tested).name == Path(case).name
            ):
                files.add(case)
                break
    # A collection report can contain only a child path relative to pytest's
    # current directory. The file-oriented runner names this report after its
    # source file, so the filename supplies that one missing path segment.
    if not files and len(report_files) == 1:
        suffix = report_files.pop()
        matches = [case for case in inventory if Path(case).name == suffix]
        if len(matches) == 1:
            files.add(matches[0])
    if requested:
        files &= set(requested)
    if (
        len(inventory) == 1
        and len(files) == 1
        and (len(requested) == 1 or (marker_proves_file and not supplied_args))
    ):
        # run_unit_tests.py runs one inventory file per report.
        return inventory[0]
    resolved = set(files)
    return resolved.pop() if len(resolved) == 1 else ""


def report_cases(directory, allow_unit_files=False, expected=None):
    """Collect exact outcomes from complete JSON reports for one group."""
    if not directory:
        raise SystemExit(
            "an inventory is configured, but no pytest report directory was supplied"
        )
    cases = {}
    credit_files = allow_unit_files
    unit_reports = (
        sorted(Path(directory).rglob("unit-*.unit")) if allow_unit_files else []
    )
    reports = sorted(
        path
        for path in Path(directory).rglob("*.pending")
        if not (allow_unit_files and path.with_suffix(".unit").exists())
    )
    if not reports and not unit_reports:
        raise SystemExit(
            f"no pytest report was written to {directory}; cannot verify inventory"
        )
    for marker in reports:
        report_path = marker.with_suffix(".json")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"pytest session {marker.name} is incomplete: {error}"
            ) from error
        tests = report.get("tests")
        if not isinstance(tests, list) or not isinstance(report.get("summary"), dict):
            raise SystemExit(f"pytest session {marker.name} has no complete test data")
        for test in tests:
            nodeid = _report_nodeid(test)
            if not nodeid:
                raise SystemExit(f"pytest session {marker.name} has a test with no ID")
            outcome = _report_outcome(test)
            if nodeid in cases:
                previous = cases[nodeid]
                if previous == outcome:
                    continue
                if {previous, outcome} == {"passed", "xfailed"}:
                    outcome = "xfailed"
                else:
                    raise SystemExit(
                        f"pytest case {nodeid} appears with both "
                        f"{cases[nodeid]} and {outcome} outcomes"
                    )
            cases[nodeid] = outcome
    # The unit runner owns one JSON report per file. Its group's own pytest
    # summary is printed by the runner itself, so it has no group session to
    # pair with a pending marker. Every per-file report must still be complete.
    for report_path in unit_reports:
        marker = report_path
        report_path = marker.with_suffix(".json")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"pytest session {report_path.name} is incomplete: {error}"
            ) from error
        tests = report.get("tests")
        if not isinstance(tests, list) or not isinstance(report.get("summary"), dict):
            raise SystemExit(
                f"pytest session {report_path.name} has no complete test data"
            )
        unit_file = (
            _unit_report_file(marker, report_path, tests, expected) if tests else ""
        )
        report_outcomes = [_report_outcome(test) for test in tests]
        if credit_files and not unit_file:
            raise SystemExit(
                f"unit report {report_path.name} has no unique shared inventory file"
            )
        for test in tests:
            nodeid = _report_nodeid(test)
            if not nodeid:
                raise SystemExit(
                    f"pytest session {report_path.name} has a test with no ID"
                )
            if not credit_files:
                cases[nodeid] = _report_outcome(test)
        # An unexpected xpass is a failure. Report it at the file level even
        # though pytest's own process did not fail without ``-rx``.
        file_failures = {
            outcome
            for outcome in report_outcomes
            if outcome not in {"passed", "xfailed"}
        }
        if "xpassed" in file_failures:
            file_failures.add("failed")
        report_outcome = (
            "passed"
            if report_outcomes
            and all(outcome in {"passed", "xfailed"} for outcome in report_outcomes)
            else ("failed" if "failed" in file_failures else sorted(file_failures)[0])
        )
        if credit_files:
            if unit_file in cases and cases[unit_file] != report_outcome:
                raise SystemExit(
                    f"unit file {unit_file} has conflicting report outcomes"
                )
            cases[unit_file] = report_outcome
    return cases


def verify_inventory(name, actual, expected):
    """Require the exact inventory to pass without skipped cases."""
    if isinstance(expected, list):
        cases, aliases = expected, {}
    elif isinstance(expected, dict):
        cases, aliases = expected["cases"], expected["aliases"]
    else:
        raise SystemExit(f"Inventory for '{name}' has an unsupported document")
    credited = set()
    alias_passed = set()
    wrong = []
    for case in cases:
        reported = [
            nodeid for nodeid in (case, *aliases.get(case, [])) if nodeid in actual
        ]
        for nodeid in reported:
            outcome = actual[nodeid]
            if outcome not in {"passed", "xfailed"}:
                wrong.append((nodeid, outcome))
            elif nodeid == case:
                credited.add(case)
            else:
                alias_passed.add(case)
    credited |= alias_passed
    missing = sorted(set(cases) - credited)
    wanted = set(cases) | {native for values in aliases.values() for native in values}
    extra = sorted(set(actual) - wanted)
    wrong.sort()
    if missing or extra or wrong:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("extra: " + ", ".join(extra))
        if wrong:
            details.append(
                "not passed: "
                + ", ".join(f"{case}={outcome}" for case, outcome in wrong)
            )
        raise SystemExit(
            f"Integration test '{name}' does not match its shared inventory; "
            + "; ".join(details)
        )
    print(
        (
            f"Inventory met: {len(cases)} shared cases completed without "
            "missing, extra or skipped cases"
        ),
        flush=True,
    )


def safe_report_directory(name):
    """A fresh report location avoids attributing a prior attempt's case."""
    directory = os.environ.get("FLAGCICD_REPORT_DIR", "")
    if not directory:
        return ""
    slug = re.sub(r"[^A-Za-z0-9_.-]", "-", name)
    path = Path(directory) / f"attempt-{slug}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def collect_unit_evidence(source, destination, expected):
    """Move a per-file runner's assigned reports into its isolated directory.

    CI supplies the per-file runner a writable report directory. The integration
    runner takes ownership of that directory before invoking the group, so copy
    the runner's reports across and let the group's normal verifier inspect the
    exact files it produced. The copied markers retain their source path.
    """
    if expected is None:
        return
    target = Path(destination)
    original = Path(source)
    if not target.is_dir() or not original.is_dir():
        return
    (target / "google-test.json").write_text(
        json.dumps(expected) + "\n", encoding="utf-8"
    )
    for name in ("unit-*.unit", "unit-*.json", "*.pending"):
        for evidence in original.glob(name):
            if evidence.is_file():
                (target / evidence.name).write_text(
                    evidence.read_text(encoding="utf-8"), encoding="utf-8"
                )


def preserve_failed_reports(original, group):
    """Keep reports written before a group fails outside its pytest run.

    A vendor image can stop a group before pytest writes a report. The merge
    step then never sees the group, so the CI log is the only evidence of the
    failure. Copy any report the group did write to its stable location; the
    runner still returns the group's failure status.
    """
    if not original or not group or original == group:
        return
    source = Path(original)
    if not source.is_dir():
        return
    target = Path(group)
    target.mkdir(parents=True, exist_ok=True)
    inventory = source / "google-test.json"
    if inventory.is_file() and not (target / inventory.name).exists():
        (target / inventory.name).write_text(
            inventory.read_text(encoding="utf-8"), encoding="utf-8"
        )
    for evidence in source.iterdir():
        if evidence.is_file() and (
            evidence.suffix in {".json", ".pending"} or evidence.name.endswith(".unit")
        ):
            destination = target / evidence.name
            if not destination.exists():
                destination.write_text(
                    evidence.read_text(encoding="utf-8"), encoding="utf-8"
                )


def prepare_failed_reports(name, source, group):
    """Give a failed environment preflight a place in the merged report.

    Without it, a group that dies before its first pytest session, such as a
    missing ``mpirun`` or canonical environment file, has no evidence in the
    report artifact. This marker records that gap and is not a passing test.
    """
    if not source or not group:
        return
    preserve_failed_reports(source, group)
    marker = Path(group) / f"environment-{uuid.uuid4().hex}.pending"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "group": name,
                "args": [],
                "error": "the integration group environment preparation failed",
            }
        ),
        encoding="utf-8",
    )


def check_inventory_binding(tests, inventory_by_group):
    """Require each shared inventory to name a group in this platform's plan."""
    group_ids = {test.get("id") for test in tests}
    missing = sorted(set(inventory_by_group) - group_ids)
    if missing:
        raise SystemExit(
            "shared inventory group(s) are absent from this platform's plan: "
            + ", ".join(missing)
        )


def validate_environment(environment, field_name):
    """Validate one integration-test environment mapping."""
    if not isinstance(environment, dict):
        raise SystemExit(f"{field_name} must be an object")
    for key, value in environment.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise SystemExit(
                f"invalid environment variable name in {field_name}: {key}"
            )
        if (
            not isinstance(value, str)
            or "\n" in value
            or "\r" in value
            or "\0" in value
        ):
            raise SystemExit(f"{field_name}[{key}] must be a single-line string")


def clean_summary(output):
    """Return pytest's summary line when it shows a completed, clean run.

    ``None`` means no such line exists (the command died before the session
    ended) or the summary reports failures -- either way the group's verdict
    is unknown or negative, never a pass.
    """
    for line in reversed(output.splitlines()):
        if _SUMMARY_RE.search(line):
            if _SUMMARY_FAILURE_RE.search(line):
                return None
            return line.strip("= ")
    return None


def teardown_crash_allowed(test_environment):
    """Whether this group tolerates a fatal signal after a clean summary."""
    return test_environment.get(_TEARDOWN_CRASH_ENV, "").lower() not in (
        "",
        "0",
        "off",
        "false",
    )


def classify_command_result(returncode, output, allow_teardown_crash):
    """Map one group command onto (ok, tolerated).

    Only a fatal *signal* after a complete, clean pytest summary is
    tolerable, and only when the group opted in: a plain nonzero exit, a
    signal before the summary, or a summary reporting failures all fail, with
    or without the opt-in.
    """
    if returncode == 0:
        return True, False
    fatal = (returncode < 0 and -returncode in _FATAL_SIGNALS) or (
        returncode > 128 and (returncode - 128) in _FATAL_SIGNALS
    )
    if fatal and allow_teardown_crash and clean_summary(output):
        return True, True
    return False, False


def signal_number(returncode):
    """Human-readable signal for a fatal-signal returncode."""
    if returncode < 0:
        return -returncode
    return returncode - 128


def required_mpirun(command, test_environment):
    """Refuse a command whose MPI launcher is absent before bash starts it.

    Without this check, the command starts, ``mpirun`` prints
    ``command not found``, and the following pytest receives SIGPIPE while
    writing its collection report. The runner then reports a crash, hiding
    the missing launcher that actually stopped the group.
    """
    try:
        words = shlex.split(command)
    except ValueError:
        return
    mpirun = next((word for word in words if Path(word).name == "mpirun"), None)
    if mpirun is None:
        return
    if Path(mpirun).is_absolute() or len(Path(mpirun).parts) > 1:
        if Path(mpirun).is_file() and os.access(mpirun, os.X_OK):
            return
    else:
        for directory in test_environment.get("PATH", os.defpath).split(os.pathsep):
            if directory and (Path(directory) / "mpirun").is_file():
                if os.access(Path(directory) / "mpirun", os.X_OK):
                    return
                break
    raise SystemExit(
        "mpirun is required by this integration group but is not installed or on PATH"
    )


def run_command(command, test_environment, workdir, command_prefix=""):
    """Run one group command, streaming its output live while capturing it.

    Streaming (not print-at-end) keeps long hardware groups observable and
    leaves the log behind if the runner itself is killed; the captured copy
    feeds the teardown-crash check.
    """
    required_mpirun(command, test_environment)
    process = subprocess.Popen(
        [
            "bash",
            "-c",
            f"set -euo pipefail\n{command_prefix}{command}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=test_environment,
        cwd=workdir,
    )
    assert process.stdout is not None
    try:
        lines = []
        for line in process.stdout:
            sys.stdout.write(line)
            lines.append(line)
        sys.stdout.flush()
        returncode = process.wait()
    except KeyboardInterrupt:
        # A cancelled runner must not leave a pytest process writing to the
        # report directory or to a retry's workspace.
        process.kill()
        process.wait()
        raise
    return returncode, "".join(lines)


def main():
    args = parse_args()
    tests, environment = load_configuration(args.allow_empty)
    inventory_by_group = load_inventory(
        [os.path.expandvars(path) for path in getattr(args, "inventory", [])]
    )
    check_inventory_binding(tests, inventory_by_group)
    if args.validate_only:
        print(f"Validated {len(tests)} configured integration test(s)")
        return 0

    command_environment = os.environ.copy()
    command_environment.update(environment)
    reporting = command_environment.get("FLAGCICD_REPORT_DIR", "")
    reporting_path = str((Path(__file__).with_name("pytest_reporting")).resolve())
    if reporting_path not in sys.path:
        sys.path.insert(0, reporting_path)
    if reporting:
        path = Path(__file__).with_name("pytest_reporting") / "flagcicd_report.py"
        spec = importlib.util.spec_from_file_location("flagcicd_report", path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load FlagCICD reporting helper: {path}")
        reporting_plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reporting_plugin)
    workdir = os.environ.get("INTEGRATION_WORKDIR")
    if workdir and not os.path.isdir(workdir):
        raise SystemExit(f"INTEGRATION_WORKDIR does not exist: {workdir}")
    # Passing the installed parent through every group disturbed vendor torch
    # facades. A small startup hook selects the built torch_fl only around a
    # command that runs a source test tree.
    import sysconfig

    package = next(
        (
            Path(entry) / "torch_fl"
            for entry in sys.path
            if (Path(entry) / "torch_fl/__init__.py").is_file()
            and any(
                (Path(entry) / "torch_fl" / f"_C{suffix}").is_file()
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
            )
        ),
        None,
    )
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    if package is not None and package.parent.resolve() != purelib:
        command_environment["PYTHONPATH"] = os.pathsep.join(
            [str(purelib), command_environment.get("PYTHONPATH", "")]
        )
    for index, test in enumerate(tests, start=1):
        print(
            f"===== [{index}/{len(tests)}] {test['name']} =====",
            flush=True,
        )
        test_environment = command_environment.copy()
        explicit_environment = inventory_environment(test.get("id"), inventory_by_group)
        for key, value in {
            **test.get("environment", {}),
            **explicit_environment,
        }.items():
            test_environment[key] = os.path.expandvars(value)
        if (
            package is not None
            and package.parent.resolve() != purelib
            and "tests/unit" in str(test.get("command", ""))
        ):
            hook = str(Path(__file__).resolve().parent)
            test_environment["PYTHONPATH"] = os.pathsep.join(
                [
                    str(package.parent),
                    hook,
                    test_environment.get("PYTHONPATH", ""),
                ]
            )
            test_environment["TORCH_FL_INSTALLED_PACKAGE_ROOT"] = str(package.parent)
            test_environment["TORCH_FL_SOURCE_ROOT"] = os.environ.get(
                "GITHUB_WORKSPACE", str(Path.cwd())
            )
        reporting = test_environment.get("FLAGCICD_REPORT_DIR", "")
        source = reporting
        if reporting:
            path = Path(__file__).with_name("pytest_reporting") / "flagcicd_report.py"
            source = test_environment["FLAGCICD_REPORT_DIR"]
            spec = importlib.util.spec_from_file_location("flagcicd_report", path)
            if spec is None or spec.loader is None:
                raise SystemExit(f"cannot load FlagCICD reporting helper: {path}")
            reporting_plugin = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reporting_plugin)
        command_prefix = ""
        group_report = safe_report_directory(test["name"]) if reporting else ""
        group_command_environment = test_environment.copy()
        expected = inventory_by_group.get(test.get("id")) if test.get("id") else None
        if expected is not None and not reporting:
            raise SystemExit(
                f"Integration test '{test['name']}' is in the shared inventory, "
                "but no pytest report directory was supplied"
            )
        if reporting:
            test_environment["FLAGCICD_REPORT_DIR"] = group_report
            test_environment["FLAGCICD_TEST_GROUP"] = test["name"]
            test_environment["FLAGCICD_FLAGGEMS_GROUP"] = test["name"]
            group_command_environment = runner_environment(
                test_environment, group_report
            )
        try:
            if reporting:
                command_prefix = reporting_plugin.guard_command(test_environment)
            if test.get("id") == "unit_tests_per_file":
                collect_unit_evidence(source, group_report, expected)
        except BaseException as error:
            prepare_failed_reports(test["name"], source, group_report)
            print(
                f"::error::Integration test '{test['name']}' failed during "
                f"environment preparation: {error}",
                flush=True,
            )
            raise
        allow_crash = teardown_crash_allowed(test_environment)
        group_workdir = workdir
        if test.get("id") in INVENTORY_SOURCES:
            # pytest puts the rootdir at the front of sys.path when it imports
            # a conftest from a source checkout. Starting this group in the
            # wheel-only workspace would therefore select that workspace's
            # torch_fl package, which lacks the extension built in the
            # checkout. Run the source-tree suite from the tree it tests.
            source = source_directory(require_workspace=True)
            group_workdir = str(source)
            if not os.path.isdir(group_workdir):
                raise SystemExit(
                    f"integration source root does not exist: {group_workdir}"
                )
            if test.get("id", "").startswith("flaggems_"):
                group_command_environment["TORCH_FL_SOURCE_ROOT"] = str(source)
        try:
            returncode, output = run_command(
                test["command"],
                group_command_environment,
                group_workdir,
                command_prefix,
            )
        except BaseException as error:
            prepare_failed_reports(test["name"], source, group_report)
            print(
                f"::error::Integration test '{test['name']}' failed during "
                f"environment preparation: {error}",
                flush=True,
            )
            raise
        ok, tolerated = classify_command_result(returncode, output, allow_crash)
        inventory_error = None
        if expected is not None:
            try:
                cases = report_cases(
                    group_report,
                    allow_unit_files=test.get("id") == "unit_tests_per_file",
                    expected=expected,
                )
                verify_inventory(test["name"], cases, expected)
            except SystemExit as error:
                inventory_error = error
        if tolerated:
            print(
                f"Integration test '{test['name']}' passed "
                f"({clean_summary(output)}), then the command died with "
                f"signal {signal_number(returncode)} during teardown "
                f"({_TEARDOWN_CRASH_ENV} is set for this group; see this "
                f"script's header comment)",
                flush=True,
            )
        elif not ok:
            print(
                f"Integration test '{test['name']}' failed with exit code {returncode}",
                flush=True,
            )
        if not ok and not tolerated:
            # A plain nonzero exit has a useful native status to preserve.
            # Signals and tolerated teardown crashes retain the inventory
            # diagnostic, which is raised when the command itself succeeded.
            preserve_failed_reports(source, group_report)
            if returncode > 0 or inventory_error is None:
                raise SystemExit(returncode)
            raise inventory_error
        if inventory_error is not None:
            preserve_failed_reports(source, group_report)
            raise inventory_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
