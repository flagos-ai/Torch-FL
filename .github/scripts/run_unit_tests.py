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

"""Run tests/unit one file per interpreter and report per-file results.

Why not a single ``pytest tests/unit`` call: a unit test file on an accelerator
platform can take the whole process down without unwinding -- a bad device
address is reported by the driver as VMFault, and the process dies before
pytest can print which test was running. In one shared interpreter that costs
every result collected after it, including in files that had already passed.
One process per file bounds the blast radius to that file and keeps the failure
attributable to it.

The exit status alone is not enough to judge a file, because a crash can also
happen *after* pytest has finished. Some DCU containers tear their RDMA stack
down during ``exit()``: libibverbs' ELF destructor calls into libnl-route-3's
``rtnl_tc_unregister``, which faults, so the interpreter dies with SIGSEGV
although every test ran and reported. Set ``UNIT_TESTS_ALLOW_TEARDOWN_CRASH=1``
to read that case -- a complete, clean pytest summary followed by a fatal
signal -- as a pass and print it as a warning. Whether a given container has the
fault is not something this script can assume either way, which is why the
tolerance is opt-in. Without the variable the crash is a failure, which is the
right reading for a signal that arrives before the summary: nothing proved the
tests themselves were fine.

Usage:
    python .github/scripts/run_unit_tests.py [--test-dir tests/unit] [--timeout 300]
"""

import argparse
import importlib
import importlib.machinery
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import uuid
from pathlib import Path

# pytest prints its counts on the last line that mentions a duration, e.g.
# "24 passed in 19.28s" (with a banner when not run under -q). Matching on the
# duration is what distinguishes the summary from the per-test progress output.
_SUMMARY_RE = re.compile(
    r"^={0,8}\s*(?:(?:\d+ (?:passed|skipped))(?:\s*,\s*|\s+)){0,2}"
    r"in \d+(?:\.\d+)?s\s*=*$"
)
_SUMMARY_FAILURE_RE = re.compile(
    r"\b(?:failed|errors?|xfailed|xpassed|no tests ran|Fatal Python error)\b"
)

# Signals that mean the interpreter died rather than exited: SIGABRT from an
# uncaught C++ exception in a static initializer, SIGSEGV from a bad address.
_FATAL_SIGNALS = {signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS, signal.SIGILL}
_ALLOW_TEARDOWN_CRASH = os.environ.get(
    "UNIT_TESTS_ALLOW_TEARDOWN_CRASH", ""
).lower() not in (
    "",
    "0",
    "off",
    "false",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="tests/unit")
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--wheel-package-root",
        help=(
            "directory holding the installed torch_fl package; children import "
            "it ahead of the source tree while tests and fixtures stay in the "
            "checkout"
        ),
    )
    return parser.parse_args()


def child_environment(wheel_package_root=""):
    """Select the built extension while keeping the source tree on the path.

    pytest adds the package's parent to its own import path before reading a
    source conftest. A source tree beside that package would then make pytest
    replace ``torch_fl`` with its Python-only copy. Subprocesses inherit the
    installed parent without any checkout entry that can shadow it.
    """
    # CPython on macOS passes the parent's path digests to the child and may
    # restart configuration from them after either path variable changes.
    # They are parent-process cache entries, never a setting for the child.
    env = {
        name: value
        for name, value in os.environ.items()
        if not (
            name.startswith("__PYVENV_LAUNCHER__:")
            or name.startswith("PYTHONHOME:")
            or _is_path_digest_entry(name)
        )
    }
    if not wheel_package_root:
        return env
    root = Path(wheel_package_root).resolve()
    selected_torch_fl(root)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(root),
            str(Path(__file__).resolve().parent),
            *_without_checked_out_torch_fl(env),
        ]
    )
    if env.get("TORCH_FL_TEST_SITE_PACKAGE") != "1":
        env["TORCH_FL_INSTALLED_PACKAGE_ROOT"] = str(root)
    repo_root = Path(__file__).resolve().parents[2]
    cwd = Path.cwd().resolve()
    source_root = cwd if (cwd / "tests/unit").is_dir() else repo_root
    env["TORCH_FL_SOURCE_ROOT"] = str(source_root)
    return env


def _without_checked_out_torch_fl(env):
    """Discard the import path of the source package from a child."""
    source_root = Path(env.get("TORCH_FL_SOURCE_ROOT", "") or Path.cwd()).resolve()
    return [
        entry
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry and Path(entry).resolve() != source_root
    ]


def selected_torch_fl(root):
    """Return the installed package, rejecting a Python-only directory."""
    package = root / "torch_fl"
    test_site = os.environ.get("TORCH_FL_TEST_SITE_PACKAGE") == "1"
    if not package.is_dir():
        raise SystemExit(f"wheel package root has no torch_fl directory: {root}")
    if not test_site and not (package / "__init__.py").is_file():
        raise SystemExit(f"installed torch_fl package has no __init__.py: {package}")
    if not test_site and not any(
        (package / f"_C{candidate}").is_file()
        for candidate in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise SystemExit(f"installed torch_fl package has no native _C: {package}")
    return package.resolve()


def _is_path_digest_entry(name):
    """Whether this is an interpreter path-cache variable, not a setting."""
    key, separator, digest = name.partition(":")
    return (
        separator == ":"
        and key == "PYTHONPATH"
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def clean_summary(output):
    """Return pytest's summary line when it shows a completed, clean run.

    ``None`` means no such line exists, so the file never reached the end of its
    test session and its result is unknown.
    """
    lines = output.splitlines()
    summary_index = next(
        (
            index
            for index in range(len(lines) - 1, -1, -1)
            if _SUMMARY_RE.match(lines[index])
        ),
        None,
    )
    if summary_index is None:
        return None
    summary = lines[summary_index].strip("= ").strip()
    # The opt-in covers a driver destructor after a complete pytest session. It
    # does not cover a test that printed a fake summary before dying, or a
    # teardown error that pytest reported after the test counts.
    if _SUMMARY_FAILURE_RE.search(summary) or any(
        _SUMMARY_FAILURE_RE.search(line) for line in lines[summary_index + 1 :]
    ):
        return None
    return summary


def classify(returncode, output):
    """Map one pytest run onto (status, tolerated) for a single test file."""
    if returncode == 0:
        return "PASS", False
    if returncode == 124:
        # ``timeout`` uses GNU coreutils' convention; the runner kills the
        # process group on its own timeout and reports that separately.
        return "TIMEOUT", False
    if returncode < 0:
        fatal = -returncode in _FATAL_SIGNALS
        if fatal and _ALLOW_TEARDOWN_CRASH and clean_summary(output):
            return "PASS (crash after summary)", True
        return ("CRASH" if fatal else f"SIGNAL {-returncode}"), False
    if returncode > 128 and (returncode - 128) in _FATAL_SIGNALS:
        if _ALLOW_TEARDOWN_CRASH and clean_summary(output):
            return "PASS (crash after summary)", True
        return "CRASH", False
    return "FAIL", False


def _start_vendor_runtime(test_file, wheel_package_root=""):
    """Let torch_fl select its vendor core before a test imports torch.

    PPU, DCU and MetaX wheels use a private torch facade whose location is
    selected while torch_fl starts. Importing torch first, as several shared
    unit files do at collection time, is already too late to choose that core.
    Start torch_fl in this runner so its selection and PYTHONPATH are inherited
    by every child.
    """
    del test_file
    env = child_environment(wheel_package_root)
    if not wheel_package_root:
        previous_path = list(sys.path)
        selected = [
            entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry
        ]
        sys.path[:] = selected + [entry for entry in sys.path if entry not in selected]
        try:
            importlib.invalidate_caches()
            import torch_fl  # noqa: F401  -- selects the vendor libtorch
        except ImportError:
            # A source-only build without the extension still supports tests
            # that do not need it. An installed package's import failure is
            # fatal because the remaining files share that broken environment.
            return env
        finally:
            sys.path[:] = previous_path
        return child_environment("")
    return _start_selected_package(Path(wheel_package_root).resolve(), env)


def _start_selected_package(wheel_root, env):
    """Import the wheel's torch_fl and return the path it selected.

    pytest's source override can make ``torch_fl`` a second package object, so
    the selected package's startup is what must run here. Testing that package
    object lets one test simulate startup without loading vendor libraries.
    """
    package_root = wheel_root / "torch_fl"
    previous_path = list(sys.path)
    selected = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    sys.path[:] = selected + [entry for entry in sys.path if entry not in selected]
    try:
        importlib.invalidate_caches()
        found = importlib.util.find_spec("torch_fl")
        source = getattr(found, "origin", None) if found else None
        if source is not None and Path(source).resolve().is_relative_to(package_root):
            importlib.import_module("torch_fl")
        else:
            spec = importlib.util.spec_from_file_location(
                "torch_fl",
                package_root / "__init__.py",
                submodule_search_locations=[str(package_root)],
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load torch_fl from {package_root}")
            package = importlib.util.module_from_spec(spec)
            previous = sys.modules.get("torch_fl")
            sys.modules["torch_fl"] = package
            try:
                spec.loader.exec_module(package)
            except BaseException:
                if previous is None:
                    sys.modules.pop("torch_fl", None)
                else:
                    sys.modules["torch_fl"] = previous
                raise
    finally:
        sys.path[:] = previous_path
    # Startup may install a private torch facade and prepend it to PYTHONPATH.
    # Rebuild from the process environment so the child sees that facade too.
    return child_environment(wheel_root)


def run_file(python, test_file, timeout, report_dir="", wheel_package_root=""):
    """Run one test file; return (returncode, output, timed_out)."""
    # The runner exports a shell wrapper for deliberate group commands; a
    # nested sample can donate its wrapper, so invoke the real executable. The
    # parent selects the installed package explicitly. pytest's pythonpath
    # setting may add the checkout, but the installed package still owns
    # torch_fl when the conftest and its subprocesses import it.
    python = os.environ.get("FLAGCICD_PYTHON_NO_LAUNCHER", python)
    env = _start_vendor_runtime(test_file, wheel_package_root)
    startup = Path(__file__).with_name("_torch_fl_test_site.py")
    command = [
        python,
        "-c",
        (
            "import importlib.util\n"
            f"spec = importlib.util.spec_from_file_location("
            f"'_torch_fl_test_site', {str(startup)!r})\n"
            "site = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(site)\n"
            "site.install()\n"
            "import pytest\n"
            "original = pytest.Config._configure_python_path\n"
            "def configure_python_path(config):\n"
            "    original(config)\n"
            "    site.install()\n"
            "pytest.Config._configure_python_path = configure_python_path\n"
            "raise SystemExit(pytest.main())"
        ),
        str(test_file.resolve()),
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    # Reasons carry the test path and message needed to fix a shared case that
    # is skipped on one platform. The path stays absolute: a unit file can live
    # outside the checkout, and the JSON report lives in the CI temp directory.
    if report_dir:
        command.extend(
            [
                "-rs",
                "--json-report",
                "--json-report-file",
                str(Path(report_dir) / f"unit-{uuid.uuid4().hex}.json"),
            ]
        )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        # Own process group so a runaway test cannot outlive the timeout.
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
        return process.returncode, output, False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return 124, output, True


def _report_belongs_to_file(document, test_file):
    """Match a unit report to the exactly one Python file it tested.

    pytest-json-report omits ``args`` when the JSON path is absolute, so the
    complete test-node IDs are the portable record of the invoked file.
    """
    tests = document.get("tests")
    if not isinstance(tests, list) or not tests:
        return False
    requested = Path(test_file).resolve()
    return all(
        isinstance(test.get("nodeid"), str)
        and test["nodeid"].split("::", 1)[0].endswith("/" + requested.name)
        for test in tests
    )


def _report_shows_a_completed_pass(document):
    """Check the child session's own verdict before trusting the runner's.

    The runner also marks an opted-in teardown crash as a PASS, so its status
    alone cannot tell a clean session from one whose report records failures.
    """
    summary = document.get("summary")
    if not isinstance(summary, dict):
        return False
    tests = document.get("tests")
    if not isinstance(tests, list):
        return False
    if not tests:
        return document.get("summary", {}).get("error", 0) == 0
    return all(
        test.get("outcome") in {"passed", "xfailed"}
        and all(
            isinstance(test.get(phase), dict)
            and test[phase].get("outcome") in {"passed", "skipped"}
            for phase in ("setup", "call", "teardown")
        )
        for test in tests
    )


def _repository_relative_file(test_file):
    """Return a test file's inventory name, or an empty string for a probe."""
    root = Path(os.environ.get("GITHUB_WORKSPACE", ""))
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2]
    try:
        return test_file.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return ""


def _normalize_child_reports(report_dir, statuses):
    """Finish the report of a unit file that the runner already classified.

    pytest-json-report can write its complete file during interpreter
    teardown. A Python destructor may then crash or time out before the native
    plugin writes the marker that tells the merger the file is complete. The
    runner has already classified the file from pytest's own summary; with a
    passing report, that file becomes the inventory's evidence for its case.
    An incomplete or failing report remains incomplete and fails the inventory.

    Resolved paths and relative source paths refer to the same file. The
    inventory names the source path, so a report in a temporary checkout still
    answers for the case the plan declares.
    """
    directory = Path(report_dir)
    known = directory / "google-test.json"
    try:
        inventory = json.loads(known.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        inventory = {"cases": [], "aliases": {}}
    if (
        not isinstance(inventory, dict)
        or not isinstance(inventory.get("aliases"), dict)
        or inventory.get("group", "unit_tests_per_file") != "unit_tests_per_file"
    ):
        raise SystemExit(f"unit inventory is malformed: {known}")
    added = False
    for report in directory.glob("unit-*.json"):
        if report.with_suffix(".pending").exists():
            continue
        try:
            document = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not _report_shows_a_completed_pass(document):
            continue
        matches = [
            (path, status)
            for path, status, _returncode, _tolerated in statuses
            if status.startswith("PASS") and _report_belongs_to_file(document, path)
        ]
        if len(matches) != 1:
            continue
        path, status = matches[0]
        if not _report_shows_a_completed_pass(document):
            continue
        case = _repository_relative_file(path)
        if not case:
            continue
        if case not in inventory["cases"]:
            continue
        inventory["aliases"].setdefault(case, [case])
        added = True
        report.with_suffix(".pending").write_text(
            json.dumps(
                {
                    "group": os.environ.get("FLAGCICD_TEST_GROUP", "") or str(path),
                    "args": document.get("args", []),
                    "unit_file": str(path),
                    "unit_status": status,
                }
            ),
            encoding="utf-8",
        )
    if added:
        known.write_text(json.dumps(inventory) + "\n", encoding="utf-8")


def _record_one_file_per_report(report_dir, results):
    """Attach each per-file report to the one source file it ran.

    The integration runner supplies the shared inventory. Its report directory
    can also contain the group pytest session that launched this script; that
    session is not a unit report and must not become one of its files.
    """
    directory = Path(report_dir)
    if not (directory / "google-test.json").is_file():
        return
    try:
        inventory = json.loads(
            (directory / "google-test.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return
    group = inventory.get("group", "") if isinstance(inventory, dict) else ""
    group = os.environ.get("FLAGCICD_TEST_GROUP", "") or group
    passing = {
        Path(result[0]).resolve(): result
        for result in results
        if result[1].startswith("PASS")
    }
    for report in directory.glob("unit-*.json"):
        report.with_suffix(".pending").unlink(missing_ok=True)
        try:
            document = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        matches = [
            result
            for result in passing.values()
            if _report_belongs_to_file(document, result[0])
        ]
        if len(matches) != 1:
            continue
        path, status, returncode, tolerated = matches[0]
        report.with_suffix(".unit").write_text(
            json.dumps(
                {
                    "group": group or str(path),
                    "args": [str(path)],
                    "unit_file": str(path),
                    "unit_status": status,
                    "unit_returncode": returncode,
                    "tolerated_teardown_crash": tolerated,
                }
            ),
            encoding="utf-8",
        )
    complete = set()
    for report in directory.glob("unit-*.unit"):
        try:
            marker = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(marker, dict) and marker.get("unit_file"):
            complete.add(Path(marker["unit_file"]).resolve())
    for path, (_path, _status, _returncode, tolerated) in passing.items():
        case = _repository_relative_file(path)
        if (
            not case
            or path in complete
            or not path.is_file()
            or case not in inventory["cases"]
            or _completed_report(directory, path)
        ):
            continue
        pending = directory / f"unit-{uuid.uuid4().hex}.pending"
        pending.write_text(
            json.dumps(
                {
                    "group": group or str(path),
                    "args": [str(path)],
                    "unit_file": str(path),
                    "error": "the individual pytest run wrote no report",
                }
            ),
            encoding="utf-8",
        )


def _completed_report(directory, test_file):
    """Whether one child run left a complete report for a tolerated crash."""
    for report in directory.glob("unit-*.json"):
        try:
            document = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if _report_belongs_to_file(document, test_file):
            return _report_shows_a_completed_pass(document)
    return False


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        raise SystemExit(f"test directory does not exist: {test_dir}")
    wheel_package_root = args.wheel_package_root
    if not wheel_package_root:
        original = next(
            (
                str(Path(entry) / "torch_fl")
                for entry in sys.path
                if (Path(entry) / "torch_fl/__init__.py").is_file()
                and any(
                    (Path(entry) / "torch_fl" / f"_C{suffix}").is_file()
                    for suffix in importlib.machinery.EXTENSION_SUFFIXES
                )
            ),
            "",
        )
        if original:
            wheel_package_root = str(Path(original).parent)
    # Check once before starting files, rather than once in every child.
    child_environment(wheel_package_root)

    files = sorted(test_dir.rglob(args.pattern))
    files = [path for path in files if path.is_file()]
    if not files:
        raise SystemExit(f"no test files matched {test_dir}/{args.pattern}")
    report_dir = os.environ.get("FLAGCICD_REPORT_DIR", "")
    if report_dir:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        os.environ["FLAGCICD_TEST_GROUP"] = "unit_tests_per_file"
    results = []
    for test_file in files:
        print(f"FILE {test_file}", flush=True)
        print(f"::group::{test_file}", flush=True)
        returncode, output, timed_out = run_file(
            args.python,
            test_file,
            args.timeout,
            report_dir,
            wheel_package_root=wheel_package_root,
        )
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        status, tolerated = classify(returncode, output)
        if timed_out:
            print(f"::error::{test_file} did not finish within {args.timeout}s")
        elif tolerated:
            print(
                f"::warning::{test_file} passed, then the interpreter died with "
                f"signal {-returncode} during teardown (see this script's docstring)"
            )
        elif status != "PASS":
            print(f"::error::{test_file} failed ({status}, exit {returncode})")
        print("::endgroup::", flush=True)
        results.append((test_file, status, returncode, tolerated))
        # An opted-in teardown crash carries a complete report but no marker;
        # finish that report here as well, after the interpreter has exited.
        if report_dir:
            _normalize_child_reports(report_dir, [results[-1]])
            _record_one_file_per_report(report_dir, [results[-1]])

    if report_dir:
        _normalize_child_reports(report_dir, results)
        _record_one_file_per_report(report_dir, results)
    # A tolerated teardown crash counted as a pass above, so it must not count as
    # a failure here either -- otherwise `UNIT_TESTS_ALLOW_TEARDOWN_CRASH=1`
    # reports every affected file as a warning and then still exits 1.
    failed = [r for r in results if r[1] != "PASS" and not r[3]]
    width = max(len(str(r[0])) for r in results)
    print("===== unit test summary =====")
    for test_file, status, returncode, _ in results:
        print(f"{status:<28} exit={returncode:<5} {str(test_file):<{width}}")
    print(f"{len(results) - len(failed)}/{len(results)} test files passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
