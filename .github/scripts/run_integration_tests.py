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
import json
import os
import re
import signal
import threading
import time
import shlex
import subprocess
import sys
from pathlib import Path


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
_SUMMARY_RE = re.compile(r"\bin \d+(?:\.\d+)?s\b")
_SUMMARY_FAILURE_RE = re.compile(r"\b(?:failed|errors?|xfailed|xpassed|no tests ran)\b")

# Signals that mean the process died rather than exited: SIGSEGV from a bad
# address in a teardown destructor is the observed case; the rest are the same
# class.
_FATAL_SIGNALS = {signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS, signal.SIGILL}

_TEARDOWN_CRASH_ENV = "INTEGRATION_ALLOW_TEARDOWN_CRASH"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
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
        validate_environment(
            test_environment,
            f"integration_tests[{index}].environment",
        )

    validate_environment(environment, "integration_environment")

    return tests, environment


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


def descendant_pids(root_pid):
    """Return descendants using procfs, with a portable ps fallback."""
    children = {}
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for task in proc_root.glob("*/task/*/children"):
            try:
                parent = int(task.parts[2])
                values = task.read_text().split()
                children.setdefault(parent, []).extend(int(value) for value in values)
            except (OSError, ValueError):
                continue
    else:
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,ppid="],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            result = None
        if result and result.returncode == 0:
            for line in result.stdout.splitlines():
                fields = line.split()
                if len(fields) == 2:
                    try:
                        children.setdefault(int(fields[1]), []).append(int(fields[0]))
                    except ValueError:
                        continue
    found = []
    pending = list(children.get(root_pid, ()))
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.append(pid)
        pending.extend(children.get(pid, ()))
    return found


def run_command(command, test_environment, workdir):
    """Stream one command; timeout and cancellation also kill its descendants."""
    required_mpirun(command, test_environment)
    timeout = int(test_environment.get("FLAGCICD_CASE_TIMEOUT", "1800"))
    if not 1 <= timeout <= 43200:
        raise ValueError("FLAGCICD_CASE_TIMEOUT must be between 1 and 43200 seconds")
    process = subprocess.Popen(
        ["bash", "-c", f"set -euo pipefail\n{command}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=test_environment,
        cwd=workdir,
        start_new_session=True,
    )
    timed_out = threading.Event()

    def kill_group():
        descendants = set(descendant_pids(process.pid))
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        # Vendor launchers may escape the process group with setsid().
        descendants.update(descendant_pids(process.pid))
        for pid in descendants:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if process.stdout and not process.stdout.closed:
            process.stdout.close()

    def expire():
        timed_out.set()
        kill_group()

    timer = threading.Timer(timeout, expire)
    timer.daemon = True
    timer.start()
    assert process.stdout is not None
    try:
        lines = []
        for line in process.stdout:
            sys.stdout.write(line)
            lines.append(line)
        sys.stdout.flush()
        returncode = process.wait()
    except (OSError, ValueError):
        if not timed_out.is_set():
            kill_group()
            process.wait()
            raise
        returncode = process.wait()
    except BaseException:
        kill_group()
        process.wait()
        raise
    finally:
        timer.cancel()
        process.stdout.close()
    return (124 if timed_out.is_set() else returncode), "".join(lines)


def main():
    args = parse_args()
    tests, environment = load_configuration(args.allow_empty)
    if args.validate_only:
        print(f"Validated {len(tests)} configured integration test(s)")
        return 0

    command_environment = os.environ.copy()
    command_environment.update(environment)
    workdir = os.environ.get("INTEGRATION_WORKDIR")
    if workdir and not os.path.isdir(workdir):
        raise SystemExit(f"INTEGRATION_WORKDIR does not exist: {workdir}")
    records = [
        {
            "name": test["name"],
            "command": test["command"],
            "status": "not_run",
            "requires_pytest": any(
                Path(word).name in ("pytest", "run_unit_tests.py")
                for word in shlex.split(test["command"])
            ),
            "min_passed": test.get("min_passed", 0),
        }
        for test in tests
    ]
    report_dir = command_environment.get("FLAGCICD_REPORT_DIR")

    def save_execution():
        if report_dir:
            Path(report_dir).mkdir(parents=True, exist_ok=True)
            Path(report_dir, "groups.json").write_text(
                json.dumps(
                    {"expected": [r["name"] for r in records], "groups": records}
                )
            )

    save_execution()
    for index, test in enumerate(tests, start=1):
        print(
            f"===== [{index}/{len(tests)}] {test['name']} =====",
            flush=True,
        )
        test_environment = command_environment.copy()
        for key, value in test.get("environment", {}).items():
            test_environment[key] = os.path.expandvars(value)
        if test_environment.get("FLAGCICD_REPORT_DIR"):
            test_environment["FLAGCICD_TEST_GROUP"] = test["name"]
        record = records[index - 1]
        record.update(status="running", started_at=time.time())
        save_execution()
        allow_crash = teardown_crash_allowed(test_environment)
        try:
            returncode, output = run_command(test["command"], test_environment, workdir)
        except BaseException as error:
            record.update(
                status="cancelled"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                error=str(error),
                finished_at=time.time(),
            )
            save_execution()
            raise
        ok, tolerated = classify_command_result(returncode, output, allow_crash)
        if ok and test.get("min_passed", 0):
            passed = sum(int(n) for n in re.findall(r"\b(\d+) passed\b", output))
            ok = passed >= test["min_passed"]
        record.update(
            status="passed" if ok else "failed",
            returncode=returncode,
            finished_at=time.time(),
        )
        save_execution()
        if tolerated and ok:
            print(
                f"Integration test '{test['name']}' passed "
                f"({clean_summary(output)}), then the command died with "
                f"signal {signal_number(returncode)} during teardown "
                f"({_TEARDOWN_CRASH_ENV} is set for this group; see this "
                f"script's header comment)",
                flush=True,
            )
            continue
        if not ok:
            print(
                f"Integration test '{test['name']}' failed with exit code {returncode}",
                flush=True,
            )
            raise SystemExit(returncode or 1)
    return 0


if __name__ == "__main__":

    def cancel(signum, frame):
        raise KeyboardInterrupt("integration runner cancelled")

    signal.signal(signal.SIGTERM, cancel)
    raise SystemExit(main())
