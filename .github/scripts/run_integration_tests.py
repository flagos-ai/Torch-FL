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
import codecs
import ctypes
import json
import os
import re
import resource
import signal
import selectors
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
    parser.add_argument("--supervise-command", help=argparse.SUPPRESS)
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


_SHELL_PARAMETER = r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))"


def _unset_variables(command, test_environment):
    """Return the shell parameters ``command`` expands that nothing supplies.

    The command runs under ``set -u``. A parameter no layer defines then ends
    the group with bash's own "unbound variable", which reports the *group
    name* and nothing about the variable or the configuration that owes it.
    ``MODEL_PATH`` is the standing example: the manifest lists the Qwen3 groups
    for a platform whose ``integration_environment`` has no model mount.

    ``$name`` and ``${name}`` are the forms a manifest command uses; shell
    lookup variables (``$?``, ``$@``, ``$$``) and commands that spell a
    parameter as ``${name:-default}`` are left alone.
    """
    missing = []
    for match in re.finditer(_SHELL_PARAMETER, command):
        name = match.group(1) or match.group(2)
        if name not in test_environment and name not in missing:
            missing.append(name)
    return missing


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


def supervise_command(command):
    """Keep escaped Linux workers parented to a dedicated command supervisor.

    A launcher can exit before its setsid worker. A Linux child subreaper keeps
    that worker in this command's tree instead of losing it to PID 1. The
    supervisor is a separate process: unrelated children of the caller must
    never be adopted or killed. Other POSIX hosts retain process-group cleanup.
    """
    if sys.platform != "linux":
        os.execvp("bash", ["bash", "-c", f"set -euo pipefail\n{command}"])
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    prctl.restype = ctypes.c_int
    if prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    process = subprocess.Popen(["bash", "-c", f"set -euo pipefail\n{command}"])
    code = process.wait()
    while True:
        try:
            os.waitpid(-1, 0)
        except InterruptedError:
            continue
        except ChildProcessError:
            break
    if code < 0:
        # Preserve Popen's negative signal result without creating a second
        # core dump for the supervisor. The command kept its original limit.
        resource.setrlimit(
            resource.RLIMIT_CORE, (0, resource.getrlimit(resource.RLIMIT_CORE)[1])
        )
        signum = -code
        if signum not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    return code


def run_command(command, test_environment, workdir):
    """Stream one command with a bounded reader and command-owned cleanup."""
    required_mpirun(command, test_environment)
    timeout = int(test_environment.get("FLAGCICD_CASE_TIMEOUT", "1800"))
    if not 1 <= timeout <= 43200:
        raise ValueError("FLAGCICD_CASE_TIMEOUT must be between 1 and 43200 seconds")
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            f"--supervise-command={command}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=test_environment,
        cwd=workdir,
        start_new_session=True,
    )

    def kill_group():
        descendants = descendant_pids(process.pid)
        for pid in reversed(descendants):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if sys.platform == "linux" and descendants:
            # Let the dedicated supervisor reap workers before terminating it;
            # otherwise a container's PID 1 may retain them as zombies.
            try:
                process.wait(timeout=0.5)
                return
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    chunks = []
    timed_out = False
    deadline = time.monotonic() + timeout

    def emit(data, final=False):
        text = decoder.decode(data, final=final)
        if text:
            chunks.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()

    try:
        os.set_blocking(process.stdout.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            if not selector.select(timeout=remaining):
                timed_out = True
                break
            data = os.read(process.stdout.fileno(), 65536)
            if not data:
                break
            emit(data)
        if not timed_out:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
        if timed_out:
            kill_group()
        returncode = process.wait()
        # Drain only currently available bytes; an unclosed pipe must never
        # turn the deadline into another blocking read.
        while selector.select(timeout=0):
            data = os.read(process.stdout.fileno(), 65536)
            if not data:
                break
            emit(data)
        emit(b"", final=True)
    except BaseException:
        kill_group()
        process.wait()
        raise
    finally:
        selector.close()
        process.stdout.close()
    return (124 if timed_out else returncode), "".join(chunks)


def main():
    args = parse_args()
    if getattr(args, "supervise_command", None) is not None:
        return supervise_command(args.supervise_command)
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
            "group_id": test.get("group_id", test.get("manifest_id", test["name"])),
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
            test_environment["FLAGCICD_GROUP_ID"] = records[index - 1]["group_id"]
        record = records[index - 1]
        record.update(status="running", started_at=time.time())
        save_execution()
        missing = _unset_variables(test["command"], test_environment)
        if missing:
            # The command runs under `set -u`, so a variable no layer supplies
            # aborts with "unbound variable": a bash message that never names
            # the group or the configuration that forgot it. thead is the live
            # example -- integration_environment is empty because its Qwen3
            # checkpoint mount went away, while the manifest still lists both
            # Qwen3 groups for it.
            output = (
                f"integration group '{test['name']}' was not run: "
                f"nothing supplies {', '.join(f'${name}' for name in missing)}, "
                f"which its command expands\n"
            )
            record.update(
                status="failed", error=output.strip(), finished_at=time.time()
            )
            save_execution()
            print(output, flush=True)
            raise SystemExit(1)
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
        evidence_error = ""
        if ok and test.get("min_passed", 0):
            from merge_pytest_reports import check_group_minimum

            try:
                check_group_minimum(report_dir, test)
            except (OSError, ValueError, TypeError, KeyError) as error:
                ok, evidence_error = False, str(error)
                record["error"] = evidence_error
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
                f"Integration test '{test['name']}' "
                + (
                    f"evidence check failed: {evidence_error}"
                    if evidence_error
                    else f"failed with exit code {returncode}"
                ),
                flush=True,
            )
            raise SystemExit(returncode or 1)
    return 0


if __name__ == "__main__":

    def cancel(signum, frame):
        raise KeyboardInterrupt("integration runner cancelled")

    signal.signal(signal.SIGTERM, cancel)
    raise SystemExit(main())
