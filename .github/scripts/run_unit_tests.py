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
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

# pytest prints its counts on the last line that mentions a duration, e.g.
# "24 passed in 19.28s" (with a banner when not run under -q). Matching on the
# duration is what distinguishes the summary from the per-test progress output.
_SUMMARY_RE = re.compile(r"\bin \d+(?:\.\d+)?s\b")
_SUMMARY_FAILURE_RE = re.compile(r"\b(failed|error|errors|no tests ran)\b")

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
    return parser.parse_args()


def clean_summary(output):
    """Return pytest's summary line when it shows a completed, clean run.

    ``None`` means no such line exists, so the file never reached the end of its
    test session and its result is unknown.
    """
    for line in reversed(output.splitlines()):
        if _SUMMARY_RE.search(line):
            if _SUMMARY_FAILURE_RE.search(line):
                return None
            return line.strip(" =")
    return None


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


def run_file(python, test_file, timeout):
    """Run one test file; return (returncode, output, timed_out)."""
    command = [python, "-m", "pytest", str(test_file), "-q", "-p", "no:cacheprovider"]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        raise SystemExit(f"test directory does not exist: {test_dir}")

    files = sorted(test_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"no test files matched {test_dir}/{args.pattern}")

    results = []
    for test_file in files:
        print(f"::group::{test_file}", flush=True)
        returncode, output, timed_out = run_file(args.python, test_file, args.timeout)
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
