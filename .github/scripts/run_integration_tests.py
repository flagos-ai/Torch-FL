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
import subprocess


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
    for index, test in enumerate(tests, start=1):
        print(
            f"===== [{index}/{len(tests)}] {test['name']} =====",
            flush=True,
        )
        test_environment = command_environment.copy()
        for key, value in test.get("environment", {}).items():
            test_environment[key] = os.path.expandvars(value)
        result = subprocess.run(
            ["bash", "-c", f"set -euo pipefail\n{test['command']}"],
            check=False,
            env=test_environment,
            cwd=workdir,
        )
        if result.returncode:
            print(
                f"Integration test '{test['name']}' failed with exit code "
                f"{result.returncode}"
            )
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
