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

"""Run one container command and publish its environment to the host runner."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


class EnvironmentHookError(Exception):
    """A container command cannot keep the runner and image in sync."""


def _files(settings):
    """Name the container copy and the fresh command output for each hook."""
    state = Path(settings["state"])
    return {
        name: {
            "command": state / f"{suffix}.command",
            "live": Path(settings["environment"] if name == "GITHUB_ENV" else path),
            "host": Path(settings["host_files"][name]),
        }
        for name, suffix, path in (
            ("GITHUB_ENV", "env", settings["environment"]),
            ("GITHUB_PATH", "path", settings["path"]),
        )
    }


def _additions(command, inherited):
    """Return the exact bytes the command appended after inheriting its file."""
    value = command.read_bytes() if command.is_file() else b""
    if not value:
        return b""
    if not value.startswith(inherited):
        raise EnvironmentHookError(f"the command replaced {command.name}")
    return value[len(inherited) :]


def _append(settings, name, additions):
    """Append a hook's entries to the host and the corresponding container file."""
    if not additions:
        return
    files = _files(settings)
    destinations = (Path(settings["host_files"][name]), files[name]["live"])
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        inherited = destination.read_bytes() if destination.is_file() else b""
        if inherited and not inherited.endswith(b"\n"):
            inherited += b"\n"
        destination.write_bytes(inherited + additions)


def _set_environment(settings, environment, additions, host):
    """Replace inline values and append the command's own environment additions."""
    current = Path(environment).read_text(encoding="utf-8").splitlines()
    inline = set(settings.get("inline_environment", []))
    if not inline:
        return
    names = {line.split("=", 1)[0] for line in current if "=" in line}
    replaced = [
        line for line in current if "=" in line and line.split("=", 1)[0] not in inline
    ]
    for name in settings["inline_environment"]:
        if name in names:
            replaced.append(f"{name}={shlex.quote(additions[name])}")
    Path(environment).write_text("".join(f"{line}\n" for line in replaced))
    host.write_text(Path(environment).read_text())


def publish_inline(settings, values, command):
    """Publish the paths that only the current host step can resolve correctly."""
    missing = set(settings["inline_environment"]) - set(values)
    if missing:
        raise EnvironmentHookError(
            "inline environment values are missing: " + ", ".join(sorted(missing))
        )
    _set_environment(
        settings,
        settings["environment"],
        values,
        Path(settings["host_files"]["GITHUB_ENV"]),
    )


def _run_in_container(settings, command_file):
    """Run a hook from the inherited files and preserve its exit status."""
    files = _files(settings)
    wrapper = Path(settings["runtime"]) / "container-command.sh"
    wrapper.write_text(
        'cd "$GITHUB_WORKSPACE"\n'
        'source "$INTEGRATION_PROFILE"\n'
        + f"bash {shlex.quote(str(command_file))}\n"
        + "status=$?\n"
        + 'if [[ -s "$GITHUB_ENV" ]]; then\n'
        + f"  copy={shlex.quote(str(files['GITHUB_ENV']['command']))}.container-$$\n"
        + '  cp "$GITHUB_ENV" "$copy"\n'
        + f'  mv -f "$copy" {shlex.quote(str(files["GITHUB_ENV"]["command"]))}\n'
        + "fi\n"
        + 'if [[ -s "$GITHUB_PATH" ]]; then\n'
        + f"  copy={shlex.quote(str(files['GITHUB_PATH']['command']))}.container-$$\n"
        + '  cp "$GITHUB_PATH" "$copy"\n'
        + f'  mv -f "$copy" {shlex.quote(str(files["GITHUB_PATH"]["command"]))}\n'
        + "fi\n"
        + 'exit "$status"\n'
    )
    try:
        return subprocess.run(
            [
                "docker",
                "exec",
                settings["container"],
                "--env",
                "HOME=/tmp",
                "bash",
                settings["shell"],
                "-c",
                f"bash {shlex.quote(str(wrapper))}",
            ],
            timeout=3600,
            check=False,
        ).returncode
    except subprocess.TimeoutExpired:
        print(
            "::error::the container hook exceeded its one-hour limit", file=sys.stderr
        )
        return 124
    except OSError as error:
        print(f"::error::cannot run the container hook: {error}", file=sys.stderr)
        return 1


def hook(settings, command):
    """Execute a script in the configured container and copy its files back."""
    if not command:
        raise EnvironmentHookError("--command is required")
    if not settings.get("image") or not settings.get("container"):
        raise EnvironmentHookError("the hook needs a running vendor container")
    if Path(command).name != command or not command.endswith(".sh"):
        raise EnvironmentHookError("--command must name a shell script")
    command_file = Path(settings["workspace"]) / command
    # Scripts are shared with the container through the mounted workspace.
    # Keep their edit after the step so a later host step cannot execute a
    # different copy.
    if not command_file.is_file():
        raise EnvironmentHookError(f"the hook script is missing: {command_file}")
    files = _files(settings)
    inherited = {}
    for name, values in files.items():
        inherited[name] = (
            values["live"].read_bytes() if values["live"].is_file() else b""
        )
        values["command"].write_bytes(inherited[name])
    status = _run_in_container(settings, command_file)
    try:
        additions = {
            name: _additions(values["command"], inherited[name])
            for name, values in files.items()
        }
        for name, value in additions.items():
            _append(settings, name, value)
    except (OSError, subprocess.SubprocessError, EnvironmentHookError) as error:
        print(
            f"::error::cannot publish the container environment: {error}",
            file=sys.stderr,
        )
        if not status:
            status = 1
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("publish-inline", "hook"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--inline", default="{}")
    arguments = parser.parse_args(argv)
    try:
        settings = json.loads(Path(arguments.config).read_text(encoding="utf-8"))
        if arguments.command == "publish-inline":
            publish_inline(settings, json.loads(arguments.inline), arguments.command)
            return 0
        return hook(settings, arguments.command)
    except (KeyError, OSError, TypeError, ValueError, EnvironmentHookError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
