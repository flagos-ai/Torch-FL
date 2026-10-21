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

"""Give each declared pytest invocation its own standard JSON report.

Some unit tests start their own pytest process. That probe is a sample, which
must not add a CI session. The integration runner declares each group
explicitly; a sample receives a different launcher marker. Pytest imports
plugins before its own runner can update ``PYTEST_CURRENT_TEST``, so that
variable cannot identify which kind of process this is.
"""

import json
import os
import shlex
import shutil
import sys
import uuid
from pathlib import Path

import pytest
from _pytest.config.exceptions import UsageError

from flagcicd_launch import GROUP, SAMPLE, current, strip_child_activation

_PLUGINS_ENV = "PYTEST_PLUGINS"
_LAUNCHER_ENV = "FLAGCICD_LAUNCHER"
_OUTER_ENV = "FLAGCICD_OUTER_PROCESS"
_PYTHON_ENV = "FLAGCICD_PYTHON_EXECUTABLE"
_ROOT_ENV = "FLAGCICD_PYTHON_ROOT"


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config, parser, args):
    _activate(early_config, args)


def _activate(early_config, args):
    """Capture activation before pytest overwrites the sample marker."""
    launch = current()
    directory = launch.report_dir
    plugins = _plugins_without("", "flagcicd_report")
    _set_plugins(plugins)
    os.environ.pop(_LAUNCHER_ENV, "")
    if not directory:
        return
    if launch.launcher == SAMPLE:
        return
    if launch.launcher == GROUP:
        if not launch.group:
            raise UsageError(
                f"{_LAUNCHER_ENV}=group requires a non-empty FLAGCICD_TEST_GROUP"
            )
        strip_child_activation()
        # The opted-in JSON options are available by this hook. Requesting
        # them on the command line makes the native plugin's registration
        # independent of hook ordering with other configuration plugins.
        # pytest-json-report resolves this path against the test's cwd.
        if "--json-report" not in args:
            args.append("--json-report")
        early_config.pluginmanager.register(
            _Suite(directory, launch.group),
            f"flagcicd_report_suite_{id(early_config)}",
        )
        return
    raise UsageError(
        f"FlagCICD reporting requires {_LAUNCHER_ENV}=group for this pytest session"
    )


def _set_plugins(names):
    if names:
        os.environ[_PLUGINS_ENV] = ",".join(names)
    else:
        os.environ.pop(_PLUGINS_ENV, None)


def _plugins_without(*removed):
    return [
        name
        for name in os.environ.get(_PLUGINS_ENV, "").split(",")
        if name.strip() not in ("", *removed)
    ]


class _Suite:
    def __init__(self, directory, group):
        self.directory = directory
        self.group = group
        self.report = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_configure(self, config):
        _configure_report(config, self.directory, self.group)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session):
        _verify_report(session)


def _configure_report(config, directory, group):
    if "json_report" not in config.option:
        raise UsageError("FlagCICD reporting requires pytest-json-report")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = (directory / f"pytest-{uuid.uuid4().hex}.json").resolve()
    # This marker is written before native report setup, so a startup failure
    # remains visible to the merger as an incomplete session.
    report.with_suffix(".pending").write_text(
        json.dumps({"group": group, "args": config.invocation_params.args}),
        encoding="utf-8",
    )
    # pytest-json-report owns this private attribute in the session that was
    # actually started. A nested session must not be mistaken for this one.
    setattr(config, "_flagcicd_report", report)
    config.option.json_report_file = str(report)
    config.option.json_report_relative = True


def _verify_report(session):
    """Reject a successful pytest exit that silently lacks its JSON report."""
    report = getattr(session.config, "_flagcicd_report", None)
    if report is None:
        return
    native = getattr(session.config, "_json_report", None)
    if native is None:
        session.exitstatus = pytest.ExitCode.INTERNAL_ERROR
        print("::error::FlagCICD reporting did not configure pytest-json-report")
    elif native.report is None or not report.is_file():
        session.exitstatus = pytest.ExitCode.INTERNAL_ERROR
        print(f"::error::pytest did not write its requested report: {report}")


def guard_command(environment):
    """Install the guarded Python entry point for one outer integration run."""
    name = environment.get(_PYTHON_ENV, "python")
    executable = shutil.which(name, path=environment.get("PATH"))
    if executable is None:
        raise ValueError("FlagCICD reporting requires Python on PATH")
    root_bin = str(Path(executable).resolve())
    return (
        f"export {_PYTHON_ENV}={shlex.quote(str(executable))}\n"
        f"export {_ROOT_ENV}={shlex.quote(root_bin)}\n"
        f"export {_OUTER_ENV}=$$\n"
        f"export {_LAUNCHER_ENV}=group\n"
        "python() {\n"
        f'  "${{{_PYTHON_ENV}}}" "$@"\n'
        "  local status=$?\n"
        f'  if [[ "$$" == "${{{_OUTER_ENV}}}" ]]; then\n'
        f'    export {_PYTHON_ENV}="{sys.executable}"\n'
        f"    unset {_OUTER_ENV}\n"
        "  fi\n"
        '  return "$status"\n'
        "}\n"
        "export -f python\n"
    )
