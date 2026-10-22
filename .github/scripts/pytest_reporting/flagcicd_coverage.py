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

"""Collect each top-level pytest process before conftest/test-module imports."""

import json
import os
import sys
import uuid
from pathlib import Path

import pytest

from flagcicd_launch import GROUP, SAMPLE, current, strip_child_activation


class CoverageSession:
    def __init__(self, collector, marker):
        self.collector = collector
        self.marker = marker
        self.saved = False

    def save(self, exitstatus=None):
        if self.saved:
            return
        self.saved = True
        try:
            self.collector.stop()
            self.collector.save()
            if exitstatus is not None:
                self.marker.with_suffix(".complete").write_text(
                    json.dumps({"exitcode": int(exitstatus)}), encoding="utf-8"
                )
        except Exception as error:
            # Metrics are best effort; preserve pytest's original verdict.
            print(f"::warning::Cannot save Python coverage: {error}", file=sys.stderr)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        self.save(exitstatus)

    def pytest_unconfigure(self, config):
        self.save()


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config, parser, args):
    launch = current()
    directory = launch.coverage_dir
    if not directory:
        os.environ.pop("FLAGCICD_COVERAGE_DIR", None)
        return
    if launch.launcher == SAMPLE:
        return
    if launch.launcher != GROUP or not launch.group:
        raise pytest.UsageError(
            "Python coverage requires a group with FLAGCICD_TEST_GROUP"
        )
    strip_child_activation()
    _without_session_plugins()
    try:
        import coverage

        directory = Path(directory).resolve()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        marker = directory / f"coverage-{uuid.uuid4().hex}.pending"
        marker.write_text(
            json.dumps({"group": launch.group}),
            encoding="utf-8",
        )
        collector = coverage.Coverage(
            data_file=str(marker.with_suffix(".data")),
            config_file=False,
            source=[manifest["package_root"]],
            branch=False,
        )
        collector.start()
        early_config.pluginmanager.register(
            CoverageSession(collector, marker), "flagcicd_python_coverage_session"
        )
    except Exception as error:
        print(f"::warning::Cannot start Python coverage: {error}", file=sys.stderr)


def _without_session_plugins():
    """Keep a deliberate sample pytest process out of the parent session."""
    plugins = [
        name
        for name in os.environ.get("PYTEST_PLUGINS", "").split(",")
        if name.strip() not in ("", "flagcicd_report", "flagcicd_coverage")
    ]
    if plugins:
        os.environ["PYTEST_PLUGINS"] = ",".join(plugins)
    else:
        os.environ.pop("PYTEST_PLUGINS", None)
