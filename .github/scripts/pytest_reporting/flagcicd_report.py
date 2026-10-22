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

"""Give each top-level pytest invocation its own standard JSON report.

Loaded only by the hardware test step through PYTEST_PLUGINS. Removing this
plugin and its report directory from the pytest process's environment keeps
subprocess fixtures from reporting their intentionally failing sample suites.
The outer integration/per-file runner retains its environment, so each actual
suite still loads the plugin, including suites run from the wheel-only cwd.
"""

import json
import os
import uuid
from pathlib import Path

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    directory = os.environ.pop("FLAGCICD_REPORT_DIR", "")
    if not directory:
        return
    group = os.environ.pop("FLAGCICD_TEST_GROUP", "")
    plugins = os.environ.get("PYTEST_PLUGINS", "").split(",")
    plugins = [name for name in plugins if name.strip() not in ("", "flagcicd_report")]
    if plugins:
        os.environ["PYTEST_PLUGINS"] = ",".join(plugins)
    else:
        os.environ.pop("PYTEST_PLUGINS", None)

    if not hasattr(config.option, "json_report"):
        raise pytest.UsageError("FlagCICD reporting requires pytest-json-report")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / f"pytest-{uuid.uuid4().hex}.json"
    report.with_suffix(".pending").write_text(
        json.dumps({"group": group, "args": config.invocation_params.args}),
        encoding="utf-8",
    )
    # Run before pytest-json-report's configure hook, retaining its native schema.
    config.option.json_report = True
    config.option.json_report_file = str(report)
