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

"""Verify that superseded CI runs do not send completion notifications."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
NOTIFIER = yaml.safe_load((ROOT / ".github/workflows/notify-feishu.yml").read_text())


def test_notifier_ignores_intentionally_cancelled_runs():
    condition = NOTIFIER["jobs"]["notify"]["if"]
    assert "skipped" in condition
    assert "cancelled" in condition
    assert "github.event.workflow_run.conclusion" in condition
