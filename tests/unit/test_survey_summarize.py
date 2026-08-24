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

"""Survey summarize() tests: verdicts and the strided-failure report."""

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "flaggems_overload_survey",
    _REPO / "tests" / "manual" / "flaggems_overload_survey.py",
)
survey = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(survey)


def _case(status, profile="2d-f32"):
    return {"profile": profile, "status": status}


def test_strided_failures_reported():
    results = {
        "a": {"cases": [_case("PASS"), _case("PASS", "2d-f32-strided")]},
        "b": {"cases": [_case("PASS"), _case("WRONG", "2d-f32-strided")]},
        "c": {"cases": [_case("PASS"), _case("ERROR", "2d-f32-strided")]},
    }
    summary = survey.summarize(["a", "b", "c"], results)
    assert summary["strided_failures"] == ["b", "c"]


def test_strided_failures_absent_when_none():
    results = {"a": {"cases": [_case("PASS"), _case("PASS", "2d-f32-strided")]}}
    summary = survey.summarize(["a"], results)
    assert summary["strided_failures"] == []


def test_verdicts_unchanged():
    results = {"a": {"cases": [_case("PASS")]}, "b": {"cases": [_case("WRONG")]}}
    summary = survey.summarize(["a", "b"], results)
    assert summary["verdicts"] == {"STRICT": 1, "FAILED": 1}
