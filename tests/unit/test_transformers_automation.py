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
"""Pure tests for the Transformers triage and verification tools."""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = REPO_ROOT / "scripts" / "transformers" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


triage = load("transformers_triage")
verify = load("transformers_verify")
file_issues = load("transformers_file_issues")


def test_cpu_fallback_becomes_confirmed_operator_finding():
    result = triage.triage_failures(
        {
            "environment": {"device": "flagos"},
            "run": {"context_poison": False},
            "tests": [
                {
                    "nodeid": "tests/models/qwen3/test_modeling_qwen3.py::test_ok",
                    "status": "PASS",
                    "cpu_fallback_ops": ["aten::div"],
                }
            ],
        }
    )
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["class"] == "OP_CPU_FALLBACK"
    assert finding["component"] == "flagos"
    assert finding["subject"] == "aten::div"
    assert finding["verdict"] == "CONFIRMED"
    assert not finding["verification_required"]


def test_fingerprint_separates_backend_components():
    first = triage.compute_fingerprint("OP_UNSUPPORTED", "musa", "aten::div", "x")
    second = triage.compute_fingerprint("OP_UNSUPPORTED", "ascend", "aten::div", "x")
    assert first != second


def test_run_level_poison_does_not_classify_every_failure_as_crash():
    result = triage.triage_failures(
        {
            "environment": {"device": "flagos"},
            "run": {"context_poison": True},
            "tests": [
                {
                    "nodeid": "tests/models/qwen3/test.py::test_distributed",
                    "status": "FAIL",
                    "detail": "RuntimeError: unsupported device type flagos",
                }
            ],
        }
    )
    finding = result["findings"][0]
    assert finding["class"] == "FEATURE_UNSUPPORTED"
    assert finding["subject"] == "device_flagos"


def test_issue_body_metadata_accepts_platform_and_legacy_chip_keys():
    body = "- **Platform**: MUSA MTT S5000\n- **Transformers**: 5.16.1\n"
    assert file_issues.extract_body_metadata(body, "Platform") == "MUSA MTT S5000"
    assert file_issues.extract_body_metadata(body, "Transformers") == "5.16.1"

    legacy = "- **Chip**: GCU S60\n"
    assert file_issues.extract_body_metadata(legacy, "Chip") == "GCU S60"


def test_verification_error_is_inconclusive():
    assert verify.determine_verdict("ERROR", "CRASH") == "INCONCLUSIVE"
    assert verify.determine_verdict("FAIL", "CRASH") == "CONFIRMED"
    assert verify.determine_verdict("PASS", "CRASH") == "COLLATERAL"


def test_resolve_test_source_accepts_exact_tree(tmp_path):
    (tmp_path / "tests" / "models").mkdir(parents=True)
    assert verify.resolve_test_source(tmp_path, None) == tmp_path


def test_resolve_test_source_uses_numeric_version_order(tmp_path):
    old = tmp_path / "transformers-5.9.0"
    new = tmp_path / "transformers-5.16.1"
    old.mkdir()
    new.mkdir()
    assert verify.resolve_test_source(tmp_path, None) == new
