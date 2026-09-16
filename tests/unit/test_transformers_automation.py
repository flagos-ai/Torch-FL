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
"""Regression tests for the transformers triage, dedup and filing tools.

The fixtures are built by running the runner's own ``reduce_records()`` over
synthetic pytest reports instead of hand-writing result JSON. That is the point
of this file: the two defects these tests guard against --- ``--all`` triaging
to zero findings and an all-``ENVIRONMENT_ERROR`` run summarized as clean ---
both shipped because the fixtures were hand-written and had drifted from what
the runner actually emits.
"""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DEVICE = "flagos"
ENVIRONMENT = {
    "device": DEVICE,
    "torch": "2.10.0",
    "transformers": "5.16.1",
    "torch_fl_commit": "abc1234",
}

UNSUPPORTED_DETAIL = (
    "Traceback (most recent call last):\n"
    '  File "/tmp/pytest-of-root/pytest-12/test_x0/test_modeling.py", line 88, in test_save\n'
    "    model.save_pretrained(tmp)\n"
    "RuntimeError: Could not run 'aten::index_copy_.out' with arguments from the "
    "'flagos' backend at 0x7f8a9b2c1d40 after 1.234s for shape [3, 4]"
)


def load(name, path=None):
    """Load a tool by path, as the manual tree and the scripts tree are not
    importable packages."""
    if path is None:
        path = REPO_ROOT / "scripts" / "transformers" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = load(
    "transformers_hf_source",
    REPO_ROOT / "tests" / "manual" / "transformers_hf_source.py",
)
runner = load(
    "transformers_hf_tests", REPO_ROOT / "tests" / "manual" / "transformers_hf_tests.py"
)
triage = load("transformers_triage")
verify = load("transformers_verify")
dedup = load("transformers_deduplicate")
preview = load("transformers_preview_issues")
file_issues = load("transformers_file_issues")


# --- fixtures built from the runner's own reducer -----------------------------


def raw_test(
    nodeid,
    outcome="passed",
    when="call",
    longrepr=None,
    duration=0.0,
    wasxfail=False,
    cpu_fallback_ops=None,
):
    """One report exactly as the in-child plugin writes it."""
    return {
        "kind": "test",
        "nodeid": nodeid,
        "when": when,
        "outcome": outcome,
        "duration": duration,
        "wasxfail": wasxfail,
        "longrepr": longrepr,
        "sections": [],
        "cpu_fallback_ops": cpu_fallback_ops or [],
    }


def raw_collect(nodeid, longrepr):
    return {
        "kind": "collect",
        "nodeid": nodeid,
        "outcome": "failed",
        "longrepr": longrepr,
    }


def measured_model(model, records, device=DEVICE):
    """One model's result, assembled the way ``prepare_result`` assembles it."""
    reduced = runner.reduce_records(records)
    result = {
        "schema_version": runner.SCHEMA_VERSION,
        "model": {"requested": model, "module": runner.module_name(model)},
        "environment": {**ENVIRONMENT, "device": device},
        "run": {
            "status": "RAN",
            "timed_out": False,
            "crashed": False,
            "context_poison": False,
            "collect_only": False,
        },
        "tests": reduced["tests"],
        "collect_errors": reduced["collect_errors"],
    }
    result["summary"] = runner.summarize_statuses(
        reduced["tests"], reduced["collect_errors"]
    )
    result["verdict"] = runner.verdict(result)
    for test in result["tests"]:
        if test["status"] in ("FAIL", "ERROR", "ENVIRONMENT_ERROR"):
            test["occurrence_fingerprint"] = runner.occurrence_fingerprint(
                model, device, test
            )
    return result


def all_aggregate(models):
    return runner.all_result(
        models, [m["model"]["requested"] for m in models], ENVIRONMENT
    )


# --- triage ------------------------------------------------------------------


def test_triage_reads_the_all_architecture_aggregate():
    """``--all`` keeps its models under ``models[]``; triage must still see them."""
    aggregate = all_aggregate(
        [
            measured_model(
                "bert",
                [
                    raw_test("tests/models/bert/test_modeling_bert.py::test_ok"),
                    raw_test(
                        "tests/models/bert/test_modeling_bert.py::test_save",
                        outcome="failed",
                        longrepr=UNSUPPORTED_DETAIL,
                    ),
                ],
            ),
            measured_model(
                "qwen3",
                [
                    raw_test(
                        "tests/models/qwen3/test_modeling_qwen3.py::test_masked",
                        outcome="failed",
                        longrepr=(
                            "AttributeError: 'Qwen3Model' object has no attribute "
                            "'flagos_kernel'"
                        ),
                    )
                ],
            ),
        ]
    )

    result = triage.triage_failures(aggregate)

    assert result["summary"]["total_failures"] == 2
    assert {finding["subject"] for finding in result["findings"]} == {
        "aten::index_copy_.out",
        "flagos_kernel",
    }
    assert all(finding["component"] == DEVICE for finding in result["findings"])
    assert result["summary"]["actionable"] == 2


def test_triage_merges_one_cause_across_models_and_separates_occurrences():
    """One defect is one finding, however many tests observed it."""
    second_trace = (
        UNSUPPORTED_DETAIL.replace("0x7f8a9b2c1d40", "0x55d3ab")
        .replace("1.234s", "9.87s")
        .replace("[3, 4]", "[17, 2]")
    )
    aggregate = all_aggregate(
        [
            measured_model(
                "bert",
                [
                    raw_test(
                        "tests/models/bert/test_modeling_bert.py::test_save",
                        outcome="failed",
                        longrepr=UNSUPPORTED_DETAIL,
                    )
                ],
            ),
            measured_model(
                "qwen3",
                [
                    raw_test(
                        "tests/models/qwen3/test_modeling_qwen3.py::test_save",
                        outcome="failed",
                        longrepr=second_trace,
                    )
                ],
            ),
        ]
    )

    result = triage.triage_failures(aggregate)

    assert len(result["findings"]) == 1
    assert result["findings"][0]["count"] == 2
    occurrence = [
        model["tests"][0]["occurrence_fingerprint"] for model in aggregate["models"]
    ]
    assert occurrence[0] != occurrence[1]


def test_triage_reports_an_all_environment_error_run_as_not_actionable():
    """An environment that never reached the assertion is not a clean run."""
    result = triage.triage_failures(
        measured_model(
            "bert",
            [
                raw_test(
                    "tests/models/bert/test_modeling_bert.py::test_audio",
                    outcome="failed",
                    longrepr="ModuleNotFoundError: No module named 'librosa'",
                ),
                raw_test(
                    "tests/models/bert/test_modeling_bert.py::test_vision",
                    outcome="failed",
                    longrepr="ModuleNotFoundError: No module named 'librosa'",
                ),
            ],
        )
    )

    finding = result["findings"][0]
    assert finding["class"] == "ENVIRONMENT_ERROR"
    assert finding["subject"] == "librosa"
    assert finding["actionable"] is False
    assert finding["verification_required"] is False
    assert finding["verdict"] == "INCONCLUSIVE"
    assert result["summary"]["total_failures"] == 2
    assert result["summary"]["environment_error"] == 1
    assert result["summary"]["actionable"] == 0
    assert result["summary"]["statuses"] == {"ENVIRONMENT_ERROR": 2}


def test_triage_reports_setup_errors_as_test_errors():
    result = triage.triage_failures(
        measured_model(
            "bert",
            [
                raw_test(
                    "tests/models/bert/test_modeling_bert.py::test_train",
                    outcome="failed",
                    when="setup",
                    longrepr="fixture 'tokenizer' not found",
                )
            ],
        )
    )

    finding = result["findings"][0]
    assert finding["class"] == "TEST_ERROR"
    assert finding["subject"] == "setup_or_teardown_error"
    assert finding["actionable"] is False


def test_triage_reads_collection_failures():
    result = triage.triage_failures(
        measured_model(
            "bert",
            [
                raw_collect(
                    "tests/models/bert", "TypeError: unsupported operand type(s)"
                )
            ],
        )
    )

    assert result["summary"]["statuses"] == {"COLLECT_ERROR": 1}
    finding = result["findings"][0]
    assert finding["class"] == "TEST_ERROR"
    assert finding["subject"] == "collect_error"
    assert finding["actionable"] is False


def test_cpu_fallback_becomes_confirmed_operator_finding():
    result = triage.triage_failures(
        measured_model(
            "qwen3",
            [
                raw_test(
                    "tests/models/qwen3/test_modeling_qwen3.py::test_ok",
                    cpu_fallback_ops=["aten::div"],
                )
            ],
        )
    )
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["class"] == "OP_CPU_FALLBACK"
    assert finding["component"] == DEVICE
    assert finding["subject"] == "aten::div"
    assert finding["verdict"] == "CONFIRMED"
    assert not finding["verification_required"]


def test_run_level_poison_does_not_classify_every_failure_as_crash():
    result = triage.triage_failures(
        {
            "environment": {"device": DEVICE},
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


# --- fingerprints -------------------------------------------------------------


def test_fingerprint_separates_backend_components():
    first = triage.generate_fingerprint("OP_UNSUPPORTED", "musa", "aten::div", "x")
    second = triage.generate_fingerprint("OP_UNSUPPORTED", "ascend", "aten::div", "x")
    assert first != second


def test_compute_fingerprint_alias_still_resolves():
    assert triage.compute_fingerprint is triage.generate_fingerprint


def test_normalize_error_is_idempotent():
    text = (
        "RuntimeError: bad at 0x7f8a9b2c1d40 after 1.234s in "
        "/tmp/pytest-of-root/pytest-12/test_x0/test.py for shape [3, 4] (code 17)"
    )
    once = triage.normalize_error(text)
    assert triage.normalize_error(once) == once


def test_normalize_error_folds_a_legacy_address_token_without_growing_it():
    """``0xADDR`` is itself valid hex; re-normalizing it used to append an ``R``."""
    once = triage.normalize_error("failed at 0xADDR")
    assert once == "failed at <ADDR>"
    assert triage.normalize_error(once) == once


def test_fingerprint_is_stable_across_run_noise():
    second = (
        UNSUPPORTED_DETAIL.replace("0x7f8a9b2c1d40", "0x55d3ab")
        .replace("1.234s", "9.87s")
        .replace("[3, 4]", "[17, 2]")
        .replace("pytest-12", "pytest-99")
    )

    first_mechanism = triage.mechanism_from(UNSUPPORTED_DETAIL)
    second_mechanism = triage.mechanism_from(second)
    assert first_mechanism == second_mechanism
    assert triage.generate_fingerprint(
        "OP_UNSUPPORTED", DEVICE, "aten::index_copy_.out", first_mechanism
    ) == triage.generate_fingerprint(
        "OP_UNSUPPORTED", DEVICE, "aten::index_copy_.out", second_mechanism
    )


# --- baseline round-trip ------------------------------------------------------

COVERAGE_DOC = """# HuggingFace Transformers Coverage

## Baseline: MUSA MTT S5000

| Field | Value |
| --- | --- |
| Date | 2026-01-01 |

| Class | Subject | Affected tests | Issue |
| --- | --- | --- | --- |
| `CRASH` | old cause | 3 | [#250](https://example.invalid/250) |

### Failed test inventory

The prose below the table must survive an update.
"""


def test_baseline_round_trip_keeps_what_it_writes(tmp_path):
    coverage = tmp_path / "hf-coverage.md"
    coverage.write_text(COVERAGE_DOC)
    findings_json = {
        "findings": [
            {
                "fingerprint": "aa11bb22cc33",
                "class": "OP_UNSUPPORTED",
                "subject": "aten::index_copy_.out",
                "count": 2,
            },
            {
                "fingerprint": "dd44ee55ff66",
                "class": "CRASH",
                "subject": "segfault",
                "count": 1,
            },
        ]
    }

    file_issues.update_baseline_with_issues(
        findings_json,
        {"aa11bb22cc33": 301, "dd44ee55ff66": 302},
        coverage,
        "flagos-ai/Torch-FL",
    )

    written = coverage.read_text()
    assert "| Fingerprint | Class | Subject | Affected tests | Issue |" in written
    assert "| `aa11bb22cc33` | `OP_UNSUPPORTED` |" in written
    assert "The prose below the table must survive an update." in written

    assert dedup.extract_baseline_fingerprints(coverage) == {
        "aa11bb22cc33": "issue #301",
        "dd44ee55ff66": "issue #302",
    }


def test_baseline_reader_accepts_the_standalone_fingerprint_marker(tmp_path):
    """Issue bodies carry a bare ``Fingerprint: `hash``` line."""
    coverage = tmp_path / "hf-coverage.md"
    coverage.write_text(
        "# Coverage\n\n## Baseline: Ascend 910B\n\nFingerprint: `99887766aabb`\n"
    )
    assert dedup.extract_baseline_fingerprints(coverage) == {
        "99887766aabb": "baseline:Ascend 910B"
    }


def test_baseline_update_refuses_a_document_without_a_cause_table(tmp_path):
    coverage = tmp_path / "hf-coverage.md"
    coverage.write_text("# Coverage\n\nNo baseline section here.\n")
    with pytest.raises(ValueError, match="no '## Baseline:' section"):
        file_issues.update_baseline_with_issues(
            {
                "findings": [
                    {"fingerprint": "aa11bb22cc33", "class": "CRASH", "subject": "x"}
                ]
            },
            {"aa11bb22cc33": 301},
            coverage,
            "flagos-ai/Torch-FL",
        )


# --- deduplication ------------------------------------------------------------


def actionable_finding(fingerprint="aa11bb22cc33", **overrides):
    finding = {
        "fingerprint": fingerprint,
        "class": "OP_UNSUPPORTED",
        "component": DEVICE,
        "subject": "aten::index_copy_.out",
        "mechanism": "RuntimeError: Could not run 'aten::index_copy_.out'",
        "nodeids": ["tests/models/bert/test_modeling_bert.py::test_save"],
        "models": ["bert"],
        "count": 1,
        "actionable": True,
        "verification_required": True,
        "verdict": "CONFIRMED",
    }
    finding.update(overrides)
    return finding


def empty_baseline(tmp_path):
    coverage = tmp_path / "hf-coverage.md"
    coverage.write_text("# HuggingFace Transformers Coverage\n")
    return coverage


def test_github_timeout_does_not_declare_a_new_finding(tmp_path, monkeypatch):
    """A duplicate check that could not run must not read as "nothing found"."""

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("gh", 30)

    monkeypatch.setattr(dedup.subprocess, "run", timeout)
    with pytest.raises(dedup.GitHubSearchUnavailable, match="timed out"):
        dedup.gh_search(["gh", "api", "repos/x/y/issues"], "issue body search")

    result = dedup.deduplicate_findings(
        {"findings": [actionable_finding()], "summary": {}},
        empty_baseline(tmp_path),
        "flagos-ai/Torch-FL",
        skip_github=False,
    )
    assert result["findings"] == []
    assert result["summary"]["dedup"]["DEDUP_UNAVAILABLE"] == 1


def test_github_failure_exit_status_does_not_declare_a_new_finding(
    tmp_path, monkeypatch
):
    def rejected(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="gh: Not Found (HTTP 404)\n"
        )

    monkeypatch.setattr(dedup.subprocess, "run", rejected)
    with pytest.raises(dedup.GitHubSearchUnavailable, match="HTTP 404"):
        dedup.gh_search(["gh", "api", "repos/x/y/issues"], "issue body search")

    result = dedup.deduplicate_findings(
        {"findings": [actionable_finding()], "summary": {}},
        empty_baseline(tmp_path),
        "flagos-ai/Torch-FL",
        skip_github=False,
    )
    assert result["findings"] == []
    assert result["summary"]["dedup"]["DEDUP_UNAVAILABLE"] == 1


def test_skip_github_marks_not_checked_rather_than_new(tmp_path):
    result = dedup.deduplicate_findings(
        {"findings": [actionable_finding()], "summary": {}},
        empty_baseline(tmp_path),
        "flagos-ai/Torch-FL",
        skip_github=True,
    )
    assert result["findings"] == []
    assert result["summary"]["dedup"]["NOT_CHECKED"] == 1
    assert result["summary"]["dedup"]["NEW"] == 0


def test_dedup_reports_every_status_the_sweep_wrapper_reads(tmp_path, monkeypatch):
    """``transformers_auto_sweep.sh`` counts unchecked findings, so they must be
    countable from the summary alone.

    Only ``NEW`` findings survive into ``findings``; every other status is a
    summary entry. A wrapper that looked for ``DEDUP_UNAVAILABLE`` in the
    findings array would always count zero and could never refuse a preview.
    """

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("gh", 30)

    monkeypatch.setattr(dedup.subprocess, "run", timeout)
    finding = actionable_finding()
    result = dedup.deduplicate_findings(
        {"findings": [finding], "summary": {}},
        empty_baseline(tmp_path),
        "flagos-ai/Torch-FL",
        skip_github=False,
    )

    blocked = {"IN_BASELINE", "DUPLICATE", "COLLATERAL", "INCONCLUSIVE"}
    blocked |= {"NOT_ACTIONABLE", "DEDUP_UNAVAILABLE", "NOT_CHECKED"}
    assert blocked <= set(result["summary"]["dedup"])
    assert result["summary"]["dedup"]["DEDUP_UNAVAILABLE"] == 1
    assert all(f["dedup_status"] == "NEW" for f in result["findings"])


def test_a_finding_that_claims_no_defect_never_reaches_github(tmp_path, monkeypatch):
    def bomb(*args, **kwargs):
        raise AssertionError("no tracker call may be made for a non-actionable finding")

    monkeypatch.setattr(dedup.subprocess, "run", bomb)
    result = dedup.deduplicate_findings(
        {
            "findings": [
                actionable_finding(
                    **{"class": "ENVIRONMENT_ERROR"},
                    actionable=False,
                    verdict="INCONCLUSIVE",
                )
            ],
            "summary": {},
        },
        empty_baseline(tmp_path),
        "flagos-ai/Torch-FL",
        skip_github=False,
    )
    assert result["findings"] == []
    assert result["summary"]["dedup"]["NOT_ACTIONABLE"] == 1


def test_baseline_hit_is_not_checked_against_github(tmp_path, monkeypatch):
    def bomb(*args, **kwargs):
        raise AssertionError("a finding already in the baseline needs no search")

    monkeypatch.setattr(dedup.subprocess, "run", bomb)
    coverage = tmp_path / "hf-coverage.md"
    coverage.write_text(
        "# Coverage\n\n## Baseline: MUSA MTT S5000\n\n"
        "| Fingerprint | Class | Subject | Affected tests | Issue |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `aa11bb22cc33` | `OP_UNSUPPORTED` | aten::index_copy_.out | 1 | "
        "[#301](https://example.invalid/301) |\n"
    )
    result = dedup.deduplicate_findings(
        {"findings": [actionable_finding()], "summary": {}},
        coverage,
        "flagos-ai/Torch-FL",
        skip_github=False,
    )
    assert result["findings"] == []
    assert result["summary"]["dedup"]["IN_BASELINE"] == 1


# --- verification -------------------------------------------------------------


def test_verification_error_is_inconclusive():
    assert verify.determine_verdict("ERROR", "CRASH") == "INCONCLUSIVE"
    assert verify.determine_verdict("FAIL", "CRASH") == "CONFIRMED"
    assert verify.determine_verdict("PASS", "CRASH") == "COLLATERAL"


def test_isolated_timeout_reclassifies_a_finding_as_a_crash():
    finding = actionable_finding()
    verify.apply_isolation(
        finding,
        {
            "status": "TIMEOUT",
            "detail": "Test exceeded 120s timeout",
            "duration_s": 120.0,
            "command": "pytest ...",
        },
    )
    assert finding["class"] == "CRASH"
    assert "reclassified" in finding["isolation_note"]
    assert finding["fingerprint"] != actionable_finding()["fingerprint"]
    assert finding["verdict"] == "CONFIRMED"


def test_resolve_test_source_accepts_exact_tree(tmp_path):
    (tmp_path / "tests" / "models").mkdir(parents=True)
    assert verify.resolve_test_source(tmp_path, None) == tmp_path


def test_resolve_test_source_uses_numeric_version_order(tmp_path):
    old = tmp_path / "transformers-5.9.0"
    new = tmp_path / "transformers-5.16.1"
    old.mkdir()
    new.mkdir()
    assert verify.resolve_test_source(tmp_path, None) == new


# --- drafts and filing --------------------------------------------------------


def confirmed_finding(**overrides):
    finding = triage.triage_failures(
        measured_model(
            "bert",
            [
                raw_test(
                    "tests/models/bert/test_modeling_bert.py::test_save",
                    outcome="failed",
                    longrepr=UNSUPPORTED_DETAIL,
                )
            ],
        )
    )["findings"][0]
    finding["verdict"] = "CONFIRMED"
    finding["isolation_status"] = "FAIL"
    finding["isolation_detail"] = UNSUPPORTED_DETAIL
    finding.update(overrides)
    return finding


def test_preview_sidecar_is_exactly_what_the_filer_submits(tmp_path):
    finding = confirmed_finding()
    title = preview.generate_issue_title(finding, "MUSA MTT S5000", "5.16.1")
    labels = preview.issue_class(finding["class"])["labels"]

    preview.write_sidecar(tmp_path, finding, title, labels)
    metadata = file_issues.read_sidecar(tmp_path, finding["fingerprint"])

    assert metadata["title"] == title
    assert metadata["labels"] == labels
    assert metadata["class"] == finding["class"]
    assert metadata["subject"] == finding["subject"]


def test_sidecar_must_record_the_fingerprint_it_is_named_for(tmp_path, monkeypatch):
    finding = actionable_finding()
    preview.write_sidecar(
        tmp_path, finding, preview.generate_issue_title(finding, "GCU", "5.16.1"), []
    )
    with pytest.raises(ValueError, match="records no labels"):
        file_issues.read_sidecar(tmp_path, finding["fingerprint"])


def test_generated_body_selects_the_issue_type_that_matches_its_class():
    context = {
        "python_version": "3.11.9",
        "pytorch_version": "2.10.0",
        "torch_fl_commit": "abc1234",
        "transformers_version": "5.16.1",
    }
    body = preview.generate_issue_body(confirmed_finding(), "GCU", context)

    assert "- [x] Operator Implementation" in body
    assert "- [x] Bug Report" not in body
    assert "## Environment (for bug reports)" in body
    assert "**Build config:**" in body
    assert "<!-- UNFILLED:" in body


def test_unready_gate_rejects_placeholder_prose_and_says_why():
    context = {
        "python_version": "3.11.9",
        "pytorch_version": "2.10.0",
        "torch_fl_commit": "abc1234",
        "transformers_version": "5.16.1",
    }
    markers = file_issues.unready_markers(
        preview.generate_issue_body(confirmed_finding(), "GCU", context)
    )

    assert any(marker.startswith("unfilled field: ") for marker in markers)
    assert "human review checklist is still unticked" in markers

    prose = "**Hardware**: fill in driver and SDK versions before filing\n"
    assert file_issues.unready_markers(prose) == [
        "placeholder prose: **Hardware**: fill in driver and SDK versions before filing"
    ]

    assert file_issues.unready_markers("- [x] Human reviewer has completed\n") == []


def test_issue_class_table_covers_every_class_the_reporter_can_emit():
    for failure_class in (
        "OP_UNSUPPORTED",
        "OP_CPU_FALLBACK",
        "FEATURE_UNSUPPORTED",
        "PRECISION",
        "CRASH",
        "TEST_ERROR",
        "ENVIRONMENT_ERROR",
    ):
        entry = preview.issue_class(failure_class)
        assert entry["labels"], failure_class
        assert entry["type"] in preview.ISSUE_TYPE_NAMES


def test_issue_body_metadata_accepts_platform_and_legacy_chip_keys():
    body = "- **Platform**: MUSA MTT S5000\n- **Transformers**: 5.16.1\n"
    assert file_issues.extract_body_metadata(body, "Platform") == "MUSA MTT S5000"
    assert file_issues.extract_body_metadata(body, "Transformers") == "5.16.1"

    legacy = "- **Chip**: GCU S60\n"
    assert file_issues.extract_body_metadata(legacy, "Chip") == "GCU S60"


# --- the device name has exactly one source -----------------------------------


def test_device_name_comes_from_the_installed_device_spec():
    spec = REPO_ROOT / "tests" / "manual" / "hf_device_spec.py"
    assert source.device_name(spec) == DEVICE
    assert runner.spec_device() == DEVICE


def test_device_name_rejects_a_spec_without_the_assignment(tmp_path):
    spec = tmp_path / "hf_device_spec.py"
    spec.write_text("MANUAL_SEED_FN = None\n")
    with pytest.raises(source.SourceError, match="no DEVICE_NAME"):
        source.device_name(spec)


# --- the sweep's interpreter probe --------------------------------------------
#
# The probe decides whether a run is worth starting, so it has to answer the
# question the test children will ask. Two ways of getting that wrong both
# shipped, and neither is visible from the Python side: the interpreter's own
# warnings were read as a missing-module verdict, and the repository root was
# allowed to answer for a checkout that was never installed. These tests drive
# the real script with a stand-in interpreter.

AUTO_SWEEP = REPO_ROOT / "scripts" / "transformers" / "transformers_auto_sweep.sh"


def fake_interpreter(tmp_path, stdout="", stderr="", status=0, cwd_log=None):
    """A stand-in for the accelerator interpreter.

    It answers the step-zero probe and fails every other invocation, so a test
    never starts the real suite.
    """
    lines = ["#!/bin/bash", 'if [ "${1:-}" = "-c" ]; then']
    if cwd_log is not None:
        lines.append(f'  pwd >> "{cwd_log}"')
    if stderr:
        lines.append(f'  echo "{stderr}" >&2')
    if stdout:
        lines.append(f'  echo "{stdout}"')
    lines.append(f"  exit {status}")
    lines.append("fi")
    lines.append('echo "fake interpreter: cannot run the suite" >&2')
    lines.append("exit 2")
    path = tmp_path / "fake-interpreter"
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)
    return path


def run_sweep(interpreter, tmp_path, model="bert"):
    return subprocess.run(
        ["bash", str(AUTO_SWEEP), model, "MetaX"],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHON": str(interpreter), "TMPDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_sweep_probe_does_not_read_interpreter_chatter_as_a_verdict(tmp_path):
    interpreter = fake_interpreter(
        tmp_path,
        stderr="UserWarning: Could not find flash_attn package installed",
        stdout="",
    )
    proc = run_sweep(interpreter, tmp_path)
    assert "cannot import the test environment" not in proc.stderr
    assert "has torch, transformers and torch_fl" in proc.stdout
    # The probe passed, so the sweep moved on to the measurement.
    assert "[1/6]" in proc.stdout


def test_sweep_probe_reports_the_missing_module_it_was_told_about(tmp_path):
    interpreter = fake_interpreter(
        tmp_path, stdout="torch_fl (No module named 'torch_fl')"
    )
    proc = run_sweep(interpreter, tmp_path)
    assert proc.returncode == 2
    assert "cannot import the test environment" in proc.stderr
    assert "torch_fl (No module named 'torch_fl')" in proc.stderr
    assert "[1/6]" not in proc.stdout


def test_sweep_probe_runs_outside_the_repository(tmp_path):
    """The working directory is on ``sys.path`` for ``python -c``.

    Probing from the repository root would import a checkout's ``torch_fl`` and
    call an interpreter healthy while every test child, which runs from a private
    work directory, cannot import it.
    """
    cwd_log = tmp_path / "probe-cwd"
    interpreter = fake_interpreter(tmp_path, cwd_log=cwd_log)
    run_sweep(interpreter, tmp_path)

    probed = [Path(line) for line in cwd_log.read_text().splitlines() if line]
    assert probed, "the probe never ran"
    for path in probed:
        assert path != REPO_ROOT
        assert not path.is_relative_to(REPO_ROOT)


# --- what the sweep tells its caller -------------------------------------------
#
# The sweep's exit code is how the batch driver learns what the model measured:
# ``transformers_batch_sweep.sh`` maps 0 to "nothing to file", 1 to "findings
# awaiting review", and anything else to "nothing was measured". The sweep used
# to fall off the end of step 6, so a run that had just written a preview of new
# findings exited 0 and the batch summary counted it as clean --- the measurement
# reached the preview and stopped there, and the driver's one signal that
# something was measured was gone.

STAGE_INTERPRETER = '''#!/usr/bin/env python3
"""A stand-in for the tools the sweep drives.

It answers the probe, records a measurement, and writes each stage's output, so
the sweep's own control flow --- and the exit code it hands its caller --- is
exercised without an accelerator, an architecture suite, or a tracker.
"""
import json
import sys

argv = sys.argv[1:]
if argv and argv[0] == "-c":
    if "importlib.import_module" in argv[1]:
        # The step-zero probe: this interpreter is healthy, so it prints no
        # missing module. Every other snippet is a count the sweep reads back
        # from a file, so it runs for real.
        raise SystemExit(0)
    exec(compile(argv[1], "<script>", "exec"), {})
    raise SystemExit(0)

script = argv[0]
out = argv[argv.index("--out") + 1]


def write(payload):
    with open(out, "w") as handle:
        json.dump(payload, handle)


def write_text():
    with open(out, "w") as handle:
        handle.write("# Issue Preview\\n")


if script.endswith("transformers_hf_tests.py"):
    write(
        {
            "run": {"status": "COMPLETED_RESILIENT"},
            "tests": [
                {
                    "nodeid": "tests/models/bert/test_modeling_bert.py"
                    "::BertModelTest::test_x",
                    "status": "__TEST_STATUS__",
                }
            ],
        }
    )
    raise SystemExit(__RUNNER_EXIT__)
if script.endswith("transformers_triage.py"):
    findings = [] if "__OUTCOME__" == "clean" else [{"actionable": True}]
    write({"findings": findings, "summary": {"environment_error": 0}})
elif script.endswith("transformers_verify.py"):
    write({"findings": [{"fingerprint": "aa11bb22cc33"}]})
elif script.endswith("transformers_deduplicate.py"):
    write(
        {
            "findings": [],
            "summary": {"dedup": {"NEW": 0 if "__OUTCOME__" == "clean" else 1}},
        }
    )
elif script.endswith("transformers_preview_issues.py"):
    write_text()
raise SystemExit(0)
'''


def stage_interpreter(tmp_path, outcome):
    """The interpreter above, set up to report ``outcome``."""
    clean = outcome == "clean"
    source = (
        STAGE_INTERPRETER.replace("__OUTCOME__", outcome)
        .replace("__RUNNER_EXIT__", "0" if clean else "1")
        .replace("__TEST_STATUS__", "PASS" if clean else "FAIL")
    )
    path = tmp_path / "stage-interpreter"
    path.write_text(source)
    path.chmod(0o755)
    return path


def test_a_sweep_that_produced_findings_exits_one(tmp_path):
    """Measured-with-findings has to reach the driver as its own outcome."""
    proc = run_sweep(stage_interpreter(tmp_path, "findings"), tmp_path)
    assert "Issue Preview (1 issues)" in proc.stdout
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Exit code 1" in proc.stdout


def test_a_sweep_with_nothing_to_file_exits_zero(tmp_path):
    """The clean outcome stays distinguishable from the findings one."""
    proc = run_sweep(stage_interpreter(tmp_path, "clean"), tmp_path)
    assert "No failures to triage. All tests passed!" in proc.stdout
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ``--chip`` is the hardware label that reaches the report title and the issue
# preview. Vendors name boards "vendor + part number", and the allowlist is a
# list of vendors, so the safe wrapper rejected every real board name --- the
# mixed-case vendors in its own allowlist included --- while accepting only an
# all-uppercase vendor word.

CHIP_LABELS = [
    "MetaX",
    "MetaX C550",
    "MUSA MTT S5000",
    "Enflame GCU S60",
    "ASCEND 910B",
    "GCU",
]


@pytest.mark.parametrize("label", CHIP_LABELS)
def test_chip_label_accepts_a_board_name_and_keeps_the_caller_spelling(label):
    wrapper = load("safe_transformers_wrapper")
    assert wrapper.validate_chip(label) == label


@pytest.mark.parametrize("label", ["nosuchchip", "MetaX-adjacent", ""])
def test_chip_label_rejects_a_label_that_names_no_known_vendor(label):
    wrapper = load("safe_transformers_wrapper")
    with pytest.raises(SystemExit):
        wrapper.validate_chip(label)


# --- what a run reports about itself ------------------------------------------

BERT_NODEID = (
    "tests/models/bert/test_modeling_bert.py::BertModelTest::test_attention_outputs"
)


def test_triage_keeps_the_environment_the_verifier_reads_the_version_from():
    """The version must come from the run, not from the cache directory listing.

    Triage returned findings and a summary only, so the verifier fell back to
    the newest cached source tree: a 5.12.1 measurement was isolated against a
    5.14.1 checkout, with one warning as the only trace of it.
    """
    result = triage.triage_failures(
        measured_model(
            "bert",
            [raw_test(BERT_NODEID, outcome="failed", longrepr=UNSUPPORTED_DETAIL)],
        )
    )
    assert result["environment"]["transformers"] == ENVIRONMENT["transformers"]
    assert verify.recorded_version(result) == ENVIRONMENT["transformers"]


def test_triage_keeps_the_environment_of_a_full_sweep():
    """``--all`` keeps the environment per model block; one interpreter measured
    them all, so the run's own environment still has to reach the verifier."""
    aggregate = all_aggregate(
        [
            measured_model(
                "bert",
                [raw_test(BERT_NODEID, outcome="failed", longrepr=UNSUPPORTED_DETAIL)],
            )
        ]
    )
    assert (
        verify.recorded_version(triage.triage_failures(aggregate))
        == (ENVIRONMENT["transformers"])
    )


def test_reported_nodeid_gets_its_file_back():
    """pytest reports a nodeid it was given without the file part.

    Every batch recorded ``::BertModelTest::test_x``, and the verifier passed
    that back to pytest, which rejected it as a directory argument containing a
    selection. All nine isolations of a full sweep collected nothing.
    """
    tests = [
        {"nodeid": "::BertModelTest::test_attention_outputs"},
        {"nodeid": "::BertModelTest::test_cpu_offload"},
    ]
    runner.canonicalize_nodeids(
        tests,
        [
            BERT_NODEID,
            "tests/models/bert/test_modeling_bert.py::BertModelTest::test_cpu_offload",
        ],
    )
    assert tests[0]["nodeid"] == BERT_NODEID
    assert tests[1]["nodeid"].endswith("::BertModelTest::test_cpu_offload")
    assert "test_modeling_bert.py" in tests[1]["nodeid"]


def test_a_nodeid_tail_two_files_share_is_left_as_reported():
    """Restoring a shared tail by guess would attribute a result to a file that
    did not produce it, so the reported form is kept instead."""
    tests = [{"nodeid": "::BertModelTest::test_save"}]
    runner.canonicalize_nodeids(
        tests,
        [
            "tests/models/bert/test_modeling_bert.py::BertModelTest::test_save",
            "tests/models/bert/test_tokenization_bert.py::BertModelTest::test_save",
        ],
    )
    assert tests[0]["nodeid"] == "::BertModelTest::test_save"


def test_a_nodeid_that_already_names_its_file_is_unchanged():
    tests = [{"nodeid": BERT_NODEID}]
    runner.canonicalize_nodeids(tests, [BERT_NODEID])
    assert tests[0]["nodeid"] == BERT_NODEID


def test_isolation_counts_only_the_summary_line_of_its_own_run():
    assert verify.isolated_outcomes("no tests ran in 0.00s") == (0, set())
    # A traceback that happens to mention a count must not become the result.
    assert verify.isolated_outcomes("assert 2 failed in 5s\n1 passed in 5.00s") == (
        1,
        {"passed"},
    )


def isolated_source(tmp_path):
    """A source tree with the two directories the isolation workdir links to."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "src").mkdir()
    return tmp_path


def test_the_isolated_run_keeps_the_repository_importable(tmp_path, monkeypatch):
    """``hf_device_spec.py`` imports ``torch_fl``, so the repository root the
    parent was given must survive into the child; it is moved behind the source
    tree rather than dropped, because the source tree has to own
    ``tests.models...`` and the repository root has to provide ``torch_fl``."""
    source = isolated_source(tmp_path)
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(verify.REPO_ROOT), "/tmp/elsewhere"])
    )
    entries = verify.isolated_env(source, tmp_path / "work")["PYTHONPATH"].split(
        os.pathsep
    )
    assert str(verify.REPO_ROOT) in entries
    assert entries.index(str(source)) < entries.index(str(verify.REPO_ROOT))
    assert entries.index("/tmp/elsewhere") < entries.index(str(verify.REPO_ROOT))
    assert entries[0] == str(tmp_path / "work")


def test_an_isolation_that_selected_no_test_is_an_error_not_evidence(
    tmp_path, monkeypatch
):
    def usage_error(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=4,
            stdout="",
            stderr=(
                "ERROR: directory argument cannot contain :: selection parts: "
                "::BertModelTest::test_attention_outputs\nno tests ran in 0.00s\n"
            ),
        )

    monkeypatch.setattr(verify.subprocess, "run", usage_error)
    result = verify.run_isolated_test(
        "::BertModelTest::test_attention_outputs", isolated_source(tmp_path), 5
    )
    assert result["status"] == "ERROR"
    assert result["reported_tests"] == 0
    assert "instead of one" in result["detail"]
    assert verify.determine_verdict(result["status"], "CRASH") == "INCONCLUSIVE"


def test_an_isolated_setup_error_is_not_a_confirmed_defect(tmp_path, monkeypatch):
    def setup_error(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="1 error in 3.10s\n"
        )

    monkeypatch.setattr(verify.subprocess, "run", setup_error)
    result = verify.run_isolated_test(BERT_NODEID, isolated_source(tmp_path), 5)
    assert result["status"] == "ERROR"
    assert result["reported_tests"] == 1


def test_one_isolated_failure_is_still_confirmed_evidence(tmp_path, monkeypatch):
    def one_failure(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="1 failed, 54 warnings in 10.30s\n",
        )

    monkeypatch.setattr(verify.subprocess, "run", one_failure)
    result = verify.run_isolated_test(BERT_NODEID, isolated_source(tmp_path), 5)
    assert result["status"] == "FAIL"
    assert result["reported_tests"] == 1
    assert verify.determine_verdict(result["status"], "CRASH") == "CONFIRMED"


TWO_VENDOR_BASELINE = """# HuggingFace Transformers Coverage

## Baseline: MUSA MTT S5000

| Fingerprint | Class | Subject | Affected tests | Issue |
| --- | --- | --- | --- | --- |
| `aa11bb22cc33` | `CRASH` | musa crash | 1 | [#250](https://example.invalid/250) |

## Baseline: MetaX C550

| Fingerprint | Class | Subject | Affected tests | Issue |
| --- | --- | --- | --- | --- |
| `dd44ee55ff66` | `CRASH` | metax crash | 1 | [#302](https://example.invalid/302) |
"""


def test_baseline_read_is_scoped_to_the_measured_hardware(tmp_path):
    coverage = tmp_path / "hf-coverage.md"
    coverage.write_text(TWO_VENDOR_BASELINE)
    assert dedup.extract_baseline_fingerprints(coverage, "MetaX C550") == {
        "dd44ee55ff66": "issue #302"
    }
    assert dedup.extract_baseline_fingerprints(coverage, "MUSA MTT S5000") == {
        "aa11bb22cc33": "issue #250"
    }
    assert dedup.extract_baseline_fingerprints(coverage, "GCU") == {}


def test_a_finding_is_not_known_from_another_vendors_baseline(tmp_path):
    """Both vendors register their device as ``flagos``, so the finding's
    component cannot tell them apart and the section heading has to."""
    coverage = tmp_path / "hf-coverage.md"
    coverage.write_text(TWO_VENDOR_BASELINE)

    elsewhere = dedup.deduplicate_findings(
        {"findings": [actionable_finding("aa11bb22cc33")], "summary": {}},
        coverage,
        "flagos-ai/Torch-FL",
        skip_github=True,
        hardware="MetaX C550",
    )
    assert elsewhere["summary"]["dedup"]["IN_BASELINE"] == 0
    assert elsewhere["summary"]["dedup"]["NOT_CHECKED"] == 1

    same_board = dedup.deduplicate_findings(
        {"findings": [actionable_finding("aa11bb22cc33")], "summary": {}},
        coverage,
        "flagos-ai/Torch-FL",
        skip_github=True,
        hardware="MUSA MTT S5000",
    )
    assert same_board["summary"]["dedup"]["IN_BASELINE"] == 1
    assert same_board["findings"] == []
