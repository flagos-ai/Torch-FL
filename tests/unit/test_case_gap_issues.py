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

"""The issue gate reads live state and fails closed when verification is lost."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "ci" / "check_case_gap_issues.py"
REGISTRY = ROOT / "tests/integration" / "case_gap_issues.py"


def _checker():
    spec = importlib.util.spec_from_file_location("case_gap_checker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _script_arguments(module):
    return [
        str(module.DEFAULT_SCRIPT_ROOT / SCRIPT.name),
        "--script-root",
        str(module.DEFAULT_SCRIPT_ROOT),
        "--registry",
        str(REGISTRY),
    ]


def test_registry_reads_the_shared_issue_and_its_component_fix():
    gap = _checker().registered_gaps(REGISTRY)[0]
    assert (gap.number, gap.repository, gap.upstream) == (
        "#710",
        "flagos-ai/Torch-FL",
        "flagos-ai/FlagGems/issues/8411",
    )


def test_closed_and_unavailable_issues_fail_the_gate():
    module = _checker()
    gap = module.registered_gaps(REGISTRY)[0]
    calls = []

    def lookup(repository, number):
        calls.append((repository, number))
        return "closed"

    errors = module.validate_gaps((gap,), lookup)
    assert len(errors) == 2
    assert "issue flagos-ai/Torch-FL/issues/710 is closed" in errors[0]
    assert "upstream flagos-ai/FlagGems/issues/8411 is closed" in errors[1]
    assert calls == [
        ("flagos-ai/Torch-FL", "710"),
        ("flagos-ai/FlagGems", "8411"),
    ]

    def upstream_closed(repository, number):
        return "closed" if repository == "flagos-ai/FlagGems" else "open"

    errors = module.validate_gaps((gap,), upstream_closed)
    assert len(errors) == 1
    assert "upstream flagos-ai/FlagGems/issues/8411 is closed" in errors[0]

    def unavailable(*_):
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    errors = module.validate_gaps((gap,), unavailable)
    assert len(errors) == 2
    assert all("could not verify" in error for error in errors)


@pytest.mark.parametrize(
    "row",
    [
        'CaseGapIssue("#710", cases=("x::test_x",), routes=("x",), upstream="a/b/issues/1")',
        'CaseGapIssue("#710", cases=("x::test_x",), routes=(), upstream="a/b/issues/1")',
        'CaseGapIssue(number="#710", cases=(), routes=(), upstream="a/b/issues/1")',
        'CaseGapIssue(number="710", cases=("x::test_x",), routes=(), upstream="a/b/issues/1")',
        'CaseGapIssue(number="#710", cases=("x::test_x",), routes=(), upstream="a/b/pulls/1")',
        'CaseGapIssue(number="#710", cases=("x::test_x",), routes=(), upstream="a/b/issues/1", state="open")',
        'CaseGapIssue(number="#710", cases=(), routes=("#710",), upstream="")',
    ],
)
def test_registry_rejects_untracked_or_incomplete_rows(row, tmp_path):
    path = tmp_path / "case_gap_issues.py"
    path.write_text(f"_ISSUES = ({row},)\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _checker().registered_gaps(path)


def test_registry_rejects_duplicate_case_owners(tmp_path):
    row = (
        "CaseGapIssue(number='#{0}', cases=('x::test_x',), routes=(), "
        "upstream='a/b/issues/{1}')"
    )
    path = tmp_path / "case_gap_issues.py"
    path.write_text(
        f"_ISSUES = ({row.format(1, 10)}, {row.format(2, 20)})\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly one issue owner"):
        _checker().registered_gaps(path)


def test_registry_groups_rows_with_the_same_issue(tmp_path):
    path = tmp_path / "case_gap_issues.py"
    path.write_text(
        "_ISSUES = ("
        "CaseGapIssue(number='#710', cases=('x::test_x',), routes=(), "
        "upstream='a/b/issues/1'), "
        "CaseGapIssue(number='#710', cases=(), routes=('x',), "
        "upstream='a/b/issues/1'),"
        ")\n",
        encoding="utf-8",
    )
    gaps = _checker().registered_gaps(path)
    assert [(gap.number, gap.repository, gap.upstream) for gap in gaps] == [
        ("#710", "flagos-ai/Torch-FL", "a/b/issues/1")
    ]


def test_runtime_skip_finds_one_issue_across_case_and_route_rows():
    runtime = importlib.import_module("tests.integration.case_gap_issues")
    assert runtime.open_case_gap("#710").number == "#710"
    assert runtime.open_case_gap("all").number == "#710"


def test_registry_rejects_different_upstreams_for_one_issue(tmp_path):
    path = tmp_path / "case_gap_issues.py"
    path.write_text(
        "_ISSUES = ("
        "CaseGapIssue(number='#710', cases=('x::test_x',), routes=(), "
        "upstream='a/b/issues/10'), "
        "CaseGapIssue(number='#710', cases=(), routes=('x',), "
        "upstream='a/b/issues/1'),"
        ")\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="one upstream"):
        _checker().registered_gaps(path)


def test_api_issue_rejects_a_pull_request(monkeypatch):
    module = _checker()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, '{"pull_request": {}}', ""
        ),
    )
    with pytest.raises(ValueError, match="must name an issue"):
        module._api_issue("flagos-ai/Torch-FL", "723")


def test_api_issue_does_not_accept_user_supplied_gh_environment(monkeypatch):
    module = _checker()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, '{"state": "OPEN"}', "")

    monkeypatch.setenv("GH_TOKEN", "workflow-token")
    monkeypatch.setattr(module.subprocess, "run", run)
    assert module._api_issue("flagos-ai/Torch-FL", "710") == "open"
    assert "env" not in calls[0][1]


def test_main_fails_when_an_issue_is_closed(monkeypatch, capsys):
    module = _checker()
    gap = module.registered_gaps(REGISTRY)[0]
    monkeypatch.setattr(module, "registered_gaps", lambda _: (gap,))
    monkeypatch.setattr(
        module,
        "_api_issue",
        lambda repository, _: (
            "closed" if repository == "flagos-ai/Torch-FL" else "open"
        ),
    )
    monkeypatch.setattr(sys, "argv", _script_arguments(module))
    assert module.main() == 1
    assert "issue flagos-ai/Torch-FL/issues/710 is closed" in capsys.readouterr().out


def test_main_fails_when_github_cannot_be_queried(monkeypatch):
    module = _checker()
    gap = module.registered_gaps(REGISTRY)[0]
    monkeypatch.setattr(module, "registered_gaps", lambda _: (gap,))

    def unavailable(*_):
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(module, "_api_issue", unavailable)
    monkeypatch.setattr(sys, "argv", _script_arguments(module))
    assert module.main() == 1


def test_main_rejects_the_script_outside_its_trusted_root(tmp_path):
    module = _checker()
    script = tmp_path / "check_case_gap_issues.py"
    script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--script-root",
            str(module.DEFAULT_SCRIPT_ROOT),
            "--registry",
            str(REGISTRY),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1
    assert "must run from the trusted base script" in process.stdout


def test_main_checks_skip_forms_in_every_migrated_source(tmp_path):
    module = _checker()
    relative = "tests/integration/ops/test_all_dispatch.py"
    contract = module.CaseGapContract("", (relative,))
    source = "import pytest\ndef test_x():\n    pytest.skip('gap')\n"
    submitted = tmp_path / relative
    submitted.parent.mkdir(parents=True)
    submitted.write_text(source, encoding="utf-8")
    assert module._read_contract_sources(tmp_path, contract) == {relative: source}
    errors = module.source_violations(
        {relative: source}, (), frozenset(contract.sources)
    )
    assert any(
        error.startswith(f"{relative}:") and "test skips must use" in error
        for error in errors
    )


_GAP = """CaseGapIssue(number="#710", cases=("tests/integration/ops/test_x.py::test_x",), routes=(), upstream="flagos-ai/FlagGems/issues/8411")
"""
_CONTRACT = "MIGRATED_CASE_FILES = frozenset({'tests/integration/ops/test_x.py'})\n"
_VALID_SOURCE = (
    "from tests.integration.case_gap_issues import tracked_skip\n"
    "@tracked_skip(condition=True, issue='#710')\n"
    "def test_x():\n"
    "    return True\n"
)


def _source_errors(source):
    module = _checker()
    gaps = module.parse_registered_gaps(
        __import__("ast").parse(f"_ISSUES = ({_GAP},)"), require_upstream=True
    )
    return module.source_violations(
        {"tests/integration/ops/test_x.py": source},
        gaps,
        frozenset({"tests/integration/ops/test_x.py"}),
    )


def test_source_contract_accepts_only_the_registered_decorator():
    assert _source_errors(_VALID_SOURCE) == []


def test_migrated_source_without_a_skip_has_nothing_to_import():
    module = _checker()
    relative = "tests/integration/ops/test_mps_argmax.py"
    source = "def test_x():\n    return True\n"
    assert module.source_violations({relative: source}, (), frozenset({relative})) == []


@pytest.mark.parametrize(
    "source, message",
    [
        (
            _VALID_SOURCE.replace(
                "from tests.integration.case_gap_issues import tracked_skip\n", ""
            ),
            "shared registry",
        ),
        (
            _VALID_SOURCE.replace("@tracked_skip", "@pytest.mark.skipif"),
            "must use the tracked_skip",
        ),
        (
            _VALID_SOURCE.replace(
                "    return True\n", "    import pytest\n    pytest.skip('gap')\n"
            ),
            "must use the tracked_skip",
        ),
        (
            _VALID_SOURCE.replace(
                "    return True\n", "    pytest.importorskip('x')\n"
            ),
            "must use the tracked_skip",
        ),
        (
            _VALID_SOURCE.replace(
                "from tests.integration.case_gap_issues import tracked_skip\n",
                "from tests.integration.case_gap_issues import tracked_skip as skip\n",
            ),
            "must not be aliased",
        ),
        (
            _VALID_SOURCE
            + "original = tracked_skip\n"
            + "tracked_skip = lambda *args, **kwargs: (lambda f: f)\n",
            "cannot be rebound",
        ),
        (
            "import pytest\n"
            "from tests.integration.case_gap_issues import tracked_skip\n"
            "pytestmark = pytest.mark.skipif(True, reason='gap')\n"
            + _VALID_SOURCE.split("\n", 1)[1],
            "module-wide",
        ),
        (
            "from tests.integration.case_gap_issues import tracked_skip\n"
            "@tracked_skip(condition=True, issue='#710', reason='ignored')\n"
            "def test_x():\n"
            "    return True\n",
            "accepts only",
        ),
        (
            _VALID_SOURCE.replace("issue='#710'", "issue='#711'"),
            "no registry row",
        ),
        (
            _VALID_SOURCE.replace("def test_x", "def test_other"),
            "no cases owned",
        ),
    ],
)
def test_source_contract_rejects_unregistered_skip_forms(source, message):
    assert any(message in error for error in _source_errors(source))
