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

"""The issue gate and legacy skip ratchet fail closed."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "ci" / "check_case_gap_issues.py"
REGISTRY = ROOT / "tests/integration" / "case_gap_issues.py"
BASELINE = ROOT / "tests/integration" / "skip_policy_baseline.json"


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
        "--baseline",
        str(BASELINE),
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


def test_main_rejects_tracked_skip_in_an_unregistered_unmigrated_case(
    tmp_path, monkeypatch, capsys
):
    module = _checker()
    repository = tmp_path.resolve()
    (repository / "tests/integration/ops").mkdir(parents=True)
    (repository / "tests/integration/flaggems").mkdir()

    migrated = "tests/integration/ops/test_registered.py"
    unmigrated = "tests/integration/ops/test_legacy.py"
    registry = repository / "tests/integration/case_gap_issues.py"
    registry.write_text(
        f"MIGRATED_CASE_FILES = frozenset({{{migrated!r}}})\n"
        "_ISSUES = ("
        "CaseGapIssue("
        "number='#710', "
        f"cases=({migrated + '::test_registered'!r},), "
        "routes=(), "
        "upstream='flagos-ai/FlagGems/issues/8411'"
        "),"
        ")\n",
        encoding="utf-8",
    )
    (repository / migrated).write_text(
        "from tests.integration.case_gap_issues import tracked_skip\n"
        "@tracked_skip(condition=True, issue='#710')\n"
        "def test_registered():\n"
        "    return True\n",
        encoding="utf-8",
    )
    (repository / unmigrated).write_text(
        "from tests.integration.case_gap_issues import tracked_skip\n"
        "@tracked_skip(condition=True, issue='#710')\n"
        "def test_unregistered():\n"
        "    return True\n",
        encoding="utf-8",
    )
    baseline = repository / "tests/integration/skip_policy_baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "version": 1,
                "roots": [
                    "tests/integration/flaggems/",
                    "tests/integration/ops/",
                ],
                "files": {},
                "kinds": {},
            }
        ),
        encoding="utf-8",
    )

    arguments = _script_arguments(module)
    arguments[arguments.index("--registry") + 1] = str(registry)
    arguments[arguments.index("--baseline") + 1] = str(baseline)
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(module, "_api_issue", lambda *_: "open")

    assert module.main() == 1
    assert (
        f"{unmigrated}:2: #710 has no cases owned by {unmigrated}::test_unregistered"
        in capsys.readouterr().out
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


def _baseline(module, files):
    return module.SkipPolicyBaseline(
        roots=module.POLICY_ROOTS,
        files=files,
        kinds={},
    )


def test_checked_in_skip_baseline_matches_the_exact_source_inventory():
    module = _checker()
    sources = module.read_policy_sources(ROOT)
    observed = {
        relative: counts
        for relative, source in sources.items()
        if (counts := module.skip_call_counts(source, filename=relative))
    }
    generated = module.build_skip_policy_baseline(observed)
    stored = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert stored == generated

    baseline = module.load_skip_policy_baseline(BASELINE)
    contract = module.case_gap_contract(
        REGISTRY.read_text(encoding="utf-8"),
        (ROOT / "tests/unit/test_integration_source_contract.py").read_text(
            encoding="utf-8"
        ),
    )
    assert (
        module.skip_policy_violations(
            sources, baseline, migrated=frozenset(contract.sources)
        )
        == []
    )


@pytest.mark.parametrize(
    "source, kind",
    [
        ("import pytest\npytest.skip('new')\n", "pytest.skip"),
        (
            "import pytest\n@pytest.mark.skipif(True, reason='new')\n"
            "def test_x(): pass\n",
            "pytest.mark.skipif",
        ),
    ],
)
def test_existing_file_cannot_gain_a_legacy_skip(source, kind):
    module = _checker()
    relative = "tests/integration/ops/test_existing.py"
    baseline = _baseline(module, {relative: {kind: 1}})
    errors = module.skip_policy_violations({relative: source}, baseline)
    assert errors == []

    errors = module.skip_policy_violations({relative: source + source}, baseline)
    assert any(
        f"{relative}:{kind}: 2 legacy skip calls, baseline allows 1" in error
        for error in errors
    )


@pytest.mark.parametrize(
    "source",
    [
        "import pytest\npytest.skip('new')\n",
        (
            "import pytest\n@pytest.mark.skipif(True, reason='new')\n"
            "def test_x(): pass\n"
        ),
    ],
)
def test_new_file_cannot_inherit_a_legacy_skip_allowance(source):
    module = _checker()
    baseline = _baseline(
        module,
        {"tests/integration/ops/test_existing.py": {"pytest.skip": 1}},
    )
    relative = "tests/integration/ops/test_new.py"
    errors = module.skip_policy_violations({relative: source}, baseline)
    assert any(
        error.startswith(f"{relative}:") and "new legacy skip calls" in error
        for error in errors
    )


def test_removing_a_bare_skip_requires_shrinking_the_baseline():
    module = _checker()
    relative = "tests/integration/ops/test_one.py"
    baseline = _baseline(module, {relative: {"pytest.mark.skipif": 2}})
    source = (
        "import pytest\n@pytest.mark.skipif(True, reason='one')\ndef test_one(): pass\n"
    )
    errors = module.skip_policy_violations({relative: source}, baseline)
    assert any("1 legacy skip calls, baseline allows 2" in error for error in errors)


def test_stale_baseline_file_and_kind_rows_are_rejected():
    module = _checker()
    present = "tests/integration/ops/test_present.py"
    gone = "tests/integration/ops/test_gone.py"
    source = "import pytest\npytest.skip('one')\n"
    baseline = _baseline(
        module,
        {
            present: {"pytest.skip": 1, "pytest.mark.xfail": 1},
            gone: {"pytest.skip": 1},
        },
    )
    errors = module.skip_policy_violations({present: source}, baseline)
    assert any(f"{present}:pytest.mark.xfail" in error for error in errors)
    assert any(f"{gone}: baseline file has no" in error for error in errors)


def test_migrated_file_rejects_any_legacy_skip_even_in_a_baseline():
    module = _checker()
    relative = "tests/integration/ops/test_migrated.py"
    source = "import pytest\npytest.skip('gap')\n"
    baseline = _baseline(module, {relative: {"pytest.skip": 1}})
    errors = module.skip_policy_violations(
        {relative: source}, baseline, migrated=frozenset({relative})
    )
    assert any("migrated cross-chip files may not contain" in error for error in errors)


def test_direct_policy_markers_are_reserved_for_the_shared_helpers():
    module = _checker()
    source = (
        "import pytest\n"
        "@pytest.mark.flagos_tracked_gap(issue='#710')\n"
        "@pytest.mark.flagos_platform_gate(gate='device_count', detail='two')\n"
        "def test_x(): pass\n"
    )
    relative = "tests/integration/ops/test_markers.py"
    errors = module.skip_policy_violations({relative: source}, _baseline(module, {}))
    assert (
        len(
            [
                error
                for error in errors
                if "markers must come from the shared helper" in error
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    "files, kinds, message",
    [
        ("not-a-mapping", {}, "files and kinds objects"),
        ({"tests/integration/ops/test_x.py": {"pytest.skip": 0}}, {}, "positive"),
        ({"tests/integration/unit/test_x.py": {"pytest.skip": 1}}, {}, "invalid"),
        (
            {"tests/integration/ops/test_x.py": {"pytest.skip": 1}},
            {"pytest.skip": 2},
            "do not match",
        ),
    ],
)
def test_skip_baseline_loader_rejects_invalid_rows(tmp_path, files, kinds, message):
    module = _checker()
    path = tmp_path / "baseline.json"
    payload = {
        "version": 1,
        "roots": [
            "tests/integration/flaggems/",
            "tests/integration/ops/",
        ],
        "files": files,
        "kinds": kinds,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        module.load_skip_policy_baseline(path)


def test_skip_baseline_loader_rejects_noncanonical_roots(tmp_path):
    module = _checker()
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "roots": ["tests/integration/"],
                "files": {},
                "kinds": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="roots must be"):
        module.load_skip_policy_baseline(path)


def test_case_gap_contract_rejects_a_file_outside_the_policy_roots(tmp_path):
    module = _checker()
    registry = tmp_path / "case_gap_issues.py"
    registry.write_text(
        "MIGRATED_CASE_FILES = frozenset({'tests/integration/test_other.py'})\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="governed integration root"):
        module.case_gap_contract(
            registry.read_text(encoding="utf-8"), "tests/unit/test_source.py"
        )


def test_main_rejects_an_unreadable_skip_baseline(tmp_path, monkeypatch, capsys):
    module = _checker()
    arguments = _script_arguments(module)
    arguments[arguments.index("--baseline") + 1] = str(tmp_path / "missing.json")
    monkeypatch.setattr(sys, "argv", arguments)
    assert module.main() == 1
    assert "invalid case-gap contract" in capsys.readouterr().out


def test_skip_policy_contract_is_identical_in_the_runtime_and_report_paths():
    runtime = importlib.import_module("tests.integration.skip_policy")
    checker = _checker()
    spec = importlib.util.spec_from_file_location(
        "skip_policy_contract_test",
        ROOT / ".github/scripts/pytest_reporting/skip_policy_contract.py",
    )
    stable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stable)
    assert runtime.TRACKED_GAP_MARKER == stable.TRACKED_GAP_MARKER
    assert runtime.PLATFORM_GATE_MARKER == stable.PLATFORM_GATE_MARKER
    assert runtime.TRACKED_GAP_CATEGORY == stable.TRACKED_GAP_CATEGORY
    assert runtime.PLATFORM_GATE_CATEGORY == stable.PLATFORM_GATE_CATEGORY
    assert runtime.PLATFORM_GATES == stable.PLATFORM_GATES
    assert runtime.SKIP_OUTCOMES == stable.SKIP_OUTCOMES
    assert runtime.POLICY_ROOTS == stable.POLICY_ROOTS
    assert runtime.POLICY_ROOTS == checker.POLICY_ROOTS
