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

"""AI reviewer policy: verdict parsing, approval gating, and API writes."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts/ci"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AI = _load("ai_reviewer")
SHA = "b" * 40
REPOSITORY = "flagos-ai/Torch-FL"

CLEAN_VERDICT = {
    "decision": "approve",
    "summary": "Adds the missing test and fixes the root cause.",
    "findings": [],
}


def make_pull(**overrides):
    pull = {
        "number": 7,
        "state": "open",
        "draft": False,
        "title": "fix: resolve race",
        "body": "Root cause: the stream was not synced.",
        "user": {"login": "contributor"},
        "head": {"sha": SHA},
    }
    pull.update(overrides)
    return pull


def make_config(**overrides):
    config = {
        "provider": "anthropic",
        "api_base": "https://api.anthropic.com",
        "api_key": "key",
        "model": "model",
        "reviewer_login": "torch-fl-reviewer",
        "approve_enabled": True,
        "max_additions": 5000,
        "dry_run": False,
    }
    config.update(overrides)
    return config


def file_entry(name, patch="@@ -1 +1 @@\n-a\n+b\n", **overrides):
    entry = {
        "filename": name,
        "patch": patch,
        "additions": 1,
        "deletions": 1,
    }
    entry.update(overrides)
    return entry


class FakeGitHub:
    repository = REPOSITORY

    def __init__(self, pull=None, files=None):
        self.calls = []
        self.pull = pull if pull is not None else make_pull()
        self.files = files if files is not None else [file_entry("torch_fl/core.py")]

    def api(self, path, **kwargs):
        self.calls.append(("api", path, kwargs))
        return {"login": "torch-fl-reviewer"}

    def repo_api(self, path, **kwargs):
        self.calls.append(("api", path, kwargs))
        if kwargs.get("method", "GET") != "GET":
            return None
        if path.startswith("pulls/") and path.endswith("/files?per_page=100"):
            return [self.files]
        if path.startswith("pulls/"):
            return self.pull
        raise AssertionError(f"Unexpected API path: {path}")

    def writes(self):
        return [
            call
            for call in self.calls
            if call[0] == "api" and call[2].get("method", "GET") != "GET"
        ]


# --- verdict parsing -------------------------------------------------------


def test_extract_json_accepts_fenced_reply():
    raw = 'Here you go:\n```json\n{"decision": "approve"}\n```'
    assert AI.extract_json(raw) == {"decision": "approve"}


def test_extract_json_accepts_bare_reply_with_prose():
    raw = 'Sure. {"decision": "approve", "summary": "ok"} done'
    assert AI.extract_json(raw)["decision"] == "approve"


@pytest.mark.parametrize("raw", ["", "no json here", None, "[1, 2, 3]"])
def test_extract_json_rejects_garbage(raw):
    assert AI.extract_json(raw) is None


def test_normalize_rejects_unknown_decision():
    verdict = AI.normalize_verdict('{"decision": "ship it"}')
    assert verdict["ok"] is True
    assert verdict["decision"] == "request_changes"


def test_normalize_survives_unparseable_reply():
    verdict = AI.normalize_verdict("I could not review this.")
    assert verdict["ok"] is False
    assert verdict["decision"] == "request_changes"


def test_normalize_coerces_unknown_severity_to_low():
    raw = json.dumps(
        {"decision": "approve", "summary": "s", "findings": [{"severity": "banana"}]}
    )
    assert AI.normalize_verdict(raw)["findings"][0]["severity"] == "low"


# --- approval gating -------------------------------------------------------


def test_clean_diff_approves():
    files = [file_entry("torch_fl/core.py")]
    state, reasons = AI.decide(
        make_pull(),
        files,
        AI.normalize_verdict(json.dumps(CLEAN_VERDICT)),
        "torch-fl-reviewer",
        make_config(),
    )
    assert state == "APPROVE"
    assert reasons == []


def test_draft_is_never_approved():
    state, reasons = AI.decide(
        make_pull(draft=True),
        [file_entry("torch_fl/core.py")],
        AI.normalize_verdict(json.dumps(CLEAN_VERDICT)),
        "r",
        make_config(),
    )
    assert state == "COMMENT"
    assert any("draft" in reason for reason in reasons)


def test_self_authored_pull_request_is_not_approved():
    state, reasons = AI.decide(
        make_pull(user={"login": "torch-fl-reviewer"}),
        [file_entry("a.py")],
        AI.normalize_verdict(json.dumps(CLEAN_VERDICT)),
        "torch-fl-reviewer",
        make_config(),
    )
    assert state == "COMMENT"
    assert any("authored" in reason for reason in reasons)


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/ai-review.yml",
        ".github/scripts/ci/ai_reviewer.py",
        ".github/configs/nvidia.yml",
        "CODEOWNERS",
    ],
)
def test_protected_paths_force_a_human_review(path):
    state, reasons = AI.decide(
        make_pull(),
        [file_entry(path)],
        AI.normalize_verdict(json.dumps(CLEAN_VERDICT)),
        "r",
        make_config(),
    )
    assert state == "COMMENT"
    assert any("human reviewer" in reason for reason in reasons)


def test_high_severity_finding_blocks_approval():
    verdict = AI.normalize_verdict(
        json.dumps(
            {
                "decision": "approve",
                "summary": "mostly fine",
                "findings": [{"file": "a.py", "severity": "high", "comment": "bug"}],
            }
        )
    )
    state, reasons = AI.decide(
        make_pull(), [file_entry("a.py")], verdict, "r", make_config()
    )
    assert state == "COMMENT"
    assert any("high-severity" in reason for reason in reasons)


def test_binary_file_cannot_be_approved():
    state, reasons = AI.decide(
        make_pull(),
        [file_entry("logo.png", patch=None)],
        AI.normalize_verdict(json.dumps(CLEAN_VERDICT)),
        "r",
        make_config(),
    )
    assert state == "COMMENT"
    assert any("cannot be reviewed" in reason for reason in reasons)


def test_oversized_diff_cannot_be_approved():
    files = [file_entry("a.py", additions=6000)]
    state, reasons = AI.decide(
        make_pull(),
        files,
        AI.normalize_verdict(json.dumps(CLEAN_VERDICT)),
        "r",
        make_config(),
    )
    assert state == "COMMENT"
    assert any("too large" in reason for reason in reasons)


def test_approval_can_be_disabled_by_configuration():
    state, reasons = AI.decide(
        make_pull(),
        [file_entry("a.py")],
        AI.normalize_verdict(json.dumps(CLEAN_VERDICT)),
        "r",
        make_config(approve_enabled=False),
    )
    assert state == "COMMENT"
    assert any("disabled" in reason for reason in reasons)


def test_unparsed_verdict_is_never_approved():
    state, _ = AI.decide(
        make_pull(),
        [file_entry("a.py")],
        AI.normalize_verdict("nonsense"),
        "r",
        make_config(),
    )
    assert state == "COMMENT"


# --- review submission -----------------------------------------------------


def run_review(files, reply, pull=None, config=None):
    github = FakeGitHub(pull=pull or make_pull(), files=files)
    config = config or make_config()
    status = AI.review_pull_request(7, github, lambda prompt, cfg: reply, config)
    return github, status


def test_clean_review_submits_an_approve_review():
    github, status = run_review(
        [file_entry("torch_fl/core.py")], json.dumps(CLEAN_VERDICT)
    )
    assert status.startswith("approved:")
    reviews = [c for c in github.writes() if c[1] == "pulls/7/reviews"]
    assert len(reviews) == 1
    payload = reviews[0][2]["payload"]
    assert payload["event"] == "APPROVE"
    assert payload["commit_id"] == SHA


def test_protected_path_submits_a_comment_not_an_approval():
    github, status = run_review(
        [file_entry(".github/workflows/ci.yml")], json.dumps(CLEAN_VERDICT)
    )
    assert status.startswith("commented:")
    payload = [c for c in github.writes() if c[1] == "pulls/7/reviews"][0][2]["payload"]
    assert payload["event"] == "COMMENT"


def test_unparseable_reply_comments_rather_than_approving():
    github, status = run_review([file_entry("a.py")], "no json")
    assert status.startswith("commented:")
    payload = [c for c in github.writes() if c[1] == "pulls/7/reviews"][0][2]["payload"]
    assert payload["event"] == "COMMENT"


def test_failed_llm_call_posts_an_error_comment():
    def boom(prompt, cfg):
        raise AI.GitHubError("provider down")

    github = FakeGitHub()
    status = AI.review_pull_request(7, github, boom, make_config())
    assert status.startswith("error:")
    comments = [c for c in github.writes() if c[1] == "issues/7/comments"]
    assert len(comments) == 1


def test_closed_pull_request_is_ignored():
    github = FakeGitHub(pull=make_pull(state="closed"))
    status = AI.review_pull_request(7, github, lambda p, c: "{}", make_config())
    assert status.startswith("ignored:")
    assert github.writes() == []


def test_draft_pull_request_is_skipped_before_any_llm_call():
    calls = []

    def llm(prompt, cfg):
        calls.append(prompt)
        return json.dumps(CLEAN_VERDICT)

    github = FakeGitHub(pull=make_pull(draft=True))
    status = AI.review_pull_request(7, github, llm, make_config())
    assert status.startswith("skipped:")
    assert calls == []


def test_dry_run_writes_nothing():
    github, status = run_review(
        [file_entry("a.py")],
        json.dumps(CLEAN_VERDICT),
        config=make_config(dry_run=True),
    )
    assert status.startswith("dry-run:")
    assert github.writes() == []


def test_truncated_diff_never_approves():
    files = [file_entry("a.py", patch="x" * (AI.MAX_PATCH_CHARS + 1))]
    github, status = run_review(files, json.dumps(CLEAN_VERDICT))
    assert status.startswith("commented:")
    payload = [c for c in github.writes() if c[1] == "pulls/7/reviews"][0][2]["payload"]
    assert payload["event"] == "COMMENT"


# --- prompt and rendering --------------------------------------------------


def test_prompt_labels_the_diff_as_untrusted():
    prompt = AI.build_prompt(make_pull(), [file_entry("a.py")], False)
    assert "untrusted data" in prompt
    assert "diff --git a/a.py b/a.py" in prompt


def test_review_body_escapes_table_pipes():
    verdict = AI.normalize_verdict(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "s",
                "findings": [
                    {"file": "a.py", "severity": "low", "comment": "a | b\nc"}
                ],
            }
        )
    )
    body = AI.render_review("COMMENT", ["reason"], verdict)
    assert "a \\| b c" in body
    assert AI.REVIEW_MARKER in body


def test_protected_prefix_matching():
    assert AI._is_protected(".github/workflows/ci.yml")
    assert AI._is_protected("CODEOWNERS")
    assert not AI._is_protected("torch_fl/core.py")
    assert not AI._is_protected(".github/PULL_REQUEST_TEMPLATE.md")


# --- configuration and entry point -----------------------------------------


def test_empty_actions_variable_does_not_blank_the_default():
    # An unset Actions variable expands to "", which must not override the base
    # URL or the model id.
    config = AI.load_config(
        {
            "AI_REVIEW_API_KEY": "k",
            "AI_REVIEW_PROVIDER": "",
            "AI_REVIEW_MODEL": "",
            "AI_REVIEW_API_BASE": "",
        }
    )
    assert config["provider"] == "anthropic"
    assert config["model"] == "claude-sonnet-4-5"
    assert config["api_base"] == "https://api.anthropic.com"


def test_resolve_pull_number_prefers_the_cli_argument():
    assert AI.resolve_pull_number({"pull_request": {"number": 3}}, "9") == 9


def test_resolve_pull_number_falls_back_to_the_event():
    assert AI.resolve_pull_number({"pull_request": {"number": 3}}, "") == 3


def test_resolve_pull_number_rejects_garbage():
    with pytest.raises(ValueError):
        AI.resolve_pull_number({}, "abc")


def test_main_skips_cleanly_without_credentials(monkeypatch, capsys):
    monkeypatch.delenv("AI_REVIEW_API_KEY", raising=False)
    monkeypatch.setenv("GH_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    assert AI.main(["--pr", "7"]) == 0
    assert "skipping review" in capsys.readouterr().out
