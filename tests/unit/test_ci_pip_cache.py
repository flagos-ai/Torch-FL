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

"""Replay cache-action outputs, including download errors reported as success."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / ".github/workflows"
# The two reusable entrypoints that carry a pip download cache: the shared wheel
# build, and the single parameterized integration pipeline that replaced the
# per-platform integration-test-<platform>.yml files.
WORKFLOWS = [
    "build-wheel-common.yml",
    "integration-tests.yml",
]


def _steps(workflow):
    document = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    return next(job["steps"] for job in document["jobs"].values() if "steps" in job)


def _condition(expression, states, cancelled=False, success=True):
    """Evaluate the workflow's boolean expression against recorded action outputs."""

    def output(match):
        step, field, name = match.groups()
        state = states.get(step, {})
        value = state.get("outputs", {}).get(name, "") if name else state.get(field, "")
        return repr(value)

    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    # Actions supplies success() unless the expression already checks status.
    if not re.search(r"\b(?:success|failure|cancelled|always)\(\)", expression):
        if not success:
            return False
    expression = re.sub(
        r"steps\.([a-z-]+)\.(outcome|outputs\.([a-z-]+))", output, expression
    )
    expression = expression.replace("cancelled()", repr(cancelled))
    expression = expression.replace("success()", repr(success))
    expression = expression.replace("failure()", repr(not success and not cancelled))
    expression = expression.replace("always()", "True")
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", "not ", expression)
    return eval(expression.strip(), {"__builtins__": {}}, {})


@pytest.mark.parametrize("workflow", WORKFLOWS)
@pytest.mark.parametrize("job_success", [True, False])
@pytest.mark.parametrize(
    "lookup_outcome,primary,hit,matched,restore_outcome,restored,cancelled,download,save",
    [
        pytest.param(
            "success", "new", "", "", "skipped", "", False, False, True, id="cold-miss"
        ),
        pytest.param(
            "failure",
            "new",
            "",
            "",
            "skipped",
            "",
            False,
            False,
            False,
            id="lookup-failed",
        ),
        pytest.param(
            "failure",
            "",
            "",
            "",
            "skipped",
            "",
            False,
            False,
            False,
            id="lookup-timeout",
        ),
        pytest.param(
            "success", "", "", "", "skipped", "", False, False, False, id="missing-key"
        ),
        pytest.param(
            "success",
            "new",
            "true",
            "new",
            "success",
            "new",
            False,
            True,
            False,
            id="exact-hit",
        ),
        pytest.param(
            "success",
            "new",
            "true",
            "new",
            "success",
            "",
            False,
            True,
            False,
            id="cuda-swallowed-timeout",
        ),
        pytest.param(
            "success",
            "new",
            "false",
            "old",
            "success",
            "old",
            False,
            True,
            True,
            id="fallback-restored",
        ),
        pytest.param(
            "success",
            "new",
            "false",
            "old",
            "success",
            "",
            False,
            True,
            False,
            id="gcu-swallowed-timeout",
        ),
        pytest.param(
            "success",
            "new",
            "false",
            "old",
            "failure",
            "",
            False,
            True,
            False,
            id="transfer-step-timeout",
        ),
        pytest.param(
            "success",
            "new",
            "false",
            "old",
            "skipped",
            "",
            False,
            True,
            False,
            id="transfer-skipped",
        ),
        pytest.param(
            "success",
            "new",
            "",
            "",
            "skipped",
            "",
            True,
            False,
            False,
            id="cancelled-cold-miss",
        ),
        pytest.param(
            "success",
            "new",
            "false",
            "old",
            "success",
            "old",
            True,
            True,
            False,
            id="cancelled-fallback",
        ),
    ],
)
def test_cache_transfer_decisions(
    workflow,
    job_success,
    lookup_outcome,
    primary,
    hit,
    matched,
    restore_outcome,
    restored,
    cancelled,
    download,
    save,
):
    steps = _steps(workflow)
    lookup = next(step for step in steps if step.get("id") == "pip-cache-lookup")
    restore = next(step for step in steps if step.get("id") == "pip-cache")
    upload = next(step for step in steps if step.get("uses") == "actions/cache/save@v4")
    states = {
        "pip-cache-lookup": {
            "outcome": lookup_outcome,
            "outputs": {
                "cache-primary-key": primary,
                "cache-hit": hit,
                "cache-matched-key": matched,
            },
        },
        "pip-cache": {
            "outcome": restore_outcome,
            "outputs": {"cache-matched-key": restored},
        },
    }
    assert lookup["with"]["lookup-only"] is True
    assert _condition(restore["if"], states, cancelled, success=job_success) is (
        download and job_success
    )
    assert _condition(upload["if"], states, cancelled, success=job_success) is save
    assert (
        upload["with"]["key"]
        == "${{ steps.pip-cache-lookup.outputs.cache-primary-key }}"
    )


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_cache_failure_budgets_do_not_gate_installation(workflow):
    # Every transfer is bounded so a stalled cache service cannot turn into an
    # unbounded wait before the install and the tests. Keep these caps explicit
    # so tuning stays visible.
    restore_maximum, segment_maximum = 3, 2
    save_maximum = 3
    for step in _steps(workflow):
        if step.get("uses", "").startswith("actions/cache/"):
            assert step["continue-on-error"] is True
            maximum = {
                "pip-cache-lookup": 1,
                "pip-cache": restore_maximum,
            }.get(step.get("id"), save_maximum)
            assert 0 < step["timeout-minutes"] <= maximum
            if step.get("id") == "pip-cache":
                segment_timeout = int(step["env"]["SEGMENT_DOWNLOAD_TIMEOUT_MINS"])
                assert (
                    0 < segment_timeout <= min(segment_maximum, step["timeout-minutes"])
                )
        elif "run" in step:
            assert "pip-cache" not in step.get("if", "")


def test_every_routed_integration_workflow_is_covered():
    # ci.yml dispatches one platform pipeline per discovered platform through the
    # single integration-tests.yml; that file is what must carry the cache.
    document = yaml.safe_load((WORKFLOW_DIR / "ci.yml").read_text())
    routed = {
        Path(job["uses"]).name
        for job in document["jobs"].values()
        if job.get("uses", "").startswith("./.github/workflows/integration-tests")
    }
    assert routed == {"integration-tests.yml"}
    assert routed <= set(WORKFLOWS)


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_cache_key_tracks_the_effective_dependencies(workflow):
    lookup = next(
        step for step in _steps(workflow) if step.get("id") == "pip-cache-lookup"
    )
    key = lookup["with"]["key"]
    for dependency in (
        "pyproject.toml",
        "setup.py",
        ".github/version-pins.env",
        ".github/constraints.txt",
        ".github/scripts/set_env.sh",
        ".github/scripts/lib/set_env_common.sh",
    ):
        assert dependency in key
    # The hook and the config are named through format() rather than baked in,
    # because hashFiles does not interpolate the raw input string but does
    # evaluate a format() argument.
    assert "set_env_*.sh" not in key
    assert f".github/workflows/{workflow}" in key
    assert "format('.github/scripts/hooks/set_env_{0}.sh'" in key
    if workflow == "build-wheel-common.yml":
        assert "inputs.setup_platform" in key
        assert "format('.github/configs/{0}.yml', inputs.setup_platform)" in key
    else:
        assert "inputs.platform" in key
        assert "format('.github/configs/{0}.yml', inputs.platform)" in key


def test_gcu_can_reuse_caches_from_the_retired_split_jobs():
    # GCU was the last platform on the shared common workflow, whose cache keys
    # were built from the display name. The unified pipeline keeps those prefixes
    # as restore-keys so the first run after the collapse still finds them.
    lookup = next(
        step
        for step in _steps("integration-tests.yml")
        if step.get("id") == "pip-cache-lookup"
    )
    assert lookup["with"]["key"].startswith("${{ inputs.platform }}-pip-")
    for prefix in (
        "${{ inputs.platform }}-pip-${{ runner.os }}-",
        "${{ needs.configuration.outputs.display_name }}-pip-${{ runner.os }}-",
        "${{ needs.configuration.outputs.display_name }}-build-pip-${{ runner.os }}-",
        (
            "${{ needs.configuration.outputs.display_name }}"
            "-integration-pip-${{ runner.os }}-"
        ),
    ):
        assert prefix in lookup["with"]["restore-keys"]


@pytest.mark.parametrize("workflow", WORKFLOWS)
@pytest.mark.parametrize("runner_temp", [True, False])
def test_cache_cleanup_actions_and_setup_share_one_directory(
    tmp_path, workflow, runner_temp
):
    """Execute the real cleanup and entrypoint, using an empty platform hook."""
    checkout = tmp_path / "checkout with spaces"
    scripts = checkout / ".github/scripts"
    shutil.copytree(ROOT / ".github/scripts", scripts)
    shutil.copy(ROOT / ".github/version-pins.env", scripts.parent / "version-pins.env")
    (scripts / "hooks/set_env_cache_test.sh").write_text(":\n")
    github_env = tmp_path / "github-env"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("RUNNER_TEMP", "PIP_CACHE_DIR")
    }
    env.update(
        GITHUB_WORKSPACE=str(checkout), GITHUB_ENV=str(github_env), CI_STAGE="build"
    )
    if runner_temp:
        env["RUNNER_TEMP"] = str(tmp_path / "runner temp")
    expected = (
        Path(env["RUNNER_TEMP"]) / "pip-cache"
        if runner_temp
        else checkout / ".ci/pip-cache"
    )
    expected.mkdir(parents=True)
    (expected / "stale").write_text("leftover")
    steps = _steps(workflow)
    reset = next(step for step in steps if step.get("name") == "Reset pip cache dir")
    subprocess.run(["bash", "-e", "-c", reset["run"]], env=env, check=True)
    assert expected.is_dir()
    assert not (expected / "stale").exists()
    assert github_env.read_text().splitlines() == [f"PIP_CACHE_DIR={expected}"]
    env["PIP_CACHE_DIR"] = str(expected)
    subprocess.run(
        ["bash", str(scripts / "set_env.sh"), "--platform", "cache_test"],
        env=env,
        check=True,
    )
    assert github_env.read_text().splitlines() == [f"PIP_CACHE_DIR={expected}"] * 2
    for step in steps:
        if step.get("uses", "").startswith("actions/cache/"):
            assert step["with"]["path"] == "${{ env.PIP_CACHE_DIR }}"


def test_cuda_caches_cpu_torch_without_the_vendor_wheel_set():
    hook = (ROOT / ".github/scripts/hooks/set_env_cuda.sh").read_text()
    assert re.search(r"^install_cpu_torch\s*$", hook, re.M)
    assert re.search(
        r"pip_retry --no-cache-dir --index-url \"\$VENDOR_TORCH_INDEX_URL\"", hook
    )
