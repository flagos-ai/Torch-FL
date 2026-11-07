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


def _condition(expression, states, cancelled=False, success=True, event_name="push"):
    """Evaluate the workflow's boolean expression against recorded action outputs.

    `event_name` stands in for the context's `github.event_name`. It defaults to
    `push`, the run that is allowed to write the cache.
    """

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
    expression = re.sub(r"github\.event_name", repr(event_name), expression)
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

    # A pull request restores exactly as a push does, but never writes: its
    # cache would land in a refs/pull/<n>/merge scope no branch can read, while
    # counting against the same 10 GB budget.
    assert _condition(restore["if"], states, cancelled, success=job_success) is (
        download and job_success
    )
    assert (
        _condition(
            upload["if"],
            states,
            cancelled,
            success=job_success,
            event_name="pull_request",
        )
        is False
    )


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_cache_failure_budgets_do_not_gate_installation(workflow):
    # Every transfer is bounded so a stalled cache service cannot turn into an
    # unbounded wait before the install and the tests. Keep these caps explicit
    # so tuning stays visible.
    #
    # The restore cap is 10 minutes because a failed transfer is a cold miss, not
    # a failure: at 3 minutes the largest cache (Ascend's, which holds the
    # FlagTree wheel) timed out on every run of PRs #707/#712 and was then
    # re-downloaded from the index, so the cap cost a second transfer instead of
    # saving one. The wait is still bounded, and the segment stall detector is
    # what catches a wedged transfer inside that budget.
    restore_maximum, segment_maximum = 10, 5
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
        # The base platform, not the lane: a platform's two lanes install the
        # same package set, so keying on the lane stored a duplicate entry.
        assert "needs.configuration.outputs.platform" in key
        assert (
            "format('.github/configs/{0}.yml', needs.configuration.outputs.platform)"
            in key
        )
        assert "${{ inputs.platform }}-pip-" not in key


def test_gcu_can_reuse_caches_from_the_retired_split_jobs():
    # GCU was the last platform on the shared common workflow, whose cache keys
    # were built from the display name. The unified pipeline keeps those prefixes
    # as restore-keys so the first run after the collapse still finds them.
    lookup = next(
        step
        for step in _steps("integration-tests.yml")
        if step.get("id") == "pip-cache-lookup"
    )
    assert lookup["with"]["key"].startswith(
        "${{ needs.configuration.outputs.platform }}-pip-"
    )
    for prefix in (
        "${{ needs.configuration.outputs.platform }}-pip-${{ runner.os }}-",
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
    hook = (ROOT / ".github/scripts/hooks/set_env_nvidia.sh").read_text()
    assert re.search(r"^install_cpu_torch\s*$", hook, re.M)
    assert re.search(
        r"pip_retry --no-cache-dir --index-url \"\$VENDOR_TORCH_INDEX_URL\"", hook
    )


def test_cuda_bootstraps_the_vendor_torch_only_on_the_boxing_lane():
    """The vendor lane runs on the +cpu torch; only boxing stages a libtorch_cuda.

    The vendor lane builds the wheel whose premise is that it carries no
    libtorch_cuda.so, and nothing downstream of the bootstrap reads
    VENDOR_PYTHON there. Bootstrapping it anyway downloaded the multi-GB
    torch+cu130 wheel for an unread path, and that download is what failed the
    lane while the PyTorch index answered 503 through the runner proxy.
    """
    hook = (ROOT / ".github/scripts/hooks/set_env_nvidia.sh").read_text()

    # The bootstrap is guarded by BOXING_LANE: the vendor lane resolves the
    # source as "none" instead of installing one.
    assert (
        'elif [[ "${BOXING_LANE:-}" == "1" ]]; then\n'
        '  VENDOR_SOURCE="bootstrap"\n'
        "  ensure_system_prerequisites\n"
        "  bootstrap_vendor_python\n"
        "else\n"
        "  # The vendor lane: no accelerator torch is installed, and none is read below.\n"
        '  VENDOR_SOURCE="none"\n'
        "fi" in hook
    )

    # The explicit mode override cannot smuggle a bootstrap onto the vendor lane.
    assert "TORCH_FL_CUDA_VENDOR_MODE=bootstrap asks for an accelerator PyTorch" in hook

    # The probe reads the accelerator torch, so it must not run without one.
    assert (
        'if [[ "$VENDOR_SOURCE" != "none" ]]; then\n'
        "  resolve_vendor_torch\n"
        "else\n"
        '  echo "Vendor lane: no accelerator PyTorch is resolved or installed; '
        'tests run on the +cpu torch"\n'
        "fi" in hook
    )


def test_reporting_tools_install_through_the_pypi_fallback():
    """A reporting tool must not fail a lane whose tests can still pass.

    The vendor images inherit a pip index of their own (MUSA's is TUNA) and
    those are VCS mirrors: TUNA listed pytest-json-report and then answered 403
    for the wheel's own file, so PR #714's MUSA lane died before running any
    test. The step installs through install_with_pypi_fallback, which retries on
    the pinned index, rather than a bare `pip install` that lands on whatever
    index the image configured.
    """
    steps = _steps(WORKFLOW_DIR / "integration-tests.yml")
    step = next(
        step for step in steps if step.get("name") == "Install pytest JSON reporting"
    )
    run = step["run"]
    assert "install_with_pypi_fallback" in run
    assert ".github/version-pins.env" in run
    assert re.search(r"^\s*python -m pip install\b", run, re.M) is None

    # The helper falls back only when the caller did not pin an index itself.
    common = (ROOT / ".github/scripts/lib/set_env_common.sh").read_text()
    helper = re.search(
        r"^install_with_pypi_fallback\(\) \{\n(.*?)^\}", common, re.M | re.S
    )
    assert helper
    assert '--index-url "$pinned"' in helper.group(1)


def test_every_platform_takes_flagtree_out_of_the_cache():
    """FlagTree wheels are 190-3283 MB and restore slower than they download.

    Measured across the platforms that install one: the cache restored at
    5.8-7.7 MB/s (341 MB in 55 s up to 3348 MB in 435 s, run 41976848532),
    while resource.flagos.net served at 54.7 MB/s (run 42084793856). Every
    wheel is faster uncached -- 48-375 s per platform -- and together they were
    holding ~2.4 GB of the 10 GB budget.

    The opt-out lives in the shared install_flagtree rather than in each hook,
    because there is no platform whose wheel is small enough for the cache to
    win. Raising PIP_RETRY_NO_CACHE for that one call is what keeps every other
    install in a hook cached.
    """
    common = (ROOT / ".github/scripts/lib/set_env_common.sh").read_text()
    tree = re.search(r"^install_flagtree\(\) \{\n(.*?)^\}", common, re.M | re.S)
    assert tree
    assert "PIP_RETRY_NO_CACHE=1 pip_retry" in tree.group(1)
    # The override is per-call, so the rest of the library still caches.
    assert not re.search(r"^export PIP_RETRY_NO_CACHE=1", common, re.M)

    # No hook re-implements it; the shared phase is the one place it is decided.
    for name in (
        "nvidia",
        "ascend",
        "mthreads",
        "hygon",
        "metax",
        "enflame",
        "kunlunxin",
        "iluvatar",
        "thead",
    ):
        hook = (ROOT / f".github/scripts/hooks/set_env_{name}.sh").read_text()
        assert not re.search(r"PIP_RETRY_NO_CACHE=1", hook), name


def test_a_platforms_two_lanes_share_one_cache_entry():
    """Boxing changes the kernels compiled, not the packages downloaded.

    Measured on run 41976848532: thead and thead-boxing install a byte-identical
    package set, and their cache entries were both 619 MB -- duplicates of each
    other. Keying on the lane cost about 2 GB of the 4.3 GB main held across
    nvidia, metax, hygon and thead, which is what kept the 10 GB budget full.
    """
    lookup = next(
        step
        for step in _steps("integration-tests.yml")
        if step.get("id") == "pip-cache-lookup"
    )
    key = lookup["with"]["key"]
    # `needs.configuration.outputs.platform` is the config basename with the
    # -boxing suffix already stripped (load-platform-config.yml documents it).
    assert "needs.configuration.outputs.platform" in key
    assert "inputs.platform" not in key
    for pattern in re.findall(r"format\('([^']+)'", key):
        assert "{0}" in pattern
    # The hook and config are still hashed, now through the base platform.
    assert "format('.github/scripts/hooks/set_env_{0}.sh'" in key
