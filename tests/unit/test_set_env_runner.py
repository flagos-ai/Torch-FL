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

"""The set_env entrypoint: one runner, per-platform hooks, thin wrappers.

`.github/scripts/set_env.sh --platform <p>` owns the boilerplate every platform
shares and sources `hooks/set_env_<p>.sh`, which is the platform-specific part.
The old `set_env_<p>.sh` files are thin wrappers around the runner, and the
integration pipeline calls the runner directly. This pins that shape: the runner
dispatches, every platform has a hook and a wrapper, and the pipeline uses the
parameterized entrypoint.

The runner is exercised with `--help`/an unknown platform, which return before
any provisioning runs.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / ".github" / "scripts"
RUNNER = SCRIPTS / "set_env.sh"
HOOKS = SCRIPTS / "hooks"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

PLATFORMS = [
    "cuda",
    "ascend",
    "dcu",
    "gcu",
    "iluvatar",
    "kunlunxin",
    "metax",
    "musa",
    "ppu",
]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(RUNNER), *args],
        capture_output=True,
        text=True,
        env={"CI_STAGE": "build", "PATH": "/usr/bin:/bin"},
    )


def test_runner_rejects_a_missing_platform():
    done = _run()
    assert done.returncode == 2
    assert "--platform is required" in done.stderr


def test_runner_rejects_an_unknown_platform():
    done = _run("--platform", "bogus")
    assert done.returncode == 2
    assert "unknown platform 'bogus'" in done.stderr


def test_runner_rejects_an_unknown_argument():
    done = _run("--nope")
    assert done.returncode == 2
    assert "unknown argument" in done.stderr


def test_every_platform_has_a_hook():
    for platform in PLATFORMS:
        assert (HOOKS / f"set_env_{platform}.sh").is_file(), platform


def test_every_wrapper_routes_to_the_runner():
    for platform in PLATFORMS:
        text = (SCRIPTS / f"set_env_{platform}.sh").read_text(encoding="utf-8")
        assert "set_env.sh" in text and f"--platform {platform}" in text, platform


def test_the_parameterized_workflow_provisions_every_platform():
    """One workflow provisions every platform, from that platform's config.

    There used to be a file per platform under `integration-test-<p>.yml`, each
    repeating `bash .github/scripts/set_env.sh --platform <p>` and hard-coding
    its stage. The pipeline reads setup_script/setup_platform/setup_stage from
    the config now, so a platform that needs the `build` stage (Ascend,
    Kunlunxin) is no longer the only shape the file can express.
    """
    text = (WORKFLOWS / "integration-tests.yml").read_text(encoding="utf-8")
    assert (
        'bash "${{ needs.configuration.outputs.setup_script }}" '
        '--platform "${{ needs.configuration.outputs.setup_platform }}"' in text
    )
    assert "CI_STAGE: ${{ needs.configuration.outputs.setup_stage }}" in text


def test_config_driven_path_uses_the_runner():
    """The configs name the runner and the platform, so no wrapper hop.

    The pipeline reads setup_script/setup_platform from the config and runs
    `bash set_env.sh --platform <p>`; the wrappers stay only for the docs and
    local use.
    """
    for platform in PLATFORMS:
        text = (REPO_ROOT / ".github" / "configs" / f"{platform}.yml").read_text(
            encoding="utf-8"
        )
        assert "setup_script: .github/scripts/set_env.sh" in text, platform
        assert f"setup_platform: {platform}" in text, platform


def test_common_workflows_pass_the_platform():
    """The config-driven wheel build calls the runner with --platform."""
    text = (WORKFLOWS / "build-wheel-common.yml").read_text(encoding="utf-8")
    assert (
        'bash "${{ inputs.setup_script }}" --platform "${{ inputs.setup_platform }}"'
        in text
    )
    assert "      setup_platform:" in text


def test_config_discovery_exports_the_platform():
    """load-platform-config.yml must declare setup_platform as a job output.

    It is read into the config step's GITHUB_OUTPUT, but `needs.<job>.outputs`
    only exposes what the job's `outputs:` block lists; without this the
    pipeline receives an empty platform and the runner exits 2.
    """
    text = (WORKFLOWS / "load-platform-config.yml").read_text(encoding="utf-8")
    assert "setup_platform: ${{ steps.config.outputs.setup_platform }}" in text
    assert 'emit setup_platform "$SETUP_PLATFORM"' in text


def test_runner_applies_the_pip_constraints():
    """The runner exports PIP_CONSTRAINT; the file exists and pins the known drift.

    The constraints file is what fixes the transitive pulls the pin table cannot
    reach, so it must be applied to every install (the runner exports it before
    sourcing the hook) and must still name them.
    """
    runner = RUNNER.read_text(encoding="utf-8")
    assert 'export PIP_CONSTRAINT="${REPO_ROOT}/.github/constraints.txt"' in runner
    constraints = (REPO_ROOT / ".github" / "constraints.txt").read_text(
        encoding="utf-8"
    )
    for pin in ("torch==", "numpy<2", "transformers>="):
        assert pin in constraints, pin
