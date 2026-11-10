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

"""The integration pipeline keeps pip downloads on the runner, not in a cache entry.

GitHub's managed cache transfers through blob storage this repository's vendor
pods cannot reach: measured on 2026-11-10, hygon moved 0 bytes in the 100 s
client timeout and ascend 11.8% at 0.2 MB/s, on every vendor lane, while the
macOS lane restored 176 MB in 5 s. The failure is silent (continue-on-error,
then "Cache not found"), so it cost ~110 s per lane and delivered nothing.

A self-hosted runner keeps its `_work` directory between jobs, so the downloads
can simply stay there.
"""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/integration-tests.yml"
STEP_NAME = "Prepare pip download cache"


def _steps():
    document = yaml.safe_load(WORKFLOW.read_text())
    return next(job["steps"] for job in document["jobs"].values() if "steps" in job)


def _step():
    return next(step for step in _steps() if step.get("name") == STEP_NAME)


def test_the_pipeline_no_longer_uses_a_managed_cache_entry():
    """A restore that cannot transfer is not worth a continue-on-error.

    Keeping one would also hide a regression: the step reports success and
    "Cache not found", so nothing in the job says the transfer failed.
    """
    uses = [step.get("uses", "") for step in _steps()]
    assert not [use for use in uses if use.startswith("actions/cache")]


def test_the_cache_lives_above_runner_temp():
    """RUNNER_TEMP and the checkout are cleared per job; `_work` is not.

    The runner mounts its host `_work` directory into the container and keeps
    `_tool`/`_actions` there for exactly that reason. RUNNER_WORKSPACE points
    inside it, and the checkout is one level further down.
    """
    script = _step()["run"]
    assert 'root="$(dirname "${RUNNER_WORKSPACE:-${GITHUB_WORKSPACE}/..}")"' in script
    # The directory the runner clears for every job must not be involved at all.
    assert "RUNNER_TEMP" not in script
    assert 'PIP_CACHE_DIR="$cache_root/${PLATFORM' in script


def test_the_key_tracks_dependencies_but_not_this_workflow():
    """Editing the pipeline changes what runs, not what pip downloads.

    The key used to hash this workflow file, which purged every platform's cache
    15 times in 30 days -- more often than the version pins (8) or the shared
    install library (8) that actually decide the package set.
    """
    key = _step()["env"]["PIP_CACHE_KEY"]
    for dependency in (
        "pyproject.toml",
        "setup.py",
        ".github/version-pins.env",
        ".github/constraints.txt",
        ".github/scripts/set_env.sh",
        ".github/scripts/lib/set_env_common.sh",
    ):
        assert dependency in key
    assert "integration-tests.yml" not in key
    assert "format('.github/scripts/hooks/set_env_{0}.sh'" in key
    assert "format('.github/configs/{0}.yml'" in key


def test_the_key_names_the_base_platform_so_both_lanes_share_it():
    """Boxing changes the kernels compiled, not the packages downloaded.

    Measured on run 41976848532: thead and thead-boxing install a byte-identical
    package set. `needs.configuration.outputs.platform` is the config basename
    with the -boxing suffix already stripped, so one directory serves both.
    """
    key = _step()["env"]["PIP_CACHE_KEY"]
    assert key.startswith("${{ needs.configuration.outputs.platform }}-pip-")
    assert "inputs.platform" not in key


def _run_step(tmp_path, key):
    """Execute the step's script against a fake runner and return (env, files)."""
    work = tmp_path / "_work"
    (work / "Torch-FL" / "Torch-FL").mkdir(parents=True, exist_ok=True)
    genv = tmp_path / "github_env"
    genv.write_text("")
    environment = {
        **os.environ,
        "RUNNER_WORKSPACE": str(work / "Torch-FL"),
        "GITHUB_WORKSPACE": str(work / "Torch-FL" / "Torch-FL"),
        "GITHUB_ENV": str(genv),
        "PLATFORM": "hygon",
        "PIP_CACHE_KEY": key,
    }
    done = subprocess.run(
        ["bash", "-c", _step()["run"]],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    cache = work / ".pip-cache" / "hygon"
    return genv.read_text(), cache, done.stdout


def test_a_warm_directory_is_reused_for_the_same_key(tmp_path):
    env_text, cache, _ = _run_step(tmp_path, "hygon-pip-Linux-aaa")
    (cache / "torch-2.10.0+cpu.whl").write_text("wheel")

    env_text, cache, _ = _run_step(tmp_path, "hygon-pip-Linux-aaa")
    assert (cache / "torch-2.10.0+cpu.whl").exists()
    assert f"PIP_CACHE_DIR={cache}" in env_text


def test_a_changed_key_discards_the_previous_downloads(tmp_path):
    """The wheels are content-addressed, but a stale set is still the wrong set."""
    _env_text, cache, _ = _run_step(tmp_path, "hygon-pip-Linux-aaa")
    (cache / "torch-2.10.0+cpu.whl").write_text("wheel")

    _env_text, cache, output = _run_step(tmp_path, "hygon-pip-Linux-bbb")
    assert not (cache / "torch-2.10.0+cpu.whl").exists()
    assert "pip cache key changed for hygon" in output


@pytest.mark.parametrize("platform", ["hygon", "thead"])
def test_each_platform_gets_its_own_directory(tmp_path, platform):
    """A runner that hosts two platforms must not have them purge each other."""
    step = _step()["run"]
    work = tmp_path / "_work"
    (work / "Torch-FL" / "Torch-FL").mkdir(parents=True, exist_ok=True)
    genv = tmp_path / "github_env"
    done = subprocess.run(
        ["bash", "-c", step],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RUNNER_WORKSPACE": str(work / "Torch-FL"),
            "GITHUB_WORKSPACE": str(work / "Torch-FL" / "Torch-FL"),
            "GITHUB_ENV": str(genv),
            "PLATFORM": platform,
            "PIP_CACHE_KEY": f"{platform}-pip-Linux-aaa",
        },
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert f"/.pip-cache/{platform}" in genv.read_text()
