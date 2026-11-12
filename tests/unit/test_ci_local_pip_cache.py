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


@pytest.mark.parametrize("platform", ["hygon", "enflame"])
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


CACHE_HELPER = ROOT / ".github/scripts/lib/pip_cache.sh"
RUNTIME_PLATFORMS = ["ascend", "iluvatar", "kunlunxin", "mthreads", "thead"]


def _run_shared_cache(
    tmp_path, platform, *, cache=None, key="", before="", **overrides
):
    env_file = tmp_path / "shared-github-env"
    env_file.write_text("")
    env = {
        **os.environ,
        "REPO_ROOT": str(ROOT),
        "RUNNER_TEMP": str(tmp_path / "job"),
        "PIP_CACHE_DIR": str(cache or tmp_path / "persistent" / platform),
        "PIP_RETRY_NO_CACHE": "0",
        "PIP_NO_CACHE_DIR": "",
        "GITHUB_ENV": str(env_file),
        **overrides,
    }
    import shlex

    script = (
        "set -euo pipefail\n"
        + f"source {shlex.quote(str(CACHE_HELPER))}\n"
        + before
        + f"\nprepare_pip_cache {shlex.quote(platform)} {shlex.quote(key)}\n"
    )
    result = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30
    )
    rows = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    return result, rows, env


@pytest.mark.parametrize("platform", RUNTIME_PLATFORMS)
def test_runtime_lanes_keep_warm_downloads_and_reset_on_key_change(tmp_path, platform):
    result, rows, _ = _run_shared_cache(tmp_path, platform, key="first")
    assert result.returncode == 0, result.stderr
    cache = Path(rows["PIP_CACHE_DIR"])
    wheel = cache / "wheel-payload"
    wheel.write_text("cached wheel")
    result, rows, _ = _run_shared_cache(tmp_path, platform, key="first")
    assert result.returncode == 0, result.stderr
    assert wheel.read_text() == "cached wheel"
    assert rows["PIP_RETRY_NO_CACHE"] == "0"
    assert rows["PIP_NO_CACHE_DIR"] == ""
    result, rows, _ = _run_shared_cache(tmp_path, platform, key="second")
    assert result.returncode == 0, result.stderr
    assert Path(rows["PIP_CACHE_DIR"]) == cache
    assert not wheel.exists()


@pytest.mark.parametrize("platform", RUNTIME_PLATFORMS)
def test_a_blocked_cache_uses_a_persistent_uid_directory(tmp_path, platform):
    original = tmp_path / "blocked cache"
    original.write_text("do not remove")
    result, rows, _ = _run_shared_cache(tmp_path, platform, cache=original)
    assert result.returncode == 0, result.stderr
    assert rows["PIP_CACHE_DIR"] == str(original) + f"-uid-{os.geteuid()}"
    assert original.read_text() == "do not remove"
    assert "::warning::" in result.stderr
    assert rows["PIP_RETRY_NO_CACHE"] == "0"


@pytest.mark.parametrize("platform", RUNTIME_PLATFORMS)
def test_foreign_owned_cache_is_preserved_instead_of_failing(tmp_path, platform):
    if os.geteuid() == 0:
        cache = tmp_path / "foreign"
        cache.mkdir()
        os.chown(cache, 65534, 65534)
    else:
        cache = Path("/usr/share")
        assert cache.stat().st_uid != os.geteuid()
    original_owner = cache.stat().st_uid
    result, rows, _ = _run_shared_cache(tmp_path, platform, cache=cache)
    assert result.returncode == 0, result.stderr
    assert Path(rows["PIP_CACHE_DIR"]) != cache
    assert cache.stat().st_uid == original_owner
    assert Path(rows["PIP_CACHE_DIR"]).stat().st_uid == os.geteuid()


@pytest.mark.parametrize("platform", RUNTIME_PLATFORMS)
def test_readonly_mount_probe_failure_falls_back(tmp_path, platform):
    result, rows, env = _run_shared_cache(
        tmp_path,
        platform,
        before="""
mktemp() {
  case "$*" in
    *persistent*) return 1 ;;
    *) command mktemp "$@" ;;
  esac
}
""",
    )
    assert result.returncode == 0, result.stderr
    assert rows["PIP_CACHE_DIR"] == str(
        Path(env["RUNNER_TEMP"]) / "pip-cache" / platform
    )
    assert rows["PIP_RETRY_NO_CACHE"] == "0"


@pytest.mark.parametrize("platform", RUNTIME_PLATFORMS)
def test_all_unusable_cache_paths_warn_and_disable_without_stopping_build(
    tmp_path, platform
):
    result, rows, _ = _run_shared_cache(
        tmp_path, platform, before="mktemp() { return 1; }\n"
    )
    assert result.returncode == 0, result.stderr
    assert rows["PIP_CACHE_DIR"] == ""
    assert rows["PIP_RETRY_NO_CACHE"] == "1"
    assert rows["PIP_NO_CACHE_DIR"] == "1"
    assert "continuing without download caching" in result.stderr


@pytest.mark.parametrize("platform", RUNTIME_PLATFORMS)
@pytest.mark.parametrize(
    "override", [{"PIP_RETRY_NO_CACHE": "1"}, {"PIP_NO_CACHE_DIR": "false"}]
)
def test_explicit_no_cache_is_honored_and_persisted(tmp_path, platform, override):
    result, rows, env = _run_shared_cache(tmp_path, platform, **override)
    assert result.returncode == 0, result.stderr
    assert rows["PIP_RETRY_NO_CACHE"] == "1"
    assert rows["PIP_NO_CACHE_DIR"] == "1"
    assert not Path(env["PIP_CACHE_DIR"]).exists()


@pytest.mark.parametrize("platform", RUNTIME_PLATFORMS)
def test_selected_fallback_survives_the_next_workflow_shell(tmp_path, platform):
    blocked = tmp_path / "blocked"
    blocked.touch()
    result, rows, env = _run_shared_cache(tmp_path, platform, cache=blocked)
    assert result.returncode == 0, result.stderr
    selected = Path(rows["PIP_CACHE_DIR"])
    (selected / "download").write_text("warm")
    result, next_rows, _ = _run_shared_cache(tmp_path, platform, **{**env, **rows})
    assert result.returncode == 0, result.stderr
    assert next_rows == rows
    assert (selected / "download").read_text() == "warm"


@pytest.mark.parametrize("platform", RUNTIME_PLATFORMS)
def test_real_workflow_uses_shared_cache_selection(tmp_path, platform):
    work = tmp_path / "_work"
    work.mkdir()
    blocked = work / ".pip-cache"
    blocked.write_text("unavailable persistent mount")
    env_file = tmp_path / "env"
    result = subprocess.run(
        ["bash", "-c", _step()["run"]],
        cwd=ROOT,
        env={
            **os.environ,
            "REPO_ROOT": str(ROOT),
            "RUNNER_WORKSPACE": str(work / "repo"),
            "RUNNER_TEMP": str(tmp_path / "job"),
            "GITHUB_ENV": str(env_file),
            "PLATFORM": platform,
            "PIP_CACHE_KEY": "test-key",
            "PIP_NO_CACHE_DIR": "",
            "PIP_RETRY_NO_CACHE": "0",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    rows = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    assert rows["PIP_CACHE_DIR"] == str(tmp_path / "job/pip-cache" / platform)
    assert blocked.read_text() == "unavailable persistent mount"


def test_key_and_entrypoint_include_the_shared_helper():
    assert ".github/scripts/lib/pip_cache.sh" in _step()["env"]["PIP_CACHE_KEY"]
    runner = (ROOT / ".github/scripts/set_env.sh").read_text()
    assert "ascend|iluvatar|kunlunxin|mthreads|thead)" in runner
    assert 'prepare_pip_cache "$PLATFORM"' in runner


@pytest.mark.parametrize("platform", RUNTIME_PLATFORMS)
def test_manual_wheel_workflow_uses_the_same_cache_helper(tmp_path, platform):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/build-wheel-common.yml").read_text()
    )
    steps = next(job["steps"] for job in workflow["jobs"].values() if "steps" in job)
    step = next(s for s in steps if s.get("name") == "Reset pip cache dir")
    assert step["env"]["CACHE_PLATFORM"] == "${{ inputs.setup_platform }}"
    temp = tmp_path / "job"
    temp.mkdir()
    (temp / "pip-cache").write_text("old blocker")
    env_file = tmp_path / "env"
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=ROOT,
        env={
            **os.environ,
            "CACHE_PLATFORM": platform,
            "RUNNER_TEMP": str(temp),
            "GITHUB_ENV": str(env_file),
            "PIP_NO_CACHE_DIR": "",
            "PIP_RETRY_NO_CACHE": "0",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    rows = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    assert rows["PIP_CACHE_DIR"] == str(temp / "pip-cache") + f"-uid-{os.geteuid()}"
    assert (temp / "pip-cache").read_text() == "old blocker"


@pytest.mark.parametrize("platform", RUNTIME_PLATFORMS)
def test_real_entrypoint_selects_cache_before_loading_runtime_hooks(tmp_path, platform):
    repo = tmp_path / "repo"
    scripts = repo / ".github/scripts"
    (scripts / "hooks").mkdir(parents=True)
    (scripts / "set_env.sh").write_text(
        (ROOT / ".github/scripts/set_env.sh").read_text()
    )
    (scripts / "lib").symlink_to(ROOT / ".github/scripts/lib", target_is_directory=True)
    (repo / ".github/version-pins.env").write_text(
        (ROOT / ".github/version-pins.env").read_text()
    )
    (scripts / f"hooks/set_env_{platform}.sh").write_text(
        'test -d "$PIP_CACHE_DIR"\n'
        'test "$PIP_RETRY_NO_CACHE" = 0\n'
        'test -z "$PIP_NO_CACHE_DIR"\n'
        'echo "hook-cache=$PIP_CACHE_DIR"\n'
    )
    blocked = tmp_path / "blocked"
    blocked.touch()
    env_file = tmp_path / "env"
    result = subprocess.run(
        ["bash", str(scripts / "set_env.sh"), "--platform", platform],
        env={
            **os.environ,
            "CI_STAGE": "build",
            "RUNNER_TEMP": str(tmp_path / "job"),
            "PIP_CACHE_DIR": str(blocked),
            "PIP_RETRY_NO_CACHE": "0",
            "PIP_NO_CACHE_DIR": "",
            "GITHUB_ENV": str(env_file),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    rows = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    assert f"hook-cache={rows['PIP_CACHE_DIR']}" in result.stdout
    assert rows["PIP_CACHE_DIR"] == str(blocked) + f"-uid-{os.geteuid()}"


@pytest.mark.parametrize("old_key,preserved", [("same-key", True), ("old-key", False)])
def test_legacy_runner_key_files_are_honored(tmp_path, old_key, preserved):
    cache = tmp_path / "persistent/ascend"
    cache.mkdir(parents=True)
    payload = cache / "warm-wheel"
    payload.write_text("download")
    Path(str(cache) + ".key").write_text(old_key + "\n")
    result, rows, _ = _run_shared_cache(tmp_path, "ascend", key="same-key")
    assert result.returncode == 0, result.stderr
    assert rows["PIP_CACHE_DIR"] == str(cache)
    assert payload.exists() == preserved
    assert (cache / ".torch-fl-ci-cache-key").read_text() == "same-key\n"
