# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise warm-image reuse and job-local compiler/operator replacements."""

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / ".github/scripts/lib/iluvatar_packages.sh"


def test_hook_clears_legacy_vendor_core_for_build_and_later_steps(tmp_path):
    """Reuse the published image without carrying its vendor-core default."""
    image = tmp_path / "image"
    venv = image / "torch-fl-iluvatar-venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin/python"
    python.write_text("#!/bin/bash\nexit 0\n")
    python.chmod(0o755)
    (image / "torch-fl-ci").mkdir()
    (image / "torch-fl-ci/verify.py").touch()
    repo = tmp_path / "repo"
    helpers = repo / ".github/scripts/lib"
    helpers.mkdir(parents=True)
    (helpers / "iluvatar_packages.sh").write_text(
        "prepare_iluvatar_packages() { :; }\n"
    )
    hook = tmp_path / "hook.sh"
    hook.write_text(
        (ROOT / ".github/scripts/hooks/set_env_iluvatar.sh")
        .read_text()
        .replace("/opt/torch-fl-", str(image / "torch-fl-"))
    )
    github_env = tmp_path / "github-env"
    script = r"""
set -euo pipefail
source "$1/.github/scripts/lib/set_env_common.sh"
build_flagos_inplace() {
  test -z "$FLAGOS_VENDOR_TORCH_LIB"
  test -z "$FLAGOS_VENDOR_TORCH_VERSION"
  echo CPU-core-build
}
source "$2"
"""
    done = subprocess.run(
        ["bash", "-c", script, "test", str(ROOT), str(hook)],
        env=dict(
            os.environ,
            REPO_ROOT=str(repo),
            CI_STAGE="build",
            GITHUB_ENV=str(github_env),
            GITHUB_PATH="",
            FLAGOS_VENDOR_TORCH_LIB="/legacy/vendor/torch/lib",
            FLAGOS_VENDOR_TORCH_VERSION="vendor-version",
            FLAGTREE_VERSION_iluvatar="test-tree",
            FLAGTREE_INDEX_URL_DEFAULT="https://example.invalid/tree",
            FLAGGEMS_VERSION_DEFAULT="test-gems",
            FLAGOS_WHEEL_ROOT_DEFAULT="https://example.invalid",
            PIP_CONSTRAINT="checkout-constraints",
            LIBRARY_PATH="",
            LD_LIBRARY_PATH="",
            CPATH="",
        ),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "CPU-core-build" in done.stdout
    rows = github_env.read_text().splitlines()
    assert "FLAGOS_VENDOR_TORCH_LIB=" in rows
    assert "FLAGOS_VENDOR_TORCH_VERSION=" in rows


def provision(tmp_path, *, tree_matches=True, gems_matches=True, failure=""):
    log = tmp_path / "calls"
    python = tmp_path / "python"
    python.write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == - ]]; then exit "$TREE_STATUS"; fi\n'
        'printf "%s\\n" "$*" >> "$CALL_LOG"\n'
        'if [[ "$*" == *"$FAIL_COMMAND"* && -n "$FAIL_COMMAND" ]]; then exit 47; fi\n'
    )
    python.chmod(0o755)
    env = dict(
        os.environ,
        VENV_PYTHON=str(python),
        CALL_LOG=str(log),
        TREE_STATUS="0" if tree_matches else "1",
        GEMS_STATUS="0" if gems_matches else "1",
        FAIL_COMMAND=failure,
        FLAGTREE_VERSION="new+iluvatar",
        FLAGGEMS_VERSION="new-gems",
        FLAGTREE_INDEX_URL="https://example.invalid/tree",
        FLAGGEMS_INDEX_URL="https://example.invalid/gems",
    )
    script = r"""
set -euo pipefail
source "$1"
prefer_direct_route() { printf 'route %s\n' "$1" >> "$CALL_LOG"; }
install_flagtree() { printf 'tree %s\n' "$FLAGTREE_VERSION" >> "$CALL_LOG"; }
flag_gems_installed() { return "$GEMS_STATUS"; }
install_flag_gems() {
  if ! flag_gems_installed; then
    printf 'gems %s\n' "$FLAGGEMS_VERSION" >> "$CALL_LOG"
  fi
}
prepare_iluvatar_packages
"""
    done = subprocess.run(
        ["bash", "-c", script, "test", str(HELPER)],
        env=env,
        text=True,
        capture_output=True,
    )
    return done, log.read_text().splitlines() if log.exists() else []


def test_matching_baseline_needs_no_download_or_install(tmp_path):
    done, calls = provision(tmp_path)
    assert done.returncode == 0, done.stderr
    assert calls == ["-m pip check"]


def test_changed_flagtree_replaces_the_complete_triton_provider(tmp_path):
    done, calls = provision(tmp_path, tree_matches=False)
    assert done.returncode == 0, done.stderr
    assert calls == [
        "-m pip uninstall -y flagtree triton triton_kernels",
        "route https://example.invalid/tree",
        "tree new+iluvatar",
        "-m pip check",
    ]


def test_changed_gems_preserves_the_compiler_and_cpu_torch(tmp_path):
    done, calls = provision(tmp_path, gems_matches=False)
    assert done.returncode == 0, done.stderr
    assert calls == [
        "route https://example.invalid/gems",
        "gems new-gems",
        "-m pip check",
    ]


@pytest.mark.parametrize("failure", ["uninstall", "check"])
def test_failed_replacement_or_dependency_check_stops_provisioning(tmp_path, failure):
    done, calls = provision(tmp_path, tree_matches=False, failure=failure)
    assert done.returncode == 47, done.stderr
    if failure == "uninstall":
        assert len(calls) == 1
