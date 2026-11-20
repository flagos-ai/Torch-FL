# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run Kunlunxin provisioning with local SDK/package boundaries and real shells."""

import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / ".github/scripts/hooks/set_env_kunlunxin.sh"


def run_hook(
    tmp_path, *, prepared=True, pip_ready=True, fail="", stage="build", target="ci"
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".github").symlink_to(ROOT / ".github", target_is_directory=True)
    sdk = tmp_path / "xpu"
    for name in ("include/xpu/runtime.h", "include/cuda_runtime.h", "so/libcudart.so"):
        path = sdk / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    image = tmp_path / "image"
    xdnn = image / "lib/python3.10/site-packages/torch_xmlir"
    (xdnn / "include/xdnn_pytorch").mkdir(parents=True)
    (xdnn / "include/xdnn_pytorch/xdnn_pytorch.h").touch()
    (xdnn / "libxdnn_pytorch.so").touch()
    image_bin = image / "bin"
    image_bin.mkdir()
    ci_python = tmp_path / "ci-python"
    ci_python.write_text(
        r"""#!/bin/bash
printf 'python %s\n' "$*" >> "$CALL_LOG"
if [[ "$*" == *--check-installed* ]]; then
  if [[ "$3" == 'pip>=23.2' ]]; then
    [[ "$PIP_READY" == 1 || -f "$PIP_BOOTSTRAPPED" ]]; exit $?
  fi
  if grep -Fxq -- "installed $3" "$CALL_LOG"; then exit 0; fi
  if [[ "$3" == cmake* && "$FAIL_PHASE" == cmake-metadata ]]; then exit 1; fi
  exit "$PREPARED_STATUS"
fi
if [[ "$1 $2" == '-m pip' ]]; then
  [[ "$3" != check || "$FAIL_PHASE" != check ]]
  exit $?
fi
if [[ "$1" == '-' ]]; then
  input="$(cat)"
  if [[ "$input" == *'metadata.distribution("flagtree")'* ]]; then
    exit "$PREPARED_STATUS"
  fi
  if [[ "$input" == *'import triton'* && "$FAIL_PHASE" == backend ]]; then exit 1; fi
fi
if [[ $# == 0 ]]; then printf 'ci-python\n'; fi
"""
    )
    ci_python.chmod(0o755)
    vendor = image_bin / "python3.10"
    vendor.write_text(
        r"""#!/bin/bash
set -e
printf 'vendor %s\n' "$*" >> "$CALL_LOG"
if [[ "$1 $2" == '-m venv' ]]; then
  [[ "$FAIL_PHASE" != venv && "$FAIL_PHASE" != uv ]]
  mkdir -p "$4/bin"; cp "$CI_PYTHON" "$4/bin/python"; exit 0
fi
if [[ "$*" == *seed_runtime.py* ]]; then [[ "$FAIL_PHASE" != seed ]]; exit $?; fi
if [[ "$1 $2" == '-I -c' ]]; then
  if [[ "$3" == *sys.version_info* ]]; then [[ "$FAIL_PHASE" != python ]]; exit $?; fi
  printf '%s\n' "$VIRTUAL_ENV/lib/python3.10/site-packages"; exit 0
fi
if [[ "$1 $2" == '-I -' ]]; then
  exec "$REAL_PYTHON" -I -c 'import os,sys; sys.prefix=os.environ["IMAGE_PREFIX"]; sys.base_prefix=sys.prefix; sys.argv=["-",*sys.argv[1:]]; exec(sys.stdin.read())' "$3" "$4"
fi
exit 2
"""
    )
    vendor.chmod(0o755)
    uv = image_bin / "uv"
    uv.write_text(
        '#!/bin/bash\nprintf "uv %s\\n" "$*" >> "$CALL_LOG"\n'
        '[[ "$FAIL_PHASE" != uv ]] || exit 1\n'
        'target="${@: -1}"\nmkdir -p "$target/bin"\n'
        'cp "$CI_PYTHON" "$target/bin/python"\n'
    )
    uv.chmod(0o755)
    apt = image_bin / "apt-get"
    apt.write_text('#!/bin/sh\necho forbidden-apt >> "$CALL_LOG"\nexit 1\n')
    apt.chmod(0o755)
    startup = tmp_path / "bash-env"
    startup.write_text(
        f"export PATH={shlex.quote(str(image_bin))}:/usr/bin:/bin\n"
        "export PYTHONPATH=/opt/flagtree:/flagos/site-packages\n"
        'printf "startup\\n" >> "$CALL_LOG"\n'
    )
    env_file = tmp_path / "github-env"
    path_file = tmp_path / "github-path"
    calls = tmp_path / "calls"
    env = dict(
        os.environ,
        PATH=f"{image_bin}:/usr/bin:/bin",
        BASH_ENV=str(startup),
        VIRTUAL_ENV=str(image),
        IMAGE_PREFIX=str(image),
        REAL_PYTHON=sys.executable,
        REPO_ROOT=str(repo),
        CI_STAGE=stage,
        TORCH_FL_VENDOR_PYTHON=str(vendor),
        TORCH_FL_VENV_ROOT=str(tmp_path / target),
        XPU_ROOT=str(sdk),
        XDNN_ROOT="" if fail != "xdnn" else str(tmp_path / "missing"),
        GITHUB_ENV=str(env_file),
        GITHUB_PATH=str(path_file),
        CALL_LOG=str(calls),
        CI_PYTHON=str(ci_python),
        PREPARED_STATUS="0" if prepared else "1",
        PIP_READY="1" if pip_ready else "0",
        PIP_BOOTSTRAPPED=str(tmp_path / "pip-bootstrapped"),
        PIP_CACHE_DIR=str(tmp_path / "runner-pip-cache"),
        BOXING_LANE="1" if fail == "boxing" else "",
        FAIL_PHASE=fail,
        LD_LIBRARY_PATH="/host/driver",
        PIP_CONSTRAINT=str(ROOT / ".github/constraints.txt"),
    )
    if fail == "sdk":
        (sdk / "include/xpu/runtime.h").unlink()
    if fail == "cudart":
        (sdk / "so/libcudart.so").unlink()
    script = r"""
set -euo pipefail
source "$REPO_ROOT/.github/version-pins.env"
source "$REPO_ROOT/.github/scripts/lib/set_env_common.sh"
install_cpu_torch() { echo cpu-torch >> "$CALL_LOG"; [[ "$FAIL_PHASE" != cpu ]]; }
install_flagtree() { echo flagtree >> "$CALL_LOG"; [[ "$FAIL_PHASE" != flagtree ]]; }
install_flag_gems() { echo flag-gems >> "$CALL_LOG"; [[ "$FAIL_PHASE" != gems ]]; }
pip_retry() {
  printf 'pip %s\n' "$*" >> "$CALL_LOG"
  [[ "$FAIL_PHASE" != pip ]] || return 1
  if [[ "$*" == *'pip>=23.2'* ]]; then
    [[ "$FAIL_PHASE" != bootstrap ]] || return 1
    [[ "$FAIL_PHASE" == stale-pip ]] || touch "$PIP_BOOTSTRAPPED"
  fi
  printf 'installed %s\n' "${@: -1}" >> "$CALL_LOG"
}
build_flagos_inplace() {
  test "$FLAGOS_BUILD_BOXING" = 0
  test "$FLAGOS_BUILD_FLAGGEMS_CPP" = 0
  test "$FLAGOS_BUILD_VENDOR" = 1
  echo native-build >> "$CALL_LOG"
}
source "$REPO_ROOT/.github/scripts/hooks/set_env_kunlunxin.sh"
"""
    done = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )
    return done, calls.read_text().splitlines(), env, env_file, path_file


def test_runtime_reuse_keeps_the_next_shell_in_the_isolated_venv(tmp_path):
    done, calls, env, env_file, path_file = run_hook(tmp_path)
    assert done.returncode == 0, done.stderr
    assert calls.count("native-build") == 1
    assert "cpu-torch" not in calls and "flagtree" not in calls
    assert not any(row.startswith("pip ") for row in calls)
    assert not any("--flagcx-version" in row for row in calls)
    assert any("--report-environment" in row for row in calls)
    assert "forbidden-apt" not in calls
    rows = dict(row.split("=", 1) for row in env_file.read_text().splitlines())
    assert rows["BASH_ENV"] == "/dev/null"
    assert rows["PYTHONPATH"] == ""
    assert rows["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert "/host/driver" in rows["LD_LIBRARY_PATH"]
    assert rows["XDNN_ROOT"].startswith(env["IMAGE_PREFIX"])
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", "python"],
        env={**env, **rows, "PATH": path_file.read_text().strip() + ":" + rows["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "ci-python"
    assert Path(env["CALL_LOG"]).read_text().splitlines().count("startup") == 1


def test_missing_packages_install_declared_python_build_requirements(tmp_path):
    done, calls, *_ = run_hook(tmp_path, prepared=False)
    assert done.returncode == 0, done.stderr
    assert "cpu-torch" in calls and "flagtree" in calls
    assert any(row.startswith("pip ") and row.endswith("numpy") for row in calls)
    assert any(
        row.startswith("pip ") and row.endswith("cmake>=3.19,<4") for row in calls
    )
    assert "native-build" in calls


def test_old_pip_is_bootstrapped_and_rechecked_before_torch_resolution(tmp_path):
    done, calls, *_ = run_hook(tmp_path, prepared=False, pip_ready=False)
    assert done.returncode == 0, done.stderr
    bootstrap = next(
        i
        for i, row in enumerate(calls)
        if row.startswith("pip ") and "pip>=23.2" in row
    )
    checks = [i for i, row in enumerate(calls) if "--check-installed pip>=23.2" in row]
    assert checks[0] < bootstrap < checks[1] < calls.index("cpu-torch")
    assert "--only-binary=:all:" in calls[bootstrap]
    assert "--no-cache-dir" not in calls[bootstrap]
    assert "download.pytorch.org" not in calls[bootstrap]
    assert any(row == "python -m pip --version" for row in calls)


@pytest.mark.parametrize("fail", ["bootstrap", "stale-pip"])
def test_failed_or_ineffective_pip_bootstrap_stops_before_torch(tmp_path, fail):
    done, calls, *_ = run_hook(tmp_path, prepared=False, pip_ready=False, fail=fail)
    assert done.returncode != 0
    assert "cpu-torch" not in calls
    assert "native-build" not in calls


def test_system_cmake_cannot_replace_the_declared_python_distribution(tmp_path):
    done, calls, *_ = run_hook(tmp_path, fail="cmake-metadata")
    assert done.returncode == 0, done.stderr
    installs = [row for row in calls if row.startswith("pip ")]
    assert len(installs) == 1 and installs[0].endswith("cmake>=3.19,<4")
    assert any("--check-installed cmake>=3.19,<4" in row for row in calls)
    assert "cpu-torch" not in calls and "flagtree" not in calls
    assert "forbidden-apt" not in calls
    assert "native-build" in calls


def test_venv_failure_uses_the_runtime_uv_without_apt(tmp_path):
    done, calls, *_ = run_hook(tmp_path, fail="venv")
    assert done.returncode == 0, done.stderr
    assert any(row.startswith("uv venv --clear --seed --python ") for row in calls)
    assert "forbidden-apt" not in calls


@pytest.mark.parametrize(
    "fail",
    ["python", "boxing", "sdk", "cudart", "xdnn", "seed", "uv", "check", "backend"],
)
def test_invalid_runtime_or_dependency_state_stops_before_build(tmp_path, fail):
    done, calls, *_ = run_hook(tmp_path, fail=fail)
    assert done.returncode != 0
    assert "native-build" not in calls
    assert "forbidden-apt" not in calls


@pytest.mark.parametrize("fail", ["cpu", "pip", "flagtree", "gems"])
def test_package_fallback_failure_stops_before_build(tmp_path, fail):
    done, calls, *_ = run_hook(tmp_path, prepared=False, fail=fail)
    assert done.returncode != 0
    assert "native-build" not in calls


@pytest.mark.parametrize("target", ["image", "image/subdir", ".", "repo"])
def test_venv_override_cannot_clear_runtime_or_checkout(tmp_path, target):
    done, calls, env, *_ = run_hook(tmp_path, target=target)
    assert done.returncode != 0
    assert not any("-m venv" in row for row in calls)
    assert (
        Path(env["IMAGE_PREFIX"])
        / "lib/python3.10/site-packages/torch_xmlir/libxdnn_pytorch.so"
    ).exists()


def test_integration_setup_does_not_add_another_native_build(tmp_path):
    done, calls, *_ = run_hook(tmp_path, stage="integration")
    assert done.returncode == 0, done.stderr
    assert "native-build" not in calls


def test_lane_uses_pinned_runtime_and_prepares_tools_in_the_hook():
    config = yaml.safe_load((ROOT / ".github/configs/kunlunxin.yml").read_text())
    assert config["ci_image"].endswith(
        ":2.3.0@sha256:cc0deb9c09cd49977e754892e85f4414835d142379949cc2d66261f63b47959d"
    )
    assert config["setup_stage"] == "build"
    assert config["integration_build_tools"] == []
    assert config["integration_test_dependencies"] == []


@pytest.mark.parametrize(
    "state", ["local", "wrong-version", "shadowed", "no-xpu", "external"]
)
def test_flagtree_reuse_probe_checks_real_metadata_and_import_origin(tmp_path, state):
    """Run the actual metadata gate, including a matching external provider."""
    prefix = tmp_path / "ci"
    site = prefix / "site-packages"
    site.mkdir(parents=True)
    provider = site if state != "external" else tmp_path / "external"
    compiler = provider / "triton"
    (compiler / "backends/xpu").mkdir(parents=True)
    (compiler / "__init__.py").touch()
    info = provider / "flagtree-0.7.0.dist-info"
    info.mkdir()
    version = "0.6.0+xpu3.6" if state == "wrong-version" else "0.7.0+xpu3.6"
    (info / "METADATA").write_text(f"Name: flagtree\nVersion: {version}\n")
    if state == "no-xpu":
        (compiler / "backends/xpu").rmdir()
    paths = [str(provider)]
    if state == "shadowed":
        shadow = tmp_path / "shadow"
        (shadow / "triton").mkdir(parents=True)
        (shadow / "triton/__init__.py").touch()
        paths.insert(0, str(shadow))
    probe = (
        HOOK.read_text()
        .split('if "$VENV_PYTHON" - "$FLAGTREE_VERSION" <<\'PY\'\n', 1)[1]
        .split("\nPY", 1)[0]
    )
    code = f"import sys; sys.prefix={str(prefix)!r}; sys.path[:0]={paths!r}\n" + probe
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, "0.7.0+xpu3.6"],
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (state == "local"), result.stderr
