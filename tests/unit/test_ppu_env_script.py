# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise both PPU lanes without a device, network, or vendor Python import."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / ".github/scripts/hooks/set_env_thead.sh"


def run_hook(
    tmp_path, *, boxing=False, stage="integration", prepared=True, fail="", target="ci"
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".github").symlink_to(ROOT / ".github", target_is_directory=True)
    (repo / "torch_fl/configs").mkdir(parents=True)
    (repo / "torch_fl/configs/backends_thead_boxing.conf").touch()
    sdk = tmp_path / "sdk"
    (sdk / "CUDA_SDK/lib64").mkdir(parents=True)
    (sdk / "CUDA_SDK/lib64/libcudart.so").touch()
    image = tmp_path / "image"
    vendor_root = image / "site-packages/torch"
    (vendor_root / "lib").mkdir(parents=True)
    for name in (
        "libc10",
        "libtorch_cpu",
        "libtorch",
        "libtorch_global_deps",
        "libtorch_python",
        "libc10_cuda",
        "libtorch_cuda",
    ):
        (vendor_root / f"lib/{name}.so").touch()
    image_bin = image / "bin"
    image_bin.mkdir()
    ci_python = tmp_path / "ci-python"
    ci_python.write_text(r"""#!/bin/bash
printf 'python %s\n' "$*" >> "$CALL_LOG"
if [[ "$*" == *--check-installed* ]]; then
  [[ "$PREPARED_STATUS" == 0 ]] || grep -Fxq -- "$3" "$PREPARED_FILE"
  exit $?
fi
if [[ "$1 $2" == '-m pip' ]]; then
  [[ "$FAIL_PHASE" != uninstall && "$FAIL_PHASE" != check ]]; exit $?
fi
if [[ "$1" == '-' ]]; then
  input="$(cat)"
  if [[ "$input" == *'sys.argv[1].removesuffix'* ]]; then
    [[ "$PREPARED_STATUS" == 0 || -f "$CPU_INSTALLED" ]]; exit $?
  fi
  if [[ "$input" == *'metadata.distribution("flagtree")'* ]]; then
    exit "$PREPARED_STATUS"
  fi
  if [[ "$input" == *'print(Path(torch.__file__).resolve().parent)'* ]]; then
    printf '%s\n' "$CI_TORCH_ROOT"
  fi
  if [[ "$input" == *'root = Path(sys.argv[1]).resolve()'* ]]; then
    [[ "$FAIL_PHASE" != isolation ]]; exit $?
  fi
  if [[ "$input" == *'import torch_fl'* ]]; then
    [[ "$FAIL_PHASE" != backend ]]; exit $?
  fi
fi
if [[ $# == 0 ]]; then printf 'ci-python\n'; fi
""")
    ci_python.chmod(0o755)
    vendor = image_bin / "python"
    vendor.write_text(r"""#!/bin/bash
set -e
printf 'vendor %s\n' "$*" >> "$CALL_LOG"
if [[ "$1 $2" == '-m venv' ]]; then
  [[ "$FAIL_PHASE" != venv && "$FAIL_PHASE" != uv ]] || exit 1
  mkdir -p "$4/bin"; cp "$CI_PYTHON" "$4/bin/python"; exit 0
fi
if [[ "$*" == *seed_runtime.py* ]]; then
  [[ "$FAIL_PHASE" != seed ]]; exit $?
fi
if [[ "$1 $2" == '-I -c' ]]; then exit 0; fi
input="$(cat)"
if [[ "$input" == *'print(json.dumps'* ]]; then
  printf '%s\n' "$VENDOR_INFO_JSON"
elif [[ "$input" == *'info = json.loads'* ]]; then
  printf '%s' "$input" | "$REAL_PYTHON" -
elif [[ "$input" == *'target, repo ='* ]]; then
  printf '%s' "$input" | "$REAL_PYTHON" -I -c 'import os,sys; sys.prefix=os.environ["IMAGE_PREFIX"]; sys.base_prefix=sys.prefix; sys.version_info=(3,12,0); sys.argv=["-",*sys.argv[1:]]; exec(sys.stdin.read())' "$3" "$4"
else
  exit 2
fi
""")
    vendor.chmod(0o755)
    uv = image_bin / "uv"
    uv.write_text(
        '#!/bin/bash\nprintf "uv %s\\n" "$*" >> "$CALL_LOG"\n'
        '[[ "$FAIL_PHASE" != uv ]] || exit 1\ntarget="${@: -1}"\n'
        'mkdir -p "$target/bin"; cp "$CI_PYTHON" "$target/bin/python"\n'
    )
    uv.chmod(0o755)
    smi = image_bin / "ppu-smi"
    smi.write_text('#!/bin/sh\necho ppu-smi >> "$CALL_LOG"\n')
    smi.chmod(0o755)
    startup = tmp_path / "bash-env"
    startup.write_text(
        f"export PATH={shlex.quote(str(image_bin))}:/usr/bin:/bin\n"
        "export PYTHONPATH=/opt/flagtree:/flagos/site-packages\n"
        'echo startup >> "$CALL_LOG"\n'
    )
    calls, env_file, path_file = (tmp_path / name for name in ("calls", "env", "path"))
    prepared_file = tmp_path / "prepared"
    prepared_file.touch()
    hook = tmp_path / "hook.sh"
    hook.write_text(
        HOOK.read_text().replace(
            "/dev/alixpu_ppu0",
            "/dev/null" if fail != "device" else "/dev/no-ppu-device",
        )
    )
    if fail == "sdk":
        (sdk / "CUDA_SDK/lib64/libcudart.so").unlink()
    if fail == "vendor-lib":
        (vendor_root / "lib/libtorch_cuda.so").unlink()
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
        PPU_SDK=str(sdk),
        PPU_VERSION="v2.1.0",
        BOXING_LANE="1" if boxing else "",
        GITHUB_ENV=str(env_file),
        GITHUB_PATH=str(path_file),
        CALL_LOG=str(calls),
        CI_PYTHON=str(ci_python),
        CI_TORCH_ROOT=str(tmp_path / "ci/torch"),
        PREPARED_STATUS="0" if prepared else "1",
        PREPARED_FILE=str(prepared_file),
        CPU_INSTALLED=str(tmp_path / "cpu-installed"),
        FAIL_PHASE=fail,
        VENDOR_INFO_JSON=json.dumps(
            dict(
                version="2.10.0+v0.1.0.ppu2.1.1",
                base_version="2.10.0",
                cuda="13.0",
                root=str(vendor_root),
                site=str(vendor_root.parent),
                python="3.12.3",
            )
        ),
        LD_LIBRARY_PATH=f"{vendor_root}/lib:/host/driver",
        FLAGOS_BUILD_BOXING="",
        FLAGOS_BACKEND_CONFIG="",
    )
    script = r"""
set -euo pipefail
source "$REPO_ROOT/.github/version-pins.env"
source "$REPO_ROOT/.github/scripts/lib/set_env_common.sh"
install_cpu_torch() { echo cpu-torch >> "$CALL_LOG"; [[ "$FAIL_PHASE" != cpu ]] || return 1; touch "$CPU_INSTALLED"; }
install_flagtree() { echo flagtree >> "$CALL_LOG"; [[ "$FAIL_PHASE" != tree ]]; }
install_flag_gems() { echo flag-gems >> "$CALL_LOG"; [[ "$FAIL_PHASE" != gems ]]; }
pip_retry() {
  printf 'pip %s\n' "$*" >> "$CALL_LOG"
  [[ "$FAIL_PHASE" != pip ]] || return 1
  [[ "$FAIL_PHASE" == stale-package ]] || printf '%s\n' "${@: -1}" >> "$PREPARED_FILE"
}
build_flagos_inplace() {
  test "$FLAGOS_ACCELERATOR" = ppu
  test "$FLAGOS_BUILD_FLAGGEMS_CPP" = 0
  echo native-build >> "$CALL_LOG"
}
bash() { test "$1" = scripts/vendor/bundle_thead_libtorch.sh; echo bundle >> "$CALL_LOG"; }
source "$1"
"""
    done = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script, "test", str(hook)],
        env=env,
        capture_output=True,
        text=True,
    )
    return done, calls.read_text().splitlines(), env, env_file, path_file


@pytest.mark.parametrize("boxing", [False, True])
def test_runtime_reuse_preserves_lane_and_next_shell_isolation(tmp_path, boxing):
    done, calls, env, env_file, path_file = run_hook(tmp_path, boxing=boxing)
    assert done.returncode == 0, done.stderr
    assert calls.count("native-build") == 1
    assert ("bundle" in calls) == boxing
    assert "cpu-torch" not in calls and "flagtree" not in calls
    assert not any(row.startswith("pip ") for row in calls)
    assert any("--requirements-file" in row for row in calls)
    assert not any("--flagcx-version" in row for row in calls)
    rows = dict(row.split("=", 1) for row in env_file.read_text().splitlines())
    assert rows["BASH_ENV"] == "/dev/null"
    assert rows["PYTHONPATH"] == ""
    assert rows["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert rows["FLAGOS_SDK_VERSION"] == "2.1.0"
    assert rows["FLAGOS_VENDOR_TORCH_VERSION"] == (
        "2.10.0+v0.1.0.ppu2.1.1" if boxing else ""
    )
    assert "/host/driver" in rows["LD_LIBRARY_PATH"]
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", "python"],
        env={**env, **rows, "PATH": path_file.read_text().strip() + ":" + rows["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "ci-python"
    assert Path(env["CALL_LOG"]).read_text().splitlines().count("startup") == 1


def test_missing_packages_use_published_wheels_and_conflicting_compiler_cleanup(
    tmp_path,
):
    done, calls, *_ = run_hook(tmp_path, prepared=False)
    assert done.returncode == 0, done.stderr
    assert "cpu-torch" in calls and "flagtree" in calls
    assert "python -m pip uninstall -y flagtree triton triton_kernels" in calls
    assert any(
        row.startswith("pip ") and row.endswith("sqlalchemy==2.0.48") for row in calls
    )


def test_build_stage_preserves_existing_compile_count_without_model_packages(tmp_path):
    done, calls, *_ = run_hook(tmp_path, stage="build")
    assert done.returncode == 0, done.stderr
    assert calls.count("native-build") == 1
    assert "flag-gems" not in calls and "flagtree" not in calls
    assert not any("transformers" in row for row in calls)


@pytest.mark.parametrize(
    "fail",
    [
        "sdk",
        "vendor-lib",
        "device",
        "seed",
        "isolation",
        "pip",
        "stale-package",
        "cpu",
        "uninstall",
        "check",
        "tree",
        "gems",
        "uv",
    ],
)
def test_invalid_environment_or_failed_provisioning_stops_before_build(tmp_path, fail):
    done, calls, *_ = run_hook(tmp_path, prepared=False, fail=fail)
    assert done.returncode != 0
    assert "native-build" not in calls


@pytest.mark.parametrize("target", ["image", "image/child", "."])
def test_clear_never_overwrites_runtime_or_checkout(tmp_path, target):
    done, calls, *_ = run_hook(tmp_path, target=target)
    assert done.returncode != 0
    assert not any(row.startswith("vendor -m venv") for row in calls)


def test_image_uv_fallback_does_not_install_into_runtime_python(tmp_path):
    done, calls, *_ = run_hook(tmp_path, fail="venv")
    assert done.returncode == 0, done.stderr
    assert any(row.startswith("uv ") for row in calls)
    assert not any(row.startswith("vendor -m pip") for row in calls)


@pytest.mark.parametrize(
    "state", ["local", "external", "shadowed", "wrong-version", "no-ppu"]
)
def test_compiler_reuse_validates_version_origin_and_ppu_backend(tmp_path, state):
    prefix = tmp_path / "ci"
    provider = (
        prefix / "site-packages" if state != "external" else tmp_path / "external"
    )
    compiler = provider / "triton"
    (compiler / "backends/ppu").mkdir(parents=True)
    (compiler / "__init__.py").touch()
    info = provider / "flagtree-0.7.0.dist-info"
    info.mkdir()
    version = "0.6.1+ppu3.6" if state == "wrong-version" else "0.7.0+ppu3.6"
    (info / "METADATA").write_text(f"Name: flagtree\nVersion: {version}\n")
    if state == "no-ppu":
        (compiler / "backends/ppu").rmdir()
    paths = [str(provider)]
    if state == "shadowed":
        shadow = tmp_path / "shadow"
        (shadow / "triton").mkdir(parents=True)
        (shadow / "triton/__init__.py").touch()
        paths.insert(0, str(shadow))
    probe = (
        HOOK.read_text()
        .split("ppu_runtime_flagtree_is_usable() {", 1)[1]
        .split("<<'PYCODE'\n", 1)[1]
        .split("\nPYCODE", 1)[0]
    )
    code = f"import sys; sys.prefix={str(prefix)!r}; sys.path[:0]={paths!r}\n" + probe
    done = subprocess.run(
        [sys.executable, "-I", "-c", code, "0.7.0+ppu3.6"],
        capture_output=True,
        text=True,
    )
    assert (done.returncode == 0) == (state == "local"), done.stderr


def test_pinned_runtime_retains_device_access_and_both_lane_contracts():
    config = yaml.safe_load((ROOT / ".github/configs/thead.yml").read_text())
    assert (
        config["ci_image"]
        == "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0:2.3.0@sha256:7837256c2dab931c762f86713a37d3fbecc902792392d67731c982ecda632cef"
    )
    assert config["runner_labels"] == ["flagcicd-810e"]
    assert config["boxing_lane"] is True
    assert "--privileged" not in config["container_options"]
    assert "--ipc=host" not in config["container_options"]
    assert "--device=/dev/alixpu_ppu15" in config["container_options"]
    assert config["environment_check"]["cuda_version"] is None
    assert config["boxing_environment_check"]["cuda_version"] == "13.0"


@pytest.mark.parametrize("state", ["isolated", "system-site", "wrong-prefix"])
def test_venv_probe_checks_real_interpreter_and_site_policy(tmp_path, state):
    root = tmp_path / "venv"
    command = [sys.executable, "-m", "venv", "--without-pip", str(root)]
    if state == "system-site":
        command.insert(-1, "--system-site-packages")
    subprocess.run(command, check=True, capture_output=True, text=True)
    probe = (
        HOOK.read_text()
        .split('"$VENV_PYTHON" - "$VENV_ROOT" <<\'PY\'\n', 1)[1]
        .split("\nPY", 1)[0]
    )
    expected = root if state != "wrong-prefix" else tmp_path / "other"
    done = subprocess.run(
        [str(root / "bin/python"), "-I", "-c", probe, str(expected)],
        capture_output=True,
        text=True,
    )
    assert (done.returncode == 0) == (state == "isolated"), done.stderr
