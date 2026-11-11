# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise warm-image reuse and job-local compiler/operator replacements."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / ".github/scripts/lib/iluvatar_packages.sh"


def run_hook(tmp_path, *, prepared=True, fail="", boxing=False, cached_occupancy=False):
    """Run the real hook with local package/build boundaries and a fake SDK."""
    import shlex

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".github").symlink_to(ROOT / ".github", target_is_directory=True)
    if cached_occupancy:
        (repo / "build").mkdir()
        (repo / "build/CMakeCache.txt").write_text(
            "CUDA_OCCUPANCY_INCLUDE_DIR:PATH=/image/cuda/include\n"
            "UNRELATED_SETTING:STRING=retained\n"
        )
    sdk = tmp_path / "corex"
    (sdk / "bin").mkdir(parents=True)
    (sdk / "lib").mkdir()
    (sdk / "lib64").mkdir()
    (sdk / "include").mkdir()
    tool = sdk / "bin/ixsmi"
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    image_bin = tmp_path / "image/bin"
    image_bin.mkdir(parents=True)
    ci_python = tmp_path / "ci-python"
    ci_python.write_text(
        "#!/bin/bash\n"
        'if [[ "$*" == *--check-installed* ]]; then\n'
        '  printf "check %s\\n" "$3" >> "$CALL_LOG"\n'
        '  exit "$PREPARED_STATUS"\n'
        "fi\n"
        'printf "python %s\\n" "$*" >> "$CALL_LOG"\n'
        'if [[ $# == 0 ]]; then printf "ci-python\\n"; fi\n'
    )
    ci_python.chmod(0o755)
    vendor = image_bin / "python3.12"
    vendor.write_text(
        "#!/bin/bash\nset -e\n"
        'if [[ "$1 $2" == "-m venv" ]]; then\n'
        '  mkdir -p "$4/bin"\n'
        '  cp "$CI_PYTHON" "$4/bin/python"\n'
        "  exit 0\n"
        "fi\n"
        'if [[ "$*" == *seed_runtime.py* ]]; then\n'
        '  printf "seed %s\\n" "$*" >> "$CALL_LOG"\n'
        'elif [[ "$1 $2" == "-I -" ]]; then\n'
        '  input="$(cat)"\n'
        '  if [[ "$input" == *metadata.distribution* ]]; then\n'
        '    printf "%s\\n" "$IMAGE_TORCH"\n'
        "  fi\n"
        "fi\n"
    )
    vendor.chmod(0o755)
    startup = tmp_path / "bash-env"
    startup.write_text(
        f"export PATH={shlex.quote(str(image_bin))}:/usr/bin:/bin\n"
        "export PYTHONPATH=/opt/flagtree:/image/site-packages\n"
        'printf "startup\\n" >> "$CALL_LOG"\n'
    )
    calls = tmp_path / "calls"
    env_file = tmp_path / "github-env"
    path_file = tmp_path / "github-path"
    image_torch = "/image/vendor/torch"
    env = dict(
        os.environ,
        PATH=f"{image_bin}:/usr/bin:/bin",
        BASH_ENV=str(startup),
        REPO_ROOT=str(repo),
        CI_STAGE="integration",
        TORCH_FL_VENDOR_PYTHON=str(vendor),
        TORCH_FL_VENV_ROOT=str(tmp_path / "ci"),
        COREX_PATH=str(sdk),
        COREX_ROOT=str(sdk),
        GITHUB_ENV=str(env_file),
        GITHUB_PATH=str(path_file),
        CALL_LOG=str(calls),
        CI_PYTHON=str(ci_python),
        IMAGE_TORCH=image_torch,
        PREPARED_STATUS="0" if prepared else "1",
        BOXING_LANE="1" if boxing else "",
        PIP_CONSTRAINT=str(ROOT / ".github/constraints.txt"),
        FAIL_PHASE=fail,
        FLAGOS_VENDOR_TORCH_LIB=f"{image_torch}/lib",
        FLAGOS_VENDOR_TORCH_VERSION="2.10.0+corex",
        CPATH=f"{image_torch}/include:/system/include",
        LIBRARY_PATH=f"{image_torch}/lib:/system/lib",
        LD_LIBRARY_PATH=f"{image_torch}/lib:/host/driver",
        CMAKE_PREFIX_PATH=f"{image_torch}/share/cmake:/system/cmake",
    )
    script = r"""
set -euo pipefail
source "$REPO_ROOT/.github/version-pins.env"
source "$REPO_ROOT/.github/scripts/lib/set_env_common.sh"
install_cpu_torch() {
  echo cpu-torch >> "$CALL_LOG"
  [[ "$FAIL_PHASE" != cpu ]]
}
pip_retry() {
  printf 'pip %s\n' "$*" >> "$CALL_LOG"
  [[ "$FAIL_PHASE" != pip ]]
}
prepare_iluvatar_packages() { :; }
build_flagos_inplace() {
  test -z "$FLAGOS_VENDOR_TORCH_LIB"
  test -z "$FLAGOS_VENDOR_TORCH_VERSION"
  test "$FLAGOS_BUILD_BOXING" = 0
  echo CPU-core-build >> "$CALL_LOG"
}
source() {
  if [[ "$1" == */iluvatar_packages.sh ]]; then return 0; fi
  builtin source "$@"
}
source "$REPO_ROOT/.github/scripts/hooks/set_env_iluvatar.sh"
"""
    done = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )
    return done, calls.read_text().splitlines(), env, env_file, path_file


def test_hook_reuses_packages_and_persists_cpu_isolation(tmp_path):
    done, calls, env, env_file, path_file = run_hook(tmp_path)
    assert done.returncode == 0, done.stderr
    assert "CPU-core-build" in calls
    assert not any(row == "cpu-torch" or row.startswith("pip ") for row in calls)
    assert any(row.startswith("seed ") for row in calls)
    assert not any("--flagcx-version" in row for row in calls)
    rows = dict(row.split("=", 1) for row in env_file.read_text().splitlines())
    assert rows["FLAGOS_VENDOR_TORCH_LIB"] == ""
    assert rows["FLAGOS_VENDOR_TORCH_VERSION"] == ""
    assert rows["FLAGOS_BUILD_BOXING"] == "0"
    assert rows["BASH_ENV"] == "/dev/null"
    assert rows["PYTHONPATH"] == ""
    for name in ("CPATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "CMAKE_PREFIX_PATH"):
        assert env["IMAGE_TORCH"] not in rows[name]
    next_env = {**env, **rows}
    next_env["PATH"] = path_file.read_text().replace("\n", os.pathsep) + rows["PATH"]
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", "python"],
        env=next_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "ci-python"
    assert Path(env["CALL_LOG"]).read_text().splitlines().count("startup") == 1


def test_missing_runtime_packages_use_cpu_torch_and_pip_fallback(tmp_path):
    done, calls, *_ = run_hook(tmp_path, prepared=False)
    assert done.returncode == 0, done.stderr
    assert "cpu-torch" in calls
    assert any(row.endswith("numpy<2") and row.startswith("pip ") for row in calls)
    assert "CPU-core-build" in calls


@pytest.mark.parametrize("cached_occupancy", [False, True])
def test_new_image_keeps_the_previous_optional_compile_path(tmp_path, cached_occupancy):
    """A newly discoverable header must not change compiler paths or features."""
    if not shutil.which("cmake") or not shutil.which("c++"):
        pytest.skip("CMake and a host C++ compiler are required")
    done, _, env, *_ = run_hook(tmp_path, cached_occupancy=cached_occupancy)
    assert done.returncode == 0, done.stderr
    repo = Path(env["REPO_ROOT"])
    cuda = tmp_path / "cuda/include"
    cuda.mkdir(parents=True)
    (cuda / "cuda_occupancy.h").write_text('#error "new occupancy path enabled"\n')
    (cuda / "cublas_v2.h").write_text('#error "foreign CUDA header selected"\n')
    sdk = Path(env["COREX_PATH"]) / "include"
    (sdk / "cublas_v2.h").write_text("typedef int ixblasDataType_t;\n")
    (repo / "probe.cc").write_text(
        "#include <cublas_v2.h>\nixblasDataType_t value = 0;\n"
    )
    occupancy = (
        (ROOT / "csrc/CMakeLists.txt")
        .read_text()
        .split("set(CUDA_OCCUPANCY_SEARCH_PATHS", 1)[1]
        .split("# ROCtracer headers", 1)[0]
    )
    (repo / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.19)\nproject(probe LANGUAGES CXX)\n"
        "set(CMAKE_CXX_STANDARD 17)\nset(CMAKE_EXPORT_COMPILE_COMMANDS ON)\n"
        "set(LIBRARY_NAME probe)\n"
        f'add_library(probe OBJECT probe.cc "{ROOT}/csrc/profiler/cupti_device_tracer.cc")\n'
        "add_library(cuda_runtime_compat INTERFACE IMPORTED)\n"
        f'set_target_properties(cuda_runtime_compat PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "{sdk}")\n'
        "target_link_libraries(probe PRIVATE cuda_runtime_compat)\n"
        f'target_include_directories(probe PRIVATE "{ROOT}/csrc/include")\n'
        "set(CUDA_OCCUPANCY_SEARCH_PATHS"
        + occupancy
        + "\nif(CUDA_OCCUPANCY_INCLUDE_DIR)\n"
        'message(FATAL_ERROR "Image migration enabled occupancy")\nendif()\n'
    )
    # Setup builds in-place, then the wheel step configures the same directory.
    for _ in range(2):
        configured = subprocess.run(
            ["cmake", "-S", str(repo), "-B", str(repo / "build")],
            env={**os.environ, "CUDA_HOME": str(cuda.parent)},
            capture_output=True,
            text=True,
        )
        assert configured.returncode == 0, configured.stdout + configured.stderr
        assert "cuda_occupancy.h NOT found" in configured.stdout
        built = subprocess.run(
            ["cmake", "--build", str(repo / "build")], capture_output=True, text=True
        )
        assert built.returncode == 0, built.stdout + built.stderr
    commands = (repo / "build/compile_commands.json").read_text()
    assert "FLAGOS_HAVE_CUDA_OCCUPANCY" not in commands
    assert str(cuda) not in commands
    if cached_occupancy:
        assert (
            "UNRELATED_SETTING:STRING=retained"
            in (repo / "build/CMakeCache.txt").read_text()
        )
    unseeded = subprocess.run(
        ["cmake", "-S", str(repo), "-B", str(repo / "unseeded-build")],
        env={**os.environ, "CUDA_HOME": str(cuda.parent)},
        capture_output=True,
        text=True,
    )
    assert unseeded.returncode != 0
    assert "Image migration enabled occupancy" in unseeded.stderr


def test_iluvatar_cannot_silently_build_a_boxing_lane(tmp_path):
    done, calls, *_ = run_hook(tmp_path, boxing=True)
    assert done.returncode != 0
    assert "has no boxing lane" in done.stderr
    assert not any(row.startswith("seed ") for row in calls)
    assert "CPU-core-build" not in calls


@pytest.mark.parametrize("phase", ["cpu", "pip"])
def test_failed_install_stops_before_native_build(tmp_path, phase):
    done, calls, *_ = run_hook(tmp_path, prepared=False, fail=phase)
    assert done.returncode != 0
    assert "CPU-core-build" not in calls


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
    # The unproxied-route probe is no longer this helper's job: it lives in the
    # shared install_flagtree, which every platform calls, so the helper drops
    # its own call. This stub records installs, not routes.
    done, calls = provision(tmp_path, tree_matches=False)
    assert done.returncode == 0, done.stderr
    assert calls == [
        "-m pip uninstall -y flagtree triton triton_kernels",
        "tree new+iluvatar",
        "-m pip check",
    ]


def test_changed_gems_preserves_the_compiler_and_cpu_torch(tmp_path):
    done, calls = provision(tmp_path, gems_matches=False)
    assert done.returncode == 0, done.stderr
    assert calls == [
        "gems new-gems",
        "-m pip check",
    ]


@pytest.mark.parametrize("failure", ["uninstall", "check"])
def test_failed_replacement_or_dependency_check_stops_provisioning(tmp_path, failure):
    done, calls = provision(tmp_path, tree_matches=False, failure=failure)
    assert done.returncode == 47, done.stderr
    if failure == "uninstall":
        assert len(calls) == 1
