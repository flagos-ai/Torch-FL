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

"""Exercise MUSA environment reuse without hardware, downloads or package installs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_PATH = REPO_ROOT / ".github/scripts/lib/set_env_common.sh"
COMMON = COMMON_PATH.read_text()
SCRIPT = (REPO_ROOT / ".github/scripts/hooks/set_env_mthreads.sh").read_text()
VENDORED_BUILD_VERSION = "1.4.0"


def shell_function(name, script=SCRIPT):
    start = script.index(f"{name}() {{")
    end = script.index("\n}\n", start) + 3
    return script[start:end]


def run_shell(body, *, cwd=None, **environment):
    # Keep the build frontend and the host's Python modules out of process
    # launchers that the shell script only emulates.
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LC_ALL": "C",
        "PYTHONPATH": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "REPO_ROOT": str(REPO_ROOT),
    }
    env.update(environment)
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + body],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def fake_build_tools(site, *, editable=True, frontend=True):
    site.mkdir(parents=True, exist_ok=True)
    setuptools = site / "setuptools"
    commands = setuptools / "command"
    commands.mkdir(parents=True)
    version = "64.0.0" if editable else "59.6.0"
    (setuptools / "__init__.py").write_text(f"__version__ = {version!r}\n")
    (commands / "__init__.py").touch()
    if editable:
        (commands / "editable_wheel.py").write_text("class editable_wheel: pass\n")
    if frontend:
        (site / "build.py").write_text("__version__ = '1.4.0'\n")


@pytest.mark.parametrize(
    "editable,frontend,usable",
    [(True, True, True), (False, True, False), (True, False, False)],
)
@pytest.mark.parametrize("built_checkout", [False, True])
@pytest.mark.parametrize("ambient_build", [False, True])
def test_build_tools_require_real_imports(
    tmp_path, monkeypatch, editable, frontend, usable, built_checkout, ambient_build
):
    if ambient_build:
        ambient = tmp_path / "ambient-python"
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(ambient)], check=True
        )
        ambient_site = next((ambient / "lib").glob("python*/site-packages"))
        (ambient_site / "build.py").write_text("__version__ = '9.9.9'\n")
        monkeypatch.setattr(sys, "executable", str(ambient / "bin/python"))
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    if built_checkout:
        (checkout / "build").mkdir()
    monkeypatch.chdir(checkout)
    root = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(root)], check=True
    )
    site = next((root / "lib").glob("python*/site-packages"))
    fake_build_tools(site, editable=editable, frontend=frontend)
    result = run_shell(
        shell_function("validate_runtime_build_tools", COMMON)
        + '\nvalidate_runtime_build_tools\necho "AUTOLOAD=$TORCH_DEVICE_BACKEND_AUTOLOAD"',
        # Do not import the checkout's build/ directory as a namespace package.
        cwd=tmp_path,
        VENV_PYTHON=str(root / "bin/python"),
        TORCH_DEVICE_BACKEND_AUTOLOAD="1",
    )
    assert (result.returncode == 0) == usable, result.stdout + result.stderr
    if usable:
        assert "editable_wheel available" in result.stdout
        assert f"Build frontend: {VENDORED_BUILD_VERSION}" in result.stdout
        assert "AUTOLOAD=1" in result.stdout
    else:
        assert "ModuleNotFoundError" in result.stderr


def test_musa_uses_the_entrypoint_cache_policy():
    assert "musa_prepare_pip_cache" not in SCRIPT
    runner = (REPO_ROOT / ".github/scripts/set_env.sh").read_text()
    assert 'prepare_pip_cache "$PLATFORM"' in runner
    assert "ascend|iluvatar|kunlunxin|mthreads|thead|enflame|metax)" in runner


def test_cache_directory_is_persisted_to_later_ci_steps(tmp_path):
    # Exercise the actual export block rather than just its helper.
    start = SCRIPT.rindex('if [[ -n "${GITHUB_ENV:-}" ]]; then')
    end = SCRIPT.index('\ncd "$REPO_ROOT"', start)
    names = (
        "PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH FLAGOS_ACCELERATOR MUSA_HOME "
        "FLAGOS_BUILD_VENDOR FLAGOS_BUILD_FLAGGEMS_CPP FLAGOS_BUILD_FLAGGEMS "
        "FLAGOS_DISABLE_CUDA_ASSETS MTHREADS_VISIBLE_DEVICES CPATH LIBRARY_PATH "
        "LD_LIBRARY_PATH"
    ).split()
    environment = {name: "" for name in names if name != "PATH"}
    environment.update(BASH_ENV="/dev/null", FLAGOS_SDK_VERSION="5.2.0")
    github_env = tmp_path / "github-env"
    cache = tmp_path / "cache"
    result = run_shell(
        shell_function("export_ci_env", COMMON) + "\n" + SCRIPT[start:end],
        **environment,
        PIP_CACHE_DIR=str(cache),
        GITHUB_ENV=str(github_env),
    )
    assert result.returncode == 0, result.stderr
    assert f"PIP_CACHE_DIR={cache}\n" in github_env.read_text()


@pytest.mark.parametrize(
    "condition,usable",
    [
        ("isolated", True),
        ("system_packages", False),
        ("conflicting_config", False),
        ("vendor_module", False),
        ("orphan_vendor_metadata", False),
        ("missing_pip", False),
        ("broken_pip", False),
    ],
)
def test_venv_reuse_requires_isolation(tmp_path, condition, usable):
    root = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(root)], check=True
    )
    site = next((root / "lib").glob("python*/site-packages"))
    if condition != "missing_pip":
        (site / "pip.py").write_text(
            "raise SystemExit(1)" if condition == "broken_pip" else ""
        )
    if condition == "system_packages":
        config = root / "pyvenv.cfg"
        config.write_text(config.read_text().replace("= false", "= true"))
    if condition == "conflicting_config":
        config = root / "pyvenv.cfg"
        config.write_text(config.read_text() + "include-system-site-packages = true\n")
    if condition == "vendor_module":
        (site / "torch_musa.py").touch()
    if condition == "orphan_vendor_metadata":
        dist = site / "torch_musa-2.9.1.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text("Name: torch_musa\nVersion: 2.9.1\n")

    result = run_shell(
        shell_function("venv_is_usable", COMMON)
        + "\n"
        + shell_function("runtime_venv_is_isolated", COMMON)
        + "\nruntime_venv_is_isolated torch_musa",
        VENV_ROOT=str(root),
        VENV_PYTHON=str(root / "bin/python"),
    )
    assert (result.returncode == 0) == usable, result.stdout + result.stderr


@pytest.mark.parametrize(
    "condition,reuse",
    [
        ("same_version", True),
        ("different_version", False),
        ("missing_module", False),
        ("missing_metadata", False),
    ],
)
@pytest.mark.parametrize("ambient_flaggems", [False, True])
def test_flaggems_reuse_requires_published_version(
    tmp_path, monkeypatch, condition, reuse, ambient_flaggems
):
    if ambient_flaggems:
        ambient = tmp_path / "ambient-python"
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(ambient)], check=True
        )
        ambient_site = next((ambient / "lib").glob("python*/site-packages"))
        (ambient_site / "flag_gems.py").touch()
        ambient_dist = ambient_site / "flag_gems-5.3.0.dist-info"
        ambient_dist.mkdir()
        (ambient_dist / "METADATA").write_text("Name: flag_gems\nVersion: 5.3.0\n")
        # A module of the requested version still has to be importable from the
        # selected environment; the host copy must not make the probe pass.
        ambient_dist = ambient_site / "flag_gems-5.4.0.dist-info"
        ambient_dist.mkdir()
        (ambient_dist / "METADATA").write_text("Name: flag_gems\nVersion: 5.4.0\n")
        monkeypatch.setattr(sys, "executable", str(ambient / "bin/python"))
    # Missing-module cases must not discover the runner's installed FlagGems.
    root = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(root)], check=True
    )
    site = next((root / "lib").glob("python*/site-packages"))
    dist = site / "flag_gems-5.4.0.dist-info"
    dist.mkdir()
    if condition != "missing_metadata":
        version = "5.3.0" if condition == "different_version" else "5.4.0"
        (dist / "METADATA").write_text(f"Name: flag_gems\nVersion: {version}\n")
    if condition != "missing_module":
        (site / "flag_gems.py").touch()

    result = run_shell(
        # install_flag_gems probes the unproxied route before it installs; the
        # sandbox has no proxy_route.sh, so record it and carry on.
        "prefer_direct_route() { printf 'ROUTE_PROBED %s\\n' \"$1\"; }\n"
        "pip_retry() {\n"
        '  printf "INSTALL_REQUESTED %s\\n" "$*"\n'
        '  printf "Name: flag_gems\\nVersion: 5.4.0\\n" > "$STUB_METADATA"\n'
        '  touch "$STUB_MODULE"\n'
        "}\n"
        + shell_function("flag_gems_installed", COMMON)
        + "\n"
        + shell_function("install_flag_gems", COMMON)
        + "\ninstall_flag_gems",
        cwd=tmp_path,
        VENV_PYTHON=str(root / "bin/python"),
        FLAGGEMS_VERSION="5.4.0",
        FLAGGEMS_INDEX_URL="https://example.invalid/mthreads/simple",
        STUB_METADATA=str(dist / "METADATA"),
        STUB_MODULE=str(site / "flag_gems.py"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("INSTALL_REQUESTED" not in result.stdout) == reuse
    if not reuse:
        assert "--only-binary=:all:" in result.stdout
        assert "flag-gems===5.4.0" in result.stdout
        assert "git+" not in result.stdout


def test_flaggems_rejects_an_unusable_installed_wheel(tmp_path):
    result = run_shell(
        # install_flag_gems probes the unproxied route first; the sandbox has no
        # proxy_route.sh, so record it and carry on.
        "prefer_direct_route() { printf 'ROUTE_PROBED %s\\n' \"$1\"; }\n"
        "pip_retry() { return 0; }\n"
        + shell_function("flag_gems_installed", COMMON)
        + "\n"
        + shell_function("install_flag_gems", COMMON)
        + "\ninstall_flag_gems",
        VENV_PYTHON=sys.executable,
        PYTHONPATH=str(tmp_path),
        FLAGGEMS_VERSION="0.0.0-test-missing",
        FLAGGEMS_INDEX_URL="https://example.invalid/simple",
    )
    assert result.returncode != 0
    assert "not importable after wheel installation" in result.stdout


@pytest.mark.parametrize("qwen", ["0", "1"])
def test_integration_keeps_bert_and_gates_tokenizer_deps(qwen):
    block = shell_function("prepare_musa_environment")
    start = block.index('  if [[ "$CI_STAGE" == integration ]]; then')
    end = block.index("  export FLAGOS_SDK_VERSION", start)
    result = run_shell(
        'ensure_runtime_requirement() { printf "%s\\n" "$@"; }\n' + block[start:end],
        CI_STAGE="integration",
        PIP_INDEX_URL_ARG="https://example.invalid/simple",
        TORCH_FL_INSTALL_QWEN_DEPS=qwen,
    )
    assert result.returncode == 0, result.stderr
    assert "transformers>=4.51,<5" in result.stdout
    for package in ("sentencepiece", "tiktoken", "protobuf"):
        assert (package in result.stdout) == (qwen == "1")


@pytest.mark.parametrize(
    "version,cuda,usable",
    [
        ("2.10.0+cpu", None, True),
        ("2.9.1+cpu", None, False),
        ("2.10.0", None, False),
        ("2.10.0+cpu", "12.8", False),
    ],
)
def test_cpu_torch_probe_requires_the_pinned_cpu_build(tmp_path, version, cuda, usable):
    (tmp_path / "torch.py").write_text(
        "import os\n"
        "assert os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] == '0'\n"
        f"__version__ = {version!r}\n"
        f"class version: cuda = {cuda!r}\n"
    )
    result = run_shell(
        shell_function("runtime_cpu_torch_is_usable", COMMON)
        + '\nruntime_cpu_torch_is_usable\necho "AUTOLOAD=$TORCH_DEVICE_BACKEND_AUTOLOAD"',
        VENV_PYTHON=sys.executable,
        CPU_TORCH_VERSION="2.10.0",
        PYTHONPATH=str(tmp_path),
        TORCH_DEVICE_BACKEND_AUTOLOAD="1",
    )
    assert (result.returncode == 0) == usable, result.stdout + result.stderr
    if usable:
        assert "AUTOLOAD=1" in result.stdout


@pytest.mark.parametrize("cache", ["0", "1"])
def test_shared_pip_retry_honors_musa_cache_setting(tmp_path, cache):
    # With caching on, the no-cache flag is absent and `extra` is empty. macOS
    # `/bin/bash` 3.2 needs the alternate expansion so that `set -u` accepts it.
    recorder = tmp_path / "python"
    recorder.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    recorder.chmod(0o755)
    # `pip_retry` is the shared library's, and it calls `pip_index_host` and
    # `ensure_index_host_route` from that same library -- extracting the one
    # function would leave both undefined. Extract all three rather than
    # sourcing the whole library: the rest of it provisions a venv, and this
    # test only observes the cache flag the retry helper passes to pip.
    body = "\n".join(
        shell_function(name, COMMON)
        for name in ("pip_index_host", "ensure_index_host_route", "pip_retry")
    )
    result = run_shell(
        body + '\nPIP_RETRY_PYTHON="$VENV_PYTHON" pip_retry package-for-test',
        VENV_PYTHON=str(recorder),
        PIP_RETRY_NO_CACHE=cache,
    )
    assert result.returncode == 0, result.stderr
    assert ("--no-cache-dir" in result.stdout) == (cache == "1")


def run_runtime_hook(
    tmp_path, *, prepared=True, fail="", stage="integration", target="ci"
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".github").symlink_to(REPO_ROOT / ".github", target_is_directory=True)
    image = tmp_path / "image"
    image.mkdir()
    sdk = tmp_path / "sdk"
    for asset in (
        "include/mudnncxx/mudnn.h",
        "include/murand.h",
        "lib/libmudnn.so",
        "lib/libmurand.so",
    ):
        path = sdk / asset
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    ci = tmp_path / "ci-python"
    ci.write_text(r"""#!/bin/bash
printf 'python %s\n' "$*" >> "$CALL_LOG"
if [[ "$*" == *--check-installed* ]]; then
  [[ "$PREPARED" == 1 ]] || grep -Fxq -- "$3" "$INSTALLED" 2>/dev/null
  exit $?
fi
if [[ "$1 $2" == '-m pip' ]]; then
  [[ "$3" != check || "$FAIL_PHASE" != check ]]; exit $?
fi
if [[ "$1" == '-c' ]]; then
  [[ "$2" != 'import torch_fl._C' ]]; exit $?
fi
if [[ "$1" == '-' ]]; then
  input="$(cat)"
  if [[ "$input" == *'expected = sys.argv[1]'* ]]; then
    [[ "$PREPARED" == 1 || -f "$CPU_READY" ]]; exit $?
  fi
  if [[ "$input" == *'metadata.distribution("flagtree")'* ]]; then
    [[ "$PREPARED" == 1 && "$FAIL_PHASE" != backend ]]; exit $?
  fi
fi
if [[ $# == 0 ]]; then echo ci-python; fi
""")
    ci.chmod(0o755)
    vendor = image / "python3.10"
    vendor.write_text(r"""#!/bin/bash
set -e
printf 'vendor %s\n' "$*" >> "$CALL_LOG"
if [[ "$1 $2" == '-I -c' ]]; then [[ "$FAIL_PHASE" != python ]]; exit $?; fi
if [[ "$1 $2" == '-I -' ]]; then
  exec "$REAL_PYTHON" -I -c 'import os,sys; sys.prefix=os.environ["IMAGE_PREFIX"]; sys.base_prefix=sys.prefix; exec(sys.stdin.read())' "$3" "$4"
fi
if [[ "$1 $2" == '-m venv' ]]; then
  [[ "$FAIL_PHASE" != venv && "$FAIL_PHASE" != uv ]] || exit 1
  mkdir -p "$4/bin"; cp "$CI_PYTHON" "$4/bin/python"; exit 0
fi
if [[ "$*" == *seed_runtime.py* ]]; then [[ "$FAIL_PHASE" != seed ]]; exit $?; fi
exit 2
""")
    vendor.chmod(0o755)
    uv = image / "uv"
    uv.write_text(r"""#!/bin/bash
printf 'uv %s\n' "$*" >> "$CALL_LOG"
[[ "$FAIL_PHASE" != uv ]] || exit 1
target="${@: -1}"
mkdir -p "$target/bin"; cp "$CI_PYTHON" "$target/bin/python"
""")
    uv.chmod(0o755)
    apt = image / "apt-get"
    apt.write_text('#!/bin/sh\necho forbidden-apt >> "$CALL_LOG"\nexit 1\n')
    apt.chmod(0o755)
    gmi = image / "mthreads-gmi"
    gmi.write_text('#!/bin/sh\necho gmi >> "$CALL_LOG"\n')
    gmi.chmod(0o755)
    startup = tmp_path / "startup"
    startup.write_text('echo startup >> "$CALL_LOG"\nexport PYTHONPATH=/opt/flagtree\n')
    calls = tmp_path / "calls"
    env_file = tmp_path / "github-env"
    path_file = tmp_path / "github-path"
    env = dict(
        os.environ,
        PATH=f"{image}:/usr/bin:/bin",
        BASH_ENV=str(startup),
        REAL_PYTHON=sys.executable,
        IMAGE_PREFIX=str(image),
        TORCH_FL_VENDOR_PYTHON=str(vendor),
        TORCH_FL_VENV_ROOT=str(tmp_path / target),
        MUSA_HOME=str(sdk),
        REPO_ROOT=str(repo),
        CI_STAGE=stage,
        GITHUB_ENV=str(env_file),
        GITHUB_PATH=str(path_file),
        CALL_LOG=str(calls),
        CI_PYTHON=str(ci),
        CPU_READY=str(tmp_path / "cpu-ready"),
        INSTALLED=str(tmp_path / "installed"),
        PREPARED="1" if prepared else "0",
        FAIL_PHASE=fail,
        PIP_CACHE_DIR=str(tmp_path / "cache"),
        PIP_RETRY_NO_CACHE="0",
        PIP_CONSTRAINT=str(REPO_ROOT / ".github/constraints.txt"),
    )
    if fail == "sdk":
        (sdk / "include/mudnncxx/mudnn.h").unlink()
    script = r"""
set -euo pipefail
source "$REPO_ROOT/.github/version-pins.env"
source "$REPO_ROOT/.github/scripts/lib/set_env_common.sh"
venv_is_usable() { [[ "$FAIL_PHASE" != isolated ]]; }
install_cpu_torch() { echo cpu-torch >> "$CALL_LOG"; touch "$CPU_READY"; [[ "$FAIL_PHASE" != cpu ]]; }
install_flagtree() { echo flagtree >> "$CALL_LOG"; [[ "$FAIL_PHASE" != flagtree ]]; }
install_flag_gems() { echo flag-gems >> "$CALL_LOG"; [[ "$FAIL_PHASE" != gems ]]; }
pip_retry() {
  printf 'pip %s\n' "$*" >> "$CALL_LOG"
  [[ "$FAIL_PHASE" != pip ]] || return 1
  [[ "$FAIL_PHASE" == ineffective-pip ]] || printf '%s\n' "${@: -1}" >> "$INSTALLED"
}
build_flagos_inplace() { echo native-build >> "$CALL_LOG"; }
ls() { if [[ "$1" == /dev/mtgpu* ]]; then return 0; fi; command ls "$@"; }
source "$REPO_ROOT/.github/scripts/hooks/set_env_mthreads.sh"
"""
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )
    return result, calls.read_text().splitlines(), env_file, path_file, env


@pytest.mark.parametrize("stage", ["build", "integration"])
def test_runtime_image_reuses_packages_and_keeps_next_shell_isolated(tmp_path, stage):
    done, calls, env_file, path_file, env = run_runtime_hook(tmp_path, stage=stage)
    assert done.returncode == 0, done.stderr
    assert calls.count("native-build") == 1
    assert "flagtree" not in calls and "cpu-torch" not in calls
    assert not any(row.startswith("pip ") for row in calls)
    assert "forbidden-apt" not in calls
    assert not any("--flagcx-version" in row for row in calls)
    rows = dict(row.split("=", 1) for row in env_file.read_text().splitlines())
    assert rows["BASH_ENV"] == "/dev/null"
    assert rows["PYTHONPATH"] == ""
    assert rows["FLAGOS_SDK_VERSION"] == "5.2.0"
    assert rows["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert any("--report-environment" in row for row in calls)
    result = subprocess.run(
        ["bash", "-c", "python"],
        env={**env, **rows, "PATH": path_file.read_text().strip() + ":" + rows["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "ci-python"
    assert Path(env["CALL_LOG"]).read_text().splitlines().count("startup") == 1


@pytest.mark.parametrize("stage", ["build", "integration"])
def test_runtime_image_installs_only_missing_python_requirements(tmp_path, stage):
    done, calls, *_ = run_runtime_hook(tmp_path, prepared=False, stage=stage)
    assert done.returncode == 0, done.stderr
    assert "cpu-torch" in calls and "flagtree" in calls
    assert any(
        row.endswith("cmake>=3.19,<4") and row.startswith("pip ") for row in calls
    )
    assert any(row.endswith("sqlalchemy") and row.startswith("pip ") for row in calls)
    assert any("transformers>=4.51,<5" in row for row in calls) == (
        stage == "integration"
    )
    assert next(
        i
        for i, row in enumerate(calls)
        if row.startswith("pip ") and row.endswith("pip>=23.2")
    ) < calls.index("cpu-torch")
    assert "forbidden-apt" not in calls


@pytest.mark.parametrize(
    "fail",
    [
        "sdk",
        "python",
        "seed",
        "isolated",
        "uv",
        "pip",
        "ineffective-pip",
        "cpu",
        "gems",
        "check",
        "flagtree",
    ],
)
def test_invalid_runtime_or_failed_install_stops_before_native_build(tmp_path, fail):
    done, calls, *_ = run_runtime_hook(tmp_path, prepared=False, fail=fail)
    assert done.returncode != 0
    assert "native-build" not in calls
    assert "forbidden-apt" not in calls


def test_runtime_image_uses_uv_if_the_venv_module_cannot_seed_pip(tmp_path):
    done, calls, *_ = run_runtime_hook(tmp_path, fail="venv")
    assert done.returncode == 0, done.stderr
    assert any(row.startswith("uv venv --clear --seed --python ") for row in calls)
    assert "forbidden-apt" not in calls


@pytest.mark.parametrize("target", ["image", "image/subdir", ".", "repo"])
def test_runtime_venv_override_cannot_clear_the_image_or_checkout(tmp_path, target):
    done, calls, *_ = run_runtime_hook(tmp_path, target=target)
    assert done.returncode != 0
    assert not any("-m venv" in row for row in calls)
    assert "native-build" not in calls


@pytest.mark.parametrize(
    "state", ["local", "wrong-version", "shadowed", "no-mthreads", "external"]
)
def test_runtime_compiler_reuse_requires_local_matching_provider(tmp_path, state):
    prefix = tmp_path / "ci"
    site = prefix / "site-packages"
    site.mkdir(parents=True)
    provider = site if state != "external" else tmp_path / "external"
    compiler = provider / "triton"
    (compiler / "backends/mthreads").mkdir(parents=True)
    (compiler / "__init__.py").touch()
    info = provider / "flagtree-0.7.0.dist-info"
    info.mkdir()
    version = "0.6.1+mthreads3.6" if state == "wrong-version" else "0.7.0+mthreads3.6"
    (info / "METADATA").write_text(f"Name: flagtree\nVersion: {version}\n")
    if state == "no-mthreads":
        (compiler / "backends/mthreads").rmdir()
    paths = [str(provider)]
    if state == "shadowed":
        shadow = tmp_path / "shadow"
        (shadow / "triton").mkdir(parents=True)
        (shadow / "triton/__init__.py").touch()
        paths.insert(0, str(shadow))
    probe = (
        COMMON.split("runtime_flagtree_is_usable() {", 1)[1]
        .split("<<'PY'\n", 1)[1]
        .split("\nPY", 1)[0]
    )
    code = f"import sys; sys.prefix={str(prefix)!r}; sys.path[:0]={paths!r}\n" + probe
    done = subprocess.run(
        [sys.executable, "-I", "-c", code, "0.7.0+mthreads3.6", "mthreads", ""],
        capture_output=True,
        text=True,
    )
    assert (done.returncode == 0) == (state == "local"), done.stderr


def test_musa_selects_the_verified_runtime_without_changing_the_build_contract():
    import yaml

    config = yaml.safe_load((REPO_ROOT / ".github/configs/mthreads.yml").read_text())
    assert config["ci_image"].endswith(
        ":2.3.0@sha256:1870e059f98d95465009f55db78d723cf77640b211ff1c5f1378d59b8e200c27"
    )
    assert config["setup_platform"] == "mthreads"
    assert config["runner_labels"] == ["mt-8g-cicd-pytorch"]
    assert SCRIPT.count("\nbuild_flagos_inplace\n") == 1
    assert "seed_runtime.py" in SCRIPT
    assert "apt-get" not in SCRIPT
