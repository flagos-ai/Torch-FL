# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise CUDA's shared test-venv phases without GPU Torch or a network."""

import os
from pathlib import Path
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / ".github/scripts/hooks/set_env_nvidia.sh"
SCRIPT = HOOK.read_text()


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("stage", ["build", "integration"])
def test_shared_provisioning_reuses_packages_and_preserves_cache_policy(
    tmp_path, prepared, stage
):
    source = tmp_path / "image-python"
    source.write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == *seed_runtime.py ]]; then\n'
        '  printf "seed %s\\n" "$*" >> "$CALL_LOG"; exit 0\n'
        "fi\n"
        'if [[ "$1" == -m ]]; then\n'
        '  printf "%s\\n" "$*" >> "$CALL_LOG"; exit 0\n'
        "fi\n"
        'if [[ "$1" == -c ]]; then echo 3.12.3; exit 0; fi\n'
        "cat >/dev/null\n"
        'printf "%s/torch\\n" "$VENV_ROOT"\n'
    )
    source.chmod(0o755)
    start = SCRIPT.index("# Sets TEST_PYTHON")
    end = SCRIPT.index("export FLAGOS_ACCELERATOR=cuda", start)
    log = tmp_path / "calls"
    done = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\n"
            'source "$REPO_ROOT/.github/scripts/lib/set_env_common.sh"\n'
            'select_test_python() { echo "$SOURCE_PYTHON"; }\n'
            "getconf() { echo 9.9; }\n"
            "create_runtime_venv() {\n"
            '  echo "create $1" >> "$CALL_LOG"\n'
            '  mkdir -p "$VENV_ROOT/bin"\n'
            '  cp "$SOURCE_PYTHON" "$VENV_ROOT/bin/python"\n'
            '  VENV_PYTHON="$VENV_ROOT/bin/python"\n'
            "}\n"
            'ensure_runtime_requirements() { echo "requirements $*" >> "$CALL_LOG"; }\n'
            'ensure_runtime_requirement() { echo "requirement $*" >> "$CALL_LOG"; }\n'
            'validate_runtime_build_tools() { echo build-tools >> "$CALL_LOG"; }\n'
            'runtime_cpu_torch_is_usable() { test "$PREPARED" = 1 || test -f "$VENV_ROOT/cpu-installed"; }\n'
            'runtime_flagtree_is_usable() { test "$PREPARED" = 1; }\n'
            'install_cpu_torch() { echo "cpu-torch $CPU_TORCH_VERSION" >> "$CALL_LOG"; touch "$VENV_ROOT/cpu-installed"; }\n'
            'install_flagtree() { echo flagtree >> "$CALL_LOG"; }\n'
            'install_flag_gems() { echo flaggems >> "$CALL_LOG"; }\n'
            + SCRIPT[start:end]
            + '\nprintf "CACHE=%s VENV=%s\\n" "$PIP_RETRY_NO_CACHE" "$VENV_ROOT"',
        ],
        env={
            **os.environ,
            "REPO_ROOT": str(ROOT),
            "RUNNER_TEMP": str(tmp_path),
            "CALL_LOG": str(log),
            "SOURCE_PYTHON": str(source),
            "VENDOR_PYTHON": "",
            "VENDOR_TORCH_ROOT": "",
            "CI_STAGE": stage,
            "CPU_TORCH_VERSION": "2.10.0",
            "FLAGTREE_PYTHON_VERSION": "3.12",
            "FLAGTREE_VERSION": "0.7.0",
            "FLAGGEMS_VERSION": "5.4.0",
            "FLAGTREE_MIN_GLIBC": "2.0",
            "PIP_INDEX_URL_ARG": "https://pypi.org/simple",
            "PIP_RETRY_NO_CACHE": "1",
            "PREPARED": "1" if prepared else "0",
            "TORCH_FL_VENV_ROOT": "",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    calls = log.read_text().splitlines()
    assert f"create {source}" in calls
    assert any("--requirements-file" in row for row in calls if row.startswith("seed "))
    assert any("requirements-nvidia.txt" in row for row in calls)
    assert "build-tools" in calls
    assert "flaggems" in calls
    assert ("requirement transformers https://pypi.org/simple" in calls) == (
        stage == "integration"
    )
    assert "-m pip check" in calls
    assert ("cpu-torch 2.10.0+cpu" in calls) is not prepared
    if not prepared:
        assert calls.index("-m pip uninstall -y torch") < calls.index(
            "cpu-torch 2.10.0+cpu"
        )
    else:
        assert "-m pip uninstall -y torch" not in calls
    assert ("flagtree" in calls) is not prepared
    uninstall = "-m pip uninstall -y flagtree triton triton_kernels"
    assert (uninstall in calls) is not prepared
    assert f"CACHE=1 VENV={tmp_path}/torch-fl-cuda-{stage}" in done.stdout


def test_cuda_keeps_its_image_and_enables_isolation_check():
    config = yaml.safe_load((ROOT / ".github/configs/nvidia.yml").read_text())
    assert (
        config["ci_image"] == "harbor.baai.ac.cn/flagos-dev/torch-fl:20261021-cuda-dev"
    )
    assert config["environment_check"]["venv_isolation"] is True


def test_cuda_exports_isolation_and_does_not_reset_the_selected_cache():
    assert "export PIP_RETRY_NO_CACHE=0" not in SCRIPT
    assert "export BASH_ENV=/dev/null" in SCRIPT
    assert "export TORCH_DEVICE_BACKEND_AUTOLOAD=0" in SCRIPT
    assert "export_ci_env BASH_ENV PATH VIRTUAL_ENV" in SCRIPT
    # Startup scripts and user-site packages must be disabled before probing
    # the image's accelerator interpreter, not just after creating the venv.
    assert SCRIPT.index("export BASH_ENV=/dev/null") < SCRIPT.index(
        'VENDOR_PYTHON="$(find_image_vendor_python)"'
    )
