# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise GCU runtime isolation without accelerator libraries or downloads."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / ".github/scripts/hooks/set_env_enflame.sh").read_text()
COMMON = (ROOT / ".github/scripts/lib/set_env_common.sh").read_text()


def shell_function(name, source=SCRIPT):
    start = source.index(f"{name}() {{")
    return source[start : source.index("\n}\n", start) + 3]


def run_shell(body, **environment):
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + body],
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONNOUSERSITE": "1",
            "REPO_ROOT": str(ROOT),
            "PYTHONPATH": "",
            **environment,
        },
        capture_output=True,
        text=True,
        timeout=30,
    )


def make_venv(tmp_path):
    root = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(root)], check=True
    )
    return root, next((root / "lib").glob("python*/site-packages"))


@pytest.mark.parametrize("colocated_headers", [True, False])
def test_sdk_discovery_allows_separate_topsaten_headers(tmp_path, colocated_headers):
    sdk = tmp_path / "tops"
    (sdk / "lib").mkdir(parents=True)
    (sdk / "lib/libtopsrt.so.1.10").touch()
    (sdk / "lib/libtopsrt.so").symlink_to("libtopsrt.so.1.10")
    if colocated_headers:
        header = sdk / "include/gcu/topsaten/topsaten.h"
        header.parent.mkdir(parents=True)
        header.touch()
    result = run_shell(
        shell_function("discover_tops_root") + "\ndiscover_tops_root",
        TOPS_HOME=str(sdk),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(sdk)


@pytest.mark.parametrize(
    "version,cuda,usable",
    [("2.10.0+cpu", None, True), ("2.11.0+cpu", None, False), ("2.10.0", "13", False)],
)
def test_cpu_torch_rejects_the_runtime_cohort(tmp_path, version, cuda, usable):
    root, site = make_venv(tmp_path)
    (site / "torch.py").write_text(
        f"__version__ = {version!r}\nclass version: cuda = {cuda!r}\n"
    )
    result = run_shell(
        shell_function("runtime_cpu_torch_is_usable", COMMON)
        + "\nruntime_cpu_torch_is_usable",
        VENV_PYTHON=str(root / "bin/python"),
        CPU_TORCH_VERSION="2.10.0",
    )
    assert (result.returncode == 0) == usable, result.stdout + result.stderr


@pytest.mark.parametrize("leak", [None, "module", "metadata", "system-site"])
def test_venv_rejects_vendor_torch_and_system_packages(tmp_path, leak):
    root, site = make_venv(tmp_path)
    if leak == "module":
        (site / "torch_gcu.py").write_text("raise RuntimeError('must not import')\n")
    elif leak == "metadata":
        info = site / "torch_gcu-2.11.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Name: torch-gcu\nVersion: 2.11.0\n")
    elif leak == "system-site":
        config = root / "pyvenv.cfg"
        config.write_text(
            config.read_text().replace(
                "include-system-site-packages = false",
                "include-system-site-packages = true",
            )
        )
    result = run_shell(
        "venv_is_usable() { return 0; }\n"
        + shell_function("runtime_venv_is_isolated", COMMON)
        + "\nruntime_venv_is_isolated torch_gcu",
        VENV_PYTHON=str(root / "bin/python"),
        VENV_ROOT=str(root),
    )
    assert (result.returncode == 0) == (leak is None), result.stdout + result.stderr
    assert "must not import" not in result.stderr


@pytest.mark.parametrize(
    "version,exit_code,accepted",
    [
        ("1.10.6-1", 0, True),
        ("1:1.10.6-2", 0, True),
        ("1.9.10-1", 0, False),
        ("", 1, False),
    ],
)
def test_sdk_evidence_is_required(version, exit_code, accepted):
    result = run_shell(
        """dpkg-query() { printf '%s' "$TEST_VERSION"; return "$TEST_EXIT"; }
"""
        + shell_function("gcu_record_runtime_sdk")
        + '\ngcu_record_runtime_sdk\nprintf "SDK=%s\\n" "$FLAGOS_SDK_VERSION"',
        TEST_VERSION=version,
        TEST_EXIT=str(exit_code),
    )
    assert (result.returncode == 0) == accepted, result.stdout + result.stderr
    if accepted:
        assert "SDK=1.10.6" in result.stdout


def test_later_shell_cannot_reactivate_the_image(tmp_path):
    start = SCRIPT.rindex('if [[ -n "${GITHUB_ENV:-}" ]]; then')
    end = SCRIPT.index("\n# Print the device/driver inventory", start)
    common = (ROOT / ".github/scripts/lib/set_env_common.sh").read_text()
    a = common.index("export_ci_env() {")
    b = common.index("\n}\n", a) + 3
    env_file = tmp_path / "github-env"
    names = (
        "PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH FLAGOS_ACCELERATOR "
        "FLAGOS_BUILD_VENDOR FLAGOS_BUILD_FLAGGEMS_CPP FLAGOS_BUILD_FLAGGEMS "
        "FLAGOS_DISABLE_CUDA_ASSETS TOPS_HOME TOPSATEN_LIB CPATH LIBRARY_PATH "
        "LD_LIBRARY_PATH"
    ).split()
    result = run_shell(
        common[a:b] + "\n" + SCRIPT[start:end],
        **{name: "" for name in names if name != "PATH"},
        BASH_ENV="/dev/null",
        FLAGOS_SDK_VERSION="1.10.6",
        GITHUB_ENV=str(env_file),
    )
    assert result.returncode == 0, result.stderr
    rows = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    assert rows["BASH_ENV"] == "/dev/null"
    activation = tmp_path / "activate.sh"
    activation.write_text("export VIRTUAL_ENV=/flagos\n")
    # Explicitly read BASH_ENV as well: privileged local bash can suppress its
    # automatic startup read, while the CI shell runs with matching UID/EUID.
    command = 'bash -c \'. "$BASH_ENV"; test "${VIRTUAL_ENV:-}" != /flagos\''
    contaminated = run_shell(command, BASH_ENV=str(activation))
    assert contaminated.returncode != 0
    result = run_shell(command, BASH_ENV=rows["BASH_ENV"])
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "stage,expected", [("build", "build\navailable\n"), ("integration", "")]
)
def test_native_build_count_is_preserved(tmp_path, stage, expected):
    start = SCRIPT.rindex('\ncd "$REPO_ROOT"')
    result = run_shell(
        "build_flagos_inplace() { echo build; }\n"
        "python() { cat >/dev/null; echo available; }\n" + SCRIPT[start:],
        REPO_ROOT=str(tmp_path),
        CI_STAGE=stage,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected
