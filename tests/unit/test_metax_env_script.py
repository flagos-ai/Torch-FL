# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise MetaX runtime provisioning without hardware or network installs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / ".github/scripts/hooks/set_env_metax.sh").read_text()
COMMON = (ROOT / ".github/scripts/lib/set_env_common.sh").read_text()


def shell_function(name, source=SCRIPT):
    start = source.index(f"{name}() {{")
    return source[start : source.index("\n}\n", start) + 3]


def run_shell(body, **environment):
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + body],
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": "",
            "PYTHONNOUSERSITE": "1",
            "REPO_ROOT": str(ROOT),
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


@pytest.mark.parametrize(
    "version,accepted",
    [
        ("2.10.0+metax3.8.1.0", True),
        ("2.10.0+cpu", False),
        ("2.9.1+metax3.8.1.0", False),
    ],
)
def test_vendor_provenance_never_imports_torch(tmp_path, version, accepted):
    root, site = make_venv(tmp_path)
    info = site / f"torch-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Name: torch\nVersion: {version}\n")
    torch = site / "torch"
    (torch / "lib").mkdir(parents=True)
    (torch / "__init__.py").write_text(
        "raise RuntimeError('vendor import forbidden')\n"
    )
    (torch / "version.py").write_text(f"__version__ = {version!r}\n")
    result = run_shell(
        shell_function("metax_read_vendor_torch")
        + '\nmetax_read_vendor_torch "$VENDOR_PYTHON"\n'
        + 'printf "%s|%s|%s\\n" "$FLAGOS_VENDOR_TORCH_LIB" '
        + '"$FLAGOS_VENDOR_TORCH_VERSION" "$TORCH_FL_VENDOR_TORCH_VERSION"',
        VENDOR_PYTHON=str(root / "bin/python"),
        CPU_TORCH_VERSION="2.10.0",
    )
    assert (result.returncode == 0) == accepted, result.stdout + result.stderr
    assert "vendor import forbidden" not in result.stderr
    if accepted:
        assert result.stdout.strip() == f"{torch / 'lib'}|{version}|2.10.0"


@pytest.mark.parametrize(
    "version,cuda,accepted",
    [
        ("2.10.0+cpu", None, True),
        ("2.10.0+metax3.8.1.0", "11.6", False),
        ("2.11.0+cpu", None, False),
    ],
)
def test_build_torch_remains_the_cpu_wheel(tmp_path, version, cuda, accepted):
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
    assert (result.returncode == 0) == accepted, result.stdout + result.stderr


@pytest.mark.parametrize("leak", [None, "system-site", "outside-prefix"])
def test_venv_does_not_inherit_the_runtime_interpreter(tmp_path, leak):
    root, _site = make_venv(tmp_path)
    if leak == "system-site":
        config = root / "pyvenv.cfg"
        config.write_text(config.read_text().replace("= false", "= true"))
    python = sys.executable if leak == "outside-prefix" else str(root / "bin/python")
    result = run_shell(
        "venv_is_usable() { return 0; }\n"
        + shell_function("runtime_venv_is_isolated", COMMON)
        + "\nruntime_venv_is_isolated ",
        VENV_ROOT=str(root),
        VENV_PYTHON=python,
    )
    assert (result.returncode == 0) == (leak is None), result.stdout + result.stderr


@pytest.mark.parametrize(
    "version,accepted",
    [
        ("Version:3.8.1.3\n", True),
        ("Version:3.8.1.3\r\n", True),
        ("Version:3.8.0\n", False),
        (None, False),
    ],
)
def test_sdk_version_requires_the_installed_version_file(tmp_path, version, accepted):
    if version is not None:
        (tmp_path / "Version.txt").write_text(version)
    result = run_shell(
        shell_function("metax_record_runtime_sdk")
        + '\nmetax_record_runtime_sdk\nprintf "%s\\n" "$FLAGOS_SDK_VERSION"',
        MACA_PATH=str(tmp_path),
    )
    assert (result.returncode == 0) == accepted, result.stdout + result.stderr
    if accepted:
        assert result.stdout == "3.8.1.3\n"


@pytest.mark.parametrize(
    "version,backend,spec,accepted",
    [
        ("0.7.0.post2+metax3.6", "metax", True, True),
        ("0.7.0+metax3.6", "metax", True, False),
        ("0.7.0.post2+metax3.6", "nvidia", True, False),
        ("0.7.0.post2+metax3.6", "metax", False, False),
    ],
)
def test_compiler_reuse_keeps_the_current_post2_cohort(
    tmp_path, version, backend, spec, accepted
):
    root, site = make_venv(tmp_path)
    info = site / f"flagtree-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Name: flagtree\nVersion: {version}\n")
    triton = site / "triton"
    (triton / "backends" / backend).mkdir(parents=True)
    (triton / "__init__.py").touch()
    if spec:
        (triton / "_flagtree_spec.py").touch()
    result = run_shell(
        shell_function("runtime_flagtree_is_usable", COMMON)
        + "\nruntime_flagtree_is_usable metax _flagtree_spec.py",
        VENV_PYTHON=str(root / "bin/python"),
        FLAGTREE_VERSION="0.7.0.post2+metax3.6",
    )
    assert (result.returncode == 0) == accepted, result.stdout + result.stderr


@pytest.mark.parametrize("boxing", ["", "1"])
def test_both_lanes_keep_the_image_inherited_runtime_flags(tmp_path, boxing):
    conf = tmp_path / "torch_fl/configs/backends_metax_boxing.conf"
    conf.parent.mkdir(parents=True)
    conf.touch()
    start = SCRIPT.index("export FLAGOS_BUILD_VENDOR=OFF")
    end = SCRIPT.index("\nexport LD_LIBRARY_PATH=", start)
    result = run_shell(
        SCRIPT[start:end]
        + '\nprintf "%s|%s|%s|%s|%s\\n" "$FLAGOS_BUILD_VENDOR" '
        + '"${FLAGOS_BUILD_BOXING:-}" "$FLAGOS_METAX_CUDART_SHIM" '
        + '"$FLAGOS_DISABLE_CUDA_ASSETS" "$FLAGOS_BUILD_FLAGGEMS"',
        BOXING_LANE=boxing,
        REPO_ROOT=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"OFF|{boxing}|1|1|1"


@pytest.mark.parametrize("stage", ["build", "integration"])
def test_native_build_and_bundle_order_is_unchanged(tmp_path, stage):
    start = SCRIPT.index('\ncd "$REPO_ROOT"\n\nif [[ "$CI_STAGE"')
    end = SCRIPT.index("\n# --- FlagGems cohort", start)
    result = run_shell(
        "build_flagos_inplace() { echo native; }\n"
        'bash() { test "$1" = scripts/vendor/bundle_metax_libtorch.sh; echo bundle; }\n'
        + SCRIPT[start:end],
        CI_STAGE=stage,
        REPO_ROOT=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "native\nbundle\n"


def test_runtime_isolation_and_provenance_reach_later_steps(tmp_path):
    start = SCRIPT.index('if [[ -n "${GITHUB_ENV:-}" ]]; then')
    end = SCRIPT.index('\ncd "$REPO_ROOT"', start)
    names = (
        "VIRTUAL_ENV PYTHONNOUSERSITE FLAGOS_ACCELERATOR MACA_PATH MACA_HOME "
        "FLAGOS_BUILD_VENDOR FLAGOS_METAX_CUDART_SHIM FLAGOS_DISABLE_CUDA_ASSETS "
        "FLAGOS_BUILD_FLAGGEMS_CPP FLAGOS_BUILD_FLAGGEMS FLAGOS_VENDOR_TORCH_LIB "
        "TORCH_FL_VENDOR_TORCH_VERSION LD_LIBRARY_PATH LIBRARY_PATH CPATH"
    ).split()
    env_file = tmp_path / "github-env"
    result = run_shell(
        shell_function("export_ci_env", COMMON) + "\n" + SCRIPT[start:end],
        **{name: "" for name in names},
        BASH_ENV="/dev/null",
        PYTHONPATH="",
        FLAGOS_SDK_VERSION="3.8.1.3",
        FLAGOS_VENDOR_TORCH_VERSION="2.10.0+metax3.8.1.0",
        GITHUB_ENV=str(env_file),
    )
    assert result.returncode == 0, result.stderr
    rows = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    assert rows["BASH_ENV"] == "/dev/null"
    assert rows["PYTHONPATH"] == ""
    assert rows["FLAGOS_SDK_VERSION"] == "3.8.1.3"
    assert rows["FLAGOS_VENDOR_TORCH_VERSION"] == "2.10.0+metax3.8.1.0"


@pytest.mark.parametrize("prepared", [True, False])
def test_runtime_flagcx_reuses_or_installs_the_declared_dependency(tmp_path, prepared):
    python = tmp_path / "python"
    python.write_text(f"#!/bin/sh\nexit {0 if prepared else 1}\n")
    python.chmod(0o755)
    start = SCRIPT.index('FLAGCX_VERSION="${TORCH_FL_FLAGCX_VERSION:')
    end = SCRIPT.index('\n"$VENV_PYTHON" -m pip check', start)
    result = run_shell(
        "install_flagcx() { echo install-flagcx; }\n"
        + SCRIPT[start:end]
        + '\ntest "$FLAGCX_TORCH_BACKEND" = flagos\ntest "$TORCH_DEVICE_BACKEND_AUTOLOAD" = 0',
        FLAGOS_VENDOR_TORCH_LIB="/flagos/vendor-torch/lib",
        VENV_PYTHON=str(python),
        REPO_ROOT=str(ROOT),
        FLAGCX_VERSION_metax="0.14.0+maca3.8.1.3",
    )
    assert result.returncode == 0, result.stderr
    assert ("install-flagcx" in result.stdout) == (not prepared)
    assert ("Using prepared FlagCX" in result.stdout) == prepared
