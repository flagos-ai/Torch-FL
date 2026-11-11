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

"""Keep runtime shell initialization from replacing the isolated CI Python."""

import os
import base64
import csv
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
HOOK = (ROOT / ".github/scripts/hooks/set_env_ascend.sh").read_text()
COMMON = ROOT / ".github/scripts/lib/set_env_common.sh"
SPEC = importlib.util.spec_from_file_location(
    "seed_runtime", ROOT / ".github/scripts/seed_runtime.py"
)
SEED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEED)


def test_ci_python_survives_a_new_workflow_shell(tmp_path):
    """Replay an image's PATH/PYTHONPATH reset, then the real hook's env export."""
    image_bin = tmp_path / "image/bin"
    ci_bin = tmp_path / "ci/bin"
    for directory, name in ((image_bin, "image"), (ci_bin, "ci")):
        directory.mkdir(parents=True)
        python = directory / "python"
        python.write_text(f"#!/bin/sh\nprintf '%s\\n' '{name}'\n")
        python.chmod(0o755)

    marker = tmp_path / "image-starts"
    startup = tmp_path / "bash_env.sh"
    startup.write_text(
        f"printf 'start\\n' >> {shlex.quote(str(marker))}\n"
        f"export PATH={shlex.quote(str(image_bin))}:/usr/bin:/bin\n"
        "export PYTHONPATH=/opt/flagtree:/image/sdk\n"
        "export ASCEND_HOME_PATH=/image/sdk\n"
        "export ASCEND_OPP_PATH=/image/sdk/opp\n"
        "export ASCEND_TOOLKIT_HOME=/image/sdk\n"
        "export TOOLCHAIN_HOME=/image/sdk/toolkit\n"
        "export ATB_HOME_PATH=/image/atb\n"
        "export CMAKE_PREFIX_PATH=/image/sdk/lib/cmake\n"
    )
    env_file = tmp_path / "github-env"
    path_file = tmp_path / "github-path"
    persist = HOOK.split('if [[ -n "${GITHUB_PATH:-}" ]]', 1)[1]
    persist = (
        'if [[ -n "${GITHUB_PATH:-}" ]]' + persist.split('\ncd "$REPO_ROOT"', 1)[0]
    )
    names = persist.split("export_ci_env ", 1)[1].splitlines()[0].split()
    env = {name: "" for name in names}
    env.update(
        PATH="/usr/bin:/bin",
        BASH_ENV=str(startup),
        GITHUB_ENV=str(env_file),
        GITHUB_PATH=str(path_file),
        VENV_ROOT=str(ci_bin.parent),
        VIRTUAL_ENV=str(ci_bin.parent),
        ASCEND_HOME="/image/sdk",
        LD_LIBRARY_PATH="/image/sdk/lib:/host/driver/lib",
    )
    isolation = HOOK.split("# --- Python isolation", 1)[1]
    isolation = (
        "# --- Python isolation" + isolation.split("# --- CANN toolkit root", 1)[0]
    )
    prepared = subprocess.run(
        [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            "set -euo pipefail\n"
            + isolation
            + '\nexport PATH="$VENV_ROOT/bin:$PATH"\n'
            + f"source {shlex.quote(str(COMMON))}\n"
            + persist,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert not prepared.stderr
    assert marker.read_text().splitlines() == ["start"]
    next_env = {
        **env,
        **dict(line.split("=", 1) for line in env_file.read_text().splitlines()),
    }
    next_env["PATH"] = path_file.read_text().strip() + os.pathsep + next_env["PATH"]
    result = subprocess.run(
        [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            'set -euo pipefail; python; test -z "$PYTHONPATH"; '
            'test "$PYTHONNOUSERSITE" = 1; test "$ASCEND_HOME" = /image/sdk; '
            'test "$ASCEND_HOME_PATH" = /image/sdk; '
            'test "$ASCEND_OPP_PATH" = /image/sdk/opp; '
            'test "$ASCEND_TOOLKIT_HOME" = /image/sdk; '
            'test "$TOOLCHAIN_HOME" = /image/sdk/toolkit; '
            'test "$ATB_HOME_PATH" = /image/atb; '
            'test "$CMAKE_PREFIX_PATH" = /image/sdk/lib/cmake; '
            'test "$LD_LIBRARY_PATH" = /image/sdk/lib:/host/driver/lib',
        ],
        env=next_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "ci"
    assert marker.read_text().splitlines() == ["start"]


def runtime_distribution(root, name="torch", version="2.10.0+cpu", extra=()):
    root.mkdir(parents=True, exist_ok=True)
    package = root / name
    package.mkdir()
    module = package / "__init__.py"
    # Seeding must not import an accelerator package just to inspect it.
    module.write_text("raise RuntimeError('source package imported')\n")
    native = package / "compiler"
    native.write_text("#!/bin/sh\nexit 0\n")
    native.chmod(0o755)
    info = root / f"{name}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    (info / "entry_points.txt").write_text("[console_scripts]\ntorchrun = torch:main\n")
    files = [
        module.relative_to(root),
        native.relative_to(root),
        info.relative_to(root) / "METADATA",
        info.relative_to(root) / "entry_points.txt",
        info.relative_to(root) / "RECORD",
    ]
    for relative in extra:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unsafe\n")
        files.append(Path(relative))
    with (info / "RECORD").open("w", newline="") as output:
        csv.writer(output).writerows((str(path), "", "") for path in files)
    return metadata.Distribution.at(info)


def test_runtime_copy_preserves_isolation_and_executable_files(tmp_path):
    source = tmp_path / "runtime"
    dist = runtime_distribution(source)
    (source / "torch_npu").mkdir()
    (source / "torch_npu/__init__.py").write_text("raise RuntimeError('leaked')\n")
    venv = tmp_path / "ci"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True
    )
    target = next((venv / "lib").glob("python*/site-packages"))
    binary = venv / "bin"
    python = binary / "python"
    assert SEED.copy_distribution(dist, target, binary, python)
    assert not (target / "torch_npu").exists()
    assert not list(target.rglob("*.pth"))
    assert not any(path.is_symlink() for path in target.rglob("*"))
    assert os.access(target / "torch/compiler", os.X_OK)
    assert (binary / "torchrun").read_text().startswith(f"#!{python}\n")
    assert metadata.Distribution.at(target / dist._path.name).version == "2.10.0+cpu"
    (target / "torch/__init__.py").write_text("def main():\n    print('CI copy')\n")
    console = subprocess.run(
        [str(binary / "torchrun")],
        env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert console.stdout.strip() == "CI copy"
    (target / "torch/__init__.py").unlink()
    assert (source / "torch/__init__.py").is_file()


@pytest.mark.parametrize("extra", [("inject.pth",), ("torch_npu/__init__.py",)])
def test_runtime_copy_refuses_import_hooks_and_vendor_packages(tmp_path, extra):
    dist = runtime_distribution(tmp_path / "runtime", extra=extra)
    target = tmp_path / "ci/site-packages"
    assert not SEED.copy_distribution(
        dist, target, tmp_path / "bin", Path(sys.executable)
    )
    assert not target.exists()


def test_runtime_copy_falls_back_when_recorded_files_are_missing(tmp_path):
    source = tmp_path / "runtime"
    dist = runtime_distribution(source)
    (source / "torch/compiler").unlink()
    target = tmp_path / "ci/site-packages"
    assert not SEED.copy_distribution(
        dist, target, tmp_path / "bin", Path(sys.executable)
    )
    assert not target.exists()


def test_runtime_copy_refuses_unhandled_external_data_files(tmp_path):
    dist = runtime_distribution(tmp_path / "runtime")
    record = dist._path / "RECORD"
    with record.open("a", newline="") as output:
        csv.writer(output).writerow(("../../../share/required-tool-data", "", ""))
    target = tmp_path / "ci/site-packages"
    assert not SEED.copy_distribution(
        dist, target, tmp_path / "bin", Path(sys.executable)
    )
    assert not target.exists()


def test_runtime_copy_regenerates_recorded_console_scripts(tmp_path):
    dist = runtime_distribution(tmp_path / "runtime")
    with (dist._path / "RECORD").open("a", newline="") as output:
        csv.writer(output).writerow(("../../../bin/torchrun", "", ""))
    binary = tmp_path / "ci/bin"
    binary.mkdir(parents=True)
    assert SEED.copy_distribution(
        dist, tmp_path / "ci/site-packages", binary, binary / "python"
    )
    assert (binary / "torchrun").is_file()


@pytest.mark.parametrize(
    "recorded_script", [None, "../../../bin/torchrun", "../../bin/torchrun"]
)
def test_copied_record_tracks_rewritten_scripts_and_pip_uninstall(
    tmp_path, recorded_script
):
    source = tmp_path / "runtime/lib/python3.10/site-packages"
    dist = runtime_distribution(source)
    (source / "torch/__init__.py").write_text("def main():\n    print('copied tool')\n")
    record = dist._path / "RECORD"
    rows = list(csv.reader(record.open()))
    if recorded_script:
        script = source / recorded_script
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/old/python\nfrom torch import main\nmain()\n")
        rows.append([recorded_script, "", ""])
    for row in rows:
        if row[0].endswith(".dist-info/RECORD"):
            continue
        content = (source / row[0]).read_bytes()
        row[1] = "sha256=" + base64.urlsafe_b64encode(
            hashlib.sha256(content).digest()
        ).decode().rstrip("=")
        row[2] = str(len(content))
    with record.open("w", newline="") as output:
        csv.writer(output).writerows(rows)
    original = {
        (source / row[0]).resolve(): (source / row[0]).read_bytes() for row in rows
    }
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    target = next((venv / "lib").glob("python*/site-packages"))
    python = venv / "bin/python"
    assert SEED.copy_distribution(dist, target, venv / "bin", python)
    installed_record = target / dist._path.name / "RECORD"
    installed_rows = list(csv.reader(installed_record.open()))
    for name, digest, size in installed_rows:
        path = target / name
        assert path.is_file(), name
        content = path.read_bytes()
        if digest:
            algorithm, value = digest.split("=", 1)
            actual = (
                base64.urlsafe_b64encode(hashlib.new(algorithm, content).digest())
                .decode()
                .rstrip("=")
            )
            assert actual == value, name
        if size:
            assert int(size) == len(content), name
    launcher = venv / "bin/torchrun"
    relative_launcher = os.path.relpath(launcher, target)
    assert sum(row[0] == relative_launcher for row in installed_rows) == 1
    assert [str(installed_record.relative_to(target)), "", ""] in installed_rows
    executed = subprocess.run(
        [str(launcher)], capture_output=True, text=True, check=True
    )
    assert executed.stdout.strip() == "copied tool"
    removed = subprocess.run(
        [str(python), "-I", "-m", "pip", "uninstall", "-y", "torch"],
        capture_output=True,
        text=True,
    )
    assert removed.returncode == 0, removed.stdout + removed.stderr
    assert not launcher.exists()
    assert not (target / "torch").exists()
    assert not installed_record.parent.exists()
    assert all(path.read_bytes() == content for path, content in original.items())


def test_runtime_copy_does_not_overwrite_an_existing_console_command(tmp_path):
    dist = runtime_distribution(tmp_path / "runtime")
    binary = tmp_path / "venv/bin"
    binary.mkdir(parents=True)
    launcher = binary / "torchrun"
    launcher.write_text("existing command\n")
    target = tmp_path / "venv/site-packages"
    assert not SEED.copy_distribution(dist, target, binary, binary / "python")
    assert launcher.read_text() == "existing command\n"
    assert not target.exists()


@pytest.mark.parametrize("broken_dependencies", [False, True])
def test_ascend_final_dependency_check_precedes_report_and_build(
    tmp_path, broken_dependencies
):
    script = HOOK.split(
        "# Validate the complete environment after copying and filling dependencies.\n",
        1,
    )[1]
    script = script.split("# --- Verify the isolation held", 1)[0]
    calls = tmp_path / "calls"
    python = tmp_path / "python"
    python.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
        'if [ "$*" = "-m pip check" ]; then exit "$CHECK_STATUS"; fi\n'
    )
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + script + '\necho build >> "$CALL_LOG"'],
        env={
            "PATH": "/usr/bin:/bin",
            "VENV_PYTHON": str(python),
            "REPO_ROOT": str(ROOT),
            "CALL_LOG": str(calls),
            "CHECK_STATUS": "47" if broken_dependencies else "0",
        },
        capture_output=True,
        text=True,
    )
    rows = calls.read_text().splitlines()
    assert rows[0] == "-m pip check"
    assert (result.returncode != 0) == broken_dependencies
    assert any("--report-environment" in row for row in rows) == (
        not broken_dependencies
    )
    assert ("build" in rows) == (not broken_dependencies)


def test_environment_report_uses_metadata_without_importing_packages(
    tmp_path, monkeypatch
):
    dist = runtime_distribution(
        tmp_path / "runtime", name="vendor_demo", version="1.2.3"
    )
    monkeypatch.setattr(SEED.metadata, "distributions", lambda: [dist])
    monkeypatch.setattr(
        SEED.shutil, "which", lambda name: "/tools/cmake" if name == "cmake" else None
    )
    monkeypatch.setattr(
        SEED.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout="cmake version 3.31.10\n", stderr="", returncode=0
        ),
    )
    report = SEED.environment_report()
    assert report["python"]["executable"] == sys.executable
    assert report["packages"] == [
        {
            "name": "vendor_demo",
            "version": "1.2.3",
            "location": str(dist.locate_file("").resolve()),
        }
    ]
    assert report["tools"]["cmake"] == {
        "path": "/tools/cmake",
        "version": "cmake version 3.31.10",
        "exit_code": 0,
    }
    assert report["tools"]["gcc"] == {"path": None}


def test_environment_report_bounds_tool_probes(tmp_path, monkeypatch):
    monkeypatch.setattr(SEED.metadata, "distributions", lambda: [])
    monkeypatch.setattr(SEED.shutil, "which", lambda name: "/tools/" + name)

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 5
        raise subprocess.TimeoutExpired(command, 5)

    monkeypatch.setattr(SEED.subprocess, "run", timeout)
    report = SEED.environment_report()
    assert all(tool["error"] == "TimeoutExpired" for tool in report["tools"].values())


@pytest.mark.parametrize(
    "dependency_version,optional,expected",
    [
        (None, False, False),
        ("1.0", False, False),
        ("2.0", False, True),
        (None, True, True),
    ],
)
def test_prepared_requirement_checks_its_dependencies(
    tmp_path, monkeypatch, dependency_version, optional, expected
):
    root = tmp_path / "packages"
    parent = runtime_distribution(root, name="prepared", version="1.0")
    marker = '; extra == "test"' if optional else ""
    with (parent._path / "METADATA").open("a") as output:
        output.write(f"Requires-Dist: dependency>=2{marker}\n")
    distributions = {"prepared": parent}
    if dependency_version:
        distributions["dependency"] = runtime_distribution(
            root, name="dependency", version=dependency_version
        )

    def find(name):
        if name not in distributions:
            raise metadata.PackageNotFoundError(name)
        return distributions[name]

    monkeypatch.setattr(SEED.metadata, "distribution", find)
    assert SEED.requirement_satisfied("prepared==1.0") == expected
    assert not SEED.requirement_satisfied("prepared==2.0")


def test_satisfied_tool_does_not_call_pip_or_probe_the_network(tmp_path):
    script = HOOK.split("while IFS= read -r requirement", 1)[1]
    end = 'done < "$REPO_ROOT/.github/requirements-ascend.txt"'
    script = "while IFS= read -r requirement" + script.split(end, 1)[0] + end
    source = tmp_path / "source/.github"
    source.mkdir(parents=True)
    (source / "requirements-ascend.txt").write_text("wheel\n")
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            "set -euo pipefail\npip_retry() { echo unexpected-network >&2; return 99; }\n"
            + script,
        ],
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_ROOT": str(source.parent),
            "VENV_PYTHON": str(python),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Using prepared requirement: wheel" in result.stdout
    assert not result.stderr


@pytest.mark.parametrize(
    "version,reused",
    [("0.14.0+cann9.0.0.910c", True), ("0.14.0+cann9.0.0", False)],
)
def test_flagcx_reuse_and_install_both_select_the_flagos_backend(
    tmp_path, version, reused
):
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True
    )
    target = next((venv / "lib").glob("python*/site-packages"))
    runtime_distribution(target, name="flagcx", version=version)
    script = HOOK.split("# Reuse the matching FlagCX", 1)[1]
    script = (
        "# Reuse the matching FlagCX" + script.split("# Matching runtime tools", 1)[0]
    )
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            "set -euo pipefail\ninstall_flagcx() { echo download; export FLAGCX_TORCH_BACKEND=flagos; }\n"
            + script
            + '\ntest "$FLAGCX_TORCH_BACKEND" = flagos',
        ],
        env={
            "PATH": "/usr/bin:/bin",
            "VENV_PYTHON": str(venv / "bin/python"),
            "FLAGCX_VERSION_ascend": "0.14.0+cann9.0.0.910c",
            "PYTHONNOUSERSITE": "1",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert ("Using runtime-seeded FlagCX" in result.stdout) == reused
    assert ("download" in result.stdout) == (not reused)


def test_runtime_copy_does_not_overwrite_existing_target_packages(tmp_path):
    dist = runtime_distribution(tmp_path / "runtime")
    target = tmp_path / "ci/site-packages"
    (target / "torch").mkdir(parents=True)
    existing = target / "torch/__init__.py"
    existing.write_text("existing package\n")
    assert not SEED.copy_distribution(
        dist, target, tmp_path / "bin", Path(sys.executable)
    )
    assert existing.read_text() == "existing package\n"
    assert not list(target.glob("*.dist-info"))


@pytest.mark.parametrize(
    "name,version,requirement,reused",
    [
        ("pybind11", "3.0.3", "pybind11", True),
        ("PyYAML", "6.0.1", "PyYAML==6.0.1", True),
        ("PyYAML", "6.0.2", "PyYAML==6.0.1", False),
        ("packaging", "26.0", "packaging>=26.0", True),
        ("packaging", "25.0", "packaging>=26.0", False),
        ("sqlalchemy", "2.0.55", "sqlalchemy==2.0.48", False),
        ("wheel", "0.45.1", "wheel", True),
    ],
)
def test_tool_reuse_obeys_the_same_requirements_as_pip(
    tmp_path, monkeypatch, name, version, requirement, reused
):
    dist = runtime_distribution(tmp_path / "runtime", name=name, version=version)
    target = tmp_path / "ci/site-packages"
    binary = tmp_path / "ci/bin"
    binary.mkdir(parents=True)
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("# tools\n\n" + requirement + "\n")
    monkeypatch.setattr(SEED.metadata, "distributions", lambda **kwargs: [dist])
    monkeypatch.setattr(
        SEED.subprocess,
        "check_output",
        lambda *args, **kwargs: json.dumps(
            {"site": str(target), "version": list(sys.version_info[:2])}
        ),
    )
    SEED.seed(
        binary / "python",
        "2.10.0",
        "0.7.0+ascend3.5",
        "5.4.0",
        "0.14.0+cann9.0.0.910c",
        requirements,
    )
    assert target.exists() == reused


def test_installed_check_works_with_only_venv_pip(tmp_path):
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    command = [
        str(venv / "bin/python"),
        str(ROOT / ".github/scripts/seed_runtime.py"),
        "--check-installed",
    ]
    env = {"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"}
    assert subprocess.run(command + ["pip>=1"], env=env).returncode == 0
    assert subprocess.run(command + ["nonexistent-ci-tool==1"], env=env).returncode == 1


@pytest.mark.parametrize(
    "name,version,python_version,reused",
    [
        ("torch", "2.13.0+cpu", list(sys.version_info[:2]), False),
        ("torch", "2.10.0+corex.4.5.0.20260804", list(sys.version_info[:2]), False),
        ("torch", "2.10.0+cpu", [3, 9], False),
        ("torch", "2.10.0+cpu", list(sys.version_info[:2]), True),
        ("flagcx", "0.14.0+cann9.0.0", list(sys.version_info[:2]), False),
        ("flagcx", "0.14.0+cann9.0.0.910c", list(sys.version_info[:2]), True),
    ],
)
def test_seed_requires_matching_versions_and_python(
    tmp_path, monkeypatch, name, version, python_version, reused
):
    dist = runtime_distribution(tmp_path / "runtime", name=name, version=version)
    target = tmp_path / "ci/site-packages"
    binary = tmp_path / "ci/bin"
    binary.mkdir(parents=True)
    monkeypatch.setattr(SEED.metadata, "distributions", lambda **kwargs: [dist])
    monkeypatch.setattr(
        SEED.subprocess,
        "check_output",
        lambda *args, **kwargs: json.dumps(
            {"site": str(target), "version": python_version}
        ),
    )
    SEED.seed(
        binary / "python", "2.10.0", "0.7.0+ascend3.5", "5.4.0", "0.14.0+cann9.0.0.910c"
    )
    assert target.exists() == reused


def test_a_lane_without_flagcx_keeps_the_runtime_communication_package_out(
    tmp_path, monkeypatch
):
    dist = runtime_distribution(
        tmp_path / "runtime", name="flagcx", version="0.14.0+corex4.5.0"
    )
    target = tmp_path / "ci/site-packages"
    binary = tmp_path / "ci/bin"
    binary.mkdir(parents=True)
    monkeypatch.setattr(SEED.metadata, "distributions", lambda **kwargs: [dist])
    monkeypatch.setattr(
        SEED.subprocess,
        "check_output",
        lambda *args, **kwargs: json.dumps(
            {"site": str(target), "version": list(sys.version_info[:2])}
        ),
    )
    SEED.seed(binary / "python", "2.10.0", "0.7.0+iluvatar3.6", "5.4.0")
    assert not target.exists()


@pytest.mark.parametrize(
    "name,version,reused",
    [
        ("flagtree", "0.7.1a1+iluvatar3.6", True),
        ("flagtree", "0.7.0+iluvatar3.6", False),
        ("sqlalchemy", "2.0.55", True),
        ("sqlalchemy", "2.1.0", False),
    ],
)
def test_iluvatar_reuses_image_versions_with_checkout_requirements(
    tmp_path, monkeypatch, name, version, reused
):
    dist = runtime_distribution(tmp_path / "runtime", name=name, version=version)
    target = tmp_path / "ci/site-packages"
    binary = tmp_path / "ci/bin"
    binary.mkdir(parents=True)
    pins = dict(
        line.split("=", 1)
        for line in (ROOT / ".github/version-pins.env").read_text().splitlines()
        if line and not line.startswith("#")
    )
    monkeypatch.setattr(SEED.metadata, "distributions", lambda **kwargs: [dist])
    monkeypatch.setattr(
        SEED.subprocess,
        "check_output",
        lambda *args, **kwargs: json.dumps(
            {"site": str(target), "version": list(sys.version_info[:2])}
        ),
    )
    SEED.seed(
        binary / "python",
        pins["CPU_TORCH_VERSION_DEFAULT"],
        pins["FLAGTREE_VERSION_iluvatar"],
        pins["FLAGGEMS_VERSION_DEFAULT"],
        requirements_file=ROOT / ".github/requirements-iluvatar.txt",
    )
    assert target.exists() == reused
