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

"""Validate shared runtime copying, CI version ranges and final environment checks."""

import os
import base64
import csv
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "seed_runtime", ROOT / ".github/scripts/seed_runtime.py"
)
SEED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEED)


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


@pytest.fixture(params=[False, True], ids=["native-metadata", "filtered-metadata"])
def metadata_filters_missing(request, monkeypatch):
    """Exercise the Python 3.12 metadata behavior on older test interpreters too."""
    if request.param:
        original = metadata.Distribution.files.fget
        monkeypatch.setattr(
            metadata.Distribution,
            "files",
            property(
                lambda dist: [
                    entry
                    for entry in original(dist) or []
                    if dist.locate_file(entry).exists()
                ]
            ),
        )


def test_runtime_copy_falls_back_when_recorded_files_are_missing(
    tmp_path, metadata_filters_missing
):
    source = tmp_path / "runtime"
    dist = runtime_distribution(source)
    (source / "torch/compiler").unlink()
    target = tmp_path / "ci/site-packages"
    assert not SEED.copy_distribution(
        dist, target, tmp_path / "bin", Path(sys.executable)
    )
    assert not target.exists()


def test_runtime_copy_refuses_unhandled_external_data_files(
    tmp_path, metadata_filters_missing
):
    dist = runtime_distribution(tmp_path / "runtime")
    record = dist._path / "RECORD"
    with record.open("a", newline="") as output:
        csv.writer(output).writerow(("../../../share/required-tool-data", "", ""))
    target = tmp_path / "ci/site-packages"
    assert not SEED.copy_distribution(
        dist, target, tmp_path / "bin", Path(sys.executable)
    )
    assert not target.exists()


def test_runtime_copy_regenerates_recorded_console_scripts(
    tmp_path, metadata_filters_missing
):
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
    monkeypatch.setenv(
        "PIP_CONSTRAINT", str(ROOT / ".github/constraints-runtime-ci.txt")
    )
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


@pytest.mark.parametrize(
    "platform",
    [
        "ascend",
        "iluvatar",
        "kunlunxin",
        "mthreads",
        "thead",
        "enflame",
        "metax",
        "hygon",
    ],
)
@pytest.mark.parametrize(
    "version,accepted",
    [("2.0.48", False), ("2.0.55", True), ("2.0.57", True), ("2.0.58", False)],
)
def test_runtime_ranges_agree_for_copy_and_installed_checks(
    tmp_path, monkeypatch, platform, version, accepted
):
    """All migrated lanes accept both image cohorts and reject out-of-range ones."""
    monkeypatch.setenv(
        "PIP_CONSTRAINT", str(ROOT / ".github/constraints-runtime-ci.txt")
    )
    dist = runtime_distribution(
        tmp_path / "runtime", name="SQLAlchemy", version=version
    )
    target = tmp_path / "ci/site-packages"
    binary = tmp_path / "ci/bin"
    binary.mkdir(parents=True)
    monkeypatch.setattr(SEED.metadata, "distributions", lambda **kwargs: [dist])
    monkeypatch.setattr(SEED.metadata, "distribution", lambda name: dist)
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
        "unused-platform-compiler",
        "5.4.0",
        requirements_file=ROOT / f".github/requirements-{platform}.txt",
    )
    assert target.exists() == accepted
    assert SEED.requirement_satisfied("sqlalchemy>=1.4.31,<2.1") == accepted


@pytest.mark.parametrize("version,accepted", [("2.0.48", False), ("2.0.57", True)])
def test_installed_dependency_closure_obeys_runtime_constraints(
    tmp_path, monkeypatch, version, accepted
):
    monkeypatch.setenv(
        "PIP_CONSTRAINT", str(ROOT / ".github/constraints-runtime-ci.txt")
    )
    parent = runtime_distribution(tmp_path / "runtime", name="prepared", version="1.0")
    with (parent._path / "METADATA").open("a") as stream:
        stream.write("Requires-Dist: SQLAlchemy>=1.4.31,<2.1\n")
    dependency = runtime_distribution(
        tmp_path / "runtime", name="SQLAlchemy", version=version
    )
    monkeypatch.setattr(
        SEED.metadata,
        "distribution",
        lambda name: parent if name == "prepared" else dependency,
    )
    assert SEED.requirement_satisfied("prepared") == accepted


@pytest.mark.parametrize(
    "platform,accepted",
    [
        ("ascend", True),
        ("thead", True),
        ("metax", False),
        ("mthreads", False),
        ("hygon", False),
        ("enflame", False),
        ("kunlunxin", False),
        ("iluvatar", False),
    ],
)
def test_runtime_ranges_preserve_platform_cmake_restrictions(
    monkeypatch, platform, accepted
):
    monkeypatch.setenv(
        "PIP_CONSTRAINT", str(ROOT / ".github/constraints-runtime-ci.txt")
    )
    limits = SEED.tool_requirements(ROOT / f".github/requirements-{platform}.txt")
    assert limits["cmake"].contains("3.31.10")
    assert limits["cmake"].contains("4.4.4") == accepted
    assert not limits["cmake"].contains("5.0.0")


def test_nested_constraints_keep_markers_and_intersect_repeated_limits(
    tmp_path, monkeypatch
):
    directory = tmp_path / "nested constraints"
    directory.mkdir()
    (directory / "base.txt").write_text("wheel>=0.45\n")
    constraints = tmp_path / "constraints.txt"
    constraints.write_text(
        '-c "nested constraints/base.txt"\n'
        'wheel<0.49; python_version >= "3.10"\n'
        'wheel==0.1; python_version < "3"\n'
    )
    monkeypatch.setenv("PIP_CONSTRAINT", str(constraints))
    limits = SEED.constraint_requirements()["wheel"]
    assert limits.contains("0.45.1") and limits.contains("0.48.0")
    assert not limits.contains("0.44.0") and not limits.contains("0.49.0")
    constraints.write_text('-c "nested constraints/base.txt"\n-c constraints.txt\n')
    with pytest.raises(ValueError, match="Cyclic CI constraints"):
        SEED.constraint_requirements()


@pytest.mark.parametrize(
    "name,version,accepted",
    [
        ("numpy", "1.25.2", False),
        ("numpy", "1.26.4", True),
        ("numpy", "2.0.0", False),
        ("torch", "2.13.0+cpu", False),
        ("torch", "2.10.0+corex.4.5.0.20260804", False),
        ("torch", "2.10.0+cpu", True),
    ],
)
def test_runtime_ranges_do_not_bypass_numpy_or_torch_guards(
    tmp_path, monkeypatch, name, version, accepted
):
    monkeypatch.setenv(
        "PIP_CONSTRAINT", str(ROOT / ".github/constraints-runtime-ci.txt")
    )
    dist = runtime_distribution(tmp_path / "runtime", name=name, version=version)
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
    SEED.seed(binary / "python", "2.10.0", "unused-platform-compiler", "5.4.0")
    assert target.exists() == accepted


def test_real_pip_fallback_uses_the_same_runtime_range_without_network(tmp_path):
    """Local wheel candidates exercise pip's actual nested-constraint resolver."""
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for version in ["2.0.48", "2.0.55", "2.0.57", "2.0.58"]:
        info = f"sqlalchemy-{version}.dist-info"
        with zipfile.ZipFile(
            wheels / f"sqlalchemy-{version}-py3-none-any.whl", "w"
        ) as wheel:
            wheel.writestr(
                f"{info}/METADATA",
                f"Metadata-Version: 2.1\nName: SQLAlchemy\nVersion: {version}\n",
            )
            wheel.writestr(
                f"{info}/WHEEL",
                "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            )
            wheel.writestr(f"{info}/RECORD", "")
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    target = tmp_path / "installed"
    command = [
        str(venv / "bin/python"),
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--find-links",
        str(wheels),
        "--target",
        str(target),
    ]
    env = {
        **os.environ,
        "PIP_CONSTRAINT": str(ROOT / ".github/constraints-runtime-ci.txt"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    result = subprocess.run(
        command + ["SQLAlchemy"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    installed = list(metadata.distributions(path=[str(target)]))
    assert len(installed) == 1 and installed[0].version == "2.0.57"
    result = subprocess.run(
        command + ["SQLAlchemy==2.0.58"], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "conflict" in result.stderr.lower()
    # A compatible image copy must not be upgraded merely because pip is run.
    command = command[:-2]
    for requirement in ["SQLAlchemy==2.0.55", "SQLAlchemy"]:
        result = subprocess.run(
            command + [requirement], env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr
    version = subprocess.check_output(
        [
            str(venv / "bin/python"),
            "-c",
            "from importlib.metadata import version; print(version('SQLAlchemy'))",
        ],
        env=env,
        text=True,
    )
    assert version.strip() == "2.0.55"


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "ci-ranges",
        "ci-tools",
        "wheel-identity",
        "pip-check",
        "installed-wheel",
        "flagcx-import",
        "timeout",
    ],
)
def test_final_environment_records_all_installs_and_retains_failures(
    tmp_path, monkeypatch, failure
):
    wheel = tmp_path / "torch_fl.whl"
    manifest = {
        "platform": "metax",
        "wheel_version": "2.10.0+maca3.8.1.3",
        "build": {"sdk_version": "3.8.1.3"},
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("torch_fl/compatibility.json", json.dumps(manifest))
    output = tmp_path / "reports/environment.json"
    monkeypatch.setenv(
        "PIP_CONSTRAINT", str(ROOT / ".github/constraints-runtime-ci.txt")
    )
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "metax")
    monkeypatch.setenv("FLAGOS_SDK_VERSION", "3.8.1.3")
    monkeypatch.setenv("SOURCE_REF", "tested-commit")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-recorded")
    monkeypatch.setattr(
        SEED,
        "environment_report",
        lambda: {
            "schema_version": "torch-fl.ci-environment/v1",
            "packages": [
                {
                    "name": "torch-fl",
                    "version": "2.10.1+maca3.8.1.3"
                    if failure == "wheel-identity"
                    else "2.10.0+maca3.8.1.3",
                },
                {"name": "coverage", "version": "7.10.7"},
            ],
        },
    )
    monkeypatch.setattr(
        SEED.metadata,
        "distribution",
        lambda name: SimpleNamespace(version="0.14.0+maca3.8.1.3"),
    )
    monkeypatch.setattr(
        SEED.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(
                metadata={"Name": "SQLAlchemy"},
                version="2.0.48" if failure == "ci-ranges" else "2.0.57",
            )
        ],
    )
    calls = []
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("build\n")
    monkeypatch.setattr(
        SEED, "requirement_satisfied", lambda value: failure != "ci-tools"
    )

    def run(command, **kwargs):
        calls.append(command)
        assert Path(kwargs["cwd"]).is_dir()
        assert Path(kwargs["cwd"]).resolve() != ROOT
        assert kwargs["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        name = (
            "flagcx-import"
            if "-c" in command
            else "pip-check"
            if command[-1] == "check"
            else "installed-wheel"
        )
        return SimpleNamespace(
            returncode=int(name == failure),
            stdout=name,
            stderr="failed" if name == failure else "",
        )

    monkeypatch.setattr(SEED.subprocess, "run", run)
    assert SEED.finalize_environment(wheel, output, requirements) == int(
        failure is not None
    )
    report = json.loads(output.read_text())
    assert report["valid"] == (failure is None)
    assert report["phase"] == "before-tests" and report["source_ref"] == "tested-commit"
    assert report["wheel"]["manifest"] == manifest
    assert report["wheel"]["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert {x["name"] for x in report["packages"]} == {"torch-fl", "coverage"}
    assert "must-not-be-recorded" not in output.read_text()
    assert report["constraints"]["requirements"]["sqlalchemy"] == "<2.0.58,>=2.0.55"
    assert report["requirements"]["build"] == "<2,>=1.2"
    assert any(
        "--check-installed" in call and "--check-build-env" in call for call in calls
    )
    assert any("--sdk-version" in call and "3.8.1.3" in call for call in calls)
    assert any("-c" in call for call in calls) == (failure in (None, "flagcx-import"))


def test_final_environment_retains_an_invalid_wheel_report(tmp_path, monkeypatch):
    monkeypatch.setattr(SEED, "environment_report", lambda: {"packages": []})
    wheel = tmp_path / "broken.whl"
    wheel.write_bytes(b"not a wheel")
    output = tmp_path / "reports/environment.json"
    assert SEED.finalize_environment(wheel, output) == 1
    report = json.loads(output.read_text())
    assert not report["valid"]
    assert report["validation"][0]["name"] == "environment-record"


@pytest.mark.parametrize(
    "platform", ["gcu", "enflame", "musa", "mthreads", "metax", "ascend", "ppu"]
)
def test_only_explicit_platforms_defer_missing_flagcx(monkeypatch, platform):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(SEED.metadata, "distribution", missing)
    manifest = {
        "wheel_version": "2.10.0+test",
        "requirements": ["flagcx==0.14.0+test"],
        "build": {"distributions": {"flagcx": None}},
    }
    checks = [
        {
            "name": "pip-check",
            "exit_code": 1,
            "stdout": "torch-fl 2.10.0+test requires flagcx, which is not installed.",
            "stderr": "",
        },
        {
            "name": "installed-wheel",
            "exit_code": 1,
            "stdout": "",
            "stderr": "ERROR: Required flagcx==0.14.0+test; installed missing\n",
        },
    ]
    accepted = platform in {"gcu", "enflame", "musa", "mthreads"}
    assert SEED.defer_missing_flagcx(checks, manifest, platform) == accepted
    assert (
        all(check.get("deferred_dependency") == "flagcx" for check in checks)
        == accepted
    )


@pytest.mark.parametrize(
    "failure",
    [
        "other-package",
        "version-conflict",
        "pip-error",
        "preflight-error",
        "truncated",
        "build-record",
        "installed",
    ],
)
def test_flagcx_deferral_never_hides_other_failures(monkeypatch, failure):
    def distribution(name):
        if failure == "installed":
            return SimpleNamespace(version="0.14.0+test")
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(SEED.metadata, "distribution", distribution)
    manifest = {
        "wheel_version": "2.10.0+test",
        "requirements": ["flagcx==0.14.0+test"],
        "build": {
            "distributions": {
                "flagcx": "0.14.0+test" if failure == "build-record" else None
            }
        },
    }
    check = {
        "name": "pip-check",
        "exit_code": 1,
        "stdout": "torch-fl 2.10.0+test requires flagcx, which is not installed.",
        "stderr": "",
    }
    if failure == "other-package":
        check["stdout"] += "\nother 1.0 requires dependency, which is not installed."
    if failure == "version-conflict":
        check["stdout"] = (
            "torch-fl 2.10.0+test has requirement flagcx==0.14.0, but you have flagcx 0.13.0."
        )
    if failure == "pip-error":
        check["stderr"] = "No module named pip"
    if failure == "preflight-error":
        check.update(
            name="installed-wheel",
            stdout="",
            stderr="ERROR: Required flagcx==0.14.0+test; installed missing\nERROR: Required torch>=2.10,<2.11; installed 2.13.0",
        )
    if failure == "truncated":
        check["truncated"] = True
    assert not SEED.defer_missing_flagcx([check], manifest, "gcu")
    assert "deferred_dependency" not in check


def test_final_environment_labels_deferred_dependencies_incomplete(
    tmp_path, monkeypatch
):
    version = "2.10.0+tops1.9.10"
    manifest = {
        "platform": "gcu",
        "wheel_version": version,
        "requirements": ["flagcx==0.14.0+tops1.9.10"],
        "build": {"distributions": {"flagcx": None}},
    }
    wheel = tmp_path / "torch_fl.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("torch_fl/compatibility.json", json.dumps(manifest))
    monkeypatch.setattr(
        SEED,
        "environment_report",
        lambda: {"packages": [{"name": "torch-fl", "version": version}]},
    )
    monkeypatch.setattr(SEED.metadata, "distributions", lambda: [])

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(SEED.metadata, "distribution", missing)
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "enflame")

    def run(command, **kwargs):
        if command[-1] == "check":
            return SimpleNamespace(
                returncode=1,
                stdout=f"torch-fl {version} requires flagcx, which is not installed.\n",
                stderr="",
            )
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="ERROR: Required flagcx==0.14.0+tops1.9.10; installed missing\n",
        )

    monkeypatch.setattr(SEED.subprocess, "run", run)
    output = tmp_path / "environment.json"
    assert SEED.finalize_environment(wheel, output) == 0
    report = json.loads(output.read_text())
    assert report["ready_for_tests"] and not report["valid"]
    assert report["deferred_dependencies"][0]["name"] == "flagcx"
    assert [item["exit_code"] for item in report["validation"]] == [0, 1, 1]


@pytest.mark.parametrize("missing_field", ["Name", "Version"])
def test_final_environment_retains_corrupt_package_metadata(
    tmp_path, monkeypatch, missing_field
):
    distribution = runtime_distribution(tmp_path / "runtime", name="broken")
    metadata_file = distribution._path / "METADATA"
    metadata_file.write_text(
        "".join(
            line + "\n"
            for line in metadata_file.read_text().splitlines()
            if not line.startswith(missing_field + ":")
        )
    )
    monkeypatch.setattr(SEED.metadata, "distributions", lambda: [distribution])
    monkeypatch.setattr(SEED.shutil, "which", lambda name: None)
    monkeypatch.setenv("SOURCE_REF", "tested-commit")
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "metax")
    output = tmp_path / "reports/environment.json"
    assert SEED.finalize_environment(tmp_path / "torch_fl.whl", output) == 1
    report = json.loads(output.read_text())
    assert report["schema_version"] == "torch-fl.ci-environment/v1"
    assert report["phase"] == "before-tests"
    assert report["source_ref"] == "tested-commit"
    assert report["runtime"]["FLAGOS_ACCELERATOR"] == "metax"
    assert not report["ready_for_tests"] and not report["valid"]
    failure = report["validation"][0]
    assert failure["name"] == "environment-record" and failure["exit_code"] == 1
    assert "Invalid installed distribution metadata" in failure["stderr"]
    assert str(tmp_path / "runtime") in failure["stderr"]
