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

"""Exercise real pytest processes and native coverage.py data without hardware."""

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts"
SPEC = importlib.util.spec_from_file_location(
    "python_coverage", SCRIPTS / "python_coverage.py"
)
REPORTING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORTING)
PYTHON = shlex.quote(sys.executable)


def _setup(tmp_path, initializer="INITIAL = 1\n"):
    site = tmp_path / "installed wheel"
    package = site / "torch_fl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(initializer)
    (package / "model.py").write_text(
        "def choose(value):\n"
        "    if value:\n"
        "        return 'yes'\n"
        "    return 'no'\n"
        "\n"
        "def unused():\n"
        "    return 'unused'\n"
    )
    (package / "unimported.py").write_text("def untouched():\n    return 42\n")
    metadata = site / "torch_fl-99.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: torch_fl\nVersion: 99.0\n")
    workdir = tmp_path / "wheel-only"
    workdir.mkdir()
    reports = tmp_path / "coverage sessions"
    environment = os.environ.copy()
    for name in (
        "PYTEST_ADDOPTS",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "PYTEST_PLUGINS",
        "FLAGCICD_REPORT_DIR",
        "FLAGCICD_COVERAGE_DIR",
        "FLAGCICD_LAUNCHER",
        "FLAGCICD_TEST_GROUP",
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(site), str(SCRIPTS / "pytest_reporting"))
    )
    env_file = tmp_path / "github-env"
    prepared = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "python_coverage.py"),
            "prepare",
            "--report-dir",
            str(reports),
            "--env-file",
            str(env_file),
        ],
        env=environment,
        cwd=workdir,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    assert env_file.read_text() == f"FLAGCICD_COVERAGE_DIR={reports}\n"
    environment.update(
        FLAGCICD_COVERAGE_DIR=str(reports),
        FLAGCICD_REPORT_DIR=str(tmp_path / "pytest sessions"),
        FLAGCICD_PYTHON_EXECUTABLE=sys.executable,
        FLAGCICD_LAUNCHER="group",
        PYTEST_PLUGINS="flagcicd_report,flagcicd_coverage",
        INTEGRATION_WORKDIR=str(workdir),
        INTEGRATION_ENVIRONMENT="{}",
    )
    return workdir, reports, environment


def _run(environment, groups):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "run_integration_tests.py")],
        env={**environment, "INTEGRATION_TESTS": json.dumps(groups)},
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )


def _group(name, file):
    return {"name": name, "command": f"{PYTHON} -m pytest {file} -q"}


def _merge(reports, outcome="success"):
    output = reports.parent / "coverage.json"
    REPORTING.merge(reports, output, "cuda", outcome, "flagos-ai/Torch-FL", "a" * 40)
    return json.loads(output.read_text())


def _metadata(reports):
    return json.loads((reports / "metadata.json").read_text())


def test_preparation_resolves_installed_wheel_without_importing_it(tmp_path):
    _, reports, _ = _setup(tmp_path, "raise RuntimeError('accelerator imported')\n")
    manifest = json.loads((reports / "manifest.json").read_text())
    assert Path(manifest["package_root"]) == tmp_path / "installed wheel/torch_fl"


def test_groups_union_executed_lines_and_count_unimported_modules(tmp_path):
    workdir, reports, environment = _setup(tmp_path)
    for name, value in (("first", True), ("second", False)):
        (workdir / f"test_{name}.py").write_text(
            "from torch_fl import model\n"
            f"def test_case(): assert model.choose({value!r})\n"
        )
    result = _run(
        environment,
        [_group("first", "test_first.py"), _group("second", "test_second.py")],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = _merge(reports)
    assert report["meta"]["branch_coverage"] is False
    assert set(report["files"]) == {
        "torch_fl/__init__.py",
        "torch_fl/model.py",
        "torch_fl/unimported.py",
    }
    assert report["files"]["torch_fl/model.py"]["executed_lines"] == [1, 2, 3, 4, 6]
    assert report["files"]["torch_fl/model.py"]["missing_lines"] == [7]
    assert report["files"]["torch_fl/unimported.py"]["executed_lines"] == []
    assert report["totals"]["covered_lines"] == 6
    assert report["totals"]["num_statements"] == 9
    assert report["totals"]["percent_covered"] == pytest.approx(100 * 6 / 9)
    assert _metadata(reports)["complete"]
    assert {s["group"] for s in _metadata(reports)["sessions"]} == {"first", "second"}
    assert len(list(Path(environment["FLAGCICD_REPORT_DIR"]).rglob("*.json"))) == 2


def test_collection_imports_and_failed_test_lines_are_measured(tmp_path):
    workdir, reports, environment = _setup(tmp_path)
    (workdir / "conftest.py").write_text(
        "from torch_fl import model\nmodel.choose(True)\n"
    )
    (workdir / "test_failure.py").write_text(
        "from torch_fl import model\n"
        "def test_failure():\n    model.choose(False)\n    assert False\n"
    )
    result = _run(environment, [_group("failure", "test_failure.py")])
    assert result.returncode == 1, result.stdout + result.stderr
    report = _merge(reports, "failure")
    assert report["files"]["torch_fl/model.py"]["executed_lines"] == [1, 2, 3, 4, 6]
    assert not _metadata(reports)["complete"]
    assert _metadata(reports)["sessions"][0]["exitcode"] == 1
    assert _metadata(reports)["incomplete_sessions"] == []


def test_per_file_runner_does_not_collect_fixture_sample_subprocesses(tmp_path):
    workdir, reports, environment = _setup(tmp_path)
    child_source = (
        "from torch_fl import model\n"
        "def test_child():\n    model.choose(False)\n    assert False\n"
    )
    (workdir / "test_first.py").write_text(
        "import os, subprocess, sys\nfrom torch_fl import model\n"
        "def test_parent(tmp_path):\n"
        "    assert 'FLAGCICD_COVERAGE_DIR' not in os.environ\n"
        "    assert model.choose(True)\n"
        "    child = tmp_path / 'test_child.py'\n"
        f"    child.write_text({child_source!r})\n"
        "    result = subprocess.run([sys.executable, '-m', 'pytest', str(child), '-q'])\n"
        "    assert result.returncode == 1\n"
    )
    (workdir / "test_second.py").write_text("def test_other(): pass\n")
    groups = [
        {
            "name": "unit files",
            "command": f"{PYTHON} {shlex.quote(str(SCRIPTS / 'run_unit_tests.py'))} --test-dir .",
        }
    ]
    result = _run(environment, groups)
    assert result.returncode == 0, result.stdout + result.stderr
    report = _merge(reports)
    assert len(_metadata(reports)["sessions"]) == 2
    assert report["files"]["torch_fl/model.py"]["missing_lines"] == [4, 7]
    assert len(list(Path(environment["FLAGCICD_REPORT_DIR"]).rglob("*.json"))) == 2


@pytest.mark.parametrize("damage", ["crash", "corrupt"])
def test_partial_coverage_preserves_completed_data_and_missing_sessions(
    tmp_path, damage
):
    workdir, reports, environment = _setup(tmp_path)
    (workdir / "test_first.py").write_text(
        "from torch_fl import model\ndef test_first(): assert model.choose(True)\n"
    )
    (workdir / "test_second.py").write_text(
        "import os\nfrom torch_fl import model\n"
        "def test_second():\n    model.choose(False)\n"
        + ("    os._exit(3)\n" if damage == "crash" else "    assert True\n")
    )
    result = _run(
        environment,
        [_group("first", "test_first.py"), _group("second", "test_second.py")],
    )
    assert result.returncode == (3 if damage == "crash" else 0)
    if damage == "corrupt":
        for marker in reports.glob("*.pending"):
            if json.loads(marker.read_text())["group"] == "second":
                marker.with_suffix(".data").write_bytes(b"truncated database")
    report = _merge(reports, "failure" if damage == "crash" else "success")
    assert not _metadata(reports)["complete"]
    assert len(_metadata(reports)["incomplete_sessions"]) == 1
    assert report["files"]["torch_fl/model.py"]["missing_lines"] == [4, 7]


def test_no_pytest_or_no_measured_package_does_not_fabricate_coverage(tmp_path):
    workdir, reports, environment = _setup(tmp_path)
    result = _run(environment, [{"name": "environment", "command": "exit 7"}])
    assert result.returncode == 7
    with pytest.raises(ValueError, match="No pytest coverage data"):
        _merge(reports, "failure")
    (workdir / "test_unrelated.py").write_text("def test_pass(): pass\n")
    result = _run(environment, [_group("unrelated", "test_unrelated.py")])
    assert result.returncode == 0, result.stdout + result.stderr
    with pytest.raises(ValueError, match="No installed torch_fl Python files"):
        _merge(reports)


def test_workflow_merger_and_uploader_use_same_container_relative_report(tmp_path):
    workdir, reports, environment = _setup(tmp_path)
    (workdir / "test_good.py").write_text(
        "from torch_fl import model\ndef test_good(): assert model.choose(True)\n"
    )
    result = _run(environment, [_group("good", "test_good.py")])
    assert result.returncode == 0, result.stdout + result.stderr
    job = yaml.safe_load(
        (ROOT / ".github/workflows/integration-tests.yml").read_text()
    )["jobs"]["integration-tests"]
    steps = {s.get("id", s.get("name")): s for s in job["steps"]}
    workspace = tmp_path / "torch-fl-workspace"
    workspace.mkdir()
    result = subprocess.run(
        ["bash", "-e", "-c", steps["merge-coverage"]["run"]],
        cwd=ROOT,
        env={
            **environment,
            "GITHUB_WORKSPACE": str(workspace),
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
            "REPORT_PLATFORM": "cuda",
            "TEST_OUTCOME": "success",
            "SOURCE_REPOSITORY": "flagos-ai/Torch-FL",
            "SOURCE_REF": "b" * 40,
            "PIP_CACHE_DIR": str(tmp_path / "pip"),
            "CONTAINER_CONFIG": "",
        },
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    upload = steps["Upload Python coverage to FlagCICD"]
    assert upload["with"]["file_type"] == "coverage"
    assert upload["with"]["job_name"] == "coverage-${{ inputs.platform }}"
    assert upload["with"]["fail_on_error"] == "false"
    assert upload["continue-on-error"] is True
    assert "steps.merge-coverage.outcome == 'success'" in upload["if"]
    assert "steps.flagcicd-auth.outputs.available == 'true'" in upload["if"]
    filename = (
        upload["with"]["report_path"]
        .replace("${{ env.PIP_CACHE_DIR }}", str(tmp_path / "pip"))
        .replace("${{ inputs.platform }}", "cuda")
    )
    output = Path(filename).resolve()
    report = json.loads(output.read_text())
    assert _metadata(reports)["source_ref"] == "b" * 40
    assert report["totals"]["percent_covered"] > 0
    assert (
        upload["with"]["report_path"] in steps["Retain Python coverage"]["with"]["path"]
    )


def test_uploaded_json_matches_native_coverage_json_export(tmp_path):
    workdir, reports, environment = _setup(tmp_path)
    (workdir / "test_good.py").write_text(
        "from torch_fl import model\ndef test_good(): assert model.choose(True)\n"
    )
    result = _run(environment, [_group("good", "test_good.py")])
    assert result.returncode == 0, result.stdout + result.stderr
    report = _merge(reports)
    package = Path(json.loads((reports / "manifest.json").read_text())["package_root"])
    config = tmp_path / "coverage.ini"
    config.write_text(f"[run]\nsource = {package}\nbranch = false\n")
    native_output = tmp_path / "native-coverage.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "json",
            "--rcfile",
            str(config),
            "--data-file",
            str(reports / "combined.data"),
            "-o",
            str(native_output),
        ],
        cwd=package.parent,
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    native = json.loads(native_output.read_text())
    assert set(report) == {"meta", "files", "totals"}
    report["meta"].pop("timestamp")
    native["meta"].pop("timestamp")
    assert report == native
    for filename, data in report["files"].items():
        assert filename.startswith("torch_fl/")
        assert data["summary"]["covered_lines"] == len(data["executed_lines"])
        assert data["summary"]["missing_lines"] == len(data["missing_lines"])
    assert report["totals"]["percent_covered"] == pytest.approx(
        100 * report["totals"]["covered_lines"] / report["totals"]["num_statements"]
    )


@pytest.mark.parametrize(
    "installed,apt_fails", [(True, False), (False, False), (False, True)]
)
def test_metax_upload_tool_preparation_is_idempotent_and_reports_install_failure(
    tmp_path, installed, apt_fails
):
    job = yaml.safe_load(
        (ROOT / ".github/workflows/integration-tests.yml").read_text()
    )["jobs"]["integration-tests"]
    step = next(
        s for s in job["steps"] if s.get("name") == "Prepare MetaX upload tools"
    )
    assert step["continue-on-error"] is True
    assert "inputs.platform == 'metax'" in step["if"]
    assert "steps.flagcicd-auth.outputs.available == 'true'" in step["if"]
    binaries = tmp_path / "bin"
    binaries.mkdir()
    curl = binaries / "curl"
    curl_body = "#!/bin/sh\nprintf 'curl test stub\\n'\n"
    if installed:
        curl.write_text(curl_body)
        curl.chmod(0o755)
    log = tmp_path / "apt.log"
    apt = binaries / "apt-get"
    apt.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$APT_LOG"\n'
        + (
            "exit 42\n"
            if apt_fails
            else 'if [ "$1" = install ]; then\n'
            f"  printf '%s' {shlex.quote(curl_body)} > \"$CURL_STUB\"\n"
            '  /bin/chmod +x "$CURL_STUB"\nfi\n'
        )
    )
    apt.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", "-e", "-o", "pipefail", "-c", step["run"]],
        env={
            **os.environ,
            "PATH": str(binaries),
            "APT_LOG": str(log),
            "CURL_STUB": str(curl),
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == (42 if apt_fails else 0), result.stderr
    if installed:
        assert not log.exists()
    elif not apt_fails:
        assert log.read_text().splitlines() == [
            "update",
            "install -y --no-install-recommends curl ca-certificates",
        ]
