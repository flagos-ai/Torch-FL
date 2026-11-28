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


def _merge(reports, outcome="success", platform="cuda"):
    output = reports.parent / "coverage.json"
    REPORTING.merge(reports, output, platform, outcome, "flagos-ai/Torch-FL", "a" * 40)
    return json.loads(output.read_text())


def _metadata(reports):
    return json.loads((reports / "metadata.json").read_text())


@pytest.fixture
def native_host_wheel(tmp_path):
    workdir, reports, environment = _setup(tmp_path)
    runner_temp = tmp_path / "runner"
    install = runner_temp / "torch-fl-native-host-install"
    package = install / "torch_fl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from types import SimpleNamespace\n"
        "native_host = SimpleNamespace(_model_runtime=lambda: True)\n"
    )
    (package / "model.py").write_text(
        "def common():\n"
        "    return 'common'\n"
        "def eager():\n"
        "    return 'eager'\n"
        "def sample():\n"
        "    return 'sample'\n"
    )
    evidence = runner_temp / "torch-fl-native-host-evidence"
    evidence.mkdir()
    (evidence / "wheel.json").write_text(
        json.dumps({"package": str(package), "sha256": "1" * 64, "plan": {}})
    )
    environment.update(
        RUNNER_TEMP=str(runner_temp), REPORT_PLATFORM="nvidia", CI_PLAN_CONTEXT=""
    )
    return workdir, reports, environment, install


def test_nvidia_pilot_collects_eager_and_common_child_in_their_own_wheel(
    native_host_wheel,
):
    workdir, reports, environment, install = native_host_wheel
    helper = ROOT / "tests/integration/common/_native_host.py"
    sample = (
        f"import sys; sys.path.insert(0, {str(install)!r}); "
        "from torch_fl import model; model.sample()"
    )
    (workdir / "torch.py").touch()
    (workdir / "test_common.py").write_text(
        "import runpy, subprocess, sys\n"
        "from torch_fl import model\n"
        "def common_child():\n"
        "    assert model.common() == 'common'\n"
        "def test_common(pytestconfig):\n"
        "    assert model.choose(True) == 'yes'\n"
        f"    runpy.run_path({str(helper)!r})['run_native_host_case'](common_child, pytestconfig)\n"
        f"    subprocess.run([sys.executable, '-c', {sample!r}], check=True)\n"
    )
    (workdir / "test_eager.py").write_text(
        "from torch_fl import model\ndef test_eager(): assert model.eager() == 'eager'\n"
    )
    common = {
        **_group("Common regression", "test_common.py"),
        "group_id": "common_regression",
    }
    eager = {
        **_group("Eager operator regression", "test_eager.py"),
        "group_id": "eager_ops_regression",
        "environment": {
            "PYTHONPATH": os.pathsep.join((str(install), environment["PYTHONPATH"]))
        },
    }
    result = _run(environment, [common, eager])
    assert result.returncode == 0, result.stdout + result.stderr
    ordinary = _merge(reports, platform="nvidia")
    native = reports / "native-host"
    report = json.loads((native / "coverage-native-host.json").read_text())
    metadata = json.loads((native / "native-host-metadata.json").read_text())
    assert metadata["complete"]
    assert metadata["wheel_variant"] == "native-host"
    assert metadata["wheel_sha256"] == "1" * 64
    assert {s["group"] for s in metadata["sessions"]} == {
        "Common regression",
        "Eager operator regression",
    }
    assert len(metadata["sessions"]) == 2
    assert report["files"]["torch_fl/model.py"]["executed_lines"] == [1, 2, 3, 4, 5]
    assert report["files"]["torch_fl/model.py"]["missing_lines"] == [6]
    assert report["files"]["torch_fl/__init__.py"]["executed_lines"] == [1, 2]
    assert ordinary["files"]["torch_fl/model.py"]["missing_lines"] == [4, 7]
    assert [s["group"] for s in _metadata(reports)["sessions"]] == ["Common regression"]
    assert list(reports.rglob("metadata.json")) == [reports / "metadata.json"]

    warning = "Common regression: Native-host child uses a different wheel"
    (native / "collection-warnings.txt").write_text(f"{warning}\n")
    _merge(reports, platform="nvidia")
    metadata = json.loads((native / "native-host-metadata.json").read_text())
    assert not metadata["complete"]
    assert metadata["collection_warnings"] == [warning]
    assert _metadata(reports)["complete"]


def test_nvidia_pilot_rejects_wrong_eager_wheel_without_losing_ordinary_coverage(
    native_host_wheel, capsys
):
    workdir, reports, environment, _ = native_host_wheel
    (workdir / "test_good.py").write_text(
        "from torch_fl import model\ndef test_good(): assert model.choose(True)\n"
    )
    result = _run(
        environment,
        [
            _group("Ordinary regression", "test_good.py"),
            {
                **_group("Eager operator regression", "test_good.py"),
                "group_id": "eager_ops_regression",
            },
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    warnings = (reports / "native-host/collection-warnings.txt").read_text()
    assert "Eager operator regression imports a different wheel" in warnings
    _merge(reports, platform="nvidia")
    assert "Cannot report native-host Python coverage" in capsys.readouterr().err
    assert not (reports / "native-host/coverage-native-host.json").exists()
    metadata = _metadata(reports)
    assert metadata["complete"]
    assert [s["group"] for s in metadata["sessions"]] == ["Ordinary regression"]


def test_nvidia_experiment_counts_matching_lines_once_and_isolates_unit_probe(
    native_host_wheel,
):
    workdir, reports, environment, install = native_host_wheel
    ordinary_package = Path(
        json.loads((reports / "manifest.json").read_text())["package_root"]
    )
    native_package = install / "torch_fl"
    shared = (
        "def baseline():\n    return 1\n"
        "def native():\n    return 2\n"
        "def unit():\n    return 3\n"
    )
    for package in (ordinary_package, native_package):
        (package / "shared.py").write_text(shared)
    (native_package / "native_only.py").write_text("VALUE = 1\n")
    (workdir / "test_base.py").write_text(
        "from torch_fl import shared\ndef test_base(): assert shared.baseline() == 1\n"
    )
    (workdir / "test_eager.py").write_text(
        "from torch_fl import model, native_only, shared\n"
        "def test_eager():\n"
        "    assert (shared.baseline(), shared.native()) == (1, 2)\n"
        "    assert model.eager() == 'eager' and native_only.VALUE == 1\n"
    )
    (workdir / "test_units.py").write_text(
        "from torch_fl import shared\n"
        "def test_units(): assert (shared.native(), shared.unit()) == (2, 3)\n"
    )
    eager = {
        **_group("Eager operator regression", "test_eager.py"),
        "group_id": "eager_ops_regression",
        "environment": {
            "PYTHONPATH": os.pathsep.join((str(install), environment["PYTHONPATH"]))
        },
    }
    integration = _run(environment, [_group("Ordinary", "test_base.py"), eager])
    assert integration.returncode == 0, integration.stdout + integration.stderr

    probe = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_units.py"],
        cwd=workdir,
        env={
            **environment,
            "PYTEST_PLUGINS": "flagcicd_coverage",
            "FLAGCICD_NVIDIA_UNIT_PROBE": "1",
            "FLAGCICD_TEST_GROUP": "Wheel unit probe",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    baseline = _merge(reports, platform="nvidia")
    path = "torch_fl/shared.py"
    assert baseline["files"][path]["executed_lines"] == [1, 2, 3, 5]
    summary = reports / "nvidia-experiment.json"
    experiment = json.loads(summary.read_text())
    assert experiment["native_host"]["new_lines_by_file"] == {path: [4]}
    assert (
        experiment["native_host"]["omitted_files"]["torch_fl/model.py"]
        == "different_source"
    )
    assert (
        experiment["native_host"]["omitted_files"]["torch_fl/native_only.py"]
        == "outside_ordinary_wheel"
    )
    assert experiment["unit_probe"]["relative_to"] == "native_host"
    assert experiment["unit_probe"]["new_lines_by_file"] == {path: [6]}
    assert (
        experiment["unit_probe"]["covered_lines"]
        == baseline["totals"]["covered_lines"] + 2
    )
    assert list(reports.rglob("metadata.json")) == [reports / "metadata.json"]
    complete = next((reports / "unit-probe").glob("coverage-*.complete"))
    complete.write_text('{"exitcode":1}')
    assert _merge(reports, platform="nvidia")["totals"] == baseline["totals"]
    experiment = json.loads(summary.read_text())
    assert experiment["native_host"]["added_lines"] == 1
    assert experiment["unit_probe"] == {
        "available": False,
        "reason": "incomplete_evidence",
    }


@pytest.fixture
def scoped_wheel(tmp_path):
    workdir, reports, environment = _setup(tmp_path)
    package = tmp_path / "installed wheel/torch_fl"
    modules = {
        "accelerator/__init__.py": "",
        "accelerator/nvidia/__init__.py": "CUDA = True\n",
        "accelerator/ascend/__init__.py": "ASCEND = True\n",
        "native_host/__init__.py": "",
        "native_host/backends/__init__.py": "",
        "native_host/backends/driver.py": "SHARED = True\n",
        "native_host/backends/metax.py": "METAX = True\n",
        "compile/__init__.py": "",
        "compile/metal.py": "METAL = True\n",
        "compile/flagtree_ascend_policy.py": "ASCEND = True\n",
        "compile/flagtree_ascend_tle.py": "SHARED = True\n",
    }
    for name, source in modules.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    excluded_files = {
        "torch_fl/accelerator/ascend/__init__.py",
        "torch_fl/native_host/backends/metax.py",
        "torch_fl/compile/metal.py",
        "torch_fl/compile/flagtree_ascend_policy.py",
    }
    return workdir, reports, environment, excluded_files


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
    assert (
        len(list(Path(environment["FLAGCICD_REPORT_DIR"]).glob("pytest-*.json"))) == 2
    )


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
    assert (
        len(list(Path(environment["FLAGCICD_REPORT_DIR"]).glob("pytest-*.json"))) == 2
    )


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


def test_workflow_merger_and_uploader_use_same_container_relative_report(
    tmp_path, scoped_wheel
):
    workdir, reports, environment, vendor_files = scoped_wheel
    (workdir / "test_good.py").write_text(
        "from torch_fl import model\n"
        "from torch_fl.accelerator import nvidia\n"
        "from torch_fl.compile import flagtree_ascend_tle\n"
        "def test_good():\n"
        "    assert model.choose(True) and nvidia.CUDA and flagtree_ascend_tle.SHARED\n"
    )
    result = _run(environment, [_group("good", "test_good.py")])
    assert result.returncode == 0, result.stdout + result.stderr
    job = yaml.safe_load(
        (ROOT / ".github/workflows/integration-tests.yml").read_text()
    )["jobs"]["integration-tests"]
    steps = {s.get("id", s.get("name")): s for s in job["steps"]}
    workspace = tmp_path / "container workspace"
    workspace.mkdir()
    result = subprocess.run(
        ["bash", "-e", "-c", steps["merge-coverage"]["run"]],
        cwd=ROOT,
        env={
            **environment,
            "GITHUB_WORKSPACE": str(workspace),
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
            "REPORT_PLATFORM": "nvidia",
            "TEST_OUTCOME": "success",
            "SOURCE_REPOSITORY": "flagos-ai/Torch-FL",
            "SOURCE_REF": "b" * 40,
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
    filename = upload["with"]["report_path"].replace("${{ inputs.platform }}", "nvidia")
    output = workspace / filename
    report = json.loads(output.read_text())
    metadata = _metadata(reports)
    assert metadata["source_ref"] == "b" * 40
    assert metadata["scope_version"] == "nvidia-v1"
    assert metadata["complete"]
    assert set(metadata["excluded_files"]) == vendor_files
    assert metadata["excluded_statements"] == 4
    assert metadata["scope_conflicts"] == []
    full = json.loads((reports / "coverage-full-wheel.json").read_text())
    assert set(full["files"]) - set(report["files"]) == vendor_files
    assert report["files"]["torch_fl/unimported.py"]["summary"]["num_statements"] == 2
    assert report["files"]["torch_fl/unimported.py"]["executed_lines"] == []
    assert "torch_fl/native_host/backends/driver.py" in report["files"]
    assert report["totals"]["covered_lines"] == full["totals"]["covered_lines"]
    assert report["totals"]["num_statements"] == full["totals"]["num_statements"] - 4
    replay = REPORTING.summarize_nvidia(full, metadata, "b" * 40)
    assert replay["applicable"]["available"]
    assert replay["applicable"]["covered_lines"] == report["totals"]["covered_lines"]
    assert replay["applicable"]["num_statements"] == report["totals"]["num_statements"]
    assert replay["applicable"]["percent_covered"] == pytest.approx(
        report["totals"]["percent_covered"]
    )
    with pytest.raises(ValueError, match="Expected full-wheel coverage"):
        REPORTING.summarize_nvidia(report, metadata, "b" * 40)
    assert report["totals"]["percent_covered"] > 0
    assert (
        upload["with"]["report_path"] in steps["Retain Python coverage"]["with"]["path"]
    )


@pytest.mark.parametrize(
    "change", [{"platform": "ascend"}, {"source_ref": "c" * 40}, {"complete": False}]
)
def test_nvidia_replay_rejects_incomplete_or_wrong_metadata(change):
    metadata = {
        "platform": "nvidia",
        "source_ref": "b" * 40,
        "complete": True,
        **change,
    }
    with pytest.raises(ValueError, match="Expected complete NVIDIA line coverage"):
        REPORTING.summarize_nvidia({}, metadata, "b" * 40)


def test_non_nvidia_report_keeps_its_full_wheel_scope(scoped_wheel):
    workdir, reports, environment, vendor_files = scoped_wheel
    (workdir / "test_good.py").write_text(
        "from torch_fl import model\ndef test_good(): assert model.choose(True)\n"
    )
    result = _run(environment, [_group("good", "test_good.py")])
    assert result.returncode == 0, result.stdout + result.stderr
    report = _merge(reports, platform="ascend")
    assert vendor_files <= set(report["files"])
    assert "scope_version" not in _metadata(reports)
    assert not (reports / "coverage-full-wheel.json").exists()


def test_nvidia_report_flags_execution_outside_its_scope(scoped_wheel):
    workdir, reports, environment, vendor_files = scoped_wheel
    (workdir / "test_good.py").write_text(
        "from torch_fl.accelerator import ascend\n"
        "def test_good(): assert ascend.ASCEND\n"
    )
    result = _run(environment, [_group("good", "test_good.py")])
    assert result.returncode == 0, result.stdout + result.stderr
    with pytest.raises(ValueError, match="NVIDIA coverage executed excluded files"):
        _merge(reports, platform="nvidia")
    metadata = _metadata(reports)
    assert not metadata["complete"]
    assert metadata["scope_conflicts"] == ["torch_fl/accelerator/ascend/__init__.py"]
    full = json.loads((reports / "coverage-full-wheel.json").read_text())
    assert set(metadata["excluded_files"]) == vendor_files
    assert full["files"][metadata["scope_conflicts"][0]]["executed_lines"] == [1]


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
