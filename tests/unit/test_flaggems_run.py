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

"""Run the FlagGems selector and its reporting in real pytest sessions."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts"


@pytest.fixture
def conformance_tree(tmp_path):
    tree = tmp_path / "tree"
    suite = tree / "tests/integration/flaggems"
    suite.mkdir(parents=True)
    (tree / "tests/__init__.py").write_text("")
    for name in ("profiler_support.py", "amp_support.py"):
        (tree / "tests/integration" / name).write_text("\n")
    for name in ("integration_rules.py", "device_init.py"):
        (tree / "tests/integration" / name).write_text(
            (ROOT / "tests/integration" / name).read_text()
        )
    (tree / "torch_fl").mkdir()
    (tree / "torch_fl/__init__.py").write_text(
        "class Env:\n"
        "    def build_kernels(self):\n"
        "        return ('flaggems',)\n"
        "_env = Env()\n"
        "def is_available():\n"
        "    return True\n"
        "class Flagos:\n"
        "    is_available = staticmethod(is_available)\n"
        "    init = staticmethod(lambda: None)\n"
        "flagos = Flagos()\n"
    )
    (suite / "__init__.py").write_text("")
    (suite / "conftest.py").write_text(
        (ROOT / "tests/integration/flaggems/conftest.py").read_text()
    )
    (suite / "test_shared.py").write_text(
        textwrap.dedent(
            """
            import pytest

            @pytest.fixture(autouse=True)
            def _does_not_load_a_second_report(tmp_path, monkeypatch):
                monkeypatch.delenv("FLAGCICD_REPORT_DIR", raising=False)

            def test_native():
                pass

            @pytest.mark.flaggems_python
            def test_kernel():
                pass
            """
        )
    )
    (suite / "run.py").write_text(
        (ROOT / "tests/integration/flaggems/run.py")
        .read_text()
        .replace("parents[2]", "parents[3]", 1)
    )
    return tree


def test_an_outer_integration_session_ignores_the_runner_only_option(
    conformance_tree, tmp_path
):
    """A marker outside the shared directory must not collect its suite."""
    (conformance_tree / "tests/integration/test_outer.py").write_text(
        "import pytest\n@pytest.mark.outside_shared_suite\ndef test_outer(): pass\n"
    )
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/",
            "-m",
            "outside_shared_suite",
            "-q",
        ],
        cwd=conformance_tree,
        env=_environment(tmp_path / "reports", "outer", str(ROOT / "tests/unit")),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "1 passed" in process.stdout
    assert "FLAGGEMS-CONFORMANCE" not in process.stdout


def test_ci_plan_requires_the_exact_collected_inventory(conformance_tree, tmp_path):
    plan = {
        "flaggems_conformance": [
            "tests/integration/flaggems/test_shared.py::test_native",
            "tests/integration/flaggems/test_shared.py::test_kernel",
        ]
    }
    environment = _environment(
        tmp_path / "reports",
        "flaggems_kernel_conformance",
        str(ROOT / "tests/unit"),
    )
    environment["FLAGCICD_EXPECTED_CASES"] = json.dumps(plan)
    environment["FLAGCICD_FLAGGEMS_PLAN"] = json.dumps(plan)
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/flaggems/",
            "-q",
            "--flaggems-conformance-suite=flaggems",
        ],
        cwd=conformance_tree,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "2 passed" in process.stdout
    assert "FLAGGEMS-CONFORMANCE-COMPLETE suite=flaggems" in process.stdout

    plan["flaggems_conformance"].pop()
    environment["FLAGCICD_EXPECTED_CASES"] = json.dumps(plan)
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/flaggems/",
            "-q",
            "--flaggems-conformance-suite=flaggems",
        ],
        cwd=conformance_tree,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode != 0
    assert "differs from the CI inventory" in process.stdout + process.stderr


def test_ci_selection_runs_once_in_its_own_group_session(conformance_tree, tmp_path):
    """A CI group owns its pytest pass and its single JSON report."""
    reports = tmp_path / "reports"
    environment = _environment(
        reports, "flaggems_conformance", str(SCRIPTS / "pytest_reporting")
    )
    environment["PYTEST_PLUGINS"] = "flagcicd_report"
    environment["PYTHONPATH"] += os.pathsep + str(conformance_tree)
    process = subprocess.run(
        [
            sys.executable,
            str(conformance_tree / "tests/integration/flaggems/run.py"),
            "native",
        ],
        cwd=conformance_tree,
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    reports_by_suffix = {path.suffix: path for path in reports.glob("pytest-*")}
    assert set(reports_by_suffix) == {".pending", ".json"}
    assert json.loads(reports_by_suffix[".pending"].read_text())["group"] == (
        "flaggems_conformance"
    )


def test_ci_inventory_environment_uses_one_plan_for_both_passes():
    scripts = _load_module(SCRIPTS / "run_integration_tests.py", "ci_inventory")
    inventory = {"flaggems_conformance": {"cases": ["case"]}}
    native = scripts.inventory_environment("flaggems_conformance", inventory)
    kernel = scripts.inventory_environment("flaggems_kernel_conformance", inventory)
    assert native == {
        "FLAGCICD_EXPECTED_CASES": '{"flaggems_conformance":["case"]}',
    }
    assert kernel == native


def test_the_runner_uses_the_same_case_plan_for_both_passes():
    """A wheel conformance run must not assign a host-specific case subset."""
    scripts = _load_module(SCRIPTS / "run_integration_tests.py", "ci_runner_plan")
    tests = [
        {"id": "flaggems_conformance"},
        {"id": "flaggems_kernel_conformance"},
    ]
    inventory = {
        "flaggems_conformance": {"cases": ["first", "second"]},
        "flaggems_kernel_conformance": {
            "cases": ["unmatched-first", "unmatched-second"]
        },
    }
    assert scripts.check_inventory_binding(tests, inventory) is None
    environments = [
        scripts.inventory_environment(test["id"], inventory) for test in tests
    ]
    assert environments[0] == environments[1]
    assert json.loads(environments[0]["FLAGCICD_EXPECTED_CASES"]) == {
        "flaggems_conformance": ["first", "second"]
    }


def test_the_runner_selection_remains_active_inside_its_directory(
    conformance_tree, tmp_path
):
    """The outer-session guard must not weaken the runner's own selection."""
    (conformance_tree / "tests/integration/test_outer.py").write_text(
        "import pytest\n@pytest.mark.anyplatform\ndef test_outer(): pass\n"
    )
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/flaggems/",
            "--flaggems-conformance-suite=native",
            "-m",
            "anyplatform",
            "-q",
        ],
        cwd=conformance_tree,
        env=_environment(tmp_path / "reports", "native", str(ROOT / "tests/unit")),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 5, process.stdout + process.stderr
    assert "FLAGGEMS-CONFORMANCE-FAILURE" in process.stdout + process.stderr


def _environment(directory, group, python_path):
    environment = os.environ.copy()
    for name in (
        "PYTEST_ADDOPTS",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "PYTEST_PLUGINS",
        "FLAGCICD_REPORT_DIR",
        "FLAGCICD_TEST_GROUP",
        "FLAGCICD_LAUNCHER",
        "FLAGCICD_PYTHON_NO_LAUNCHER",
        "FLAGCICD_EXPECTED_CASES",
        "FLAGCICD_FLAGGEMS_PLAN",
        "FLAGCICD_FLAGGEMS_GROUP",
    ):
        environment.pop(name, None)
    environment.update(
        PYTHONPATH=os.pathsep.join(
            value for value in (str(python_path), str(conformance_tree)) if value
        ),
        FLAGCICD_REPORT_DIR=str(directory),
        FLAGCICD_TEST_GROUP=group,
        FLAGCICD_LAUNCHER="group" if group else "",
    )
    return environment


def test_a_long_test_output_keeps_the_conformance_tail(conformance_tree, tmp_path):
    suite = conformance_tree / "tests/integration/flaggems"
    (suite / "test_shared.py").write_text(
        "def test_native():\n    print('x' * 70000)\n\ndef test_kernel():\n    pass\n"
    )
    environment = _environment(tmp_path / "reports", "", "")
    environment["PYTHONPATH"] += os.pathsep + str(conformance_tree)
    process = subprocess.run(
        [sys.executable, str(suite / "run.py"), "native"],
        cwd=conformance_tree,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert process.returncode == 0, process.stdout[-500:] + process.stderr[-500:]
    assert "x" * 65536 in process.stdout
    assert "FLAGGEMS-CONFORMANCE-COMPLETE suite=native" in process.stdout


@pytest.mark.parametrize("suite", ["native", "flaggems"])
def test_local_runner_selection_uses_its_group(conformance_tree, tmp_path, suite):
    """Each locally started pass owns only the suite named by its argument."""
    (conformance_tree / "tests/integration/conftest.py").write_text(
        "def pytest_addoption(parser):\n"
        "    parser.addoption('--model', default=None)\n"
        "    parser.addoption('--max-new-tokens', type=int, default=None)\n"
        "    parser.addoption('--steps', type=int, default=None)\n"
        "    parser.addoption('--batch-size', type=int, default=None)\n"
        "    parser.addoption('--seq-len', type=int, default=None)\n"
        "    parser.addoption('--lr', type=float, default=None)\n"
    )
    environment = _environment(
        tmp_path / "reports",
        "flaggems_conformance" if suite == "native" else "flaggems_kernel_conformance",
        "",
    )
    environment["PYTHONPATH"] += os.pathsep + str(conformance_tree)
    process = subprocess.run(
        [
            sys.executable,
            str(conformance_tree / "tests/integration/flaggems/run.py"),
            suite,
        ],
        cwd=conformance_tree,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 0, repr(process.stdout) + repr(process.stderr)
    assert f"FLAGGEMS-CONFORMANCE-COMPLETE suite={suite}" in process.stdout
    sessions = [
        json.loads(path.read_text())
        for path in (tmp_path / "reports").glob("*.pending")
    ]
    assert not sessions


def test_a_local_child_uses_the_physical_python_from_the_ci_launcher(
    conformance_tree, tmp_path
):
    """The nested interpreter must not invoke the group's shell function."""
    reports = tmp_path / "reports"
    environment = _environment(reports, "flaggems_conformance", "")
    physical_python = conformance_tree / "physical-python"
    physical_python.write_text(f"#!{sys.executable}\n")
    physical_python.chmod(0o700)
    environment["FLAGCICD_PYTHON_ROOT"] = str(physical_python)
    (conformance_tree / "tests/integration/conftest.py").write_text(
        "def pytest_collection_finish(session):\n    print('FLAGCICD-CHILD-REPORT')\n"
    )
    process = subprocess.run(
        [
            sys.executable,
            str(conformance_tree / "tests/integration/flaggems/run.py"),
            "native",
        ],
        cwd=conformance_tree,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "FLAGCICD-CHILD-REPORT" in process.stdout
    assert not list(reports.glob("*.pending"))


def _copy_integration_conftest(tree):
    (tree / "tests/integration/conftest.py").write_text(
        (ROOT / "tests/integration/conftest.py")
        .read_text()
        .replace(
            "pytest_plugins = (",
            'pytest_plugins = ("profiler_support", "amp_support",',
            1,
        )
    )


def _load_module(path, name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_invocation_keeps_the_outer_group_out_of_its_child(
    conformance_tree, tmp_path
):
    reports = tmp_path / "reports"
    environment = _environment(reports, "flaggems_conformance", "")
    environment["PYTHONPATH"] += os.pathsep + str(conformance_tree)
    process = subprocess.run(
        [
            sys.executable,
            str(conformance_tree / "tests/integration/flaggems/run.py"),
            "native",
        ],
        cwd=conformance_tree,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    sessions = [json.loads(path.read_text()) for path in reports.glob("*.pending")]
    assert not sessions


def test_the_runner_rejects_a_pytest_suite_choice(conformance_tree, tmp_path):
    """The pass comes from the runner; a second selection could disagree."""
    environment = _environment(tmp_path / "reports", "", "")
    environment["PYTHONPATH"] += os.pathsep + str(conformance_tree)
    process = subprocess.run(
        [
            sys.executable,
            str(conformance_tree / "tests/integration/flaggems/run.py"),
            "native",
            "--flaggems-conformance-suite=flaggems",
        ],
        cwd=conformance_tree,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert process.returncode == 2, process.stdout + process.stderr
    assert "unrecognized arguments" in process.stderr


def test_both_passes_execute_the_same_ordered_cases(conformance_tree, tmp_path):
    suite = conformance_tree / "tests/integration/flaggems"
    # The copied runner resolves the repository root from its directory.
    # Place the recording plugin alongside that root, as the real runner does.
    integration = suite.parent
    (suite / "test_shared.py").write_text(
        textwrap.dedent(
            """
            import pytest

            @pytest.fixture(autouse=True)
            def _forbid_ci_plan(monkeypatch):
                monkeypatch.delenv("FLAGCICD_EXPECTED_CASES", raising=False)

            def test_unmarked():
                pass

            @pytest.mark.flaggems_python
            def test_kernel():
                pass

            def test_after():
                pass
            """
        )
    )
    environment = _environment(tmp_path / "reports", "", "")
    environment["PYTEST_ADDOPTS"] = "-p no:randomly"
    environment["PYTHONPATH"] += os.pathsep + str(conformance_tree)
    passport = tmp_path / "passes.json"
    (integration / "conftest.py").write_text(
        textwrap.dedent(
            f"""
            import json

            def pytest_terminal_summary(terminalreporter):
                passport = {str(passport)!r}
                try:
                    data = json.load(open(passport))
                except FileNotFoundError:
                    data = {{}}
                data[terminalreporter.config.getoption(
                    "--flaggems-conformance-suite"
                )] = [
                    report.nodeid
                    for report in terminalreporter.stats.get("passed", ())
                ]
                with open(passport, "w") as handle:
                    json.dump(data, handle)
            """
        )
    )
    environment["PYTHONPATH"] = (
        str(integration) + os.pathsep + environment["PYTHONPATH"]
    )
    for suite_name in ("native", "flaggems"):
        process = subprocess.run(
            [sys.executable, str(suite / "run.py"), suite_name],
            cwd=conformance_tree,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert process.returncode == 0, process.stdout[-500:] + process.stderr[-500:]
        assert f"FLAGGEMS-CONFORMANCE-COMPLETE suite={suite_name}" in process.stdout

    selected = list(json.loads(passport.read_text()).values())
    assert selected[0] == selected[1]
    assert selected[0] == [
        "tests/integration/flaggems/test_shared.py::test_unmarked",
        "tests/integration/flaggems/test_shared.py::test_kernel",
        "tests/integration/flaggems/test_shared.py::test_after",
    ]


@pytest.mark.parametrize(
    ("source", "status", "message"),
    [
        ("def test_kernel(): assert False\n", 1, "test_kernel"),
        ("@pytest.mark.flaggems_python\ndef test_native(): pass\n", 0, "PASSED"),
        ("", 5, "FLAGGEMS-CONFORMANCE-FAILURE"),
    ],
)
def test_failure_or_empty_selection_never_reports_complete(
    conformance_tree, tmp_path, source, status, message
):
    suite = conformance_tree / "tests/integration/flaggems"
    (suite / "test_shared.py").write_text(
        "import pytest\n" + source if "@pytest" in source else source
    )
    environment = _environment(
        tmp_path / "reports",
        "flaggems_conformance",
        "",
    )
    environment["PYTHONPATH"] += os.pathsep + str(conformance_tree)
    process = subprocess.run(
        [
            sys.executable,
            str(suite / "run.py"),
            "native",
        ],
        cwd=conformance_tree,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert process.returncode == status, process.stdout + process.stderr
    if status:
        assert "FLAGGEMS-CONFORMANCE-COMPLETE" not in process.stdout
        assert message in process.stdout + process.stderr
    else:
        assert "FLAGGEMS-CONFORMANCE-COMPLETE" in process.stdout
        assert message in process.stdout
