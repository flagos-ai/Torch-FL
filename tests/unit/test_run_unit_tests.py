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

"""Tests for the per-file unit test runner's result classification."""

import hashlib
import importlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_RUNNER_PATH = Path(__file__).resolve().parents[2] / ".github/scripts/run_unit_tests.py"
_SPEC = importlib.util.spec_from_file_location("run_unit_tests", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


_CLEAN = "........................                                                 [100%]\n24 passed in 19.28s\n"
_BANNERED_CLEAN = (
    "........................                                                 [100%]\n"
    "===== 24 passed, 1 skipped in 19.28s =====\n"
)
_WITH_FAILURE = "F.                                                                        [100%]\n1 failed, 1 passed in 3.10s\n"


@pytest.fixture(autouse=True)
def _default_teardown_policy(monkeypatch):
    """Pin the module's opt-in to its default for every test in this file.

    ``_ALLOW_TEARDOWN_CRASH`` is read from the environment at import time, and
    the CI step that runs this very file sets ``UNIT_TESTS_ALLOW_TEARDOWN_CRASH=1``
    (that is what makes the surrounding suite pass on containers whose RDMA stack
    faults during ``exit()``). Without this pin the assertions below would mean
    different things depending on how the runner was invoked -- which is how the
    signal-classification test failed in the first CI run of that step.
    """
    monkeypatch.setattr(_RUNNER, "_ALLOW_TEARDOWN_CRASH", False)


@pytest.fixture
def allow_teardown_crash(monkeypatch):
    monkeypatch.setattr(_RUNNER, "_ALLOW_TEARDOWN_CRASH", True)


def test_success_is_a_pass():
    assert _RUNNER.classify(0, _CLEAN) == ("PASS", False)


def test_pytest_reported_failure_is_a_failure():
    assert _RUNNER.classify(1, _WITH_FAILURE) == ("FAIL", False)


def test_summary_line_is_recognised_with_a_banner_and_without_one():
    # ``-q`` prints the bare counts; the default reporter wraps them in "=".
    assert _RUNNER.clean_summary("24 passed in 19.28s") == "24 passed in 19.28s"
    assert (
        _RUNNER.clean_summary("===== 24 passed, 1 skipped in 19.28s =====")
        == "24 passed, 1 skipped in 19.28s"
    )


def test_progress_lines_that_look_like_durations_are_not_the_summary():
    # A slow test prints its own "in Ns" only in the summary; a stack trace that
    # quotes "took 1.5s" must not be mistaken for a completed run.
    assert _RUNNER.clean_summary("Traceback: worker took 1.5s then died") is None


def test_crash_without_a_summary_is_never_tolerated(allow_teardown_crash):
    # SIGSEGV before pytest could report: nothing says the tests themselves ran,
    # so the opt-in for teardown crashes must not swallow it.
    assert (
        _RUNNER.classify(-signal.SIGSEGV, "Fatal Python error: Segmentation fault\n")[0]
        == "CRASH"
    )


def test_crash_after_a_clean_summary_is_tolerated_only_when_opted_in(
    monkeypatch, allow_teardown_crash
):
    monkeypatch.setattr(_RUNNER, "_ALLOW_TEARDOWN_CRASH", False)
    assert _RUNNER.classify(-signal.SIGSEGV, _CLEAN)[0] == "CRASH"

    monkeypatch.setattr(_RUNNER, "_ALLOW_TEARDOWN_CRASH", True)
    assert _RUNNER.classify(-signal.SIGSEGV, _CLEAN) == (
        "PASS (crash after summary)",
        True,
    )


def test_crash_after_a_bannered_clean_summary_is_tolerated(allow_teardown_crash):
    assert _RUNNER.classify(-signal.SIGSEGV, _BANNERED_CLEAN) == (
        "PASS (crash after summary)",
        True,
    )


def test_subprocess_style_exit_codes_are_classified_like_signals():
    # subprocess reports a signal death as a negative return code, but a shell
    # wrapper reports 128 + signal; both spellings must reach the same verdict.
    assert _RUNNER.classify(128 + signal.SIGSEGV, _CLEAN)[0] == "CRASH"
    assert _RUNNER.classify(128 + signal.SIGABRT, _CLEAN)[0] == "CRASH"


def test_crash_after_a_failed_summary_stays_a_failure(allow_teardown_crash):
    # The tests did report, and what they reported was a failure; the crash that
    # followed must not launder it into a pass.
    assert _RUNNER.classify(-signal.SIGSEGV, _WITH_FAILURE)[0] == "CRASH"


def test_timeout_is_reported_as_such():
    assert _RUNNER.classify(124, _CLEAN) == ("TIMEOUT", False)


def test_tolerated_teardown_crash_does_not_fail_the_run(
    monkeypatch, tmp_path, allow_teardown_crash
):
    """A file forgiven under the opt-in must not then be counted as a failure.

    ``main()`` used to build its failure list from the status string alone, so
    every tolerated file was reported as a ``::warning::`` and the run still
    exited 1 -- an opt-in that bought nothing. The CI step exits on that code.
    """
    (tmp_path / "test_one.py").write_text("")
    monkeypatch.setattr(
        _RUNNER, "run_file", lambda *args, **kwargs: (-signal.SIGSEGV, _CLEAN, False)
    )
    monkeypatch.setattr(sys, "argv", ["run_unit_tests.py", "--test-dir", str(tmp_path)])
    assert _RUNNER.main() == 0


def test_a_real_failure_still_fails_the_run(monkeypatch, tmp_path):
    (tmp_path / "test_one.py").write_text("")
    monkeypatch.setattr(
        _RUNNER, "run_file", lambda *args, **kwargs: (1, _WITH_FAILURE, False)
    )
    monkeypatch.setattr(sys, "argv", ["run_unit_tests.py", "--test-dir", str(tmp_path)])
    assert _RUNNER.main() == 1


def test_each_file_writes_its_own_pytest_report(monkeypatch, tmp_path):
    """The per-file runner owns complete reports for its child sessions."""
    test_file = tmp_path / "test_one.py"
    test_file.write_text("def test_one():\n    assert True\n")
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(report_dir))
    monkeypatch.delenv("FLAGCICD_PYTHON_NO_LAUNCHER", raising=False)
    monkeypatch.setattr(sys, "argv", ["run_unit_tests.py", "--test-dir", str(tmp_path)])
    assert _RUNNER.main() == 0
    reports = list(report_dir.glob("unit-*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["summary"]["passed"] == 1


def test_the_inventory_group_reaches_every_child_report(monkeypatch, tmp_path):
    """A group launch passes its identity to each child pytest session."""
    test_file = tmp_path / "test_one.py"
    test_file.write_text("def test_one():\n    assert True\n")
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "google-test.json").write_text(
        json.dumps(
            {
                "cases": [],
                "aliases": {},
                "group": "unit_tests_per_file",
            }
        )
    )
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(report_dir))
    monkeypatch.delenv("FLAGCICD_TEST_GROUP", raising=False)
    monkeypatch.delenv("FLAGCICD_PYTHON_NO_LAUNCHER", raising=False)
    monkeypatch.setattr(sys, "argv", ["run_unit_tests.py", "--test-dir", str(tmp_path)])
    assert _RUNNER.main() == 0
    captured = []

    def run_file(python, test_file, timeout, report_dir="", **_kwargs):
        captured.append((os.environ.get("FLAGCICD_TEST_GROUP"), report_dir))
        return 0, "1 passed in 0.01s\n", False

    monkeypatch.setattr(_RUNNER, "run_file", run_file)
    assert _RUNNER.main() == 0
    assert captured == [("unit_tests_per_file", str(report_dir))]


def test_pytest_child_inherits_the_facade_selected_during_startup(
    monkeypatch, tmp_path
):
    """PPU selects a private torch facade, then starts pytest in a child."""
    package = tmp_path / "installed" / "torch_fl"
    facade = tmp_path / "facade"
    root = tmp_path / "root"
    checkout = root
    package.mkdir(parents=True)
    (package / _C_EXTENSION).write_text("")
    facade.mkdir()
    checkout.mkdir()
    (package / "__init__.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "entry = str(Path(__file__).resolve().parent.parent)\n"
        "facade = os.environ['SYNTHETIC_FACADE']\n"
        "os.environ['PYTHONPATH'] = os.pathsep.join((facade, entry))\n"
    )
    test_file = checkout / "test_child.py"
    test_file.write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "def test_facade_and_tests_are_selected():\n"
        "    entries = os.environ['PYTHONPATH'].split(os.pathsep)\n"
        "    facade = Path(os.environ['SYNTHETIC_FACADE'])\n"
        "    assert Path(entries[0]) == facade\n"
        "def test_the_site_owns_torch_fl():\n"
        "    import torch_fl\n"
        "    installed = Path(os.environ['SYNTHETIC_INSTALLED']).resolve()\n"
        "    assert Path(torch_fl.__file__).resolve().is_relative_to(installed)\n"
    )
    (facade / "sitecustomize.py").write_text(
        "import hashlib\n"
        "import os\n"
        "from pathlib import Path\n"
        "checkout = Path(os.environ['SYNTHETIC_CHECKOUT'])\n"
        "digest = hashlib.sha256(str(checkout).encode()).hexdigest()\n"
        "os.environ['PYTHONPATH:' + digest] = 'stale'\n"
    )
    probe = root / "probe.py"
    probe.write_text(f"""\
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

package_root = Path({str(tmp_path / "installed")!r})
checkout = Path({str(checkout)!r})
facade = Path({str(facade)!r})
test_file = checkout / 'test_child.py'
os.environ['SYNTHETIC_FACADE'] = str(facade)
os.environ['SYNTHETIC_CHECKOUT'] = str(checkout)
os.environ['SYNTHETIC_INSTALLED'] = str(package_root)
os.environ['PYTHONPATH'] = str(checkout)
spec = importlib.util.spec_from_file_location(
    'torch_fl', package_root / 'torch_fl' / '__init__.py',
    submodule_search_locations=[str(package_root / 'torch_fl')],
)
module = importlib.util.module_from_spec(spec)
sys.modules['torch_fl'] = module
spec.loader.exec_module(module)
environment = {{}}
environment['__file__'] = {str(Path(".github/scripts/run_unit_tests.py").resolve())!r}
exec(compile(
    Path({str(Path(".github/scripts/run_unit_tests.py").resolve())!r}).read_text(),
    {str(Path(".github/scripts/run_unit_tests.py").resolve())!r}, 'exec',
), environment)
status, output, timed_out = environment['run_file'](
    sys.executable, test_file, 30, wheel_package_root=package_root,
)
assert not timed_out
print(output)
assert status == 0, output
""")
    monkeypatch.chdir(root)
    monkeypatch.setenv("SYNTHETIC_FACADE", str(facade))
    monkeypatch.setenv("SYNTHETIC_CHECKOUT", str(checkout))
    monkeypatch.setenv("SYNTHETIC_INSTALLED", str(package.parent))
    completed = subprocess.run(
        [sys.executable, str(probe)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output


def test_failed_package_startup_fails_before_a_child_is_created(monkeypatch, tmp_path):
    wheel_root = tmp_path / "installed"
    package = wheel_root / "torch_fl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise ImportError('broken extension')\n")
    (package / _C_EXTENSION).write_text("")
    test_file = tmp_path / "test_one.py"
    test_file.write_text("")
    created = []
    monkeypatch.setattr(
        _RUNNER.subprocess,
        "Popen",
        lambda *args, **kwargs: created.append((args, kwargs)),
    )

    with pytest.raises(ImportError, match="broken extension"):
        _RUNNER.run_file(sys.executable, test_file, 30, wheel_package_root=wheel_root)
    assert created == []


def test_main_only_preloads_a_package_with_a_native_extension(monkeypatch, tmp_path):
    source = tmp_path / "checkout" / "torch_fl"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (tmp_path / "checkout" / "test_one.py").write_text("")
    selected = SimpleNamespace(origin=str(source / "__init__.py"))
    roots = []
    monkeypatch.setattr(_RUNNER.importlib.util, "find_spec", lambda _name: selected)
    monkeypatch.setattr(_RUNNER, "child_environment", lambda root: {})
    monkeypatch.setattr(
        _RUNNER,
        "run_file",
        lambda *args, **kwargs: (
            roots.append(kwargs["wheel_package_root"]),
            (0, "1 passed in 0.01s\n", False),
        )[1],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_unit_tests.py",
            "--test-dir",
            str(tmp_path / "checkout"),
        ],
    )

    assert _RUNNER.main() == 0
    assert roots == [None]


def test_a_standalone_run_uses_its_supplied_inventory_group(monkeypatch, tmp_path):
    """A direct child run keeps the group supplied by its materialized inventory."""
    (tmp_path / "test_one.py").write_text("")
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "google-test.json").write_text(
        json.dumps(
            {
                "cases": [],
                "aliases": {},
                "group": "unit_tests_per_file",
            }
        )
    )
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(report_dir))
    monkeypatch.delenv("FLAGCICD_TEST_GROUP", raising=False)
    monkeypatch.delenv("FLAGCICD_PYTHON_NO_LAUNCHER", raising=False)
    monkeypatch.setattr(sys, "argv", ["run_unit_tests.py", "--test-dir", str(tmp_path)])
    captured = []

    def run_file(python, test_file, timeout, report_dir="", **_kwargs):
        captured.append(os.environ.get("FLAGCICD_TEST_GROUP"))
        return 0, "1 passed in 0.01s\n", False

    monkeypatch.setattr(_RUNNER, "run_file", run_file)
    assert _RUNNER.main() == 0
    assert captured == ["unit_tests_per_file"]


def test_the_installed_package_precedes_the_checkout_without_hiding_its_scripts(
    monkeypatch, tmp_path
):
    """The wheel owns ``torch_fl``; source tests still find checkout modules."""
    wheel_root = tmp_path / "wheel"
    (wheel_root / "torch_fl").mkdir(parents=True)
    (wheel_root / "torch_fl" / "__init__.py").write_text("")
    (wheel_root / "torch_fl" / _C_EXTENSION).write_text("")
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "scripts" / "checkout_only.py").write_text("")
    monkeypatch.setenv("PYTHONPATH", str(checkout))

    env = _RUNNER.child_environment(wheel_root)
    startup = Path(_RUNNER.__file__).with_name("_torch_fl_test_site.py")
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util; import sys; "
                "from pathlib import Path; "
                f"spec = importlib.util.spec_from_file_location('site', {str(startup)!r}); "
                "module = importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(module); module.install(); "
                "package = importlib.util.find_spec('torch_fl'); "
                "checkout = importlib.util.find_spec('scripts.checkout_only'); "
                "print(Path(package.origin).resolve()); "
                "print(checkout is not None)"
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert probe.stdout.splitlines() == [
        str((wheel_root / "torch_fl" / "__init__.py").resolve()),
        "True",
    ]


def test_the_child_does_not_inherit_a_stale_macos_path_digest(monkeypatch, tmp_path):
    """The child must use the selected wheel path, including on macOS."""
    wheel_root = tmp_path / "installed-wheel"
    (wheel_root / "torch_fl").mkdir(parents=True)
    (wheel_root / "torch_fl" / "__init__.py").write_text("")
    (wheel_root / "torch_fl" / _C_EXTENSION).write_text("")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "checkout"))
    digest = hashlib.sha256(str(tmp_path / "checkout").encode()).hexdigest()
    monkeypatch.setenv(f"PYTHONPATH:{digest}", "stale")
    env = _RUNNER.child_environment(wheel_root)
    assert all(not name.startswith("PYTHONPATH:") for name in env)


_C_EXTENSION = f"_C{importlib.machinery.EXTENSION_SUFFIXES[0]}"


def test_pytest_conftest_and_child_keep_source_and_installed_modules(
    monkeypatch, tmp_path
):
    """pytest's pythonpath entry cannot replace the selected native package."""
    checkout = tmp_path / "checkout"
    source = checkout / "torch_fl"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("raise ModuleNotFoundError('torch_fl._C')\n")
    (checkout / "scripts").mkdir()
    (checkout / "scripts" / "__init__.py").write_text("")
    (checkout / "scripts" / "source_support.py").write_text("VALUE = 7\n")
    tests = checkout / "tests" / "unit"
    tests.mkdir(parents=True)
    (checkout / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npythonpath = ["."]\n'
    )
    (tests / "conftest.py").write_text(
        "import os\n"
        "import torch_fl\n"
        "assert torch_fl.ORIGIN == 'installed'\n"
        "torch_fl.SELECTED_BY_CONFTEST = True\n"
        "os.environ['TORCH_FL_CONFTEST_SELECTED'] = torch_fl.ORIGIN\n"
    )
    test_file = tests / "test_selected.py"
    test_file.write_text(
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "import torch_fl\n"
        "from scripts import source_support\n"
        "def test_parent():\n"
        "    assert os.environ['TORCH_FL_SELECTED_TEST'] == 'installed'\n"
        "    assert os.environ['TORCH_FL_CONFTEST_SELECTED'] == 'installed'\n"
        "    assert torch_fl.SELECTED_BY_CONFTEST\n"
        "    assert source_support.VALUE == 7\n"
        "def test_child():\n"
        "    probe = subprocess.run(\n"
        "        [sys.executable, '-c', 'import torch_fl; "
        "print(torch_fl.ORIGIN)'],\n"
        "        cwd=os.environ['TORCH_FL_CURRENT'],\n"
        "        env=os.environ, text=True, capture_output=True, check=True,\n"
        "    )\n"
        "    assert probe.stdout.strip() == 'installed', probe.stderr\n"
    )
    installed = tmp_path / "installed"
    package = installed / "torch_fl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "import os\n"
        "ORIGIN = 'installed'\n"
        "os.environ['TORCH_FL_SELECTED_TEST'] = ORIGIN\n"
    )
    (package / _C_EXTENSION).write_text("")
    monkeypatch.chdir(checkout)
    report_dir = tmp_path / "reports"
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(report_dir))
    monkeypatch.setenv("FLAGCICD_TEST_GROUP", "unit_tests_per_file")
    monkeypatch.setenv("TORCH_FL_CURRENT", str(checkout))
    monkeypatch.delenv("FLAGCICD_PYTHON_NO_LAUNCHER", raising=False)
    completed = subprocess.run(
        [
            sys.executable,
            str(_RUNNER_PATH),
            "--test-dir",
            str(tests),
            "--wheel-package-root",
            str(installed),
        ],
        cwd=checkout,
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output


def test_a_nested_unit_file_is_discovered_by_the_real_runner(monkeypatch, tmp_path):
    """BPU tests live in a subpackage; leaving them out would hide them."""
    nested = tmp_path / "bpu"
    nested.mkdir()
    (nested / "test_one.py").write_text("def test_nested():\n    assert True\n")
    monkeypatch.setattr(sys, "argv", ["run_unit_tests.py", "--test-dir", str(tmp_path)])
    monkeypatch.delenv("FLAGCICD_REPORT_DIR", raising=False)
    monkeypatch.delenv("FLAGCICD_PYTHON_NO_LAUNCHER", raising=False)
    assert _RUNNER.main() == 0


def test_a_unit_child_never_uses_the_python_p_switch(monkeypatch, tmp_path):
    """The MUSA interpreter launcher rejects ``-P`` before starting Python."""
    test_file = tmp_path / "test_one.py"
    test_file.write_text("")
    created = []

    class Finished:
        returncode = 0

        def communicate(self, timeout=None):
            return "1 passed in 0.01s\n", None

    monkeypatch.setattr(
        _RUNNER.subprocess,
        "Popen",
        lambda command, **_kwargs: (created.append(command), Finished())[1],
    )
    returncode, _output, timed_out = _RUNNER.run_file(sys.executable, test_file, 30)
    assert (returncode, timed_out) == (0, False)
    assert created and "-P" not in created[0]


def test_a_unit_directory_outside_the_checkout_writes_a_report(monkeypatch, tmp_path):
    """A probe may run pytest on a temp file; the report directory is writable."""
    test_file = tmp_path / "test_outside.py"
    test_file.write_text("def test_outside():\n    assert True\n")
    report_dir = tmp_path / "reports"
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(report_dir))
    monkeypatch.delenv("FLAGCICD_PYTHON_NO_LAUNCHER", raising=False)
    monkeypatch.setattr(sys, "argv", ["run_unit_tests.py", "--test-dir", str(tmp_path)])
    assert _RUNNER.main() == 0
    assert len(list(report_dir.glob("unit-*.json"))) == 1


def test_a_failing_child_file_really_fails_the_run(monkeypatch, tmp_path):
    (tmp_path / "test_one.py").write_text("def test_one():\n    assert False\n")
    monkeypatch.delenv("FLAGCICD_REPORT_DIR", raising=False)
    monkeypatch.delenv("FLAGCICD_PYTHON_NO_LAUNCHER", raising=False)
    monkeypatch.setattr(sys, "argv", ["run_unit_tests.py", "--test-dir", str(tmp_path)])
    assert _RUNNER.main() == 1


def test_a_failing_report_is_not_marked_complete_by_a_passing_runner_status(
    monkeypatch, tmp_path
):
    """The runner's PASS must not override the child's recorded failure."""
    root = tmp_path / "checkout"
    test_file = root / "tests/unit/test_one.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("")
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    case = "tests/unit/test_one.py"
    (report_dir / "google-test.json").write_text(
        json.dumps({"cases": [case], "aliases": {}})
    )
    (report_dir / "unit-one.json").write_text(
        json.dumps(
            {
                "args": [str(test_file)],
                "tests": [{"nodeid": case + "::test_one", "outcome": "passed"}],
                "summary": {"passed": 1, "failed": 1},
            }
        )
    )
    monkeypatch.setattr(
        _RUNNER, "run_file", lambda *args, **kwargs: (-signal.SIGSEGV, _CLEAN, False)
    )
    monkeypatch.setattr(_RUNNER, "_ALLOW_TEARDOWN_CRASH", True)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(root))
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(report_dir))
    monkeypatch.setattr(
        sys, "argv", ["run_unit_tests.py", "--test-dir", str(test_file.parent)]
    )
    assert _RUNNER.main() == 0
    assert not (report_dir / "unit-one.pending").exists()


def test_a_clean_report_is_marked_complete_after_a_tolerated_crash(
    monkeypatch, tmp_path
):
    root = tmp_path / "checkout"
    test_file = root / "tests/unit/test_one.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("")
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    case = "tests/unit/test_one.py"
    inventory = {"cases": [case], "aliases": {}}
    (report_dir / "google-test.json").write_text(json.dumps(inventory))
    monkeypatch.setattr(
        _RUNNER,
        "run_file",
        lambda *args, **kwargs: (
            (report_dir / "unit-one.json").write_text(
                json.dumps(
                    {
                        "args": [str(test_file)],
                        "tests": [
                            {
                                "nodeid": case + "::test_one",
                                "outcome": "passed",
                                "setup": {"outcome": "passed"},
                                "call": {"outcome": "passed"},
                                "teardown": {"outcome": "passed"},
                            }
                        ],
                        "summary": {"passed": 1, "failed": 0, "error": 0},
                    }
                )
            ),
            (-signal.SIGSEGV, _CLEAN, False),
        )[-1],
    )
    monkeypatch.setattr(_RUNNER, "_ALLOW_TEARDOWN_CRASH", True)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(root))
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(report_dir))
    monkeypatch.setattr(
        sys, "argv", ["run_unit_tests.py", "--test-dir", str(test_file.parent)]
    )
    assert _RUNNER.main() == 0
    assert not list(report_dir.glob("unit-*.pending"))
    assert list(report_dir.glob("unit-*.unit"))
    assert json.loads((report_dir / "google-test.json").read_text())["aliases"] == {
        case: [case]
    }
