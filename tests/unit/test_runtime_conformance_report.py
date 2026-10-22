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

"""Unit coverage for tests/integration/runtime/conftest.py's summary gate.

The suite registers its terminal-summary plugin from ``pytest_configure``, and
pytest imports a conftest whenever its directory is *collected* -- running none
of its tests is enough. The `Multi-device contracts` CI step
(``pytest tests/integration/ -m "multi_device"``) collects the whole integration
tree, so the plugin came up and printed a RUNTIME-CONFORMANCE line whose
``passed=58 skipped=2`` belonged to that step's own marker selection while every
runtime test sat deselected: a number that reads as this suite's verdict but
cannot fail for this suite. The line now counts only this package's selected
tests and stays silent when none of them ran.

Run: pytest tests/unit/test_runtime_conformance_report.py -v
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_CONFTEST_PATH = (
    _REPO_ROOT / "tests" / "integration" / "runtime" / "conftest.py"
)

_LINE = "RUNTIME-CONFORMANCE "
_RUNTIME_OPS = [
    "add.Tensor",
    "mul.Tensor",
    "mm",
    "bmm",
    "cat",
    "neg",
    "sum.dim_IntList",
    "_softmax",
    "embedding",
    "where.self",
]


def _load_runtime_conftest():
    """Import runtime/conftest.py outside pytest's plugin machinery."""
    spec = importlib.util.spec_from_file_location(
        "runtime.conftest", _RUNTIME_CONFTEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime_conftest = _load_runtime_conftest()


class _FakeItem:
    """The two attributes ``_is_runtime_item`` reads."""

    def __init__(self, path=None, fspath=None, nodeid=""):
        if path is not None:
            self.path = path
        if fspath is not None:
            self.fspath = fspath
        self.nodeid = nodeid


class _FakeReport:
    def __init__(self, nodeid, outcome="passed"):
        self.nodeid = nodeid
        self.outcome = outcome


def test_is_runtime_item_accepts_a_test_in_the_package():
    package = runtime_conftest._PACKAGE_PATH
    item = _FakeItem(path=package / "test_runtime_allocator.py")
    assert runtime_conftest._is_runtime_item(item) is True


def test_is_runtime_item_rejects_a_sibling_directory():
    """``tests/integration/test_ops.py`` is collected by the same step that
    collects this package; it is not this suite's test."""
    item = _FakeItem(path=runtime_conftest._PACKAGE_PATH.parent / "test_ops.py")
    assert runtime_conftest._is_runtime_item(item) is False


def test_is_runtime_item_falls_back_to_fspath():
    """pytest < 7 had no ``Item.path``; the package still has to recognise it."""
    package = runtime_conftest._PACKAGE_PATH
    item = _FakeItem(fspath=package / "test_runtime_dispatch_route.py")
    assert runtime_conftest._is_runtime_item(item) is True


def test_count_outcomes_ignores_other_selections():
    """The session's stats are everyone's; only ours may be counted."""
    mine = "tests/integration/runtime/test_runtime_allocator.py::test_x"
    theirs = "tests/integration/test_ops.py::test_y"
    stats = {
        "passed": [_FakeReport(mine), _FakeReport(theirs), _FakeReport(theirs)],
        "skipped": [_FakeReport(theirs, "skipped")],
    }
    assert runtime_conftest._count_outcomes(stats, {mine}) == (1, 0, 0, 0)


def _load_runtime_caps(tmp_path, monkeypatch, routes):
    conf = tmp_path / "backends_mps.conf"
    conf.write_text(
        "".join(f"{op} = {value}\n" for op, value in routes.items()),
        encoding="utf-8",
    )

    class FakeTorchFl:
        @staticmethod
        def backend_config_path():
            return str(conf)

    monkeypatch.setitem(sys.modules, "torch_fl", FakeTorchFl)
    spec = importlib.util.spec_from_file_location(
        "runtime_caps_under_test",
        runtime_conftest._PACKAGE_PATH / "runtime_caps.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_effective_route_separates_an_unimplemented_op_from_a_missing_one(
    tmp_path, monkeypatch
):
    runtime_caps = _load_runtime_caps(
        tmp_path,
        monkeypatch,
        {"add.Tensor": "mps", "cat": "none"},
    )
    assert runtime_caps.effective_route("add.Tensor") == ("mps", "routed")
    assert runtime_caps.effective_route("cat") == (None, "unimplemented")
    with pytest.raises(AssertionError, match="embedding is not listed"):
        runtime_caps.effective_route("embedding")


def test_unimplemented_ops_retain_their_names_and_missing_routes_still_fail(
    tmp_path, monkeypatch
):
    routes = {
        op: "none" if op in {"cat", "embedding", "where.self"} else "mps"
        for op in _RUNTIME_OPS
    }
    runtime_caps = _load_runtime_caps(tmp_path, monkeypatch, routes)
    assert runtime_caps.unimplemented_runtime_ops() == [
        "cat",
        "embedding",
        "where.self",
    ]
    conf = runtime_caps.conf_path()
    conf.write_text(conf.read_text().split("where.self", 1)[0], encoding="utf-8")
    with pytest.raises(AssertionError, match="where.self"):
        runtime_caps.unimplemented_runtime_ops()


def test_effective_route_does_not_translate_an_empty_conf_value(tmp_path, monkeypatch):
    runtime_caps = _load_runtime_caps(
        tmp_path,
        monkeypatch,
        {"add.Tensor": "mps", "cat": ""},
    )
    assert runtime_caps.effective_route("cat") == ("", "routed")


def _write_tree(root, *, runtime_tests, other_tests):
    """A miniature of the integration tree: the suite plus a sibling test."""
    package = root / "tests" / "integration" / "runtime"
    package.mkdir(parents=True)
    (package / "conftest.py").write_text(_RUNTIME_CONFTEST_PATH.read_text())
    (package / "runtime_caps.py").write_text(
        (
            _REPO_ROOT / "tests" / "integration" / "runtime" / "runtime_caps.py"
        ).read_text()
    )
    (root / "stub_torch_fl.py").write_text(
        "def backend_config_path():\n"
        "    return 'backends_mps.conf'\n"
        "\n"
        "\n"
        "class _env:\n"
        "    @staticmethod\n"
        "    def build_kernels():\n"
        "        return frozenset({'vendor'})\n"
        "\n"
        "\n"
        "def report_facts():\n"
        "    from runtime_caps import RUNTIME_OPS, unimplemented_runtime_ops\n"
        "    return {'ops_total': len(RUNTIME_OPS), "
        "'unimplemented_ops': unimplemented_runtime_ops()}\n",
        encoding="utf-8",
    )
    (package / "test_runtime_probe.py").write_text(runtime_tests)
    (root / "tests" / "integration" / "test_sibling.py").write_text(other_tests)
    (root / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\nmarkers = ["multi_device: probe marker"]\n'
    )
    conf = root / "backends_mps.conf"
    conf.write_text(
        "".join(
            f"{op} = {'none' if op in {'cat', 'embedding', 'where.self'} else 'mps'}\n"
            for op in _RUNTIME_OPS
        ),
        encoding="utf-8",
    )
    return root


def _run_pytest(root, *args):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, importlib.util; "
                "spec = importlib.util.spec_from_file_location("
                "'torch_fl', 'stub_torch_fl.py'); "
                "torch_fl = importlib.util.module_from_spec(spec); "
                "sys.modules['torch_fl'] = torch_fl; "
                "spec.loader.exec_module(torch_fl); "
                "from pytest import console_main; "
                "raise SystemExit(console_main())"
            ),
            *args,
            "-p",
            "no:cacheprovider",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result


def _report_facts(conformance_tree):
    result = _run_pytest(conformance_tree, "tests/integration/runtime", "-s")
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(
        (conformance_tree / "runtime-report-facts.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def conformance_tree(tmp_path):
    return _write_tree(
        tmp_path,
        runtime_tests=(
            "def test_runtime_probe():\n"
            "    import json\n"
            "    from pathlib import Path\n"
            "    from runtime_caps import RUNTIME_OPS, unimplemented_runtime_ops\n"
            "    Path('runtime-report-facts.json').write_text(\n"
            "        json.dumps(\n"
            "            {\n"
            "                'ops_total': len(RUNTIME_OPS),\n"
            "                'unimplemented_ops': unimplemented_runtime_ops(),\n"
            "            }\n"
            "        )\n"
            "    )\n"
        ),
        other_tests=(
            "import pytest\n\n\n"
            "@pytest.mark.multi_device\n"
            "def test_sibling_probe():\n"
            "    assert True\n"
        ),
    )


def test_no_line_when_the_selection_is_the_wider_tree(conformance_tree):
    """The `Multi-device contracts` step's shape: the directory is collected,
    its tests are not selected, and no line may claim otherwise."""
    result = _run_pytest(
        conformance_tree, "tests/integration/", "-m", "multi_device", "-v", "-s"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert _LINE not in result.stdout
    assert not list(conformance_tree.glob("runtime-conformance-*.json"))


def test_line_prints_for_the_suite_itself(conformance_tree):
    result = _run_pytest(conformance_tree, "tests/integration/runtime", "-v", "-s")
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(row for row in result.stdout.splitlines() if row.startswith(_LINE))
    assert "passed=1 failed=0 skipped=0 errors=0" in line


def test_line_counts_only_the_suite_in_a_mixed_selection(conformance_tree):
    """Run both at once: the line still describes the runtime tests alone."""
    result = _run_pytest(
        conformance_tree,
        "tests/integration/runtime",
        "tests/integration/test_sibling.py",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    line = next(row for row in result.stdout.splitlines() if row.startswith(_LINE))
    assert "passed=1 failed=0 skipped=0 errors=0" in line


def test_unimplemented_routes_are_named_in_the_report(conformance_tree):
    report = _run_pytest(conformance_tree, "tests/integration/runtime", "-s")
    assert report.returncode == 0, report.stdout + report.stderr
    line = next(row for row in report.stdout.splitlines() if row.startswith(_LINE))
    assert "accel=mps" in line
    assert "ops_total=10 ops_routed=7 ops_none=3" in line
    assert "RUNTIME-COVERAGE unimplemented=cat,embedding,where.self" in (report.stdout)
    report_json = json.loads(
        (conformance_tree / "runtime-conformance-mps.json").read_text(encoding="utf-8")
    )
    assert report_json["ops_none"] == 3
    assert report_json["unimplemented_ops"] == ["cat", "embedding", "where.self"]
    facts = _report_facts(conformance_tree)
    assert facts["ops_total"] == 10
    assert facts["unimplemented_ops"] == ["cat", "embedding", "where.self"]
