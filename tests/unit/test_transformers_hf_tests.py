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
"""Pure tests for the official HuggingFace per-model runner."""

import importlib.util
import io
import os
import sys
import tarfile
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SCRIPT = REPO_ROOT / "tests" / "manual" / "transformers_hf_source.py"
RUNNER_SCRIPT = REPO_ROOT / "tests" / "manual" / "transformers_hf_tests.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = load("transformers_hf_source", SOURCE_SCRIPT)
runner = load("transformers_hf_tests", RUNNER_SCRIPT)


# --- source tree --------------------------------------------------------------


def test_source_version_reads_exact_declared_version(tmp_path):
    init = tmp_path / "src" / "transformers"
    init.mkdir(parents=True)
    (init / "__init__.py").write_text('__version__ = "5.16.1"\n')
    (tmp_path / "tests" / "models").mkdir(parents=True)
    source.atomic_write(
        tmp_path / source.CACHE_MARKER,
        {"format": source.CACHE_FORMAT, "version": "5.16.1"},
    )
    assert source.source_version(tmp_path) == "5.16.1"
    assert source.is_usable(tmp_path, "5.16.1")
    assert not source.is_usable(tmp_path, "5.16.0")


def test_source_dir_is_version_scoped(tmp_path):
    assert source.source_dir("5.16.1", tmp_path) == tmp_path / "transformers-5.16.1"
    assert source.source_dir("4.50.2", tmp_path) != source.source_dir(
        "5.16.1", tmp_path
    )


def make_archive(tmp_path, unsafe=False):
    archive = tmp_path / "transformers.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        files = {
            "transformers-5.16.1/tests/models/bert/test.py": "def test_ok(): pass\n",
            "transformers-5.16.1/conftest.py": "# conftest\n",
            "transformers-5.16.1/pyproject.toml": "[tool.pytest.ini_options]\n",
            "transformers-5.16.1/src/transformers/__init__.py": '__version__ = "5.16.1"\n',
            "transformers-5.16.1/.torch-fl-hf-source": '{"format": 2, "version": "5.16.1"}\n',
        }
        if unsafe:
            files["transformers-5.16.1/../../escape"] = "no\n"
        for name, text in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(text.encode())
            tar.addfile(info, io.BytesIO(text.encode()))
    return archive


def test_extract_archive_is_complete_and_version_checked(tmp_path):
    dest = tmp_path / "tree"
    source.extract_archive(make_archive(tmp_path), dest)
    assert (dest / "tests/models/bert/test.py").exists()
    assert source.source_version(dest) == "5.16.1"


def test_extract_archive_rejects_unsafe_member(tmp_path):
    with pytest.raises(source.SourceError, match="unsafe"):
        source.extract_archive(make_archive(tmp_path, unsafe=True), tmp_path / "tree")


def make_link_archive(tmp_path, link_target):
    archive = tmp_path / "transformers-links.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        files = {
            "transformers-5.16.1/tests/models/bert/test.py": "def test_ok(): pass\n",
            "transformers-5.16.1/src/transformers/__init__.py": '__version__ = "5.16.1"\n',
            "transformers-5.16.1/.torch-fl-hf-source": '{"format": 2, "version": "5.16.1"}\n',
            "transformers-5.16.1/target.txt": "target\n",
        }
        for name, text in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(text.encode())
            tar.addfile(info, io.BytesIO(text.encode()))
        link = tarfile.TarInfo("transformers-5.16.1/docs/link.txt")
        link.type = tarfile.SYMTYPE
        link.linkname = link_target
        tar.addfile(link)
    return archive


def test_extract_archive_allows_safe_relative_symlink(tmp_path):
    dest = tmp_path / "tree"
    source.extract_archive(make_link_archive(tmp_path, "../target.txt"), dest)
    assert (dest / "docs/link.txt").is_symlink()
    assert (dest / "docs/link.txt").read_text() == "target\n"


@pytest.mark.parametrize("link_target", ["/tmp/outside", "../../../../outside"])
def test_extract_archive_rejects_unsafe_symlink(tmp_path, link_target):
    with pytest.raises(source.SourceError, match="unsafe archive link"):
        source.extract_archive(
            make_link_archive(tmp_path, link_target), tmp_path / "tree"
        )


def test_use_source_rejects_version_mismatch(tmp_path):
    init = tmp_path / "src" / "transformers"
    init.mkdir(parents=True)
    (init / "__init__.py").write_text('__version__ = "4.50.2"\n')
    (tmp_path / "tests" / "models" / "bert").mkdir(parents=True)
    with pytest.raises(source.SourceError, match="declares transformers 4.50.2"):
        source.use_source("5.16.1", tmp_path, None, True)


# --- model resolution and environment ---------------------------------------


def install_fake_transformers_mapping(monkeypatch):
    transformers = types.ModuleType("transformers")
    models = types.ModuleType("transformers.models")
    auto = types.ModuleType("transformers.models.auto")
    configuration = types.ModuleType("transformers.models.auto.configuration_auto")
    configuration.CONFIG_MAPPING_NAMES = {"qwen3": "Qwen3Config", "bert": "BertConfig"}
    configuration.model_type_to_module_name = lambda key: (
        "blip_2" if key == "blip-2" else key
    )
    transformers.models = models
    models.auto = auto
    auto.configuration_auto = configuration
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.models", models)
    monkeypatch.setitem(sys.modules, "transformers.models.auto", auto)
    monkeypatch.setitem(
        sys.modules, "transformers.models.auto.configuration_auto", configuration
    )


def test_module_name_uses_transformers_mapping(monkeypatch, tmp_path):
    install_fake_transformers_mapping(monkeypatch)
    assert runner.module_name("blip-2") == "blip_2"
    assert runner.test_dir(tmp_path, "blip-2") == tmp_path / "tests/models/blip_2"


def test_child_env_sets_hf_device_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    env = runner.child_env(tmp_path, "flagos", tmp_path / "report.jsonl", True)
    assert "TRANSFORMERS_TEST_DEVICE" not in env
    assert env["TRANSFORMERS_TEST_DEVICE_SPEC"] == "hf_device_spec.py"
    assert env["_HF_TESTS_SOURCE"] == str(tmp_path)
    assert env["HF_HUB_OFFLINE"] == "1"
    # HF's own ``tests`` package must win over this repository's.
    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(tmp_path)
    assert str(REPO_ROOT) not in entries


def test_child_env_does_not_override_test_behaviour(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    env = runner.child_env(tmp_path, "flagos", tmp_path / "report.jsonl", False)
    assert "HF_HUB_OFFLINE" not in env
    assert "TRANSFORMERS_OFFLINE" not in env


def test_pytest_command_runs_upstream_tests_unfiltered(tmp_path):
    command = runner.pytest_command(tmp_path / "tests/models/qwen3", "", [], False)
    assert command[1:3] == ["-m", "pytest"]
    assert command[3:] == [str(tmp_path / "tests/models/qwen3")]


def test_known_model_listing_delegates_to_mapping(monkeypatch):
    install_fake_transformers_mapping(monkeypatch)
    assert runner.known_models() == ["bert", "qwen3"]


# --- report reduction ---------------------------------------------------------


def record(nodeid, outcome, when="call", detail=None, wasxfail=False, duration=0.1):
    return {
        "kind": "test",
        "nodeid": nodeid,
        "outcome": outcome,
        "when": when,
        "longrepr": detail,
        "wasxfail": wasxfail,
        "duration": duration,
        "sections": [],
    }


def test_reduce_records_preserves_per_test_outcomes():
    reduced = runner.reduce_records(
        [
            record("a::pass", "passed"),
            record("a::fail", "failed", detail="assertion failed"),
            record("a::cuda", "skipped", detail="test requires CUDA"),
            record("a::skip", "skipped", detail="not applicable"),
            record("a::xfail", "skipped", detail="known issue", wasxfail=True),
        ]
    )
    statuses = {item["nodeid"]: item["status"] for item in reduced["tests"]}
    assert statuses == {
        "a::pass": "PASS",
        "a::fail": "FAIL",
        "a::cuda": "SKIP_CUDA_ONLY",
        "a::skip": "SKIP_OTHER",
        "a::xfail": "XFAIL",
    }


def test_reduce_records_marks_setup_import_error_as_environment_error():
    reduced = runner.reduce_records(
        [
            record(
                "a::setup",
                "failed",
                when="setup",
                detail="ModuleNotFoundError: No module named 'accelerate'",
            )
        ]
    )
    assert reduced["tests"][0]["status"] == "ENVIRONMENT_ERROR"


def test_reduce_records_marks_collection_error_separately():
    reduced = runner.reduce_records(
        [
            {
                "kind": "collect",
                "nodeid": "tests/models/bert",
                "longrepr": "SyntaxError: bad",
            }
        ]
    )
    assert reduced["collect_errors"][0]["status"] == "COLLECT_ERROR"
    assert runner.summarize_statuses([], reduced["collect_errors"]) == {
        "COLLECT_ERROR": 1
    }


def test_reduce_records_aggregates_setup_and_call_without_hiding_setup_error():
    reduced = runner.reduce_records(
        [
            record("a::test", "passed", when="call"),
            record(
                "a::test", "failed", when="teardown", detail="fixture teardown failed"
            ),
        ]
    )
    assert reduced["tests"][0]["status"] == "ERROR"


def test_classification_distinguishes_cuda_and_environment():
    assert runner.classify_skip("test requires a CUDA GPU") == "SKIP_CUDA_ONLY"
    assert runner.classify_skip("not supported on this model") == "SKIP_OTHER"
    assert (
        runner.classify_failure("ImportError: optional package missing")
        == "ENVIRONMENT_ERROR"
    )
    assert runner.classify_failure("assert 1 == 2") == "FAIL"


# --- process/verdict ----------------------------------------------------------


def test_verdict_not_run_is_environment_or_no_tests():
    assert (
        runner.verdict({"run": {"timed_out": False, "crashed": False}, "summary": {}})
        == "NO_TESTS_RUN"
    )
    assert (
        runner.verdict(
            {
                "run": {"timed_out": False, "crashed": False},
                "summary": {"ENVIRONMENT_ERROR": 1},
            }
        )
        == "ENVIRONMENT_ERROR"
    )


def test_verdict_prioritizes_timeout_and_poison():
    base = {"summary": {"FAIL": 2}, "run": {"timed_out": False, "crashed": False}}
    assert runner.verdict({**base, "run": {"timed_out": True}}) == "TIMEOUT"
    assert runner.verdict({**base, "run": {"context_poison": True}}) == "CRASH"
    assert runner.verdict({**base, "run": {"crashed": True}}) == "CRASH"


def test_not_in_version_result_has_no_tests(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "module_name", lambda model: model)
    args = runner.argparse.Namespace(device="flagos", offline=True)
    result = runner.run_tests("bert", tmp_path, args)
    assert result["run"]["status"] == "NOT_IN_VERSION"
    assert result["tests"] == []


def test_collect_only_with_collected_tests_is_a_pass():
    result = {
        "run": {
            "timed_out": False,
            "crashed": False,
            "collect_only": True,
            "collected": 3,
        },
        "summary": {},
    }
    assert runner.verdict(result) == "PASS"


def test_pytest_internal_and_usage_exit_codes_are_not_crashes():
    for returncode in (3, 4, 5):
        assert not runner.pytest_process_crashed(returncode)
    assert runner.pytest_process_crashed(6)
    assert runner.pytest_process_crashed(-11)


def test_fingerprint_normalizes_addresses_and_paths():
    first = {"status": "FAIL", "nodeid": "x", "detail": "ptr 0xabc /tmp/foo 123"}
    second = {"status": "FAIL", "nodeid": "x", "detail": "ptr 0xdef /tmp/bar 456"}
    assert runner.fingerprint("bert", "flagos", first) == runner.fingerprint(
        "bert", "flagos", second
    )


def test_atomic_result_write(tmp_path):
    target = tmp_path / "nested" / "result.json"
    runner.atomic_write(target, {"verdict": "PASS"})
    assert target.exists()
    assert [p.name for p in target.parent.iterdir()] == ["result.json"]
