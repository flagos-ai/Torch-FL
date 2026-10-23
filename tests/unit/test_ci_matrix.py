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

"""Exercise real platform configurations and conservative PR selection."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/ci/generate_matrix.py"
CONFIGS = ROOT / ".github/configs"
SPEC = importlib.util.spec_from_file_location("ci_matrix", SCRIPT)
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)
PLATFORMS = MATRIX._load_platforms(CONFIGS)


def _run(event="pull_request", files="", labels="", dispatch=""):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--event-name",
            event,
            "--configs-dir",
            str(CONFIGS),
            "--changed-files",
            files,
            "--pr-labels",
            labels,
            "--dispatch-platforms",
            dispatch,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return result, None
    return result, json.loads(result.stdout.removeprefix("platforms="))


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize(
    "path",
    [
        "torch_fl/accelerator/{platform}/__init__.py",
        "csrc/aten/backends/{platform}/op.cc",
        ".github/scripts/hooks/set_env_{platform}.sh",
        ".github/scripts/set_env_{platform}.sh",
        ".github/configs/{platform}.yml",
        ".github/workflows/integration-test-{platform}.yml",
    ],
)
def test_platform_owned_files_select_only_that_lane(platform, path):
    result, selected = _run(files=path.format(platform=platform))
    assert result.returncode == 0, result.stderr
    assert selected == [platform]


@pytest.mark.parametrize(
    "files",
    [
        "",
        "README.md",
        "docs/development/testing.md",
        "docs/assets/ci.png",
        "README.md\ndocs/guide.rst",
    ],
)
def test_documentation_needs_no_hardware(files):
    assert _run(files=files)[1] == []


def test_a_conf_is_still_named_for_its_platform_id():
    """Step 4 renames the confs; until then each one selects only its own lane."""
    for lane, conf in (
        ("nvidia", "nvidia"),
        ("hygon", "hygon"),
        ("mthreads", "mthreads"),
        ("enflame", "enflame"),
        ("thead", "thead"),
        ("apple", "apple"),
    ):
        assert _run(files=f"torch_fl/configs/backends_{conf}.conf")[1] == [lane], lane


@pytest.mark.parametrize(
    "path",
    [
        "torch_fl/__init__.py",
        "csrc/aten/generated/register.inc",
        "csrc/runtime/generator.cc",
        ".github/scripts/set_env.sh",
        ".github/scripts/lib/set_env_common.sh",
        ".github/version-pins.env",
        "cmake/flagos_platforms.json",
        "tests/integration/ops/test_add_dispatch.py",
        ".github/workflows/ci.yml",
        "setup.py",
        "pyproject.toml",
        "unknown/future_backend.cc",
        "prefix/torch_fl/accelerator/ascend/op.py",
    ],
)
def test_shared_or_unknown_code_runs_every_lane_even_with_vendor_changes(path):
    assert _run(files=f"torch_fl/accelerator/ascend/op.py\n{path}")[1] == PLATFORMS


def test_multiple_vendors_select_the_union():
    assert _run(
        files="torch_fl/accelerator/ascend/op.py\ncsrc/aten/backends/mthreads/op.cc"
    )[1] == ["ascend", "mthreads"]


def test_shared_cuda_runtime_selects_all_consumers():
    assert _run(files="csrc/runtime/accelerator/nvidia/device.cc")[1] == [
        "hygon",
        "nvidia",
        "thead",
    ]
    assert _run(files="csrc/runtime/allocator/backends/nvidia_memory.h")[1] == [
        "metax",
        "nvidia",
        "thead",
    ]


@pytest.mark.parametrize(
    "path",
    [
        "csrc/runtime/accelerator/iluvatar/device.cc",
        "csrc/runtime/allocator/backends/iluvatar_memory.h",
    ],
)
def test_independent_iluvatar_runtime_selects_only_iluvatar(path):
    assert _run(files=path)[1] == ["iluvatar"]


@pytest.mark.parametrize("platform", PLATFORMS)
def test_runtime_watch_paths_cover_the_build_tables_actual_source_directory(platform):
    """`runtime_dir` is the directory the runtime sources live in; the lane name is
    the vendor. A CUDA-ABI vendor's runtime directory is shared with the platform
    that defines it, so the lane must be in the selected set rather than equal to
    the only one."""
    table = json.loads((ROOT / "cmake/flagos_platforms.json").read_text())
    row = next(r for r in table["accelerators"].values() if r["vendor"] == platform)
    runtime = row["runtime_dir"]
    assert platform in _run(files=f"csrc/runtime/accelerator/{runtime}/device.cc")[1]


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
def test_non_pr_events_default_to_every_lane(event):
    assert (
        _run(event=event, files="README.md", labels="platform:ascend")[1] == PLATFORMS
    )


@pytest.mark.parametrize(
    "labels, expected",
    [
        ("platform:mthreads,platform:ascend,platform:mthreads", ["ascend", "mthreads"]),
        ("platform:ascend,platform:all", PLATFORMS),
        ("platform:unknown", PLATFORMS),
        ("platform:ascend,platform:unknown", PLATFORMS),
    ],
)
def test_routing_labels_override_file_filters_without_duplicates(labels, expected):
    assert _run(files="README.md", labels=labels)[1] == expected


@pytest.mark.parametrize(
    "requested, expected",
    [
        ("", PLATFORMS),
        ("all", PLATFORMS),
        ("mthreads,ascend,mthreads", ["ascend", "mthreads"]),
    ],
)
def test_manual_selection(requested, expected):
    assert _run(event="workflow_dispatch", dispatch=requested)[1] == expected


@pytest.mark.parametrize("requested", ["unknown", "all,ascend", "ascend,unknown"])
def test_invalid_manual_selection_fails(requested):
    result, selected = _run(event="workflow_dispatch", dispatch=requested)
    assert result.returncode == 2
    assert selected is None


def test_missing_watch_metadata_falls_back_to_all(tmp_path):
    (tmp_path / "cuda.yml").write_text("display_name: CUDA\n")
    assert MATRIX._platforms_from_files(["op.cc"], ["nvidia"], tmp_path) == ["nvidia"]
    assert MATRIX._platforms_from_files(None, PLATFORMS, CONFIGS) == PLATFORMS


def test_shared_mps_runtime_changes_select_the_mps_lane():
    """The MPS lane runs the shared runtime suite, so its edits reach the lane."""
    for path in (
        "tests/integration/runtime/test_runtime_dispatch_route.py",
        "tests/integration/flaggems/flaggems_caps.py",
        "tests/unit/test_runtime_conformance_report.py",
    ):
        result, selected = _run(files=path)
        assert result.returncode == 0, result.stderr
        assert selected == ["apple"], path


def test_mps_suites_with_shared_filenames_keep_the_mps_lane():
    """An MPS-only change must not hide an MPS suite shared with other lanes."""
    for path in (
        "csrc/aten/strided_ops.cc",
        "tests/integration/ops/backend_conf.py",
        "tests/integration/ops/test_linear_dispatch.py",
        "tests/integration/ops/test_mps_qwen35.py",
        "tests/integration/ops/test_mps_rotary_and_compare.py",
    ):
        result, selected = _run(files=path)
        assert result.returncode == 0, result.stderr
        assert selected == ["apple"], path


def test_git_diff_preserves_deleted_and_renamed_platform_paths(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True
        ).strip()

    git("init", "-q")
    git("config", "user.name", "CI test")
    git("config", "user.email", "ci@example.invalid")
    git("config", "commit.gpgsign", "false")
    old = tmp_path / "ascend.cc"
    old.write_text("original")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    old.rename(tmp_path / "musa.cc")
    git("add", "-A")
    git("commit", "-qm", "rename")
    monkeypatch.chdir(tmp_path)
    assert MATRIX._changed_files("unused", base) == ["ascend.cc", "musa.cc"]
    assert MATRIX._changed_files("missing") is None
