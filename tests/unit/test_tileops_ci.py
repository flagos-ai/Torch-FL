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

"""The TileOPs experiment keeps its own source and hardware gates."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import subprocess

import pytest
import yaml


REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "tileops-ci.yml"
INSTALLER = REPO / ".github" / "scripts" / "tileops_metax_install.sh"
SOURCE_CHECK = REPO / "scripts" / "tools" / "tileops_source_check.py"
PINS = REPO / ".github" / "version-pins.env"
CONFIGS = REPO / "torch_fl" / "configs"


def _yaml():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _pins():
    values = {}
    for line in PINS.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key] = value
    return values


def test_tileops_is_not_added_to_the_platform_matrix():
    configs = {path.stem for path in (REPO / ".github" / "configs").glob("*.yml")}
    assert "tileops" not in configs


def test_source_job_runs_generation_and_routing_contracts():
    workflow = _yaml()
    source = workflow["jobs"]["source"]
    commands = "\n".join(step.get("run", "") for step in source["steps"])
    assert "scripts/tools/tileops_source_check.py" in commands
    assert "scripts/codegen/gen_vendor_confs.py --check" in commands
    for test in (
        "test_tileops_ci.py",
        "test_tileops_runtime.py",
        "test_codegen_tileops_recipes.py",
        "test_gen_vendor_confs.py",
        "test_conf_registration_consistency.py",
        "test_integration_workdir.py",
        "test_wheel_preflight.py",
    ):
        assert test in commands, test


def test_relevant_build_inputs_trigger_the_source_job():
    paths = _yaml()[True]["pull_request"]["paths"]
    for path in (
        ".github/scripts/hooks/set_env_metax.sh",
        ".github/constraints.txt",
        ".github/scripts/set_env.sh",
        "cmake/flagos_platforms.json",
        "scripts/vendor/bundle_metax_libtorch.sh",
        "setup.py",
    ):
        assert path in paths


def test_cpu_job_never_asserts_a_hardware_environment():
    workflow = _yaml()
    source = workflow["jobs"]["source"]
    commands = "\n".join(step.get("run", "") for step in source["steps"])
    assert "check_platform_environment.py" not in commands
    assert "FLAGOS_ACCELERATOR=metax" not in commands


def test_hardware_job_is_manual_or_nightly_only():
    workflow = _yaml()
    condition = workflow["jobs"]["metax"]["if"]
    assert "github.event_name == 'schedule'" in condition
    assert "github.event_name == 'workflow_dispatch'" in condition
    assert "inputs.metax == true" in condition
    assert "pull_request" not in condition


def test_hardware_build_enables_only_tileops_and_requires_private_access():
    workflow = _yaml()
    metax = workflow["jobs"]["metax"]
    by_name = {step["name"]: step for step in metax["steps"] if "name" in step}
    build_env = by_name["Build the non-boxing TileOPs wheel"]["env"]
    assert {
        key: value for key, value in build_env.items() if key.startswith("FLAGOS_")
    } == {
        "FLAGOS_BUILD_FLAGGEMS": "0",
        "FLAGOS_BUILD_FLAGGEMS_CPP": "0",
        "FLAGOS_BUILD_BOXING": "0",
        "FLAGOS_BUILD_TILEOPS": "1",
    }
    token = by_name["Require private provider access"]["env"]["TILEOPS_GITHUB_TOKEN"]
    assert token == "${{ secrets.TILEOPS_GITHUB_TOKEN }}"
    install = by_name["Validate and install the experiment wheel"]["run"]
    assert '("tileops",)' in install
    assert by_name["Build the non-boxing TileOPs wheel"]["run"] == (
        "python -m build --wheel --no-isolation"
    )
    stack = by_name["Install pinned TileOPs MetaX stack"]["env"]
    assert stack["PIP_EXTRA_INDEX_URL"] == ""


def test_source_check_stages_a_minimal_source_package_read_only():
    namespace = {
        "__name__": "tileops_source_check",
        "__file__": str(SOURCE_CHECK),
        "__package__": None,
    }
    exec(compile(SOURCE_CHECK.read_text(), str(SOURCE_CHECK), "exec"), namespace)
    assert namespace["SOURCE_PATHS"] == (
        ".github",
        "cmake",
        "scripts",
        "csrc",
        "torch_fl",
        "tests",
    )
    text = SOURCE_CHECK.read_text()
    assert "from torch_fl import _env, _platform, _vendor" in text
    assert "initializer.write_text(STAGED_INITIALIZER" in text
    assert "initializer.unlink()" not in text
    assert '"PYTHONPATH": os.pathsep.join(pythonpath)' in text
    assert 'if existing := os.environ.get("PYTHONPATH")' in text


def test_installer_uses_a_secret_safe_askpass_and_verifies_commits():
    text = INSTALLER.read_text(encoding="utf-8")
    assert 'GIT_ASKPASS="$askpass"' in text
    assert "$TILEOPS_GITHUB_TOKEN" in text
    assert 'mkdir -p "$destination"' in text
    assert 'git -C "$destination" rev-parse HEAD' in text
    assert 'fetch --depth 1 origin "$commit"' in text
    assert "submodule sync --recursive" in text
    assert "submodule update --init --recursive --depth 1" in text
    assert "USE_MACA=ON CIBUILDWHEEL=1" in text
    assert "3rdparty/tvm/3rdparty/tvm-ffi" in text
    assert "https://x-access-token:" not in text
    assert "${TILEOPS_METAX_COMMIT:-" in text
    assert "target.build.tilelang_maca" in text
    assert "torch-c-dlpack-ext==0.1.5" in text
    assert "repos.metax-tech.com" in text
    assert 'PIP_CONSTRAINT="${REPO_ROOT}/.github/constraints.txt"' in text


def test_all_tileops_pins_are_full_commits(tmp_path):
    pins = _pins()
    for key in (
        "TILEOPS_OFFICIAL_COMMIT_DEFAULT",
        "TILEOPS_ASCEND_COMMIT_DEFAULT",
        "TILEOPS_MUSA_COMMIT_DEFAULT",
        "TILEOPS_METAX_COMMIT_DEFAULT",
        "TILELANG_METAX_COMMIT_DEFAULT",
    ):
        assert re.fullmatch(r"[0-9a-f]{40}", pins[key]), (key, pins[key])


@pytest.mark.parametrize(
    "platform,expected",
    [
        ("ascend", {"tileops": 49, "ascend": 372, "none": 1621}),
        ("mthreads", {"tileops": 59, "mthreads": 109, "none": 1874}),
        ("metax", {"tileops": 60, "none": 1982}),
    ],
)
def test_configs_remain_closed_tileops_experiments(platform, expected):
    routes = Counter()
    for line in (
        (CONFIGS / f"backends_{platform}_tileops.conf").read_text().splitlines()
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        routes[stripped.split("=", 1)[1].split("#", 1)[0].strip()] += 1
    assert routes == expected
    assert "flaggems" not in routes
    assert "boxing" not in routes


def test_smoke_has_four_native_routes_and_fallback():
    text = (
        REPO / "tests" / "integration" / "ops" / "test_tileops_vendor_smoke.py"
    ).read_text()
    for operation in ("relu", "add.Tensor", "sum.dim_IntList", "_softmax"):
        assert operation in text
    assert "transpose.int" in text
    assert "pytest.skip" not in text

    workflow = _yaml()
    by_name = {
        step["name"]: step
        for step in workflow["jobs"]["metax"]["steps"]
        if "name" in step
    }
    command = next(
        step["run"]
        for step in workflow["jobs"]["metax"]["steps"]
        if step.get("name") == "Run the five-case TileOPs smoke"
    )
    assert "--min-passed" not in command
    assert "test_tileops_vendor_smoke.py" in command
    assert (
        by_name["Run the five-case TileOPs smoke"]["env"]["TILEOPS_LOG_DIR"]
        == "${{ runner.temp }}/tileops-dispatch-logs"
    )

    verify = next(
        step["run"]
        for step in workflow["jobs"]["metax"]["steps"]
        if step.get("name") == "Require all five smoke cases to pass"
    )
    for needle in ('"tests": 5', '"skipped": 0', '"failures": 0', '"errors": 0'):
        assert needle in verify
    evidence = next(
        step["run"]
        for step in workflow["jobs"]["metax"]["steps"]
        if step.get("name") == "Run the five-case TileOPs smoke"
    )
    assert 'target == "tileops_maca"' in evidence
    assert "torch_fl.tileops" in evidence


def test_hardware_artifact_is_uploaded_even_when_the_smoke_fails():
    workflow = _yaml()
    upload = next(
        step
        for step in workflow["jobs"]["metax"]["steps"]
        if step.get("name") == "Upload TileOPs evidence"
    )
    assert upload["if"] == "always()"
    assert "tileops-smoke.xml" in upload["with"]["path"]
    assert "*.dispatch.log" in upload["with"]["path"]
    assert "tileops-dispatch-logs/*.dispatch.log" in upload["with"]["path"]


def test_installer_is_valid_bash():
    done = subprocess.run(
        ["bash", "-n", str(INSTALLER)],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


def test_metax_hook_leaves_the_default_lane_on_flaggems():
    hook = (REPO / ".github" / "scripts" / "hooks" / "set_env_metax.sh").read_text()
    assert 'if [[ "${FLAGOS_BUILD_TILEOPS:-0}" == "1" ]]; then' in hook
    assert "export FLAGOS_BUILD_FLAGGEMS=0" in hook
    assert "export FLAGOS_BUILD_FLAGGEMS=1" in hook
    assert (
        hook.count(
            '[[ "$CI_STAGE" == "integration" && "${FLAGOS_BUILD_TILEOPS:-0}" != "1" ]]'
        )
        == 2
    )
