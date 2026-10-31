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

"""Protect diff routing, full nightly selection and honest functional results."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import time
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts"
sys.path.insert(0, str(SCRIPTS))
from flagcicd_plan import build_plan  # noqa: E402
from merge_pytest_reports import merge_reports  # noqa: E402
from ci.generate_matrix import (  # noqa: E402
    _load_platforms,
    _platforms_from_files,
    _schedule,
    boxing_platforms,
)
from compose_integration_manifest import compose, ManifestError  # noqa: E402

sys.path.pop(0)
CONTRACT = yaml.safe_load((ROOT / ".flagcicd/project.yaml").read_text())
CONFIGS = ROOT / ".github/configs"


def process_stopped(pid):
    """A disappearing /proc entry is an exited process, including ESRCH on read."""
    pid = int(pid)
    if pid <= 0:
        raise ValueError("expected a positive process ID")
    try:
        status = (Path("/proc") / str(pid) / "status").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return True
    return "State:\tZ" in status


def wait_for_pid(path, deadline):
    """Wait for the PID bytes, not only creation of the fixture's marker file."""
    while time.monotonic() < deadline:
        try:
            pid = int(path.read_text())
            if pid > 0:
                return pid
        except (FileNotFoundError, ValueError):
            pass
        time.sleep(0.02)
    pytest.fail(f"no complete PID was written to {path}")


def test_pid_wait_handles_creation_before_the_write(monkeypatch, tmp_path):
    values = iter([FileNotFoundError(), "", "12345"])

    def read(*args, **kwargs):
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(Path, "read_text", read)
    assert wait_for_pid(tmp_path / "pid", time.monotonic() + 1) == 12345


@pytest.mark.parametrize("error", [FileNotFoundError, ProcessLookupError])
def test_process_exit_during_status_read_is_stopped(monkeypatch, error):
    def vanished(*args, **kwargs):
        raise error("process was reaped during the read")

    monkeypatch.setattr(Path, "read_text", vanished)
    assert process_stopped(12345)


@pytest.mark.parametrize("state,stopped", [("Z", True), ("S", False)])
def test_process_status_distinguishes_running_from_zombie(monkeypatch, state, stopped):
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: f"State:\t{state}\n")
    assert process_stopped(12345) is stopped


def test_process_status_permission_errors_are_not_hidden(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("cannot inspect the process")

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError):
        process_stopped(12345)


def request(profile="pr", platforms="all"):
    return {
        "source_repository": "flagos-ai/Torch-FL",
        "source_sha": "a" * 40,
        "base_sha": "b" * 40,
        "requested_level": "auto",
        "platforms": platforms,
        "profile": profile,
        "task_id": "",
        "parent_run_id": "",
    }


@pytest.mark.parametrize(
    "changed,expected",
    [
        (["csrc/aten/backends/mthreads/mudnn_copy.cc"], ["mthreads"]),
        (["csrc/aten/backends/mthreads/mudnn_copy.cc", "csrc/aten/copy_ops.cc"], None),
        (["README.md"], []),
    ],
)
def test_pr_router_and_external_entry_agree(changed, expected):
    known = _load_platforms(CONFIGS)
    expected = (
        _schedule(known, boxing_platforms(CONFIGS, known))
        if expected is None
        else expected
    )
    assert _platforms_from_files(changed, known, CONFIGS) == expected
    assert (
        build_plan(request(), CONTRACT, ROOT, changed)["expected_platforms"] == expected
    )


@pytest.mark.parametrize(
    "changed", [["README.md"], [".github/configs/mthreads.yml"], None]
)
def test_nightly_always_selects_every_configured_platform(changed):
    planned = build_plan(request("nightly"), CONTRACT, ROOT, changed)
    known = _load_platforms(CONFIGS)
    assert planned["expected_platforms"] == _schedule(
        known, boxing_platforms(CONFIGS, known)
    )
    assert planned["test_level"] == "L2"
    assert planned["profile"] == "nightly"


def test_profile_identity_and_reserved_weekly():
    changed = ["setup.py"]
    assert (
        build_plan(request(), CONTRACT, ROOT, changed)["config_hash"]
        != build_plan(request("nightly"), CONTRACT, ROOT, changed)["config_hash"]
    )
    with pytest.raises(ValueError, match="weekly is reserved"):
        build_plan(request("weekly"), CONTRACT, ROOT, changed)
    with pytest.raises(ValueError, match="nightly requires platforms=all"):
        build_plan(request("nightly", "mthreads"), CONTRACT, ROOT, changed)


def test_nightly_keeps_all_existing_groups_and_isolates_cuda_profiling():
    manifest = yaml.safe_load((ROOT / ".github/integration-manifest.yml").read_text())
    known = _load_platforms(CONFIGS)
    boxing = boxing_platforms(CONFIGS, known)
    for platform in known:
        config = yaml.safe_load((CONFIGS / f"{platform}.yml").read_text())
        for boxing_lane in (False, True) if platform in boxing else (False,):
            kwargs = {"boxing": boxing, "boxing_lane": boxing_lane}
            pr = compose(manifest, config, platform, known, profile="pr", **kwargs)
            nightly = compose(
                manifest, config, platform, known, profile="nightly", **kwargs
            )
            assert {g["name"] for g in pr} <= {g["name"] for g in nightly}
            extras = {g["name"] for g in nightly} - {g["name"] for g in pr}
            expected = platform == "nvidia" and not boxing_lane
            assert len(extras) == (2 if expected else 0)
    with pytest.raises(ManifestError, match="weekly is reserved"):
        compose(manifest, config, platform, known, boxing=boxing, profile="weekly")


def test_profile_is_threaded_through_both_workflow_entries():
    workflows = ROOT / ".github/workflows"
    ci = yaml.safe_load((workflows / "ci.yml").read_text())
    assert ci["jobs"]["platform-pipeline"]["with"]["test_profile"]
    entry = yaml.safe_load((workflows / "flagcicd-entry.yml").read_text())
    calls = [
        job
        for job in entry["jobs"].values()
        if job.get("uses") == "./.github/workflows/integration-tests.yml"
    ]
    assert calls and all("test_profile" in job["with"] for job in calls)
    integration = yaml.safe_load((workflows / "integration-tests.yml").read_text())
    assert integration["jobs"]["configuration"]["with"]["test_profile"] == (
        "${{ inputs.test_profile || 'pr' }}"
    )
    loader = (workflows / "load-platform-config.yml").read_text()
    assert '--profile "${{ inputs.test_profile }}"' in loader


def test_serial_failure_records_unstarted_groups(tmp_path):
    marker = tmp_path / "must-not-run"
    groups = [
        {"name": "fails", "command": "exit 7"},
        {"name": "later", "command": "touch " + shlex.quote(str(marker))},
    ]
    env = {
        **os.environ,
        "INTEGRATION_TESTS": json.dumps(groups),
        "INTEGRATION_ENVIRONMENT": "{}",
        "FLAGCICD_REPORT_DIR": str(tmp_path),
    }
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_integration_tests.py")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 7
    assert not marker.exists()
    records = json.loads((tmp_path / "groups.json").read_text())["groups"]
    assert [g["status"] for g in records] == ["failed", "not_run"]


@pytest.mark.parametrize("escape", [False, True])
def test_command_timeout_kills_descendants(tmp_path, escape):
    spec = importlib.util.spec_from_file_location(
        "functional_runner", SCRIPTS / "run_integration_tests.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    pid_file = tmp_path / "child.pid"
    child = "import time; time.sleep(30)"
    code = f"import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session={escape!r}); Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(30)"
    command = shlex.join([sys.executable, "-c", code])
    result, _ = runner.run_command(
        command, {**os.environ, "FLAGCICD_CASE_TIMEOUT": "1"}, str(tmp_path)
    )
    assert result == 124
    assert process_stopped(pid_file.read_text())


@pytest.mark.parametrize("missing", ["result", "group", "selection", "all_skipped"])
def test_incomplete_or_empty_evidence_cannot_pass(tmp_path, missing):
    (tmp_path / "session.pending").write_text(
        json.dumps({"group": "smoke", "selected_nodeids": ["test_sample.py::test_ok"]})
    )
    report = {
        "summary": {"passed": 1},
        "tests": [{"nodeid": "test_sample.py::test_ok"}],
        "duration": 0.1,
        "exitcode": 0,
    }
    (tmp_path / "session.json").write_text(json.dumps(report))
    groups = {
        "expected": ["smoke"],
        "groups": [{"name": "smoke", "status": "passed", "requires_pytest": True}],
    }
    (tmp_path / "groups.json").write_text(json.dumps(groups))
    assert merge_reports(tmp_path, "mthreads", "success", "repo", "sha", True)[
        "flagcicd"
    ]["complete"]
    if missing == "result":
        (tmp_path / "session.json").unlink()
    elif missing == "group":
        groups["groups"][0]["status"] = "not_run"
        (tmp_path / "groups.json").write_text(json.dumps(groups))
    elif missing == "selection":
        (tmp_path / "session.pending").write_text(
            json.dumps({"group": "smoke", "selected_nodeids": ["missing"]})
        )
    else:
        report["summary"] = {"skipped": 1}
        (tmp_path / "session.json").write_text(json.dumps(report))
    actual = merge_reports(tmp_path, "mthreads", "success", "repo", "sha", True)
    assert actual["flagcicd"]["complete"] is False
    assert actual["exitcode"] != 0


@pytest.mark.parametrize(
    "escape,launcher_exits",
    [
        (False, False),
        (True, False),
        pytest.param(
            True,
            True,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="Linux orphan adoption contract"
            ),
        ),
    ],
)
def test_cancellation_kills_the_active_command_group(tmp_path, escape, launcher_exits):
    pid_file = tmp_path / "active.pid"
    code = f"import os,time; from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
    if escape:
        child = "import time; time.sleep(30)"
        code = f"import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session=True); Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(30)"
    launcher_pid = tmp_path / "launcher.pid"
    if launcher_exits:
        code = (
            "import os; "
            + code.rsplit("time.sleep(30)", 1)[0]
            + f"Path({str(launcher_pid)!r}).write_text(str(os.getpid())); sys.exit(0)"
        )
    group = {"name": "active", "command": shlex.join([sys.executable, "-c", code])}
    env = {
        **os.environ,
        "INTEGRATION_TESTS": json.dumps([group]),
        "INTEGRATION_ENVIRONMENT": "{}",
        "FLAGCICD_REPORT_DIR": str(tmp_path),
    }
    process = subprocess.Popen(
        [sys.executable, str(SCRIPTS / "run_integration_tests.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        active_pid = wait_for_pid(pid_file, deadline)
        if launcher_exits:
            launcher = wait_for_pid(launcher_pid, deadline)
            while time.monotonic() < deadline:
                if process_stopped(launcher):
                    break
                time.sleep(0.02)
            else:
                pytest.fail("the launcher did not exit before cancellation")
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        assert process.returncode != 0
        assert process_stopped(active_pid)
        assert (
            json.loads((tmp_path / "groups.json").read_text())["groups"][0]["status"]
            == "cancelled"
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


@pytest.mark.parametrize("profile,expected_code", [("nightly", 0), ("weekly", 2)])
def test_automatic_workflow_profile_selection_before_hardware(
    tmp_path, profile, expected_code
):
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    step = next(
        s
        for s in workflow["jobs"]["discover-platforms"]["steps"]
        if s.get("id") == "detect"
    )
    output = tmp_path / "output"
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "TEST_PROFILE": profile,
        "EVENT_NAME": "schedule",
        "DISPATCH_PLATFORMS": "",
        "PR_LABELS_JSON": "[]",
        "BASE_REF": "main",
        "BASE_SHA": "",
        "HEAD_SHA": "HEAD",
        "GITHUB_OUTPUT": str(output),
    }
    done = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert done.returncode == expected_code, done.stdout + done.stderr
    if expected_code == 0:
        known = _load_platforms(CONFIGS)
        assert json.loads(output.read_text().split("=", 1)[1]) == _schedule(
            known, boxing_platforms(CONFIGS, known)
        )
    else:
        assert "weekly is reserved" in done.stdout
        assert not output.exists()


def test_source_contracts_have_one_shared_cpu_owner():
    workflows = ROOT / ".github/workflows"
    commands = [
        step.get("run", "")
        for filename in ("agnostic-checks.yml", "codegen-checks.yml")
        for job in yaml.safe_load((workflows / filename).read_text())["jobs"].values()
        for step in job.get("steps", [])
    ]
    for target in (
        "scripts/codegen/gen_vendor_confs.py --check",
        "tests/unit/test_gen_vendor_confs.py",
        "tests/unit/test_dtype_cast_arm_contract.py",
    ):
        assert sum(command.count(target) for command in commands) == 1, target


def test_integration_callers_require_the_shared_cpu_owners():
    workflows = ROOT / ".github/workflows"
    for name in ("ci.yml", "flagcicd-entry.yml"):
        jobs = yaml.safe_load((workflows / name).read_text())["jobs"]
        cpu = {
            key
            for key, job in jobs.items()
            if job.get("uses", "").endswith(
                ("agnostic-checks.yml", "codegen-checks.yml")
            )
        }
        assert len(cpu) == 2

        def ancestors(name):
            needs = jobs[name].get("needs", [])
            if isinstance(needs, str):
                needs = [needs]
            return set(needs).union(*(ancestors(parent) for parent in needs))

        for key, job in jobs.items():
            if job.get("uses", "").endswith("integration-tests.yml"):
                assert cpu <= ancestors(key), (name, key)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux orphan adoption contract")
@pytest.mark.parametrize("holds_output", [False, True])
@pytest.mark.parametrize("launcher_exit", [0, 7])
def test_timeout_reaps_workers_after_their_launcher_exits(
    tmp_path, holds_output, launcher_exit
):
    spec = importlib.util.spec_from_file_location(
        "orphan_runner", SCRIPTS / "run_integration_tests.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    pid_file = tmp_path / "worker.pid"
    worker = (
        "import os,time; from pathlib import Path; "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(10)"
    )
    output = "None" if holds_output else "subprocess.DEVNULL"
    launcher = (
        "import os,subprocess,sys; "
        f"subprocess.Popen([sys.executable,'-c',{worker!r}], "
        f"start_new_session=True,stdout={output},stderr={output}); "
        f"os._exit({launcher_exit})"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    started = time.monotonic()
    try:
        code, _ = runner.run_command(
            shlex.join([sys.executable, "-c", launcher]),
            {**os.environ, "FLAGCICD_CASE_TIMEOUT": "1"},
            str(tmp_path),
        )
        assert code == 124
        assert time.monotonic() - started < 4
        status = Path("/proc") / pid_file.read_text() / "status"
        assert not status.exists()
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait()
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("exitcode", [0, 7])
def test_command_drains_partial_utf8_output_and_preserves_exit_status(
    tmp_path, exitcode
):
    spec = importlib.util.spec_from_file_location(
        "stream_runner", SCRIPTS / "run_integration_tests.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    code = (
        "import os,time; os.write(1,bytes([228])); time.sleep(0.05); "
        f"os.write(1,bytes([184,173])); raise SystemExit({exitcode})"
    )
    result, output = runner.run_command(
        shlex.join([sys.executable, "-c", code]), os.environ.copy(), str(tmp_path)
    )
    assert result == exitcode
    assert output == "中"
