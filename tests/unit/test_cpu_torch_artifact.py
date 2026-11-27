# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU wheel delivery preserves the published artifact and works offline."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/prepare_cpu_torch.py"
SPEC = importlib.util.spec_from_file_location("prepare_cpu_torch", SCRIPT)
CPU = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CPU)
VERSION = "2.10.0"
INDEX = "https://download.pytorch.org/whl/cpu"
TORCH = "torch-2.10.0+cpu-cp311-cp311-linux_aarch64.whl"
CONSUMERS = [
    ("integration-tests.yml", "integration-tests"),
    ("build-wheel-common.yml", "build"),
]


def _cohort(directory):
    files = {
        TORCH: b"original published wheel",
        "dependency-1-py3-none-any.whl": b"dep",
    }
    for name, data in files.items():
        (directory / name).write_bytes(data)
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    record = {
        "schema_version": 1,
        "index_url": INDEX,
        "torch_wheel": TORCH,
        "files": checksums,
    }
    (directory / "cohort.json").write_text(json.dumps(record))
    return checksums[TORCH]


def test_delivered_wheels_match_the_original_cohort(tmp_path):
    checksum = _cohort(tmp_path)
    result = CPU.verify(tmp_path, VERSION, checksum, INDEX)
    assert result["torch_wheel"] == TORCH


@pytest.mark.parametrize("damage", ["torch", "dependency", "missing", "extra"])
def test_incomplete_or_changed_delivery_is_rejected(tmp_path, damage):
    checksum = _cohort(tmp_path)
    if damage == "missing":
        (tmp_path / TORCH).unlink()
    elif damage == "extra":
        (tmp_path / "unexpected-1-py3-none-any.whl").write_bytes(b"extra")
    else:
        filename = TORCH if damage == "torch" else "dependency-1-py3-none-any.whl"
        (tmp_path / filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match="file set|checksum"):
        CPU.verify(tmp_path, VERSION, checksum, INDEX)


@pytest.mark.parametrize("pin", ["version", "index", "sha256"])
def test_a_different_cpu_build_is_rejected(tmp_path, pin):
    checksum = _cohort(tmp_path)
    with pytest.raises(ValueError, match="pin"):
        CPU.verify(
            tmp_path,
            "2.11.0" if pin == "version" else VERSION,
            "0" * 64 if pin == "sha256" else checksum,
            "https://other.invalid/cpu" if pin == "index" else INDEX,
        )


def test_hosted_download_targets_the_hardware_python_and_original_index(
    tmp_path, monkeypatch
):
    commands = []

    def run(command, *, check):
        assert check
        commands.append(command)
        _cohort(tmp_path)

    monkeypatch.setattr(CPU.subprocess, "run", run)
    checksum = hashlib.sha256(b"original published wheel").hexdigest()
    CPU.download(tmp_path, VERSION, "3.11", INDEX, TORCH, checksum)
    command = commands[0]
    assert command[:5] == [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
    ]
    assert command[command.index("--index-url") + 1] == INDEX
    assert command[command.index("--abi") + 1] == "cp311"
    assert command[-1] == "torch==2.10.0"
    assert "--no-deps" not in command
    assert "linux_aarch64" in command


@pytest.mark.parametrize("wrong_hash", [False, True])
def test_accelerator_download_keeps_its_exact_release_and_original_bytes(
    tmp_path, monkeypatch, wrong_hash
):
    commands = []
    wheel = "flagtree-0.7.0+ascend3.5-cp311-cp311-linux_aarch64.whl"
    payload = b"published compiler wheel"
    checksum = hashlib.sha256(payload).hexdigest()

    def run(command, *, check):
        assert check
        commands.append(command)
        if command[-1] == "torch==2.10.0":
            _cohort(tmp_path)
        else:
            (tmp_path / wheel).write_bytes(payload)

    monkeypatch.setattr(CPU.subprocess, "run", run)
    args = (
        tmp_path,
        VERSION,
        "3.11",
        INDEX,
        TORCH,
        hashlib.sha256(b"original published wheel").hexdigest(),
        [
            (
                "flagtree",
                "0.7.0+ascend3.5",
                "https://published.invalid/simple",
                wheel,
                "0" * 64 if wrong_hash else checksum,
            )
        ],
    )
    if wrong_hash:
        with pytest.raises(ValueError, match="Published wheel differs"):
            CPU.download(*args)
    else:
        result = CPU.download(*args)
        assert result["files"][wheel] == checksum
    assert len(commands) == 2
    command = commands[-1]
    assert "--no-deps" in command
    assert command[-1] == "flagtree===0.7.0+ascend3.5"
    assert (
        command[command.index("--index-url") + 1] == "https://published.invalid/simple"
    )


@pytest.mark.parametrize(
    "function,distribution,prefix",
    [
        ("install_flagtree", "flagtree", "FLAGTREE"),
        ("install_flag_gems", "flag-gems", "FLAGGEMS"),
        ("install_flagcx", "flagcx", "FLAGCX"),
    ],
)
@pytest.mark.parametrize("changed", [False, True])
def test_accelerator_install_uses_pinned_delivery_even_when_preinstalled(
    tmp_path, function, distribution, prefix, changed
):
    checksum = _cohort(tmp_path)
    wheel = f"{distribution.replace('-', '_')}-1.0-py3-none-any.whl"
    original = hashlib.sha256(b"published release").hexdigest()
    payload = b"changed release" if changed else b"published release"
    (tmp_path / wheel).write_bytes(payload)
    record_path = tmp_path / "cohort.json"
    record = json.loads(record_path.read_text())
    record["files"][wheel] = hashlib.sha256(payload).hexdigest()
    record_path.write_text(json.dumps(record))
    done = subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail\nsource "{ROOT}/.github/scripts/lib/set_env_common.sh"\n'
            "prefer_direct_route() { echo UNEXPECTED-NETWORK; return 1; }\n"
            "flag_gems_installed() { return 0; }\n"
            'pip_retry() { printf "PIP %s\\n" "$*"; }\n'
            f"{function}\n",
        ],
        env={
            **os.environ,
            "TORCH_FL_CPU_TORCH_WHEELHOUSE": str(tmp_path),
            "CPU_TORCH_VERSION": VERSION,
            "CPU_TORCH_INDEX_URL": INDEX,
            "CPU_TORCH_WHEEL_SHA256_ascend": checksum,
            "FLAGOS_ACCELERATOR": "ascend",
            "PIP_RETRY_PYTHON": sys.executable,
            f"{prefix}_VERSION": "1.0",
            f"{prefix}_WHEEL_ascend": wheel,
            f"{prefix}_WHEEL_SHA256_ascend": original,
        },
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert "UNEXPECTED-NETWORK" not in done.stdout
    if changed:
        assert done.returncode != 0
        assert "PIP " not in done.stdout
        assert "Published wheel differs" in done.stderr
    else:
        assert done.returncode == 0, done.stderr
        assert f"--no-index --find-links {tmp_path}" in done.stdout
        assert (
            f"--force-reinstall --no-deps --only-binary=:all: {distribution}===1.0"
            in done.stdout
        )


@pytest.mark.parametrize("changed", [False, True])
def test_install_uses_only_verified_delivery_without_index_probes(tmp_path, changed):
    checksum = _cohort(tmp_path)
    if changed:
        (tmp_path / TORCH).write_bytes(b"different build")
    done = subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail\nsource "{ROOT}/.github/scripts/lib/set_env_common.sh"\n'
            "prepare_cpu_torch_index_route() { echo UNEXPECTED-NETWORK; return 1; }\n"
            'pip_retry() { printf "PIP %s\\n" "$*"; }\n'
            "install_cpu_torch --no-cache-dir\n",
        ],
        env={
            **os.environ,
            "TORCH_FL_CPU_TORCH_WHEELHOUSE": str(tmp_path),
            "CPU_TORCH_VERSION": VERSION,
            "CPU_TORCH_INDEX_URL": INDEX,
            "CPU_TORCH_WHEEL_SHA256_ascend": checksum,
            "FLAGOS_ACCELERATOR": "ascend",
            "PIP_RETRY_PYTHON": sys.executable,
        },
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert "UNEXPECTED-NETWORK" not in done.stdout
    if changed:
        assert done.returncode != 0
        assert "PIP " not in done.stdout
        assert "checksum mismatch" in done.stderr
    else:
        assert done.returncode == 0, done.stderr
        assert (
            f"PIP --no-index --find-links {tmp_path} --no-cache-dir torch==2.10.0"
            in done.stdout
        )


@pytest.mark.parametrize("workflow,job", CONSUMERS)
def test_only_ascend_prefetches_and_other_hardware_lanes_still_run(workflow, job):
    data = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    jobs = data["jobs"]
    assert "== 'ascend'" in jobs["cpu-torch"]["if"]
    consumer = jobs[job]
    assert "needs.cpu-torch.result == 'skipped'" in consumer["if"]
    assert "!cancelled()" in consumer["if"]
    steps = consumer["steps"]
    delivery = [step for step in steps if "id" in step]
    setup = next(
        step for step in steps if "TORCH_FL_CPU_TORCH_WHEELHOUSE" in step.get("env", {})
    )
    receive, retry, verify = (
        next(step for step in delivery if step["id"] == step_id)
        for step_id in ("receive-wheels", "retry-wheels", "verify-wheelhouse")
    )
    assert steps.index(receive) < steps.index(retry) < steps.index(verify)
    assert steps.index(verify) < steps.index(setup)
    assert receive["uses"] == retry["uses"] == "actions/download-artifact@v4"
    assert receive["with"]["name"] == retry["with"]["name"]
    assert receive["with"]["path"] == "${{ steps.wheelhouse.outputs.path }}"
    assert retry["with"]["path"] == "${{ steps.wheelhouse.outputs.retry-path }}"
    assert retry["timeout-minutes"] < receive["timeout-minutes"]
    assert retry["if"] == (
        "needs.cpu-torch.outputs.artifact_name != '' && "
        "steps.receive-wheels.outputs.download-path == ''"
    )
    assert verify["if"] == "needs.cpu-torch.result == 'success'"
    assert verify["env"]["WHEELHOUSE_PATH"] == (
        "${{ steps.receive-wheels.outputs.download-path || "
        "steps.retry-wheels.outputs.download-path }}"
    )
    assert setup["env"]["TORCH_FL_CPU_TORCH_WHEELHOUSE"] == (
        "${{ steps.verify-wheelhouse.outputs.path }}"
    )
    assert "--check-build-env" in next(
        step["run"]
        for step in steps
        if step.get("name") == "Validate wheel compatibility manifest"
    )


@pytest.mark.parametrize("workflow,job", CONSUMERS)
def test_delivery_does_not_reuse_a_previous_jobs_wheels(tmp_path, workflow, job):
    data = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    steps = data["jobs"][job]["steps"]
    prepare = next(step for step in steps if step.get("id") == "wheelhouse")
    old = tmp_path / "cpu-torch-wheelhouse"
    old.mkdir()
    (old / "old-release.whl").write_bytes(b"previous cohort")
    output = tmp_path / "github-output"
    directories = []
    for _ in range(2):
        output.write_text("")
        subprocess.run(
            ["bash", "-c", prepare["run"]],
            env={
                **os.environ,
                "RUNNER_TEMP": str(tmp_path),
                "GITHUB_OUTPUT": str(output),
            },
            check=True,
            capture_output=True,
            timeout=10,
        )
        paths = dict(line.split("=", 1) for line in output.read_text().splitlines())
        initial, retry = Path(paths["path"]), Path(paths["retry-path"])
        assert initial.parent == retry.parent
        assert initial.parent.parent == tmp_path
        for path in (initial, retry):
            assert list(path.iterdir()) == []
            (path / "partial.whl").touch()
            directories.append(path)
    assert len(set(directories)) == 4
    assert (old / "old-release.whl").read_bytes() == b"previous cohort"


@pytest.mark.parametrize("workflow,job", CONSUMERS)
@pytest.mark.parametrize(
    "damage,message",
    [
        (None, "Verified CPU Torch artifact"),
        ("no-output", "No completed download of the pinned Ascend wheelhouse"),
        ("missing", "file set differs from its manifest"),
        ("checksum", "CPU wheelhouse checksum mismatch"),
        ("vendor-pin", "Published wheel differs from the pinned artifact"),
    ],
)
def test_only_verified_delivery_is_passed_to_setup(
    tmp_path, workflow, job, damage, message
):
    data = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    verify = next(
        step
        for step in data["jobs"][job]["steps"]
        if step.get("id") == "verify-wheelhouse"
    )
    wheelhouse = tmp_path / "received wheels"
    wheelhouse.mkdir()
    pins = {
        "CPU_TORCH_VERSION_DEFAULT": VERSION,
        "CPU_TORCH_INDEX_URL_DEFAULT": INDEX,
        "CPU_TORCH_WHEEL_SHA256_ascend": _cohort(wheelhouse),
    }
    manifest = wheelhouse / "cohort.json"
    record = json.loads(manifest.read_text())
    for package in ("FLAGTREE", "FLAGGEMS", "FLAGCX"):
        name = f"{package.lower()}-1.0-py3-none-any.whl"
        payload = package.encode()
        checksum = hashlib.sha256(payload).hexdigest()
        (wheelhouse / name).write_bytes(payload)
        record["files"][name] = checksum
        pins[f"{package}_WHEEL_ascend"] = name
        pins[f"{package}_WHEEL_SHA256_ascend"] = checksum
    manifest.write_text(json.dumps(record))
    if damage == "missing":
        (wheelhouse / TORCH).unlink()
    elif damage == "checksum":
        (wheelhouse / TORCH).write_bytes(b"truncated")
    elif damage == "vendor-pin":
        pins["FLAGCX_WHEEL_SHA256_ascend"] = "0" * 64

    scripts = tmp_path / ".github/scripts"
    scripts.mkdir(parents=True)
    (scripts / SCRIPT.name).symlink_to(SCRIPT)
    (scripts.parent / "version-pins.env").write_text(
        "".join(f"{name}={value}\n" for name, value in pins.items())
    )
    output = tmp_path / "github-output"
    output.touch()
    done = subprocess.run(
        ["bash", "-c", verify["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "WHEELHOUSE_PATH": "" if damage == "no-output" else str(wheelhouse),
            "GITHUB_OUTPUT": str(output),
        },
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert message in done.stdout + done.stderr
    if damage:
        assert done.returncode != 0
        assert not output.read_text()
    else:
        assert done.returncode == 0, done.stderr
        assert output.read_text() == f"path={wheelhouse}\n"
