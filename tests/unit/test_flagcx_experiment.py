# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""The Ascend experiment must preserve isolation and reject false acceptance."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import pytest
import yaml
from packaging.utils import parse_wheel_filename
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "flagcx_experiment", ROOT / ".github/scripts/run_flagcx_experiment.py"
)
EXPERIMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPERIMENT)


def wheel_fixture(tmp_path):
    wheel = tmp_path / "flagcx-0.14.0+cann9.0.0.910c-cp311-cp311-linux_aarch64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "flagcx-0.14.0.dist-info/METADATA",
            "Name: flagcx\nVersion: 0.14.0+cann9.0.0.910c\n",
        )
        archive.writestr("flagcx/_C.cpython-311-aarch64-linux-gnu.so", b"ELF fixture")
    return SimpleNamespace(
        wheelhouse=tmp_path,
        wheel=wheel.name,
        version="0.14.0+cann9.0.0.910c",
        sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        torch_fl_wheel=tmp_path / "torch_fl-0.1.0-py3-none-any.whl",
        index_url="https://example.invalid/simple",
        output=tmp_path / "report",
        workdir=tmp_path,
    )


@pytest.mark.parametrize(
    "vendor_library,probe_exit", [(True, None), (False, 0), (False, 7)]
)
def test_native_dependency_gate_precedes_install_and_import(
    tmp_path, monkeypatch, vendor_library, probe_exit
):
    args = wheel_fixture(tmp_path)
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "ascend")
    monkeypatch.setenv("PYTHONPATH", "/baseline/reporting")
    monkeypatch.setenv("FLAGCX_TORCH_BACKEND", "vendor")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "readelf":
            needed = "libtorch_npu.so" if vendor_library else "libtorch_cpu.so"
            return SimpleNamespace(stdout=f" (NEEDED) Shared library: [{needed}]")
        assert command[:4] == [sys.executable, "-m", "pip", "install"]
        assert "--no-deps" in command and "--no-index" in command
        assert "--target" in command
        assert command[-2:] == [
            str(args.torch_fl_wheel),
            str(args.wheelhouse / args.wheel),
        ]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def collectives(command, workdir, env, log):
        assert env["FLAGCX_TORCH_BACKEND"] == "flagos"
        assert env["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
        assert env["PYTHONPATH"] != "/baseline/reporting"
        assert workdir == tmp_path
        assert "--nproc-per-node=2" in command
        return probe_exit

    monkeypatch.setattr(EXPERIMENT.subprocess, "run", run)
    monkeypatch.setattr(EXPERIMENT, "run_collectives", collectives)
    assert EXPERIMENT.experiment(args) == int(vendor_library or probe_exit != 0)
    report = json.loads((args.output / "result.json").read_text())
    assert report["status"] == (
        "blocked" if vendor_library else "passed" if probe_exit == 0 else "failed"
    )
    assert report["admission_status"] == ("blocked" if vendor_library else "passed")
    assert report["import_attempted"] == (not vendor_library)
    assert report["runtime_compatibility"] == (
        "not_tested"
        if vendor_library
        else "validated"
        if probe_exit == 0
        else "unverified"
    )
    if vendor_library:
        assert report["reason_type"] == "environment_policy"
        assert "No import or collective was attempted" in report["reason"]
        assert (
            "Runtime compatibility: not_tested" in (tmp_path / "summary.md").read_text()
        )
    assert report["collectives_started"] == (not vendor_library)
    assert len(calls) == (1 if vendor_library else 2)
    assert report["required"] is False
    assert report["baseline_environment_modified"] is False
    assert report["torch_fl_wheel"] == args.torch_fl_wheel.name
    assert os.environ["PYTHONPATH"] == "/baseline/reporting"
    assert os.environ["FLAGCX_TORCH_BACKEND"] == "vendor"


@pytest.mark.parametrize(
    "case",
    [
        "absent",
        "shim",
        "vendor",
        "orphan",
        "foreign-child",
        "native-extension",
        "loaded-library",
    ],
)
def test_python_shim_is_distinct_from_vendor_modules_and_native_library(
    monkeypatch, case
):
    monkeypatch.syspath_prepend(str(ROOT / ".github/scripts"))
    import check_platform_environment as checks

    for name in list(sys.modules):
        if name == "torch_npu" or name.startswith("torch_npu."):
            monkeypatch.delitem(sys.modules, name)
    root = ModuleType("torch_npu")
    root.__torch_fl_shim__ = True
    root._C = ModuleType("torch_npu._C")
    if case not in ("absent", "orphan"):
        monkeypatch.setitem(sys.modules, "torch_npu", root)
        monkeypatch.setitem(sys.modules, "torch_npu._C", root._C)
    if case == "vendor":
        root.__torch_fl_shim__ = False
        root.__file__ = "/image/torch_npu/__init__.py"
    elif case == "orphan":
        monkeypatch.setitem(sys.modules, "torch_npu._C", root._C)
    elif case == "foreign-child":
        monkeypatch.setitem(
            sys.modules, "torch_npu.vendor", ModuleType("torch_npu.vendor")
        )
    elif case == "native-extension":
        root._C.__file__ = "/image/torch_npu/_C.so"
    original = checks.assert_loaded_torch_libraries
    mappings = (
        "100-200 r-xp 0000 00:00 1 /image/libtorch_npu.so\n"
        if case == "loaded-library"
        else ""
    )
    monkeypatch.setattr(
        checks,
        "assert_loaded_torch_libraries",
        lambda root, names, forbidden: original(root, names, forbidden, mappings),
    )
    if case in ("absent", "shim"):
        EXPERIMENT.assert_no_vendor_torch_npu()
    else:
        with pytest.raises(AssertionError, match="vendor|Vendor"):
            EXPERIMENT.assert_no_vendor_torch_npu()


@pytest.mark.parametrize("damage", ["sha256", "version", "platform"])
def test_corrupt_or_out_of_scope_input_is_reported_without_running_tools(
    tmp_path, monkeypatch, damage
):
    args = wheel_fixture(tmp_path)
    monkeypatch.setenv(
        "FLAGOS_ACCELERATOR", "musa" if damage == "platform" else "ascend"
    )
    if damage == "sha256":
        args.sha256 = "0" * 64
    elif damage == "version":
        args.version = "0.14.1+cann9.0.0"
    monkeypatch.setattr(
        EXPERIMENT.subprocess,
        "run",
        lambda *a, **k: pytest.fail("Unexpected tool execution"),
    )
    assert EXPERIMENT.experiment(args) == 1
    report = json.loads((args.output / "result.json").read_text())
    assert report["status"] == "failed"
    assert report["collectives_started"] is False


def test_download_failure_retains_a_separate_experiment_result(tmp_path, monkeypatch):
    args = wheel_fixture(tmp_path)
    args.wheelhouse = None
    monkeypatch.setenv("FLAGOS_ACCELERATOR", "ascend")
    monkeypatch.setattr(
        EXPERIMENT.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="download unavailable"
        ),
    )
    assert EXPERIMENT.experiment(args) == 1
    assert (args.output / "download.log").read_text() == "download unavailable"
    assert json.loads((args.output / "result.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("needs_kill", [False, True])
def test_timed_out_collectives_clean_only_their_own_process_group(
    monkeypatch, needs_kill
):
    calls = []

    class Process:
        pid = 12345

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self, timeout=None):
            calls.append(timeout)
            if len(calls) == 1 or (needs_kill and len(calls) == 2):
                raise subprocess.TimeoutExpired("collectives", timeout)
            return 0

    monkeypatch.setattr(EXPERIMENT.subprocess, "Popen", lambda *a, **k: Process())
    signals = []
    monkeypatch.setattr(
        EXPERIMENT.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    with pytest.raises(subprocess.TimeoutExpired):
        EXPERIMENT.run_collectives(["collectives"], ".", {}, None, timeout=1)
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_successful_non_flagcx_fallback_is_rejected(monkeypatch, backend):
    monkeypatch.syspath_prepend(str(ROOT / ".github/scripts"))
    import check_platform_environment as checks

    monkeypatch.setattr(checks, "assert_loaded_torch_libraries", lambda *a, **k: {})
    for name in list(sys.modules):
        if name == "torch_npu" or name.startswith("torch_npu."):
            monkeypatch.delitem(sys.modules, name)
    shim = ModuleType("torch_npu")
    shim.__torch_fl_shim__ = True
    shim._C = ModuleType("torch_npu._C")
    monkeypatch.setitem(sys.modules, "torch_npu", shim)
    monkeypatch.setitem(sys.modules, "torch_npu._C", shim._C)
    calls = []
    dist = SimpleNamespace(
        init_process_group=lambda *a, **k: calls.append("init"),
        destroy_process_group=lambda: calls.append("destroy"),
        distributed_c10d=SimpleNamespace(
            _get_default_group=lambda: SimpleNamespace(
                _inner=SimpleNamespace(_get_backend_name=lambda: backend)
            )
        ),
    )
    torch = SimpleNamespace(
        __version__="2.10.0+cpu", version=SimpleNamespace(cuda=None), distributed=dist
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setitem(
        sys.modules,
        "torch_fl",
        SimpleNamespace(flagos=SimpleNamespace(set_device=lambda _: None)),
    )
    monkeypatch.setitem(sys.modules, "flagcx", SimpleNamespace())
    monkeypatch.setattr(
        importlib.metadata, "version", lambda _: "0.14.0+cann9.0.0.910c"
    )
    monkeypatch.setenv("FLAGCX_EXPERIMENT_VERSION", "0.14.0+cann9.0.0.910c")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(AssertionError, match="Expected FlagCX"):
        EXPERIMENT.collective_worker()
    assert calls == ["init", "destroy"]


def test_prefetch_and_experiment_keep_distinct_acceptance_scopes():
    prefetch = yaml.safe_load(
        (ROOT / ".github/workflows/prepare-cpu-torch.yml").read_text()
    )
    download = next(
        s["run"]
        for s in prefetch["jobs"]["prepare"]["steps"]
        if s.get("id") == "download"
    )
    assert "--release-wheel flagcx" not in download
    assert (
        "--release-wheel flagtree" in download
        and "--release-wheel flag-gems" in download
    )
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/integration-tests.yml").read_text()
    )
    steps = next(
        j["steps"]
        for j in workflow["jobs"].values()
        if "steps" in j and any(s.get("id") == "integration-tests" for s in j["steps"])
    )
    experiment = next(s for s in steps if s.get("id") == "flagcx-experiment")
    assert "inputs.platform == 'ascend'" in experiment["if"]
    assert experiment["continue-on-error"] is True
    assert "--torch-fl-wheel dist/*.whl" in experiment["run"]
    assert steps.index(experiment) > next(
        i for i, s in enumerate(steps) if s.get("id") == "integration-tests"
    )
    pins = dict(
        line.split("=", 1)
        for line in (ROOT / ".github/version-pins.env").read_text().splitlines()
        if line and not line.startswith("#") and "=" in line
    )
    _, version, _, _ = parse_wheel_filename(pins["FLAGCX_EXPERIMENT_WHEEL_ascend"])
    assert version == Version(pins["FLAGCX_EXPERIMENT_VERSION_ascend"])
    assert pins["FLAGCX_VERSION_ascend"] == "0.14.1+cann9.0.0.910c"
    for option, pin in (
        ("--wheel", "FLAGCX_EXPERIMENT_WHEEL_ascend"),
        ("--version", "FLAGCX_EXPERIMENT_VERSION_ascend"),
        ("--sha256", "FLAGCX_EXPERIMENT_WHEEL_SHA256_ascend"),
    ):
        assert f'{option} "${pin}"' in experiment["run"]
