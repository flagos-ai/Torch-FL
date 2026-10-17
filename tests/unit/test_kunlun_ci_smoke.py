"""CPU-only regressions for the Kunlun CI startup and smoke logging."""

import os
from pathlib import Path
import runpy
import subprocess
import sys
import types

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_allocation_log_does_not_format_tensor(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch_fl", types.ModuleType("torch_fl"))
    smoke = runpy.run_path(str(REPO_ROOT / "tests/manual/kunlun_runtime_smoke.py"))
    allocation = torch.empty((8, 8))

    def unsupported_repr(*args, **kwargs):
        raise AssertionError("tensor repr would dispatch unsupported vendor reductions")

    monkeypatch.setattr(torch.Tensor, "__repr__", unsupported_repr)
    monkeypatch.setattr(torch.Tensor, "__format__", unsupported_repr)
    smoke["check"]("allocation", lambda: allocation)
    assert capsys.readouterr().out == (
        "allocation: PASS (shape=(8, 8), dtype=torch.float32, device=cpu)\n"
    )


def test_smoke_propagates_runtime_failures(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch_fl", types.ModuleType("torch_fl"))
    smoke = runpy.run_path(str(REPO_ROOT / "tests/manual/kunlun_runtime_smoke.py"))

    def failed_allocation():
        raise RuntimeError("device allocation failed")

    with pytest.raises(RuntimeError, match="device allocation failed"):
        smoke["check"]("allocation", failed_allocation)
    assert "allocation: FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("inherited_flag", [None, "1"])
def test_profiler_flag_precedes_first_torch_import(tmp_path, inherited_flag):
    probe_log = tmp_path / "probe.log"
    vendor_python = tmp_path / "vendor-python"
    vendor_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "${XPU_ENABLE_PROFILER_TRACING-unset}" > "$PROBE_LOG"\n'
        "exit 42\n"
    )
    vendor_python.chmod(0o755)
    env = os.environ.copy()
    env.pop("BASH_ENV", None)
    env.pop("XPU_ENABLE_PROFILER_TRACING", None)
    if inherited_flag is not None:
        env["XPU_ENABLE_PROFILER_TRACING"] = inherited_flag
    env.update(
        CI_STAGE="integration",
        TORCH_FL_VENDOR_PYTHON=str(vendor_python),
        PROBE_LOG=str(probe_log),
    )
    # Integration jobs copy tests to a wheel-only workspace; setup scripts
    # remain in the original checkout exposed through GITHUB_WORKSPACE.
    checkout = Path(os.environ.get("GITHUB_WORKSPACE", str(REPO_ROOT)))
    result = subprocess.run(
        ["bash", str(checkout / ".github/scripts/set_env_kunlun.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    # Stop at the interpreter probe: this test must not install or build anything.
    assert result.returncode == 1
    assert "Vendor Python cannot import torch" in result.stderr
    assert probe_log.read_text() == "1\n"
