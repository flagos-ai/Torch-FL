# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[2] / "torch_fl/native_host/load.py"
SPEC = importlib.util.spec_from_file_location("native_host_loader", PATH)
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


def test_unrelated_application_compiler_is_not_blocked():
    before = list(loader._native_commands)
    loader.audit_native_compilers("subprocess.Popen", ("gcc", ["gcc", "application.c"]))
    assert loader._native_commands == before


def test_flaggems_native_compiler_is_blocked_but_device_assembler_is_allowed():
    namespace = {"audit": loader.audit_native_compilers}
    exec(
        compile(
            "def call(command):\n    audit('subprocess.Popen', (command[0], command))\n",
            "/installed/flag_gems/utils/compiler.py",
            "exec",
        ),
        namespace,
    )
    with pytest.raises(RuntimeError, match="runtime native compilation is forbidden"):
        namespace["call"](["/usr/bin/g++", "launcher.cc"])
    namespace["call"](["/usr/local/cuda/bin/ptxas", "kernel.ptx"])
    assert loader._native_commands[-1] == ["/usr/local/cuda/bin/ptxas", "kernel.ptx"]


def test_launcher_rejects_kernel_without_flaggems_provenance():
    source = SimpleNamespace(fn=object())
    with pytest.raises(RuntimeError, match="no admitted FlagGems source"):
        loader.kernel_launcher(source, SimpleNamespace(name="foreign_kernel"))


def test_launcher_rejects_source_changed_after_admission(monkeypatch):
    function = SimpleNamespace(src="changed source")
    monkeypatch.setitem(
        loader._admitted_kernels,
        id(function),
        {"jit_sha256": hashlib.sha256(b"original source").hexdigest()},
    )
    with pytest.raises(RuntimeError, match="JIT source changed after admission"):
        loader.kernel_launcher(
            SimpleNamespace(fn=function), SimpleNamespace(name="changed_kernel")
        )
