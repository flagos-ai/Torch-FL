# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[2] / "torch_fl/native_host/load.py"
SPEC = importlib.util.spec_from_file_location("native_host_loader", PATH)
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


@pytest.mark.parametrize("prepare_hook", [False, True])
def test_native_startup_prepares_platform_and_launcher_before_device_init(
    tmp_path, monkeypatch, prepare_hook
):
    path = PATH.with_name("__init__.py")
    spec = importlib.util.spec_from_file_location("native_startup_fixture", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.__file__ = str(tmp_path / "__init__.py")
    (tmp_path / "_generated").mkdir()
    (tmp_path / "_generated/manifest.json").write_text("{}")
    monkeypatch.delenv("_TORCH_FL_BUILDING_NATIVE_HOST", raising=False)
    calls = []
    backend = ModuleType(spec.name + ".backend")
    adapter = SimpleNamespace()
    if prepare_hook:
        adapter.prepare_runtime = lambda: calls.append("compat")
    backend.get_backend = lambda name: adapter

    def install(package):
        assert calls == (["compat"] if prepare_hook else [])
        calls.append("launcher")

    def initialize():
        assert calls[-1] == "launcher"
        calls.append("device")

    class Hosts:
        def __init__(self, path):
            assert calls[-1] == "device"
            calls.append("hosts")

        def activate(self):
            calls.append("activate")

    load = ModuleType(spec.name + ".load")
    load.install_launcher = install
    load.Hosts = Hosts
    vendor = ModuleType(spec.name + ".vendor")
    vendor.activate = lambda: calls.append("vendor")
    torch_fl = ModuleType("torch_fl")
    torch_fl._build_config = SimpleNamespace(
        ACCELERATOR="ascend", NATIVE_HOSTS=True, KERNELS=("flaggems",)
    )
    torch_fl.flagos = SimpleNamespace(init=initialize)
    for name, value in (
        (backend.__name__, backend),
        (load.__name__, load),
        (vendor.__name__, vendor),
        ("torch_fl", torch_fl),
    ):
        monkeypatch.setitem(sys.modules, name, value)
    module._bootstrap_launcher()
    module._activate()
    expected = ["launcher", "device", "hosts", "activate", "vendor"]
    assert calls == (["compat", *expected] if prepare_hook else expected)


@pytest.mark.parametrize("major", [0, 8, 9])
def test_pointwise_helpers_use_published_vendor_capability(monkeypatch, major):
    calls = []
    device_info = ModuleType("flag_gems.utils.device_info")

    def capability():
        calls.append("published")
        return major, 0

    device_info.get_device_capability = capability
    for name in ("flag_gems", "flag_gems.utils"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, device_info.__name__, device_info)
    functions = [object() for _ in range(5)]
    modules = {
        "flag_gems.utils.shape_utils": SimpleNamespace(
            heuristics_for_tile_size=functions[0],
            heuristics_for_num_warps=functions[1],
            stride_order=functions[2],
        ),
        "flag_gems.utils.pointwise_dynamic": SimpleNamespace(
            _balanced_grid_partition=functions[3],
            _tensor_inputs_all_complex=functions[4],
        ),
    }
    assert loader._pointwise_helpers(modules) == (*functions, major)
    assert calls == ["published"]


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
