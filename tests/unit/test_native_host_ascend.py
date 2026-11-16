# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend packed ABI: scalar promotion, layout, and immutable launch metadata."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ascend_host_adapter", ROOT / "torch_fl/native_host/backends/ascend.py"
)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


@pytest.fixture
def inputs(monkeypatch):
    for name in (
        "TRITON_COMPILE_ONLY",
        "TRITON_DEVICE_PRINT",
        "TRITON_REGISTER_TENSOR_MSPROF",
        "TRITON_ENABLE_TASKQUEUE",
    ):
        monkeypatch.delenv(name, raising=False)
    source = SimpleNamespace(
        fn=object(), signature={"value": ("*fp16", "bf16", "i1"), "scale": "fp16"}
    )
    metadata = SimpleNamespace(
        mix_mode="aiv", enable_auto_blockify=None, debug_enabled=False
    )
    utils = SimpleNamespace(
        get_ascend_arch_from_env=lambda: "published-arch",
        is_ffts_supported=lambda arch: True,
        force_disable_ffts=lambda: False,
        _is_auto_map_parallel_blocks_enabled=lambda: True,
    )
    runtime = SimpleNamespace(
        get_aivector_core_num=lambda: 48, get_aicore_num=lambda: 24
    )
    return source, metadata, utils, runtime


def test_metadata_normalization_preserves_original_sources_and_boolean_types(inputs):
    source, metadata, utils, runtime = inputs
    prepared_source, prepared = adapter.prepare(*inputs)
    assert prepared_source.fn is source.fn
    assert prepared_source.signature == {
        "value": ("*fp16", "fp32", "i1"),
        "scale": "fp32",
    }
    assert source.signature["scale"] == "fp16"
    assert not hasattr(metadata, "_flagos_ascend_types")
    assert prepared._flagos_ascend_types == ("*fp16", "fp32", "i1", "fp32")
    assert prepared._flagos_ascend_ffts
    assert prepared._flagos_ascend_auto_blocks
    assert prepared._flagos_ascend_blocks == 48


@pytest.mark.parametrize("mode,blocks", [("aiv", 48), ("aic", 24), ("mix", 24)])
def test_physical_block_mapping_uses_the_published_runtime(inputs, mode, blocks):
    inputs[1].mix_mode = mode
    assert adapter.prepare(*inputs)[1]._flagos_ascend_blocks == blocks


def test_per_kernel_block_mapping_overrides_the_environment_default(inputs):
    inputs[1].enable_auto_blockify = False
    _, metadata = adapter.prepare(*inputs)
    assert not metadata._flagos_ascend_auto_blocks


@pytest.mark.parametrize(
    "name",
    [
        "TRITON_COMPILE_ONLY",
        "TRITON_DEVICE_PRINT",
        "TRITON_REGISTER_TENSOR_MSPROF",
        "TRITON_ENABLE_TASKQUEUE",
    ],
)
def test_unsupported_hidden_abis_fail_before_kernel_creation(monkeypatch, inputs, name):
    monkeypatch.setenv(name, "true")
    with pytest.raises(RuntimeError, match=name):
        adapter.prepare(*inputs)


@pytest.mark.parametrize("name", ["debug_enabled", "debug_launch_hidden_arg"])
def test_debugger_hidden_pointer_is_not_silently_omitted(inputs, name):
    setattr(inputs[1], name, True)
    with pytest.raises(RuntimeError, match="debugger arguments"):
        adapter.prepare(*inputs)


def test_prebuilt_driver_installation_is_repeatable_and_uses_no_host_compiler(
    monkeypatch, inputs
):
    class OriginalUtils:
        def __init__(self):
            pytest.fail("published constructor would compile C++")

        def get_arch(self):
            return self.npu_utils_mod.arch

    driver = SimpleNamespace(NPUUtils=OriginalUtils)
    compiler_utils_alias = driver.NPUUtils

    class Driver:
        def __init__(self):
            self.utils = driver.NPUUtils()
            self.launcher_cls = driver.NPULauncher

        def get_current_device(self):
            return 3

    driver.NPUDriver = Driver
    active = object.__new__(Driver)
    runtime_driver = SimpleNamespace(_active=active, _default=None)
    runtime_package = SimpleNamespace(driver=runtime_driver)
    monkeypatch.setitem(sys.modules, "triton.runtime", runtime_package)
    fake_torch = SimpleNamespace(
        flagos=SimpleNamespace(current_device=lambda: 3),
        device=lambda name, index: (name, index),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    published_utils = inputs[2]
    inputs[1].enable_auto_blockify = False
    utility = SimpleNamespace(arch="published-soc")
    modules = {
        "triton.backends.ascend.driver": driver,
        "triton.backends.ascend.utils": published_utils,
        "first.npu_utils": utility,
        "second.npu_utils": SimpleNamespace(arch="another-soc"),
    }
    monkeypatch.setattr(
        adapter, "importlib", SimpleNamespace(import_module=modules.__getitem__)
    )
    calls = []

    def factory(source, metadata):
        calls.append((source, metadata))
        return "prebuilt-kernel"

    for package in ("first", "second"):
        adapter.Backend().install(
            package,
            SimpleNamespace(current_stream=lambda device: 100 + device),
            factory,
        )
        assert active.utils.npu_utils_mod is modules[package + ".npu_utils"]
        assert driver.NPUUtils is compiler_utils_alias
        assert compiler_utils_alias() is active.utils
        future = Driver()
        assert future.utils.get_arch() == modules[package + ".npu_utils"].arch
        assert active.get_current_stream() == 103
        assert future.get_current_stream(5) == 105
        assert future.get_active_torch_device() == ("flagos", 3)
        assert future.launcher_cls(*inputs[:2]) == "prebuilt-kernel"
        assert calls[-1][0].signature["scale"] == "fp32"
        with pytest.raises(RuntimeError, match="compilation is forbidden"):
            driver.make_npu_launcher_stub("unused")
        with pytest.raises(RuntimeError, match="compilation is forbidden"):
            driver._build_npu_ext("unused")


@pytest.mark.parametrize("layout", ["legacy", "package"])
@pytest.mark.parametrize("module", ["_launcher", "npu_utils"])
def test_sdk_runtime_headers_find_their_profiling_dependency(
    tmp_path, monkeypatch, layout, module
):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("SDK include conformance requires a host C++ compiler")
    cann = tmp_path / "cann"
    runtime = cann / (
        "include/experiment/runtime/runtime"
        if layout == "legacy"
        else "pkg_inc/runtime/runtime"
    )
    profiling = cann / (
        "include/experiment/msprof" if layout == "legacy" else "pkg_inc/profiling"
    )
    runtime.mkdir(parents=True)
    profiling.mkdir(parents=True)
    (runtime / "rt.h").write_text("#include <prof_common.h>\n")
    (profiling / "prof_common.h").write_text(
        "#pragma once\n#define FLAGOS_PROFILING_HEADER 1\n"
    )
    monkeypatch.setenv("ASCEND_HOME", str(cann))
    torch = SimpleNamespace(
        __file__=str(tmp_path / "torch/__init__.py"),
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
    )
    extensions = adapter.Backend().extensions(
        "probe",
        ROOT / "torch_fl/native_host",
        torch,
        {"native_source_sha256": "0" * 64},
        tmp_path / "triton",
    )
    extension = next(e for e in extensions if e.name == "probe." + module)
    source = tmp_path / "probe.cc"
    source.write_text(
        "#include <runtime/runtime/rt.h>\n"
        "static_assert(FLAGOS_PROFILING_HEADER == 1);\n"
    )
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsyntax-only",
            *["-I" + path for path in extension.include_dirs],
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_packed_arguments_match_the_published_launcher_struct(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("packed ABI conformance requires a host C++ compiler")
    # The actual field declarations/alignment follow make_launcher() in the
    # published FlagTree 0.7.0+ascend3.5 driver (SHA256 da34b0e53e48056980cb8dfe8e5120f8dd3bae279004ceab730cb6314ec6445d).
    source = tmp_path / "conformance.cc"
    source.write_text(
        '#include "ascend_abi.h"\n#include <cassert>\n#include <cstddef>\n'
        "using flagos::native_host::ascend::Layout;\n"
        "struct __attribute__((packed)) Published {\n"
        "void* ffts __attribute__((aligned(8)));\n"
        "void* lock __attribute__((aligned(8)));\n"
        "void* workspace __attribute__((aligned(8)));\n"
        "int8_t small __attribute__((aligned(4)));\n"
        "uint16_t medium __attribute__((aligned(4)));\n"
        "int32_t boolean __attribute__((aligned(4)));\n"
        "float scale __attribute__((aligned(4)));\n"
        "uint64_t large __attribute__((aligned(8)));\n"
        "void* pointer __attribute__((aligned(8)));\n"
        "int32_t x __attribute__((aligned(4)));\n"
        "int32_t y __attribute__((aligned(4)));\n"
        "int32_t z __attribute__((aligned(4)));\n};\n"
        "int main() {\n"
        "int8_t small=-17; uint16_t medium=65535; uint8_t boolean=1;\n"
        "float scale=1.0001f; uint64_t large=UINT64_MAX, pointer=0x12345678;\n"
        "void* values[]={&small,&medium,&boolean,&scale,&large,&pointer};\n"
        'Layout layout({"i8","u16","i1","fp32","u64","*fp16"},true,false);\n'
        "auto bytes=layout.Pack(values,{3,5,7},11,22,33);\n"
        "assert(bytes.size()==sizeof(Published));\n"
        "Published actual; std::memcpy(&actual,bytes.data(),sizeof(actual));\n"
        "assert(actual.ffts==reinterpret_cast<void*>(11));\n"
        "assert(actual.lock==reinterpret_cast<void*>(22));\n"
        "assert(actual.workspace==reinterpret_cast<void*>(33));\n"
        "assert(actual.small==small && actual.medium==medium && actual.boolean==1);\n"
        "assert(actual.scale==scale && actual.large==large);\n"
        "assert(actual.pointer==reinterpret_cast<void*>(pointer));\n"
        "assert(actual.x==3 && actual.y==5 && actual.z==7);\n"
        'Layout simt({"fp32"},false,true); void* single[]={&scale};\n'
        "auto simt_bytes=simt.Pack(single,{1,2,3},0,0,0);\n"
        "assert(simt_bytes.size()==16);\n"
        "try { layout.Pack(values,{uint64_t(INT32_MAX)+1,1,1},0,0,0); return 1; }\n"
        "catch(const std::overflow_error&) {}\n"
        'try { Layout invalid({"tensordesc<fp32>"},false,false); return 2; }\n'
        "catch(const std::invalid_argument&) {}\n"
        "}\n"
    )
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Werror",
            "-I" + str(ROOT / "torch_fl/native_host/csrc"),
            str(source),
            "-o",
            str(tmp_path / "conformance"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(tmp_path / "conformance")], check=True, timeout=10)


def test_native_backend_syntax_with_cpu_torch_and_runtime_declarations(tmp_path):
    torch = pytest.importorskip("torch")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("backend syntax check requires a host C++ compiler")
    # These declarations mock the runtime boundary only. Real SDK compilation
    # and kernel execution remain hardware-CI gates.
    acl = tmp_path / "acl"
    acl.mkdir()
    (acl / "acl.h").write_text(
        "#pragma once\n#include <cstdint>\n"
        "using aclrtStream=void*; using aclmdlRI=void*;\n"
        "enum aclmdlRICaptureStatus {ACL_MODEL_RI_CAPTURE_STATUS_NONE, ACTIVE};\n"
        "constexpr int ACL_SUCCESS=0;\n"
        "int aclmdlRICaptureGetInfo(aclrtStream,aclmdlRICaptureStatus*,aclmdlRI*);\n"
    )
    runtime = tmp_path / "runtime/runtime"
    runtime.mkdir(parents=True)
    (runtime / "rt.h").write_text(
        "#pragma once\n#include <cstdint>\n"
        "using rtError_t=int; using rtStream_t=void*;\n"
        "constexpr int RT_ERROR_NONE=0,RT_MEMCPY_HOST_TO_DEVICE=1;\n"
        "struct rtArgsEx_t {void* args; uint32_t argsSize;};\n"
        "struct rtTaskCfgInfo_t {uint32_t localMemorySize;};\n"
        "int rtMemcpy(void*,uint64_t,const void*,uint64_t,int);\n"
        "int rtGetC2cCtrlAddr(uint64_t*,uint32_t*);\n"
        "int rtKernelLaunch(const void*,uint32_t,void*,uint32_t,void*,rtStream_t);\n"
        "int rtKernelLaunchWithFlagV2(const void*,uint32_t,rtArgsEx_t*,void*,rtStream_t,uint32_t,rtTaskCfgInfo_t*);\n"
    )
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsyntax-only",
            "-I" + str(tmp_path),
            "-I" + sysconfig.get_path("include"),
            "-I" + str(Path(torch.__file__).parent / "include"),
            str(ROOT / "torch_fl/native_host/csrc/backends/ascend.cc"),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
