# Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0
"""The optional vendor provider must not leak into boxing-OFF binaries."""

import ast
from pathlib import Path
import shutil
import subprocess
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_dcu_header_and_translation_unit_compile_without_vendor_headers(tmp_path):
    cxx = shutil.which("c++")
    if cxx is None:
        pytest.skip("requires a C++ compiler")
    # No torch, DTK or CUDA include directories: OFF must not consume any of
    # them through the newly added provider header/TU.
    for definitions in ([], ["-DUSE_DCU=1"]):
        subprocess.run(
            [
                cxx,
                "-std=c++17",
                *definitions,
                "-c",
                str(ROOT / "csrc/runtime/dcu_stream.cc"),
                "-o",
                str(tmp_path / "provider.o"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_flagos_provider_publication_requires_binary_capability(enabled, monkeypatch):
    import sys

    tree = ast.parse((ROOT / "torch_fl/flagos/__init__.py").read_text(encoding="utf-8"))
    statement = next(
        n
        for n in tree.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Call)
        and any(
            isinstance(a, ast.Constant) and a.value == "_dcu_stream_info"
            for a in n.test.args
        )
    )
    provider = types.ModuleType("torch_fl.accelerator.dcu.dcu_stream")
    for name in (
        "DcuEvent",
        "DcuExternalStream",
        "DcuStream",
        "current_stream",
        "default_stream",
        "set_stream",
        "stream_context",
    ):
        setattr(provider, name, object())
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    old_stream = object()
    c = types.SimpleNamespace()
    if enabled:
        c._dcu_stream_info = lambda: None
    ns = {"_C": c, "Stream": old_stream, "__all__": []}
    exec(compile(ast.Module(body=[statement], type_ignores=[]), "flagos", "exec"), ns)
    assert ns["Stream"] is (provider.DcuStream if enabled else old_stream)
    assert ("ExternalStream" in ns["__all__"]) == enabled


def test_cmake_feature_gate_is_dcu_and_boxing():
    text = (ROOT / "csrc/CMakeLists.txt").read_text(encoding="utf-8")
    gate = text.split(
        'if(FLAGOS_ACCELERATOR STREQUAL "dcu" AND FLAGOS_BUILD_BOXING)', 1
    )[1]
    on, off = gate.split("set_target_properties", 1)[0].split("else()", 1)
    assert "PUBLIC FLAGOS_DCU_VENDOR_STREAMS=1" in on
    assert "HEADER_FILE_ONLY TRUE" in off
    assert "FLAGOS_DCU_VENDOR_STREAMS" not in off


def test_dcu_stream_includes_are_standalone():
    """The provider declarations cannot depend on a translation unit's include order."""
    for source, dependent_headers in (
        ("csrc/aten/copy_ops.cc", ("contiguous_ops.h",)),
        ("csrc/aten/contiguous_ops.cc", ("contiguous_ops.h",)),
    ):
        lines = (ROOT / source).read_text(encoding="utf-8").splitlines()
        include = lines.index('#include "stream_copy.h"')
        for header in dependent_headers:
            assert include < lines.index(f'#include "{header}"'), source

    text = (ROOT / "csrc/aten/stream_copy.h").read_text(encoding="utf-8")
    dcu_include = text.index('#include "runtime/dcu_stream.h"')
    dcu_namespace = text.index("namespace at::native::flagos")
    assert dcu_include < dcu_namespace


def test_dcu_allocator_and_record_stream_keep_the_provider_identity():
    allocator = (ROOT / "csrc/runtime/allocator/caching_device_allocator.cc").read_text(
        encoding="utf-8"
    )
    allocation = allocator.split("auto dcu_ptr = c10::flagos::dcu::Allocate", 1)[1]
    allocation = allocation.split("#endif", 1)[0]
    assert "unsafe_set_device" not in allocation

    recording = allocator.split("void CachingDeviceAllocator::record_stream", 1)[1]
    recording = recording.split("backend_->caching_record_stream", 1)[0]
    assert "c10::flagos::dcu::RecordDataPtr" in recording

    provider = (ROOT / "csrc/runtime/dcu_stream.cc").read_text(encoding="utf-8")
    allocation = provider.split("c10::DataPtr Allocate(size_t size)", 1)[1]
    allocation = allocation.split("c10::DeleterFnPtr RawDeleter", 1)[0]
    assert "c10::DeviceType::PrivateUse1" in allocation
    assert "ptr.get()" in allocation
