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
