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

"""Unit coverage for PPU CUDA metadata restored from the loaded runtime."""

import sys
import types

import pytest
from dcu_module_loader import load_module

_LINK_REL = "torch_fl/accelerator/ppu/_ppu_libtorch_link.py"

link = load_module(
    _LINK_REL,
    "_flagos_test_ppu_libtorch_link",
    deps=(
        (
            "torch_fl.accelerator._vendor_libtorch",
            "torch_fl/accelerator/_vendor_libtorch.py",
        ),
    ),
)


def _fake_torch(cuda=None, available=True, compiled=13000):
    calls = []

    def get_compiled_version():
        calls.append("compiled")
        return compiled

    module = types.SimpleNamespace(
        version=types.SimpleNamespace(cuda=cuda),
        cuda=types.SimpleNamespace(is_available=lambda: available),
        _C=types.SimpleNamespace(_cuda_getCompiledVersion=get_compiled_version),
    )
    return module, calls


@pytest.mark.parametrize(
    ("compiled", "expected"),
    [
        (13000, "13.0"),
        (12080, "12.8"),
        (11070, "11.7"),
    ],
)
def test_cuda_version_is_derived_from_loaded_runtime(monkeypatch, compiled, expected):
    torch, calls = _fake_torch(compiled=compiled)
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert link.restore_ppu_cuda_version() == expected
    assert torch.version.cuda == expected
    assert calls == ["compiled"]


def test_existing_vendor_torch_metadata_is_left_unchanged(monkeypatch):
    torch, calls = _fake_torch(cuda="13.0", available=False, compiled=None)
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert link.restore_ppu_cuda_version() == "13.0"
    assert calls == []


def test_missing_ppu_runtime_is_reported(monkeypatch):
    torch, _ = _fake_torch(available=False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(RuntimeError, match=r"torch\.cuda\.is_available\(\) is false"):
        link.restore_ppu_cuda_version()


def test_missing_compiled_version_getter_is_reported(monkeypatch):
    torch, _ = _fake_torch()
    del torch._C._cuda_getCompiledVersion
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(RuntimeError, match="_cuda_getCompiledVersion"):
        link.restore_ppu_cuda_version()


@pytest.mark.parametrize("compiled", [None, "not-a-version", 0, 13001])
def test_invalid_compiled_version_is_rejected(monkeypatch, compiled):
    torch, _ = _fake_torch(compiled=compiled)
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(RuntimeError, match="invalid compiled CUDA version"):
        link.restore_ppu_cuda_version()


def test_runtime_probe_is_idempotent(monkeypatch):
    torch, calls = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert link.restore_ppu_cuda_version() == "13.0"
    assert link.restore_ppu_cuda_version() == "13.0"
    assert calls == ["compiled"]
