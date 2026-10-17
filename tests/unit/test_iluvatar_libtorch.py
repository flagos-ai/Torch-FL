# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CoreX metadata must follow the loaded binary, not a hard-coded CUDA tag."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def load_module(monkeypatch):
    shared = ModuleType("torch_fl.accelerator._vendor_libtorch")
    shared.ensure_vendor_libtorch_links = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, shared.__name__, shared)
    path = (
        Path(__file__).resolve().parents[2]
        / "torch_fl/accelerator/_iluvatar_libtorch.py"
    )
    spec = importlib.util.spec_from_file_location("iluvatar_runtime_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_metadata_comes_from_selected_binary(monkeypatch):
    module = load_module(monkeypatch)
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda=None),
        _C=SimpleNamespace(_cuda_getCompiledVersion=lambda: 10020),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert module.restore_iluvatar_cuda_version() == "10.2"
    assert torch.version.cuda == "10.2"


def test_existing_vendor_metadata_is_preserved(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda="10.2"))
    )
    assert module.restore_iluvatar_cuda_version() == "10.2"


def test_cpu_core_without_corex_selection_fails_clearly(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(version=SimpleNamespace(cuda=None), _C=SimpleNamespace()),
    )
    with pytest.raises(RuntimeError, match="not selected before torch"):
        module.restore_iluvatar_cuda_version()
