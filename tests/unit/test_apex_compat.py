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

"""Unit tests for the optional Apex multi-tensor compatibility shim."""

from types import ModuleType, SimpleNamespace

import pytest

import torch_fl.compat.apex as compat


class _FakeTensor:
    def __init__(self, device_type, value=None):
        self.device = SimpleNamespace(type=device_type, index=0)
        self.value = value

    def __repr__(self):
        return f"_FakeTensor({self.device.type!r}, {self.value!r})"


@pytest.fixture
def fake_torch(monkeypatch):
    fake = SimpleNamespace(Tensor=_FakeTensor)
    monkeypatch.setattr(compat, "torch", fake)
    return fake


@pytest.fixture
def cuda_views(monkeypatch, fake_torch):
    calls = {"to_cuda": [], "to_flagos": []}

    def to_cuda(tensor):
        calls["to_cuda"].append(tensor)
        return _FakeTensor("cuda", tensor.value)

    def to_flagos(tensor):
        calls["to_flagos"].append(tensor)
        return _FakeTensor("flagos", tensor.value)

    monkeypatch.setattr(compat, "_get_view_functions", lambda: (to_cuda, to_flagos))
    return calls


def _fake_apex_module(call):
    class MultiTensorApply:
        def check_avail(self):
            call["checked"] = True

        def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
            call["original"] = (op, noop_flag_buffer, tensor_lists, args)
            return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)

        chunk_size = 128

    module = ModuleType("apex.multi_tensor_apply.multi_tensor_apply")
    module.MultiTensorApply = MultiTensorApply
    return module, MultiTensorApply


def test_cuda_alias_capability_uses_shared_vendor_table(monkeypatch):
    monkeypatch.setattr(compat, "_active_vendor", lambda: "hygon")
    assert compat.is_apex_compat_available()

    monkeypatch.setattr(compat, "_active_vendor", lambda: "ascend")
    assert not compat.is_apex_compat_available()

    monkeypatch.setattr(compat, "_active_vendor", lambda: "unknown")
    assert not compat.is_apex_compat_available()


def test_disable_switch_prevents_install(monkeypatch):
    monkeypatch.setattr(compat, "_active_vendor", lambda: "nvidia")
    monkeypatch.setenv("FLAGOS_DISABLE_APEX_COMPAT", "1")
    assert not compat.is_apex_compat_available()


def test_recursive_inputs_and_outputs(monkeypatch, fake_torch, cuda_views):
    call = {}
    module, cls = _fake_apex_module(call)
    monkeypatch.setattr(compat, "is_apex_compat_available", lambda: True)
    monkeypatch.setattr(compat.importlib, "import_module", lambda name: module)
    assert compat.patch_apex()

    flagos_a = _FakeTensor("flagos", "a")
    flagos_b = _FakeTensor("privateuseone", "b")
    cpu = _FakeTensor("cpu", "cpu")
    cuda = _FakeTensor("cuda", "cuda")
    noop = _FakeTensor("flagos", "noop")

    def op(chunk_size, received_noop, received_lists, *args):
        call["op"] = (chunk_size, received_noop, received_lists, args)
        return (
            _FakeTensor("cuda", "result"),
            [_FakeTensor("cuda", "nested"), None],
            cpu,
        )

    result = cls()(op, noop, [[flagos_a, (flagos_b, cpu)], [cuda]], None, (flagos_a, 7))

    _, received_noop, received_lists, received_args = call["op"]
    assert received_noop.device.type == "cuda"
    assert received_lists[0][0].device.type == "cuda"
    assert received_lists[0][1][0].device.type == "cuda"
    assert received_lists[0][1][1] is cpu
    assert received_lists[1][0] is cuda
    assert received_args[0] is None
    assert received_args[1][0].device.type == "cuda"

    assert result[0].device.type == "flagos"
    assert result[1][0].device.type == "flagos"
    assert result[1][1] is None
    assert result[2] is cpu
    assert len(cuda_views["to_cuda"]) == 4
    assert len(cuda_views["to_flagos"]) == 2


def test_cpu_invocation_is_passed_through(monkeypatch, fake_torch, cuda_views):
    call = {}
    module, cls = _fake_apex_module(call)
    monkeypatch.setattr(compat, "is_apex_compat_available", lambda: True)
    monkeypatch.setattr(compat.importlib, "import_module", lambda name: module)
    assert compat.patch_apex()

    noop = _FakeTensor("cuda", "noop")
    tensors = [[_FakeTensor("cuda", "input")]]
    sentinel = object()

    def op(*args):
        call["args"] = args
        return sentinel

    result = cls()(op, noop, tensors, 3)
    assert result is sentinel
    assert call["args"][1] is noop
    assert call["args"][2] is tensors
    assert call["args"][3] == 3
    assert cuda_views["to_cuda"] == []
    assert cuda_views["to_flagos"] == []


def test_patch_is_idempotent(monkeypatch, fake_torch):
    call = {}
    module, cls = _fake_apex_module(call)
    monkeypatch.setattr(compat, "is_apex_compat_available", lambda: True)
    monkeypatch.setattr(compat.importlib, "import_module", lambda name: module)

    assert compat.patch_apex()
    first = cls.__call__
    assert compat.patch_apex()
    assert cls.__call__ is first
    assert getattr(first, compat._PATCHED_ATTR)
    assert getattr(first, compat._ORIGINAL_ATTR)


def test_import_hook_patches_apex_after_import(monkeypatch):
    compat._remove_import_hook()
    monkeypatch.setattr(compat, "is_apex_compat_available", lambda: True)
    patched = []
    monkeypatch.setattr(compat, "patch_apex", lambda: patched.append(True) or True)

    original_import = compat.builtins.__import__
    imported = ModuleType("apex")

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        assert name == "apex"
        return imported

    compat._original_import = fake_import
    compat._import_hook = compat._apex_import
    compat.builtins.__import__ = compat._import_hook
    try:
        assert compat.builtins.__import__("apex") is imported
        assert patched == [True]
    finally:
        compat.builtins.__import__ = original_import
        compat._original_import = None
        compat._import_hook = None


def test_disabled_import_hook_is_removed(monkeypatch):
    compat._remove_import_hook()
    monkeypatch.setattr(compat, "is_apex_compat_available", lambda: False)
    assert not compat.install_apex_compat()
    assert compat._import_hook is None
