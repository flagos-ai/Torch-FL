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

"""Device-independent unit coverage for the DCP overlapping staging adapter."""

from contextlib import contextmanager
import types

import pytest
import torch
import torch.distributed.checkpoint.filesystem as fs
from dcu_module_loader import load_module

_dcp_compat = load_module("torch_fl/compat/dcp.py", "_flagos_test_dcp_compat")


class _FakeStream:
    def __init__(
        self, stream_id, device_index=0, dev_type_str="flagos", dev_type_int=20
    ):
        self.stream_id = stream_id
        self.device_index = device_index
        self.device_type = dev_type_int
        self._dev_type_str = dev_type_str
        self.synced = 0
        self.waited = []

    @property
    def device(self):
        return types.SimpleNamespace(type=self._dev_type_str, index=self.device_index)

    def synchronize(self):
        self.synced += 1

    def wait_stream(self, other):
        self.waited.append(other)

    def __eq__(self, other):
        return (
            isinstance(other, _FakeStream)
            and self.stream_id == other.stream_id
            and self.device_index == other.device_index
            and self._dev_type_str == other._dev_type_str
        )


class _FakeFlagosTensor:
    """Stand-in tensor reporting ``device.type == 'flagos'`` without a GPU."""

    def __init__(self, cpu_tensor: torch.Tensor, calls: list):
        self._cpu = cpu_tensor
        self._calls = calls
        self.device = types.SimpleNamespace(type="flagos", index=0)

    def detach(self):
        return self

    def to(self, device=None, non_blocking=False):
        self._calls.append(("to", device, non_blocking))
        return self._cpu.clone()


@pytest.fixture
def dcp_patch(monkeypatch):
    orig_init = fs._OverlappingCpuLoader.__init__
    orig_refill = fs._OverlappingCpuLoader._refill
    monkeypatch.setattr(_dcp_compat, "_installed", False)

    active = [_FakeStream(100)]
    entered = []

    @contextmanager
    def _stream_scope(s):
        prev = active[0]
        active[0] = s
        entered.append(s.stream_id)
        try:
            yield
        finally:
            active[0] = prev

    fake_flagos = types.SimpleNamespace(
        is_available=lambda: True,
        current_stream=lambda: active[0],
        stream=_stream_scope,
    )
    monkeypatch.setattr(torch, "flagos", fake_flagos, raising=False)
    monkeypatch.setattr(
        torch._C, "_get_privateuse1_backend_name", lambda: "flagos", raising=False
    )
    # Simulate the CUDA-shimmed FlagOS runtime where torch.cuda.is_available()
    # answers True and _get_available_device_type() would pick "cuda" first.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert _dcp_compat.install_dcp_compat() is True
    # Idempotency check.
    patched_init = fs._OverlappingCpuLoader.__init__
    assert _dcp_compat.install_dcp_compat() is True
    assert fs._OverlappingCpuLoader.__init__ is patched_init

    try:
        yield types.SimpleNamespace(active=active, entered=entered, flagos=fake_flagos)
    finally:
        fs._OverlappingCpuLoader.__init__ = orig_init
        fs._OverlappingCpuLoader._refill = orig_refill


def test_default_overlapping_loader_selects_flagos_and_stages_to_cpu(dcp_patch):
    calls = []
    src_a = torch.arange(16, dtype=torch.float32)
    src_b = torch.arange(32, dtype=torch.float32) * 2.0
    items = {
        "a": _FakeFlagosTensor(src_a, calls),
        "b": _FakeFlagosTensor(src_b, calls),
    }

    non_default = _FakeStream(777)
    dcp_patch.active[0] = non_default

    loader = fs._OverlappingCpuLoader(lambda key: items[key], inflight_threshhold=96)
    assert loader.device_type == "flagos"
    assert loader.device_module is dcp_patch.flagos
    assert loader.stream == non_default

    loader.add(src_a.numel() * 4, "a")
    loader.add(src_b.numel() * 4, "b")
    results = dict((key, tensor) for tensor, key in loader.values())

    # Current stream must remain the caller's non-default stream throughout.
    assert dcp_patch.active[0] == non_default
    assert dcp_patch.entered
    assert all(sid == 777 for sid in dcp_patch.entered)
    assert non_default.synced >= 1
    assert calls == [("to", "cpu", True), ("to", "cpu", True)]
    assert results["a"].is_cpu and torch.equal(results["a"], src_a)
    assert results["b"].is_cpu and torch.equal(results["b"], src_b)


def test_explicit_flagos_guard_stream_with_integer_device_type(dcp_patch):
    """``torch.Stream.device_type`` is an int (20), while ``.device.type`` is ``'flagos'``."""
    calls = []
    src = torch.randn(8)
    staging_stream = _FakeStream(999, dev_type_str="flagos", dev_type_int=20)

    loader = fs._OverlappingCpuLoader(
        lambda _: _FakeFlagosTensor(src, calls),
        stream=staging_stream,
    )
    assert loader.device_type == "flagos"
    assert loader.stream is staging_stream
    # Distinct explicit stream waits on current_stream().
    assert staging_stream.waited == [dcp_patch.active[0]]

    loader.add(src.numel() * 4, "w")
    out = list(loader.values())
    assert len(out) == 1
    assert out[0][0].is_cpu and torch.equal(out[0][0], src)
    assert staging_stream.synced == 1


def test_cpu_view_with_oversized_storage_is_cloned(dcp_patch):
    base = torch.arange(100, dtype=torch.float32)
    sliced = base[10:18]
    assert sliced.untyped_storage().size() != sliced.numel() * sliced.itemsize

    loader = fs._OverlappingCpuLoader(lambda _: sliced)
    loader.add(sliced.numel() * 4, "slice")
    ((staged, key),) = list(loader.values())
    assert key == "slice"
    assert staged.is_cpu
    assert staged.untyped_storage().size() == staged.numel() * staged.itemsize
    assert torch.equal(staged, sliced)


def test_falls_back_to_stock_loader_when_flagos_is_unavailable(dcp_patch, monkeypatch):
    monkeypatch.setattr(dcp_patch.flagos, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(fs, "_get_available_device_type", lambda: "cpu")
    monkeypatch.setattr(
        fs,
        "_get_device_module",
        lambda _: types.SimpleNamespace(
            current_stream=lambda: _FakeStream(0, dev_type_str="cpu", dev_type_int=0),
            stream=dcp_patch.flagos.stream,
        ),
    )

    src = torch.ones(4)
    loader = fs._OverlappingCpuLoader(lambda _: src)
    assert loader.device_type == "cpu"
    loader.add(16, "x")
    ((staged, _),) = list(loader.values())
    assert torch.equal(staged, src)
