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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

compat = pytest.importorskip("torch_fl.compat.cosmos_attention")


class _TaggedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, device):
        obj = torch.Tensor._make_subclass(cls, torch.empty(0, device="cpu"))
        obj._reported_device = torch._C.device(device)
        return obj

    @property
    def device(self):
        return self._reported_device


def test_attention_bridges_flagos_inputs_and_tuple_outputs(monkeypatch):
    calls = []

    def original(query, key, *, metadata):
        calls.append("attention")
        assert query is key is metadata["same"]
        assert query.device.type == metadata["offsets"].device.type == "cuda"
        return _TaggedTensor("cuda:0"), _TaggedTensor("cuda:0")

    monkeypatch.setattr(torch.flagos, "synchronize", lambda _: calls.append("flagos"))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: calls.append("cuda"))
    monkeypatch.setattr(
        compat._C,
        "_flagos_to_cuda_view",
        lambda _: _TaggedTensor("cuda:0"),
        raising=False,
    )
    monkeypatch.setattr(
        compat._C,
        "_cuda_to_flagos_view",
        lambda _: _TaggedTensor("flagos:0"),
        raising=False,
    )
    module = SimpleNamespace(attention=original)
    compat._patch_frontend(module)

    q = _TaggedTensor("flagos:0")
    with torch.inference_mode():
        values = module.attention(
            q, q, metadata={"same": q, "offsets": _TaggedTensor("flagos:0")}
        )
    assert isinstance(values, tuple)
    assert [value.device.type for value in values] == ["flagos", "flagos"]
    assert calls == ["flagos", "attention", "cuda"]


def test_attention_does_not_intercept_cpu_inputs():
    x = torch.empty(1, device="cpu")
    module = SimpleNamespace(attention=lambda query, **_: query)
    compat._patch_frontend(module)
    patched = module.attention
    compat._patch_frontend(module)
    assert module.attention is patched
    assert patched(query=x) is x
