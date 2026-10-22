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

"""The pre-partition rewrites must preserve semantics and be ONNX-exportable."""

from __future__ import annotations

import operator
from pathlib import Path

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd

from torch_fl.accelerator.bpu.compiler import export_onnx
from torch_fl.accelerator.bpu.decompose import decompose, decompose_for_onnx

BN_NO_TRAINING = torch.ops.aten._native_batch_norm_legit_no_training.default
MAX_POOL_INDICES = torch.ops.aten.max_pool2d_with_indices.default


class ConvBN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.conv(x)))


def _aten_graph(mod: torch.nn.Module, x: torch.Tensor):
    """Capture the post-AOTAutograd aten graph, as the backend sees it."""
    captured = {}

    def fw(gm, inputs):
        captured["gm"] = gm
        return gm.forward

    with torch.no_grad():
        torch.compile(mod, backend=aot_autograd(fw_compiler=fw))(x)
    return captured["gm"]


def test_rewrites_batch_norm() -> None:
    gm = _aten_graph(ConvBN().eval(), torch.randn(1, 3, 8, 8))
    before = [n for n in gm.graph.nodes if n.target is BN_NO_TRAINING]
    assert before, "expected the functional batch_norm in the aten graph"

    decompose_for_onnx(gm)

    assert not [n for n in gm.graph.nodes if n.target is BN_NO_TRAINING]
    assert [n for n in gm.graph.nodes if n.target is torch.ops.aten.batch_norm.default]
    # The getitem that unpacked the tuple should be gone too.
    assert not [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target is operator.getitem
        and n.args
        and getattr(n.args[0], "target", None) is BN_NO_TRAINING
    ]


def test_rewrite_preserves_numerics() -> None:
    """The rewritten graph must still reproduce eager output exactly."""
    mod = ConvBN().eval()
    x = torch.randn(2, 3, 8, 8)

    captured = {}

    def fw(gm, inputs):
        decompose_for_onnx(gm)
        captured["gm"] = gm
        return gm.forward

    with torch.no_grad():
        ref = mod(x)
        got = torch.compile(mod, backend=aot_autograd(fw_compiler=fw))(x)

    assert "gm" in captured
    assert not [n for n in captured["gm"].graph.nodes if n.target is BN_NO_TRAINING]
    torch.testing.assert_close(got, ref)


def test_exports_to_onnx_after_rewrite(tmp_path: Path) -> None:
    """Without the rewrite this raises UnsupportedOperatorError."""
    pytest.importorskip("onnx")
    mod = ConvBN().eval()
    x = torch.randn(1, 3, 8, 8)
    gm = _aten_graph(mod, x)

    inputs = []
    for n in gm.graph.nodes:
        if n.op != "placeholder":
            continue
        v = n.meta["val"]
        inputs.append(torch.zeros(tuple(v.shape), dtype=v.dtype))

    out = tmp_path / "m.onnx"
    ins, outs = export_onnx(gm, inputs, out)
    assert out.exists() and ins and outs

    import onnx

    proto = onnx.load(str(out))
    onnx.checker.check_model(proto)
    # hbdk4's adaptor needs shapes for intermediates, which export_onnx adds.
    assert len(proto.graph.value_info) > 0


# -- max_pool2d ----------------------------------------------------------------


class Stem(torch.nn.Module):
    """A ResNet stem: this is where the graph used to be cut in half."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 7, 2, 3, bias=False)
        self.bn = torch.nn.BatchNorm2d(8)
        self.pool = torch.nn.MaxPool2d(3, 2, 1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv2(self.pool(torch.relu(self.bn(self.conv(x))))))


def test_rewrites_max_pool_with_indices() -> None:
    gm = _aten_graph(Stem().eval(), torch.randn(1, 3, 32, 32))
    assert [n for n in gm.graph.nodes if n.target is MAX_POOL_INDICES]

    decompose(gm)

    assert not [n for n in gm.graph.nodes if n.target is MAX_POOL_INDICES]
    pools = [n for n in gm.graph.nodes if n.target is torch.ops.aten.max_pool2d.default]
    assert len(pools) == 1
    # The rewritten node must carry a tensor val, not the original tuple: a
    # tuple here is what made the partitioner reject the downstream partition.
    assert isinstance(pools[0].meta.get("val"), torch.Tensor)


def test_max_pool_rewrite_preserves_numerics() -> None:
    mod = Stem().eval()
    x = torch.randn(2, 3, 32, 32)

    captured = {}

    def fw(gm, inputs):
        decompose(gm)
        captured["gm"] = gm
        return gm.forward

    with torch.no_grad():
        ref = mod(x)
        got = torch.compile(mod, backend=aot_autograd(fw_compiler=fw))(x)

    assert not [n for n in captured["gm"].graph.nodes if n.target is MAX_POOL_INDICES]
    torch.testing.assert_close(got, ref)


def test_max_pool_keeps_indices_when_they_are_used() -> None:
    """max_unpool needs the indices; that node must be left alone."""

    class WithIndices(torch.nn.Module):
        def forward(self, x):
            out, idx = torch.nn.functional.max_pool2d(x, 2, return_indices=True)
            return out.sum() + idx.sum()

    gm = _aten_graph(WithIndices().eval(), torch.randn(1, 3, 8, 8))
    before = [n for n in gm.graph.nodes if n.target is MAX_POOL_INDICES]
    assert before

    decompose(gm)

    assert [n for n in gm.graph.nodes if n.target is MAX_POOL_INDICES] == before


def test_pooling_stays_in_one_partition() -> None:
    """The regression this rewrite exists for.

    Untouched, `max_pool2d_with_indices` is unsupported, so a ResNet splits into
    a 4-node stem and an 84-node body -- and the body is then rejected outright,
    because its boundary input is a getitem off a tuple-valued node. The whole
    network ran on the CPU.
    """
    from torch_fl.accelerator.bpu.backend import _example_inputs_for
    from torch_fl.accelerator.bpu.partition import partition_graph

    gm = _aten_graph(Stem().eval(), torch.randn(1, 3, 32, 32))
    assert len(partition_graph(gm, min_nodes=2)) == 2  # split at the pool

    decompose(gm)
    parts = partition_graph(gm, min_nodes=2)

    assert len(parts) == 1
    targets = {n.target for n in parts[0].nodes}
    assert torch.ops.aten.max_pool2d.default in targets
    assert _example_inputs_for(parts[0], [], gm) is not None


def test_decompose_is_idempotent() -> None:
    """It runs before partitioning and again before export."""
    gm = _aten_graph(Stem().eval(), torch.randn(1, 3, 32, 32))
    decompose(gm)
    after_once = [(n.op, n.target) for n in gm.graph.nodes]
    decompose(gm)
    assert [(n.op, n.target) for n in gm.graph.nodes] == after_once


def test_decompose_for_onnx_is_the_same_pass() -> None:
    assert decompose_for_onnx is decompose


# -- attention -----------------------------------------------------------------


SOFTMAX_PRIVATE = torch.ops.aten._softmax.default
UNSAFE_VIEW = torch.ops.aten._unsafe_view.default
ATEN_T = torch.ops.aten.t.default


class Attention(torch.nn.Module):
    """One transformer block, which used to export not at all."""

    def __init__(self, d: int = 32, h: int = 4) -> None:
        super().__init__()
        self.n1 = torch.nn.LayerNorm(d)
        self.q = torch.nn.Linear(d, d)
        self.k = torch.nn.Linear(d, d)
        self.v = torch.nn.Linear(d, d)
        self.proj = torch.nn.Linear(d, d)
        self.h, self.dh = h, d // h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        y = self.n1(x)
        q = self.q(y).view(b, n, self.h, self.dh).transpose(1, 2)
        k = self.k(y).view(b, n, self.h, self.dh).transpose(1, 2)
        v = self.v(y).view(b, n, self.h, self.dh).transpose(1, 2)
        a = ((q @ k.transpose(-2, -1)) * (self.dh**-0.5)).softmax(-1)
        return x + self.proj((a @ v).transpose(1, 2).reshape(b, n, d))


def test_rewrites_private_softmax() -> None:
    """AOTAutograd emits `_softmax`, which has no ONNX symbolic at all."""
    gm = _aten_graph(Attention().eval(), torch.randn(1, 8, 32))
    assert [n for n in gm.graph.nodes if n.target is SOFTMAX_PRIVATE]

    decompose(gm)

    assert not [n for n in gm.graph.nodes if n.target is SOFTMAX_PRIVATE]
    assert [n for n in gm.graph.nodes if n.target is torch.ops.aten.softmax.int]


def test_rewrites_unsafe_view_and_t() -> None:
    gm = _aten_graph(Attention().eval(), torch.randn(1, 8, 32))
    assert [n for n in gm.graph.nodes if n.target in (UNSAFE_VIEW, ATEN_T)]

    decompose(gm)

    assert not [n for n in gm.graph.nodes if n.target in (UNSAFE_VIEW, ATEN_T)]


def test_attention_rewrite_preserves_numerics() -> None:
    mod = Attention().eval()
    x = torch.randn(2, 8, 32)

    captured = {}

    def fw(gm, inputs):
        decompose(gm)
        captured["gm"] = gm
        return gm.forward

    with torch.no_grad():
        ref = mod(x)
        got = torch.compile(mod, backend=aot_autograd(fw_compiler=fw))(x)

    assert not [n for n in captured["gm"].graph.nodes if n.target is SOFTMAX_PRIVATE]
    torch.testing.assert_close(got, ref)


def test_attention_becomes_one_partition() -> None:
    """Nine partitions, none of them exportable, was the old behaviour."""
    from torch_fl.accelerator.bpu.partition import partition_graph

    gm = _aten_graph(Attention().eval(), torch.randn(1, 8, 32))
    assert len(partition_graph(gm, min_nodes=2)) > 1

    decompose(gm)
    parts = partition_graph(gm, min_nodes=2)

    assert len(parts) == 1
    assert torch.ops.aten.softmax.int in {n.target for n in parts[0].nodes}


def test_attention_exports_to_onnx(tmp_path: Path) -> None:
    """The failure this rewrite exists for: UnsupportedOperatorError on _softmax."""
    pytest.importorskip("onnx")
    gm = _aten_graph(Attention().eval(), torch.randn(1, 8, 32))

    inputs = [
        torch.zeros(tuple(n.meta["val"].shape), dtype=n.meta["val"].dtype)
        for n in gm.graph.nodes
        if n.op == "placeholder"
    ]

    out = tmp_path / "attn.onnx"
    ins, outs = export_onnx(gm, inputs, out)
    assert out.exists() and ins and outs

    import onnx

    onnx.checker.check_model(onnx.load(str(out)))


def test_t_is_left_alone_below_two_dims() -> None:
    """`aten.t` is the identity on 0-D and 1-D; rewriting it would be wrong."""

    class OneD(torch.nn.Module):
        def forward(self, x):
            return torch.t(x) + 1

    gm = _aten_graph(OneD().eval(), torch.randn(5))
    ts = [n for n in gm.graph.nodes if n.target is ATEN_T]

    decompose(gm)

    assert [n for n in gm.graph.nodes if n.target is ATEN_T] == ts
