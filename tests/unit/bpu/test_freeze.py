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

"""Tests for baking parameters and buffers into the compiled artifact.

AOTAutograd lifts every parameter and buffer to a graph input, so a partition's
boundary carries all of them unless they are frozen. That costs a host copy per
tensor per call and, worse, makes the ONNX exporter emit weights as graph
inputs, which stops the QDQ pass from folding them to int8. These tests pin the
structural properties; none of them needs hbdk4 or the device.
"""

from __future__ import annotations

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd

import torch_fl.accelerator.bpu.backend as backend_mod
from torch_fl.accelerator.bpu.partition import (
    extract_subgraph,
    partition_graph,
    runtime_inputs,
)


class ConvBN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.b1 = torch.nn.BatchNorm2d(8)
        self.c2 = torch.nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.c2(torch.relu(self.b1(self.c1(x)))))


def _capture(model, x, min_nodes=2):
    """Run the real backend plumbing and hand back the aten-level artifacts."""
    captured = {}

    def outer(gm, inputs):
        names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
        token = backend_mod._OUTER_INPUTS.set((names, list(inputs)))

        def fw(aten_gm, aten_inputs):
            captured["gm"] = aten_gm
            captured["frozen"] = backend_mod._frozen_weights(aten_gm, aten_inputs)
            captured["parts"] = partition_graph(aten_gm, min_nodes=min_nodes)
            return aten_gm.forward

        try:
            return aot_autograd(fw_compiler=fw)(gm, inputs)
        finally:
            backend_mod._OUTER_INPUTS.reset(token)

    with torch.no_grad():
        torch.compile(model, backend=outer)(x)
    return captured


def test_identifies_parameters_and_buffers():
    cap = _capture(ConvBN().eval(), torch.randn(1, 3, 16, 16))
    # 2 convs (w+b) + BN (w, b, running_mean, running_var) = 8 tensors.
    assert len(cap["frozen"]) == 8
    for t in cap["frozen"].values():
        assert isinstance(t, torch.Tensor)
        assert not t.is_meta


def test_freezing_shrinks_the_boundary_to_activations():
    cap = _capture(ConvBN().eval(), torch.randn(1, 3, 16, 16))
    p = cap["parts"][0]
    ri = runtime_inputs(p, cap["frozen"])
    assert len(p.inputs) > len(ri)
    assert len(ri) == 1  # just the activation


def test_subgraph_signature_matches_runtime_inputs():
    """A mismatch here would pass the wrong tensors to the compiled artifact."""
    cap = _capture(ConvBN().eval(), torch.randn(1, 3, 16, 16))
    p, frozen = cap["parts"][0], cap["frozen"]
    sub = extract_subgraph(cap["gm"], p, frozen)

    phs = [n.name for n in sub.graph.nodes if n.op == "placeholder"]
    assert phs == [n.name for n in runtime_inputs(p, frozen)]
    assert sum(1 for n in sub.graph.nodes if n.op == "get_attr") == len(frozen)


def test_extract_subgraph_leaves_the_parent_untouched():
    """Constants are staged on the parent module; they must not linger."""
    cap = _capture(ConvBN().eval(), torch.randn(1, 3, 16, 16))
    gm, p, frozen = cap["gm"], cap["parts"][0], cap["frozen"]
    before = set(dict(gm.named_buffers())) | set(gm.__dict__)
    extract_subgraph(gm, p, frozen)
    leaked = [
        k
        for k in (set(dict(gm.named_buffers())) | set(gm.__dict__)) - before
        if k.startswith("_frozen_")
    ]
    assert not leaked


def test_frozen_subgraph_is_numerically_equivalent():
    """Freezing must not change what the subgraph computes."""
    cap = _capture(ConvBN().eval(), torch.randn(1, 3, 16, 16))
    gm, p, frozen = cap["gm"], cap["parts"][0], cap["frozen"]

    plain = extract_subgraph(gm, p)
    baked = extract_subgraph(gm, p, frozen)

    with torch.no_grad():
        args = [
            frozen[n.name]
            if n.name in frozen
            else torch.randn(tuple(n.meta["val"].shape), dtype=n.meta["val"].dtype)
            for n in p.inputs
        ]
        want = plain(*args)
        got = baked(*[a for n, a in zip(p.inputs, args) if n.name not in frozen])

    for w, g in zip(
        want if isinstance(want, (tuple, list)) else (want,),
        got if isinstance(got, (tuple, list)) else (got,),
    ):
        torch.testing.assert_close(g, w)


def test_frozen_weights_export_as_onnx_initializers():
    """The point of freezing: weights must be initializers so QDQ can fold them."""
    onnx = pytest.importorskip("onnx")
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from torch_fl.accelerator.bpu.compiler import export_onnx

    cap = _capture(ConvBN().eval(), torch.randn(1, 3, 16, 16))
    gm, p, frozen = cap["gm"], cap["parts"][0], cap["frozen"]
    sub = extract_subgraph(gm, p, frozen)
    ri = runtime_inputs(p, frozen)
    ex = [
        torch.zeros(tuple(n.meta["val"].shape), dtype=n.meta["val"].dtype) for n in ri
    ]

    with TemporaryDirectory() as td:
        path = Path(td) / "s.onnx"
        export_onnx(sub, ex, path)
        model = onnx.load(str(path))

    assert len(model.graph.input) == len(ri)
    int8 = [i for i in model.graph.initializer if i.data_type == onnx.TensorProto.INT8]
    # Two convs: an int8 weight plus a zero_point each, at minimum.
    assert len(int8) >= 2


def test_missing_outer_context_degrades_to_no_freezing():
    """Without the outer inputs we must return {}, not raise."""
    model = ConvBN().eval()
    captured = {}

    def fw(aten_gm, aten_inputs):
        captured["frozen"] = backend_mod._frozen_weights(aten_gm, aten_inputs)
        return aten_gm.forward

    with torch.no_grad():
        # aot_autograd used directly, so _OUTER_INPUTS is never set.
        torch.compile(model, backend=aot_autograd(fw_compiler=fw))(
            torch.randn(1, 3, 16, 16)
        )

    assert captured["frozen"] == {}


def test_identifies_weights_when_dynamo_keeps_them_on_the_module():
    """Freezing mode hands weights over as module attributes, not inputs.

    `is_parameter_freezing()` -- inductor's config.freezing with grad off -- is
    what makes Dynamo guard on parameter identity, which the BPU backend needs
    because it bakes weights into the artifact. In that mode the weights never
    reach `_OUTER_INPUTS`, and an index-only lookup finds nothing: every weight
    then crosses the boundary on each call, which is far slower than eager.
    """
    prior = torch._inductor.config.freezing
    torch._dynamo.reset()
    torch._inductor.config.freezing = True
    try:
        cap = _capture(ConvBN().eval(), torch.randn(1, 3, 16, 16))
    finally:
        torch._inductor.config.freezing = prior
        torch._dynamo.reset()

    assert len(cap["frozen"]) == 8
    for t in cap["frozen"].values():
        assert isinstance(t, torch.Tensor)
        assert not t.is_meta


def test_frozen_weights_line_up_with_their_placeholders():
    """A shape mismatch here means the artifact is built with shuffled weights.

    The freezing-mode path pairs placeholders with tensors positionally, so this
    is the property that catches an off-by-one or a reversed order.
    """
    prior = torch._inductor.config.freezing
    torch._dynamo.reset()
    torch._inductor.config.freezing = True
    try:
        cap = _capture(ConvBN().eval(), torch.randn(1, 3, 16, 16))
    finally:
        torch._inductor.config.freezing = prior
        torch._dynamo.reset()

    gm, frozen = cap["gm"], cap["frozen"]
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.name in frozen:
            assert tuple(node.meta["val"].shape) == tuple(frozen[node.name].shape)
            assert node.meta["val"].dtype == frozen[node.name].dtype
