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

"""Splice tests: verify graph rewriting is correct independently of hbdk4.

A stub runtime stands in for the BPU and computes the partition on CPU, so any
numerical difference is a graph-rewriting bug rather than a hardware one.
"""

import torch

from torch_fl.accelerator.bpu.backend import _BPUCall, _example_inputs_for, _splice
from torch_fl.accelerator.bpu.partition import extract_subgraph, partition_graph


class _StubRuntime:
    """Runs the extracted subgraph on CPU, mimicking BPURuntime's interface."""

    def __init__(self, sub):
        self.sub = sub
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        with torch.no_grad():
            out = self.sub(*args)
        return list(out) if isinstance(out, (tuple, list)) else [out]


def _offload_with_stub(gm, example_inputs, min_nodes=2):
    """Replace every partition with a stub-backed call node."""
    parts = partition_graph(gm, min_nodes=min_nodes)
    stubs = []
    for i, p in reversed(list(enumerate(parts))):
        ex = _example_inputs_for(p, example_inputs, gm)
        if ex is None:
            continue
        sub = extract_subgraph(gm, p)
        stub = _StubRuntime(sub)
        stubs.append(stub)
        _splice(gm, p, _BPUCall(stub, len(p.outputs)), f"_bpu_{i}")
    if stubs:
        gm.graph.lint()
        gm.recompile()
    return gm, stubs


def _run(model, *inputs, min_nodes=2):
    """Compile with stub offload, return (result, stub_call_count)."""
    from torch._dynamo.backends.common import aot_autograd

    state = {"stubs": []}

    def compile_aten(aten_gm, aten_inputs):
        gm, stubs = _offload_with_stub(aten_gm, aten_inputs, min_nodes)
        state["stubs"].extend(stubs)
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=aot_autograd(fw_compiler=compile_aten))
    with torch.no_grad():
        out = compiled(*inputs)
    return out, sum(s.calls for s in state["stubs"])


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(16)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class TwoRegions(torch.nn.Module):
    """Two offloadable regions separated by an unsupported op."""

    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = torch.nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.c1(x)) * 2
        x = torch.erfinv(x.clamp(-0.9, 0.9))
        return torch.relu(self.c2(x)) * 3


def test_single_partition_numerics():
    model = ConvNet().eval()
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        expected = model(x)

    got, n_calls = _run(model, x)
    assert n_calls == 1, "partition should have executed exactly once"
    torch.testing.assert_close(got, expected)


def test_two_partitions_numerics():
    model = TwoRegions().eval()
    x = torch.randn(2, 3, 12, 12)
    with torch.no_grad():
        expected = model(x)

    got, n_calls = _run(model, x)
    assert n_calls == 2, f"expected 2 partition calls, got {n_calls}"
    torch.testing.assert_close(got, expected)


def test_multi_output_partition():
    """A partition whose results feed two different consumers."""

    class Branch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Conv2d(3, 4, 3, padding=1)

        def forward(self, x):
            h = torch.relu(self.c(x))
            a = h * 2
            b = torch.erfinv(h.clamp(-0.9, 0.9))
            return a.sum() + b.sum()

    model = Branch().eval()
    x = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        expected = model(x)
    got, _ = _run(model, x)
    torch.testing.assert_close(got, expected)


def test_graph_is_valid_after_splice():
    """The rewritten graph must contain a call_module and no orphan nodes."""
    from torch._dynamo.backends.common import aot_autograd

    seen = {}

    def compile_aten(aten_gm, aten_inputs):
        gm, stubs = _offload_with_stub(aten_gm, aten_inputs, 2)
        seen["gm"] = gm
        seen["n"] = len(stubs)
        return gm.forward

    torch._dynamo.reset()
    with torch.no_grad():
        torch.compile(ConvNet().eval(), backend=aot_autograd(fw_compiler=compile_aten))(
            torch.randn(1, 3, 16, 16)
        )

    gm = seen["gm"]
    assert seen["n"] == 1
    call_modules = [n for n in gm.graph.nodes if n.op == "call_module"]
    assert len(call_modules) == 1
    # The offloaded aten ops must be gone from the outer graph.
    assert not any(
        "convolution" in str(n.target)
        for n in gm.graph.nodes
        if n.op == "call_function"
    )
    gm.graph.lint()


def test_repeated_calls_reuse_partition():
    """Second invocation must hit the compiled graph, not recompile."""
    model = ConvNet().eval()
    x = torch.randn(1, 3, 16, 16)

    from torch._dynamo.backends.common import aot_autograd

    stubs = []

    def compile_aten(aten_gm, aten_inputs):
        gm, s = _offload_with_stub(aten_gm, aten_inputs, 2)
        stubs.extend(s)
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=aot_autograd(fw_compiler=compile_aten))
    with torch.no_grad():
        a = compiled(x)
        b = compiled(x)

    assert len(stubs) == 1, "backend should compile once"
    assert stubs[0].calls == 2, "both runs should reuse the same partition"
    torch.testing.assert_close(a, b)
