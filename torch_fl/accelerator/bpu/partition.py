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

"""FX graph partitioning: split a graph into BPU-compilable subgraphs and CPU remainder.

The BPU only executes whole compiled graphs, so the unit of offload is a maximal
connected region of supported nodes rather than an individual operator.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.fx import Graph, GraphModule, Node

# Operators hbdk4 can compile. Conservative on purpose: an op that is wrongly
# listed here fails at compile time (recoverable, we fall back), whereas a
# missing op only costs us a smaller subgraph.
_SUPPORTED: set[Callable | str] = {
    # convolution / linear
    torch.ops.aten.convolution.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.linear.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    # normalization
    torch.ops.aten._native_batch_norm_legit_no_training.default,
    torch.ops.aten.batch_norm.default,
    torch.ops.aten.native_layer_norm.default,
    # activation
    torch.ops.aten.relu.default,
    torch.ops.aten.relu_.default,
    torch.ops.aten.hardtanh.default,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.gelu.default,
    # pooling
    torch.ops.aten.max_pool2d.default,
    # max_pool2d_with_indices is deliberately excluded: it returns an indices
    # tensor the BPU does not produce, and the decomposition is what Dynamo
    # actually emits. Use aten.max_pool2d via a graph pass if you need pooling
    # inside a partition.
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.mean.dim,
    torch.ops.aten.adaptive_avg_pool2d.default,
    torch.ops.aten._adaptive_avg_pool2d.default,
    # elementwise
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.add_.Tensor,
    torch.ops.aten.mul_.Tensor,
    # RMSNorm, which every recent LLM uses in place of LayerNorm. hbdk4 lowers
    # the lot to b30vpu (`pow` -> Pow, `rsqrt` -> Sqrt+Div in ONNX): checked
    # with convert(advice=True), 6 ops, zero CPU fallbacks. Without these two a
    # transformer block splits at every norm -- 86% coverage and two partitions
    # where one will do.
    torch.ops.aten.pow.Tensor_Scalar,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.neg.default,
    # shape
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.cat.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.unsqueeze.default,
    # `slice` is how attention takes its causal window and how any model that
    # indexes a cache reads it. Without it a transformer fragments at every
    # slice; with it decode and prefill are each a single partition.
    torch.ops.aten.slice.Tensor,
    # `expand` is what AOTAutograd inserts ahead of a batched matmul to
    # broadcast the operands; ONNX Expand covers it exactly. `_unsafe_view`
    # and `t` are rewritten to `view`/`transpose` by decompose.py rather than
    # listed here, because the exporter has no symbolic for either.
    torch.ops.aten.expand.default,
    # misc
    torch.ops.aten.softmax.int,
    # `_softmax` stays out: it has no ONNX symbolic. decompose.py rewrites it
    # to softmax.int before this runs.
    torch.ops.aten.clone.default,
}

# Structural nodes that never block a partition: they carry no computation.
_STRUCTURAL = {"placeholder", "output", "get_attr"}


@dataclass
class Partition:
    """A maximal connected region of BPU-supported nodes."""

    nodes: list[Node] = field(default_factory=list)
    inputs: list[Node] = field(default_factory=list)
    outputs: list[Node] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.nodes)


def is_supported(node: Node) -> bool:
    """Whether a single node can run on the BPU.

    Dynamic shapes are rejected: hbdk4 bakes memspace offsets and SRAM tiling
    into the compiled artifact, so every shape must be static at compile time.
    """
    if node.op in _STRUCTURAL:
        return False
    if node.op == "call_method":
        return False
    if node.op == "call_module":
        # Dynamo traces down to aten ops; a surviving call_module is opaque.
        return False
    if node.op != "call_function":
        return False
    if node.target in (operator.getitem,):
        # Tuple indexing is bookkeeping, absorbed into whichever side needs it.
        return True
    if node.target not in _SUPPORTED:
        return False

    val = node.meta.get("val")
    if val is None:
        return False
    vals = val if isinstance(val, (tuple, list)) else [val]
    for v in vals:
        if not isinstance(v, torch.Tensor):
            continue
        if any(not isinstance(d, int) for d in v.shape):
            return False  # symbolic dim
        if v.dtype not in (torch.float32, torch.float16, torch.int8, torch.int32):
            return False
    return True


def partition_graph(gm: GraphModule, min_nodes: int = 3) -> list[Partition]:
    """Find maximal connected regions of supported nodes, in topological order.

    A region is grown greedily in graph order: a supported node joins the
    current region if all of its supported predecessors are already in it,
    which keeps each region a contiguous slice of the topological order and so
    directly emittable as a standalone graph.

    Regions smaller than `min_nodes` are dropped — the host/device copy at the
    boundary costs more than the offload saves.
    """
    supported = {n: is_supported(n) for n in gm.graph.nodes}

    # getitem alone is not worth a partition; it only rides along.
    partitions: list[Partition] = []
    current: list[Node] = []
    current_set: set[Node] = set()

    def flush() -> None:
        nonlocal current, current_set
        if current:
            real = [n for n in current if n.target is not operator.getitem]
            if real:
                partitions.append(Partition(nodes=list(current)))
            current = []
            current_set = set()

    for node in gm.graph.nodes:
        if not supported.get(node, False):
            flush()
            continue
        # Join only if every supported producer is already inside this region;
        # otherwise this node belongs to a later region.
        deps_ok = all(
            arg in current_set or not supported.get(arg, False)
            for arg in node.all_input_nodes
        )
        if not deps_ok:
            flush()
        current.append(node)
        current_set.add(node)
    flush()

    for p in partitions:
        _compute_boundary(p)

    return [p for p in partitions if len(p) >= min_nodes]


def _compute_boundary(p: Partition) -> None:
    """Fill in the tensors crossing into and out of a partition."""
    inside = set(p.nodes)

    seen_in: set[Node] = set()
    for node in p.nodes:
        for arg in node.all_input_nodes:
            if arg not in inside and arg not in seen_in:
                seen_in.add(arg)
                p.inputs.append(arg)

    seen_out: set[Node] = set()
    for node in p.nodes:
        # A node is an output if anything outside the partition consumes it.
        if any(user not in inside for user in node.users):
            if node not in seen_out:
                seen_out.add(node)
                p.outputs.append(node)
    p.outputs = [n for n in p.outputs if n.target is not operator.getitem] or p.outputs


def runtime_inputs(
    p: Partition, frozen: dict[str, torch.Tensor] | None = None
) -> list[Node]:
    """The partition inputs that must be passed at call time.

    Weights baked into the compiled artifact are excluded — see
    `extract_subgraph`.
    """
    if not frozen:
        return list(p.inputs)
    return [n for n in p.inputs if n.name not in frozen]


def extract_subgraph(
    gm: GraphModule, p: Partition, frozen: dict[str, torch.Tensor] | None = None
) -> GraphModule:
    """Build a standalone GraphModule for one partition.

    The result takes the partition's boundary inputs as placeholders and
    returns its boundary outputs, so it can be exported to ONNX on its own.

    Boundary inputs named in `frozen` become module attributes instead of
    placeholders. AOTAutograd lifts every parameter and buffer to a graph
    input, so without this a 2-conv block crosses the boundary with 13 tensors
    instead of 1: each one is copied to the device on every call, and the ONNX
    exporter sees weights as graph inputs rather than initializers, which stops
    the QDQ pass from folding them to int8. Both effects are large enough to
    make the offloaded graph slower than eager.
    """
    frozen = frozen or {}
    new_graph = Graph()
    env: dict[Node, Node] = {}
    consts: dict[str, torch.Tensor] = {}

    # Placeholders first, so the signature matches runtime_inputs() order.
    for inp in p.inputs:
        if inp.name in frozen:
            continue
        ph = new_graph.placeholder(inp.name)
        ph.meta = dict(inp.meta)
        env[inp] = ph

    for inp in p.inputs:
        if inp.name not in frozen:
            continue
        attr = f"_frozen_{inp.name}"
        consts[attr] = frozen[inp.name]
        node = new_graph.get_attr(attr)
        node.meta = dict(inp.meta)
        env[inp] = node

    for node in p.nodes:
        env[node] = new_graph.node_copy(node, lambda n: env[n])

    new_graph.output(tuple(env[o] for o in p.outputs))
    new_graph.lint()

    # GraphModule resolves every get_attr target against the root at
    # construction time, so the constants have to be on `gm` before the call,
    # not registered on the result afterwards. They are removed again below to
    # leave the caller's module as it was.
    # `detach` runs under the backend's active FakeTensorMode, which rejects
    # real tensors, so it has to be suspended (as with the ONNX export).
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    with unset_fake_temporarily():
        for attr, t in consts.items():
            setattr(gm, attr, t.detach())
        try:
            sub = GraphModule(gm, new_graph)
        finally:
            for attr in consts:
                if hasattr(gm, attr):
                    delattr(gm, attr)

    sub.graph.eliminate_dead_code()
    sub.recompile()
    return sub


def summarize(gm: GraphModule, partitions: list[Partition]) -> str:
    """Human-readable partition report, for logs and debugging."""
    total = sum(1 for n in gm.graph.nodes if n.op == "call_function")
    offloaded = sum(1 for p in partitions for n in p.nodes if n.op == "call_function")
    lines = [
        f"{len(partitions)} BPU partition(s), "
        f"{offloaded}/{total} compute nodes offloaded"
    ]
    for i, p in enumerate(partitions):
        ops = [
            str(n.target).split(".")[-2] if "." in str(n.target) else str(n.target)
            for n in p.nodes
            if n.op == "call_function"
        ]
        lines.append(
            f"  [{i}] {len(p)} nodes, "
            f"{len(p.inputs)} in / {len(p.outputs)} out: {' '.join(ops[:8])}"
            + (" ..." if len(ops) > 8 else "")
        )
    return "\n".join(lines)
