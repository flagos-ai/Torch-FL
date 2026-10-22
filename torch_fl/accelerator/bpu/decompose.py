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

"""Rewrite aten ops that the ONNX exporter or the partitioner cannot handle.

AOTAutograd emits functional aten variants that the TorchScript-based ONNX
exporter has no symbolic function for, and multi-output variants whose extra
results are training-only. Each rewrite here preserves semantics exactly and
only changes which overload the graph names.

The pass runs twice: once on the whole aten graph before partitioning, because
a multi-output op splits a partition in two and its tuple cannot cross a
boundary, and once on each extracted subgraph before export. It is idempotent.
"""

from __future__ import annotations

import logging
import operator

import torch
from torch.fx import GraphModule

log = logging.getLogger("torch_fl.bpu")

_BN_NO_TRAINING = torch.ops.aten._native_batch_norm_legit_no_training.default
_MAX_POOL_INDICES = torch.ops.aten.max_pool2d_with_indices.default
_SOFTMAX = torch.ops.aten._softmax.default
_UNSAFE_VIEW = torch.ops.aten._unsafe_view.default
_T = torch.ops.aten.t.default


def _only_uses_output(node, index: int) -> bool:
    """Whether every user of `node` is `getitem(node, index)`."""
    users = list(node.users)
    if not users:
        return False
    return all(
        u.op == "call_function"
        and u.target is operator.getitem
        and len(u.args) > 1
        and u.args[1] == index
        for u in users
    )


def _replace_tuple_op(gm: GraphModule, node, target, args) -> None:
    """Swap a tuple-returning node for a single-output equivalent."""
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(target, args=args)
    new.meta.update(node.meta)
    val = node.meta.get("val")
    if isinstance(val, (tuple, list)) and val:
        new.meta["val"] = val[0]

    for u in list(node.users):
        u.replace_all_uses_with(new)
        gm.graph.erase_node(u)
    gm.graph.erase_node(node)


def _rewrite_batch_norm(gm: GraphModule) -> int:
    """`_native_batch_norm_legit_no_training` -> `aten.batch_norm`.

    The functional op returns (out, save_mean, save_invstd); in inference only
    the first element is ever consumed, and `aten.batch_norm` with
    training=False computes exactly that and does have an ONNX symbolic.
    """
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not _BN_NO_TRAINING:
            continue

        # Only the primary output may be used; save_mean/save_invstd are
        # training-only statistics and have no ONNX equivalent.
        if not _only_uses_output(node, 0):
            log.debug("batch_norm %s: non-trivial users, left alone", node.name)
            continue

        inp, weight, bias, running_mean, running_var, momentum, eps = node.args
        _replace_tuple_op(
            gm,
            node,
            torch.ops.aten.batch_norm.default,
            (inp, weight, bias, running_mean, running_var, False, momentum, eps, True),
        )
        changed += 1

    return changed


def _rewrite_max_pool(gm: GraphModule) -> int:
    """`max_pool2d_with_indices` -> `aten.max_pool2d`.

    This is what Dynamo emits for `nn.MaxPool2d`, and it is the op that used to
    cut ResNet in half. Two things go wrong with the tuple form. The partitioner
    cannot include it (the BPU produces no indices tensor), so the graph splits
    at the stem pool; and the pooled result then crosses the next partition's
    boundary as a `getitem`, whose producer's `meta['val']` is a *tuple*, which
    made `_example_inputs_for` reject that partition entirely -- reporting
    "dynamic shapes" for a graph that has none.

    The indices output only exists for max_unpool and the backward pass, so in
    an inference graph it is dead. When it is genuinely used the node is left
    alone and the old behaviour stands.
    """
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not _MAX_POOL_INDICES:
            continue
        if not _only_uses_output(node, 0):
            log.debug("max_pool2d %s: indices are used, left alone", node.name)
            continue
        _replace_tuple_op(gm, node, torch.ops.aten.max_pool2d.default, node.args)
        changed += 1
    return changed


def _replace_1to1(gm: GraphModule, node, target, args) -> None:
    """Swap a node for a single-output equivalent with the same value."""
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(target, args=args)
    new.meta.update(node.meta)
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)


def _rewrite_softmax(gm: GraphModule) -> int:
    """`aten._softmax` -> `aten.softmax.int`.

    The TorchScript ONNX exporter has a symbolic for `softmax` but none for the
    private `_softmax`, which is what AOTAutograd actually emits -- so any
    attention block failed to export with "Exporting the operator
    'aten::_softmax' to ONNX opset version 17 is not supported".

    The two differ only in the third argument: `_softmax(self, dim,
    half_to_float)` upcasts a half input to float when the flag is set, which is
    a training-time autocast concern. `softmax.int(self, dim, dtype)` expresses
    the same thing as an explicit output dtype, so the flag maps to
    dtype=float32 and False maps to dtype=None.
    """
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not _SOFTMAX:
            continue
        self_, dim, half_to_float = node.args
        dtype = torch.float32 if half_to_float else None
        _replace_1to1(gm, node, torch.ops.aten.softmax.int, (self_, dim, dtype))
        changed += 1
    return changed


def _rewrite_view_and_transpose(gm: GraphModule) -> int:
    """Rewrite the shape ops AOTAutograd emits that the partitioner rejects.

    `_unsafe_view` is `view` without the alias bookkeeping, emitted after a matmul;
    `t` is `transpose(0, 1)` restricted to 2-D. Both are pure reshapes with
    identical semantics to ops the BPU supports, and leaving them unsupported
    shattered a transformer block into nine partitions -- each one too small to
    pay for its own boundary copies.
    """
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target is _UNSAFE_VIEW:
            _replace_1to1(gm, node, torch.ops.aten.view.default, node.args)
            changed += 1
        elif node.target is _T:
            val = node.meta.get("val")
            # t() is the identity below 2-D; only the 2-D case is a transpose.
            if isinstance(val, torch.Tensor) and val.dim() == 2:
                _replace_1to1(
                    gm, node, torch.ops.aten.transpose.int, (node.args[0], 0, 1)
                )
                changed += 1
    return changed


def decompose(gm: GraphModule) -> GraphModule:
    """Apply every rewrite. Mutates in place and is safe to call twice."""
    n = (
        _rewrite_batch_norm(gm)
        + _rewrite_max_pool(gm)
        + _rewrite_softmax(gm)
        + _rewrite_view_and_transpose(gm)
    )
    if n:
        log.debug("decompose: rewrote %d node(s)", n)
        gm.graph.lint()
        gm.recompile()
    return gm


# The pass used to run only just before torch.onnx.export, which is why it was
# named for it. It now also runs before partitioning; kept as an alias because
# it is the spelling the tests and any out-of-tree caller use.
decompose_for_onnx = decompose
