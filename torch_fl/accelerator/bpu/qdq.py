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

"""Insert QuantizeLinear/DequantizeLinear pairs into a float ONNX graph.

hbdk4 lowers a conv to the BPU only when its input type is si8/si16. Wrapping
each eligible op's inputs in a Q/DQ pair is what communicates that, and the
compiler's own frontend maps those to `qnt.quantize`.

Constraints imposed by hbdk4's frontend (see onnx/opset13.py::QuantizeLinear):
scale and zero_point must be constant initializers, and the target type must be
signed. Violating any of them raises at export time, not compile time.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("torch_fl.bpu")

# Ops worth quantizing: the ones that actually run on the MAC array. Quantizing
# anything else only adds rescale nodes without moving work onto the BPU.
_QUANTIZABLE = {"Conv", "Gemm", "MatMul"}

# Ops that carry no weights and do no arithmetic on values, but that hbdk4 still
# refuses to put on the BPU while their input is f32 ("lower to cpu. P.S. The
# type of hbir.max_pool's fin is f32, which should be si8, si16, f16 on bpu").
#
# They need the *input* quantized and nothing else. Without this the ResNet stem
# max-pool was the one op in the whole network running on the CPU inside the
# artifact, because the Q/DQ pair around the preceding conv dequantizes its
# output back to float before the pool sees it.
_QUANTIZE_INPUT_ONLY = {"MaxPool", "AveragePool", "GlobalAveragePool"}

# Bumped whenever this pass changes which ops it rewrites or how. The .hbm cache
# key mixes it in, because two runs of the same graph through different versions
# of this pass produce different artifacts -- without it, adding MaxPool above
# silently reused the artifact compiled before it, and the measurement showed no
# change at all.
QDQ_VERSION = 2


def _sym_scale(arr: np.ndarray) -> float:
    m = float(np.abs(arr).max()) if arr.size else 0.0
    return (m / 127.0) if m > 0 else 1.0


def quantize_onnx(
    proto,
    act_scales: dict[str, float] | None = None,
    default_act_scale: float = 0.05,
):
    """Return a copy of `proto` with Q/DQ inserted around quantizable ops.

    Args:
        proto: a float ONNX ModelProto (already shape-inferred).
        act_scales: tensor-name -> scale, from calibration. Names not present
            fall back to `default_act_scale`.
        default_act_scale: used for activations with no calibration entry.

    Weights are quantized per-tensor from their own values, which needs no
    calibration data — only activations do.
    """
    import onnx
    from onnx import helper, numpy_helper

    act_scales = act_scales or {}
    g = proto.graph
    inits = {i.name: i for i in g.initializer}
    new_inits = []
    new_nodes = []
    uid = [0]

    def fresh(stem: str) -> str:
        uid[0] += 1
        return f"{stem}_tbpu{uid[0]}"

    def add_const(name: str, arr: np.ndarray) -> str:
        new_inits.append(numpy_helper.from_array(arr, name))
        return name

    def qdq_activation(src: str, scale: float) -> str:
        """Emit Q->DQ on an activation edge, returning the new edge name."""
        s = add_const(fresh(f"{src}_s"), np.array(scale, dtype=np.float32))
        z = add_const(fresh(f"{src}_z"), np.array(0, dtype=np.int8))
        q, dq = fresh(f"{src}_q"), fresh(f"{src}_dq")
        new_nodes.append(helper.make_node("QuantizeLinear", [src, s, z], [q]))
        new_nodes.append(helper.make_node("DequantizeLinear", [q, s, z], [dq]))
        return dq

    def dq_weight(name: str) -> str | None:
        """Replace a float weight initializer with int8 + DequantizeLinear."""
        init = inits.get(name)
        if init is None:
            return None
        w = numpy_helper.to_array(init)
        if w.dtype != np.float32:
            return None
        sc = _sym_scale(w)
        wq = np.clip(np.rint(w / sc), -127, 127).astype(np.int8)
        qn = add_const(fresh(f"{name}_q"), wq)
        s = add_const(fresh(f"{name}_s"), np.array(sc, dtype=np.float32))
        z = add_const(fresh(f"{name}_z"), np.array(0, dtype=np.int8))
        out = fresh(f"{name}_dq")
        new_nodes.append(helper.make_node("DequantizeLinear", [qn, s, z], [out]))
        return out

    # Rewrite quantizable nodes in place, preserving graph order.
    quantized = 0
    for node in g.node:
        if node.op_type in _QUANTIZE_INPUT_ONLY and node.input:
            # No weights to fold; the input type is the whole problem.
            act = node.input[0]
            args = list(node.input)
            args[0] = qdq_activation(act, act_scales.get(act, default_act_scale))
            del node.input[:]
            node.input.extend(args)
            new_nodes.append(node)
            quantized += 1
            continue

        if node.op_type not in _QUANTIZABLE or len(node.input) < 2:
            new_nodes.append(node)
            continue

        args = list(node.input)

        # Input activation: Q/DQ using the calibrated scale.
        act = args[0]
        scale = act_scales.get(act, default_act_scale)
        args[0] = qdq_activation(act, scale)

        # Weight: fold into int8 + DQ. If it isn't a constant initializer
        # (rare, e.g. a computed weight) leave it float.
        wq = dq_weight(args[1])
        if wq is not None:
            args[1] = wq

        del node.input[:]
        node.input.extend(args)
        new_nodes.append(node)
        quantized += 1

    if not quantized:
        log.warning("qdq: no quantizable ops found; graph unchanged")
        return proto

    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(new_inits)

    # Q/DQ changed the type of intermediate edges, so stale value_info would
    # now be wrong. Drop it and re-infer.
    del g.value_info[:]
    proto = onnx.shape_inference.infer_shapes(proto, data_prop=True)
    log.info("qdq: quantized %d op(s)", quantized)
    return proto
