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

"""Tests for QDQ insertion and calibration.

The property that matters is structural: hbdk4 only puts a conv on the BPU when
its input type is si8/si16, and Q/DQ pairs around the conv are how that is
expressed. These tests check the rewrite is well-formed and numerically close,
without needing the compiler or the device.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")

from torch_fl.accelerator.bpu.calibrate import (  # noqa: E402
    Calibration,
    TensorRange,
    calibrate_module,
)
from torch_fl.accelerator.bpu.qdq import quantize_onnx  # noqa: E402


class TwoConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = torch.nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.c2(torch.relu(self.c1(x))))


def _export(tmp_path, model, x):
    path = tmp_path / "m.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            (x,),
            str(path),
            input_names=["in_0"],
            output_names=["out_0"],
            dynamo=False,
            opset_version=17,
        )
    proto = onnx.shape_inference.infer_shapes(
        onnx.load(str(path)), strict_mode=False, data_prop=True
    )
    onnx.save(proto, str(path))
    return path, proto


def test_inserts_qdq_around_every_conv(tmp_path):
    _, proto = _export(tmp_path, TwoConv().eval(), torch.randn(1, 3, 16, 16))
    n_conv = sum(1 for n in proto.graph.node if n.op_type == "Conv")
    assert n_conv == 2

    out = quantize_onnx(proto)
    ops = [n.op_type for n in out.graph.node]

    # One Q + one DQ per activation edge, plus one DQ per weight.
    assert ops.count("QuantizeLinear") == n_conv
    assert ops.count("DequantizeLinear") == 2 * n_conv
    assert ops.count("Conv") == n_conv
    onnx.checker.check_model(out)


def test_conv_consumes_the_dequantized_edge(tmp_path):
    """A Q/DQ pair only helps if the conv actually reads its output."""
    _, proto = _export(tmp_path, TwoConv().eval(), torch.randn(1, 3, 16, 16))
    out = quantize_onnx(proto)

    produced_by = {o: n for n in out.graph.node for o in n.output}
    convs = [n for n in out.graph.node if n.op_type == "Conv"]
    assert convs
    for conv in convs:
        assert produced_by[conv.input[0]].op_type == "DequantizeLinear"
        assert produced_by[conv.input[1]].op_type == "DequantizeLinear"


def test_weights_become_int8_with_signed_zero_point(tmp_path):
    """hbdk4's frontend rejects unsigned targets, so zero_point must be int8."""
    from onnx import numpy_helper

    _, proto = _export(tmp_path, TwoConv().eval(), torch.randn(1, 3, 16, 16))
    out = quantize_onnx(proto)
    inits = {i.name: numpy_helper.to_array(i) for i in out.graph.initializer}

    q_inputs = [
        n.input
        for n in out.graph.node
        if n.op_type in ("QuantizeLinear", "DequantizeLinear")
    ]
    assert q_inputs
    for args in q_inputs:
        # Scale and zero_point must be constant initializers, not computed.
        assert args[1] in inits
        assert args[2] in inits
        assert inits[args[1]].dtype == np.float32
        assert inits[args[2]].dtype == np.int8

    int8_weights = [v for v in inits.values() if v.dtype == np.int8 and v.ndim == 4]
    assert len(int8_weights) == 2
    for w in int8_weights:
        assert w.min() >= -127 and w.max() <= 127


def test_quantized_graph_stays_numerically_close(tmp_path):
    ort = pytest.importorskip("onnxruntime")

    model = TwoConv().eval()
    x = torch.randn(1, 3, 16, 16)
    path, proto = _export(tmp_path, model, x)

    ref = ort.InferenceSession(
        onnx.load(str(path)).SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"in_0": x.numpy()})[0]

    out = quantize_onnx(proto, default_act_scale=float(x.abs().max()) / 127.0)
    got = ort.InferenceSession(
        out.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"in_0": x.numpy()})[0]

    a, b = ref.ravel(), got.ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert cos > 0.99, f"cosine similarity {cos}"


def test_no_quantizable_ops_leaves_graph_unchanged(tmp_path):
    class OnlyRelu(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x) * 2

    _, proto = _export(tmp_path, OnlyRelu().eval(), torch.randn(4, 4))
    before = [n.op_type for n in proto.graph.node]
    out = quantize_onnx(proto)
    assert [n.op_type for n in out.graph.node] == before


# -- pooling -------------------------------------------------------------------


class ConvPool(torch.nn.Module):
    """A ResNet stem: conv, then the pool that used to run on the CPU."""

    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv2d(3, 8, 7, 2, 3, bias=False)
        self.p = torch.nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        return self.p(torch.relu(self.c(x)))


def test_pool_input_is_quantized(tmp_path):
    """hbdk4 keeps a pool on the CPU while its input is f32.

    `convert(advice=True)` on the un-quantized ResNet said it outright:
    "lower to cpu. P.S. The type of hbir.max_pool's fin is f32, which should be
    si8, si16, f16 on bpu." It was the only op in the whole network not on the
    BPU, because the Q/DQ pair around the preceding conv hands the pool a float.
    """
    _, proto = _export(tmp_path, ConvPool().eval(), torch.randn(1, 3, 32, 32))
    out = quantize_onnx(proto)

    produced_by = {o: n for n in out.graph.node for o in n.output}
    pools = [n for n in out.graph.node if n.op_type == "MaxPool"]
    assert pools
    for pool in pools:
        assert produced_by[pool.input[0]].op_type == "DequantizeLinear"
    onnx.checker.check_model(out)


def test_pool_gets_no_weight_dequantize(tmp_path):
    """A pool has no weights; only its activation edge may be rewritten."""
    _, proto = _export(tmp_path, ConvPool().eval(), torch.randn(1, 3, 32, 32))
    out = quantize_onnx(proto)

    pools = [n for n in out.graph.node if n.op_type == "MaxPool"]
    assert all(len(p.input) == 1 for p in pools)


def test_pooling_stays_numerically_close(tmp_path):
    ort = pytest.importorskip("onnxruntime")

    model = ConvPool().eval()
    x = torch.randn(1, 3, 32, 32)
    path, proto = _export(tmp_path, model, x)

    ref = ort.InferenceSession(
        onnx.load(str(path)).SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"in_0": x.numpy()})[0]

    out = quantize_onnx(proto, default_act_scale=float(x.abs().max()) / 127.0)
    got = ort.InferenceSession(
        out.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"in_0": x.numpy()})[0]

    a, b = ref.ravel(), got.ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert cos > 0.99, f"cosine similarity {cos}"


def test_calibration_scale_covers_observed_range():
    r = TensorRange()
    r.observe(torch.tensor([-2.0, 1.0]))
    r.observe(torch.tensor([0.0, 6.5]))  # exactly representable in float32
    # Scale must map the widest magnitude to at most QMAX.
    assert r.valid
    assert 6.5 / r.scale() <= 127.0
    assert r.scale() == pytest.approx(6.5 / 127.0)


def test_calibration_handles_degenerate_input():
    empty = TensorRange()
    assert not empty.valid
    assert empty.scale() == 1.0

    zeros = TensorRange()
    zeros.observe(torch.zeros(4))
    assert zeros.scale() == 1.0  # never 0, which would divide by zero


def test_calibrate_module_records_leaf_outputs():
    model = TwoConv().eval()
    cal = calibrate_module(model, [torch.randn(1, 3, 16, 16) for _ in range(3)])
    assert isinstance(cal, Calibration)
    # Leaf modules plus the synthetic graph input.
    assert {"c1", "c2", "__input__"} <= set(cal.ranges)
    assert cal.scale_of("__input__") > 0


def test_calibrate_onnx_scales_the_conv_inputs(tmp_path):
    pytest.importorskip("onnxruntime")
    from torch_fl.accelerator.bpu.calibrate import calibrate_onnx

    model = TwoConv().eval()
    path, proto = _export(tmp_path, model, torch.randn(1, 3, 16, 16))

    scales = calibrate_onnx(path, [torch.randn(1, 3, 16, 16) for _ in range(3)])

    # One scale per conv activation edge, keyed by ONNX tensor name.
    convs = [n for n in proto.graph.node if n.op_type == "Conv"]
    assert set(scales) == {c.input[0] for c in convs}
    assert all(s > 0 for s in scales.values())


def test_calibrated_scales_beat_a_bad_default(tmp_path):
    """Calibration exists to avoid clipping; show that it does."""
    ort = pytest.importorskip("onnxruntime")
    from torch_fl.accelerator.bpu.calibrate import calibrate_onnx

    model = TwoConv().eval()
    x = torch.randn(1, 3, 16, 16) * 20.0  # far outside the default's range
    path, _ = _export(tmp_path, model, x)

    def cos_against_float(**kw):
        ref = ort.InferenceSession(
            onnx.load(str(path)).SerializeToString(),
            providers=["CPUExecutionProvider"],
        ).run(None, {"in_0": x.numpy()})[0]
        q = quantize_onnx(onnx.load(str(path)), **kw)
        got = ort.InferenceSession(
            q.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"in_0": x.numpy()})[0]
        a, b = ref.ravel(), got.ravel()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    default_only = cos_against_float(default_act_scale=0.05)
    calibrated = cos_against_float(act_scales=calibrate_onnx(path, [x]))
    assert calibrated > default_only
