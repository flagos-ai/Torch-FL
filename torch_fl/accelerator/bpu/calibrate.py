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

"""Post-training calibration: collect activation ranges and emit a QDQ ONNX.

Why this exists
---------------
The BPU's MAC array is int8/int16. Handing hbdk4 a float graph makes it lower
every conv to `native::Conv2dNHWC` on the CPU — confirmed by `convert(advice=True)`,
which reports verbatim:

    lower to cpu. P.S. The type of hbir.conv's fin is f32,
    which should be si8, si16 on bpu.

So quantization is not an optimization here, it is the precondition for the BPU
participating at all. Measured on a 2-conv net: 6.12 ms (conv on CPU) vs
0.617 ms (conv on BPU), a 10x difference.

hbdk4's ONNX frontend accepts standard `QuantizeLinear`/`DequantizeLinear`
(opset 13/19) and maps them to its own `qnt.quantize`. Three constraints come
from that frontend and are enforced here:

  * scale must be a constant initializer, not a computed value
  * zero_point must be a constant initializer
  * the quantized type must be *signed* — int8, never uint8

torch 2.13 removed `torch.ao.quantization.quantize_pt2e`, so this uses the
still-present observer machinery directly. That is also easier to control: we
only need per-tensor ranges, not a full quantizer backend config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

log = logging.getLogger("torch_fl.bpu")

# int8 symmetric range. Using -127 rather than -128 keeps the range symmetric,
# which avoids a bias the BPU's rescale path would otherwise have to absorb.
QMIN, QMAX = -127, 127


@dataclass
class TensorRange:
    """Running min/max for one tensor position."""

    lo: float = float("inf")
    hi: float = float("-inf")
    n: int = 0

    def observe(self, t: torch.Tensor) -> None:
        if t.numel() == 0:
            return
        self.lo = min(self.lo, float(t.detach().min()))
        self.hi = max(self.hi, float(t.detach().max()))
        self.n += 1

    @property
    def valid(self) -> bool:
        return self.n > 0 and self.lo <= self.hi

    def scale(self) -> float:
        """Symmetric per-tensor scale.

        Symmetric (zero_point=0) is what the BPU's conv path wants; an
        asymmetric zero point would need an extra correction term.
        """
        if not self.valid:
            return 1.0
        m = max(abs(self.lo), abs(self.hi))
        if m == 0.0:
            return 1.0
        return m / QMAX


@dataclass
class Calibration:
    """Activation ranges keyed by tensor name, collected over sample data."""

    ranges: dict[str, TensorRange] = field(default_factory=dict)

    def observe(self, name: str, t: torch.Tensor) -> None:
        self.ranges.setdefault(name, TensorRange()).observe(t)

    def scale_of(self, name: str) -> float | None:
        r = self.ranges.get(name)
        return r.scale() if r is not None and r.valid else None

    def __len__(self) -> int:
        return len(self.ranges)

    def summary(self) -> str:
        lines = [f"{len(self.ranges)} tensor(s) calibrated"]
        for k, r in list(self.ranges.items())[:8]:
            if r.valid:
                lines.append(f"  {k}: [{r.lo:.4f}, {r.hi:.4f}] scale={r.scale():.6f}")
        return "\n".join(lines)


def calibrate_onnx(
    onnx_path,
    samples: list,
    quantizable: frozenset[str] = frozenset({"Conv", "Gemm", "MatMul"}),
    max_batches: int = 32,
) -> dict[str, float]:
    """Return {onnx_tensor_name: scale} for the activations qdq.py will wrap.

    The QDQ pass keys on *ONNX* tensor names, which have no stable relation to
    torch module names, so scales have to be measured on the exported graph
    rather than on the eager module. Only the first input of each quantizable
    op needs a scale; everything else is either a constant weight (scaled from
    its own values) or not quantized at all.

    Uses onnxruntime on the float graph. If onnxruntime is missing this returns
    {} and the caller falls back to a default scale.
    """
    import onnx

    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime unavailable; falling back to a default scale")
        return {}

    proto = onnx.load(str(onnx_path))
    g = proto.graph
    produced = {o for n in g.node for o in n.output}
    initializers = {i.name for i in g.initializer}

    # Activation edges feeding a quantizable op. Graph inputs count; constants
    # do not, since dq_weight() derives their scale directly.
    wanted = [
        n.input[0]
        for n in g.node
        if n.op_type in quantizable and n.input and n.input[0] not in initializers
    ]
    wanted = list(dict.fromkeys(wanted))
    if not wanted:
        return {}

    # Intermediates are not graph outputs, so they have to be promoted before
    # onnxruntime will hand them back.
    existing = {o.name for o in g.output}
    for name in wanted:
        if name in produced and name not in existing:
            g.output.append(onnx.ValueInfoProto(name=name))

    sess = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]

    cal = Calibration()
    for sample in samples[:max_batches]:
        # A sample is either one tensor (single-input partition) or a sequence
        # matching the graph's input order.
        row = [sample] if isinstance(sample, torch.Tensor) else list(sample)
        feed = {
            n: (t.detach().numpy() if isinstance(t, torch.Tensor) else t)
            for n, t in zip(in_names, row)
        }
        outs = sess.run(out_names, feed)
        for name, val in zip(out_names, outs):
            if name in wanted:
                cal.observe(name, torch.from_numpy(val))
        for name, arr in feed.items():
            if name in wanted:
                cal.observe(name, torch.as_tensor(arr))

    scales = {n: s for n in wanted if (s := cal.scale_of(n)) is not None}
    log.info("calibration: %d/%d activation scale(s)", len(scales), len(wanted))
    return scales


def calibrate_module(
    mod: torch.nn.Module, samples: list[torch.Tensor], max_batches: int = 32
) -> Calibration:
    """Run `mod` over `samples`, recording the range of every submodule output.

    Hooks are registered on leaf modules only, so each name corresponds to one
    operation rather than a container.
    """
    cal = Calibration()
    handles = []

    def hook_for(name: str):
        def hook(_m, _inp, out):
            if isinstance(out, torch.Tensor):
                cal.observe(name, out)

        return hook

    for name, sub in mod.named_modules():
        if name and not list(sub.children()):
            handles.append(sub.register_forward_hook(hook_for(name)))

    # The graph input needs a scale too, and no hook fires for it.
    mod.eval()
    try:
        with torch.no_grad():
            for i, x in enumerate(samples[:max_batches]):
                cal.observe("__input__", x)
                mod(x)
    finally:
        for h in handles:
            h.remove()

    log.info("calibration: %s", cal.summary())
    return cal
