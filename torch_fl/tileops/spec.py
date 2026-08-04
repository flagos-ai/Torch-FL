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

"""Hand-maintained input to ``scripts/codegen_tileops.py``.

Everything the TileOPs manifest already records (signatures, dtypes, workload
shapes) is read straight from ``tileops/manifest/*.yaml`` by the generator. This
module holds only what the manifest cannot tell us: how ``ref_api`` maps onto an
aten name, which ctor shapes collapse into a shared recipe, and which ops we
refuse to route (each with the measurement that justifies it).

See ``docs/tileops_codegen_design.md``.
"""

# --- recipe names, shared with torch_fl.tileops.runtime ---------------------
UNARY = "UNARY"
BINARY = "BINARY"
REDUCE = "REDUCE"
SOFTMAX = "SOFTMAX"

#: ``ref_api`` values whose trailing segment is not the aten base name. Anything
#: not listed falls back to ``ref_api.split(".")[-1]``.
ATEN_ALIAS = {
    "torch.matmul": "mm",
    "torch.nn.functional.softmax": "_softmax",
    "torch.nn.functional.log_softmax": "_log_softmax",
    "torch.nn.functional.layer_norm": "native_layer_norm",
    "torch.nn.functional.rms_norm": "_fused_rms_norm",
    "torch.nn.functional.group_norm": "native_group_norm",
    "torch.nn.functional.batch_norm": "native_batch_norm",
    "torch.nn.functional.conv1d": "convolution",
    "torch.nn.functional.conv2d": "convolution",
    "torch.nn.functional.conv3d": "convolution",
    "torch.linalg.vector_norm": "linalg_vector_norm",
}

#: ctor parameter tuple (``self``/``kernel_map``/``tune`` stripped) -> recipe.
#: Measured on H800: every group below accepts args derived purely from the aten
#: call site, including broadcasting and non-contiguous inputs.
RECIPES = {
    ("N_total", "dtype"): UNARY,
    ("N_total", "dtype", "inplace"): UNARY,
    ("a_shape", "b_shape", "dtype"): BINARY,
    ("a_shape", "b_shape", "dtype", "alpha"): BINARY,
    ("dtype", "dim"): REDUCE,
    ("dtype", "dim", "keepdim"): REDUCE,
    ("dtype", "dim", "correction", "keepdim"): REDUCE,
    ("dtype", "ord", "dim", "keepdim"): REDUCE,
    ("dim",): SOFTMAX,
}

#: Ops we will not route, with the measurement behind each decision. Keep the
#: numbers here so a future revisit can re-test rather than re-litigate.
EXCLUDE = {
    "ArgmaxFwdOp": "perf 0.01x (7.826 ms vs torch 0.044 ms @ 8192x4096 fp16)",
    "ArgminFwdOp": "same argreduce kernel family as ArgmaxFwdOp; excluded by association",
    "CumsumFwdOp": "max err 2.12 @ 8192x4096 fp16 -- accumulation precision unusable",
    "CumprodFwdOp": "same cumulative family as CumsumFwdOp; excluded by association",
    "GemmFp8Op": "fp8 output-dtype semantics do not match aten::mm",
    "TopkSelectorOp": "forward(index_score, starts, ends) does not match aten::topk; false alignment",
    "BatchNormFwdOp": "running_mean/var in-place update semantics need review",
    "BatchNormBwdOp": "backward pairing with BatchNormFwdOp; deferred with it",
    # Caught by the codegen validator rather than assumed:
    "InfNormFwdOp": "three ops share aten::linalg_vector_norm, split by `ord`; needs a manual dispatch",
    "L1NormFwdOp": "see InfNormFwdOp -- ord-based dispatch, manual recipe",
    "L2NormFwdOp": "see InfNormFwdOp -- ord-based dispatch, manual recipe",
    "RoundFwdOp": "forward(input, decimals) is 2-arg; does not fit the UNARY recipe",
    # Measured 2026-07-31 while planning the manual recipes. Unlike the entries
    # above these are blocked on TileOPs, not on our side: forward() returns only
    # the normalized tensor, while aten::native_layer_norm / native_group_norm
    # must also return (mean, rstd). The Op keeps no mean/rstd attribute, so the
    # statistics are unrecoverable. See tileops_codegen_design.md 3.4.
    "LayerNormFwdOp": "returns 1 tensor; aten::native_layer_norm needs (out, mean, rstd)",
    "GroupNormFwdOp": "returns 1 tensor; aten::native_group_norm needs (out, mean, rstd)",
    "GroupNormNoAffineFwdOp": "see GroupNormFwdOp -- statistics are discarded",
}

#: Manual recipes still to be written, grouped the way the adapter code will be
#: rather than by literal ctor signature. Every group below was verified on H800
#: (fp16, L2 called on CUDA tensors, compared against CPU aten) before being
#: written down; the numbers live in tileops_codegen_design.md 3.4.
MANUAL_GROUPS = {
    # ctor (N_total, dtype, <scalars>), forward(x) -- UNARY plus ctor scalars.
    # NOTE: GeluFwdOp.__init__ is (*args, **kwargs); the real signature is on
    # _GeluApproximateBase, so ctor introspection must walk the MRO.
    "SCALAR_UNARY": [
        "EluFwdOp",
        "LeakyReluFwdOp",
        "HardtanhFwdOp",
        "SoftplusFwdOp",
        "NanToNumFwdOp",
        "GeluFwdOp",
        "ClampScalarFwdOp",
    ],
    # ctor takes *shape tuples* (not tensors) and broadcasts internally;
    # forward takes the tensors. Scalars must be materialized as 0-dim tensors,
    # except ClampScalarFwdOp which takes python floats (prefer it).
    "BROADCAST_TENSORS": [
        "ClampFwdOp",
        "ClampMinFwdOp",
        "ClampMaxFwdOp",
        "WhereFwdOp",
        "MaskedFillFwdOp",
        "MaskedFillScalarFwdOp",
        "LerpTensorFwdOp",
    ],
    # BINARY plus one ctor scalar; bind per aten overload since the scalar only
    # exists on some of them (div.Tensor vs div.Tensor_mode).
    "BINARY_EXTRA": ["DivFwdOp", "LerpFwdOp"],
    # ctor takes no shape, so one global instance per config. GemmOp defaults to
    # trans_b=True (NT) -- aten::mm needs trans_b=False. Conv bias-present and
    # bias-absent are separate classes, and aten::convolution has transposed /
    # output_padding that TileOPs lacks: guard and fall back on those.
    "SHAPELESS": [
        "GemmOp",
        "BmmFwdOp",
        "Conv1dFwdOp",
        "Conv2dFwdOp",
        "Conv3dFwdOp",
        "Conv1dBiasFwdOp",
        "Conv2dBiasFwdOp",
        "Conv3dBiasFwdOp",
    ],
    # ctor args map 1:1 onto the aten args.
    "POOL": ["AvgPool1dFwdOp", "AvgPool2dFwdOp", "AvgPool3dFwdOp"],
    # Three classes behind one overload, picked by `ord`; other ord values fall
    # back to aten.
    "ORD_DISPATCH": ["L1NormFwdOp", "L2NormFwdOp", "InfNormFwdOp"],
    # decimals is a forward arg, so one instance serves round and round.decimals.
    "ROUND": ["RoundFwdOp"],
    # Not in the 102 aten-aligned set (the manifest's ref_api does not name an
    # aten op) but forward returns (out, indices) matching
    # aten::max_pool{1,2,3}d_with_indices exactly -- both bit-exact on H800.
    "MAXPOOL_INDICES": [
        "MaxPool1dIndicesFwdOp",
        "MaxPool2dIndicesFwdOp",
        "MaxPool3dIndicesFwdOp",
    ],
}

#: Generated into the conf as ``cuda`` so they ship inert; enable per-op with
#: ``FLAGOS_OP_<name>=tileops``. GEMM is kernel-bound slow (0.63x even when the
#: L2 Python layer is bypassed), so it stays a plumbing sample rather than a
#: default route.
DEFAULT_OFF = {
    "GemmOp": "kernel-bound 0.63x vs cuBLAS @ 4096^3 fp16",
    "BmmFwdOp": "0.97x vs torch.bmm; no headroom to justify the boxing hop",
}

#: aten overload to bind for each recipe entry. The manifest names a functional
#: ``ref_api``; the dispatcher needs a concrete overload.
ATEN_OVERLOAD = {
    "add": "add.Tensor",
    "sub": "sub.Tensor",
    "mul": "mul.Tensor",
    "div": "div.Tensor",
    "pow": "pow.Tensor_Tensor",
    "remainder": "remainder.Tensor",
    "floor_divide": "floor_divide",
    "eq": "eq.Tensor",
    "ne": "ne.Tensor",
    "lt": "lt.Tensor",
    "le": "le.Tensor",
    "gt": "gt.Tensor",
    "ge": "ge.Tensor",
    "maximum": "maximum",
    "minimum": "minimum",
    "bitwise_and": "bitwise_and.Tensor",
    "bitwise_or": "bitwise_or.Tensor",
    "bitwise_xor": "bitwise_xor.Tensor",
    "sum": "sum.dim_IntList",
    "mean": "mean.dim",
    "prod": "prod.dim_int",
    "all": "all.dim",
    "any": "any.dim",
    "var": "var.correction",
    "std": "std.correction",
    "var_mean": "var_mean.correction",
    "count_nonzero": "count_nonzero.dim_IntList",
}
