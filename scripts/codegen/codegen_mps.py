#!/usr/bin/env python3
# Copyright (c) 2026, BAAI. All rights reserved.
#
# Generate the Apple Metal (MPSGraph) operator layer.
#
# The Apple backend is deliberately small and grows one op at a time: an op is
# claimed on PrivateUse1 only once a kernel exists for it, and the conf only
# names this backend for the ops the emitted register file claims. So there is
# no scan of the shared op list here -- the op set below IS the coverage set, and
# every entry has a body written out once.
#
# This tree carries stages 1 and 2 of
# docs/superpowers/specs/2026-10-14-apple-mps-native-backend-design.md section 7
# -- the `unary` category, the elementwise `binary` and `binary_scalar` families,
# and the `reduce` family -- plus four stage-3 increments: the `gemm` category
# (`mm` and `bmm`), the `softmax` category (`_softmax`), the elementwise `neg`
# and `pow.Tensor_Scalar`, and the `conv` category (`convolution_overrideable`),
# each matching the CPU reference within tolerance.
#
# Two consequences worth stating up front, mirroring codegen_kunlunxin.py:
#
#   * There is no conf writer. torch_fl/configs/backends_mps.conf is generated
#     by scripts/codegen/gen_vendor_confs.py from the emitted mps_register.inc
#     (mps is in VENDORS), so the register file is the single hand-off between
#     the two generators and the conf keeps one writer.
#
#   * There is no symbol probe. The MPSGraph entry points are Objective-C
#     selectors resolved at link time against the MetalPerformanceShadersGraph
#     framework the build links (csrc/CMakeLists.txt, FLAGOS_ACCELERATOR=mps), so
#     a selector that does not exist is a build error rather than a silently
#     skipped op.
#
# The generated file is Objective-C++ (.mm) and builds WITHOUT ARC, like the
# runtime under csrc/runtime/accelerator/mps/: every object this file creates is
# released explicitly through mps::MpsRelease.
#
# Run it to (re)generate csrc/aten/backends/mps/generated/; `--check` verifies the
# committed artifacts are current without writing, which is the mode CI uses.

import argparse
import os
import re
import sys
from pathlib import Path

# Reuse the authoritative naming from the CUDA codegen so the emitted
# REGISTER_IMPL_TO_DISPATCHER(FnType, dispatcher, ...) matches the
# DECLARE_DISPATCHER in generated/ops.h exactly (otherwise the build won't link).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codegen_ops import schema_to_cpp_name

REPO = Path(__file__).resolve().parent.parent.parent
OUT_MM = REPO / "csrc/aten/backends/mps/generated/mps_kernels.mm"
OUT_INC = REPO / "csrc/aten/backends/mps/generated/mps_register.inc"
REGISTER_INC = REPO / "csrc/aten/generated/register.inc"

# ---------------------------------------------------------------------------
# Op registry: schema op name -> category. The category names the template and
# the template owns the whole body. `call` is the MPSGraph selector the category
# composes the result with.
#
# Only the functional forms are listed. The `.out` and inplace spellings
# (`abs.out`, `abs_`, `add.out`, `add_`, `sum.IntList_out`, `softmax.int_out`,
# `pow.Tensor_Scalar_out`, ...) are left unregistered on PrivateUse1, exactly as
# on GCU/MUSA/Ascend: the dispatcher then sends those spellings to the generic
# fallback rather than to an empty kernel slot. A later stage adds them once
# their kernels exist -- serving one means writing into a tensor the caller
# owns, which this layer does not do.
#
# `max.dim` and `min.dim` are absent on purpose. aten defines both as a
# (values, indices) pair, and the index half has no MPSGraph selector that comes
# with a documented tie-break: `reductionArgMaximumWithTensor:axis:` exists, but
# which of two equal maxima it names is not specified, and the reference has an
# answer. They keep the CPU fallback until that can be measured on a device.
# ---------------------------------------------------------------------------
OPS = {
    "abs": "unary",
    "add.Tensor": "binary",
    "sub.Tensor": "binary",
    "mul.Tensor": "binary",
    "add.Scalar": "binary_scalar",
    "sub.Scalar": "binary_scalar",
    "mul.Scalar": "binary_scalar",
    "sum.dim_IntList": "reduce",
    "mean.dim": "reduce",
    # `softmax` is the third stage-3 lane and the op a small model calls between
    # the matmuls: one per attention block, over the (batch, heads, seq, seq)
    # scores. MPSGraph has the whole operation behind one selector,
    # `softMaxWithTensor:axis:`, which is what the body hands it once the shape
    # and dtype contract is checked.
    #
    # Only `_softmax` is registered, and it is the only spelling that needs to
    # be. On this PyTorch build `softmax.int` is CompositeImplicitAutograd: the
    # dispatcher decomposes it above the backend key, so a kernel registered for
    # it is never called and a probe confirms no `softmax.int` wrapper exists in
    # csrc/aten/generated/register.inc to register. What actually reaches the
    # PrivateUse1 key -- what `torch.softmax`, `F.softmax`, SDPA and
    # `torch.softmax(dtype=...)` all lower to -- is `aten._softmax(self, dim,
    # half_to_float)`. `softmax.int_out` is the `out=` spelling and is left
    # unregistered like every other `.out` form here: serving it means writing
    # into a tensor the caller owns, which this layer does not do.
    "_softmax": "softmax",
    # The elementwise unary a model reaches for next: `neg` is what a masked or
    # signed activation is built from, and it is the one unary whose MPSGraph
    # selector is exact on the integral widths as well (`-(-128)` wraps to
    # -128 there, exactly as aten does), so the body is the plain unary template.
    "neg": "unary",
    # A scalar exponent. MPSGraph's power is a real selector rather than an
    # exp/log composition, which is what keeps the NaN of a negative base to a
    # fractional exponent -- the case the reference produces and a composition
    # would not.
    "pow.Tensor_Scalar": "pow_scalar",
    # Stage 3 opens with the matmuls, because they are where a small model
    # spends its time: measured on an M5 (one Qwen3-0.6B layer, hidden 1024, seq
    # 512, fp32), the eight `mm` calls and two `bmm` calls were 71-73% of the
    # layer's device dispatch time, and at seq 1 -- where the whole layer's
    # arithmetic is 0.09 GFLOP -- the eight `mm` calls were still 84% of it, so
    # the cost is the host round trip around the weights rather than the flops.
    # `addmm`, `mv`, `linear` and the `.out` spellings follow the same rule as
    # the categories above: unregistered until a kernel exists.
    "mm": "gemm",
    "bmm": "gemm",
    # Convolution, the last lane of stage 3 and the op the two matmuls above are
    # not enough for: a vision model (and a Qwen3.5-class mixer's depthwise
    # conv1d) reaches `conv2d`/`conv1d` through `nn.Conv2d` and
    # `F.conv2d`. On this PyTorch build `aten.convolution` is
    # CompositeImplicitAutograd and decomposes above the backend key into
    # `aten.convolution_overrideable` -- the PrivateUse1 spelling every vendor
    # backend fills -- so that is the op to register, and it is the op a probe
    # sees `F.conv2d` land on. `convolution_overrideable.out` and
    # `aten.convolution.out` join the other `.out` forms on the fallback which is
    # the spelling that writes into a tensor the caller owns. The three other
    # lowering spellings (`_convolution`, `slow_conv2d_forward`, `_mps_convolution`)
    # are the CUDA/MPS-specific paths aten picks off the CUDA/MPS key and are not
    # the ones a PrivateUse1 tensor reaches, so they stay unregistered too.
    "convolution_overrideable": "conv",
    # The first activation a transformer reaches: `silu` is what a SwiGLU MLP
    # multiplies its gate projection by, so it runs once per layer on the
    # (batch, seq, intermediate) tensor -- a bigger operand than the softmax
    # above ever sees. MPSGraph has no `siluWithTensor:`, but it does have
    # `sigmoidWithTensor:`, and `x * sigmoid(x)` is the definition, so this is
    # the first body here whose graph is two nodes rather than one.
    #
    # Only `silu` is registered. `silu.out` and `silu_` write into a tensor the
    # caller owns, like every other `.out`/inplace spelling on this backend, and
    # `silu_backward` is a separate op with a separate selector question that
    # this increment does not answer.
    "silu": "silu",
    # The largest single item left on a small model's fallback profile, and the
    # first op here whose cost is a *read* of its operand rather than a
    # computation over it: measured on an M5 (one Qwen3-0.6B layer, hidden 1024,
    # seq 512), `embedding` was ~47% of the layer's device dispatch time.
    #
    # The reason is not the arithmetic -- a table read is nearly free -- but what
    # the fallback does instead of it. `embedding` is routed to `none` in the
    # MPS conf, so a call lands on `cpu_fallback`, and that path copies the whole
    # (151936, 1024) weight to the host and the result back: measured on the same
    # M5, one `F.embedding` call with a (1, 512) index tensor cost 55.3 ms, of
    # which 55.8 ms was the host copy of the weight alone, against 0.697 ms for
    # the same read performed entirely on the CPU. The op is therefore worth
    # serving even though the graph node it builds is a single gather.
    #
    # Only `embedding` is registered. `embedding.out` writes into a tensor the
    # caller owns like every other `.out` form here, and `embedding_dense_backward`
    # is a separate op with its own scatter question.
    "embedding": "embedding",
    # The same table read the body above serves, reached through the op the
    # caller names rather than through the embedding front end. `embedding`
    # gathers because a table read is what `F.embedding` asks for; `index_select`
    # *is* that read, and every other op that wants rows of a big tensor --
    # `nn.Embedding`'s backward, an attention KV cache, a beam search's
    # reordering -- reaches this one instead.
    #
    # It is served on the same evidence `embedding` was, and the cost is larger
    # because the operand is: measured on an M5 with a (151936, 1024) float32
    # table, `index_select` of a *single* row cost 40.4 ms and the same call
    # through `gather` cost 40.8 ms, both against 0.29 ms for the one gather node
    # this body builds. The work is the same one node at any row count: 512 rows
    # cost 0.29 ms and 4096 rows 0.78 ms, so the graph does not care how much of
    # the table is read while the fallback cares only how *big* it is. The
    # fallback's cost is the whole operand, copied device-to-host and back
    # (profiled: `aten::_to_copy` at 178 ms over three passes of the 622 MB
    # table), which is why V=10000 costs 2.4 ms and V=151936 costs 41.7 ms for
    # the same one row.
    #
    # Only `index_select` is registered. `index_select.out` writes into a tensor
    # the caller owns like every other `.out` form here, and the other spellings
    # of the same read -- `take`, `gather`, `index` -- are separate ops with their
    # own argument contracts and stay on the fallback; they were measured with
    # this one (40.6-40.8 ms) and are the same node under a different contract,
    # so they are the next increment rather than this one.
    #
    # `dim` is a host value the body reads and the operands cannot say, so it is
    # named in the key: a dim=0 call and a dim=1 call over the same shapes are
    # two graphs, and a key that left `dim` out would hand the first one to both.
    "index_select": "index_select",
    # The third spelling of the same gather, and the one with the simplest
    # contract: `take` reads the operand as a flat buffer and indexes it with an
    # index of *any* rank, returning the index's own shape. It is the read a
    # `vocab[flat_ids]` over a packed table performs, and it round-trips for the
    # same reason its two siblings did: measured on an M5 with a (151936, 1024)
    # float32 operand, `take` of a *single* element cost 45.6 ms and 512 elements
    # cost 46.3 ms, against the 0.29 ms of the one gather node this body builds.
    #
    # It is a separate body rather than a fourth entry in the index_select
    # template because its argument contract is genuinely different, and measured
    # to be so: `take` takes **int64 only** (`RuntimeError: take(): Expected a
    # long tensor for index, but got Int`, where `index_select` accepts int32 and
    # int64), it takes any index rank rather than a vector, and it accepts a
    # negative index (-1 reads the last element) where `index_select` raises. Its
    # range check is therefore over the *flattened* operand, not over axis 0, and
    # it has no `dim` to key on. Folding it into the sibling's body would mean a
    # body with two disagreement maps, which is how the wrong list gets applied
    # to the wrong op.
    "take": "take",
    # The last arithmetic gap on the elementwise path. `add`, `sub` and `mul`
    # have been served since stage 2; `div` was not, so every `x / y` and every
    # `x / 2.0` in a model took the CPU round-trip, and it is the op the
    # transformers reach for at two places that matter: the `1 / sqrt(d)`
    # attention scale and the `x / sum(exp(x))` tail of a hand-written softmax.
    #
    # The cost is the round-trip rather than the arithmetic. Measured on an M5
    # over a (2, 512, 1024) operand, `div.Scalar` cost 0.68-0.95 ms against a
    # 0.24 ms device-to-host copy of the same tensor -- three host passes over
    # the operand where the kernel does one -- and 1.36-2.66 ms for the tensor
    # form. `reciprocal` is served with it: it is the same body with a constant
    # numerator, `1 / x` is `div` with a scalar self, and MPSGraph's reciprocal
    # is the same arithmetic (and the same overflow policy) as the division
    # node. Serve one and the other is nearly free; serve `reciprocal` alone and
    # a caller who writes `1 / x` still takes the round-trip, because that is
    # `div`.
    #
    # Only the two plain overloads. `div.out` writes into a tensor the caller
    # owns like every other `.out` form here, and `div.Tensor_mode` / `.Scalar_mode`
    # carry a `rounding_mode` whose three values (`floor`, `trunc`, `None`) the
    # body would have to branch on *inside* the graph -- a divisor the truncating
    # one needs and the other two must not have, which is a different body rather
    # than a keyword. Both take the round-trip.
    "div.Tensor": "div",
    "div.Scalar": "div_scalar",
    "reciprocal": "reciprocal",
}

# The aten op a kernel falls back to on the CPU for a case the MPSGraph path does
# not compute (the SUPPORTS predicate below). The fallback keeps the result
# correct and stays on the host: it is not a delegation to a CUDA kernel.
AT_OP = {
    "abs": "abs",
    "add.Tensor": "add",
    "sub.Tensor": "sub",
    "mul.Tensor": "mul",
    "add.Scalar": "add",
    "sub.Scalar": "sub",
    "mul.Scalar": "mul",
    "sum.dim_IntList": "sum",
    "mean.dim": "mean",
    "_softmax": "_softmax",
    "neg": "neg",
    "pow.Tensor_Scalar": "pow",
    "mm": "mm",
    "bmm": "bmm",
    # The CPU reference the `conv` category round-trips to. `convolution` is the
    # composite that lowers to `convolution_overrideable`, and the reference
    # implementation behind it is the slow 2d/3d path, so this is what the
    # round-trip is written against -- it carries the same shape, stride, padding,
    # dilation, transposed, output_padding and groups arguments the kernel holds.
    "convolution_overrideable": "convolution",
    # The CPU reference the `silu` body round-trips to on a dtype it declines.
    # `silu` exists as its own aten kernel rather than a decomposition, so the
    # round-trip lands on the reference's own implementation rather than on a
    # composite that would re-enter the dispatcher.
    "silu": "silu",
    # The CPU reference the `embedding` body round-trips to on anything it
    # declines. `embedding` is the op `F.embedding`, `nn.Embedding` and
    # `nn.EmbeddingBag`'s non-bagged path all reach, and it is the reference
    # rather than `index_select` because a table read is what the caller asked
    # for and the reference is where its out-of-range contract lives.
    "embedding": "embedding",
    # The CPU reference the `index_select` body round-trips to on anything it
    # declines: a dim this body has no node for, and the index dtype and range
    # the body checks on the host. It is `at::index_select` for the same reason
    # `embedding` is `at::embedding` -- that is where the reference's own
    # `IndexError` and its own dtype list live -- and it is the reference rather
    # than `at::gather`, which carries a different argument contract for the
    # same node.
    "index_select": "index_select",
    # The CPU reference the `take` body round-trips to on anything it declines:
    # an index that is not int64, and an element the operand does not have. It is
    # `at::take` for the same reason the other two are their own aten op -- that
    # is where the reference's `IndexError` (`out of range: tried to access index
    # 20 on a tensor of 20 elements`) and its own int64-only `RuntimeError` live.
    "take": "take",
    # The CPU reference the `div` body round-trips to on anything it declines.
    # It is `at::div` rather than a reciprocal-multiply: the two readings differ
    # on the cases the gate below exists for (a zero denominator, a subnormal
    # quotient), so the round-trip has to land on the reference's own kernel.
    "div.Tensor": "div",
    "div.Scalar": "div",
    # `reciprocal` exists as its own aten kernel, so the round-trip gets the
    # reference's own `1 / x` rather than a composite that would re-enter the
    # dispatcher.
    "reciprocal": "reciprocal",
}

# The per-category dtype predicate each kernel gates on (mps_common.h).
SUPPORTS = {
    "unary": "MpsSupportsUnaryDtype",
    "binary": "MpsSupportsBinaryDtype",
    "binary_scalar": "MpsSupportsScalarDtype",
    "reduce": "MpsSupportsReduceDtype",
    "gemm": "MpsSupportsGemmDtype",
    "softmax": "MpsSupportsSoftmaxDtype",
    "pow_scalar": "MpsSupportsPowDtype",
    "conv": "MpsSupportsConvDtype",
    "silu": "MpsSupportsSiluDtype",
    "embedding": "MpsSupportsEmbeddingDtype",
    # The same predicate over the same pair, and shared rather than copied
    # because the two ops accept the same thing: one float operand (the table
    # here, the weight there) indexed by one integer tensor, with the reference's
    # own two integer widths. `index_select`'s index list is the same one -- a
    # float or bool index is a `RuntimeError` naming int32/int64 -- and its
    # out-of-range contract is the same `IndexError`, so the pair of bodies
    # declines on the same inputs and takes the same round-trip for them.
    "index_select": "MpsSupportsEmbeddingDtype",
    # Not the sibling's predicate, because the two ops do not accept the same
    # index: `take` is int64-only on the reference (`RuntimeError: take():
    # Expected a long tensor for index, but got Int`) where `index_select` takes
    # int32 as well, so a shared gate would let an int32 `take` reach the graph
    # and answer a call the reference raises on. The operand half is wider than
    # the sibling's for the opposite reason -- a read does not change its
    # operand's width, and the reference passes every one of them through.
    "take": "MpsSupportsTakeDtype",
    # Neither form takes the dtype-preserving predicate its sibling arithmetic
    # ops use: division is not dtype-preserving on the reference (`int / int` is
    # a float tensor), and a predicate that admitted the integral widths would
    # return the graph's truncated int64 where the CPU returns float32 -- a wrong
    # answer rather than an exception. So the pair and the scalar operand each
    # have their own gate over the float widths. See the notes above the two
    # predicates in mps_common.h.
    "div": "MpsSupportsDivDtype",
    "div_scalar": "MpsSupportsDivScalarDtype",
    # `reciprocal` computes on the same node family as `div` with a constant
    # numerator, so it takes the div wall rather than the unary one: the two
    # must agree on the dtypes that are served, or `x / 2` and `1 / x` would
    # answer in different widths.
    "reciprocal": "MpsSupportsDivDtype",
}

# The selector each body applies to the operand, up to and including the keyword
# its first argument binds to: `{call}` sits immediately before that argument, so
# the unary and reduce entries end at the operand (`[graph absoluteWithTensor:x
# name:nil]`, `[graph reductionSumWithTensor:x axes:axes name:nil]`) and the
# elementwise entries name the second operand in the template instead.
CALLS = {
    "abs": "absoluteWithTensor:",
    "add.Tensor": "additionWithPrimaryTensor:",
    "sub.Tensor": "subtractionWithPrimaryTensor:",
    "mul.Tensor": "multiplicationWithPrimaryTensor:",
    "add.Scalar": "additionWithPrimaryTensor:",
    "sub.Scalar": "subtractionWithPrimaryTensor:",
    "mul.Scalar": "multiplicationWithPrimaryTensor:",
    "sum.dim_IntList": "reductionSumWithTensor:",
    "mean.dim": "reductionSumWithTensor:",
    # One selector for both ops: it broadcasts, so the batched rank-3 pair bmm
    # takes is the same node as the rank-2 pair mm takes.
    "mm": "matrixMultiplicationWithPrimaryTensor:",
    "bmm": "matrixMultiplicationWithPrimaryTensor:",
    "neg": "negativeWithTensor:",
    "_softmax": "softMaxWithTensor:",
    "pow.Tensor_Scalar": "powerWithPrimaryTensor:",
    # The conv body picks between the 2-D and 3-D selectors on the operand's
    # rank, so `{call}` names the shape-independent prefix both share and the
    # template fills the rank in.
    "convolution_overrideable": "convolution{dim}DWithSourceTensor:",
    # The body builds two nodes and the unary call table names one selector, so
    # this is the first node -- the sigmoid every silu is built from. See T_SILU
    # for why the multiply is written into the template rather than named here.
    "silu": "sigmoidWithTensor:",
    # A table read is a gather with `axis = 0` and no batch dimensions, and this
    # is that selector. See T_EMBEDDING for why the *other* gather spelling
    # (`gatherAlongAxis:`) is not usable here, and for the two things the
    # reference contract forces the body to check on the host before it builds
    # the node.
    "embedding": "gatherWithUpdatesTensor:",
    # The same gather, and the same selector, because `index_select` is that
    # read named directly. T_INDEX_SELECT carries the one difference: `dim` can
    # be any axis rather than the 0 the embedding body hard-codes, so the body
    # takes the operand's rank into account when it decides whether the single
    # `axis:0` node it builds is the axis the caller asked for.
    "index_select": "gatherWithUpdatesTensor:",
    # Still the same gather, and the difference is only in what the body has to
    # describe to it: `take` reads the operand as one flat row, so the reshape is
    # to `(1, numel)` and the index is the flat list -- there is no axis 0 to
    # preserve and no row width to compute. See T_TAKE.
    "take": "gatherWithUpdatesTensor:",
    # MPSGraph's division node, and the interesting thing about it is that it is
    # the *only* arithmetic selector here whose choice is not obvious: the
    # plausible alternative is `a * reciprocalWithTensor:b`, which is what a
    # bare-metal port of `div` usually reaches for. Measured on an M5 against
    # `at::div` over the cases that separate the two -- a zero denominator, a
    # signed zero, an inf/nan pair, the smallest denormals -- the two readings
    # are bit-identical to each other *and* to the reference on every one of
    # them, so the cheaper one node is the one this body builds. See
    # MpsSupportsDivDtype for the measurement and for the one case (a subnormal
    # result) where the node and the reference do part.
    "div.Tensor": "divisionWithPrimaryTensor:",
    "div.Scalar": "divisionWithPrimaryTensor:",
    # The same family with a constant numerator, so it needs no second operand
    # and `{call}` names the one-argument selector. Sized by the same gate as
    # the div bodies above.
    "reciprocal": "reciprocalWithTensor:",
}

# The rank each op's aten contract is written for: `mm` takes two matrices, `bmm`
# two stacks of them, and a call whose operands are some other rank is declined
# rather than handed to a selector that broadcasts.
RANK = {
    "mm": 2,
    "bmm": 3,
}

# The host `alpha` an aten signature carries, as the three pieces a body needs:
# the trailing parameter to declare, the trailing argument the fallback passes on
# to the reference, and the factor the secondary operand is scaled by before the
# graph sees it. The mul forms have no alpha and pass a literal 1, which
# MpsScaledByScalar returns the operand unchanged for -- no node, like every
# alpha of 1.
ALPHA = {
    "add.Tensor": (", const at::Scalar& alpha", ", alpha", "alpha"),
    "sub.Tensor": (", const at::Scalar& alpha", ", alpha", "alpha"),
    "add.Scalar": (", const at::Scalar& alpha", ", alpha", "alpha"),
    "sub.Scalar": (", const at::Scalar& alpha", ", alpha", "alpha"),
}

# The host values a body bakes into its graph as constants, as the trailing
# argument list `mps::MpsGraphKey` takes. MpsRunGraph folds the operands' own
# signature -- dtype and shape of every operand and of the result -- into the
# cache key itself, so this only has to name what the tensors cannot say: the
# `alpha` a scaled operand carries, the host scalar a `.Scalar` form adds, the
# axes a reduction covers, the layout a gemm body decides for an operand from
# its strides. Empty for an op whose graph is a function of its operands alone.
#
# A host value a body reads and this list leaves out is not a slow path, it is a
# wrong answer: the two calls produce one cache key and the first compiled graph
# answers both. So an op with a host constant declares it here, and a reviewer
# checks this row against the body that reads it.
KEY_PARTS = {
    "abs": "",
    "neg": "",
    "add.Tensor": ", alpha",
    "sub.Tensor": ", alpha",
    "mul.Tensor": "",
    "add.Scalar": ", other, alpha",
    "sub.Scalar": ", other, alpha",
    "mul.Scalar": ", other",
    "sum.dim_IntList": ", axes",
    "mean.dim": ", axes",
    # Which of the two operands the graph has to transpose: see T_GEMM.
    "mm": ", self_transposed, mat2_transposed",
    "bmm": ", self_transposed, mat2_transposed",
    # The axis the body resolved against the operand's rank. The result dtype
    # `half_to_float` decides does not need naming here: the body widens the
    # operand to it first, and MpsRunGraph folds the operand's own dtype into the
    # cache key, so the widened (batch, heads, seq, seq) float tensor of a
    # half_to_float call and a plain half one already sit under different keys.
    "_softmax": ", axis",
    # The exponent, which is the constant the graph bakes in and the operands say
    # nothing about.
    "pow.Tensor_Scalar": ", other",
    # Every geometric argument the conv body reads, and the bias flag with them:
    # the graph node is a function of the operands' shapes (which MpsRunGraph
    # folds in) *and* of stride, padding, dilation, groups, transposed and
    # output_padding, which the shapes cannot say. `transposed` and
    # `output_padding` are declined before the graph is built, but the kernel
    # still names them because MpsConvDescriptor takes them and a body that read
    # a host value the key did not name would hand the first plan to the second
    # call. `has_bias` is the last piece of the graph's shape that the tensor
    # list does not carry: a bias-less call hands the graph one operand fewer.
    "convolution_overrideable": (
        ", stride, padding, dilation, groups, transposed, output_padding, has_bias"
    ),
    # Empty on purpose, and it is the interesting row in this table. `silu` has
    # no host argument at all -- the graph is a function of the operand's dtype
    # and shape, which MpsRunGraph folds into the key itself. A body with no
    # constants has no way to be handed a plan it did not build.
    "silu": "",
    # Empty: the body reads no host value that the operands cannot say. The three
    # non-tensor arguments are backward-pass arguments whose value the forward
    # output does not depend on -- measured, see the note above T_EMBEDDING -- so
    # they are not read and not named. The one host value the body *does* read is
    # the index range check, and its outcome is a function of the indices
    # themselves, which MpsRunGraph folds in.
    "embedding": "",
    # `dim` is the interesting row here, and the one a reviewer has to check
    # against the body: the graph is a gather over axis 0 (or over the axis
    # `dim` names, which the body resolves to a graph it can build), and the
    # operands cannot say which axis was asked for -- a (4, 6) operand read at
    # dim=0 and at dim=1 have the same dtype and shape and two different
    # answers. Naming it is what stops the first plan from answering both.
    "index_select": ", dim",
    # Empty, and that is the whole difference from its two siblings: `take` has
    # no `dim` and no other host argument, so a take graph is a function of its
    # operands' dtypes and shapes alone -- which MpsRunGraph folds in. There is
    # no host value left that an operand pair could disagree about, so there is
    # nothing to name and two calls that share a shape share a plan.
    "take": "",
    # Empty: a division graph is a function of its operands' dtypes and shapes,
    # which MpsRunGraph folds in, and of nothing else. There is no `alpha` on
    # either spelling and no host scalar the body bakes in -- `div.Scalar`'s
    # scalar *is* an operand constant and is named there.
    "div.Tensor": "",
    "div.Scalar": ", other",
    "reciprocal": "",
}

# The tail a reduce body runs once the reduction node exists, for the ops whose
# MPSGraph selector does not cover the whole semantics. Inserted as written
# between the reduction and the return, so braces here are C++ and not doubled.
FINISH = {
    "mean.dim": """\

        // mean has no MPSGraph selector: it is the sum over the reduced axes
        // divided by how many elements those axes covered, as a one-element
        // constant the graph broadcasts.
        int64_t reduced = 1;
        for (const int64_t axis : axes) {
          reduced *= self_c.size(axis);
        }
        MPSGraphTensor* divisor =
            mps::MpsConstant(graph, reduced, self_c.scalar_type());
        if (divisor == nil) {
          return nil;
        }
        y = [graph divisionWithPrimaryTensor:y
                             secondaryTensor:divisor
                                        name:nil];
""",
}

# ==========================================================================
# Templates
#
# `{kernel}`, `{fn}` and `{disp}` are filled from schema_to_cpp_name; `{at_op}`,
# `{supports}` and `{call}` from the tables above. Every other brace is C++ and
# is doubled so str.format leaves it alone.
# ==========================================================================

# Every body below has the same shape: gate on dtype, materialise a dense operand
# (MPSGraph reads and writes dense memory, so a strided operand is copied first),
# hand one graph to mps::MpsRunGraph under a key naming the host constants that
# graph bakes in, and take the CPU round-trip when that returns false -- an
# operand the Metal allocator does not own, a node the graph builder declined, or
# a graph that would not compile. Nothing runs here on every call: MpsRunGraph
# keeps the compiled graph under (this body's key, the operands' dtypes and
# shapes), so `build` runs once per distinct signature and later calls only
# schedule the work. The marshalling itself lives in MpsRunGraph, which describes
# the aten tensors to MPSGraph against the runtime's shared MTLBuffers and
# schedules the graph on the current stream's queue. Both sides of the graph are
# 1-D buffer views with the aten shape reapplied inside it, because MPSNDArray
# rounds the stride between the rows of an N-D array up to 16 bytes and would
# otherwise misread a torch-contiguous buffer whose innermost dimension is not
# already that wide: a (4, 5) float32 tensor has 20-byte rows, which MPSNDArray
# reads as 32. See the row-pitch note in mps_common.h.
T_UNARY = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!mps::{supports}(self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  at::Tensor self_c = self.contiguous();
  at::Tensor out = at::empty(self_c.sizes(), self_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{self_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        return [graph {call}args[0] name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# The elementwise body over two tensors. The graph builds each operand in its own
# aten shape and the named arithmetic selectors broadcast them exactly as aten
# does, so a (4, 1) against a (4, 5) needs no expansion on either side; the result
# shape is the reference's own (`at::infer_size`), and a pair that cannot
# broadcast takes the round-trip so the reference raises its own error.
#
# `alpha` is folded into the secondary operand as a multiplication by a constant,
# and only when it is not 1: `a + b` -- what almost every caller writes -- builds
# one node. The element type of both the constant and the operation is the
# operands' own, which the dtype gate above accepted nothing else for.
T_BINARY = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other{alpha_param}) {{
  if (!mps::{supports}(self.scalar_type(), other.scalar_type())) {{
    return at::{at_op}(self.cpu(), other.cpu(){alpha_arg}).to(self.device());
  }}
  at::Tensor self_c = self.contiguous();
  at::Tensor other_c = other.contiguous();
  std::vector<int64_t> out_shape;
  try {{
    out_shape = at::infer_size(self_c.sizes(), other_c.sizes());
  }} catch (const std::exception&) {{
    return at::{at_op}(self.cpu(), other.cpu(){alpha_arg}).to(self.device());
  }}
  at::Tensor out = at::empty(out_shape, self_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{self_c, other_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        MPSGraphTensor* other_t = mps::MpsScaledByScalar(
            graph, args[1], {alpha_operand}, self_c.scalar_type());
        if (other_t == nil) {{
          return nil;
        }}
        return [graph {call}args[0] secondaryTensor:other_t name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu(), other.cpu(){alpha_arg}).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# The elementwise body over a tensor and a host scalar. The scalar becomes a
# one-element constant in the operand's own element type and the graph broadcasts
# it; the dtype gate accepted only the scalars that element type can carry, so an
# `int32` tensor takes an integer scalar and a float tensor anything real. A
# fractional scalar that would have promoted an integral operand -- `int + 0.5` is
# a float tensor in aten -- makes MpsConstant decline, and the round-trip keeps
# the reference's dtype.
#
# A second host `alpha`, where the schema has one, is another factor on that same
# operand: `x + alpha * c`. Like the tensor form, a factor of 1 adds no node.
T_SCALAR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other{alpha_param}) {{
  if (!mps::{supports}(self.scalar_type(), other)) {{
    return at::{at_op}(self.cpu(), other{alpha_arg}).to(self.device());
  }}
  at::Tensor self_c = self.contiguous();
  at::Tensor out = at::empty(self_c.sizes(), self_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{self_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        MPSGraphTensor* other_t =
            mps::MpsConstant(graph, other, self_c.scalar_type());
        if (other_t == nil) {{
          return nil;
        }}
        other_t = mps::MpsScaledByScalar(
            graph, other_t, {alpha_operand}, self_c.scalar_type());
        if (other_t == nil) {{
          return nil;
        }}
        return [graph {call}args[0] secondaryTensor:other_t name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu(), other{alpha_arg}).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# The reduction body. `dim` is normalised once, on the host, into the axes the
# graph reduces over: an empty or absent list means every axis, negative positions
# resolve against the rank, and a position outside the rank or a repeat is a
# decline -- those are the two inputs aten rejects, and the round-trip lets the
# reference say so in its own words.
#
# The named `axes:` selectors keep the rank and leave each reduced axis at size 1,
# which is the keepdim shape; the aten shape is applied by the reshape MpsRunGraph
# reads the result through, so keepdim and its absence are the same graph.
T_REDUCE = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype) {{
  if (!mps::{supports}(self.scalar_type(), dtype)) {{
    return at::{at_op}(self.cpu(), dim, keepdim, dtype).to(self.device());
  }}
  std::vector<int64_t> axes;
  if (!mps::MpsReduceAxes(self.sizes(), dim, &axes)) {{
    return at::{at_op}(self.cpu(), dim, keepdim, dtype).to(self.device());
  }}
  at::Tensor self_c = self.contiguous();
  if (self_c.numel() == 0) {{
    // MPSGraph has no empty tensor, and a reduction over nothing is not nothing:
    // the sum over an empty dim is zeros and the mean is NaN. The reference
    // already has that arithmetic.
    return at::{at_op}(self.cpu(), dim, keepdim, dtype).to(self.device());
  }}
  at::Tensor out = at::empty(
      mps::MpsReduceOutShape(self_c.sizes(), dim, keepdim), self_c.options());
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{self_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        MPSGraphTensor* y =
            [graph {call}args[0] axes:mps::MpsAxesArray(axes) name:nil];
{finish}        return y;
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu(), dim, keepdim, dtype).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# The matrix-multiply body. One MPSGraph selector serves both ops, because it
# broadcasts: `mm`'s rank-2 pair and `bmm`'s rank-3 one are the same node, and
# the op's own rank is the only thing the body holds the graph to.
#
# Four checks come before the graph. All four are the reference's contract rather
# than the graph's, and all four would otherwise be a wrong answer instead of an
# error:
#
#   * The rank. `mm` is defined for two matrices and `bmm` for two stacks, while
#     the broadcast selector answers happily for another rank.
#   * The inner dimensions, and the batch dimensions `bmm` requires to match. The
#     selector broadcasts batch dimensions too, so a pair aten refuses to
#     multiply would otherwise build a graph.
#   * A zero-length inner dimension. `(M, 0) x (0, N)` is M*N zeros rather than
#     an empty tensor, and there is no zero-element tensor to describe to
#     MPSGraph; the reference has that arithmetic.
#   * An empty result, which is returned as the empty tensor it is.
#
# Each failure takes the CPU round-trip, where the reference raises in its own
# words: `mm` of a rank-3 operand reports the rank, and a mismatched pair reports
# the two shapes.
#
# The operands are then described to the graph in the layout it should read:
# MpsGemmOperand resolves each one, and a transpose of the last two axes is left
# for the graph to apply as a view rather than copied on the device. `nn.Linear`
# is why that case is the common one -- it hands this op `weight.t()`. The layout
# is a host value the body reads, so both flags are named in the key as well:
# MpsRunGraph folds in dtype and shape only, and a transposed (n, n) operand and
# a dense (n, n) one would otherwise share whatever plan the first call compiled.
T_GEMM = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mat2) {{
  if (!mps::{supports}(self.scalar_type(), mat2.scalar_type())) {{
    return at::{at_op}(self.cpu(), mat2.cpu()).to(self.device());
  }}
  if (self.dim() != {rank} || mat2.dim() != {rank} ||
      self.size(-1) != mat2.size(-2)) {{
    return at::{at_op}(self.cpu(), mat2.cpu()).to(self.device());
  }}
  // Empty for mm, which has no batch dimension, and aten's equal-batch rule for
  // bmm.
  for (int64_t dim = 0; dim + 2 < self.dim(); ++dim) {{
    if (self.size(dim) != mat2.size(dim)) {{
      return at::{at_op}(self.cpu(), mat2.cpu()).to(self.device());
    }}
  }}
  std::vector<int64_t> out_shape(self.sizes().vec());
  out_shape.back() = mat2.size(-1);
  at::Tensor out = at::empty(out_shape, self.options());
  if (out.numel() == 0) {{
    return out;
  }}
  if (self.numel() == 0 || mat2.numel() == 0) {{
    return at::{at_op}(self.cpu(), mat2.cpu()).to(self.device());
  }}
  const mps::MpsGemmOperand self_operand = mps::MpsGemmOperandOf(self);
  const mps::MpsGemmOperand mat2_operand = mps::MpsGemmOperandOf(mat2);
  const bool self_transposed = self_operand.transposed;
  const bool mat2_transposed = mat2_operand.transposed;
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{self_operand.buffer, mat2_operand.buffer}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        // The operands arrive at the shape their own buffers hold, so a
        // transposed one is put back the way the operand presents itself before
        // the product.
        MPSGraphTensor* primary = args[0];
        MPSGraphTensor* secondary = args[1];
        if (self_transposed) {{
          const NSUInteger last = (NSUInteger)primary.shape.count;
          primary = [graph transposeTensor:primary
                                 dimension:last - 2
                             withDimension:last - 1
                                      name:nil];
        }}
        if (mat2_transposed) {{
          const NSUInteger last = (NSUInteger)secondary.shape.count;
          secondary = [graph transposeTensor:secondary
                                   dimension:last - 2
                               withDimension:last - 1
                                        name:nil];
        }}
        return [graph {call}primary secondaryTensor:secondary name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu(), mat2.cpu()).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# The softmax body. MPSGraph has the whole operation behind one selector, so the
# work here is the contract around it rather than the arithmetic.
#
# `softMaxWithTensor:axis:` reduces along one axis of one tensor, which is exactly
# what `_softmax(self, dim, half_to_float)` asks for, so there is one node and no
# composition. What has to be settled before it runs:
#
#   * The axis. aten resolves a negative `dim` against the rank and rejects one
#     out of range, and MPSGraph takes the axis as it is given -- an out-of-range
#     axis would be a graph that either wraps or traps, not the reference's error.
#     MpsSoftmaxDim resolves the position, so a negative dim is answered and a
#     rank-0 tensor or an out-of-range one takes the round-trip and gets the
#     reference's own error or its 1.0 for the scalar it does answer.
#   * The dtype the operation computes in, which is a function of the host
#     `half_to_float` flag rather than of the tensor: it returns the operand's own
#     dtype, or float when the flag says "accumulate in float and hand back
#     float". MpsSupportsSoftmaxDtype settles it and hands back the dtype, which
#     the operand is widened to before the graph sees it and the output is
#     allocated in -- the same conversion aten's generic path makes with
#     `_to_copy`, so a declined width stays one copy away from correct. The gate
#     also fixes the widths: float32 measured within 29 ulp and float16
#     bit-identical on an M5, while bfloat16 came back as denormals, so half is
#     served and bfloat16 is not (mps_common.h carries the measurements).
#
# The empty input is declined rather than handed to the graph: MPSGraph has no
# empty tensor, and the reference's answer (an empty result) is free to produce
# on the host.
T_SOFTMAX_BODY = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float) {{
  at::ScalarType out_dtype = at::kFloat;
  int64_t axis = 0;
  if (!mps::MpsSoftmaxDim(self, dim, &axis) ||
      !mps::MpsSupportsSoftmaxDtype(self.scalar_type(), half_to_float, &out_dtype)) {{
    return at::{at_op}(self.cpu(), dim, half_to_float).to(self.device());
  }}
  at::Tensor self_c = self.to(out_dtype).contiguous();
  if (self_c.numel() == 0) {{
    return at::empty(self_c.sizes(), self_c.options());
  }}
  at::Tensor out = at::empty(self_c.sizes(), self_c.options());
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{self_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        return [graph {call}args[0] axis:axis name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu(), dim, half_to_float).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# The tensor-raised-to-a-scalar body. The exponent is a host value, so it becomes
# a one-element constant in the operand's own element type and the graph
# broadcasts it -- the shape the `.Scalar` bodies give an addend, with the
# selector deciding the arithmetic.
#
# The dtype gate is narrower than the scalar bodies' because aten is the narrow
# one here: a float operand takes any real exponent, but an integral operand
# raises on a negative one ("Integers to negative integer powers are not
# allowed") and promotes to float on a fractional one, so no integral operand is
# served and the round-trip produces both the reference's dtype and the
# reference's error. bfloat16 is declined for the reason the softmax gate
# declines it -- on the measured SDK its graph returns values the reference does
# not -- and float64 has no MPSGraph element type at all.
#
# An exponent of 1 short-circuits inside MpsPowByScalar, so `x ** 1` builds no
# node and no constant, the way an `alpha` of 1 builds no node either.
T_POW = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  if (!mps::{supports}(self.scalar_type(), other)) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
  at::Tensor self_c = self.contiguous();
  at::Tensor out = at::empty(self_c.sizes(), self_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{self_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        return mps::MpsPowByScalar(
            graph, args[0], other, self_c.scalar_type());
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# The convolution body, over the one spelling a PrivateUse1 tensor reaches:
# `aten.convolution` is CompositeImplicitAutograd on this build and decomposes
# into `aten.convolution_overrideable` (see the OPS comment).
#
# MPSGraph has one convolution pair of selectors -- `convolution2D...` and
# `convolution3D...` -- and no 1-D one, so the body resolves the spatial rank
# first and every other rank takes the CPU round-trip. What the descriptor
# carries is the whole geometry: NCHW data and OIHW weights, per-axis stride and
# dilation, `groups`, and an explicit padding style whose per-axis
# `paddingFront`/`paddingBack` are aten's own padding list -- aten's `padding` is
# [front_0, front_1, ..., back_0, back_1, ...], which is the order
# `MpsConvDescriptor` fills, and a list that is not two entries per axis is
# declined so the reference reports its own error.
#
# The checks before the graph are the reference's contract rather than the
# graph's, and each would otherwise be a wrong answer instead of an error:
#
#   * The rank pair (a 3-D input with a 4-D weight, say) and the channel rules
#     (`in_channels / groups == weight.size(1)`, `weight.size(0)` a multiple of
#     `groups`). The graph would happily misread or broadcast either.
#   * `transposed`, which on this selector means a *transposed convolution*: a
#     different selector (`convolutionTranspose2D...`), with the channel roles
#     swapped and its own argument rules. The body declines it rather than
#     folding it into the forward node.
#   * `output_padding`, which only a transposed convolution reads, so a non-zero
#     one here is the reference's error.
#   * A zero-length operand, which MPSGraph has no tensor for. A zero-length
#     batch or spatial axis produces an empty result and is returned as one, and
#     every other empty operand takes the round-trip -- where the shape error
#     the reference raises is the reference's to write.
#
# The bias is the one part of the graph with a measured shape. MPSGraph's
# convolution descriptor runs the convolution alone, so the bias is an
# `additionWithPrimaryTensor:` against the conv result -- and that add's operand
# has to broadcast against an (N, OC, OH, OW) result. The bias arrives as
# (OC,), and the shape that broadcasts is (1, OC, 1, 1), which measured within
# 1.19e-07 of the reference on an M5. The two readings that look equally
# reasonable are both wrong: (1, 1, 1, OC) crashes the MPSGraph compiler
# ("mps.add operands don't have broadcast-compatible shapes", then "original
# module failed verification") and `broadcastTensor:toShape:` compares as
# 6.6e-01, because it expands the bias along the wrong axes. So the bias is
# reshaped, once, rather than broadcast.
#
# The dtype gate is the conv category's own (MpsSupportsConvDtype), and it is
# wider than the softmax gate above on purpose: a bfloat16 convolution came back
# *bit-identical* to a bfloat16 reference convolution on an M5 (6.52e-03 against
# an exact float64 reference for both -- bfloat16's own precision, not the
# graph's error), so bfloat16 is served here where bfloat16 softmax was
# declined. That is what a per-category predicate is for. float64 has no
# MPSGraph element type and the integral and bool widths are the reference's own
# NotImplementedError, so those take the round-trip and get it.
T_CONV = """\
at::Tensor {kernel}(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups) {{
  const bool has_bias = bias.has_value() && bias->defined();
  const at::ScalarType bias_dtype =
      has_bias ? bias->scalar_type() : input.scalar_type();
  if (!mps::{supports}(input.scalar_type(), weight.scalar_type(), bias_dtype)) {{
    return at::{at_op}(input.cpu(), weight.cpu(), bias, stride, padding, dilation,
                       transposed, output_padding, groups)
        .to(input.device());
  }}
  // The spatial rank: 1-D has no MPSGraph selector, and a rank above 3 has no
  // graph either, so both take the round-trip and the reference reports what it
  // reports.
  const int64_t dim = input.dim() - 2;
  const size_t ndim = dim > 0 ? static_cast<size_t>(dim) : 0;
  if (dim < 2 || dim > 3 || weight.dim() != input.dim() ||
      stride.size() != ndim || padding.size() != 2 * ndim ||
      dilation.size() != ndim || groups < 1) {{
    return at::{at_op}(input.cpu(), weight.cpu(), bias, stride, padding, dilation,
                       transposed, output_padding, groups)
        .to(input.device());
  }}
  if (input.size(1) / groups != weight.size(1) ||
      weight.size(0) % groups != 0) {{
    return at::{at_op}(input.cpu(), weight.cpu(), bias, stride, padding, dilation,
                       transposed, output_padding, groups)
        .to(input.device());
  }}
  if (transposed || output_padding.size() != ndim) {{
    return at::{at_op}(input.cpu(), weight.cpu(), bias, stride, padding, dilation,
                       transposed, output_padding, groups)
        .to(input.device());
  }}
  for (const int64_t entry : output_padding) {{
    if (entry != 0) {{
      return at::{at_op}(input.cpu(), weight.cpu(), bias, stride, padding,
                         dilation, transposed, output_padding, groups)
          .to(input.device());
    }}
  }}
  mps::MpsConvParams params;
  std::vector<int64_t> out_shape;
  if (!mps::MpsConvResolve(
          input, weight, stride, padding, dilation, groups, &params, &out_shape)) {{
    return at::{at_op}(input.cpu(), weight.cpu(), bias, stride, padding, dilation,
                       transposed, output_padding, groups)
        .to(input.device());
  }}
  at::Tensor out = at::empty(out_shape, input.options());
  if (out.numel() == 0) {{
    return out;
  }}
  at::Tensor input_c = input.contiguous();
  at::Tensor weight_c = weight.contiguous();
  if (input_c.numel() == 0 || weight_c.numel() == 0) {{
    return at::{at_op}(input.cpu(), weight.cpu(), bias, stride, padding, dilation,
                       transposed, output_padding, groups)
        .to(input.device());
  }}
  at::Tensor bias_c;
  if (has_bias) {{
    bias_c = bias->contiguous();
    if (bias_c.numel() != weight_c.size(0)) {{
      return at::{at_op}(input.cpu(), weight.cpu(), bias, stride, padding,
                         dilation, transposed, output_padding, groups)
          .to(input.device());
    }}
  }}
  std::vector<at::Tensor> operands{{input_c, weight_c}};
  if (has_bias) {{
    operands.push_back(bias_c);
  }}
  const int64_t spatial = out.dim() - 2;
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      operands,
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        MPSGraphObject* descriptor =
            mps::MpsConvDescriptor(input_c.scalar_type(), spatial, params);
        if (descriptor == nil) {{
          return nil;
        }}
        MPSGraphTensor* y =
            (spatial == 2)
                ? [graph {call2d}args[0]
                              weightsTensor:args[1]
                                descriptor:(MPSGraphConvolution2DOpDescriptor*)descriptor
                                 name:nil]
                : [graph {call3d}args[0]
                              weightsTensor:args[1]
                                descriptor:(MPSGraphConvolution3DOpDescriptor*)descriptor
                                 name:nil];

        if (y == nil) {{
          mps::MpsRelease(descriptor);
          return nil;
        }}
        // The bias, in the one shape that broadcasts against the (N, OC, ...)
        // result -- see the note above for why it is not (1, 1, 1, OC).
        if ({has_bias}) {{
          MPSGraphTensor* bias_t = [graph reshapeTensor:args[2]
                                              withShape:mps::MpsShapeOf(params.bias_shape)
                                                   name:nil];
          y = [graph additionWithPrimaryTensor:y
                               secondaryTensor:bias_t
                                          name:nil];
        }}
        mps::MpsRelease(descriptor);
        return y;
      }});
  if (!ran) {{
    return at::{at_op}(input.cpu(), weight.cpu(), bias, stride, padding, dilation,
                       transposed, output_padding, groups)
        .to(input.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# The SiLU activation, `x * sigmoid(x)`, which is what a SwiGLU MLP multiplies
# its gate projection by. It is the first body here whose graph is a composition
# rather than one selector, so it is worth stating what makes that safe.
#
# MPSGraph has no `siluWithTensor:` -- the selector set jumps from
# `sigmoidWithTensor:` to `sineWithTensor:` -- so the activation is built from
# the two nodes that define it. Composing is only legitimate if the reference
# computes the same expression rather than something merely close, and aten's
# CPU `silu` does *not* compute `x * sigmoid(x)`: it is its own kernel, and
# evaluated on the host the two readings differ by 7.8e-03 on float16 and
# 3.1e-02 on bfloat16, which would be a wrong answer rather than a rounding
# difference. So the composition is measured against the reference rather than
# assumed, and the measurement is what the `silu` entry in MpsSupportsSiluDtype
# records (mps_common.h): on an M5 the graph was bit-identical to at::silu on
# float16 and bfloat16, and within 2.7e-07 relative on float32 -- where it and
# the reference sit 1.2353e-06 from an exact float64 evaluation to the last
# digit, so the two round differently rather than one being wrong.
#
# That is the opposite of what the conv increment found about *bfloat16*, which
# is served there and served here too -- but for a different reason, and the two
# gates are separate predicates because of it. bfloat16 is served by the conv
# gate because the graph rounds the same way the reference does on a
# convolution; it is served here because the graph matched the reference on an
# activation. Neither measurement implies the other.
#
# `sigmoid` is one operand and one node, but the operand is used twice -- once
# as the sigmoid's input and once as the multiply's -- and the graph is a DAG,
# so a plan built for one operand shape is handed back for every later call with
# that shape. Nothing here reads a host value, which is why the key has no
# parts: `MpsRunGraph` folds the operand's dtype and shape in itself, and a body
# with no constants cannot be handed a plan it did not build.
T_SILU = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!mps::{supports}(self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  at::Tensor self_c = self.contiguous();
  at::Tensor out = at::empty(self_c.sizes(), self_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{self_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        MPSGraphTensor* gate = [graph {call}args[0] name:nil];
        if (gate == nil) {{
          return nil;
        }}
        return [graph multiplicationWithPrimaryTensor:args[0]
                                      secondaryTensor:gate
                                                 name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# `embedding` is the first body here that has to decide whether the *graph* can
# answer the call at all, rather than only whether it can be built. A table read
# is one gather node -- `gatherWithUpdatesTensor:indicesTensor:axis:0
# batchDimensions:0` -- but MPSGraph's gather has no out-of-range contract of its
# own, and the reference's contract is an error:
#
#   * measured on an M5 with a 6-row table and the index list (5, 6, 2, 0), the
#     node returns the *first row* for the out-of-range index 6; with -1 it
#     returns the first row, and with 100 the first row again. That is a
#     plausible-looking wrong answer rather than a failure, which is the one
#     outcome worse than a round-trip.
#   * the reference raises `IndexError: index out of range in self` for 6, -1 and
#     100 alike.
#
# So the body validates the index range on the host before it builds the node, and
# declines -- handing the call to the reference, which raises its own error -- for
# any index outside [0, num_embeddings). The check is one host pass over the index
# tensor, which is (batch, seq) against a (vocab, hidden) table: 512 elements
# against 151936 rows, so it does not move the measurement above.
#
# The *other* gather spelling, `gatherAlongAxis:withUpdatesTensor:indicesTensor:`,
# was measured first and rejected: it requires the indices to carry the operand's
# rank (`rank of destination array (1) must equal number of outer dimensions in
# index array (2) + rank of slice to update (1)`), and its own documented
# out-of-range behaviour is to insert a 0 -- a zero vector for a genuinely bad
# index, which is a wrong answer that would look like a padding row.
#
# `padding_idx` is deliberately *not* consulted. The forward op does not read it:
# measured on the CPU reference with a (4, 3) table and the index list
# (0, 1, 2, 3), the result is bit-identical for `padding_idx` -1, -3, -4 and 2 --
# every one of them returns `weight` row for row, and an out-of-range
# `padding_idx` (7, -9) is accepted rather than rejected. The zeroing that
# `padding_idx` asks for is a property of the *gradient* (`embedding_dense_backward`
# zeroes that row of the grad table), not of this op's output, so a body that
# branched on it would decline calls whose answer does not depend on it. The first
# version of this body did exactly that and took the whole-table round-trip for
# every `F.embedding` call -- which passes `padding_idx = None`, arriving here as
# -1. It is not in the key either: two calls that differ only in `padding_idx`
# produce the same bytes, so they are one graph.
#
# Only the int and long index widths are accepted, which is what the reference
# accepts: a float index is a `RuntimeError` naming the two integer types, and a
# bool index is not one either, so both take the round-trip and get that error.
T_EMBEDDING = """\
at::Tensor {kernel}(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {{
  if (!mps::{supports}(weight.scalar_type(), indices.scalar_type())) {{
    return at::{at_op}(
        weight.cpu(), indices.cpu(), padding_idx, scale_grad_by_freq, sparse)
        .to(weight.device());
  }}
  const int64_t rows = weight.size(0);
  // `padding_idx`, `scale_grad_by_freq` and `sparse` are all backward-pass
  // arguments: the reference's forward output does not depend on any of them (see
  // the note above T_EMBEDDING), so this body does not read them either and the
  // kernel is one gather for every value they take.
  // One host pass over the indices, in their own width, before a node exists.
  // Any index the table cannot answer takes the round-trip so the caller sees
  // the reference's own IndexError rather than MPSGraph's silent first row.
  const at::Tensor indices_c = indices.contiguous();
  bool in_range = true;
  if (indices_c.numel() > 0) {{
    const at::Tensor host = indices_c.cpu();
    const int64_t* as_long = nullptr;
    std::vector<int64_t> widened;
    if (host.scalar_type() == at::kLong) {{
      as_long = host.const_data_ptr<int64_t>();
    }} else {{
      const int32_t* as_int = host.const_data_ptr<int32_t>();
      widened.assign(as_int, as_int + host.numel());
      as_long = widened.data();
    }}
    for (int64_t i = 0; i < host.numel(); ++i) {{
      if (as_long[i] < 0 || as_long[i] >= rows) {{
        in_range = false;
        break;
      }}
    }}
  }}
  if (!in_range) {{
    return at::{at_op}(
        weight.cpu(), indices.cpu(), padding_idx, scale_grad_by_freq, sparse)
        .to(weight.device());
  }}
  at::Tensor weight_c = weight.contiguous();
  std::vector<int64_t> out_shape(
      indices_c.sizes().begin(), indices_c.sizes().end());
  out_shape.push_back(weight_c.size(1));
  at::Tensor out = at::empty(out_shape, weight_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  // The graph reads a flattened index buffer and reshapes the table, because an
  // MPSNDArray over N-D memory reads with its own row pitch (see the row-pitch
  // note in mps_common.h) and the gather's result is (num_indices, num_cols).
  const int64_t index_count = indices_c.numel();
  const int64_t row_width = weight_c.size(1);
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{weight_c, indices_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        MPSGraphTensor* table = [graph reshapeTensor:args[0]
                                           withShape:mps::MpsShapeOf(
                                                         {{rows, row_width}})
                                                name:nil];
        MPSGraphTensor* flat = [graph reshapeTensor:args[1]
                                          withShape:mps::MpsShapeOf(
                                                        {{index_count}})
                                               name:nil];
        return [graph gatherWithUpdatesTensor:table
                                indicesTensor:flat
                                         axis:0
                              batchDimensions:0
                                         name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(
        weight.cpu(), indices.cpu(), padding_idx, scale_grad_by_freq, sparse)
        .to(weight.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# `index_select` is the read the `embedding` body above performs, named
# directly, so the node it builds and the two host checks it makes are the same
# ones -- `gatherWithUpdatesTensor:indicesTensor:axis:0 batchDimensions:0`, the
# index dtype list, and the range check that keeps MPSGraph's silent first-row
# answer away from an out-of-range index (see T_EMBEDDING for the measurements
# behind the second and third).
#
# The one thing that is not the same is `dim`, and the way it is handled is the
# whole of this body's design:
#
#   * **dim = 0 is the embedding gather**, row by row, and it is the case worth
#     having: measured on an M5 with a (151936, 1024) float32 table, the fallback
#     costs 40.4 ms for *one* row and the graph costs 0.29 ms -- and the graph
#     costs the same 0.29 ms for 512 rows and 0.78 ms for 4096, because the node
#     reads the rows it was asked for rather than the operand. The fallback's
#     cost is the operand, not the result (V=10000 -> 2.4 ms, V=151936 ->
#     41.7 ms, same one row), so this is the axis where serving the op removes a
#     whole-table round-trip.
#   * **dim = -1 is dim = rank - 1**, resolved on the host the way every other
#     negative axis on this backend is, and it is declined: a gather along the
#     last axis of a contiguous table is a different node (the row width would
#     have to change) and one this increment does not measure.
#   * **any other dim is declined**, including the negative ones that resolve to
#     something other than 0. The body has no node for them and does not pretend
#     to.
#
# Declining is not a failure mode here: the round-trip is the reference's own
# implementation and its own `IndexError`, so a declined dim answers correctly
# and simply costs what it costs today. The alternative -- building a node for
# the axis the body happens to know how to reach and returning its answer for a
# different axis -- is the kind of plausible-looking wrong answer T_EMBEDDING
# exists to avoid.
#
# The second thing that is not the embedding body's is the *shape* of the index,
# and it is a check rather than a case: `index_select` takes a vector and says so
# (`IndexError: index_select(): Index is supposed to be a vector`), where
# `embedding` reads any index shape as a batch dim. That is not a formality --
# measured on an M5, a (2, 2) index over a (6, 4) operand reached the graph and
# came back as a (4, 4) tensor with a silent flatten, because the body sets
# `out_shape[0] = index.numel()` and flattens the index buffer before it
# describes it. The reference raises for the same call, so without this check the
# backend answers a question the reference refuses -- and answers it plausibly,
# which is the failure mode the whole decline machinery here exists to avoid.
T_INDEX_SELECT = """\
at::Tensor {kernel}(
    const at::Tensor& self, int64_t dim, const at::Tensor& index) {{
  if (!mps::{supports}(self.scalar_type(), index.scalar_type())) {{
    return at::{at_op}(self.cpu(), dim, index.cpu()).to(self.device());
  }}
  // A vector index is the reference's own contract (`Index is supposed to be a
  // vector`), and the one below it flattens the index buffer before it
  // describes it to MPSGraph, so a higher-rank index would otherwise be read as
  // its own flattening. Declined rather than reported here: the round-trip is
  // the reference's implementation and its own message.
  if (index.dim() != 1) {{
    return at::{at_op}(self.cpu(), dim, index.cpu()).to(self.device());
  }}
  // Resolve the axis against the operand's rank, then decide whether this body
  // has a node for it. `dim` may be negative and aten normalises it (measured:
  // dim=-1 over a rank-2 operand is dim=1), so the body does the same before it
  // compares, and a dim below -rank is left to the reference rather than
  // reported here.
  const int64_t rank = self.dim();
  if (rank == 0) {{
    return at::{at_op}(self.cpu(), dim, index.cpu()).to(self.device());
  }}
  int64_t axis = dim;
  if (axis < 0) {{
    axis += rank;
  }}
  if (axis != 0) {{
    return at::{at_op}(self.cpu(), dim, index.cpu()).to(self.device());
  }}
  // One host pass over the index, in its own width, before a node exists. Any
  // index the operand cannot answer takes the round-trip so the caller sees the
  // reference's own IndexError rather than MPSGraph's silent first row.
  const at::Tensor index_c = index.contiguous();
  const int64_t rows = self.size(0);
  bool in_range = true;
  if (index_c.numel() > 0) {{
    const at::Tensor host = index_c.cpu();
    const int64_t* as_long = nullptr;
    std::vector<int64_t> widened;
    if (host.scalar_type() == at::kLong) {{
      as_long = host.const_data_ptr<int64_t>();
    }} else {{
      const int32_t* as_int = host.const_data_ptr<int32_t>();
      widened.assign(as_int, as_int + host.numel());
      as_long = widened.data();
    }}
    for (int64_t i = 0; i < host.numel(); ++i) {{
      if (as_long[i] < 0 || as_long[i] >= rows) {{
        in_range = false;
        break;
      }}
    }}
  }}
  if (!in_range) {{
    return at::{at_op}(self.cpu(), dim, index.cpu()).to(self.device());
  }}
  at::Tensor self_c = self.contiguous();
  // The result is the operand's shape with axis 0 replaced by the index count.
  // For a rank-1 operand that is just (num_indices); for a rank-n one it is
  // (num_indices, ...) and the graph's gather returns exactly that.
  std::vector<int64_t> out_shape(
      self_c.sizes().begin(), self_c.sizes().end());
  out_shape[0] = index_c.numel();
  at::Tensor out = at::empty(out_shape, self_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  // The graph reads a flattened index buffer and reshapes the table, because an
  // MPSNDArray over N-D memory reads with its own row pitch (see the row-pitch
  // note in mps_common.h). The gather's own contract is the same one the
  // embedding body relies on: `axis:0 batchDimensions:0` indexes the outermost
  // dimension and returns the slices named, in the index's order.
  const int64_t index_count = index_c.numel();
  const int64_t rows_in_table = self_c.size(0);
  const int64_t row_width = self_c.numel() / (rows_in_table > 0 ? rows_in_table : 1);
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{self_c, index_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        MPSGraphTensor* table = [graph reshapeTensor:args[0]
                                           withShape:mps::MpsShapeOf(
                                                         {{rows_in_table, row_width}})
                                                name:nil];
        MPSGraphTensor* flat = [graph reshapeTensor:args[1]
                                          withShape:mps::MpsShapeOf(
                                                        {{index_count}})
                                               name:nil];
        return [graph gatherWithUpdatesTensor:table
                                indicesTensor:flat
                                         axis:0
                              batchDimensions:0
                                         name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu(), dim, index.cpu()).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# `take` is the third spelling of the embedding gather, and it is the one with
# the *least* to get wrong and the most to get wrong quietly.
#
# What it does not have is the embedding body's problem. `embedding` reads axis 0
# of a table and `index_select` reads an axis the caller names; `take` reads the
# operand as one flat buffer of `numel()` elements, so the graph is a reshape to
# `(1, numel)` and a gather over axis 0 with the index as a flat list. There is no
# row width to compute, no axis to resolve, and no host argument at all -- which
# is why `KEY_PARTS` for this op is empty and two calls that share a shape share a
# plan.
#
# What it does have is a different contract from both siblings, measured rather
# than assumed, and each difference is a way to answer a call the reference
# rejects:
#
#   * **The index is int64 only.** `aten::take` raises `RuntimeError: take():
#     Expected a long tensor for index, but got Int` for an int32 index, where
#     `aten::index_select` accepts both widths -- so this op has its own
#     predicate rather than the sibling's, and an int32 index takes the
#     round-trip to that error.
#   * **The index may be any rank**, and the result takes the index's own shape:
#     a `(2, 2)` index returns a `(2, 2)` result, where `index_select` rejects a
#     non-vector index outright. The embedding body's flattening is therefore
#     correct *here* and wrong there, which is why the two are separate bodies
#     rather than one with a flag.
#   * **A negative index is in range.** Measured, `take(t, [-1])` reads the last
#     element where `index_select(t, 0, [-1])` raises, so the range check below is
#     written against the wrapped value and not against zero -- a check that
#     copied the sibling's would decline `-1` and cost a round-trip on a call the
#     graph answers exactly.
#
# The range check itself is over the *flattened* operand (`self.numel()` rather
# than `self.size(0)`), and it is the same defence its siblings carry: MPSGraph's
# gather has no range contract and returns the operand's first element for a bad
# index instead of failing, where the reference raises
# `IndexError: out of range: tried to access index 20 on a tensor of 20 elements`.
T_TAKE = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& index) {{
  if (!mps::{supports}(self.scalar_type(), index.scalar_type())) {{
    return at::{at_op}(self.cpu(), index.cpu()).to(self.device());
  }}
  // One host pass over the index before a node exists, resolving a negative
  // entry against the operand the way the reference does (measured: -1 reads the
  // last element, it does not raise). Anything the operand cannot answer takes
  // the round-trip so the caller sees the reference's own IndexError rather than
  // MPSGraph's silent first element.
  at::Tensor index_c = index.contiguous();
  const int64_t elements = self.numel();
  // `index_c` becomes the index the graph is handed, and that is the reason this
  // body rewrites it rather than only reading it.
  if (index_c.numel() > 0) {{
    const at::Tensor host = index_c.cpu();
    const int64_t* as_long = host.const_data_ptr<int64_t>();
    at::Tensor resolved_host = at::empty_like(host);
    int64_t* resolved = resolved_host.mutable_data_ptr<int64_t>();
    bool in_range = true;
    for (int64_t i = 0; i < host.numel(); ++i) {{
      // The int64-only predicate above is what makes this read safe.
      const int64_t at = as_long[i];
      const int64_t positive = at < 0 ? at + elements : at;
      if (positive < 0 || positive >= elements) {{
        in_range = false;
        break;
      }}
      resolved[i] = positive;
    }}
    // A *negative* index is in range on this op -- measured, `-1` reads the last
    // element where `index_select` raises -- and MPSGraph's gather does not
    // resolve it: measured through this backend, `take(t, [-1, 0, 19])` over a
    // 20-element operand returned `[0, 0, 19]` against the reference's
    // `[19, 0, 19]`, the graph having read a negative index as the *first*
    // element. So the body normalises rather than describes: the range check and
    // the wrapping are one pass, and the graph is only ever handed a
    // non-negative index. The first version of this body checked the range and
    // handed the original through, which answered every negative index with a
    // plausible wrong value -- the failure mode the check was there to prevent.
    if (!in_range) {{
      return at::{at_op}(self.cpu(), index.cpu()).to(self.device());
    }}
    index_c = resolved_host.to(index_c.device(), /*non_blocking=*/true);
  }}
  at::Tensor self_c = self.contiguous();
  // The result is the index's own shape -- not the operand's with an axis
  // replaced, which is what the two sibling bodies build. A scalar (rank-0)
  // index is not a shape `take` accepts, so there is no rank-0 case here.
  at::Tensor out = at::empty(index.sizes(), self_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  // The operand is described to the graph as one element per row, `(numel, 1)`,
  // and the index on the way in as a flat list, `(K)`. Measured on the SDK in
  // use, that pair gives `(K, 1)` for a `(K)` index and, for an index of any
  // other shape, the index's own shape carrying a trailing 1 -- `(2, 2)` gives
  // `(2, 2, 1)`, `(2, 1, 2)` gives `(2, 1, 2, 1)`. So one reshape of the gather's
  // result to the index's own shape lands on `out`, and `take`'s whole contract
  // falls out of the two nodes: the operand's structure is discarded because
  // `take` indexes a flat buffer, and the result's shape is the index's because
  // that is what the second reshape says.
  //
  // The three rescribings that do *not* work, recorded because each looks like
  // it should. `(1, numel)` with a `(K)` index gives `(K, numel)` -- the index
  // takes a whole row, so the result is rank 2 of the wrong size, which is what
  // the first version of this body shipped and MPSGraph rejected at compile time
  // ("the result shape is not compatible with the input shape") because the
  // index's runtime shape overrides the placeholder's. A flat `(numel)` operand
  // is the opposite mistake: the gather reads a `(K)` index against it as
  // selecting whole rows of a rank-1 tensor, which is a *different* answer from
  // selecting elements, and it happens to produce the right shape. And
  // reshape *after* the gather to the index's shape only works while the
  // operand's trailing 1 is still there to drop -- which is why the second
  // reshape needs the gathered shape above and not the index's shape alone.
  const int64_t element_count = self_c.numel();
  const int64_t index_count = index_c.numel();
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{self_c, index_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        MPSGraphTensor* column = [graph reshapeTensor:args[0]
                                            withShape:mps::MpsShapeOf(
                                                          {{element_count, 1}})
                                                 name:nil];
        MPSGraphTensor* flat_index = [graph reshapeTensor:args[1]
                                                withShape:mps::MpsShapeOf(
                                                              {{index_count}})
                                                     name:nil];
        MPSGraphTensor* gathered = [graph gatherWithUpdatesTensor:column
                                                    indicesTensor:flat_index
                                                             axis:0
                                                  batchDimensions:0
                                                             name:nil];
        return [graph reshapeTensor:gathered
                          withShape:mps::MpsShapeOf(index_c.sizes())
                               name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu(), index.cpu()).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# `div` closes the elementwise arithmetic set -- `add`, `sub` and `mul` have been
# served since stage 2 and this is the fourth -- so the least interesting thing
# about it is the body, which is `T_BINARY` with one selector changed. What is
# worth recording is the three ways it is *not* the binary template.
#
# **The selector.** MPSGraph offers `divisionWithPrimaryTensor:secondaryTensor:`
# and a `reciprocalWithTensor:`, and a port that reached for the reciprocal (the
# usual trick on hardware where division is slow) would be building
# `a * reciprocal(b)`. Measured on an M5 against `at::div` over the cases that
# separate the two readings -- a zero denominator (both signs), a signed zero
# numerator and denominator, an inf/nan pair, and the smallest denormals -- the
# two graphs return bit-identical results *to each other and to the reference*
# on every case except one, where both differ together (below). So the
# reciprocal is not a different answer, it is a second node, and the body takes
# the one node.
#
# **The subnormal the graph does not keep.** This is the one measured difference
# from the reference, and it is not a division property. Measured on an M5
# through this backend: a subnormal *result* comes back as zero -- `1e-38 / 2`
# is `5.0e-39` on the reference and `0` from the graph, while `2.35e-38 / 2` is
# the normal `1.175494e-38` from both -- and the same through every spelling
# (`div.Tensor` with a tensor or a folded constant, `div.Scalar`, and the
# reciprocal, which returns `inf` for `1 / 1.17e-38` where the reference reaches
# `8.5e37`). `mul.Tensor` flushes identically and has been routed to `mps` since
# stage 2 (`1e-38 * 0.5` -> `0`), as does `add.Tensor` (`1e-39 + 1e-39` -> `0`).
# So the choice here is "div behaves like the mul and add the branch already
# serves" or "div is the only arithmetic op on the device with a different
# subnormal policy from its siblings", and the first is both the smaller surprise
# and the one already shipped. The gate does not pretend otherwise --
# MpsSupportsDivDtype's note records the measurement and test_mps_div.py pins it
# as intentional beside the `mul` case it shares -- so a future reader meets it
# as a known property rather than by bisecting a model. Nothing in a
# transformer's attention or normalization path evaluates a division whose result
# is subnormal -- the values there are O(1) -- so this is a boundary the branch
# documents rather than one it meets.
#
# **The `mode` overloads are a different body, not a keyword.** `div.Tensor_mode`
# and `div.Scalar_mode` carry a `rounding_mode` whose three values are `floor`,
# `trunc` and the absent one; floor and trunc both need a `floor` node MPSGraph
# has, but they need it *after* the division and they need the division to be
# exact first, which is not what a single node does. That is a composition, and
# a composition is only allowed here where it was measured -- so those two, and
# the four `.out` spellings with them, take the round-trip.
#
# The dtype gate is a predicate over two operands because that is the form the
# tensor overload needs, and the scalar overload takes one of its own shape --
# `MpsSupportsDivScalarDtype`, not the shared `MpsSupportsScalarDtype` the other
# scalar bodies use. The shared one is dtype-preserving by construction, which is
# right for `mul`/`add`/`sub` and wrong here: `div(int, 2)` is a *float* tensor on
# the reference, so an integral operand takes the round-trip rather than the
# graph's truncated int64. On a float operand the divisor is then free, and a
# fractional one is served like any other.
T_DIV = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
  if (!mps::{supports}(self.scalar_type(), other.scalar_type())) {{
    return at::{at_op}(self.cpu(), other.cpu()).to(self.device());
  }}
  at::Tensor self_c = self.contiguous();
  at::Tensor other_c = other.contiguous();
  std::vector<int64_t> out_shape;
  try {{
    out_shape = at::infer_size(self_c.sizes(), other_c.sizes());
  }} catch (const std::exception&) {{
    return at::{at_op}(self.cpu(), other.cpu()).to(self.device());
  }}
  at::Tensor out = at::empty(out_shape, self_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{self_c, other_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        return [graph {call}args[0] secondaryTensor:args[1] name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu(), other.cpu()).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# The elementwise body over a tensor and a host scalar, for `div.Scalar`: the
# divisor becomes a one-element constant in the operand's own element type and
# the graph broadcasts it. Same shape as T_SCALAR, and the same reason for the
# narrower gate -- `x / 2` on a float tensor is a float tensor, but `x / 2` on
# an int tensor is a float tensor on the reference, which a constant of the
# operand's dtype cannot express, so an integral operand takes the round-trip
# unless the divisor is integral too.
#
# The selector is the two-operand division node rather than the reciprocal,
# exactly as in T_DIV, so a `div_scalar` body and a `div` body over the same
# values build the same node.
T_DIV_SCALAR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  if (!mps::{supports}(self.scalar_type(), other)) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
  at::Tensor self_c = self.contiguous();
  at::Tensor out = at::empty(self_c.sizes(), self_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{self_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        MPSGraphTensor* other_t =
            mps::MpsConstant(graph, other, self_c.scalar_type());
        if (other_t == nil) {{
          return nil;
        }}
        return [graph {call}args[0] secondaryTensor:other_t name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# `reciprocal` is `1 / x`, and this body is the unary template with the division
# node in it. The two things worth recording:
#
#   * **It is the same family, not a separate question.** MPSGraph has
#     `reciprocalWithTensor:`, which is a node of its own, but it is the same
#     arithmetic and the same subnormal policy as `divisionWithPrimaryTensor:`
#     (both measured on an M5 against the reference over the zero, signed-zero,
#     inf, nan and denormal cases). Serving `div` and not this would leave
#     `1 / x` -- which is what `torch.reciprocal` is, and what a caller who wants
#     the reciprocal *writes* -- on the CPU round-trip for no reason, so the two
#     land together.
#   * **The dtype gate is the div one, not the unary one.** It would be easy to
#     give this op the unary predicate, which is wider (it serves the int widths
#     as well). It must not: `x / 2` and `1 / x` are the same arithmetic, and two
#     ops built on one node family that disagree about which dtypes they serve is
#     a divergence someone would later have to prove correct rather than one a
#     reader can see. So the gate is named for the family.
T_RECIPROCAL = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!mps::{supports}(self.scalar_type(), self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  at::Tensor self_c = self.contiguous();
  at::Tensor out = at::empty(self_c.sizes(), self_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{self_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        return [graph {call}args[0] name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

CATEGORIES = {
    "unary": T_UNARY,
    "binary": T_BINARY,
    "binary_scalar": T_SCALAR,
    "reduce": T_REDUCE,
    "gemm": T_GEMM,
    "softmax": T_SOFTMAX_BODY,
    "pow_scalar": T_POW,
    "conv": T_CONV,
    "silu": T_SILU,
    "embedding": T_EMBEDDING,
    "index_select": T_INDEX_SELECT,
    "take": T_TAKE,
    "div": T_DIV,
    "div_scalar": T_DIV_SCALAR,
    "reciprocal": T_RECIPROCAL,
}

FILE_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen/codegen_mps.py -- DO NOT EDIT.
//
// MPSGraph kernels for the Apple Metal backend. Each kernel gates on dtype and
// hands one graph to mps::MpsRunGraph, which describes the aten tensors to
// MPSGraph against the runtime's shared MTLBuffers, schedules the graph on the
// current flagos stream's command queue and returns the output it allocated; a
// case the graph cannot answer exactly takes the CPU round-trip. The graph
// itself is compiled once per (body, operand shapes, dtypes and layouts) and
// reused, so the per-call cost is a schedule rather than a graph construction;
// the plan cache that does it is in mps_common.mm. Dispatchers are declared in
// generated/ops.h (shared with the CUDA codegen); here we only fill the
// Backend::kMps slot.
//
// Nothing below re-tags a tensor onto DispatchKey::MPS or CUDA: the tensors stay
// on PrivateUse1 and the runtime's shared MTLBuffers are described to MPSGraph
// directly (see mps_common.h). The CUDA-boxing path does not apply to this
// platform -- FLAGOS_BUILD_BOXING is pinned off in cmake/flagos_platforms.json --
// and the kMps slot is a plain one: Dispatcher::GetFn does not degrade it to a
// CUDA kernel when it is empty, so an op named here always has a body behind it.
//
// Whatever routes here without a kernel falls through to the CPU fallback: the
// conf only names this backend for ops in mps_register.inc.
//
// This file is Objective-C++ and builds without ARC, like the Metal runtime.

#include "../../../generated/ops.h"
#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/_softmax.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/add.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/div.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/take.h>
#include <c10/core/ScalarType.h>
#include <exception>
#include <optional>
#include <vector>
#include "../mps_common.h"

namespace at::native::flagos {
// The helpers live in the nested at::native::flagos::mps namespace, so an
// unqualified ``mps::Foo`` below already resolves there -- no alias, which
// would collide with the namespace name itself.

"""

FILE_FOOTER = "\n} // namespace at::native::flagos\n"

INC_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen/codegen_mps.py -- DO NOT EDIT.
//
// m.impl() lines for the ops that have an MPSGraph kernel. Included by
// register.cc inside TORCH_LIBRARY_IMPL(aten, PrivateUse1) when USE_MPS is set,
// in place of the full generated/register.inc list: an op registered on
// PrivateUse1 without a kernel behind it fails the dispatcher's
// "backend not registered" check, whereas an unregistered op simply reaches the
// cpu_fallback. So this file is exactly the Apple backend's coverage set.
//
// It is also what scripts/codegen/gen_vendor_confs.py reads to decide which ops
// backends_mps.conf may name for this backend, so the conf cannot claim a route
// this file does not register.

"""


def wrapper_map():
    """op name -> Wrapper<Name>, read back from the CUDA codegen's register.inc
    so the m.impl() subset we emit cannot drift from the wrappers that exist."""
    if not REGISTER_INC.exists():
        return {}
    return dict(
        re.findall(
            r'^\s*m\.impl\("([^"]+)",\s*(\w+)\);',
            REGISTER_INC.read_text(),
            flags=re.MULTILINE,
        )
    )


def render():
    """(kernels.mm text, register.inc text, [(op, category)]) for the whole op
    set. Raises on an op with no wrapper, so a stale register.inc fails loudly
    instead of emitting a file that will not link."""
    wrappers = wrapper_map()
    bodies = []
    covered = []
    for op, category in OPS.items():
        wrapper = wrappers.get(op)
        if wrapper is None:
            raise SystemExit(
                f"error: {op} has no wrapper in {REGISTER_INC.relative_to(REPO)}; "
                "run scripts/codegen/codegen_ops.py first"
            )
        fn, disp = schema_to_cpp_name(op)
        kernel = fn[:-2] + "KernelMps"  # AbsFn -> AbsKernelMps
        # alpha's three pieces for the ops that carry one:
        # (declaration, fallback argument, factor on the secondary operand).
        alpha_param, alpha_arg, alpha_operand = ALPHA.get(op, ("", "", "1"))
        bodies.append(
            CATEGORIES[category].format(
                kernel=kernel,
                fn=fn,
                disp=disp,
                at_op=AT_OP[op],
                supports=SUPPORTS[category],
                call=CALLS[op],
                rank=RANK.get(op, 0),
                alpha_param=alpha_param,
                alpha_arg=alpha_arg,
                alpha_operand=alpha_operand,
                finish=FINISH.get(op, ""),
                key_parts=KEY_PARTS[op],
                op=op,
                # The rank digit the conv body puts in the selector name. `dim`
                # is only read by the conv template, so a body that has no rank
                # to pick gets the 2-D name and never uses it.
                call2d=CALLS[op].replace("{dim}", "2"),
                call3d=CALLS[op].replace("{dim}", "3"),
                # The conv body's own reading of the optional bias, as a C++
                # literal. Every other template ignores it.
                has_bias="has_bias",
            )
        )
        covered.append((op, category))

    return (
        FILE_HEADER + "\n".join(bodies) + FILE_FOOTER,
        INC_HEADER
        + "".join(f'  m.impl("{op}", {wrappers[op]});\n' for op, _ in sorted(covered)),
        covered,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the committed artifacts are current instead of writing them",
    )
    args = ap.parse_args()

    mm_text, inc_text, covered = render()

    if args.check:
        stale = []
        for target, text in ((OUT_MM, mm_text), (OUT_INC, inc_text)):
            current = target.read_text() if target.exists() else ""
            if current != text:
                stale.append(target)
        if stale:
            for target in stale:
                print(f"[check] stale: {target.relative_to(REPO)}", file=sys.stderr)
            print(
                "error: run scripts/codegen/codegen_mps.py to regenerate",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"[check] {len(covered)} kernel(s) up to date")
        return

    OUT_MM.parent.mkdir(parents=True, exist_ok=True)
    OUT_MM.write_text(mm_text)
    OUT_INC.write_text(inc_text)

    print(f"[gen] {OUT_MM.relative_to(REPO)}  ({len(covered)} ops)")
    print(f"[gen] {OUT_INC.relative_to(REPO)}  ({len(covered)} m.impl lines)")
    for op, category in covered:
        print(f"       + {op} [{category}]")


if __name__ == "__main__":
    main()
