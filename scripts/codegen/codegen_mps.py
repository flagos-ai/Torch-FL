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
# and the `reduce` family -- plus three stage-3 increments: the `gemm` category
# (`mm` and `bmm`), the `softmax` category (`_softmax`) and the elementwise
# `neg` and `pow.Tensor_Scalar`, each matching the CPU reference within tolerance.
# The rest of stage 3 (conv) is still open.#
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
    # The activation `silu` is built from, and an op a model reaches for on its
    # own -- a hand-written attention mask, a gating projection, a `sigmoid` in a
    # loss. MPSGraph has a real selector (`sigmoidWithTensor:`) and it is the very
    # same one the `silu` composition above builds, so this increment is small,
    # and the interesting part is not the node but the width list: it is
    # *narrower* than `silu`'s even though the two bodies share the node. See
    # MpsSupportsSigmoidDtype for the measurement.
    #
    # Only `sigmoid` is registered. `sigmoid.out` and `sigmoid_` write into a
    # tensor the caller owns, like every other `.out`/inplace spelling on this
    # backend, and `sigmoid_backward` is a separate op with its own selector
    # question.
    "sigmoid": "sigmoid",
    # The largest elementwise unary left on a small model's fallback profile, and
    # the one the three ops after it are written in terms of: `log`, `rsqrt` and a
    # hand-written `tanh` are all built over an exponential, so serving this node
    # is the increment the rest of the cohort is measured against. Measured on an
    # M5 on a (3, 512, 3072) float32 operand, the `cpu_fallback` round-trip cost
    # 3.51 ms against a 1.03 ms device-to-host copy floor for the same shape --
    # the cheapest large win of the six unaries profiled, and the reason it is the
    # one this increment takes.
    #
    # MPSGraph has a real selector (`exponentWithTensor:`), so the body is the
    # plain unary template rather than a composition, and the interesting part is
    # the width list rather than the node -- see MpsSupportsExpDtype for the
    # measurement.
    #
    # Only `exp` is registered. `exp.out` and `exp_` write into a tensor the
    # caller owns, like every other `.out`/inplace spelling on this backend, and
    # `exp2`'s and `expm1`'s selectors are separate nodes with their own width
    # questions that this increment does not answer.
    "exp": "exp",
    # The inverse of the node above and the second-largest elementwise unary left
    # on a small model's fallback profile: measured on an M5 on a (3, 512, 3072)
    # float32 operand, the `cpu_fallback` round-trip cost 4.51 ms. A model reaches
    # it through a log-sum-exp, a KL divergence, a log-probability or a
    # normalising flow, and it is the op the design doc's stage-4 cohort pairs
    # with `exp`.
    #
    # MPSGraph names it directly (`logarithmWithTensor:`), so like `exp` the body
    # is the plain unary template and the increment is the width list. See
    # MpsSupportsLogDtype for the measurement -- the finding here is the *domain*
    # rather than the flush: a logarithm on a non-positive operand is `-inf`,
    # `0`/`-inf` or `nan` rather than a rounded magnitude, and the graph has to
    # reproduce each of those edges rather than a value near them.
    #
    # Only `log` is registered. `log.out` and `log_` write into a tensor the caller
    # owns, and `log2`, `log10` and `log1p` are separate ops with their own
    # selectors and their own edge behaviour.
    "log": "log",
    # The reciprocated square root, and the op a normalisation is written in
    # terms of: `x * rsqrt(v + eps)` is the tail of a layer norm, an RMS norm and
    # a batch norm, and the `1 / sqrt(d)` attention scale is the same shape.
    # Measured on an M5 on a (3, 512, 3072) float32 operand, the `cpu_fallback`
    # round-trip cost 3.703 ms against a 1.03 ms device-to-host copy floor for the
    # same shape.
    #
    # The node is the increment's constraint rather than its body: MPSGraph names
    # the fused `reciprocalSquareRootWithTensor:`, but that selector is macOS 15+
    # and the hosted `macos-14` MPS lane is on the other side of that line --
    # measured on the runner, the fused selector raises
    # `NSInvalidArgumentException: -[MPSGraph reciprocalSquareRootWithTensor:
    # name:]: unrecognized selector` and aborts the process. So the body is the
    # two-node composition `reciprocal(sqrt(x))`, built from the two selectors the
    # `reciprocal` kernel and the `layer_norm` rstd already use, which is the same
    # arithmetic at one extra node. See T_RSQRT.
    #
    # The gate is `MpsSupportsRsqrtDtype`, and the note there is the measurement:
    # this is a composition rather than one node, so both halves' subnormal
    # policy applies -- the sqrt of a subnormal, and the reciprocal of a subnormal
    # result -- and the increment's finding is which of those the platform gets
    # right at which width.
    #
    # Only `rsqrt` is registered. `rsqrt.out` and `rsqrt_` write into a tensor the
    # caller owns, like every other `.out`/inplace spelling on this backend, and
    # `_foreach_rsqrt` is the batched spelling with its own list contract.
    "rsqrt": "rsqrt",
    # The square root on its own, and the node the `rsqrt` composition above
    # already builds: `squareRootWithTensor:` is the inner half of
    # `reciprocal(sqrt(x))`, so serving it standalone adds no new arithmetic and
    # no selector this tree has not already measured. What it adds is a caller
    # who wants the root rather than its reciprocal -- a distance, a standard
    # deviation, a magnitude, or the `1 / sqrt(d)` a model writes out in two
    # steps -- and every one of those takes the `cpu_fallback` today because
    # `sqrt` is routed to `none` in the conf.
    #
    # The gate is `MpsSupportsSqrtDtype`, and the interest is that the node is
    # *not* the composition: the `rsqrt` finding split between a sqrt input half
    # and a reciprocal result half, and this body is the input half alone, so
    # whether a width is served here is a question the sibling's gate answered
    # only as part of a product. See the note there for the per-width numbers.
    #
    # Only `sqrt` is registered, exactly as `rsqrt` is: `sqrt.out` and `sqrt_`
    # write into a tensor the caller owns, like every other `.out`/inplace
    # spelling on this backend, and `_foreach_sqrt` is the batched spelling with
    # its own list contract.
    "sqrt": "sqrt",
    # The activation a transformer block reaches for at both ends -- the tanh
    # approximation inside a GELU and the one inside a recurrent or attention
    # gate -- and the reason it is the next increment after the unaries rather
    # than one of them.
    #
    # MPSGraph names `tanhWithTensor:` directly, so the body is the plain
    # one-node unary template plus one guard, and the increment is the *gate*:
    # measured at all three widths the node saturates a NaN operand to +-1
    # rather than propagating it, so `isNaNWithTensor:` selects a NaN back over
    # the node's answer -- see T_TANH. With that guard in place the gate is a
    # node-fidelity question, and it is the first one here to decline the
    # *widest* float: the node is the reference's own function at float16 and
    # bfloat16 but a coarser approximation of a tanh at float32, so the float32
    # caller keeps the round-trip. See MpsSupportsTanhDtype for the measurement.
    #
    # Only `tanh` is registered. `tanh.out` and `tanh_` write into a tensor the
    # caller owns, like every other `.out`/inplace spelling on this backend, and
    # `atanh` is a different op with its own domain -- `|x| < 1` is where it
    # answers a finite value, with its own edges at `+-1` and beyond -- so it is
    # not this increment.
    "tanh": "tanh",
    # The error function, which the explicit GELU form writes out by hand:
    # `0.5 * x * (1 + erf(x / sqrt(2)))`. `F.gelu`'s default path does not
    # dispatch `erf` on this host, so a `gelu` kernel is a separate gate. It is
    # also the kernel a diffusion sampler and an attention kernel built over erf
    # reach directly.
    #
    # MPSGraph names it directly (`erfWithTensor:`), so the body is the plain
    # one-node unary template and the increment is the *gate*: on an M5 the node
    # is closer to the exact float32 curve than `at::erf` on 63.3% of the probe
    # points where their roundings differ, and its error is at most 1.64 steps
    # at 1.0 against the reference's 4.14. It can be farther by up to 0.98 of a
    # step where the reference error is material on that host. The CI runner
    # orders the closer points differently and its absolute error is larger, so
    # the test holds both hosts to five steps from the reference and checks that
    # float32 dispatches to MPS. See MpsSupportsErfDtype for the measurement and
    # for why the two serve different width lists.
    #
    # Only `erf` is registered. `erf.out` and `erf_` write into a tensor the
    # caller owns, like every other `.out`/inplace spelling on this backend, and
    # `erfc` is a different op with its own domain and its own tail behaviour --
    # `erfc(x)` is `1 - erf(x)` only to the precision `1 - erf(x)` can carry at a
    # large `x`, which is a separate measurement -- so it is not this increment.
    "erf": "erf",
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
    # The fourth spelling of the read family, and the first one whose argument
    # contract is *not* a narrowing of `index_select`'s. `gather` and
    # `index_select` read an axis the caller names, but everything else about
    # them differs, and each difference is measured:
    #
    #   * the index has the operand's **rank**, not rank 1, and its shape is the
    #     result's shape (`take`'s rule over an axis rather than over a flat
    #     buffer);
    #   * the shape must match the operand's everywhere except `dim` --
    #     `RuntimeError: Index tensor must have the same number of dimensions as
    #     input tensor` for a rank mismatch;
    #   * an out-of-range index is a `RuntimeError` naming the dimension
    #     (`index 99 is out of bounds for dimension 0 with size 5`) rather than
    #     `index_select`'s `IndexError`;
    #   * a **negative** index is out of range and raises, where `take` reads it
    #     and `index_select` raises its own way;
    #   * the index dtype list is int32 **and** int64, the wider of the two
    #     siblings and the one `index_select` has.
    #
    # That combination is what points at a different node rather than the
    # `gatherWithUpdatesTensor:` the other three bodies build. MPSGraph's
    # `gatherAlongAxis:withUpdatesTensor:indicesTensor:` is defined as exactly
    # `res[i_{0}..i_{axis-1}, i, i_{axis+1}..] = updates[..., indices[i], ...]`
    # with `indices.shape == updates.shape` except at `axis` and `res.shape ==
    # indices.shape` -- which is `aten::gather`'s contract stated in the SDK's
    # own words, and the three clauses the sibling node cannot express. Its cost
    # is the same: measured on an M5 over a `(151936, 1024)` float32 operand, a
    # one-row `gather` at dim=0 cost 42.0 ms through the fallback against 0.29 ms
    # for the node, and the fallback's cost is the operand rather than the
    # question -- the same whole-table `aten::_to_copy` the other three pay.
    "gather": "gather",
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
    # Stage 4 opens with the normalisation every transformer block calls once,
    # and it is the first op here that answers with more than one tensor. It is
    # one MPSGraph node rather than the `native_batch_norm` + `addcmul`
    # decomposition the CPU fallback performs -- see T_LAYER_NORM.
    "native_layer_norm": "layer_norm",
    # Two ops the runtime-conformance suite lists as `RUNTIME_OPS`, which is the
    # contract that every one of them reaches an accelerator backend on every
    # platform -- MPS was the one platform handing these two to cpu_fallback.
    #
    # `where.self` is the ternary select and MPSGraph has the node
    # (`selectWithPredicateTensor:`), so it is a native body rather than a
    # round-trip: see T_WHERE for the three reference checks the node does not
    # make.
    "where.self": "where",
    # `cat` is the variadic concatenation, and the node is
    # `concatTensors:dimension:`. See T_CAT for the shape and dtype contract the
    # node would otherwise answer with a broadcasted wrong answer.
    "cat": "cat",
    # The normalisation a convolutional block calls. Its inference path is one
    # `normalizationWithTensor:...` node fed by the *running* statistics; its
    # training path is declined, because it updates the running buffers in place
    # with the unbiased batch variance and no graph node writes a caller's
    # buffer. See T_BATCH_NORM.
    "native_batch_norm": "batch_norm",
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
    # The CPU reference the `sigmoid` body round-trips to on the width it
    # declines. `sigmoid` is its own aten kernel rather than a composition, so
    # the round-trip lands on the reference's own implementation and reports its
    # own values rather than a re-entrant composite.
    "sigmoid": "sigmoid",
    # The CPU reference the `exp` body round-trips to on the widths it declines.
    # `exp` is its own aten kernel rather than a composition, so the round-trip
    # lands on the reference's own implementation -- which is where the
    # reference's own promotion rule for an integral or bool operand lives, and
    # why the gate sends those widths here rather than evaluating them in the
    # graph.
    "exp": "exp",
    # The CPU reference the `log` body round-trips to on the widths it declines.
    # `log` is its own aten kernel rather than a composition, so the round-trip
    # lands on the reference's own implementation -- which is where its domain
    # edges and its integral/bool promotion rule live, and why the gate sends
    # those widths here rather than evaluating them in the graph.
    "log": "log",
    # The CPU reference the `rsqrt` body round-trips to on the widths it
    # declines. `rsqrt` is its own aten kernel rather than a composition on the
    # reference side, so the round-trip lands on the reference's own
    # implementation -- which is where its domain edges (`rsqrt(0) = inf`,
    # `rsqrt(-0.0) = -inf`, `rsqrt(-1) = nan`) and its integral/bool promotion
    # rule live, and why the gate sends those widths here rather than evaluating
    # them in the composition.
    "rsqrt": "rsqrt",
    # The CPU reference the `sqrt` body round-trips to on the widths it declines.
    # `sqrt` is its own aten kernel rather than a composition on the reference
    # side, so the round-trip lands on the reference's own implementation --
    # which is where its domain edges (`sqrt(-1) = nan`,
    # `sqrt(-0.0) = -0.0`) and its integral/bool promotion rule live, and why
    # the gate sends those widths here rather than evaluating them in the graph.
    "sqrt": "sqrt",
    # The CPU reference the `tanh` body round-trips to on the widths it declines.
    # `tanh` is its own aten kernel rather than a composition on the reference
    # build, which is where its saturation contract (`tanh(+-inf) = +-1`) and its
    # integral/bool promotion rule live, and why the gate sends those widths here
    # rather than evaluating them in the graph.
    "tanh": "tanh",
    # The CPU reference the `erf` body round-trips to on the widths it declines.
    # `erf` is its own aten kernel rather than a composition, which is where its
    # bounded range (`erf(+-inf) = +-1`), its signed zero and its integral/bool
    # promotion rule live, and why the gate sends those widths here rather than
    # evaluating them in the graph.
    "erf": "erf",
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
    # The CPU reference the `gather` body round-trips to on anything it declines.
    # It is `at::gather` and not `at::index_select` even though the two read the
    # same kind of axis: the reference's *errors* differ -- `gather` raises
    # `RuntimeError: index 99 is out of bounds for dimension 0 with size 5` where
    # `index_select` raises an `IndexError`, and it accepts a rank-matched index
    # that `index_select` rejects as not-a-vector -- so the round-trip has to land
    # on the same op the caller named or a declined call would answer a different
    # failure than the reference does.
    "gather": "gather",
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
    # The reference op, so the round-trip lands on the kernel that computes
    # `(x - mean) / sqrt(var + eps) * gamma + beta` and answers the same
    # (out, mean, rstd) triple this body does.
    "native_layer_norm": "native_layer_norm",
    # `where.self` is a real aten kernel rather than a decomposition, so the
    # round-trip lands on the reference's own `select` and reports its own
    # `where expected condition to be a boolean tensor` message for a condition
    # width this body declines.
    "where.self": "where",
    # `cat` is likewise its own kernel, and the round-trip is where the
    # reference's shape message (`Sizes of tensors must match except in dimension
    # {dim}`) and its empty-list error live.
    "cat": "cat",
    # `native_batch_norm` is its own aten kernel, so the round-trip lands on the
    # implementation that owns the in-place running update, the unbiased
    # correction and the inference statistics the body has to reproduce.
    "native_batch_norm": "native_batch_norm",
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
    # Deliberately *not* the sibling's predicate, even though the two bodies
    # build the same node. A width list belongs to the op and not to the node:
    # see MpsSupportsSigmoidDtype for the measurement that separates them.
    "sigmoid": "MpsSupportsSigmoidDtype",
    # Its own predicate rather than the sibling `silu`/`sigmoid` one, and the
    # measurement behind it is the increment's finding: an exponential is a
    # magnitude op, not a saturated one, so the subnormal band that cost the bare
    # `sigmoid` its float16 width is not where an `exp` lives. See
    # MpsSupportsExpDtype for the per-width numbers -- float32 and bfloat16 are
    # served and float16 is declined, which is *not* the sibling's list.
    "exp": "MpsSupportsExpDtype",
    # Its own predicate, and the measurement is about the domain rather than a
    # flush: a logarithm's edges (a zero operand, a negative operand, a subnormal
    # operand the platform might flush) are not the magnitudes the sibling
    # exponentials were gated on. See MpsSupportsLogDtype.
    "log": "MpsSupportsLogDtype",
    # Its own predicate, and the measurement is about a *composition* rather than
    # a node: `rsqrt` is `reciprocal(sqrt(x))` here, so two subnormal policies
    # apply where the siblings above had one -- the sqrt of a subnormal input, and
    # the reciprocal of a subnormal result -- and the increment's finding is that
    # the two halves fail at different widths. See MpsSupportsRsqrtDtype.
    "rsqrt": "MpsSupportsRsqrtDtype",
    # Its own predicate rather than a share with `rsqrt`'s, and the measurement
    # is what separates them: `rsqrt` is `reciprocal(sqrt(x))` here, so its gate
    # had to be right about two nodes, while this body is the sqrt alone -- the
    # *input* half of that composition with no reciprocal after it. A width that
    # the composition serves is not automatically served here (a float32 input
    # the platform flushes to zero cost the sibling its non-zero answer, and this
    # body answers zero for it rather than inf), and the widening is the
    # increment's finding. See MpsSupportsSqrtDtype.
    "sqrt": "MpsSupportsSqrtDtype",
    # Its own predicate, and the measurement is a *node fidelity* question rather
    # than a flush or a domain: `tanhWithTensor:`. See MpsSupportsTanhDtype for
    # the per-width numbers.
    "tanh": "MpsSupportsTanhDtype",
    # Its own predicate, and the measurement lands on the *opposite* width list
    # to the sibling `tanh`'s even though the two are the same kind of op: the
    # node is the reference's own function at all three widths here, so the
    # increment's finding is a last-ulp question rather than an accuracy one.
    # See MpsSupportsErfDtype.
    "erf": "MpsSupportsErfDtype",
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
    # `index_select`'s pair, and shared for the same reason: the two ops accept
    # the same tensor signature here -- a table indexed by an int32/int64 tensor
    # -- and the difference between them is in the *shape* and *range* contracts,
    # which the body checks on the host rather than in a dtype predicate. The
    # subject half is `MpsSupportsEmbeddingDtype`'s subject half rather than
    # `take`'s wider one, because `gather` reads along an axis of an N-D operand
    # and the body hands that operand to MPSGraph as an N-D placeholder, where
    # `take` flattens its operand to one column and needs no N-D element type.
    "gather": "MpsSupportsEmbeddingDtype",
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
    # The reduction family's wall, not a gate of its own: `layer_norm` reduces
    # over the trailing axes exactly as `mean` and `sum` do, so it serves the
    # float widths and declines an integer input the reference computes in
    # float -- one rule, named once.
    "layer_norm": "MpsSupportsReduceDtype",
    # `where`'s select takes both branches in one datatype and promotes nothing,
    # while the reference promotes `self`/`other` to a common dtype -- so the gate
    # is over the two branch widths, and the condition width is checked with
    # them: bool is the reference's modern condition, uint8 the deprecated one
    # the body widens by comparing to zero, and any other width is the
    # reference's own error. See T_WHERE.
    "where": "MpsSupportsWhereDtype",
    # `cat`'s node requires one datatype across the whole list, where the
    # reference promotes a mixed list. The body checks the list is uniform and
    # this gate checks the one width is carryable, so a promoted or uncarryable
    # list takes the round-trip. See T_CAT.
    "cat": "MpsSupportsCatDtype",
    # The reduction family's wall again, and for the same reason `layer_norm`'s
    # is: `native_batch_norm` normalises over the per-channel axes exactly as
    # `mean` and `sum` reduce, and the reference computes an integral input in
    # float. A graph at the operand's own width cannot express that, so an
    # integral input takes the round-trip rather than a wrong answer. See
    # T_BATCH_NORM.
    "batch_norm": "MpsSupportsReduceDtype",
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
    # The same selector the `silu` body above names as its first node, here as the
    # whole body -- `sigmoid` *is* that node. Named separately because the two
    # categories carry different dtype gates and different fallbacks.
    "sigmoid": "sigmoidWithTensor:",
    # Its own selector rather than a composition, and the reason `exp` is one
    # node where a hand-written `tanh` would be two: MPSGraph names the
    # exponential directly. The node is the whole body, so the width list is the
    # only thing the increment decides.
    "exp": "exponentWithTensor:",
    # Its own selector, named directly like the exponential's, and the reason `log`
    # is one node rather than the `log2`/`loge` composition a hand-written body
    # would need.
    "log": "logarithmWithTensor:",
    # The *second* node of the two-node composition, and the reason the
    # first is named separately in T_RSQRT: `squareRootWithTensor:` builds the
    # inner node and this selector reciprocates it. The body writes both out in
    # full, so this entry is a record of the outer half rather than a `{call}`
    # prefix -- see the note there for why the fused
    # `reciprocalSquareRootWithTensor:` is not used.
    "rsqrt": "reciprocalWithTensor:",
    # The selector the `rsqrt` composition above already names as its inner node,
    # here as the whole body -- `sqrt` *is* that node. Named separately because
    # the two categories carry different dtype gates, which is the increment.
    "sqrt": "squareRootWithTensor:",
    # Named directly by the eval path, so the body is the plain unary template.
    # See MpsSupportsTanhDtype for whether the node rounds like a true tanh at a
    # width.
    "tanh": "tanhWithTensor:",
    # Named directly, exactly as the sibling `tanh` is. The node is the whole
    # body -- it is the reference's own function at every width -- so the width
    # list is the only thing the increment decides. See MpsSupportsErfDtype.
    "erf": "erfWithTensor:",
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
    # The one body here whose selector is not the sibling's, and the reason is
    # the contract rather than the operand. `gatherAlongAxis:` takes the operand
    # whole -- no row-pitch reshape, because it reads an axis of an N-D tensor
    # rather than rows of a 2-D one -- and it carries `indices.shape ==
    # updates.shape` except at `axis` plus `res.shape == indices.shape` as its own
    # shape rule, which is `aten::gather`'s. See the op-list note above.
    #
    # The axis binds to the *first* argument of this selector, which is the
    # opposite of the sibling's, so `{call}` is not a prefix of the expression
    # here: T_GATHER writes the selector out in full and this entry exists for
    # the table's completeness.
    "gather": "gatherAlongAxis:",
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
    # Unused: the layer_norm body builds its own node (`normalizationWithTensor:`)
    # and its two reductions, so it names none here. The entry exists because
    # `render` reads `CALLS[op]` for every op in OPS.
    "native_layer_norm": "",
    # Both bodies write the selector out in full -- `where` builds the predicate
    # comparison as well as the select, and `cat` passes the whole operand array
    # to a variadic selector -- so neither names a prefix here. The entries exist
    # because `render` reads `CALLS[op]` for every op in OPS.
    "where.self": "",
    "cat": "",
    # Unused: the batch_norm body builds its own node
    # (`normalizationWithTensor:...`) and its own `save_invstd` subgraph, so it
    # names no prefix here. The entry exists because `render` reads `CALLS[op]`
    # for every op in OPS.
    "native_batch_norm": "",
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
    # Empty for the same reason `silu`'s is: the graph is a function of the
    # operand's dtype and shape, which MpsRunGraph folds into the key itself, and
    # the body reads no other host value.
    "sigmoid": "",
    # Empty for the same reason `silu`'s and `sigmoid`'s are: the graph is one
    # selector over one operand, so it is a function of the operand's dtype and
    # shape -- which MpsRunGraph folds into the cache key itself -- and the body
    # reads no other host value. A body that bakes nothing into the graph has no
    # way to be handed a plan it did not build.
    "exp": "",
    # Empty, for the same reason `exp`'s is: one selector over one operand, so the
    # graph is a function of the operand's dtype and shape alone -- which
    # MpsRunGraph folds into the cache key itself -- and the body bakes no host
    # value into the graph.
    "log": "",
    # Empty for the same reason the three above are: the two nodes are a function
    # of the operand's dtype and shape, which MpsRunGraph folds into the cache key
    # itself, and the body bakes no host value into the graph.
    "rsqrt": "",
    # Empty for the same reason the four above are: the body is one selector over
    # one operand, so the graph is a function of the operand's dtype and shape --
    # which MpsRunGraph folds into the cache key itself -- and no host value is
    # baked into it.
    "sqrt": "",
    # Empty, for the same reason as `sqrt` above: one operand and one node.
    "tanh": "",
    # Empty, for the same reason as the six above it: the body is one selector
    # over one operand, so the graph is a function of the operand's dtype and
    # shape -- which MpsRunGraph folds into the cache key itself -- and no host
    # value is baked into it.
    "erf": "",
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
    # The resolved axis, and the row above that a reviewer has to check against
    # the body for the same reason `index_select`'s is: a (5, 4) operand read at
    # dim=0 and at dim=1 has one dtype and one pair of shapes and two different
    # answers, and `gatherAlongAxis:` bakes the axis into the compiled graph. A
    # body that resolved the axis and left it out of the key would hand the first
    # plan to the second call.
    #
    # `sparse_grad` is deliberately absent. It selects the *backward* pass's
    # gradient layout and does not change the forward result -- the same reading
    # the `embedding` row above takes of that op's backward arguments -- so the
    # body does not read it and two calls that differ only there share a plan.
    "gather": ", axis",
    # Empty: a division graph is a function of its operands' dtypes and shapes,
    # which MpsRunGraph folds in, and of nothing else. There is no `alpha` on
    # either spelling and no host scalar the body bakes in -- `div.Scalar`'s
    # scalar *is* an operand constant and is named there.
    "div.Tensor": "",
    "div.Scalar": ", other",
    "reciprocal": "",
    # Every host value the body bakes into the graph, and the row a reviewer has
    # to check against it: `normalized_shape` names the trailing run the
    # reductions cover (and so the mean/rstd result shapes, whose own signatures
    # are folded in below -- but the *axes* are not in any tensor), `eps` is the
    # constant added to the variance, and `has_weight` / `has_bias` say whether
    # the node is given a gamma and a beta at all. A body that read any of these
    # and left it out of the key would hand the first compiled plan to a call
    # that needed a different one.
    "native_layer_norm": ", normalized_shape, has_weight, has_bias, eps",
    # Empty: the graph is a select over three operands, and everything the body
    # decides is already in their signatures. The condition's width decides the
    # bool comparison, and MpsRunGraph folds every operand's dtype into the key,
    # so a bool and a uint8 condition already sit under different keys.
    "where.self": "",
    # The normalised axis, and the row a reviewer has to check against the body:
    # a (2, 3) list read at dim=0 and at dim=1 has one dtype and one set of
    # operand shapes and two different answers, and `concatTensors:dimension:`
    # bakes the axis into the compiled graph. A body that normalised the axis and
    # left it out of the key would hand the first plan to the second call.
    "cat": ", axis",
    # The one host value the batch_norm body bakes into the graph, and the row a
    # reviewer has to check against it: `eps` is the constant the node adds to
    # the running variance and the same constant the `save_invstd` subgraph
    # shifts by. The fixed `eps` it shadows is a non-issue -- a `fixed_eps` call
    # bakes its own value into the same key slot -- so two calls that differ only
    # in `eps` need two graphs.
    #
    # `rank` and `channels` are *not* named: they are the operands' own shapes,
    # which MpsRunGraph folds into the cache key, and the gamma/beta reshape is a
    # function of them.
    #
    # `has_weight` and `has_bias` are not named either, and that is the row worth
    # the reviewer's attention. The body always hands the node five operands --
    # an absent affine operand becomes an all-ones (gamma) or all-zeros (beta)
    # `[C]` tensor -- so a weight-less call and a weighted one differ in the
    # *values* of an operand and not in the operand list the key records. A plan
    # is compiled against the operands' dtypes and shapes and reads their
    # contents at run time, which is what makes substituting a constant tensor
    # for a missing one sound here and is also why the two calls can share a
    # graph.
    #
    # `momentum` is not read on this path at all: the graph covers inference
    # only, and `momentum` weights the running update that only the training path
    # performs. See T_BATCH_NORM.
    "native_batch_norm": ", eps",
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
#
# Every decline widens on the host when `half_to_float` is set rather than
# forwarding the flag to `at::_softmax`. The flag is only a CUDA spelling: the
# CPU kernel rejects it outright ("softmax with half to float conversion is not
# supported on CPU"), so handing it along turns a declined width into a raise
# instead of the reference's float answer. Widening the operand first and
# asking for the flag-clear result is the same thing the flag means -- compute
# in float, hand back float -- and it is what the graph path does by widening
# to `out_dtype`. The dtype gate above decides the widened result, so the two
# paths agree on the answer for every width that reaches them.
T_SOFTMAX_BODY = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float) {{
  at::ScalarType out_dtype = at::kFloat;
  int64_t axis = 0;
  if (!mps::MpsSoftmaxDim(self, dim, &axis) ||
      !mps::MpsSupportsSoftmaxDtype(self.scalar_type(), half_to_float, &out_dtype)) {{
    return mps::MpsSoftmaxOnHost(self, dim, half_to_float);
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
    return mps::MpsSoftmaxOnHost(self, dim, half_to_float);
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
# `convolution3D...` -- and no 1-D one. A 1-D convolution is served by the 2-D
# selector over a unit-high view of its input and kernel, with a reshape on each
# side of the node. Its padding list is `[front]` or `[front, back]` for the one
# axis, which `MpsConvResolve` folds into the `[0, front, 0, back]` list the
# descriptor reads; ranks above three take the CPU round-trip. What the
# descriptor carries is the whole geometry: NCHW data and OIHW weights,
# per-axis stride and dilation, `groups`, and an explicit padding style.
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
  // The spatial rank. A 1-D convolution is folded into the 2-D selector; a rank
  // above 3 has no graph and takes the round-trip.
  const int64_t dim = input.dim() - 2;
  const size_t ndim = dim > 0 ? static_cast<size_t>(dim) : 0;
  const bool padding_ok =
      (dim == 1 && (padding.size() == 1 || padding.size() == 2)) ||
      (dim > 1 && padding.size() == 2 * ndim);
  if (dim < 1 || dim > 3 || weight.dim() != input.dim() ||
      stride.size() != ndim || !padding_ok || dilation.size() != ndim ||
      groups < 1) {{
    return at::{at_op}(input.cpu(), weight.cpu(), bias, stride, padding, dilation,
                       transposed, output_padding, groups)
        .to(input.device());
  }}
  for (const int64_t entry : stride) {{
    if (entry <= 0) {{
      return at::{at_op}(input.cpu(), weight.cpu(), bias, stride, padding,
                         dilation, transposed, output_padding, groups)
          .to(input.device());
    }}
  }}
  for (const int64_t entry : dilation) {{
    if (entry <= 0) {{
      return at::{at_op}(input.cpu(), weight.cpu(), bias, stride, padding,
                         dilation, transposed, output_padding, groups)
          .to(input.device());
    }}
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
  const int64_t spatial = dim < 2 ? 2 : dim;
  std::vector<int64_t> source_shape(input_c.sizes().vec());
  std::vector<int64_t> weight_shape(weight_c.sizes().vec());
  std::vector<int64_t> result_shape(out.sizes().vec());
  if (dim == 1) {{
    source_shape.insert(source_shape.end() - 1, 1);
    weight_shape.insert(weight_shape.end() - 1, 1);
    result_shape.insert(result_shape.end() - 1, 1);
  }}
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
        MPSGraphTensor* source = args[0];
        MPSGraphTensor* kernel = args[1];
        if (dim == 1) {{
          source = mps::MpsReshape(graph, source, source_shape);
          kernel = mps::MpsReshape(graph, kernel, weight_shape);
          if (source == nil || kernel == nil) {{
            mps::MpsRelease(descriptor);
            return nil;
          }}
        }}
        MPSGraphTensor* y =
            (spatial == 2)
                ? [graph {call2d}source
                              weightsTensor:kernel
                                descriptor:(MPSGraphConvolution2DOpDescriptor*)descriptor
                                 name:nil]
                : [graph {call3d}source
                              weightsTensor:kernel
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
        if (y != nil && dim == 1) {{
          y = mps::MpsReshape(graph, y, result_shape);
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
# `sigmoid` -- the one node the `silu` composition above already builds, and the
# increment is the width list rather than the body. See MpsSupportsSigmoidDtype
# for the measurement: two ops can share a graph node and still not share a dtype
# gate, because what the node's rounding *means* depends on what the surrounding
# graph does with it. A `silu` tail is multiplied by its own subnormal x, so the
# platform's subnormal flush is what the reference does too; a bare sigmoid's
# tail has no such partner, so the same flush is a visible 3.1e-02 error and the
# width is declined.
T_SIGMOID = """\
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

# The one-node unary template for an op MPSGraph names directly -- `exp` today,
# with `log` sharing it because the two differ only in the selector and the width
# list. There is no composition to compose and no node to build by hand: the
# operand goes through `contiguous()` for the same reason every other body here
# does (the graph reads one flat row of `numel` elements and a strided operand
# would be read as the wrong bytes), and the whole increment is the gate the body
# opens with. Kept separate from T_UNARY rather than folded into it because the
# two carry different gates, which is the point of both increments; a future op
# that shares this body adds a CATEGORIES row pointing here.
T_EXP = """\
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

# `rsqrt` -- `1 / sqrt(x)`, the tail of every normalisation, and the second body
# here (after `silu`) built from two nodes rather than one.
#
# MPSGraph names the fused `reciprocalSquareRootWithTensor:`, and on a modern SDK
# that would be the whole body -- one node, one selector, exactly the arithmetic
# the reference performs. It is not usable here, and the reason is the *hosted
# runner* rather than the graph: the selector is macOS 15+, while the MPS lane
# this backend is verified on is a `macos-14` image (SDK 14.5, Xcode 15.4). The
# failure is not subtle, and it is why this was measured rather than assumed:
# built against the older SDK the fused selector is absent, and the first call
# that builds the node raises `NSInvalidArgumentException: -[MPSGraph
# reciprocalSquareRootWithTensor:name:]: unrecognized selector` -- an Objective-C
# exception from inside a `^` block with no handler, so the process aborts rather
# than the op declining. See the same note above T_LAYER_NORM's rstd node, which
# takes the identical decision for `1 / sqrt(var + eps)` for the identical
# reason.
#
# So the body composes the two selectors that *are* on both SDKs and are already
# in this tree: `squareRootWithTensor:` (the `sqrt` node) and
# `reciprocalWithTensor:` (the `reciprocal` kernel's node, and
# `layer_norm`'s rstd). Measured on an M5, the composition is the same answer as
# the fused node would have given -- the two are one rounding of the same
# arithmetic -- at one extra node in the graph, which a compiled plan amortises
# over every call that shares the cache key.
#
# The two halves are why the gate is its own predicate rather than a reuse: the
# subnormal policy that matters at the input is the *sqrt's*, and the one that
# matters at the result is the *reciprocal's*, so a width can be right on one
# half and wrong on the other. See MpsSupportsRsqrtDtype for the per-width
# measurement.
T_RSQRT = """\
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
        MPSGraphTensor* root = [graph squareRootWithTensor:args[0] name:nil];
        if (root == nil) {{
          return nil;
        }}
        return [graph reciprocalWithTensor:root name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

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

# `gather` is the fourth and last of the read family, and the one whose contract
# is neither a sibling's nor a superset of one. Where `take` reads a flat buffer
# with an index of any shape and `index_select` reads axis 0 with a vector index,
# `gather` reads the axis the caller names with an index of the operand's *rank*,
# and it is the shape rule that decides the body:
#
#   indices.shape == self.shape except at `dim`, and res.shape == indices.shape
#
# which is the same sentence the SDK writes for `gatherAlongAxis:` (measured
# against the reference and read out of the header):
#
#   res[p_0..p_{axis-1}, i, p_{axis+1}..p_{U-1}] =
#       updates[p_0..p_{axis-1}, indices[p_0..p_{axis-1}, i, p_{axis+1}..], ...]
#   P = Q, indices.shape[0:axis] == updates.shape[0:axis],
#   res.shape[0:axis] == updates.shape[0:axis], and so on around `axis`
#
# so the operand goes to the graph as itself -- no row-pitch reshape, because
# nothing here reads rows of a 2-D flattening -- and the result goes into an
# `out` allocated at the index's shape. That is the whole body; the rest of it is
# the three declines the *reference* forces, and each of them is a case where the
# node would answer plausibly rather than wrongly:
#
#   * **The index rank must *equal* the operand's, in both directions.** This is
#     the first contract that had to be measured rather than assumed, because the
#     obvious reading -- "a lower-rank index is broadcast up, the way the gathered
#     axis moves in other frameworks" -- is what the first draft of this body
#     implemented, and it was wrong. Measured: `gather((4, 5), 1, (5,))` raises
#     "Index tensor must have the same number of dimensions as input tensor", and
#     so does `gather((4, 5), 1, (4, 5, 1))`; the reference broadcasts nothing.
#     The `(1, 5)` shape that *is* accepted is not an expanded `(5,)` -- it is an
#     index that already has the operand's rank and happens to be one wide at a
#     non-axis dimension. So the body declines on `index.dim() != rank` and never
#     expands, and the two directions are one comparison rather than two.
#   * **A non-axis dimension may be smaller than the operand's, but not larger.**
#     This is the finding that decides the body. Measured: `gather((4, 5), 1,
#     (4, 3))` is `(4, 3)` and `gather((4, 5), 0, (2, 3))` is `(2, 3)`, but
#     `gather((4, 5), 1, (4, 7))` raises "Size does not match at dimension 0
#     expected index [4, 7] to be no larger than self [4, 5] apart from dimension
#     1" -- the reference checks the non-axis dimensions *one-sidedly*, and it
#     does not check them the way `gatherAlongAxis:` does at all: the node
#     requires "updates shape and indices shape must match except at axis", and
#     the first version of this body handed the operand as-is, which made the
#     MPSGraph compiler abort the process with `error: 'mps.gather_along_axis' op
#     invalid input tensor shapes, updates shape and indices shape must match
#     except at axis` -- a hard abort, not a decline the body could catch.
#
# The shape the node wants is reachable from the shape the reference accepts by
# **narrowing the operand's non-axis dimensions to the index's extents before the
# gather**: `gather((4, 5), 0, (3, 1))` reads `self[:, :1]` and gathers at axis 0,
# and a rank-3 `(2, 3, 4)` operand read at axis 1 with a `(1, 2, 4)` index reads
# `self[:1, :, :4]` and gathers at axis 1. The narrowing is exactly the sub-block
# the reference's own element mapping touches -- it writes `res[i, j] =
# self[index[i, j], j]`, so dimension 0 varies over `index.size(0)` and dimension 1
# over `index.size(1)` and no other element is read -- which is why this trims the
# operand rather than broadcasting it: enlarging a dimension would invent operand
# rows the reference never reads, and the answer would be wrong rather than merely
# slower.
#
# The trim is one-sided, and the asymmetry is the reference's rather than the
# node's: a dimension the index makes *larger* than the operand's is a
# `RuntimeError` from the reference, so the body declines it before the trim
# rather than resizing anything. That is exactly the shape the node *would*
# answer -- measured, `gatherAlongAxis:` compiles an `updates(4, 5)` against an
# `indices(4, 7)` and returns a `4x7` -- which is why the check has to be here and
# not left to the graph: the node's answer would be a plausible tensor where the
# caller is owed an error, and the error is the whole contract for that shape.
#
# The narrowing is done on the **host** with `at::` (a `narrow` per non-axis
# dimension, materialised by the `contiguous()` the body already takes) rather
# than with MPSGraph's `sliceTensor:`. That was measured, not assumed: the graph
# slice produces a tensor of the right *shape* -- `slice((4, 5), (4, 1))` is
# `4x1` and the gather over it is `3x1` -- but the backend read the wrong element
# for it, `gather((4, 5), 0, (3, 1))` returning `[0, 10, 0]` against the
# reference's `[0, 10, 15]`. The sliced tensor and the view the runtime feeds it
# do not agree on the row pitch, which is the same MPSNDArray layout constraint
# the row-pitch note in mps_common.h records for the sibling bodies; the host
# narrowing sidesteps it by handing the graph a buffer that already is the block
# (and costs one copy of the block, which is the operand's *sub-block* rather than
# the operand).
#
# The two other readings were measured and rejected. Moving the gathered axis to
# the front and transposing back is the natural first idea and it works where the
# *axis* dimension is the one that differs, but it is the identity permutation
# when `axis == 0`, which is the most common call, so it fixes nothing there.
# Flattening the operand to `(size(axis), -1)` and gathering flat rows gets the
# shape rule and loses the element rule as soon as a dimension before `axis`
# differs, because the flat row index no longer decomposes the way the
# reference's does.
#   * **An out-of-range index must raise, and the node inserts a zero.** The
#     header says so outright ("If an index is out of bounds of the updatesTensor
#     along axis a 0 is inserted"), and the reference raises `RuntimeError: index
#     99 is out of bounds for dimension 0 with size 5`. So the range check runs on
#     the host before a node exists, against `self.size(axis)` -- the *axis's*
#     extent, not `numel`, which is the mistake this op makes easy to make in
#     both directions: an index of `numel - 1` is in range for `take` over the
#     same tensor and out of range here.
#   * **A negative index is out of range.** Measured: the reference raises `index
#     -1 is out of bounds for dimension 0 with size 5` where `take` reads the last
#     element, so a body that reused `take`'s normalisation would wrap an index
#     here and answer a call the reference raises -- the same failure class, in
#     the opposite direction.
#
# `sparse_grad` is a backward-pass argument whose value the forward result does
# not depend on, so the body does not read it and `KEY_PARTS` does not name it --
# the same reading T_EMBEDDING takes of that op's three backward arguments, and
# the reason a `sparse_grad=True` and a `sparse_grad=False` call over the same
# operands share one plan.
T_GATHER = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad) {{
  if (!mps::{supports}(self.scalar_type(), index.scalar_type())) {{
    return at::{at_op}(self.cpu(), dim, index.cpu(), sparse_grad)
        .to(self.device());
  }}
  // `sparse_grad` is not read: it selects the backward pass's gradient layout
  // and the forward result does not depend on it (see the note above T_GATHER).
  const int64_t rank = self.dim();
  if (rank == 0) {{
    return at::{at_op}(self.cpu(), dim, index.cpu(), sparse_grad)
        .to(self.device());
  }}
  // The index must carry the operand's rank *exactly*, in both directions, and
  // this is the one contract a first reading gets wrong. A lower-rank index is
  // not broadcast up (`gather((4, 5), 1, (5,))` raises), and `(1, 5)` is not an
  // expanded `(5,)` -- it is an index that already has the operand's rank. So
  // the test is equality and both directions take the same round-trip; there is
  // no expansion here and nothing to reconcile, because `gatherAlongAxis:`
  // *would* have accepted and broadcast a mismatched index where the reference
  // raises `Index tensor must have the same number of dimensions as input
  // tensor`.
  if (index.dim() != rank) {{
    return at::{at_op}(self.cpu(), dim, index.cpu(), sparse_grad)
        .to(self.device());
  }}
  // Resolve the axis against the operand's rank the way aten does (measured:
  // dim=-1 over a rank-2 operand is dim=1), and leave a dim outside the rank to
  // the reference rather than reporting it here.
  int64_t axis = dim;
  if (axis < 0) {{
    axis += rank;
  }}
  if (axis < 0 || axis >= rank) {{
    return at::{at_op}(self.cpu(), dim, index.cpu(), sparse_grad)
        .to(self.device());
  }}
  // The index is materialised once, in its own width, and then read twice: the
  // host range pass below and the graph. A non-contiguous index is made dense
  // here, so both passes see the same buffer in the same order.
  const at::Tensor index_c = index.contiguous();
  // A non-axis dimension the index makes *larger* than the operand's is the
  // reference's one-sided error -- `gather((4, 5), 1, (4, 7))` raises "expected
  // index [4, 7] to be no larger than self [4, 5] apart from dimension 1" -- and
  // the node would answer it rather than raising (measured: `gatherAlongAxis:`
  // compiles that pair and returns a `4x7`). So the check runs here, and only in
  // this direction: a *smaller* non-axis dimension is legal and is what the
  // narrowing below is for.
  for (int64_t d = 0; d < rank; ++d) {{
    if (d != axis && index_c.size(d) > self.size(d)) {{
      return at::{at_op}(self.cpu(), dim, index.cpu(), sparse_grad)
          .to(self.device());
    }}
  }}
  // One host pass over the index, in its own width, before a node exists. Any
  // index the axis cannot answer takes the round-trip so the caller sees the
  // reference's own RuntimeError rather than the node's inserted zero.
  const int64_t axis_extent = self.size(axis);
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
    // Two-sided and *unwrapped*: a negative index is out of range here where
    // `take` wraps it, because the reference raises for it (measured: `index -1
    // is out of bounds for dimension 0 with size 5`).
    for (int64_t i = 0; i < host.numel(); ++i) {{
      if (as_long[i] < 0 || as_long[i] >= axis_extent) {{
        in_range = false;
        break;
      }}
    }}
  }}
  if (!in_range) {{
    return at::{at_op}(self.cpu(), dim, index.cpu(), sparse_grad)
        .to(self.device());
  }}
  // The operand is read non-const because its shape is described to the graph
  // below; a transposed or otherwise non-dense operand is declined by
  // MpsRunGraph's own buffer resolution rather than by the node.
  at::Tensor self_c = self.contiguous();
  // `gatherAlongAxis:` wants the operand and the index to agree on every
  // dimension except `axis`, where the reference lets the index be *smaller*
  // (the larger direction was declined above); and it binds the axis before the
  // operands, unlike the `gatherWithUpdatesTensor:` the three sibling bodies
  // use, so the template writes the selector out in full rather than going
  // through the `{call}` prefix. Trimming the operand's non-axis dimensions to
  // the index's extents is what reconciles the two shape rules -- see the note
  // above T_GATHER for why the other readings (broadcasting the operand, or
  // flattening it to rows) are wrong rather than merely slower. The trim is
  // `axis`-preserving, so it is one `narrow` per dimension that is not `axis`,
  // and it is done here on the host rather than with MPSGraph's `sliceTensor:`
  // because the graph slice produced the right *shape* and the wrong *element*
  // (the note above records the `[0, 10, 0]` against `[0, 10, 15]`
  // measurement): the sliced tensor and the view the runtime feeds it do not
  // agree on the row pitch. A `narrow` on the already-contiguous operand hands
  // the graph a buffer that already is the block, and costs a copy of the
  // operand's sub-block rather than of the operand.
  at::Tensor narrowed = self_c;
  for (int64_t d = 0; d < rank; ++d) {{
    if (d != axis && index_c.size(d) != self_c.size(d)) {{
      narrowed = narrowed.narrow(d, 0, index_c.size(d));
    }}
  }}
  // The narrow above is a *view*, and the graph reads its operands densely (see
  // the row-pitch note in mps_common.h), so the view has to be copied down
  // before it is described. Skipping this copy is not a slow answer but a wrong
  // one, and the wrong answer is specific: MpsRunGraph feeds the operand's own
  // buffer from its own offset with `numel` elements read densely, so the graph
  // sees the operand's *first* `numel` elements in row-major order rather than
  // the block the narrow selected. Measured here, `gather((4, 5), 0, (3, 1))`
  // over `arange(20)` read `[0, 1, 2]` where the reference reads `[0, 10, 15]`
  // -- the index values themselves, which is what makes it look like the
  // operands were swapped rather than truncated.
  if (narrowed.sizes() != self_c.sizes()) {{
    narrowed = narrowed.contiguous();
  }}
  // The result is the index's own shape, which is what the node returns and what
  // the reference returns.
  at::Tensor out = at::empty(index.sizes(), self_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{narrowed, index_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        return [graph gatherAlongAxis:(NSInteger)axis
                     withUpdatesTensor:args[0]
                         indicesTensor:args[1]
                                  name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(self.cpu(), dim, index.cpu(), sparse_grad)
        .to(self.device());
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

# `F.layer_norm` -- the normalisation a transformer calls once per block, right
# after the residual add -- and the first body here that answers with more than
# one tensor: `native_layer_norm(input, normalized_shape, weight, bias, eps)`
# returns `(out, mean, rstd)`.
#
# **It is one node, not the `native_batch_norm` + `addcmul` decomposition the
# design doc predicted.** That prediction came from reading the CPU fallback,
# and the fallback is what this op was taking: `F.layer_norm` on a device tensor
# round-tripped through the CPU, so the two ops the CPU build decomposes it into
# (`native_batch_norm` in inference mode, then `addcmul`) were what the fallback
# log showed. MPSGraph has the whole operation behind one selector,
# `normalizationWithTensor:meanTensor:varianceTensor:gammaTensor:betaTensor:
# epsilon:name:`, whose documented semantics are exactly aten's:
# `(x - mean) / sqrt(var + eps) * gamma + beta` over the trailing axes. The mean
# comes from `meanOfTensor:axes:` and the variance from
# `varianceOfTensor:axes:`, the two reduction selectors the header documents as
# the ones that feed the normalisation node.
#
# Three things decide whether the graph can answer a call, and each is the
# reference's contract rather than the graph's:
#
#   * **`normalized_shape` must be a trailing run of the input's axes.** aten
#     requires it (`normalized_shape` of `(8,)` on a `(2, 3, 8)` input is over
#     the last axis; `(3, 8)` over the last two; `(2, 3, 8)` over all three), and
#     a shape that is not the input's own trailing dimensions is a `RuntimeError`
#     naming the two. The body checks the suffix and declines otherwise, so the
#     reference raises in its own words.
#   * **The reduced axes are `[-len(normalized_shape), 0)`.** That is what the
#     "trailing run" sentence means, and it is the same axis list the reference
#     reduces over.
#   * **`mean` and `rstd` are the *keepdim* shape over those axes** -- `(2, 3,
#     1)` for the `(8,)` case -- not the reduced shape. Measured against the
#     reference: `native_layer_norm((2, 3, 8), [8], ...)` answers `mean` and
#     `rstd` both `(2, 3, 1)`. So the body allocates those two results with the
#     reduced axes already set to 1, which is also the shape MPSGraph's own
#     `meanOfTensor:axes:` returns (the named reduction keeps the rank and leaves
#     each reduced axis at size 1), so no reshape is needed to match it.
#
# `weight` and `bias` are the gamma and beta and are optional. When absent the
# node is handed `nil` for both, which is what its `_Nullable` annotations
# document, so a weight-less call is the same node with two feeds missing rather
# than a second graph shape -- hence the two `has_weight`/`has_bias` entries in
# the key (a bias-less call hands the graph one operand fewer, which the operand
# signature alone cannot say).
#
# The variance the node wants is the **population** variance (divided by N), not
# the sample one: measured on an M5, `varianceOfTensor:axes:` matches the
# reference's `unbiased=False` answer to the last digit and the reference's
# `layer_norm` uses `unbiased=False`. This is checked rather than assumed, in the
# integration suite, on a case whose sample and population variances differ.
#
# The dtype gate is the float-only one the reduction family uses: the reference
# accepts an integer input and computes in float, which a graph at the operand's
# own width cannot express, so an integral input takes the round-trip.
T_LAYER_NORM = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    double eps) {{
  const bool has_weight = weight.has_value() && weight->defined();
  const bool has_bias = bias.has_value() && bias->defined();
  // The CPU round-trip every decline below takes, in one place. Every other
  // body in this file ends its round trip with `.to(self.device())`; here all
  // three results have to be moved, and a body that returned the host tuple
  // would hand the caller values that are right on a device that is not -- the
  // `mean`/`rstd` shapes and the flagos device both belong to the contract.
  const auto round_trip = [&]() {{
    const at::Tensor host_weight = has_weight ? weight->cpu() : at::Tensor();
    const at::Tensor host_bias = has_bias ? bias->cpu() : at::Tensor();
    auto host = at::{at_op}(
        input.cpu(), normalized_shape, host_weight, host_bias, eps);
    return ::std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        std::get<0>(host).to(input.device()),
        std::get<1>(host).to(input.device()),
        std::get<2>(host).to(input.device()));
  }};
  if (!mps::{supports}(input.scalar_type(), std::nullopt)) {{
    return round_trip();
  }}
  // Every operand reaches the graph as a device buffer (see the operand
  // contract in mps_common.h): a weight or bias the caller left on the host --
  // `F.layer_norm(x, shape, w, b)` with `w`/`b` built on the host is the common
  // spelling -- is moved to the input's device rather than handed to the graph,
  // which would resolve no Metal buffer for it and silently produce a CPU
  // result. `input` is already on the device, so it is only the affine pair.
  const at::Tensor weight_d =
      has_weight && weight->device() != input.device()
      ? weight->to(input.device())
      : (has_weight ? *weight : at::Tensor());
  const at::Tensor bias_d =
      has_bias && bias->device() != input.device()
      ? bias->to(input.device())
      : (has_bias ? *bias : at::Tensor());
  const int64_t rank = input.dim();
  const int64_t normalized = static_cast<int64_t>(normalized_shape.size());
  // The suffix check, in the reference's own terms: the normalised shape has to
  // be the input's trailing axes, and a rank-0 input or a normalised shape
  // longer than the input is declined rather than described to the graph.
  if (normalized <= 0 || normalized > rank) {{
    return round_trip();
  }}
  for (int64_t i = 0; i < normalized; ++i) {{
    if (normalized_shape[normalized - 1 - i] != input.size(rank - 1 - i)) {{
      return round_trip();
    }}
  }}
  at::Tensor input_c = input.contiguous();
  if (input_c.numel() == 0) {{
    return round_trip();
  }}
  // The keepdim shape over the trailing run, which is what the reference
  // answers for mean and rstd and what the named reductions produce.
  std::vector<int64_t> stat_shape(input_c.sizes().vec());
  std::vector<int64_t> axes;
  for (int64_t axis = rank - normalized; axis < rank; ++axis) {{
    axes.push_back(axis);
    stat_shape[axis] = 1;
  }}
  at::Tensor out = at::empty(input_c.sizes(), input_c.options());
  at::Tensor mean = at::empty(stat_shape, input_c.options());
  at::Tensor rstd = at::empty(stat_shape, input_c.options());
  if (out.numel() == 0) {{
    return {{out, mean, rstd}};
  }}
  at::Tensor weight_c;
  at::Tensor bias_c;
  std::vector<at::Tensor> operands{{input_c}};
  if (has_weight) {{
    weight_c = weight_d.contiguous();
    operands.push_back(weight_c);
  }}
  if (has_bias) {{
    bias_c = bias_d.contiguous();
    operands.push_back(bias_c);
  }}
  const float epsilon = static_cast<float>(eps);
  std::vector<at::Tensor> outs{{out, mean, rstd}};
  const bool ran = mps::MpsRunGraphMulti(
      mps::MpsGraphKey("{op}"{key_parts}),
      operands,
      outs,
      ^std::vector<MPSGraphTensor*>(MPSGraph* graph,
                                    NSArray<MPSGraphTensor*>* args) {{
        MPSGraphTensor* x = args[0];
        MPSGraphTensor* avg =
            [graph meanOfTensor:x axes:mps::MpsAxesArray(axes) name:nil];
        if (avg == nil) {{
          return std::vector<MPSGraphTensor*>();
        }}
        // The two-argument variance, which takes the mean the node above
        // produced and is the spelling MPSGraph's own documentation pairs with
        // the normalisation node's `variance` parameter.
        MPSGraphTensor* var = [graph varianceOfTensor:x
                                           meanTensor:avg
                                                 axes:mps::MpsAxesArray(axes)
                                                 name:nil];
        if (var == nil) {{
          return std::vector<MPSGraphTensor*>();
        }}
        // rstd is 1 / sqrt(var + eps), built from `squareRootWithTensor:` and
        // `reciprocalWithTensor:` rather than the single fused
        // `reciprocalSquareRootWithTensor:` the composition suggests.
        //
        // The fused selector is `MPS_AVAILABLE_STARTING(macos(15.0))`, and the
        // hosted `macos-14` runner is the one lane that is *not* on 15 or
        // later. There the selector is absent from the runtime even though it
        // is absent from the SDK header too, so it compiles with only an
        // `-Wobjc-method-access` warning ("instance method not found, return
        // type defaults to 'id'") and then dies at the call rather than at the
        // build: the first case to reach the node terminates the process with
        // `NSInvalidArgumentException: -[MPSGraph reciprocalSquareRootWithTensor:
        // name:]: unrecognized selector`, which is the SIGABRT the MPS lane
        // reported as an operator-suite failure with no assertion in front of
        // it. The two nodes below are the macOS 11 spelling with no availability
        // gate, and `reciprocalWithTensor:` is the one the `reciprocal` kernel
        // in this same file already runs on.
        //
        // This is a portability constraint rather than a numerics choice: both
        // spellings compute 1/sqrt(x), and the composed form is what the older
        // MPSGraph the runner carries can actually execute.
        MPSGraphTensor* shifted =
            [graph additionWithPrimaryTensor:var
                             secondaryTensor:[graph constantWithScalar:epsilon
                                                              dataType:var.dataType]
                                        name:nil];
        if (shifted == nil) {{
          return std::vector<MPSGraphTensor*>();
        }}
        MPSGraphTensor* rooted = [graph squareRootWithTensor:shifted name:nil];
        if (rooted == nil) {{
          return std::vector<MPSGraphTensor*>();
        }}
        MPSGraphTensor* rstd_t = [graph reciprocalWithTensor:rooted name:nil];
        if (rstd_t == nil) {{
          return std::vector<MPSGraphTensor*>();
        }}
        MPSGraphTensor* gamma = {gamma};
        MPSGraphTensor* beta = {beta};
        MPSGraphTensor* y = [graph normalizationWithTensor:x
                                               meanTensor:avg
                                           varianceTensor:var
                                              gammaTensor:gamma
                                               betaTensor:beta
                                                  epsilon:epsilon
                                                     name:nil];
        if (y == nil) {{
          return std::vector<MPSGraphTensor*>();
        }}
        return std::vector<MPSGraphTensor*>{{y, avg, rstd_t}};
      }});
  if (!ran) {{
    return round_trip();
  }}
  return {{out, mean, rstd}};
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# `where.self(condition, self, other)` is MPSGraph's own ternary select node,
# `selectWithPredicateTensor:truePredicateTensor:falsePredicateTensor:name:`, and
# the node broadcasts, which is the contract the reference has as well.
#
# Three things about the reference decide what this body has to check before the
# node is built, and each of them was measured against `at::where.self`:
#
#   * The condition is bool-only in the modern reference. `where` accepts a
#     uint8 condition still -- emitting a deprecation warning -- and rejects
#     everything else itself (`where expected condition to be a boolean tensor,
#     but got a tensor with dtype Int`) for int8/int32/int64/float. The graph's
#     predicate is a bool tensor, so a uint8 condition is compared against zero
#     and the rest is declined so the reference raises its own message.
#
#   * `self` and `other` are *promoted* to a common dtype, which is the one thing
#     the graph cannot do: MPSGraph's select takes both branches in one type and
#     promotes nothing (int32 against int64 is int64 in aten, and half against
#     float is float). `MpsSupportsWhereDtype` admits the same pair
#     `MpsSupportsBinaryDtype` does and for the same reason -- equal dtypes only
#     -- and a promoted pair takes the round-trip, which keeps the reference's
#     result dtype.
#
#   * The three operands broadcast to the result shape the reference computes
#     with `at::infer_size`. That shape is not the condition's alone: `where` of
#     a (4, 1) condition against (4, 5) branches answers (4, 5). The body infers
#     it over all three and lets the node broadcast, and a triple that cannot
#     broadcast takes the round-trip so the reference raises.
#
# `native_batch_norm` is the normalisation a convolutional block calls, and the
# second body here that answers with more than one tensor:
# `native_batch_norm(input, weight?, bias?, running_mean?, running_var?,
# training, momentum, eps)` returns `(out, save_mean, save_invstd)`.
#
# **It is two different computations behind one schema, and the body has to
# branch on `training` because the difference is not a keyword argument but a
# different set of sources for the mean and variance.**
#
#   * **Inference** (`training == false`): the statistics are the *running*
#     ones the caller passes, so the graph is one `normalizationWithTensor:...`
#     node whose mean and variance operands are the running buffers. Nothing is
#     reduced, and `save_mean`/`save_invstd` are the running statistics the
#     reference returns (see below).
#   * **Training** (`training == true`): the statistics are the *batch* ones --
#     `mean` and the biased (`unbiased=False`) variance over the per-channel
#     axes -- and the reference also **updates `running_mean`/`running_var`
#     in place** with an exponential moving average. That update is the reason
#     this body is not simply "the same node with different feeds": the graph
#     cannot write the caller's buffers, so a training call is *declined* and
#     takes the round-trip, which is where the in-place update, the unbiased
#     running-variance correction (`momentum` weights the *unbiased* variance,
#     `count/(count-1)` -- a correction neither the node nor any single
#     reduction here applies) and the exact `save_mean`/`save_invstd` semantics
#     all live. Serving training while dropping the update would leave a model's
#     running statistics frozen at their initial values, which is a wrong answer
#     that a forward-output comparison cannot see.
#
# So the graph covers the inference path, which is what `model.eval()` presents
# and what an inference server runs; training stays on the reference.
#
# Three contract details the inference path has to get right, each measured
# against the reference rather than read off the header:
#
#   * **`save_mean` and `save_invstd` in inference are *empty* tensors**, shape
#     `(0,)` on the input's device and dtype -- not the running statistics and
#     not the batch ones. Measured: `native_batch_norm(x, w, b, rm, rv,
#     training=False, ...)` on the reference answers `(out, tensor([]),
#     tensor([]))`, while the same call with `training=True` answers two `[C]`
#     tensors. The reason is that the two results are the *batch* statistics the
#     backward pass consumes, and an inference call has none; a body that filled
#     them with the running statistics -- the intuitive reading, and what this
#     body originally did -- hands back a `[C]` tensor where the reference hands
#     back an empty one, which a shape check sees and a forward comparison does
#     not. The reference's empties are freshly allocated, so the graph path can
#     allocate its own empties here; there is nothing to write into them.
#   * **The channel axis is 1** for every rank the reference accepts (NCHW and
#     NCDHW alike), so `weight` and `bias` reach the node as their own `[C]`
#     selves and the node broadcasts them along the rest. MPSGraph documents
#     exactly this ("Shapes for gamma and beta must match the input data along
#     at least one dimension and will be broadcast along the rest"), and the
#     body checks the length is `channels` rather than describing a reshaped
#     operand -- a length that disagrees is the reference's error, not a
#     broadcast to be guessed at.
#   * **`momentum` is not an operand in inference.** It only weights the running
#     update, which is the part this body declines, so the inference graph is a
#     function of `eps` alone among the host scalars.
#
# The dtype gate is the reduction family's: an integral input is declined like
# every other reduction here, because the reference computes it in float and a
# graph at the operand's own width cannot express that.
T_BATCH_NORM = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var,
    bool training,
    double momentum,
    double eps) {{
  const bool has_weight = weight.has_value() && weight->defined();
  const bool has_bias = bias.has_value() && bias->defined();
  const bool has_running_mean = running_mean.has_value() && running_mean->defined();
  const bool has_running_var = running_var.has_value() && running_var->defined();
  // The CPU round-trip every decline below takes, in one place. The running
  // buffers are passed through as *host* copies for the same reason the input
  // is: the reference updates them in place, and handing it the device tensors
  // would write the update into Metal buffers the graph path does not own.
  const auto round_trip = [&]() {{
    const at::Tensor host_weight = has_weight ? weight->cpu() : at::Tensor();
    const at::Tensor host_bias = has_bias ? bias->cpu() : at::Tensor();
    const at::Tensor host_mean = has_running_mean ? running_mean->cpu() : at::Tensor();
    const at::Tensor host_var = has_running_var ? running_var->cpu() : at::Tensor();
    auto host = at::{at_op}(
        input.cpu(), host_weight, host_bias, host_mean, host_var, training,
        momentum, eps);
    // The in-place running update belongs to the caller's device tensors, so it
    // is copied back rather than left on the host copy the reference wrote.
    if (training && has_running_mean) {{
      running_mean->copy_(host_mean);
    }}
    if (training && has_running_var) {{
      running_var->copy_(host_var);
    }}
    return ::std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        std::get<0>(host).to(input.device()),
        std::get<1>(host).to(input.device()),
        std::get<2>(host).to(input.device()));
  }};
  // Training updates the running statistics in place and weights the update by
  // the *unbiased* batch variance, neither of which the inference node does --
  // see the block comment above. Declined rather than approximated.
  if (training || !has_running_mean || !has_running_var) {{
    return round_trip();
  }}
  if (!mps::{supports}(input.scalar_type(), std::nullopt)) {{
    return round_trip();
  }}
  // The channel axis is 1 for every rank the reference accepts; a rank below 2
  // has no channel axis to normalise over and is the reference's own error.
  const int64_t rank = input.dim();
  if (rank < 2) {{
    return round_trip();
  }}
  const int64_t channels = input.size(1);
  if (has_weight && weight->numel() != channels) {{
    return round_trip();
  }}
  if (has_bias && bias->numel() != channels) {{
    return round_trip();
  }}
  if (running_mean->numel() != channels || running_var->numel() != channels) {{
    return round_trip();
  }}
  at::Tensor input_c = input.contiguous();
  if (input_c.numel() == 0) {{
    return round_trip();
  }}
  // Every operand reaches the graph as a device buffer (see the operand
  // contract in mps_common.h): the running statistics and the affine pair are
  // moved to the input's device rather than described to the graph on the host,
  // which would resolve no Metal buffer for them.
  const auto to_device = [&](const at::Tensor& t) {{
    return t.device() == input.device() ? t.contiguous()
                                        : t.to(input.device()).contiguous();
  }};
  // The gamma the node broadcasts along the channel: the caller's weight when it
  // has one, and an all-ones [C] when it does not -- because the node has no
  // "no gamma" spelling here the way layer_norm's `nil` gamma is one. The
  // substitute is what keeps the operand *list* fixed across a weight-less and a
  // weighted call, so the plan cache key (which records dtypes and shapes and
  // not values) cannot hand the first compiled graph to the second call. `beta`
  // is the same with an all-zeros [C] for a missing bias.
  at::Tensor weight_d = has_weight ? to_device(*weight)
                                   : at::ones({{channels}}, input_c.options());
  at::Tensor bias_d = has_bias ? to_device(*bias)
                               : at::zeros({{channels}}, input_c.options());
  at::Tensor mean_d = to_device(*running_mean);
  at::Tensor var_d = to_device(*running_var);
  at::Tensor out = at::empty(input_c.sizes(), input_c.options());
  // The reference's inference statistics are empty -- see the block comment
  // above -- so there is nothing for the graph to write into these two. They are
  // allocated (zero elements, on the input's device and dtype, which is the
  // reference's own shape and placement) and returned untouched, and they are
  // *not* listed among the graph's results: an MPSGraph result is described
  // against an N-element buffer and MpsFlatten over zero elements is a reshape
  // the graph rejects, so a result list carrying them would fail to build. The
  // graph covers `out` alone.
  at::Tensor stat_mean = at::empty({{0}}, input_c.options());
  at::Tensor stat_var = at::empty({{0}}, input_c.options());
  if (out.numel() == 0) {{
    return {{out, stat_mean, stat_var}};
  }}
  const float epsilon = static_cast<float>(eps);
  std::vector<at::Tensor> operands{{input_c, mean_d, var_d, weight_d, bias_d}};
  // The shape the mean, variance, gamma and beta operands reach the node as: one
  // element per channel with every other axis a 1. The node's type inference
  // requires its four statistics operands to be *broadcast compatible* with the
  // input, and a `[C]` operand is not -- measured, the node rejects a `[2, 3, 4,
  // 5]` input against `[3]` statistics with "output type 'tensor<2x3x4x5xf32>'
  // and mean type 'tensor<3xf32>' are not broadcast compatible" and that abort is
  // a process exit, not a declined node. So each of the four is reshaped in the
  // graph, which is a metadata operation over the same buffer.
  std::vector<int64_t> channel_shape(static_cast<size_t>(rank), 1);
  channel_shape[1] = channels;
  std::vector<at::Tensor> outs{{out}};
  const bool ran = mps::MpsRunGraphMulti(
      mps::MpsGraphKey("{op}"{key_parts}),
      operands,
      outs,
      ^std::vector<MPSGraphTensor*>(MPSGraph* graph,
                                    NSArray<MPSGraphTensor*>* args) {{
        // The operands arrive at their aten shapes: MpsRunGraph describes each
        // one to the graph as the 1-D `numel` buffer the row-pitch note in
        // mps_common.h requires and reapplies the aten shape over it before
        // `build` is called, so `args[0]` is already the ranked input. The four
        // statistics operands are `[C]` and are reshaped to the channel view
        // above -- MPSGraph documents gamma and beta as broadcast along the axes
        // they do not name ("shapes for gamma and beta must match the input data
        // along at least one dimension and will be broadcast along the rest"),
        // but its type inference wants a broadcast *compatible* rank rather than
        // a trailing one, and a `[C]` operand is not that.
        MPSGraphTensor* x = args[0];
        MPSGraphTensor* mean =
            mps::MpsReshape(graph, args[1], at::IntArrayRef(channel_shape));
        MPSGraphTensor* var =
            mps::MpsReshape(graph, args[2], at::IntArrayRef(channel_shape));
        MPSGraphTensor* gamma =
            mps::MpsReshape(graph, args[3], at::IntArrayRef(channel_shape));
        MPSGraphTensor* beta =
            mps::MpsReshape(graph, args[4], at::IntArrayRef(channel_shape));
        if (mean == nil || var == nil || gamma == nil || beta == nil) {{
          return std::vector<MPSGraphTensor*>();
        }}
        // The inference normalisation: `(x - mean) / sqrt(var + eps) * gamma +
        // beta` over the channel, which is exactly what the running statistics
        // and this node express.
        MPSGraphTensor* y = [graph normalizationWithTensor:x
                                               meanTensor:mean
                                           varianceTensor:var
                                              gammaTensor:gamma
                                               betaTensor:beta
                                                  epsilon:epsilon
                                                     name:nil];
        if (y == nil) {{
          return std::vector<MPSGraphTensor*>();
        }}
        return std::vector<MPSGraphTensor*>{{y}};
      }});
  if (!ran) {{
    return round_trip();
  }}
  return {{out, stat_mean, stat_var}};
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

T_WHERE = """\
at::Tensor {kernel}(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {{
  if (!mps::{supports}(self.scalar_type(), other.scalar_type(), condition.scalar_type())) {{
    return at::{at_op}(condition.cpu(), self.cpu(), other.cpu()).to(self.device());
  }}
  at::Tensor condition_c = condition.contiguous();
  at::Tensor self_c = self.contiguous();
  at::Tensor other_c = other.contiguous();
  std::vector<int64_t> out_shape;
  try {{
    out_shape = at::infer_size(
        at::infer_size(condition_c.sizes(), self_c.sizes()), other_c.sizes());
  }} catch (const std::exception&) {{
    return at::{at_op}(condition.cpu(), self.cpu(), other.cpu()).to(self.device());
  }}
  at::Tensor out = at::empty(out_shape, self_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  // The graph's predicate is bool, so a uint8 condition -- the one other width
  // the reference still accepts, with a deprecation warning -- is compared to
  // zero. `condition.dataType` is read inside the block because the node is
  // built once per plan and the comparison is a function of the dtype, which is
  // already in the cache key through the operand.
  const bool wide_condition = condition_c.scalar_type() == at::kByte;
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      {{condition_c, self_c, other_c}},
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        MPSGraphTensor* predicate = args[0];
        if (wide_condition) {{
          predicate = [graph notEqualWithPrimaryTensor:predicate
                                       secondaryTensor:[graph constantWithScalar:0.0
                                                                        dataType:predicate.dataType]
                                                  name:nil];
        }}
        if (predicate == nil) {{
          return nil;
        }}
        return [graph selectWithPredicateTensor:predicate
                            truePredicateTensor:args[1]
                           falsePredicateTensor:args[2]
                                           name:nil];
      }});
  if (!ran) {{
    return at::{at_op}(condition.cpu(), self.cpu(), other.cpu()).to(self.device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""

# `cat(tensors, dim)` is MPSGraph's variadic `concatTensors:dimension:name:`, and
# the interesting part of this body is everything the node does *not* check.
#
# The node concatenates along `dim` and requires the inputs broadcast-compatible
# along every other dimension, which is close to the reference's contract but not
# the same one, and the differences are what the body resolves on the host:
#
#   * `dim` is normalised against the operands' rank. aten accepts a negative
#     `dim` and normalises it (measured: `dim=-1` over a rank-2 pair is `dim=1`);
#     the node's `dimensionIndex` is `-rank <= dim < rank` in the same terms, so
#     the body normalises to a non-negative axis and a `dim` outside the rank is
#     left to the reference.
#
#   * Every operand must carry the same dtype. The reference promotes a mixed
#     list to a common dtype; the node requires one datatype and does not promote,
#     so a mixed list takes the round-trip and keeps the reference's result dtype.
#
#   * The non-`dim` dimensions must match exactly. The reference requires it
#     (`Sizes of tensors must match except in dimension {dim}`) where the node
#     would *broadcast* a (4, 1) against a (4, 5) along the mismatched axis,
#     which is a plausible-looking wrong answer rather than an error. The body
#     checks every other axis on the host and declines the mismatch.
#
# A single-operand `cat` is the operand itself, and `torch.cat([])` is a reference
# error; the body declines an empty list rather than building a node with nothing
# to concatenate, so both reach the reference.
#
# The result shape is the reference's own: every operand's shape with `dim`
# replaced by the sum of the operands' extents along it.
T_CAT = """\
at::Tensor {kernel}(const at::ITensorListRef& tensors, int64_t dim) {{
  // Materialise the list once: the reference and every check below read it more
  // than once, and `ITensorListRef` is a view whose lifetime is the caller's.
  const std::vector<at::Tensor> ops(tensors.begin(), tensors.end());
  // The reference is called on host copies, exactly as every other body here
  // does, because the operands are flagos tensors and `at::cat(ops, dim)` would
  // re-enter this same kernel. That is not a theoretical hazard: the fallback
  // spelling was written with the original list first and a bool `cat` -- which
  // this body declines, because bool has no MPSGraph element type -- recursed
  // until the stack ran out and the process died with SIGSEGV. Built once,
  // before any early return, so no decline path can be written against the live
  // list by mistake.
  std::vector<at::Tensor> host_ops;
  host_ops.reserve(ops.size());
  for (const at::Tensor& t : ops) {{
    host_ops.push_back(t.cpu());
  }}
  // An empty list is the reference's own error, and it is raised here rather
  // than returned as an empty tensor. `ops[0]` below would also be out of
  // range, so the check has to come first for the shape of the body as much as
  // for the message.
  if (ops.empty()) {{
    return at::{at_op}(host_ops, dim);
  }}
  // One dtype for the whole list, and one the graph can carry. A mixed list is
  // declined so the reference promotes it.
  const at::ScalarType dtype = ops[0].scalar_type();
  for (const at::Tensor& t : ops) {{
    if (t.scalar_type() != dtype ||
        !mps::{supports}(dtype, dtype)) {{
      return at::{at_op}(host_ops, dim).to(ops[0].device());
    }}
  }}
  const int64_t rank = ops[0].dim();
  if (rank == 0) {{
    return at::{at_op}(host_ops, dim).to(ops[0].device());
  }}
  int64_t axis = dim;
  if (axis < 0) {{
    axis += rank;
  }}
  if (axis < 0 || axis >= rank) {{
    return at::{at_op}(host_ops, dim).to(ops[0].device());
  }}
  // The result shape: every operand's shape with `axis` summed. A rank or a
  // non-`axis` extent that disagrees is the reference's error, not the node's
  // broadcast, so the body checks rather than describes it.
  std::vector<int64_t> out_shape(ops[0].sizes().vec());
  out_shape[axis] = 0;
  for (const at::Tensor& t : ops) {{
    if (t.dim() != rank) {{
      return at::{at_op}(host_ops, dim).to(ops[0].device());
    }}
    for (int64_t d = 0; d < rank; ++d) {{
      if (d == axis) {{
        continue;
      }}
      if (t.size(d) != out_shape[d]) {{
        return at::{at_op}(host_ops, dim).to(ops[0].device());
      }}
    }}
    out_shape[axis] += t.size(axis);
  }}
  std::vector<at::Tensor> operands;
  operands.reserve(ops.size());
  for (const at::Tensor& t : ops) {{
    operands.push_back(t.contiguous());
  }}
  at::Tensor out = at::empty(out_shape, ops[0].options());
  if (out.numel() == 0) {{
    return out;
  }}
  // Each operand reaches the node as a two-dimensional `(outer, inner)` view,
  // with the concatenation over the *inner* axis rather than over `axis`
  // directly.
  //
  // Concatenating along `axis` is exactly concatenating the inner axis of the
  // `(prod(shape[:axis]), prod(shape[axis:]))` view: there are `outer` blocks,
  // each block is `inner` elements wide, and the operands join end to end inside
  // every block. The result is `(outer, sum(inner))`, which reshapes back to the
  // aten shape.
  //
  // This indirection is not cosmetic, and the direct spelling is what found it.
  // Measured through this backend, `cat([tensor([0., 1.]), tensor([2., 3.])],
  // dim=0)` -- two rank-1 operands handed to `concatTensors:` as their ranked
  // selves -- returned all zeros, and an int32 form returned
  // `[0, 0, 1, 0, 2, 0]`: the eight-byte row pitch of a four-byte element, the
  // same symptom the row-pitch note in mps_common.h records. A (2, 3) pair on
  // dim=0 was correct, which is what makes the rank-1 case the interesting one:
  // MPSGraph reads a rank-1 input to `concatTensors:` as one row and pays the
  // pitch, and the two-dimensional view above sidesteps that by never handing
  // the node a rank below two. `inner` is at least 1 because `axis < rank`.
  //
  // `outer` is the same on every operand -- the shape check above rejected any
  // list where a non-`axis` extent disagreed -- but `inner` is not: it is each
  // operand's own extent along `axis` times the extents after it, and the whole
  // point of the concatenation is that those differ. Read per operand, from the
  // same `(prod(shape[:axis]), prod(shape[axis:]))` split, so a (2,) operand
  // beside a (1,) one reshapes to (1, 2) and (1, 1) rather than both to the
  // first operand's width. Sharing one `inner` across the list is not a
  // cosmetic error: a three-operand rank-1 `cat` of (2,), (2,) and (1,) built
  // `mps.reshape` to a shape the third operand's element count cannot fill and
  // MPSGraph aborted the process ("the result shape is not compatible with the
  // input shape", then "original module failed verification") rather than
  // declining.
  int64_t outer = 1;
  for (int64_t d = 0; d < axis; ++d) {{
    outer *= ops[0].size(d);
  }}
  std::vector<int64_t> inner;
  inner.reserve(ops.size());
  for (const at::Tensor& t : ops) {{
    inner.push_back(t.numel() / (outer > 0 ? outer : 1));
  }}
  const bool ran = mps::MpsRunGraph(
      mps::MpsGraphKey("{op}"{key_parts}),
      operands,
      out,
      ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {{
        NSMutableArray<MPSGraphTensor*>* rows =
            [NSMutableArray arrayWithCapacity:[args count]];
        for (NSUInteger i = 0; i < [args count]; ++i) {{
          [rows addObject:mps::MpsReshape(graph, args[i],
                                          at::IntArrayRef({{outer, inner[i]}}))];
        }}
        MPSGraphTensor* joined = [graph concatTensors:rows
                                            dimension:1
                                                 name:nil];
        if (joined == nil) {{
          return nil;
        }}
        // `outer` is the operand side; the concatenation's own outer extent is
        // the sum the result shape already carries along `axis`, which is what
        // the reshape back to the aten shape reads.
        return mps::MpsReshape(graph, joined, out_shape);
      }});
  if (!ran) {{
    return at::{at_op}(host_ops, dim).to(ops[0].device());
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMps, {kernel})
"""
# `tanh` is the activation a transformer block reaches for at both ends -- the
# tanh approximation inside a GELU and the one inside a recurrent or attention
# gate -- and the body is the shared one-node unary shape with exactly one
# addition.
#
# The addition is a NaN guard, and it is the node's doing rather than a
# defensive check: measured at all three widths, `tanhWithTensor:` *saturates* a
# NaN operand to +-1 instead of propagating it -- `tanh(nan)` is 1.0 on the graph
# where the reference answers nan -- so the body builds `isNaNWithTensor:` over
# the operand and selects a NaN back over the node's answer. `isNaNWithTensor:`
# is on both the SDK this backend is verified against and the one after it (no
# availability annotation in `MPSGraphArithmeticOps.h`), unlike the fused
# `reciprocalSquareRootWithTensor:` T_RSQRT had to avoid, so the guard is
# available where the lane runs. The library helper is a second `nil` to check,
# and a `nil` from it round-trips like any other declined build.
#
# Everything else this body does is the shared unary contract, and the increment
# is the gate rather than the graph. See MpsSupportsTanhDtype: the node is the
# reference's own function at float16 and bfloat16 -- bfloat16 bit-identical
# with exact odd symmetry, float16 within two last-ulp roundings of 400001 -- and
# a coarser approximation of a tanh at float32, where it breaks odd symmetry by
# 1.788e-07 and sits ~100x further from the exact answer at every scale. That
# width keeps the round-trip.
#
# Only `tanh` is registered. `tanh.out` and `tanh_` write into a tensor the
# caller owns, like every other `.out`/inplace spelling on this backend, and
# `tanh_backward`, `atanh` and `_foreach_tanh` are separate ops with their own
# selectors and contracts -- `atanh`'s domain is `|x| < 1`, which is not this
# node's domain at all.

T_TANH = """\
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
        MPSGraphTensor* tanh = [graph {call}args[0] name:nil];
        if (tanh == nil) {{
          return nil;
        }}
        // The node saturates a NaN operand to +-1 rather than propagating it --
        // measured, `tanh(nan)` is 1.0 at all three widths where the reference
        // answers nan -- so a NaN input is selected back to NaN rather than
        // left to the node's own answer. This is the one deviation here that is
        // a wrong *answer* rather than a rounding, and it is the check that
        // makes the body a tanh rather than a saturating approximation.
        MPSGraphTensor* nan = [graph isNaNWithTensor:args[0] name:nil];
        if (nan == nil) {{
          return nil;
        }}
        return [graph selectWithPredicateTensor:nan
                            truePredicateTensor:[graph constantWithScalar:NAN
                                                                 dataType:args[0].dataType]
                           falsePredicateTensor:tanh
                                           name:nil];
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
    "sigmoid": T_SIGMOID,
    "exp": T_EXP,
    "log": T_EXP,
    "rsqrt": T_RSQRT,
    # The shared one-node unary template, and the increment is which node and
    # which gate rather than a body: the graph is `squareRootWithTensor:` over
    # one operand, exactly the inner node the `rsqrt` composition builds, so
    # there is nothing to compose and the whole of this body is
    # `MpsSupportsSqrtDtype`. See the note there for the per-width measurement.
    "sqrt": T_EXP,
    # The shared one-node unary template again, and the increment is the gate
    # rather than a body: `erfWithTensor:` is a real selector and needs no guard
    # -- measured, it propagates a NaN and answers the `+-1` bound at `+-inf`
    # without help -- so `MpsSupportsErfDtype` is the whole of the finding. See
    # the note there for what separates its width list from the sibling
    # `tanh`'s.
    "erf": T_EXP,
    # Its own template rather than the shared unary one, because the node needs
    # a guard the siblings do not: MPSGraph's `tanhWithTensor:` saturates a NaN
    # operand to +-1 at every width, so the body selects the input's NaN back
    # over the node's answer. Everything else is the one-node unary shape, and
    # the gate is where the increment is -- MpsSupportsTanhDtype serves the two
    # narrow widths and declines float32.
    "tanh": T_TANH,
    "embedding": T_EMBEDDING,
    "index_select": T_INDEX_SELECT,
    "take": T_TAKE,
    "gather": T_GATHER,
    "div": T_DIV,
    "div_scalar": T_DIV_SCALAR,
    "reciprocal": T_RECIPROCAL,
    "layer_norm": T_LAYER_NORM,
    "where": T_WHERE,
    "cat": T_CAT,
    "batch_norm": T_BATCH_NORM,
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
#include <ATen/ops/cat.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/div.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/native_batch_norm.h>
#include <ATen/ops/native_layer_norm.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/take.h>
#include <ATen/ops/where.h>
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
                # The two optional operands the normalisation bodies read out of
                # their own operand lists, as C++ literals.
                #
                # layer_norm builds `{input, weight?, bias?}` and passes the
                # missing affine operands to the node as the `nil` its own
                # `_Nullable` gamma and beta document, so both spellings are one
                # graph shape.
                #
                # batch_norm builds `{input, mean, var, weight, bias}` and always
                # fills the last two -- an absent affine operand becomes a
                # constant tensor of the channel shape rather than a missing
                # feed -- so gamma and beta are the two *trailing* entries of
                # that list, unconditionally.
                gamma=(
                    "args[3]"
                    if category == "batch_norm"
                    else "has_weight ? args[1] : nil"
                ),
                beta=(
                    "args[4]"
                    if category == "batch_norm"
                    else "has_weight && has_bias ? args[2] : (has_bias ? args[1] : nil)"
                ),
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
