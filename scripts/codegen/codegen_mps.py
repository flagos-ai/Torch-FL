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
# docs/superpowers/specs/2026-10-14-apple-mps-native-backend-design.md section 7:
# the `unary` category, the elementwise `binary` and `binary_scalar` families, and
# the `reduce` family, each matching the CPU reference within tolerance. Stage 3
# adds the gemm/conv/softmax set Qwen3-0.6B needs.
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
# (`abs.out`, `abs_`, `add.out`, `add_`, `sum.IntList_out`, ...) are left
# unregistered on PrivateUse1, exactly as on GCU/MUSA/Ascend: the dispatcher then
# sends those spellings to the generic fallback rather than to an empty kernel
# slot. A later stage adds them once their kernels exist.
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
}

# The per-category dtype predicate each kernel gates on (mps_common.h).
SUPPORTS = {
    "unary": "MpsSupportsUnaryDtype",
    "binary": "MpsSupportsBinaryDtype",
    "binary_scalar": "MpsSupportsScalarDtype",
    "reduce": "MpsSupportsReduceDtype",
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
# axes a reduction covers. Empty for an op whose graph is a function of its
# operands alone.
#
# A host value a body reads and this list leaves out is not a slow path, it is a
# wrong answer: the two calls produce one cache key and the first compiled graph
# answers both. So an op with a host constant declares it here, and a reviewer
# checks this row against the body that reads it.
KEY_PARTS = {
    "abs": "",
    "add.Tensor": ", alpha",
    "sub.Tensor": ", alpha",
    "mul.Tensor": "",
    "add.Scalar": ", other, alpha",
    "sub.Scalar": ", other, alpha",
    "mul.Scalar": ", other",
    "sum.dim_IntList": ", axes",
    "mean.dim": ", axes",
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

CATEGORIES = {
    "unary": T_UNARY,
    "binary": T_BINARY,
    "binary_scalar": T_SCALAR,
    "reduce": T_REDUCE,
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
// itself is compiled once per (body, operand shapes and dtypes) and reused, so
// the per-call cost is a schedule rather than a graph construction -- see the
// plan cache in mps_common.mm. Dispatchers are declared in generated/ops.h
// (shared with the CUDA codegen); here we only fill the Backend::kMps slot.
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
#include <ATen/ops/abs.h>
#include <ATen/ops/add.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sum.h>
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
                alpha_param=alpha_param,
                alpha_arg=alpha_arg,
                alpha_operand=alpha_operand,
                finish=FINISH.get(op, ""),
                key_parts=KEY_PARTS[op],
                op=op,
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
