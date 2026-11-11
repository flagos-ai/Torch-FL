// Copyright (c) 2026, BAAI. All rights reserved.
//
// Fast replay of a captured FlagGems Triton kernel, inside the caller.
//
// The FlagGems route spends most of its per-call time outside the kernel: acquiring the
// GIL, resolving and calling the Python wrapper, marshalling arguments and allocating the
// result. All of that is needed to *describe* a launch and none of it to *repeat* one, so
// a launch can be described once and replayed from C++ with three pointer stores.
//
// A plan is registered from Python (which owns the Triton kernel and its metadata) and
// replayed here. The callers in python_op_caller.cc consult this module first, so a
// replay never enters the interpreter at all.
//
// Correctness rests on two rules:
//   * the argument layout comes from the compiled kernel, not from the Python signature;
//   * a plan is replayed only for operands that match what it was captured with, and any
//     mismatch falls back to the Python route rather than launching with wrong addresses.
// Both matter because a wrong launch does not crash - it returns a plausible wrong answer.

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/core/ivalue.h>

#include <cstdint>
#include <string>
#include <vector>

namespace at::native::flagos {

// Register a plan that replays `op` (the FlagGems callable name, e.g. "flag_gems.mul").
//
// `cubin` is the compiled kernel image, `kernel` its entry name. `kinds` describes each
// kernel parameter in order: "in" (operand, taken from the call), "out" (pooled result)
// or "zero" (a scratch pointer Triton declares but this path never fills). `ints` are the
// scalar parameters in kernel order, `operand_shapes`/`operand_dtypes` the operands the
// plan was captured for, and `out_shape`/`out_dtype` the result's. The plan is replayed
// only for operands that match those shapes and types exactly.
//
// Returns false when the driver, the cubin or the kernel cannot be resolved; the caller
// then keeps the Python route.
bool RegisterFastPlan(const std::string& op, const std::string& kernel,
                      const void* cubin, size_t cubin_size, int64_t grid, int64_t threads,
                      int64_t shared, const std::vector<std::string>& kinds,
                      const std::vector<int64_t>& ints,
                      const std::vector<std::vector<int64_t>>& operand_shapes,
                      const std::vector<int64_t>& operand_dtypes,
                      const std::vector<int64_t>& scalar_values,
                      const std::vector<int64_t>& out_shape, int64_t out_dtype);

// True when `op` has a plan, so a caller can decide before doing any work.
bool HasFastPlan(const char* op);

// Replay `op` for the call's positional arguments, returning a pooled result. Returns an
// undefined tensor when there is no plan, or when the operands' shapes/types or the
// call's scalar arguments differ from what the plan was captured with - a plan bakes the
// scalars it saw (softmax's `dim`, for instance), so a different value must fall back
// rather than replay the captured one.
at::Tensor TryFastReplay(const char* op, const std::vector<c10::IValue>& args);

// Diagnostics: plans registered, and replays served per plan.
int64_t FastPlanCount();
int64_t FastPlanReplays(const char* op);

}  // namespace at::native::flagos