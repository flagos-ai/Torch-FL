// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

// Host-side launch interface for the hipcc-compiled DCU elementwise kernels.
//
// This header is the seam between two compilers. The generated kernel wrappers
// in generated/dtk_kernels.cc are ordinary C++ compiled by the host compiler and
// know about at::Tensor; the device code in dtk_elementwise.hip is compiled by
// DTK's hipcc and knows nothing about torch. Only plain types cross this
// boundary -- void* data, element counts, a dtype tag and an op tag -- so the
// device translation unit never includes an ATen header and cannot pick up a
// dependency on the vendor's libtorch_hip.
//
// Keeping the dtype as c10::ScalarType is deliberate and safe: it is a plain
// enum from a header-only part of c10, so hipcc can consume it without linking
// any torch library. That gives the device side a single authoritative dtype
// vocabulary instead of a parallel enum that could drift.

#include <ATen/core/ScalarType.h>

#include <cstddef>
#include <cstdint>

struct Stream;
using Stream_t = Stream*;

namespace at::native::flagos::dtk_ops {

// Device-side op selectors. Values are positional and carry no ABI meaning --
// both sides of this call are always compiled together into
// libdcu_aten_ops.so -- so entries may be reordered freely. The names are
// produced by codegen_dtk.py:op_tag() from the schema name.
enum class UnaryOp : int32_t {
  ABS,
  ACOS,
  ACOSH,
  ASIN,
  ATAN,
  ATANH,
  COS,
  ERF,
  ERFC,
  EXP2,
  FRAC,
  LOG,
  LOG2,
};

enum class BinaryOp : int32_t {
  BITWISE_AND_TENSOR,
  BITWISE_OR_TENSOR,
  BITWISE_XOR_TENSOR,
  GCD,
  HYPOT,
  NEXTAFTER,
};

// out[i] = op(in[i]) for i in [0, numel).
//
// Both pointers must be device pointers to contiguous buffers of `numel`
// elements of `dtype`; the generated wrappers guarantee that. Launches on the
// current DCU stream and does not synchronize -- ordering against the caller's
// subsequent work is the stream's job, and forcing a sync here would serialize
// every elementwise op.
//
// Throws c10::Error if `dtype` is not supported by `op` (e.g. a bitwise op on a
// floating type), rather than silently producing garbage.
void LaunchUnary(
    const void* in,
    void* out,
    int64_t numel,
    c10::ScalarType dtype,
    UnaryOp op,
    Stream_t stream);

// out[i] = op(a[i], b[i]). Operands must already be broadcast to a common shape
// and promoted to `dtype` -- the generated wrappers do that with aten's own
// utilities so the semantics match CPU exactly.
void LaunchBinary(
    const void* a,
    const void* b,
    void* out,
    int64_t numel,
    c10::ScalarType dtype,
    BinaryOp op,
    Stream_t stream);

} // namespace at::native::flagos::dtk_ops
