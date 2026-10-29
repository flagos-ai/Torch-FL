// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Operator-library runtime for the D-Robotics BPU.
//
// The BPU has no per-operator entry point: hbDNNInferV2 takes a compiled .hbm
// artifact, so serving one aten op means loading the artifact that carries it
// and submitting it. Those artifacts are produced off-board by
// scripts/vendor/build_bpu_oplib.py and found through FLAGOS_BPU_OPLIB.
//
// Everything here is lazy. A board without the library, or with no
// FLAGOS_BPU_OPLIB set, answers "unsupported" from BpuOplibLookup and every
// kernel falls through to the CPU round-trip, so a missing library costs
// speed and never correctness.
//
// Shapes are bucketed, not dynamic: HBRT 4.7.5 declares a dynamic axis with
// stride 0 while its task validator requires at least 64, so no caller can
// satisfy both (docs/vendors/bpu/dynamic-shape-probe.md). A request rounds up
// to the smallest bucket at or above its width and is cropped back. That is
// sound here because every op in the library is elementwise -- the padded
// lanes cannot influence the ones before them.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty_like.h>

#include <cstdint>
#include <utility>

namespace flagos {
namespace bpu {
namespace oplib {

// Where the operator library lives: FLAGOS_BPU_OPLIB, or "" when unset.
const char* Root();

// Smallest bucket at or above `width` for `op`, or 0 when the op is not in the
// library or every bucket is smaller than the request. `op` is the bare hbir
// name ("relu", "add"); `nargs` is 1 or 2.
int64_t BucketFor(const char* op, int nargs, int64_t width);

// Run `op` over `nargs` f32 buffers of `width` elements, writing `width`
// results to `out`. Returns false when the op has no bucket for this width or
// the artifact fails, which sends the caller to the reference path.
bool Run(const char* op, int nargs, const float* const* in, int64_t width,
         float* out);

// One-line accounting of hits and misses, for the same reason
// EagerDispatcher.summary() exists on the Python side: a path that silently
// degrades to the CPU must be visible rather than assumed.
const char* Summary();

// Tensor-level entry points the generated kernels call. A request the library
// cannot serve -- a non-f32 tensor, operands that do not match shape for
// shape, an empty tensor, or a width past the largest bucket -- is handed to
// `fallback` unchanged, which is the kernel's own CPU round-trip.
template <typename Fallback>
at::Tensor RunUnary(const char* op, const at::Tensor& self, Fallback fallback) {
  if (self.scalar_type() != at::kFloat || self.numel() == 0) {
    return fallback(self);
  }
  const at::Tensor self_c = self.contiguous();
  const int64_t width = self_c.numel();
  const float* in[1] = {self_c.const_data_ptr<float>()};
  at::Tensor out = at::empty_like(self_c);
  if (!Run(op, 1, in, width, out.mutable_data_ptr<float>())) {
    return fallback(self);
  }
  return out;
}

template <typename Fallback>
at::Tensor RunBinary(const char* op, const at::Tensor& a, const at::Tensor& b,
                     Fallback fallback) {
  if (a.scalar_type() != at::kFloat || b.scalar_type() != at::kFloat ||
      a.sizes() != b.sizes() || a.numel() == 0) {
    return fallback(a, b);
  }
  const at::Tensor a_c = a.contiguous();
  const at::Tensor b_c = b.contiguous();
  const int64_t width = a_c.numel();
  const float* in[2] = {a_c.const_data_ptr<float>(), b_c.const_data_ptr<float>()};
  at::Tensor out = at::empty_like(a_c);
  if (!Run(op, 2, in, width, out.mutable_data_ptr<float>())) {
    return fallback(a, b);
  }
  return out;
}

} // namespace oplib
} // namespace bpu
} // namespace flagos