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
#include <flagos.h>
#include <ATen/ops/empty_like.h>

#include <cstdint>
#include <utility>
#include <string>
#include <vector>

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

// `Run` with a C ABI, so a caller that is not going through ATen -- the eager
// path in torch_fl/accelerator/bpu -- can reach it with ctypes and skip the
// dispatcher. Same semantics and same return value.
extern "C" FLAGOS_EXPORT bool FlagosBpuOplibRun(const char* op, int nargs,
                                                const float* const* in,
                                                int64_t width, float* out);

// Submit the artifact at `hbm_path` with tensor arguments.
//
// Elementwise operators need none of this: every operand is the same width, so
// `Run` can size one bucket and copy. An operator with parameters cannot. Its
// shape is fixed when it is compiled, so it has no bucket, and the number and
// extent of its arguments are whatever that particular graph declares.
//
// A weight is *not* among those arguments. Compiling one as a runtime input
// leaves the toolchain no constant to fold, and the artifact it produces is a
// CPU node: measured at 185 ms for one 64-channel 3x3 layer, against 0.9 ms for
// the same layer with its weight baked in. So the caller compiles one artifact
// per weight set -- see ParamKernelCache on the Python side, which keys them by
// pulse and shape -- and this only has to run it.
//
// `hbm_path` is therefore an absolute path, not a name under FLAGOS_BPU_OPLIB:
// these artifacts live wherever the compiler cached them.
//
// The shapes are answered from the file, and every `at::Tensor` is checked
// against what it declares, which keeps a mismatch a miss rather than a wrong
// answer. Returns false when the file is absent, when `inputs` does not match
// its declaration, or when submission fails -- all of which send the caller to
// the reference path.
FLAGOS_EXPORT bool RunArtifact(
    const char* hbm_path, const std::vector<at::Tensor>& inputs, at::Tensor& out);


// Register the artifact a parameterised operator compiled to, and look it up
// again.
//
// aten hands a convolution its weight as a tensor, not as the nn.Module the
// Python compiler traces, so the two sides cannot describe the same operator
// the same way. The compiler knows which weight set produced which artifact --
// it is the thing that traced, quantised and compiled it -- so it publishes the
// mapping and the kernel reads it back. `key` is opaque to this layer.
FLAGOS_EXPORT void RegisterArtifact(const char* key, const char* hbm_path);

// The path registered for `key`, or nullptr. Returned pointer stays valid for
// the life of the process.
FLAGOS_EXPORT const char* LookupArtifact(const char* key);

// The key RunWeightedOp derives. Exposed so the compiler can register under the
// same string it will look up.
FLAGOS_EXPORT std::string WeightedKey(const char* op,
                                      const std::vector<at::Tensor>& operands);

// Submit the artifact the compiler registered for these operands.
//
// `op` names the aten operator; the operands are the tensors its artifact was
// compiled from, in the order the compiler keyed them. The key is derived here
// from their storage identity and version rather than passed in, so that a
// kernel never has to know how the compiler names things.
//
// Falls to `fallback` when nothing is registered for this operand set, which
// keeps an uncompiled model correct rather than merely slow.
template <typename Fallback>
at::Tensor RunWeightedOp(const char* op, const std::vector<at::Tensor>& operands,
                         Fallback fallback) {
  const std::string key = WeightedKey(op, operands);
  const char* path = LookupArtifact(key.c_str());
  if (path == nullptr) {
    return fallback();
  }
  at::Tensor out;
  if (!RunArtifact(path, {operands.front()}, out)) {
    return fallback();
  }
  return out;
}

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