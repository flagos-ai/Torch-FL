// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <macros.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "native_host/enqueue_trace.h"

namespace at::native::flagos::mps {

// Inline metadata is copied by the encoder before run returns.
using MetalArg = std::variant<at::Tensor, int64_t, c10::IntArrayRef>;
using MetalGrid = std::array<uint64_t, 3>;

class FLAGOS_EXPORT MetalKernel {
 public:
  explicit MetalKernel(const std::string& source);
  ~MetalKernel();

  void run(c10::ArrayRef<MetalArg> args, const MetalGrid& threads,
           const std::optional<MetalGrid>& group_size,
           bool compiled_host = false) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The registered operator uses PyTorch's bundled dense Metal kernel.
bool RunMetalUnary(const at::Tensor& out, const at::Tensor& input,
                   const char* name);
bool RunMetalBinaryAlpha(const at::Tensor& out, const at::Tensor& input,
                        const at::Tensor& other, const at::Scalar& alpha,
                        const char* name);
bool RunMetalFp32ScaledAdd(const at::Tensor& out, const at::Tensor& input,
                           const at::Tensor& left, const at::Tensor& right,
                           const at::Scalar& alpha, const char* term_name);
void RunMetalTernaryAlpha(const MetalKernel& kernel, const at::Tensor& out,
                          const at::Tensor& input, const at::Tensor& left,
                          const at::Tensor& right, float alpha);

FLAGOS_EXPORT ::flagos::native_host::EnqueueTrace::Events CaptureEnqueues(
    const std::function<void()>& run);

} // namespace at::native::flagos::mps
