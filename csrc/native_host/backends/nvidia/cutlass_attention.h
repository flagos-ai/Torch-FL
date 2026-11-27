// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <cuda_runtime_api.h>

namespace at { class Tensor; }

namespace flagos::nvidia {

struct AttentionInput {
  const void* data;
  int64_t batch_stride;
  int head_stride;
  int token_stride;
};

cudaError_t LaunchCutlassAttention(const AttentionInput& query, const AttentionInput& key,
                                   const AttentionInput& value, int batches, int heads,
                                   int queries, int keys, int qk_width, int value_width,
                                   float scale, void* output, void* accumulator,
                                   cudaStream_t stream);

at::Tensor TryCutlassAttention(const at::Tensor& query, const at::Tensor& key,
                              const at::Tensor& value, std::optional<double> scale);

}  // namespace flagos::nvidia
