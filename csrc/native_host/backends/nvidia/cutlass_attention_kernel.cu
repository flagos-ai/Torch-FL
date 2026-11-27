// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "cutlass_attention.h"

#include <tuple>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

namespace flagos::nvidia {
namespace {

// The published CUTLASS forward configuration for aligned wide BF16 heads.
using WideForward = PyTorchMemEffAttention::AttentionKernel<
    cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>;

__global__ void __launch_bounds__(WideForward::kNumThreads, WideForward::kMinBlocksPerSm)
CutlassForward(WideForward::Params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if (params.advance_to_block()) WideForward::attention_kernel(params);
#endif
}

}  // namespace

cudaError_t LaunchCutlassAttention(const AttentionInput& query, const AttentionInput& key,
                                   const AttentionInput& value, int batches, int heads,
                                   int queries, int keys, int qk_width, int value_width,
                                   float scale, void* output, void* accumulator,
                                   cudaStream_t stream) {
  WideForward::Params args;
  args.query_ptr = static_cast<const cutlass::bfloat16_t*>(query.data);
  args.key_ptr = static_cast<const cutlass::bfloat16_t*>(key.data);
  args.value_ptr = static_cast<const cutlass::bfloat16_t*>(value.data);
  args.output_ptr = static_cast<cutlass::bfloat16_t*>(output);
  args.output_accum_ptr = static_cast<float*>(accumulator);
  args.num_batches = batches;
  args.num_heads = heads;
  args.num_queries = queries;
  args.num_keys = keys;
  args.head_dim = qk_width;
  args.head_dim_value = value_width;
  args.q_strideB = query.batch_stride;
  args.q_strideH = query.head_stride;
  args.q_strideM = query.token_stride;
  args.k_strideB = key.batch_stride;
  args.k_strideH = key.head_stride;
  args.k_strideM = key.token_stride;
  args.v_strideB = value.batch_stride;
  args.v_strideH = value.head_stride;
  args.v_strideM = value.token_stride;
  args.o_strideM = heads * value_width;
  args.scale = scale;

  constexpr auto shared = sizeof(WideForward::SharedStorage);
  const auto status = cudaFuncSetAttribute(
      CutlassForward, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  CutlassForward<<<args.getBlocksGrid(), args.getThreadsGrid(), shared, stream>>>(args);
  return cudaGetLastError();
}

}  // namespace flagos::nvidia
