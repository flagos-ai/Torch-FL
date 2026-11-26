// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "adamw_precision.h"

#include <cuda_runtime.h>

namespace at::native::flagos::thead_adamw {
namespace {

__device__ float BFloat16ToFloat(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ uint16_t FloatToBFloat16(float value) {
  const auto bits = __float_as_uint(value);
  if ((bits & 0x7fffffff) > 0x7f800000)
    return static_cast<uint16_t>((bits >> 16) | 0x40);
  return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

template <bool Divide>
__global__ void Update(uint16_t* result, const uint16_t* first, const uint16_t* second,
                       int count, float alpha) {
  const auto index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float current = BFloat16ToFloat(result[index]);
  const float left = BFloat16ToFloat(first[index]);
  const float right = BFloat16ToFloat(second[index]);
  float updated;
  if constexpr (Divide) {
    // Stock Torch rounds the quotient before the fused BF16 update.
    const float quotient = __fdiv_rn(left, right);
    updated = fmaf(alpha, quotient, current);
  } else {
    // The stock BF16 AdamW kernel forms the gradient product before applying alpha.
    const float product = __fmul_rn(left, right);
    updated = fmaf(alpha, product, current);
  }
  result[index] = FloatToBFloat16(updated);
}

template <bool Divide>
cudaError_t Launch(uint16_t* result, const uint16_t* first, const uint16_t* second,
                   int count, float alpha, cudaStream_t stream) {
  constexpr int threads = 128;
  Update<Divide><<<(count - 1) / threads + 1, threads, 0, stream>>>(
      result, first, second, count, alpha);
  return cudaGetLastError();
}

}  // namespace

cudaError_t Addcmul(uint16_t* result, const uint16_t* first, const uint16_t* second,
                    int count, float alpha, cudaStream_t stream) {
  return Launch<false>(result, first, second, count, alpha, stream);
}

cudaError_t Addcdiv(uint16_t* result, const uint16_t* first, const uint16_t* second,
                    int count, float alpha, cudaStream_t stream) {
  return Launch<true>(result, first, second, count, alpha, stream);
}

}  // namespace at::native::flagos::thead_adamw
