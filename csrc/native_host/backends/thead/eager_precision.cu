// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "eager_precision.h"

#include <cuda_runtime.h>

#include <cfloat>

namespace at::native::flagos::thead_eager {
namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kWarpMask = 0xffffffff;
constexpr int kAttentionWidth = 128;
constexpr int kAttentionRowsPerBlock = 4;
constexpr int kReductionRowsPerBlock = 16;
constexpr int kWideThreads = 1024;
constexpr int kWideBackwardThreads = 512;
constexpr int kElementwiseThreads = 128;

template <bool Sum>
__device__ float Combine(float left, float right) {
  if constexpr (Sum) return left + right;
  return fmaxf(left, right);
}

template <bool Sum>
__device__ float ReduceWarp(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = Combine<Sum>(value, __shfl_down_sync(kWarpMask, value, offset));
  }
  return __shfl_sync(kWarpMask, value, 0);
}

template <bool Sum>
__device__ float ReduceWideBlock(float value, float* warp_totals) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const float partial = ReduceWarp<Sum>(value);
  if (lane == 0) warp_totals[warp] = partial;
  __syncthreads();
  if (warp == 0) {
    const float result = ReduceWarp<Sum>(warp_totals[lane]);
    if (lane == 0) warp_totals[0] = result;
  }
  __syncthreads();
  const float total = warp_totals[0];
  __syncthreads();
  return total;
}

template <bool Mean>
__global__ void ReduceRows(const float* input, float* output, int rows, int width) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) return;
  const auto* vectors = reinterpret_cast<const float4*>(
      input + static_cast<int64_t>(row) * width);
  float4 partial{0, 0, 0, 0};
  // Accumulate each vector component independently before the warp tree.
  for (int col = threadIdx.x; col < width / 4; col += kWarpSize) {
    const float4 value = vectors[col];
    partial.x += value.x;
    partial.y += value.y;
    partial.z += value.z;
    partial.w += value.w;
  }
  const float total = ReduceWarp<true>(((partial.x + partial.y) + partial.z) + partial.w);
  if (threadIdx.x == 0) output[row] = Mean ? total * (1.0f / width) : total;
}

__global__ void InverseSquareRoot(const float* input, float* output, int count) {
  const auto index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = rsqrtf(input[index]);
}

__global__ void AttentionSoftmaxRows(const float* input, float* output, int rows) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) return;
  input += static_cast<int64_t>(row) * kAttentionWidth;
  output += static_cast<int64_t>(row) * kAttentionWidth;
  float maximum = -INFINITY;
  for (int col = threadIdx.x; col < kAttentionWidth; col += kWarpSize) {
    maximum = fmaxf(maximum, input[col]);
  }
  maximum = ReduceWarp<false>(maximum);
  float partial = 0.0f;
  float exponents[kAttentionWidth / kWarpSize];
  #pragma unroll
  for (int i = 0; i < kAttentionWidth / kWarpSize; ++i) {
    exponents[i] = expf(input[threadIdx.x + i * kWarpSize] - maximum);
    partial += exponents[i];
  }
  const float total = ReduceWarp<true>(partial);
  #pragma unroll
  for (int i = 0; i < kAttentionWidth / kWarpSize; ++i) {
    output[threadIdx.x + i * kWarpSize] = exponents[i] / total;
  }
}

__global__ void AttentionSoftmaxBackwardRows(const float* grad, const float* probability,
                                            float* output, int rows) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) return;
  const auto start = static_cast<int64_t>(row) * kAttentionWidth + threadIdx.x;
  float partial = 0.0f;
  float weighted[kAttentionWidth / kWarpSize];
  #pragma unroll
  for (int i = 0; i < kAttentionWidth / kWarpSize; ++i) {
    const auto index = start + i * kWarpSize;
    // The vendor forms this product before reducing the weighted gradient.
    weighted[i] = __fmul_rn(grad[index], probability[index]);
    partial += weighted[i];
  }
  const float total = ReduceWarp<true>(partial);
  #pragma unroll
  for (int i = 0; i < kAttentionWidth / kWarpSize; ++i) {
    const auto index = start + i * kWarpSize;
    output[index] = fmaf(-probability[index], total, weighted[i]);
  }
}

__global__ void WideLogSoftmaxRows(const float* input, float* output, int width) {
  constexpr int columns_per_thread = 4;
  constexpr int columns_per_tile = kWideThreads * columns_per_thread;
  __shared__ float warp_totals[kWideThreads / kWarpSize];
  const auto row_offset = static_cast<int64_t>(blockIdx.x) * width;
  input += row_offset;
  output += row_offset;
  const int complete_columns = width - width % columns_per_tile;
  float maximum = -FLT_MAX;
  for (int col = threadIdx.x; col < width; col += kWideThreads) {
    maximum = fmaxf(maximum, input[col]);
  }
  maximum = ReduceWideBlock<false>(maximum, warp_totals);
  float partial = 0.0f;
  // Complete tiles use adjacent groups of four. The remainder distributes
  // individual columns across the block in the stock reduction order.
  for (int col = columns_per_thread * threadIdx.x; col < complete_columns;
       col += columns_per_tile) {
    #pragma unroll
    for (int i = 0; i < columns_per_thread; ++i) {
      partial += expf(input[col + i] - maximum);
    }
  }
  for (int col = complete_columns + threadIdx.x; col < width; col += kWideThreads) {
    partial += expf(input[col] - maximum);
  }
  const float normalization = logf(ReduceWideBlock<true>(partial, warp_totals));
  for (int col = threadIdx.x; col < width; col += kWideThreads) {
    output[col] = (input[col] - maximum) - normalization;
  }
}

__global__ void WideLogSoftmaxBackwardRows(const float* grad, const float* log_prob,
                                            float* output, int width) {
  constexpr int columns_per_thread = 4;
  constexpr int columns_per_tile = kWideBackwardThreads * columns_per_thread;
  constexpr int warps = kWideBackwardThreads / kWarpSize;
  __shared__ float warp_totals[warps];
  const auto row_offset = static_cast<int64_t>(blockIdx.x) * width;
  grad += row_offset;
  log_prob += row_offset;
  output += row_offset;
  const int complete_columns = width - width % columns_per_tile;
  float partial = 0.0f;
  for (int col = columns_per_thread * threadIdx.x; col < complete_columns;
       col += columns_per_tile) {
    #pragma unroll
    for (int i = 0; i < columns_per_thread; ++i) partial += grad[col + i];
  }
  for (int col = complete_columns + threadIdx.x; col < width; col += kWideBackwardThreads) {
    partial += grad[col];
  }

  // The vendor combines lanes in order, then combines the warp totals in order.
  float warp_total = 0.0f;
  for (int lane = 0; lane < kWarpSize; ++lane) {
    warp_total += __shfl_sync(kWarpMask, partial, lane);
  }
  if (threadIdx.x % kWarpSize == 0) warp_totals[threadIdx.x / kWarpSize] = warp_total;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int warp = 0; warp < warps; ++warp) total += warp_totals[warp];
    warp_totals[0] = total;
  }
  __syncthreads();
  const float total = warp_totals[0];
  for (int col = threadIdx.x; col < width; col += kWideBackwardThreads) {
    output[col] = fmaf(-expf(log_prob[col]), total, grad[col]);
  }
}

__device__ inline float BFloat16ToFloat(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ inline uint16_t FloatToBFloat16(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x7fffffff) > 0x7f800000
      ? 0x7fffu : (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
}

constexpr int kLeadingColumns = 128;
constexpr int kLeadingVectorWidth = 4;
constexpr int kLeadingColumnsPerBlock = 32;
constexpr int kLeadingRowsPerLane = 16;
constexpr int kLeadingLanes = 4;
constexpr int kLeadingMaxPartitions = 32;

__device__ float SumFourLanes(float a, float b, float c, float d) {
  return (a + c) + (b + d);
}

__global__ void SumLeadingBFloat16Rows(const uint16_t* input, uint16_t* output,
                                      int partitions) {
  // The stock reduction splits physical rows across 16 or 32 blocks. Keep
  // the same partition and combine order inside each output column tile.
  __shared__ float partial[kLeadingMaxPartitions][kLeadingLanes][kLeadingColumnsPerBlock];
  const int partition = threadIdx.z;
  const int lane = threadIdx.y;
  const int tile_column = threadIdx.x * kLeadingVectorWidth;
  const int column = blockIdx.x * kLeadingColumnsPerBlock + tile_column;
  const int stride = partitions * kLeadingLanes;
  const int first_row = partition * kLeadingLanes + lane;
  float accum[kLeadingVectorWidth][kLeadingVectorWidth] = {};
  #pragma unroll
  for (int iteration = 0; iteration < kLeadingRowsPerLane / kLeadingVectorWidth; ++iteration) {
    #pragma unroll
    for (int vector = 0; vector < kLeadingVectorWidth; ++vector) {
      const int row = first_row + (iteration * kLeadingVectorWidth + vector) * stride;
      const auto packed = *reinterpret_cast<const ushort4*>(input + row * kLeadingColumns + column);
      accum[vector][0] += BFloat16ToFloat(packed.x);
      accum[vector][1] += BFloat16ToFloat(packed.y);
      accum[vector][2] += BFloat16ToFloat(packed.z);
      accum[vector][3] += BFloat16ToFloat(packed.w);
    }
  }
  #pragma unroll
  for (int i = 0; i < kLeadingVectorWidth; ++i) {
    partial[partition][lane][tile_column + i] =
        ((accum[0][i] + accum[1][i]) + accum[2][i]) + accum[3][i];
  }
  __syncthreads();
  if (lane == 0) {
    #pragma unroll
    for (int i = 0; i < kLeadingVectorWidth; ++i) {
      const int col = tile_column + i;
      partial[partition][0][col] = SumFourLanes(partial[partition][0][col],
          partial[partition][1][col], partial[partition][2][col], partial[partition][3][col]);
    }
  }
  __syncthreads();
  if (partition == 0) {
    #pragma unroll
    for (int i = 0; i < kLeadingVectorWidth; ++i) {
      float subtotal = 0.0f;
      for (int block = lane; block < partitions; block += kLeadingLanes) {
        subtotal += partial[block][0][tile_column + i];
      }
      partial[0][lane][tile_column + i] = subtotal;
    }
  }
  __syncthreads();
  if (partition == 0 && lane == 0) {
    #pragma unroll
    for (int i = 0; i < kLeadingVectorWidth; ++i) {
      const int col = tile_column + i;
      output[column + i] = FloatToBFloat16(SumFourLanes(
          partial[0][0][col], partial[0][1][col], partial[0][2][col], partial[0][3][col]));
    }
  }
}

__global__ void SiluElements(const uint16_t* input, uint16_t* output, int count) {
  const auto index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float value = BFloat16ToFloat(input[index]);
  // Direct division matches the stock rounding at the BF16 midpoint.
  output[index] = FloatToBFloat16(value / (1.0f + expf(-value)));
}

__global__ void SiluBackwardElements(const uint16_t* grad, const uint16_t* input,
                                     uint16_t* output, int count) {
  const auto index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float value = BFloat16ToFloat(input[index]);
  const float dy = BFloat16ToFloat(grad[index]);
  const float sigmoid = 1.0f / (1.0f + expf(-value));
  output[index] = FloatToBFloat16((dy * sigmoid) * (1.0f + value * (1.0f - sigmoid)));
}

int Blocks(int count, int tile) {
  return 1 + (count - 1) / tile;
}

}  // namespace

cudaError_t ReduceLastDim(const float* input, float* output, int rows, int width,
                          bool mean, cudaStream_t stream) {
  const dim3 block(kWarpSize, kReductionRowsPerBlock);
  const int grid = Blocks(rows, kReductionRowsPerBlock);
  if (mean) {
    ReduceRows<true><<<grid, block, 0, stream>>>(input, output, rows, width);
  } else {
    ReduceRows<false><<<grid, block, 0, stream>>>(input, output, rows, width);
  }
  return cudaGetLastError();
}

cudaError_t SumLeadingBFloat16(const uint16_t* input, uint16_t* output, int rows,
                               cudaStream_t stream) {
  const int partitions = rows / (kLeadingRowsPerLane * kLeadingLanes);
  const dim3 block(kLeadingColumnsPerBlock / kLeadingVectorWidth, kLeadingLanes, partitions);
  SumLeadingBFloat16Rows<<<kLeadingColumns / kLeadingColumnsPerBlock, block, 0, stream>>>(
      input, output, partitions);
  return cudaGetLastError();
}

cudaError_t Rsqrt(const float* input, float* output, int count, cudaStream_t stream) {
  InverseSquareRoot<<<Blocks(count, kElementwiseThreads), kElementwiseThreads, 0, stream>>>(
      input, output, count);
  return cudaGetLastError();
}

cudaError_t Softmax(const float* input, float* output, int rows, cudaStream_t stream) {
  const dim3 block(kWarpSize, kAttentionRowsPerBlock);
  AttentionSoftmaxRows<<<Blocks(rows, kAttentionRowsPerBlock), block, 0, stream>>>(
      input, output, rows);
  return cudaGetLastError();
}

cudaError_t SoftmaxBackward(const float* grad, const float* probability, float* output,
                            int rows, cudaStream_t stream) {
  const dim3 block(kWarpSize, kAttentionRowsPerBlock);
  AttentionSoftmaxBackwardRows<<<Blocks(rows, kAttentionRowsPerBlock), block, 0, stream>>>(
      grad, probability, output, rows);
  return cudaGetLastError();
}

cudaError_t WideLogSoftmax(const float* input, float* output, int rows, int width,
                           cudaStream_t stream) {
  WideLogSoftmaxRows<<<rows, kWideThreads, 0, stream>>>(input, output, width);
  return cudaGetLastError();
}

cudaError_t WideLogSoftmaxBackward(const float* grad, const float* log_prob, float* output,
                                   int rows, int width, cudaStream_t stream) {
  WideLogSoftmaxBackwardRows<<<rows, kWideBackwardThreads, 0, stream>>>(
      grad, log_prob, output, width);
  return cudaGetLastError();
}

cudaError_t Silu(const uint16_t* input, uint16_t* output, int count, cudaStream_t stream) {
  SiluElements<<<Blocks(count, kElementwiseThreads), kElementwiseThreads, 0, stream>>>(
      input, output, count);
  return cudaGetLastError();
}

cudaError_t SiluBackward(const uint16_t* grad, const uint16_t* input, uint16_t* output,
                         int count, cudaStream_t stream) {
  SiluBackwardElements<<<Blocks(count, kElementwiseThreads), kElementwiseThreads, 0, stream>>>(
      grad, input, output, count);
  return cudaGetLastError();
}

}  // namespace at::native::flagos::thead_eager
