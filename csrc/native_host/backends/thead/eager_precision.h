/* Copyright 2026 FlagOS Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace at::native::flagos::thead_eager {

cudaError_t ReduceLastDim(const float* input, float* output, int rows, int width,
                          bool mean, cudaStream_t stream);
cudaError_t SumLeadingBFloat16(const uint16_t* input, uint16_t* output, int rows,
                               cudaStream_t stream);
cudaError_t Rsqrt(const float* input, float* output, int count, cudaStream_t stream);
cudaError_t Softmax(const float* input, float* output, int rows, cudaStream_t stream);
cudaError_t SoftmaxBackward(const float* grad, const float* probability, float* output,
                            int rows, cudaStream_t stream);
cudaError_t WideLogSoftmax(const float* input, float* output, int rows, int width,
                           cudaStream_t stream);
cudaError_t WideLogSoftmaxBackward(const float* grad, const float* log_prob, float* output,
                                   int rows, int width, cudaStream_t stream);
cudaError_t Silu(const uint16_t* input, uint16_t* output, int count, cudaStream_t stream);
cudaError_t SiluBackward(const uint16_t* grad, const uint16_t* input, uint16_t* output,
                         int count, cudaStream_t stream);

}  // namespace at::native::flagos::thead_eager
