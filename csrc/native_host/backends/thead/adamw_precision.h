// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace at::native::flagos::thead_adamw {

cudaError_t Addcmul(uint16_t* result, const uint16_t* first, const uint16_t* second,
                    int count, float alpha, cudaStream_t stream);
cudaError_t Addcdiv(uint16_t* result, const uint16_t* first, const uint16_t* second,
                    int count, float alpha, cudaStream_t stream);

}  // namespace at::native::flagos::thead_adamw
