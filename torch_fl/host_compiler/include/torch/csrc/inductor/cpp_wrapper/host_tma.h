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

#pragma once
#include <cuda.h>
#include <cstdint>
#include <stdexcept>

// Native representation of a symbolic TensorDescriptor. Kernel-specific
// physical tiles and swizzle come from the compiled Triton metadata, not from
// the logical block shape or an operator-specific launch implementation.
struct FlagosTMADescriptor {
  alignas(64) CUtensorMap m;
  uint32_t block_shape[5];
  uint64_t global_shape[5];
  uint64_t strides[5];
  void* base_ptr = nullptr;
};

inline void flagos_init_tma(FlagosTMADescriptor& descriptor, int rank,
                            int element_size, int element_type, int swizzle,
                            const uint32_t* physical_block) {
  uint64_t shape[5], stride[5];
  uint32_t block[5], element_stride[5] = {1, 1, 1, 1, 1};
  for (int i = 0; i < rank; ++i) {
    shape[rank - i - 1] = descriptor.global_shape[i];
    block[rank - i - 1] = physical_block[i];
  }
  for (int i = 0; i + 1 < rank; ++i)
    stride[rank - i - 2] = element_size * descriptor.strides[i];
  stride[rank - 1] = shape[rank - 1] * (rank == 1 ? element_size : stride[rank - 2]);
  auto result = cuTensorMapEncodeTiled(
      &descriptor.m, static_cast<CUtensorMapDataType>(element_type), rank,
      descriptor.base_ptr, shape, stride, block, element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, static_cast<CUtensorMapSwizzle>(swizzle),
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    const char* error = nullptr;
    cuGetErrorString(result, &error);
    throw std::runtime_error(error ? error : "TMA descriptor creation failed");
  }
}
