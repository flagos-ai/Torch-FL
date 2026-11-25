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

#include "mps_copy.h"

#include <ATen/mps/IndexKernels.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <c10/util/SmallVector.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>

#import <Metal/Metal.h>

#include "mps_common.h"
#include "runtime/accelerator/apple/mps.h"
#include "runtime/allocator/caching_device_allocator.h"

namespace at::native::flagos::mps {
namespace {

std::string GatherTemplateWithOffsets() {
  std::string source(at::mps::GATHER_OPS_TEMPLATE);
  // Keep the upstream indexing and specialize the two pointer origins. Metal
  // requires aligned buffer bindings even when the view starts inside a word.
  const auto replace = [&source](std::string_view from, std::string_view to) {
    size_t pos = 0;
    int replaced = 0;
    while ((pos = source.find(from, pos)) != std::string::npos) {
      source.replace(pos, from.size(), to);
      pos += to.size();
      ++replaced;
    }
    return replaced;
  };
  // The input and output have the same element type, so the type and cast
  // parameters can be shared and the slots reused for the two offsets.
  replace("{1}", "{0}");
  if (replace("{2}", "x") != 1 ||
      replace("*)src_;", "*)src_ + {1};") != 5 ||
      replace("*)dst_;", "*)dst_ + {2};") != 5) {
    return {};
  }
  return source;
}

} // namespace

bool CopyStridedToDense(const at::Tensor& src, const at::Tensor& dst) {
  if (!src.defined() || !dst.defined() || !src.is_privateuseone() ||
      src.device() != dst.device() || !src.sizes().equals(dst.sizes()) ||
      src.scalar_type() != dst.scalar_type() || src.layout() != c10::kStrided ||
      dst.layout() != c10::kStrided || !dst.is_contiguous() || src.numel() == 0) {
    return false;
  }

  // PyTorch's specialized gather shaders index views with 32-bit operands.
  constexpr auto max_index = std::numeric_limits<int32_t>::max();
  if (src.numel() > max_index ||
      src.storage().nbytes() / src.element_size() > max_index) {
    return false;
  }
  const char* element;
  switch (src.element_size()) {
    case 1: element = "uchar"; break;
    case 2: element = "ushort"; break;
    case 4: element = "uint"; break;
    case 8: element = "ulong"; break;
    default: return false;
  }

  c10::SmallVector<uint32_t, 4> sizes;
  c10::SmallVector<uint32_t, 4> strides;
  for (int64_t dim = 0; dim < src.dim(); ++dim) {
    const int64_t stride = src.stride(dim);
    if (stride < 0 || stride > max_index) return false;
    sizes.push_back(static_cast<uint32_t>(src.size(dim)));
    strides.push_back(static_cast<uint32_t>(stride));
  }
  if (sizes.empty()) {
    sizes.push_back(1);
    strides.push_back(1);
  }

  id<MTLBuffer> src_buffer = nil;
  id<MTLBuffer> dst_buffer = nil;
  size_t src_offset = 0;
  size_t dst_offset = 0;
  if (!MpsBufferOf(src, &src_buffer, &src_offset) ||
      !MpsBufferOf(dst, &dst_buffer, &dst_offset)) {
    return false;
  }
  constexpr size_t binding_alignment = sizeof(uint32_t);
  const size_t element_size = src.element_size();
  if (src_offset % element_size || dst_offset % element_size) return false;
  const size_t src_prefix = src_offset % binding_alignment;
  const size_t dst_prefix = dst_offset % binding_alignment;
  const auto stream = ::GetCurrentStreamForDevice(src.device().index());
  const auto queue = (id<MTLCommandQueue>)::FlagosMPSCommandQueue(stream);
  TORCH_CHECK(queue != nil, "mps: no Metal command queue for the flagos device");

  const auto count = static_cast<uint32_t>(src.numel());
  const auto rank = static_cast<int32_t>(sizes.size());
  @autoreleasepool {
    // Reuse PyTorch's gather and copy the storage bits without a graph cast.
    static const auto shader = GatherTemplateWithOffsets();
    if (shader.empty()) return false;
    static at::native::mps::MetalShaderLibrary library(shader, 3);
    static std::mutex library_mutex;
    const std::string name = "gather_kernel_" + (rank <= 4 ? std::to_string(rank) : "n");
    id<MTLComputePipelineState> pipeline;
    {
      std::lock_guard<std::mutex> lock(library_mutex);
      pipeline = library.getPipelineStateForFunc(
          name, {element, std::to_string(src_prefix / element_size),
                 std::to_string(dst_prefix / element_size)});
    }
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    TORCH_CHECK(encoder != nil, "mps: could not encode a strided copy");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:src_buffer offset:src_offset - src_prefix atIndex:0];
    [encoder setBuffer:dst_buffer offset:dst_offset - dst_prefix atIndex:1];
    [encoder setBytes:sizes.data() length:sizes.size() * sizeof(uint32_t) atIndex:2];
    [encoder setBytes:strides.data() length:strides.size() * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&count length:sizeof(count) atIndex:4];
    if (rank > 4) [encoder setBytes:&rank length:sizeof(rank) atIndex:5];
    const auto group = std::min<NSUInteger>(count, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    [encoder endEncoding];
    ::FlagosMPSCommitCommandBuffer(stream, command);
  }

  if (c10::flagos::CachingDeviceAllocator::is_enabled()) {
    auto* allocator = c10::flagos::GetCachingAllocator();
    allocator->record_stream(src.storage().data_ptr(), stream);
    allocator->record_stream(dst.storage().data_ptr(), stream);
  }
  return true;
}

} // namespace at::native::flagos::mps
