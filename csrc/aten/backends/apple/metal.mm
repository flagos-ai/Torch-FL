// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "metal.h"

#include <ATen/native/mps/MetalShaderLibrary.h>
#include <ATen/ops/empty.h>
#include <runtime/accelerator/apple/mps.h>
#include <runtime/allocator/caching_device_allocator.h>
#include <runtime/functions.h>

#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace at::native::flagos::mps {
namespace {

using ::flagos::native_host::EnqueueTrace;

MTLSize Size(const MetalGrid& size) {
  return MTLSizeMake(size[0], size[1], size[2]);
}

struct BufferView {
  id<MTLBuffer> buffer;
  size_t offset;
};

BufferView ResolveBuffer(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.device().is_privateuseone() && tensor.device().index() == 0,
              "Metal kernels require FlagOS tensors on device 0");
  size_t offset = 0;
  auto buffer = (id<MTLBuffer>)::FlagosMPSBufferForPointer(tensor.data_ptr(), &offset);
  TORCH_CHECK(buffer != nil, "Metal kernel argument has no FlagOS buffer");
  return {buffer, offset};
}

MetalGrid GroupSize(id<MTLComputePipelineState> pipeline, const MetalGrid& grid,
                   const std::optional<MetalGrid>& requested) {
  auto capacity = pipeline.maxTotalThreadsPerThreadgroup;
  if (!requested) {
    return {std::min<uint64_t>(grid[0], capacity), 1, 1};
  }
  for (auto width : *requested) {
    TORCH_CHECK(width > 0 && width <= capacity, "Invalid Metal threadgroup size");
    capacity /= width;
  }
  return *requested;
}

void Submit(id<MTLComputePipelineState> pipeline, c10::ArrayRef<MetalArg> args,
            const MetalGrid& grid, const std::optional<MetalGrid>& group_size,
            bool compiled_host) {
  if (std::any_of(grid.begin(), grid.end(), [](auto width) { return width == 0; })) {
    return;
  }
  for (auto width : grid) {
    TORCH_CHECK(width <= std::numeric_limits<uint32_t>::max(), "Metal grid exceeds its index width");
  }

  @autoreleasepool {
    std::vector<std::variant<BufferView, int64_t, c10::IntArrayRef>> bindings;
    bindings.reserve(args.size());
    for (const auto& arg : args) {
      if (const auto* tensor = std::get_if<at::Tensor>(&arg)) {
        bindings.emplace_back(ResolveBuffer(*tensor));
      } else if (const auto* value = std::get_if<int64_t>(&arg)) {
        bindings.emplace_back(*value);
      } else {
        bindings.emplace_back(std::get<c10::IntArrayRef>(arg));
      }
    }
    const auto group = GroupSize(pipeline, grid, group_size);
    const auto handle = c10::flagos::CurrentStreamHandle(0);
    auto stream = reinterpret_cast<Stream_t>(handle);
    auto queue = (id<MTLCommandQueue>)::FlagosMPSCommandQueue(stream);
    TORCH_CHECK(queue != nil, "FlagOS Metal command queue is unavailable");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    TORCH_CHECK(encoder != nil, "Failed to encode the FlagOS Metal kernel");
    [encoder setComputePipelineState:pipeline];
    for (size_t index = 0; index < bindings.size(); ++index) {
      const auto& binding = bindings[index];
      if (const auto* view = std::get_if<BufferView>(&binding)) {
        [encoder setBuffer:view->buffer offset:view->offset atIndex:index];
      } else if (const auto* value = std::get_if<int64_t>(&binding)) {
        [encoder setBytes:value length:sizeof(*value) atIndex:index];
      } else {
        const auto values = std::get<c10::IntArrayRef>(binding);
        [encoder setBytes:values.data()
                   length:values.size() * sizeof(int64_t)
                  atIndex:index];
      }
    }
    [encoder dispatchThreads:Size(grid) threadsPerThreadgroup:Size(group)];
    [encoder endEncoding];
    ::FlagosMPSCommitCommandBuffer(stream, command);

    if (auto* allocator = c10::flagos::GetCachingAllocator()) {
      for (const auto& arg : args) {
        if (const auto* tensor = std::get_if<at::Tensor>(&arg)) {
          allocator->record_stream(tensor->storage().data_ptr(), stream);
        }
      }
    }
    EnqueueTrace::record(handle, compiled_host);
  }
}

id<MTLComputePipelineState> GetBundledPipeline(const std::string& name) {
  static std::mutex mutex;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
  std::lock_guard<std::mutex> lock(mutex);
  if (const auto it = pipelines.find(name); it != pipelines.end()) {
    return it->second;
  }
  auto& library = at::native::mps::MetalShaderLibrary::getBundledLibrary();
  auto pipeline = library.getPipelineStateForFunc(name);
  pipelines.emplace(name, pipeline);
  return pipeline;
}

const char* MetalFloatName(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat: return "float";
    case at::kHalf: return "half";
    case at::kBFloat16: return "bfloat";
    default: return nullptr;
  }
}

template <typename T>
int64_t MetalScalarBytes(T value) {
  int64_t bytes = 0;
  std::memcpy(&bytes, &value, sizeof(value));
  return bytes;
}

} // namespace

struct MetalKernel::Impl {
  explicit Impl(const std::string& source)
      : library(source), pipeline(library.getPipelineStateForFunc("generated_kernel")) {}

  at::native::mps::DynamicMetalShaderLibrary library;
  id<MTLComputePipelineState> pipeline;
};

MetalKernel::MetalKernel(const std::string& source) {
  @autoreleasepool {
    impl_ = std::make_unique<Impl>(source);
  }
}

MetalKernel::~MetalKernel() = default;

void MetalKernel::run(c10::ArrayRef<MetalArg> args, const MetalGrid& threads,
                      const std::optional<MetalGrid>& group_size,
                      bool compiled_host) const {
  Submit(impl_->pipeline, args, threads, group_size, compiled_host);
}

bool RunMetalUnary(const at::Tensor& out, const at::Tensor& input,
                   const char* name) {
  const char* dtype = MetalFloatName(input.scalar_type());
  if (dtype == nullptr || input.numel() > std::numeric_limits<uint32_t>::max()) return false;

  @autoreleasepool {
    const std::string function = std::string(name) + "_dense_" + dtype + "_" + dtype;
    const auto pipeline = GetBundledPipeline(function);
    const std::array<MetalArg, 2> args{out, input};
    Submit(pipeline, args, {static_cast<uint64_t>(input.numel()), 1, 1},
           std::nullopt, true);
  }
  return true;
}

bool RunMetalBinaryAlpha(const at::Tensor& out, const at::Tensor& input,
                        const at::Tensor& other, const at::Scalar& alpha,
                        const char* name) {
  const auto type = input.scalar_type();
  const char* dtype = MetalFloatName(type);
  if (dtype == nullptr || input.numel() > std::numeric_limits<uint32_t>::max()) return false;
  int64_t bytes;
  switch (type) {
    case at::kFloat: bytes = MetalScalarBytes(alpha.toFloat()); break;
    case at::kHalf: bytes = MetalScalarBytes(alpha.toHalf()); break;
    case at::kBFloat16: bytes = MetalScalarBytes(alpha.toBFloat16()); break;
    default: return false;
  }
  @autoreleasepool {
    const std::string function = std::string(name) + "_alpha_dense_" + dtype + "_" + dtype + "_" + dtype;
    const auto pipeline = GetBundledPipeline(function);
    const std::array<MetalArg, 4> args{out, input, other, bytes};
    Submit(pipeline, args, {static_cast<uint64_t>(input.numel()), 1, 1},
           std::nullopt, true);
  }
  return true;
}

bool RunMetalFp32ScaledAdd(const at::Tensor& out, const at::Tensor& input,
                           const at::Tensor& left, const at::Tensor& right,
                           const at::Scalar& alpha, const char* term_name) {
  if (input.scalar_type() != at::kFloat ||
      input.numel() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  // Reuse the bundled dense kernels on devices that cannot run MPSGraph.
  const auto term = at::empty(input.sizes(), input.options());
  @autoreleasepool {
    const auto pipeline = GetBundledPipeline(std::string(term_name) + "_dense_float_float");
    const std::array<MetalArg, 3> args{term, left, right};
    Submit(pipeline, args, {static_cast<uint64_t>(input.numel()), 1, 1},
           std::nullopt, true);
  }
  return RunMetalBinaryAlpha(out, input, term, alpha, "add");
}

void RunMetalTernaryAlpha(const MetalKernel& kernel, const at::Tensor& out,
                          const at::Tensor& input, const at::Tensor& left,
                          const at::Tensor& right, float alpha) {
  const auto count = static_cast<uint64_t>(out.numel());
  const std::array<MetalArg, 5> args{out, input, left, right, MetalScalarBytes(alpha)};
  kernel.run(args, {count, 1, 1}, std::nullopt, true);
}

EnqueueTrace::Events CaptureEnqueues(const std::function<void()>& run) {
  EnqueueTrace trace;
  run();
  return trace.events();
}

} // namespace at::native::flagos::mps
