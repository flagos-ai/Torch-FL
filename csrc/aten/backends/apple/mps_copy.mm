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

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "mps_common.h"

#include <flagos.h>

#include "runtime/accelerator/apple/mps.h"
#include "runtime/accelerator/apple/mps_runtime.h"
#include "runtime/allocator/caching_device_allocator.h"

namespace at::native::flagos::mps {

namespace torch_fl_mps {

void DrainAll();

} // namespace torch_fl_mps

namespace {

using MpsShape = std::vector<int64_t>;

// The source layout and the compiled gather. The graph's index tensor is the
// only layout-specific input; changing a shape or stride creates a new plan.
struct StridedCopyPlan {
  MPSGraph* graph = nil;
  MPSGraphTensor* result = nil;
  MPSGraphExecutable* executable = nil;
  MPSGraphExecutableExecutionDescriptor* descriptor = nil;
};

// One MPSGraph is not documented as thread-safe. Holding this lock through the
// run also keeps a plan alive while another thread evicts the oldest entry.
std::mutex& PlansMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<std::string, std::shared_ptr<StridedCopyPlan>>& PlanCache() {
  static auto* plans =
      new std::unordered_map<std::string, std::shared_ptr<StridedCopyPlan>>();
  return *plans;
}

std::deque<std::string>& PlanOrder() {
  static auto* order = new std::deque<std::string>();
  return *order;
}

// A model can present a different shape, offset and stride for each view. Keep
// the same bound as the operator plans so these graphs cannot pile up.
constexpr size_t kMaxPlans = 64;

Stream_t CurrentStream() {
  int device = 0;
  if (::GetDevice(&device) != ::Success) {
    return nullptr;
  }
  return ::GetCurrentStreamForDevice(device);
}

void ReleasePlan(StridedCopyPlan* plan) {
  MpsRelease(plan->descriptor);
  MpsRelease(plan->executable);
  MpsRelease(plan->result);
  MpsRelease(plan->graph);
  delete plan;
}

std::string PlanKey(
    const at::Tensor& src,
    int64_t storage_numel,
    id<MTLDevice> device) {
  std::string key = "strided-to-dense";
  MpsKeyAppend(key, storage_numel);
  MpsKeyAppend(key, src.sizes());
  MpsKeyAppend(key, src.storage_offset());
  MpsKeyAppend(key, src.strides());
  MpsKeyAppend(key, static_cast<int64_t>(src.scalar_type()));
  MpsKeyAppend(key, static_cast<int64_t>(device.registryID));
  return key;
}

// True for one dense run that starts at the view's storage offset. Any trailing
// extent-1 axes are skipped: an empty view has no such run, and over-strided
// singleton axes do not make its one element less dense.
bool DensePrefix(at::IntArrayRef sizes, at::IntArrayRef strides) {
  if (sizes.empty()) {
    return true;
  }
  int64_t last = static_cast<int64_t>(sizes.size()) - 1;
  while (last > 0 && sizes[last] == 1) {
    --last;
  }
  if (sizes[last] == 0) {
    return false;
  }
  int64_t expected = 1;
  for (int64_t axis = last; axis >= 0; --axis) {
    if (sizes[axis] > 1 && strides[axis] != expected) {
      return false;
    }
    expected = strides[axis] * std::max<int64_t>(sizes[axis], 1);
  }
  return true;
}

MPSGraphTensor* Int64Constant(MPSGraph* graph, int64_t value) {
  return [graph constantWithScalar:static_cast<double>(value)
                              shape:@[@1]
                           dataType:MPSDataTypeInt64];
}

// Builds the gather and compiles it for this Metal device. Called
// with PlansMutex held because MPSGraph is not documented as thread-safe.
std::shared_ptr<StridedCopyPlan> BuildPlan(
    const at::Tensor& src,
    int64_t storage_numel,
    MPSDataType dtype,
    MPSGraphDevice* device) {
  const auto sizes = src.sizes();
  const auto strides = src.strides();
  const int64_t numel = src.numel();
  if (sizes.size() != strides.size()) {
    return nullptr;
  }
  // A dense prefix starting at any storage element is one device slice, so it
  // needs no coordinate graph at all. The generic path folds each logical axis
  // into the flat source order. There, a negative stride is addressed by its
  // absolute stride plus a per-axis coordinate delta.
  const bool dense_prefix = DensePrefix(sizes, strides);

  auto plan = std::make_unique<StridedCopyPlan>();
  @autoreleasepool {
    @try {
      MPSGraph* graph = [[MPSGraph alloc] init];
      if (graph == nil) {
        return nullptr;
      }
      MPSGraphTensor* flat = MpsFlatPlaceholder(graph, storage_numel, dtype);
      MPSGraphTensor* result = nil;
      if (dense_prefix) {
        result = [graph sliceTensor:flat
                          dimension:0
                              start:src.storage_offset()
                             length:numel
                               name:nil];
        result = [graph reshapeTensor:result withShape:MpsShapeOf(
            sizes.empty() ? MpsShape{1} : MpsShape(sizes.begin(), sizes.end()))
                                 name:nil];
      }
      // Each axis' coordinate is built with the view's shape, then multiplied
      // by its absolute stride. A negative axis is offset so that coordinate 0
      // reads its last element. Summing the axis terms before flattening yields
      // one index into the storage per output element.
      MPSGraphTensor* all_indices = nil;
      std::vector<MPSGraphTensor*> axis_indices;
      std::vector<MPSGraphTensor*> axis_coordinates;
      if (!dense_prefix) {
        for (size_t axis = 0; axis < sizes.size(); ++axis) {
          MPSGraphTensor* coordinate = [graph
              coordinateAlongAxis:static_cast<NSInteger>(axis)
                         withShape:MpsShapeOf(sizes)
                              name:nil];
          axis_coordinates.push_back(coordinate);
          MPSGraphTensor* scaled = [graph
              multiplicationWithPrimaryTensor:[graph castTensor:coordinate
                                                        toType:MPSDataTypeInt64
                                                          name:nil]
                             secondaryTensor:Int64Constant(
                                 graph, std::abs(strides[axis]))
                                        name:nil];
          axis_indices.push_back(scaled);
        }
      }
      if (!dense_prefix) {
        // A negative axis starts at its last element: for size s, coordinate c
        // reads (s - 1 - c) * -stride elements past the view's first element.
        for (size_t axis = 0; axis < sizes.size(); ++axis) {
          if (strides[axis] < 0) {
            MPSGraphTensor* delta = [graph
                multiplicationWithPrimaryTensor:axis_coordinates[axis]
                               secondaryTensor:Int64Constant(graph, -2)
                                          name:nil];
            delta = [graph additionWithPrimaryTensor:delta
                                     secondaryTensor:Int64Constant(
                                                         graph,
                                                         sizes[axis] - 1)
                                                name:nil];
            delta = [graph multiplicationWithPrimaryTensor:delta
                                           secondaryTensor:
                                               Int64Constant(
                                                   graph, std::abs(strides[axis]))
                                                      name:nil];
            axis_indices[axis] = [graph
                additionWithPrimaryTensor:axis_indices[axis]
                          secondaryTensor:delta
                                     name:nil];
          }
        }
        all_indices = axis_indices[0];
        for (size_t axis = 1; axis < sizes.size(); ++axis) {
          all_indices = [graph additionWithPrimaryTensor:all_indices
                                         secondaryTensor:axis_indices[axis]
                                                    name:nil];
        }
        MPSGraphTensor* flat_indices =
            [MpsFlatten(graph, all_indices, numel) retain];
        result = [graph gatherWithUpdatesTensor:flat
                                  indicesTensor:flat_indices
                                           axis:0
                                batchDimensions:0
                                           name:nil];
        MpsRelease(flat_indices);
      }
      result = result == nil ? nil : [MpsFlatten(graph, result, numel) retain];
      MPSGraphShapedType* source_type = [[MPSGraphShapedType alloc]
          initWithShape:MpsShapeOf({storage_numel})
               dataType:dtype];
      MPSGraphExecutable* executable =
          result == nil
          ? nil
          : [[graph compileWithDevice:device
                                feeds:@{flat: source_type}
                        targetTensors:@[ result ]
                     targetOperations:nil
                compilationDescriptor:nil] retain];
      MpsRelease(source_type);
      if (executable == nil) {
        MpsRelease(result);
        MpsRelease(graph);
        return nullptr;
      }
      plan->graph = graph;
      plan->result = result;
      plan->executable = executable;
      plan->descriptor = [[MPSGraphExecutableExecutionDescriptor alloc] init];
      plan->descriptor.waitUntilCompleted = NO;
    } @catch (NSException* exception) {
      TORCH_WARN("mps strided copy declined: ", exception.reason.UTF8String);
      return nullptr;
    }
  }
  return std::shared_ptr<StridedCopyPlan>(plan.release());
}

} // namespace

bool CopyStridedToDense(const at::Tensor& src, const at::Tensor& dst) {
  if (!src.defined() || !dst.defined() || !src.is_privateuseone() ||
      !dst.is_privateuseone() || src.device() != dst.device() ||
      src.numel() <= 0 || dst.numel() != src.numel() ||
      !src.sizes().equals(dst.sizes()) ||
      src.scalar_type() != dst.scalar_type() || src.layout() != c10::kStrided ||
      dst.layout() != c10::kStrided || !dst.is_contiguous()) {
    return false;
  }
  MPSDataType dtype;
  if (!MpsDataTypeOf(src.scalar_type(), &dtype) ||
      !::torch_fl_mps::MetalGraphUsable()) {
    return false;
  }
  id<MTLCommandQueue> queue = MpsCurrentQueue();
  TORCH_CHECK(
      queue != nil, "mps: no Metal command queue for the flagos device");
  id<MTLBuffer> src_buffer = nil;
  id<MTLBuffer> dst_buffer = nil;
  size_t src_offset = 0;
  size_t dst_offset = 0;
  if (!MpsBufferOf(src, &src_buffer, &src_offset) ||
      !MpsBufferOf(dst, &dst_buffer, &dst_offset)) {
    return false;
  }
  const int64_t numel = src.numel();
  const int64_t storage_numel =
      src.storage().nbytes() / src.element_size();
  if (src.strides().empty()) {
    // The scalar has no logical axes: its only element is its storage value,
    // wherever that lies inside a parent allocation. Two raw MTLBuffer offsets
    // are available without an MPSGraphTensorData description.
    if (src.storage_offset() < 0 ||
        src.storage_offset() >= storage_numel) {
      return false;
    }
    @autoreleasepool {
      // A single-element run goes through a blit, without the coordinate graph.
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
      [encoder copyFromBuffer:src_buffer
                 sourceOffset:src_offset
                     toBuffer:dst_buffer
            destinationOffset:dst_offset
                         size:static_cast<NSUInteger>(src.element_size())];
      [encoder endEncoding];
      [command commit];
    }
    const Stream_t stream = CurrentStream();
    ::FlagosMPSStreamSentinel(stream);
    if (c10::flagos::CachingDeviceAllocator::is_enabled()) {
      if (auto* alloc = c10::flagos::GetCachingAllocator()) {
        alloc->record_stream(src.storage().data_ptr(), stream);
        alloc->record_stream(dst.storage().data_ptr(), stream);
      }
    }
    return true;
  }
  if ((src_offset != 0 || dst_offset != 0) &&
      !MpsOffsetTensorDataAvailable()) {
    // Older SDKs can describe only a buffer from byte zero. A staging copy here
    // would need asynchronous lifetimes the plan cache does not own, so use the
    // caller's existing host fallback.
    return false;
  }
  const std::string key = PlanKey(src, storage_numel, queue.device);
  std::shared_ptr<StridedCopyPlan> plan;
  {
    std::lock_guard<std::mutex> lock(PlansMutex());
    const auto found = PlanCache().find(key);
    if (found != PlanCache().end()) {
      plan = found->second;
    } else {
      @autoreleasepool {
        plan = BuildPlan(
            src,
            storage_numel,
            dtype,
            [MPSGraphDevice deviceWithMTLDevice:queue.device]);
      }
      if (plan == nullptr) {
        return false;
      }
      if (PlanCache().size() >= kMaxPlans) {
        auto oldest = PlanCache().find(PlanOrder().front());
        if (oldest != PlanCache().end()) {
          std::shared_ptr<StridedCopyPlan> evicted = oldest->second;
          PlanCache().erase(oldest);
        }
        PlanOrder().pop_front();
      }
      PlanCache().emplace(key, plan);
      PlanOrder().push_back(key);
    }
  }

  bool ran = false;
  @autoreleasepool {
    MPSGraphTensorData* source =
        MpsTensorData(src_buffer, src_offset, storage_numel, dtype);
    MPSGraphTensorData* target =
        MpsTensorData(dst_buffer, dst_offset, numel, dtype);
    if (source != nil && target != nil) {
      // Ordered on the current queue after every producer that uses that
      // stream, and read by consumers that follow it there.
      [plan->executable runAsyncWithMTLCommandQueue:queue
                                        inputsArray:@[ source ]
                                       resultsArray:@[ target ]
                                executionDescriptor:plan->descriptor];
      ran = true;
    }
    MpsRelease(source);
    MpsRelease(target);
  }
  if (ran) {
    const Stream_t stream = CurrentStream();
    ::FlagosMPSStreamSentinel(stream);
    if (c10::flagos::CachingDeviceAllocator::is_enabled()) {
      if (auto* alloc = c10::flagos::GetCachingAllocator()) {
        alloc->record_stream(src.storage().data_ptr(), stream);
        alloc->record_stream(dst.storage().data_ptr(), stream);
      }
    }
  }
  return ran;
}

} // namespace at::native::flagos::mps
