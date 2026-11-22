// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "launch_params.h"

#include <musa.h>
#include <pybind11/stl.h>

#include <optional>
#include <vector>

namespace py = pybind11;

namespace flagos::native_host::backend {
namespace {

void Check(MUresult status) {
  if (status == MUSA_SUCCESS) return;
  const char* error = nullptr;
  muGetErrorString(status, &error);
  throw std::runtime_error(error ? error : "MUSA driver operation failed");
}

class MusaKernel final : public WarpKernel {
 public:
  explicit MusaKernel(py::handle metadata)
      : WarpKernel(metadata), warp_size_(py::cast<unsigned>(metadata.attr("warp_size"))) {
    if (!warp_size_ || cooperative_ || pdl_)
      throw std::invalid_argument("unsupported MUSA launch geometry or capability");
    // The published launcher sizes global scratch by logical grid, before CTA
    // expansion. Its second scratch slot is a zero profile pointer.
    scratch_multiplier = 1;
  }

  void Enqueue(const Launch& launch) const override {
    static_assert(sizeof(MUdeviceptr) == sizeof(uint64_t));
    auto function = reinterpret_cast<MUfunction>(launch.function);
    auto stream = reinterpret_cast<MUstream>(launch.stream);
    const auto block_x = GridSize(CheckedProduct(warps_, warp_size_));
    const auto grid_x = GridSize(CheckedProduct(launch.grid[0], ctas_));
    const auto grid_y = GridSize(launch.grid[1]);
    const auto grid_z = GridSize(launch.grid[2]);
    if (ctas_ == 1) {
      Check(muLaunchKernel(function, grid_x, grid_y, grid_z, block_x, 1, 1,
                           shared_, stream, launch.arguments, nullptr));
      return;
    }

    MUlaunchAttribute attrs[2]{};
    attrs[0].id = MU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attrs[0].value.clusterDim.x = ctas_;
    attrs[0].value.clusterDim.y = attrs[0].value.clusterDim.z = 1;
    attrs[1].id = MU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    attrs[1].value.clusterSchedulingPolicyPreference = MU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    MUlaunchConfig config{};
    config.gridDimX = grid_x;
    config.gridDimY = grid_y;
    config.gridDimZ = grid_z;
    config.blockDimX = block_x;
    config.blockDimY = config.blockDimZ = 1;
    config.sharedMemBytes = shared_;
    config.hStream = stream;
    config.attrs = attrs;
    config.numAttrs = 2;
    Check(muLaunchKernelEx(&config, function, launch.arguments, nullptr));
  }

 private:
  unsigned warp_size_;
};

// The driver passes a by-value TME descriptor followed by shape and strides.
class TmeDescriptor final : public Descriptor {
 public:
  explicit TmeDescriptor(py::handle metadata) {
    auto values = py::cast<py::dict>(metadata);
    const auto block = py::cast<std::vector<uint32_t>>(values["block_size"]);
    rank_ = block.size();
    element_bytes_ = py::cast<uint64_t>(values["elem_size"]);
    type_ = static_cast<MUtensorDescriptorDataType>(py::cast<int>(values["elem_type"]));
    nan_fill_ = py::cast<std::optional<uint64_t>>(values["nan_fill"]);
    if (!rank_ || rank_ > 5 || !element_bytes_)
      throw std::invalid_argument("invalid MUSA tensor descriptor metadata");
    uint64_t block_bytes = element_bytes_;
    for (auto extent : block) block_bytes = CheckedProduct(block_bytes, extent);
    if (block_bytes < 32) throw std::invalid_argument("MUSA descriptor block must span 32 bytes");
  }

  size_t Capacity() const override { return 1 + 2 * rank_; }

  size_t Pack(const DescriptorValue& value, Slot* slots) const override {
    if (value.rank != rank_)
      throw std::invalid_argument("MUSA tensor descriptor rank mismatch");
    if (value.nan_padding && !nan_fill_)
      throw std::invalid_argument("MUSA NaN padding requires a floating-point descriptor");
    uint64_t shape[5]{}, byte_strides[5]{};
    uint64_t stride = 1;
    for (size_t i = 0; i < rank_; ++i) {
      const auto dim = rank_ - 1 - i;
      const auto extent = value.shape[dim];
      if (extent > INT32_MAX) throw std::overflow_error("descriptor dimension exceeds the kernel ABI");
      if (extent > 1 && value.strides[dim] != stride)
        throw std::invalid_argument("MUSA hardware tensor descriptors require dense storage");
      shape[i] = extent;
      stride = CheckedProduct(stride, extent);
      byte_strides[i] = CheckedProduct(stride, element_bytes_);
    }
    alignas(64) MUtensorDescriptor descriptor{};
    Check(muTensorDescriptorEncode(&descriptor, type_, rank_,
                                   reinterpret_cast<void*>(value.pointer), shape, byte_strides,
                                   MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE,
                                   value.nan_padding ? *nan_fill_ : 0));
    slots[0].Set(descriptor);
    for (size_t i = 0; i < rank_; ++i) slots[i + 1].Set(static_cast<int32_t>(value.shape[i]));
    for (size_t i = 0; i < rank_; ++i) slots[i + 1 + rank_].Set(value.strides[i]);
    return Capacity();
  }

 private:
  size_t rank_;
  uint64_t element_bytes_;
  MUtensorDescriptorDataType type_;
  std::optional<uint64_t> nan_fill_;
};

bool IsCapturing(uint64_t stream) {
  MUstreamCaptureStatus status;
  Check(muStreamIsCapturing(reinterpret_cast<MUstream>(stream), &status));
  return status != MU_STREAM_CAPTURE_STATUS_NONE;
}

const Api api{
    kAbiVersion, sizeof(Api), "mthreads", "musa",
    SoftwareDescriptor | HardwareDescriptor | GlobalScratch | ClusterLaunch | StreamCaptureQuery,
    256,
    [](_object* metadata) -> std::unique_ptr<Kernel> {
      return std::make_unique<MusaKernel>(py::handle(metadata));
    },
    [](_object* metadata, _object*) -> std::shared_ptr<const Descriptor> {
      py::handle values(metadata);
      if (values.is_none()) return std::make_shared<SoftwareTensorDescriptor>(values);
      return std::make_shared<TmeDescriptor>(values);
    },
    &IsCapturing};

}  // namespace

const Api& CompiledBackend() { return api; }

}  // namespace flagos::native_host::backend
