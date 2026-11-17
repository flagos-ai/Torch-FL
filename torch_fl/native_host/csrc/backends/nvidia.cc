// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Backend mechanics only. Device code remains the published FlagGems kernels;
// module loading remains the published FlagTree driver's prebuilt utilities.
#include "../backend_api.h"

#include <cuda.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

namespace py = pybind11;

namespace flagos::native_host::backend {
namespace {

void Check(CUresult status) {
  if (status != CUDA_SUCCESS) {
    const char* error = nullptr;
    cuGetErrorString(status, &error);
    throw std::runtime_error(error ? error : "CUDA driver operation failed");
  }
}

unsigned GridSize(uint64_t value) {
  if (value > std::numeric_limits<unsigned>::max())
    throw std::overflow_error("Triton grid dimension overflow");
  return static_cast<unsigned>(value);
}

class NvidiaKernel final : public Kernel {
 public:
  explicit NvidiaKernel(py::handle metadata) {
    warps_ = py::cast<unsigned>(metadata.attr("num_warps"));
    ctas_ = py::cast<unsigned>(py::getattr(metadata, "num_ctas", py::int_(1)));
    shared_ = py::cast<unsigned>(metadata.attr("shared"));
    cluster_ = py::cast<std::array<unsigned, 3>>(
        py::getattr(metadata, "cluster_dims", py::make_tuple(1, 1, 1)));
    cooperative_ =
        py::cast<bool>(py::getattr(metadata, "launch_cooperative_grid", py::bool_(false)));
    pdl_ = py::cast<bool>(py::getattr(metadata, "launch_pdl", py::bool_(false)));
    global = {py::cast<uint64_t>(metadata.attr("global_scratch_size")),
              py::cast<uint64_t>(metadata.attr("global_scratch_align"))};
    profile = {py::cast<uint64_t>(metadata.attr("profile_scratch_size")),
               py::cast<uint64_t>(metadata.attr("profile_scratch_align"))};
    scratch_multiplier = ctas_;
    scratch_slots = 2;
    if (!warps_ || !ctas_ || !cluster_[0] || !cluster_[1] || !cluster_[2])
      throw std::invalid_argument("invalid NVIDIA launch geometry");
  }

  void PackScratch(uint64_t global_pointer, uint64_t profile_pointer, Slot* slots) const override {
    slots[0].Set(global_pointer);
    slots[1].Set(profile_pointer);
  }

  void Enqueue(const Launch& launch) const override {
    const auto& dims = launch.grid;
    auto handle = reinterpret_cast<CUfunction>(launch.function);
    auto stream = reinterpret_cast<CUstream>(launch.stream);
    if (cooperative_ && launch.global_scratch)
      Check(cuMemsetD8Async(launch.global_scratch, 0, launch.global_bytes, stream));
    CUlaunchAttribute attrs[4]{};
    unsigned count = 0;
    if (pdl_) {
      attrs[count].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      attrs[count++].value.programmaticStreamSerializationAllowed = 1;
    }
    if (cooperative_) {
      attrs[count].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
      attrs[count++].value.cooperative = 1;
    }
    const auto cluster = CheckedProduct(CheckedProduct(cluster_[0], cluster_[1]), cluster_[2]);
    if (cluster != 1) {
      attrs[count].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attrs[count].value.clusterDim.x = cluster_[0];
      attrs[count].value.clusterDim.y = cluster_[1];
      attrs[count++].value.clusterDim.z = cluster_[2];
      attrs[count].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      attrs[count++].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    }
    if (ctas_ == 16 || cluster == 16)
      Check(cuFuncSetAttribute(handle, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
    CUlaunchConfig config{};
    config.gridDimX = GridSize(CheckedProduct(CheckedProduct(dims[0], ctas_), cluster_[0]));
    config.gridDimY = GridSize(CheckedProduct(dims[1], cluster_[1]));
    config.gridDimZ = GridSize(CheckedProduct(dims[2], cluster_[2]));
    config.blockDimX = GridSize(CheckedProduct(warps_, 32));
    config.blockDimY = config.blockDimZ = 1;
    config.sharedMemBytes = shared_;
    config.hStream = stream;
    config.attrs = attrs;
    config.numAttrs = count;
    if (!launch.extended && count == 0) {
      Check(cuLaunchKernel(handle, config.gridDimX, config.gridDimY, config.gridDimZ,
                          config.blockDimX, config.blockDimY, config.blockDimZ,
                          config.sharedMemBytes, stream, launch.arguments, nullptr));
    } else
      Check(cuLaunchKernelEx(&config, handle, launch.arguments, nullptr));
  }

 private:
  unsigned warps_, ctas_, shared_;
  std::array<unsigned, 3> cluster_;
  bool cooperative_, pdl_;
};

class NvidiaDescriptor final : public Descriptor {
 public:
  explicit NvidiaDescriptor(py::handle metadata) : hardware_(!metadata.is_none()) {
    if (!hardware_) return;
    auto values = py::cast<py::dict>(metadata);
    block_ = py::cast<std::vector<uint32_t>>(values["block_size"]);
    element_size_ = py::cast<uint64_t>(values["elem_size"]);
    element_type_ = py::cast<int>(values["elem_type"]);
    swizzle_ = py::cast<int>(values["swizzle"]);
    fp4_padded_ = py::cast<bool>(values["fp4_padded"]);
    if (block_.empty() || block_.size() > 5)
      throw std::invalid_argument("invalid NVIDIA descriptor block rank");
  }

  size_t Capacity() const override { return hardware_ ? 1 + 2 * block_.size() : 22; }

  size_t Pack(const DescriptorValue& value, Slot* slots) const override {
    const size_t rank = value.rank;
    if (!rank || rank > 5 || value.strides[rank - 1] != 1)
      throw std::invalid_argument("invalid Triton tensor descriptor dimensions");
    uint64_t shape[5]{};
    for (size_t i = 0; i < rank; ++i) shape[i] = value.shape[i];
    size_t count = 0;
    if (!hardware_) {
      slots[count++].Set(value.pointer);
      for (size_t i = 0; i < rank; ++i) slots[count++].Set(shape[i]);
      for (size_t i = 0; i < rank; ++i) slots[count++].Set(value.strides[i]);
      slots[count++].Set(static_cast<uint8_t>(value.nan_padding));
    } else {
      if (block_.size() != rank) throw std::invalid_argument("TMA descriptor block rank mismatch");
      if (fp4_padded_) shape[rank - 1] = CheckedProduct(shape[rank - 1], 2);
      uint64_t reversed_shape[5]{}, reversed_stride[5]{};
      uint32_t reversed_block[5]{}, element_stride[5]{1, 1, 1, 1, 1};
      for (size_t i = 0; i < rank; ++i) {
        reversed_shape[rank - i - 1] = shape[i];
        reversed_block[rank - i - 1] = block_[i];
        if (i + 1 < rank)
          reversed_stride[rank - i - 2] = CheckedProduct(value.strides[i], element_size_);
      }
      int element_type = element_type_;
      if (element_type == 8)
        element_type = 10;
      else if (element_type == 9)
        element_type = 8;
      else if (element_type == 10)
        element_type = 9;
      static_assert(sizeof(CUtensorMap) <= kSlotBytes && alignof(CUtensorMap) <= kSlotAlignment);
      Check(cuTensorMapEncodeTiled(
          reinterpret_cast<CUtensorMap*>(slots[count++].bytes.data()),
          static_cast<CUtensorMapDataType>(element_type), rank,
          reinterpret_cast<void*>(value.pointer), reversed_shape, reversed_stride,
          reversed_block, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
          static_cast<CUtensorMapSwizzle>(swizzle_), CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
          value.nan_padding ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                            : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    for (size_t i = 0; i < rank; ++i) {
      if (shape[i] > INT32_MAX)
        throw std::overflow_error("descriptor dimension exceeds kernel ABI");
      slots[count++].Set(static_cast<int32_t>(shape[i]));
    }
    for (size_t i = 0; i < rank; ++i) slots[count++].Set(value.strides[i]);
    return count;
  }

 private:
  bool hardware_, fp4_padded_ = false;
  std::vector<uint32_t> block_;
  uint64_t element_size_ = 0;
  int element_type_ = 0, swizzle_ = 0;
};

bool IsCapturing(uint64_t stream) {
  CUstreamCaptureStatus status;
  Check(cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &status));
  return status != CU_STREAM_CAPTURE_STATUS_NONE;
}

const Api api{
    kAbiVersion, sizeof(Api), "nvidia", "cuda",
    SoftwareDescriptor | HardwareDescriptor | GlobalScratch | ProfileScratch |
        CooperativeLaunch | ClusterLaunch | ProgrammaticLaunch | StreamCaptureQuery,
    256,
    [](_object* metadata) -> std::unique_ptr<Kernel> {
      return std::make_unique<NvidiaKernel>(py::handle(metadata));
    },
    [](_object* metadata) -> std::shared_ptr<const Descriptor> {
      return std::make_shared<NvidiaDescriptor>(py::handle(metadata));
    },
    &IsCapturing};

}  // namespace

const Api& CompiledBackend() { return api; }

}  // namespace flagos::native_host::backend
