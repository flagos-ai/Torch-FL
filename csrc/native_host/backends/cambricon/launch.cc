// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "launch_params.h"

#include <cn_api.h>

namespace py = pybind11;

namespace flagos::native_host::backend {
namespace {

class MluKernel final : public Kernel {
 public:
  explicit MluKernel(py::handle metadata)
      : warps_(py::cast<unsigned>(metadata.attr("num_warps"))),
        function_class_(py::cast<unsigned>(metadata.attr("func_type"))) {
    if (!warps_ || !function_class_ ||
        py::cast<unsigned>(metadata.attr("shared")) != 0) {
      throw std::invalid_argument("unsupported Cambricon Triton launch metadata");
    }
    scratch_slots = 1;
  }

  void PackScratch(uint64_t, uint64_t, Slot* slots) const override {
    slots[0].Set(uint64_t{0});
  }

  void Enqueue(const Launch& launch) const override {
    const auto status = cnInvokeKernel(
        reinterpret_cast<CNkernel>(launch.function),
        GridSize(CheckedProduct(launch.grid[0], warps_)), GridSize(launch.grid[1]),
        GridSize(launch.grid[2]), static_cast<KernelClass>(function_class_), 0,
        reinterpret_cast<CNqueue>(launch.stream), launch.arguments, nullptr);
    if (status != CN_SUCCESS) {
      const char* message = nullptr;
      cnGetErrorString(status, &message);
      throw std::runtime_error(message ? message : "Cambricon Triton launch failed");
    }
  }

 private:
  unsigned warps_;
  unsigned function_class_;
};

class MluTensorDescriptor final : public Descriptor {
 public:
  size_t Capacity() const override { return 21; }

  size_t Pack(const DescriptorValue& value, Slot* slots) const override {
    if (!value.rank || value.rank > 5 || value.nan_padding) {
      throw std::invalid_argument("unsupported Cambricon Triton tensor descriptor");
    }
    size_t count = 0;
    slots[count++].Set(value.pointer);
    for (size_t axis = 0; axis < value.rank; ++axis)
      slots[count++].Set(value.shape[axis]);
    for (size_t axis = 0; axis < value.rank; ++axis)
      slots[count++].Set(value.strides[axis]);
    for (size_t axis = 0; axis < value.rank; ++axis) {
      if (value.shape[axis] > INT32_MAX)
        throw std::overflow_error("descriptor dimension exceeds the kernel ABI");
      slots[count++].Set(static_cast<int32_t>(value.shape[axis]));
    }
    for (size_t axis = 0; axis < value.rank; ++axis)
      slots[count++].Set(value.strides[axis]);
    return count;
  }
};

bool IsCapturing(uint64_t stream) {
  CNqueueCaptureStatus status{};
  if (cnQueueIsCapturing(reinterpret_cast<CNqueue>(stream), &status) != CN_SUCCESS)
    throw std::runtime_error("Cambricon queue capture query failed");
  return status != CN_QUEUE_CAPTURE_STATUS_NONE;
}

const Api api{
    kAbiVersion, sizeof(Api), "cambricon", "mlu",
    SoftwareDescriptor | StreamCaptureQuery, 256,
    [](_object* metadata) -> std::unique_ptr<Kernel> {
      return std::make_unique<MluKernel>(py::handle(metadata));
    },
    [](_object*, _object*) -> std::shared_ptr<const Descriptor> {
      return std::make_shared<MluTensorDescriptor>();
    },
    &IsCapturing};

}  // namespace

const Api& CompiledBackend() { return api; }

}  // namespace flagos::native_host::backend
