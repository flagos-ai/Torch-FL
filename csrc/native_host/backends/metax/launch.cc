// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Submission mechanics follow the published MetaX FlagTree launcher. Numerical
// kernels and the prebuilt module loader remain the released implementations.
#include "backend_api.h"

#include <mcr/mc_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/library.h>

namespace py = pybind11;

namespace flagos::native_host::backend {
namespace {

void Check(mcError_t status) {
  if (status != mcSuccess) throw std::runtime_error(mcGetErrorString(status));
}

int64_t DeviceAttribute(mcDeviceAttribute_t attribute) {
  int device;
  Check(mcGetDevice(&device));
  int value;
  Check(mcDeviceGetAttribute(&value, attribute, device));
  return value;
}

TORCH_LIBRARY_FRAGMENT(flagos_source_native, m) {
  m.def("get_sm_count() -> int",
        [] { return DeviceAttribute(mcDeviceAttributeMultiProcessorCount); });
  m.def("get_l2_cache_size() -> int",
        [] { return DeviceAttribute(mcDeviceAttributeL2CacheSize); });
}

unsigned Dimension(uint64_t value) {
  if (value > std::numeric_limits<unsigned>::max())
    throw std::overflow_error("MetaX launch dimension overflow");
  return static_cast<unsigned>(value);
}

class MetaxKernel final : public Kernel {
 public:
  explicit MetaxKernel(py::handle metadata) {
    warps_ = py::cast<unsigned>(metadata.attr("num_warps"));
    shared_ = py::cast<unsigned>(metadata.attr("shared"));
    const auto ctas = py::cast<unsigned>(py::getattr(metadata, "num_ctas", py::int_(1)));
    const auto cluster = py::cast<std::array<unsigned, 3>>(
        py::getattr(metadata, "cluster_dims", py::make_tuple(1, 1, 1)));
    if (!warps_ || ctas != 1 || cluster != std::array<unsigned, 3>{1, 1, 1} ||
        py::cast<bool>(py::getattr(metadata, "launch_cooperative_grid", py::bool_(false))) ||
        py::cast<bool>(py::getattr(metadata, "launch_pdl", py::bool_(false))))
      throw std::invalid_argument("unsupported MetaX launch geometry or capability");
    global = {py::cast<uint64_t>(metadata.attr("global_scratch_size")),
              py::cast<uint64_t>(metadata.attr("global_scratch_align"))};
    profile = {py::cast<uint64_t>(metadata.attr("profile_scratch_size")),
               py::cast<uint64_t>(metadata.attr("profile_scratch_align"))};
    scratch_slots = 2;
    static_assert(sizeof(mcDeviceptr_t) == sizeof(uint64_t));
  }

  void PackScratch(uint64_t global_pointer, uint64_t profile_pointer, Slot* slots) const override {
    slots[0].Set(global_pointer);
    slots[1].Set(profile_pointer);
  }

  void Enqueue(const Launch& launch) const override {
    Check(mcModuleLaunchKernel(
        reinterpret_cast<mcFunction_t>(launch.function),
        Dimension(launch.grid[0]), Dimension(launch.grid[1]), Dimension(launch.grid[2]),
        Dimension(CheckedProduct(warps_, 64)), 1, 1, shared_,
        reinterpret_cast<mcStream_t>(launch.stream), launch.arguments, nullptr));
  }

 private:
  unsigned warps_, shared_;
};

class MetaxDescriptor final : public Descriptor {
 public:
  explicit MetaxDescriptor(py::handle metadata) {
    if (!metadata.is_none())
      throw std::invalid_argument("MetaX hardware tensor descriptors are unsupported");
  }

  size_t Capacity() const override { return 22; }

  size_t Pack(const DescriptorValue& value, Slot* slots) const override {
    if (!value.rank || value.rank > 5 || value.strides[value.rank - 1] != 1)
      throw std::invalid_argument("invalid MetaX software descriptor dimensions");
    size_t count = 0;
    slots[count++].Set(value.pointer);
    for (size_t i = 0; i < value.rank; ++i) slots[count++].Set(value.shape[i]);
    for (size_t i = 0; i < value.rank; ++i) slots[count++].Set(value.strides[i]);
    slots[count++].Set(static_cast<uint8_t>(value.nan_padding));
    for (size_t i = 0; i < value.rank; ++i) {
      if (value.shape[i] > INT32_MAX)
        throw std::overflow_error("descriptor dimension exceeds the kernel ABI");
      slots[count++].Set(static_cast<int32_t>(value.shape[i]));
    }
    for (size_t i = 0; i < value.rank; ++i) slots[count++].Set(value.strides[i]);
    return count;
  }
};

bool IsCapturing(uint64_t stream) {
  mcStreamCaptureStatus status;
  Check(mcStreamIsCapturing(reinterpret_cast<mcStream_t>(stream), &status));
  return status != mcStreamCaptureStatusNone;
}

const Api api{
    kAbiVersion, sizeof(Api), "metax", "maca",
    SoftwareDescriptor | GlobalScratch | ProfileScratch | StreamCaptureQuery,
    256,
    [](_object* metadata) -> std::unique_ptr<Kernel> {
      return std::make_unique<MetaxKernel>(py::handle(metadata));
    },
    [](_object* metadata, _object*) -> std::shared_ptr<const Descriptor> {
      return std::make_shared<MetaxDescriptor>(py::handle(metadata));
    },
    &IsCapturing};

}  // namespace

const Api& CompiledBackend() { return api; }

}  // namespace flagos::native_host::backend
