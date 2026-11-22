// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backend_api.h"

#include <pybind11/pybind11.h>

namespace flagos::native_host::backend {

inline unsigned GridSize(uint64_t value) {
  if (value > std::numeric_limits<unsigned>::max())
    throw std::overflow_error("Triton grid dimension overflow");
  return static_cast<unsigned>(value);
}

// Common metadata and trailing scratch parameters for the published warp-based
// Triton launchers. Each driver defines its own launch attributes and handles.
class WarpKernel : public Kernel {
 public:
  explicit WarpKernel(pybind11::handle metadata) {
    namespace py = pybind11;
    warps_ = py::cast<unsigned>(metadata.attr("num_warps"));
    ctas_ = py::cast<unsigned>(py::getattr(metadata, "num_ctas", py::int_(1)));
    shared_ = py::cast<unsigned>(metadata.attr("shared"));
    cooperative_ =
        py::cast<bool>(py::getattr(metadata, "launch_cooperative_grid", py::bool_(false)));
    pdl_ = py::cast<bool>(py::getattr(metadata, "launch_pdl", py::bool_(false)));
    global = {py::cast<uint64_t>(metadata.attr("global_scratch_size")),
              py::cast<uint64_t>(metadata.attr("global_scratch_align"))};
    profile = {py::cast<uint64_t>(metadata.attr("profile_scratch_size")),
               py::cast<uint64_t>(metadata.attr("profile_scratch_align"))};
    if (!warps_ || !ctas_) throw std::invalid_argument("invalid Triton launch geometry");
    scratch_multiplier = ctas_;
    scratch_slots = 2;
  }

  void PackScratch(uint64_t global_pointer, uint64_t profile_pointer, Slot* slots) const override {
    slots[0].Set(global_pointer);
    slots[1].Set(profile_pointer);
  }

 protected:
  unsigned warps_, ctas_, shared_;
  bool cooperative_, pdl_;
};

constexpr size_t kSoftwareDescriptorSlots = 22;

inline size_t PackSoftwareDescriptor(const DescriptorValue& value, Slot* slots) {
  const auto rank = value.rank;
  if (!rank || rank > 5 || value.strides[rank - 1] != 1)
    throw std::invalid_argument("invalid Triton tensor descriptor dimensions");
  size_t count = 0;
  slots[count++].Set(value.pointer);
  for (size_t i = 0; i < rank; ++i) slots[count++].Set(value.shape[i]);
  for (size_t i = 0; i < rank; ++i) slots[count++].Set(value.strides[i]);
  slots[count++].Set(static_cast<uint8_t>(value.nan_padding));
  for (size_t i = 0; i < rank; ++i) {
    if (value.shape[i] > INT32_MAX)
      throw std::overflow_error("descriptor dimension exceeds the kernel ABI");
    slots[count++].Set(static_cast<int32_t>(value.shape[i]));
  }
  for (size_t i = 0; i < rank; ++i) slots[count++].Set(value.strides[i]);
  return count;
}

class SoftwareTensorDescriptor final : public Descriptor {
 public:
  explicit SoftwareTensorDescriptor(pybind11::handle metadata) {
    if (!metadata.is_none())
      throw std::invalid_argument("native host backend requires software tensor descriptors");
  }

  size_t Capacity() const override { return kSoftwareDescriptorSlots; }

  size_t Pack(const DescriptorValue& value, Slot* slots) const override {
    return PackSoftwareDescriptor(value, slots);
  }
};

}  // namespace flagos::native_host::backend
