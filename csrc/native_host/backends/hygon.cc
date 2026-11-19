// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Launch metadata and submission only; numerical kernels and module loading
// remain the published FlagGems / FlagTree providers.
#include "../backend_api.h"

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>

namespace py = pybind11;

namespace flagos::native_host::backend {
namespace {

#include "hcu_descriptor.h"

struct Driver {
  void* library = dlopen(FLAGOS_HCU_RUNTIME_LIBRARY, RTLD_NOW | RTLD_LOCAL);
  decltype(&hipModuleLaunchKernel) launch;
  using Cooperative = hipError_t (*)(hipFunction_t, unsigned, unsigned, unsigned,
                                     unsigned, unsigned, unsigned, unsigned,
                                     hipStream_t, void**, void**);
  Cooperative cooperative;
  decltype(&hipStreamIsCapturing) capturing;
  decltype(&hipGetErrorString) error;
  using Extended = hipError_t (*)(const HIP_LAUNCH_CONFIG*, hipFunction_t, void**, void**);
  Extended extended;

  template <class T> T Symbol(const char* name, bool required = true) {
    auto result = reinterpret_cast<T>(dlsym(library, name));
    if (!result && required) throw std::runtime_error(std::string("missing HCU driver symbol: ") + name);
    return result;
  }
  Driver() {
    if (!library) throw std::runtime_error("cannot open the build-time DTK runtime");
    launch = Symbol<decltype(launch)>("hipModuleLaunchKernel");
    cooperative = Symbol<decltype(cooperative)>("hipModuleLaunchCooperativeKernel");
    capturing = Symbol<decltype(capturing)>("hipStreamIsCapturing");
    error = Symbol<decltype(error)>("hipGetErrorString");
    extended = Symbol<Extended>("hipDrvLaunchKernelEx", false);
  }
};

const Driver& Hip() { static const Driver driver; return driver; }

void Check(hipError_t status) {
  if (status != hipSuccess) throw std::runtime_error(Hip().error(status));
}

unsigned Dimension(uint64_t value) {
  if (value > std::numeric_limits<unsigned>::max())
    throw std::overflow_error("HCU launch dimension overflow");
  return static_cast<unsigned>(value);
}

class HygonKernel final : public Kernel {
 public:
  explicit HygonKernel(py::handle metadata) {
    warps_ = py::cast<unsigned>(metadata.attr("num_warps"));
    wave_ = py::cast<unsigned>(metadata.attr("warp_size"));
    ctas_ = py::cast<unsigned>(py::getattr(metadata, "num_ctas", py::int_(1)));
    shared_ = py::cast<unsigned>(metadata.attr("shared"));
    cooperative_ = py::cast<bool>(py::getattr(metadata, "launch_cooperative_grid", py::bool_(false)));
    profile = {py::cast<uint64_t>(metadata.attr("profile_scratch_size")),
               py::cast<uint64_t>(metadata.attr("profile_scratch_align"))};
    const auto global_bytes = py::cast<uint64_t>(py::getattr(metadata, "global_scratch_size", py::int_(0)));
    if (global_bytes) throw std::invalid_argument("published HCU ABI does not allocate global scratch");
    scratch_slots = 2;
    if (!warps_ || (wave_ != 32 && wave_ != 64) || !ctas_ ||
        (warps_ & (warps_ - 1)) || CheckedProduct(warps_, wave_) > 1024)
      throw std::invalid_argument("invalid HCU launch geometry");
    if (ctas_ > 1 && !Hip().extended)
      throw std::invalid_argument("DTK lacks hipDrvLaunchKernelEx for a multi-CTA kernel");
  }

  void PackScratch(uint64_t global_pointer, uint64_t profile_pointer, Slot* slots) const override {
    if (global_pointer) throw std::invalid_argument("HCU global scratch must be null");
    slots[0].Set(uint64_t(0));
    slots[1].Set(profile_pointer);
  }

  void Enqueue(const Launch& value) const override {
    const auto x = Dimension(value.grid[0]), y = Dimension(value.grid[1]), z = Dimension(value.grid[2]);
    const auto block = Dimension(CheckedProduct(warps_, wave_));
    auto function = reinterpret_cast<hipFunction_t>(value.function);
    auto stream = reinterpret_cast<hipStream_t>(value.stream);
    if (ctas_ > 1) {
      hipLaunchAttribute attributes[2]{};
      attributes[0].id = static_cast<hipLaunchAttributeID>(4);
      int cluster[3]{static_cast<int>(ctas_), 1, 1};
      std::memcpy(attributes[0].val.pad, cluster, sizeof(cluster));
      attributes[1].id = hipLaunchAttributeCooperative;
      attributes[1].val.cooperative = cooperative_;
      HIP_LAUNCH_CONFIG config{Dimension(CheckedProduct(x, ctas_)), y, z,
                               block, 1, 1, shared_, stream, attributes, 2};
      Check(Hip().extended(&config, function, value.arguments, nullptr));
    } else if (cooperative_)
      Check(Hip().cooperative(function, x, y, z, block, 1, 1, shared_, stream, value.arguments, nullptr));
    else
      Check(Hip().launch(function, x, y, z, block, 1, 1, shared_, stream, value.arguments, nullptr));
  }

 private:
  unsigned warps_, wave_, ctas_, shared_;
  bool cooperative_;
};

class HygonDescriptor final : public Descriptor {
 public:
  HygonDescriptor(py::handle metadata, py::handle kernel) : hardware_(!metadata.is_none()) {
    if (!hardware_) return;
    auto values = py::cast<py::dict>(metadata);
    block_ = py::cast<std::vector<uint32_t>>(values["block_size"]);
    bits_ = py::cast<int>(values["elem_bits"]);
    warps_ = py::cast<int>(kernel.attr("num_warps"));
    auto padding = py::cast<std::vector<std::vector<int>>>(values.contains("interval_padding_pairs")
        ? values["interval_padding_pairs"] : py::list());
    if (block_.size() != 2 || !block_[0] || !block_[1] || warps_ <= 0 ||
        (warps_ & (warps_ - 1)) || (bits_ != 8 && bits_ != 16 && bits_ != 32 && bits_ != 64) ||
        padding.size() > 1 || (!padding.empty() && padding[0].size() != 2))
      throw std::invalid_argument("invalid HCU TDM descriptor metadata");
    if (!padding.empty()) { interval_ = padding[0][0]; amount_ = padding[0][1]; }
    if (interval_ < 0 || amount_ < 0) throw std::invalid_argument("invalid HCU TDM padding");
  }

  size_t Capacity() const override { return hardware_ ? 5 : 22; }

  size_t Pack(const DescriptorValue& value, Slot* slots) const override {
    const auto rank = value.rank;
    if (!rank || rank > 5 || value.strides[rank - 1] != 1)
      throw std::invalid_argument("invalid HCU tensor descriptor dimensions");
    size_t count = 0;
    if (hardware_) {
      if (rank != 2 || value.nan_padding)
        throw std::invalid_argument("HCU TDM requires rank 2 and zero padding");
      uint32_t shape[2], strides[2];
      for (size_t i = 0; i < rank; ++i) {
        if (value.shape[i] > INT32_MAX || value.strides[i] > UINT32_MAX)
          throw std::overflow_error("HCU TDM dimensions exceed the published encoder ABI");
        shape[i] = value.shape[i]; strides[i] = value.strides[i];
      }
      TDMDescriptor encoded{};
      auto block = block_;
      if (!encodeTDMDescriptor(&encoded, bits_, block.data(), warps_, interval_, amount_,
                               shape, strides, value.pointer, rank))
        throw std::invalid_argument("HCU TDM block cannot be distributed across its warps");
      static_assert(sizeof(encoded) <= kSlotBytes);
      slots[count++].Set(encoded);
    } else {
      slots[count++].Set(value.pointer);
      for (size_t i = 0; i < rank; ++i) slots[count++].Set(value.shape[i]);
      for (size_t i = 0; i < rank; ++i) slots[count++].Set(value.strides[i]);
      slots[count++].Set(static_cast<uint8_t>(value.nan_padding));
    }
    for (size_t i = 0; i < rank; ++i) {
      if (value.shape[i] > INT32_MAX) throw std::overflow_error("descriptor dimension exceeds kernel ABI");
      slots[count++].Set(static_cast<int32_t>(value.shape[i]));
    }
    for (size_t i = 0; i < rank; ++i) slots[count++].Set(value.strides[i]);
    return count;
  }

 private:
  bool hardware_;
  std::vector<uint32_t> block_;
  int bits_ = 0, warps_ = 0, interval_ = 0, amount_ = 0;
};

bool IsCapturing(uint64_t stream) {
  hipStreamCaptureStatus status;
  Check(Hip().capturing(reinterpret_cast<hipStream_t>(stream), &status));
  return status != hipStreamCaptureStatusNone;
}

const Api api{kAbiVersion, sizeof(Api), "hygon", "hip",
    SoftwareDescriptor | HardwareDescriptor | ProfileScratch | CooperativeLaunch | ClusterLaunch | StreamCaptureQuery,
    256,
    [](_object* metadata) -> std::unique_ptr<Kernel> { return std::make_unique<HygonKernel>(py::handle(metadata)); },
    [](_object* metadata, _object* kernel) -> std::shared_ptr<const Descriptor> {
      return std::make_shared<HygonDescriptor>(py::handle(metadata), py::handle(kernel));
    }, &IsCapturing};

}  // namespace
const Api& CompiledBackend() { return api; }
}  // namespace flagos::native_host::backend
