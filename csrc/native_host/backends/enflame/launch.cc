// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// TOPS submission mechanics, derived from the published FlagTree launcher.
// All numerical device code continues to come from the installed FlagGems wheel.
#include "backend_api.h"
#include <tops/tops_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <dlfcn.h>

namespace py = pybind11;
namespace flagos::native_host::backend {
namespace {

void Check(topsError_t status) {
  if (status != topsSuccess) throw std::runtime_error(topsGetErrorString(status));
}

class Submission {
 public:
  explicit Submission(void* stream) : stream_(stream) {
    static auto begin = reinterpret_cast<int (*)(void*, uint32_t)>(
        dlsym(RTLD_DEFAULT, "FlagosEnflameBeginLaunch"));
    static auto end = reinterpret_cast<int (*)(void*)>(
        dlsym(RTLD_DEFAULT, "FlagosEnflameEndLaunch"));
    if (!begin || !end) throw std::runtime_error("missing Enflame stream interop ABI");
    end_ = end;
    Check(static_cast<topsError_t>(begin(stream_, 2)));
  }
  ~Submission() { if (active_) end_(stream_); }
  void Finish() {
    active_ = false;
    Check(static_cast<topsError_t>(end_(stream_)));
  }
 private:
  void* stream_;
  int (*end_)(void*);
  bool active_ = true;
};

class EnflameKernel final : public Kernel {
 public:
  explicit EnflameKernel(py::handle metadata) {
    arch_ = py::cast<std::string>(metadata.attr("arch"));
    warps_ = py::cast<unsigned>(metadata.attr("num_warps"));
    shared_ = py::cast<unsigned>(metadata.attr("shared"));
    cluster_ = py::cast<std::array<unsigned, 3>>(
        py::getattr(metadata, "cluster_dims", py::make_tuple(1, 1, 1)));
    cooperative_ = py::cast<bool>(py::getattr(metadata, "launch_cooperative_grid", py::bool_(false)));
    redundant_ = py::cast<bool>(py::getattr(metadata, "redundant_sip", py::bool_(false)));
    pointers_ = py::cast<std::vector<size_t>>(
        py::getattr(metadata, "ptr_int_args", py::tuple()));
    if (arch_ != "gcu300" && arch_ != "gcu400" && arch_ != "gcu410" && arch_ != "gcu500")
      throw std::invalid_argument("unsupported Enflame launch architecture");
    if (!warps_ || !cluster_[0] || !cluster_[1] || cluster_[2] != 1 ||
        CheckedProduct(cluster_[0], cluster_[1]) > 6)
      throw std::invalid_argument("invalid Enflame launch geometry");
    if (arch_ != "gcu400" && arch_ != "gcu410" &&
        (cooperative_ || redundant_ || cluster_[0] != 1 || cluster_[1] != 1))
      throw std::invalid_argument("Enflame architecture does not support extended launch");
    for (const char* field : {"global_scratch_size", "profile_scratch_size", "launch_pdl"})
      if (py::cast<uint64_t>(py::getattr(metadata, field, py::int_(0))))
        throw std::invalid_argument("unsupported Enflame scratch or programmatic launch");
  }

  void Enqueue(const Launch& value) const override {
    if (value.grid[0] > 65535 || value.grid[1] > 255 || value.grid[2] > 255)
      throw std::invalid_argument("Enflame launch grid exceeds published limits");
    // The published compiler marks integer arguments that carry generic
    // pointers. Resolve each current-call address, never retain device pointers.
    for (auto position : pointers_) {
      uint64_t pointer, translated = 0;
      std::memcpy(&pointer, value.arguments[position], sizeof(pointer));
      if (pointer && topsPointerGetAttribute(&translated, TOPS_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                            reinterpret_cast<void*>(pointer)) == topsSuccess && translated)
        std::memcpy(value.arguments[position], &translated, sizeof(translated));
    }
    auto function = reinterpret_cast<topsFunction_t>(value.function);
    auto stream = reinterpret_cast<topsStream_t>(value.stream);
    Submission submission(stream);
    if (arch_ == "gcu400" || arch_ == "gcu410") {
      topsLaunchAttribute attrs[2]{};
      attrs[0].id = topsLaunchAttributeThreadDimension;
      attrs[0].val.ThreadDim = {warps_, 1, 1};
      unsigned count = 1;
      if (redundant_) {
        attrs[count].id = topsLaunchAttributeSchedulingPolicy;
        attrs[count++].val.SchedulingPolicyPreference = topsSchedulingPolicyRedundant;
      } else if (cooperative_) {
        attrs[count].id = topsLaunchAttributeCooperative;
        attrs[count++].val.cooperative = 1;
      }
      topsLaunchConfig_t config{};
      config.gridDim = {static_cast<unsigned>(value.grid[0]), static_cast<unsigned>(value.grid[1]),
                        static_cast<unsigned>(value.grid[2])};
      config.blockDim = redundant_ ? dim3(1, 1, 1) : dim3(cluster_[0], cluster_[1], cluster_[2]);
      config.dynamicSmemBytes = redundant_ ? 0 : shared_;
      config.stream = stream;
      config.attrs = attrs;
      config.numAttrs = count;
      Check(topsModuleLaunchKernelEx(&config, function, value.arguments, nullptr));
    } else {
      const auto threads = CheckedProduct(warps_, arch_ == "gcu500" ? 128 : 1);
      if (threads > UINT32_MAX) throw std::overflow_error("Enflame block size overflow");
      Check(topsModuleLaunchKernel(function, value.grid[0], value.grid[1], value.grid[2],
                                  threads, 1, 1, shared_, stream, value.arguments, nullptr));
    }
    submission.Finish();
  }

 private:
  std::string arch_;
  unsigned warps_, shared_;
  std::array<unsigned, 3> cluster_;
  bool cooperative_, redundant_;
  std::vector<size_t> pointers_;
};

class EnflameDescriptor final : public Descriptor {
 public:
  explicit EnflameDescriptor(py::handle metadata) {
    if (!metadata.is_none()) throw std::invalid_argument("Enflame requires software descriptors");
  }
  size_t Capacity() const override { return 22; }
  size_t Pack(const DescriptorValue& value, Slot* slots) const override {
    if (!value.rank || value.rank > 5 || value.strides[value.rank - 1] != 1)
      throw std::invalid_argument("invalid Enflame descriptor dimensions");
    size_t count = 0;
    slots[count++].Set(value.pointer);
    for (size_t i = 0; i < value.rank; ++i) slots[count++].Set(value.shape[i]);
    for (size_t i = 0; i < value.rank; ++i) slots[count++].Set(value.strides[i]);
    slots[count++].Set(static_cast<uint8_t>(value.nan_padding));
    for (size_t i = 0; i < value.rank; ++i) {
      if (value.shape[i] > INT32_MAX) throw std::overflow_error("descriptor dimension exceeds kernel ABI");
      slots[count++].Set(static_cast<int32_t>(value.shape[i]));
    }
    for (size_t i = 0; i < value.rank; ++i) slots[count++].Set(value.strides[i]);
    return count;
  }
};

bool IsCapturing(uint64_t stream) {
  topsStreamCaptureStatus status;
  Check(topsStreamIsCapturing(reinterpret_cast<topsStream_t>(stream), &status));
  return status != topsStreamCaptureStatusNone;
}

const Api api{
    kAbiVersion, sizeof(Api), "enflame", "gcu",
    SoftwareDescriptor | CooperativeLaunch | ClusterLaunch | StreamCaptureQuery, 256,
    [](_object* metadata) -> std::unique_ptr<Kernel> {
      return std::make_unique<EnflameKernel>(py::handle(metadata));
    },
    [](_object* metadata, _object*) -> std::shared_ptr<const Descriptor> {
      return std::make_shared<EnflameDescriptor>(py::handle(metadata));
    }, &IsCapturing};
}  // namespace
const Api& CompiledBackend() { return api; }
}  // namespace flagos::native_host::backend
