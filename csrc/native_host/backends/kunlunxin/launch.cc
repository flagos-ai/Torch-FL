// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "backend_api.h"

#include <c10/util/SmallVector.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xpu/runtime.h>

#include <string>
#include <tuple>
#include <vector>

// This entry point is linked from the installed FlagTree XPU3 launch library.
int xpuLaunchKernel(const char* name, void* function, int x, int y, int z,
                    int clusters, int cores, void* stream, void** parameters,
                    void** constants, void** extra);

namespace py = pybind11;

namespace flagos::native_host::backend {
namespace {

int Dimension(uint64_t value) {
  if (value > std::numeric_limits<int>::max())
    throw std::overflow_error("XPU launch dimension overflow");
  return static_cast<int>(value);
}

class XpuKernel final : public Kernel {
 public:
  explicit XpuKernel(py::handle metadata)
      : name_(py::cast<std::string>(metadata.attr("name"))),
        geometry_(py::cast<std::array<int, 2>>(metadata.attr("_flagos_xpu_geometry"))),
        parameters_(py::cast<std::vector<Parameter>>(metadata.attr("_flagos_xpu_parameters"))),
        ewtable_(py::cast<std::string>(metadata.attr("_flagos_xpu_ewtable"))) {}

  void Enqueue(const Launch& launch) const override {
    static_assert(sizeof(void*) == sizeof(int64_t));
    c10::SmallVector<std::array<int64_t, 5>, 16> records(parameters_.size() + 1);
    c10::SmallVector<int32_t, 8> booleans(parameters_.size());
    for (size_t i = 0; i < parameters_.size(); ++i) {
      const auto [bytes, type, widen_bool] = parameters_[i];
      const void* value = launch.arguments[i];
      if (widen_bool) {
        booleans[i] = *static_cast<const uint8_t*>(value);
        value = &booleans[i];
      }
      records[i] = {static_cast<int64_t>(i), reinterpret_cast<int64_t>(value),
                    bytes, 0, type};
    }
    int64_t constants[3]{};
    void* extra[] = {ewtable_.empty() ? nullptr : const_cast<char*>(ewtable_.data())};
    const int status = xpuLaunchKernel(
        name_.c_str(), reinterpret_cast<void*>(launch.function),
        Dimension(launch.grid[0]), Dimension(launch.grid[1]), Dimension(launch.grid[2]),
        geometry_[0], geometry_[1], reinterpret_cast<void*>(launch.stream),
        reinterpret_cast<void**>(records.data()), reinterpret_cast<void**>(constants),
        extra);
    if (status != XPU_SUCCESS)
      throw std::runtime_error(std::string("XPU launch failed: ") + xpu_strerror(status));
  }

 private:
  using Parameter = std::tuple<int64_t, int64_t, bool>;
  std::string name_;
  std::array<int, 2> geometry_;
  std::vector<Parameter> parameters_;
  std::string ewtable_;
};

class XpuDescriptor final : public Descriptor {
 public:
  explicit XpuDescriptor(py::handle rank) : rank_(py::cast<size_t>(rank)) {
    if (!rank_ || rank_ > 5) throw std::invalid_argument("invalid XPU descriptor rank");
  }

  size_t Capacity() const override { return 4 * rank_ + 2; }

  size_t Pack(const DescriptorValue& value, Slot* slots) const override {
    if (value.rank != rank_ || value.strides[rank_ - 1] != 1)
      throw std::invalid_argument("invalid XPU tensor descriptor dimensions");
    size_t count = 0;
    slots[count++].Set(value.pointer);
    for (size_t i = 0; i < rank_; ++i) slots[count++].Set(value.shape[i]);
    for (size_t i = 0; i < rank_; ++i) slots[count++].Set(value.strides[i]);
    slots[count++].Set(static_cast<uint8_t>(value.nan_padding));
    for (size_t i = 0; i < rank_; ++i) {
      if (value.shape[i] > INT32_MAX)
        throw std::overflow_error("descriptor dimension exceeds the kernel ABI");
      slots[count++].Set(static_cast<int32_t>(value.shape[i]));
    }
    for (size_t i = 0; i < rank_; ++i) slots[count++].Set(value.strides[i]);
    return count;
  }

 private:
  size_t rank_;
};

bool IsCapturing(uint64_t stream) {
  cudaStreamCaptureStatus status;
  const auto result = cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(stream), &status);
  if (result != cudaSuccess)
    throw std::runtime_error(std::string("XPU stream capture query failed: ") +
                             cudaGetErrorString(result));
  return status != cudaStreamCaptureStatusNone;
}

const Api api{
    kAbiVersion, sizeof(Api), "kunlunxin", "xpu", SoftwareDescriptor | StreamCaptureQuery, 128,
    [](_object* metadata) -> std::unique_ptr<Kernel> {
      return std::make_unique<XpuKernel>(py::handle(metadata));
    },
    [](_object* rank, _object*) -> std::shared_ptr<const Descriptor> {
      return std::make_shared<XpuDescriptor>(py::handle(rank));
    },
    &IsCapturing};

}  // namespace

const Api& CompiledBackend() { return api; }

}  // namespace flagos::native_host::backend
