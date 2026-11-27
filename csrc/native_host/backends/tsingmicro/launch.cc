// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "launch_params.h"

#include <c10/util/SmallVector.h>
#include <pybind11/stl.h>
#include <tx_runtime.h>

#include <string>
#include <vector>

namespace py = pybind11;

namespace flagos::native_host::backend {
namespace {

class TxKernel final : public Kernel {
 public:
  explicit TxKernel(py::handle metadata)
      : widths_(py::cast<std::vector<unsigned>>(metadata.attr("argument_widths"))) {
    size_t words = 6;
    for (const unsigned width : widths_) {
      if (width != 0 && width != 1 && width != 2 && width != 4 && width != 8)
        throw std::invalid_argument("invalid TsingMicro argument width");
      words += width == 0 ? 2 : 1;
    }
    if (words > UINT32_MAX / sizeof(uint64_t))
      throw std::overflow_error("TsingMicro launch argument buffer overflow");
    words_ = words;
  }

  void Enqueue(const Launch& launch) const override {
    if (!launch.function)
      throw std::invalid_argument("TsingMicro function handle is missing");
    c10::SmallVector<uint64_t, 32> arguments;
    arguments.reserve(words_);
    for (size_t index = 0; index < widths_.size(); ++index) {
      const unsigned width = widths_[index];
      if (!width) arguments.push_back(1);
      uint64_t value = 0;
      std::memcpy(&value, launch.arguments[index], width ? width : sizeof(value));
      arguments.push_back(value);
    }
    for (const auto dimension : launch.grid) arguments.push_back(dimension);
    arguments.resize(words_, 0);
    const dim3 grid{GridSize(launch.grid[0]), GridSize(launch.grid[1]),
                    GridSize(launch.grid[2])};
    const dim3 block{1, 1, 1};
    const auto status = txLaunchKernel(
        reinterpret_cast<txFunction_t>(launch.function), grid, block,
        arguments.data(), static_cast<uint32_t>(arguments.size() * sizeof(uint64_t)),
        0, reinterpret_cast<txStream_t>(launch.stream));
    if (status != TX_SUCCESS)
      throw std::runtime_error("TsingMicro Triton launch failed: " + std::to_string(status));
  }

 private:
  std::vector<unsigned> widths_;
  size_t words_;
};

bool IsCapturing(uint64_t stream) {
  txStreamCaptureStatus status{};
  const auto result = txStreamIsCapturing(reinterpret_cast<txStream_t>(stream), &status);
  if (result != TX_SUCCESS)
    throw std::runtime_error("TsingMicro stream capture query failed: " + std::to_string(result));
  return status != txStreamCaptureStatusNone;
}

const Api api{
    kAbiVersion, sizeof(Api), "tsingmicro", "txda", StreamCaptureQuery, 256,
    [](_object* metadata) -> std::unique_ptr<Kernel> {
      return std::make_unique<TxKernel>(py::handle(metadata));
    },
    nullptr, &IsCapturing};

}  // namespace

const Api& CompiledBackend() { return api; }

}  // namespace flagos::native_host::backend
