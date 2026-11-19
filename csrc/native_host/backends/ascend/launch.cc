// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Submission mechanics of the original published FlagTree Ascend launcher.
#include "abi.h"
#include "backend_api.h"

#include <algorithm>
#include <ATen/ATen.h>
#include <acl/acl.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <runtime/runtime/rt.h>

namespace py = pybind11;

namespace flagos::native_host::backend {
namespace {

void Check(rtError_t status) {
  if (status != RT_ERROR_NONE)
    throw std::runtime_error("Ascend runtime launch failed: " + std::to_string(status));
}

at::Tensor Workspace(uint64_t bytes) {
  if (!bytes) return {};
  if (bytes > INT64_MAX) throw std::overflow_error("Ascend workspace exceeds int64");
  return at::empty({static_cast<int64_t>(bytes)},
                   at::TensorOptions().device(at::kPrivateUse1).dtype(at::kByte));
}

class AscendKernel final : public Kernel {
 public:
  explicit AscendKernel(py::handle metadata)
      : layout_(py::cast<std::vector<std::string>>(metadata.attr("_flagos_ascend_types")),
                py::cast<bool>(metadata.attr("_flagos_ascend_ffts")),
                py::cast<bool>(metadata.attr("force_simt_only"))),
        ffts_(py::cast<bool>(metadata.attr("_flagos_ascend_ffts"))),
        simt_only_(py::cast<bool>(metadata.attr("force_simt_only"))),
        auto_blocks_(py::cast<bool>(metadata.attr("_flagos_ascend_auto_blocks"))),
        blocks_(py::cast<uint32_t>(metadata.attr("_flagos_ascend_blocks"))) {
    const auto workspace = py::cast<int64_t>(py::getattr(metadata, "workspace_size", py::int_(-1)));
    workspace_ = workspace > 0 ? static_cast<uint64_t>(workspace) : 0;
    const auto locks = py::cast<int64_t>(py::getattr(metadata, "lock_num", py::int_(-1)));
    locks_ = locks > 0 ? static_cast<uint64_t>(locks) : 0;
    lock_value_ = py::cast<int64_t>(py::getattr(metadata, "lock_init_value", py::int_(0)));
    const auto parallel = py::cast<std::string>(metadata.attr("parallel_mode"));
    extended_ = py::cast<bool>(metadata.attr("compile_on_910_95")) &&
                (simt_only_ || parallel.find("simt") != std::string::npos);
    if (extended_)
      local_bytes_ = py::cast<uint32_t>(metadata.attr("shared_mem_dynamic_size"));
    if (auto_blocks_ && !blocks_)
      throw std::invalid_argument("invalid Ascend physical block count");
  }

  void Enqueue(const Launch& launch) const override {
    const auto count = CheckedProduct(CheckedProduct(launch.grid[0], launch.grid[1]), launch.grid[2]);
    if (count > UINT32_MAX) throw std::overflow_error("Ascend block count exceeds uint32");
    auto workspace = Workspace(CheckedProduct(workspace_, count));
    auto locks = Workspace(CheckedProduct(locks_, sizeof(int64_t)));
    if (locks_) {
      std::vector<int64_t> initial(locks_, lock_value_);
      Check(rtMemcpy(locks.mutable_data_ptr(), initial.size() * sizeof(int64_t),
                     initial.data(), initial.size() * sizeof(int64_t), RT_MEMCPY_HOST_TO_DEVICE));
    }
    uint64_t ffts = 0;
    if (ffts_) {
      uint32_t length = 0;
      Check(rtGetC2cCtrlAddr(&ffts, &length));
    }
    auto args = layout_.Pack(launch.arguments, launch.grid, ffts,
                            locks.defined() ? reinterpret_cast<uint64_t>(locks.const_data_ptr()) : 0,
                            workspace.defined() ? reinterpret_cast<uint64_t>(workspace.const_data_ptr()) : 0);
    const auto blocks = auto_blocks_ ? std::min(count, static_cast<uint64_t>(blocks_)) : count;
    const auto function = reinterpret_cast<const void*>(launch.function);
    const auto stream = reinterpret_cast<rtStream_t>(launch.stream);
    if (extended_) {
      rtArgsEx_t info{};
      info.args = args.data();
      info.argsSize = args.size();
      rtTaskCfgInfo_t config{};
      config.localMemorySize = local_bytes_;
      Check(rtKernelLaunchWithFlagV2(function, static_cast<uint32_t>(blocks), &info,
                                     nullptr, stream, 0, &config));
    } else {
      Check(rtKernelLaunch(function, static_cast<uint32_t>(blocks), args.data(),
                           args.size(), nullptr, stream));
    }
  }

 private:
  ascend::Layout layout_;
  bool ffts_, simt_only_, auto_blocks_, extended_;
  uint32_t blocks_, local_bytes_ = 0;
  uint64_t workspace_, locks_;
  int64_t lock_value_;
};

bool IsCapturing(uint64_t stream) {
  aclmdlRICaptureStatus status;
  aclmdlRI model = nullptr;
  const auto result = aclmdlRICaptureGetInfo(reinterpret_cast<aclrtStream>(stream), &status, &model);
  if (result != ACL_SUCCESS)
    throw std::runtime_error("Ascend stream capture query failed: " + std::to_string(result));
  return status != ACL_MODEL_RI_CAPTURE_STATUS_NONE;
}

const Api api{
    kAbiVersion, sizeof(Api), "ascend", "npu", GlobalScratch | StreamCaptureQuery, 512,
    [](_object* metadata) -> std::unique_ptr<Kernel> {
      return std::make_unique<AscendKernel>(py::handle(metadata));
    },
    nullptr, &IsCapturing};

}  // namespace

const Api& CompiledBackend() { return api; }

}  // namespace flagos::native_host::backend
