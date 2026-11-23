// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "launch_params.h"

#include <cuda.h>

namespace py = pybind11;

namespace flagos::native_host::backend {
namespace {

void Check(CUresult status) {
  if (status == CUDA_SUCCESS) return;
  const char* error = nullptr;
  cuGetErrorString(status, &error);
  throw std::runtime_error(error ? error : "CoreX driver operation failed");
}

class CorexKernel final : public WarpKernel {
 public:
  using WarpKernel::WarpKernel;

  void Enqueue(const Launch& launch) const override {
    static_assert(sizeof(CUdeviceptr) == sizeof(uint64_t));
    auto function = reinterpret_cast<CUfunction>(launch.function);
    auto stream = reinterpret_cast<CUstream>(launch.stream);
    const auto block_x = GridSize(CheckedProduct(warps_, 64));
    const auto grid_y = GridSize(launch.grid[1]);
    const auto grid_z = GridSize(launch.grid[2]);
    // Follow the published CoreX launcher. A CTA uses the ordinary driver call.
    if (ctas_ == 1) {
      Check(cuLaunchKernel(function, GridSize(launch.grid[0]), grid_y, grid_z,
                           block_x, 1, 1, shared_, stream, launch.arguments, nullptr));
      return;
    }
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
    attrs[count].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attrs[count].value.clusterDim.x = ctas_;
    attrs[count].value.clusterDim.y = 1;
    attrs[count++].value.clusterDim.z = 1;
    attrs[count].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    attrs[count++].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;

    CUlaunchConfig config{};
    config.gridDimX = GridSize(CheckedProduct(launch.grid[0], ctas_));
    config.gridDimY = grid_y;
    config.gridDimZ = grid_z;
    config.blockDimX = block_x;
    config.blockDimY = config.blockDimZ = 1;
    config.sharedMemBytes = shared_;
    config.hStream = stream;
    config.attrs = attrs;
    config.numAttrs = count;
    Check(cuLaunchKernelEx(&config, function, launch.arguments, nullptr));
  }
};

bool IsCapturing(uint64_t stream) {
  CUstreamCaptureStatus status;
  Check(cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &status));
  return status != CU_STREAM_CAPTURE_STATUS_NONE;
}

const Api api{
    kAbiVersion, sizeof(Api), "iluvatar", "corex",
    SoftwareDescriptor | GlobalScratch | ProfileScratch | CooperativeLaunch |
        ClusterLaunch | ProgrammaticLaunch | StreamCaptureQuery,
    256,
    [](_object* metadata) -> std::unique_ptr<Kernel> {
      return std::make_unique<CorexKernel>(py::handle(metadata));
    },
    [](_object* metadata, _object*) -> std::shared_ptr<const Descriptor> {
      return std::make_shared<SoftwareTensorDescriptor>(py::handle(metadata));
    },
    &IsCapturing};

}  // namespace

const Api& CompiledBackend() { return api; }

}  // namespace flagos::native_host::backend
