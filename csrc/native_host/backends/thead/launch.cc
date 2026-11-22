// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "launch_params.h"

#include <hggc.h>

namespace py = pybind11;

namespace flagos::native_host::backend {
namespace {

void Check(HGresult status) {
  if (status == HGGC_SUCCESS) return;
  const char* error = nullptr;
  hgGetErrorString(status, &error);
  throw std::runtime_error(error ? error : "HGGC driver operation failed");
}

class PpuKernel final : public WarpKernel {
 public:
  using WarpKernel::WarpKernel;

  void Enqueue(const Launch& launch) const override {
    static_assert(sizeof(HGdeviceptr) == sizeof(uint64_t));
    auto function = reinterpret_cast<HGfunction>(launch.function);
    HGlaunchAttribute attrs[4]{};
    unsigned count = 0;
    if (pdl_) {
      attrs[count].id = HG_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      attrs[count++].value.programmaticStreamSerializationAllowed = 1;
    }
    if (cooperative_) {
      attrs[count].id = HG_LAUNCH_ATTRIBUTE_COOPERATIVE;
      attrs[count++].value.cooperative = 1;
    }
    if (ctas_ != 1) {
      attrs[count].id = HG_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attrs[count].value.clusterDim.x = ctas_;
      attrs[count].value.clusterDim.y = 1;
      attrs[count++].value.clusterDim.z = 1;
      attrs[count].id = HG_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      attrs[count++].value.clusterSchedulingPolicyPreference = HG_CLUSTER_SCHEDULING_POLICY_SPREAD;
    }
    if (ctas_ == 16)
      Check(hgFuncSetAttribute(function, HG_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));

    HGlaunchConfig config{};
    config.gridDimX = GridSize(CheckedProduct(launch.grid[0], ctas_));
    config.gridDimY = GridSize(launch.grid[1]);
    config.gridDimZ = GridSize(launch.grid[2]);
    config.blockDimX = GridSize(CheckedProduct(warps_, 32));
    config.blockDimY = config.blockDimZ = 1;
    config.sharedMemBytes = shared_;
    config.hStream = reinterpret_cast<HGstream>(launch.stream);
    config.attrs = attrs;
    config.numAttrs = count;
    Check(hgLaunchKernelEx(&config, function, launch.arguments, nullptr));
  }
};

class PpuDescriptor final : public Descriptor {
 public:
  explicit PpuDescriptor(py::handle metadata) {
    if (!metadata.is_none())
      throw std::invalid_argument("PPU native hosts require software tensor descriptors");
  }

  size_t Capacity() const override { return kSoftwareDescriptorSlots; }

  size_t Pack(const DescriptorValue& value, Slot* slots) const override {
    return PackSoftwareDescriptor(value, slots);
  }
};

bool IsCapturing(uint64_t stream) {
  HGstreamCaptureStatus status;
  Check(hgStreamIsCapturing(reinterpret_cast<HGstream>(stream), &status));
  return status != HG_STREAM_CAPTURE_STATUS_NONE;
}

const Api api{
    kAbiVersion, sizeof(Api), "thead", "ppu",
    SoftwareDescriptor | GlobalScratch | ProfileScratch | CooperativeLaunch |
        ClusterLaunch | ProgrammaticLaunch | StreamCaptureQuery,
    256,
    [](_object* metadata) -> std::unique_ptr<Kernel> {
      return std::make_unique<PpuKernel>(py::handle(metadata));
    },
    [](_object* metadata, _object*) -> std::shared_ptr<const Descriptor> {
      return std::make_shared<PpuDescriptor>(py::handle(metadata));
    },
    &IsCapturing};

}  // namespace

const Api& CompiledBackend() { return api; }

}  // namespace flagos::native_host::backend
