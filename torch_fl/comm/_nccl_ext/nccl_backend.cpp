// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Minimal C++ extension that exposes c10d::ProcessGroupNCCL to Python on a
// CPU-only torch build and keeps the vendor runtime's current stream aligned
// with the active FlagOS stream across collective launch and Work::wait().
//
// Why stream alignment is required: on non-boxing builds with owned streams
// (such as Hygon DCU's FLAGOS_HYGON_NATIVE_STREAM, where default_streams[dev]
// is created with cudaStreamNonBlocking), FlagOS compute kernels execute on
// GetCurrentStreamForDevice(dev) in libflagos.so, whereas ProcessGroupNCCL in
// the external vendor libtorch synchronizes its NCCL/RCCL communicator stream
// against c10::{cuda,hip}::getCurrent{CUDA,HIP}Stream(dev). Binding the vendor
// TLS current stream to GetCurrentStreamForDevice(dev) before collective entry
// and Work synchronization orders compute -> collective -> compute via native
// GPU events without host-blocking stalls.

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include <dlfcn.h>
#include <link.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#if __has_include(<hip_runtime_api.h>)
#define FLAGOS_NCCL_EXT_HIP 1
struct ihipStream_t;
namespace c10::hip {
class HIPStream {
 public:
  explicit HIPStream(c10::Stream stream) : stream_(stream) {}
  c10::Stream unwrap() const {
    return stream_;
  }

 private:
  c10::Stream stream_;
};
HIPStream getCurrentHIPStream(c10::DeviceIndex device_index = -1);
void setCurrentHIPStream(HIPStream stream);
HIPStream getStreamFromExternal(
    ihipStream_t* ext_stream,
    c10::DeviceIndex device_index);
} // namespace c10::hip
#else
#define FLAGOS_NCCL_EXT_HIP 0
#include <c10/cuda/CUDAStream.h>
#endif

namespace {

using GetFlagosStreamFn = void* (*)(int);
using GetFlagosDeviceFn = int (*)(int*);

bool ends_with(const char* str, const char* suffix) {
  if (str == nullptr || suffix == nullptr) {
    return false;
  }
  const size_t str_len = std::strlen(str);
  const size_t suf_len = std::strlen(suffix);
  return str_len >= suf_len &&
      std::strcmp(str + str_len - suf_len, suffix) == 0;
}

void* resolve_flagos_runtime_handle() {
  void* handle = dlopen("libflagos.so", RTLD_LAZY | RTLD_NOLOAD);
  if (handle != nullptr) {
    return handle;
  }
  dl_iterate_phdr(
      [](struct dl_phdr_info* info, size_t /*size*/, void* data) -> int {
        if (info->dlpi_name != nullptr &&
            (ends_with(info->dlpi_name, "/libflagos.so") ||
             std::strcmp(info->dlpi_name, "libflagos.so") == 0)) {
          void* h = dlopen(info->dlpi_name, RTLD_LAZY | RTLD_NOLOAD);
          if (h != nullptr) {
            *static_cast<void**>(data) = h;
            return 1;
          }
        }
        return 0;
      },
      &handle);
  return handle;
}

GetFlagosStreamFn resolve_get_flagos_stream() {
  static GetFlagosStreamFn fn = nullptr;
  if (fn != nullptr) {
    return fn;
  }
  if (void* sym = dlsym(RTLD_DEFAULT, "GetCurrentStreamForDevice")) {
    fn = reinterpret_cast<GetFlagosStreamFn>(sym);
    return fn;
  }
  if (void* handle = resolve_flagos_runtime_handle()) {
    fn = reinterpret_cast<GetFlagosStreamFn>(
        dlsym(handle, "GetCurrentStreamForDevice"));
  }
  return fn;
}

int current_flagos_device() {
  static GetFlagosDeviceFn fn = nullptr;
  if (fn == nullptr) {
    if (void* sym = dlsym(RTLD_DEFAULT, "GetDevice")) {
      fn = reinterpret_cast<GetFlagosDeviceFn>(sym);
    } else if (void* handle = resolve_flagos_runtime_handle()) {
      fn = reinterpret_cast<GetFlagosDeviceFn>(dlsym(handle, "GetDevice"));
    }
  }
  int dev = 0;
  if (fn != nullptr && fn(&dev) == 0 && dev >= 0) {
    return dev;
  }
  return 0;
}

void sync_vendor_stream_for_device(c10::DeviceIndex dev) {
  if (dev < 0) {
    return;
  }
  auto get_stream = resolve_get_flagos_stream();
  if (get_stream == nullptr) {
    return;
  }
  void* raw = get_stream(static_cast<int>(dev));
  if (raw == nullptr) {
    return;
  }
#if FLAGOS_NCCL_EXT_HIP
  c10::hip::setCurrentHIPStream(
      c10::hip::getStreamFromExternal(static_cast<ihipStream_t*>(raw), dev));
#else
  c10::cuda::setCurrentCUDAStream(
      c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(raw), dev));
#endif
}

void add_unique_device(
    std::vector<c10::DeviceIndex>& devices,
    c10::DeviceIndex dev) {
  if (dev < 0) {
    return;
  }
  for (auto existing : devices) {
    if (existing == dev) {
      return;
    }
  }
  devices.push_back(dev);
}

void collect_devices(
    std::vector<c10::DeviceIndex>& devices,
    const at::Tensor& tensor) {
  if (tensor.defined() && !tensor.device().is_cpu()) {
    add_unique_device(devices, tensor.device().index());
  }
}

void collect_devices(
    std::vector<c10::DeviceIndex>& devices,
    const std::vector<at::Tensor>& tensors) {
  for (const auto& t : tensors) {
    collect_devices(devices, t);
  }
}

void collect_devices(
    std::vector<c10::DeviceIndex>& devices,
    const std::vector<std::vector<at::Tensor>>& tensor_lists) {
  for (const auto& list : tensor_lists) {
    collect_devices(devices, list);
  }
}

void sync_devices(const std::vector<c10::DeviceIndex>& devices) {
  for (auto dev : devices) {
    sync_vendor_stream_for_device(dev);
  }
}

class FlagosWorkNCCL final : public c10d::Work {
 public:
  FlagosWorkNCCL(
      c10::intrusive_ptr<c10d::Work> inner,
      std::vector<c10::DeviceIndex> devices)
      : c10d::Work(-1, inner->retrieveOpType()),
        inner_(std::move(inner)),
        devices_(std::move(devices)) {}

  bool isCompleted() override {
    return inner_->isCompleted();
  }

  bool isSuccess() const override {
    return inner_->isSuccess();
  }

  std::exception_ptr exception() const override {
    return inner_->exception();
  }

  int sourceRank() const override {
    return inner_->sourceRank();
  }

  std::vector<at::Tensor> result() override {
    return inner_->result();
  }

  void synchronize() override {
    sync_devices(devices_);
    inner_->synchronize();
  }

  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    sync_devices(devices_);
    return inner_->wait(timeout);
  }

  void blockCurrentStream() override {
    sync_devices(devices_);
    inner_->blockCurrentStream();
  }

  void abort() override {
    inner_->abort();
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    return inner_->getFuture();
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFutureResult() override {
    return inner_->getFutureResult();
  }

  float getDuration() const override {
    return inner_->getDuration();
  }

  uint64_t getSequencenumber() const override {
    return inner_->getSequencenumber();
  }

 private:
  c10::intrusive_ptr<c10d::Work> inner_;
  std::vector<c10::DeviceIndex> devices_;
};

c10::intrusive_ptr<c10d::Work> wrap_work(
    c10::intrusive_ptr<c10d::Work> work,
    std::vector<c10::DeviceIndex> devices) {
  if (!work) {
    return work;
  }
  return c10::make_intrusive<FlagosWorkNCCL>(
      std::move(work), std::move(devices));
}

class FlagosProcessGroupNCCL final : public c10d::ProcessGroupNCCL {
 public:
  using c10d::ProcessGroupNCCL::ProcessGroupNCCL;

  void startCoalescing() override {
    coalesced_devices_.clear();
    add_unique_device(
        coalesced_devices_, static_cast<c10::DeviceIndex>(current_flagos_device()));
    sync_devices(coalesced_devices_);
    c10d::ProcessGroupNCCL::startCoalescing();
  }

  c10::intrusive_ptr<c10d::Work> endCoalescing() override {
    sync_devices(coalesced_devices_);
    auto work = c10d::ProcessGroupNCCL::endCoalescing();
    auto devs = std::move(coalesced_devices_);
    coalesced_devices_.clear();
    return wrap_work(std::move(work), std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override {
    auto devs = sync_for(tensors);
    return wrap_work(
        c10d::ProcessGroupNCCL::broadcast(tensors, opts), std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override {
    auto devs = sync_for(tensors);
    return wrap_work(
        c10d::ProcessGroupNCCL::allreduce_sparse(tensors, opts),
        std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override {
    auto devs = sync_for(tensors);
    return wrap_work(
        c10d::ProcessGroupNCCL::allreduce(tensors, opts), std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts =
          c10d::AllreduceCoalescedOptions()) override {
    auto devs = sync_for(tensors);
    return wrap_work(
        c10d::ProcessGroupNCCL::allreduce_coalesced(tensors, opts),
        std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override {
    auto devs = sync_for(tensors);
    return wrap_work(
        c10d::ProcessGroupNCCL::reduce(tensors, opts), std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    auto devs = sync_for(outputTensors, inputTensors);
    return wrap_work(
        c10d::ProcessGroupNCCL::allgather(outputTensors, inputTensors, opts),
        std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> _allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    auto devs = sync_for(outputbuffer, inputbuffer);
    return wrap_work(
        c10d::ProcessGroupNCCL::_allgather_base(outputbuffer, inputbuffer, opts),
        std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    auto devs = sync_for(outputTensorLists, inputTensors);
    return wrap_work(
        c10d::ProcessGroupNCCL::allgather_coalesced(
            outputTensorLists, inputTensors, opts),
        std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    auto devs = sync_for(outputs, inputs);
    return wrap_work(
        c10d::ProcessGroupNCCL::allgather_into_tensor_coalesced(
            outputs, inputs, opts),
        std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override {
    auto devs = sync_for(outputTensors, inputTensors);
    return wrap_work(
        c10d::ProcessGroupNCCL::reduce_scatter(
            outputTensors, inputTensors, opts),
        std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override {
    auto devs = sync_for(outputTensor, inputTensor);
    return wrap_work(
        c10d::ProcessGroupNCCL::_reduce_scatter_base(
            outputTensor, inputTensor, opts),
        std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override {
    auto devs = sync_for(outputs, inputs);
    return wrap_work(
        c10d::ProcessGroupNCCL::reduce_scatter_tensor_coalesced(
            outputs, inputs, opts),
        std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override {
    std::vector<c10::DeviceIndex> devs;
    for (auto id : opts.device_ids) {
      add_unique_device(devs, static_cast<c10::DeviceIndex>(id));
    }
    if (devs.empty()) {
      add_unique_device(
          devs, static_cast<c10::DeviceIndex>(current_flagos_device()));
    }
    sync_devices(devs);
    return wrap_work(
        c10d::ProcessGroupNCCL::barrier(opts), std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override {
    auto devs = sync_for(outputTensor, inputTensor);
    return wrap_work(
        c10d::ProcessGroupNCCL::alltoall_base(
            outputTensor, inputTensor, outputSplitSizes, inputSplitSizes, opts),
        std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override {
    auto devs = sync_for(outputTensors, inputTensors);
    return wrap_work(
        c10d::ProcessGroupNCCL::alltoall(outputTensors, inputTensors, opts),
        std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    auto devs = sync_for(tensors);
    return wrap_work(
        c10d::ProcessGroupNCCL::send(tensors, dstRank, tag), std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override {
    auto devs = sync_for(tensors);
    return wrap_work(
        c10d::ProcessGroupNCCL::recv(tensors, srcRank, tag), std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override {
    auto devs = sync_for(outputTensors, inputTensors);
    return wrap_work(
        c10d::ProcessGroupNCCL::gather(outputTensors, inputTensors, opts),
        std::move(devs));
  }

  c10::intrusive_ptr<c10d::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override {
    auto devs = sync_for(outputTensors, inputTensors);
    return wrap_work(
        c10d::ProcessGroupNCCL::scatter(outputTensors, inputTensors, opts),
        std::move(devs));
  }

 private:
  template <typename... Args>
  std::vector<c10::DeviceIndex> sync_for(const Args&... args) {
    std::vector<c10::DeviceIndex> devs;
    (collect_devices(devs, args), ...);
    for (auto d : devs) {
      add_unique_device(coalesced_devices_, d);
    }
    sync_devices(devs);
    return devs;
  }

  std::vector<c10::DeviceIndex> coalesced_devices_;
};

// Build a FlagosProcessGroupNCCL and up-cast to the Backend base so Python sees
// the already-bound c10d.Backend interface.
c10::intrusive_ptr<c10d::Backend> make_nccl_backend(
    c10::intrusive_ptr<c10d::Store> store,
    int64_t rank,
    int64_t size,
    int64_t timeout_ms,
    bool high_priority_stream) {
  auto opts = c10d::ProcessGroupNCCL::Options::create(high_priority_stream);
  if (timeout_ms > 0) {
    opts->timeout = std::chrono::milliseconds(timeout_ms);
  }
  auto pg = c10::make_intrusive<FlagosProcessGroupNCCL>(
      std::move(store),
      static_cast<int>(rank),
      static_cast<int>(size),
      std::move(opts));
  return pg;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "flagos internal: expose ProcessGroupNCCL for CPU-torch + external "
            "vendor libtorch";
  m.def(
      "make_nccl_backend",
      &make_nccl_backend,
      pybind11::arg("store"),
      pybind11::arg("rank"),
      pybind11::arg("size"),
      pybind11::arg("timeout_ms") = 0,
      pybind11::arg("high_priority_stream") = false,
      "Construct a c10d::ProcessGroupNCCL, returned as a c10d.Backend.");
}
