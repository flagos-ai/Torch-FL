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
// CPU-only torch build.
//
// Why this exists: the pip `torch==2.10.0+cpu` wheel is compiled WITHOUT
// USE_C10D_NCCL, so its Python bindings never expose `ProcessGroupNCCL`
// (torch.distributed.is_nccl_available() == False). An externally supplied
// vendor libtorch -- libtorch_cuda.so for NVIDIA or libtorch_hip.so for Hygon
// DCU -- can still contain the full c10d::ProcessGroupNCCL implementation.
//
// We only need to *construct* a ProcessGroupNCCL and hand it back as an
// intrusive_ptr<Backend>. The c10d.Backend base class already exposes every
// collective method (allreduce/broadcast/allgather/reduce_scatter/alltoall/
// gather/scatter/send/recv/barrier) to Python via pybind, and those are C++
// virtuals -> calling them on the returned object dispatches to the NCCL impl.
// So ProcessGroupFlagOS can use it as an inner backend with zero extra binding.
//
// Build: compiled with -DUSE_C10D_NCCL so ProcessGroupNCCL.hpp and the
// NCCL_HAS_* feature macros are derived from the NCCL/RCCL headers matching the
// external vendor libtorch.

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include <chrono>
#include <memory>

namespace {

// Build a ProcessGroupNCCL and up-cast to the Backend base so Python sees the
// already-bound c10d.Backend interface.
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
  auto pg = c10::make_intrusive<c10d::ProcessGroupNCCL>(
      std::move(store), static_cast<int>(rank), static_cast<int>(size),
      std::move(opts));
  return pg;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "flagos internal: expose ProcessGroupNCCL for CPU-torch + external "
            "vendor libtorch";
  m.def("make_nccl_backend", &make_nccl_backend,
        pybind11::arg("store"),
        pybind11::arg("rank"),
        pybind11::arg("size"),
        pybind11::arg("timeout_ms") = 0,
        pybind11::arg("high_priority_stream") = false,
        "Construct a c10d::ProcessGroupNCCL, returned as a c10d.Backend.");
}
