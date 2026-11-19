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

// DataParallel's communication primitives for the flagos device.
//
// torch.nn.parallel.comm is a thin veneer over seven functions on torch._C --
// _broadcast_coalesced, _broadcast, _broadcast_out, _scatter, _scatter_out,
// _gather, _gather_out -- and stock torch registers all seven for CUDA only
// (torch/csrc/cuda/python_comm.cpp over torch/csrc/cuda/comm.cpp). On a flagos
// build that layer is wrong in two opposite ways at once, both measured on PPU:
//
//   * a flagos tensor is not is_cuda(), so the checks reject it:
//         torch._C._gather([flagos:0, flagos:1], 0, 0)
//         -> RuntimeError: Expected all input tensors to be CUDA tensors, but
//            tensor at index 0 has device flagos:0
//   * a flagos device index *is* the CUDA index of the same card on a CUDA-ABI
//     vendor, so what the checks do not cover silently succeeds on the wrong
//     device type:
//         torch._C._scatter(flagos:0 tensor, [0, 1], None, 0, None)
//         -> [flagos:0, cuda:1]
//
// DataParallel reaches all seven on its forward path: scatter for the inputs,
// replicate -> broadcast_coalesced for the parameters, gather for the outputs
// (plus the *_out forms through the documented out= argument). None of them can
// be left to the CUDA layer.
//
// This file is that layer for flagos: the same seven operations, under stock's
// names, signatures, argument names, checks and return shapes
// (torch/csrc/cuda/comm.cpp is the model, down to the messages), written
// against the flagos device instead of CUDA. They are the same tensor
// operations every other flagos op is -- at::flatten_dense_tensors, at::empty,
// copy_, Tensor::to, split_with_sizes -- so a CUDA-compatible vendor runs them
// on the vendor's kernels through the flagos dispatch, and a non-CUDA vendor
// runs them through its own.
//
// InitDataParallelComm() publishes them by rebinding the seven attributes on
// the already-imported torch._C -- the same shape torch_npu uses for the same
// problem (torch_npu/csrc/npu/Module.cpp's initCommMethods() over
// torch_npu/csrc/npu/DataParallelComm.cpp). The rebinding has to be a
// py::setattr and not a module_::def: def passes whatever already sits under
// the name as a pybind11 *sibling*, i.e. one more overload of the very
// function being replaced, and the original is then still tried first. See the
// comment in InitDataParallelComm(). The originals are captured first, and
// every replacement delegates to its original as soon as no flagos tensor is
// involved, so a CUDA- or CPU-placed DataParallel keeps stock behaviour
// exactly. Where the build has no CUDA comm layer to capture -- a +cpu torch,
// which is what this project's own CUDA, Ascend and GCU pipelines install --
// there is no original and nothing to delegate to: the seven are published
// rather than replaced, and this layer serves every call. See
// original_or_none().
//
// Two deliberate departures from torch/csrc/cuda/comm.cpp, both because the
// device *type* is the only fixed thing here and the vendor is not:
//
//   * the streams argument is accepted and dropped. Stock runs the copies on
//     caller-supplied CUDA streams; a flagos stream is a CUDA stream only on a
//     CUDA-ABI vendor (an ACL stream on Ascend, a GCU stream on Enflame), so
//     taking one would make this layer vendor-specific. The copies run on the
//     target device's current stream instead, which is a strictly-ordered
//     subset of what the argument asks for: Scatter.forward's
//     wait_stream/record_stream pair afterwards stays valid, because a stream
//     that was never submitted to is already complete.
//   * a CPU *source* is copied without non_blocking. Stock can copy it
//     asynchronously because it submitted the copy to a stream the caller then
//     waits on; with no such stream here, the copy has to be complete before
//     the call returns, which is also what the pure-Python layer this replaces
//     did (Tensor.to defaults to non_blocking=False).
//
// There is no NCCL branch either: torch.cuda.nccl is a CUDA implementation, and
// the collectives a flagos build offers are the vendor's, reached through the
// platform's own distributed backend rather than from here. What is left is the
// memcpy path stock falls back to without NCCL, which is the path DataParallel
// uses on every backend that has no nccl.is_available().

#include "dataparallel_comm.h"

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/tensor_flatten.h>

#include <ATen/ATen.h>
#include <ATen/WrapDimUtils.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/irange.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

namespace torch_fl::comm {
namespace {

namespace py = pybind11;

constexpr c10::DeviceType kFlagos = c10::DeviceType::PrivateUse1;

// torch/csrc/cuda/comm.h's alias for "one list of tensors per device".
// Declared here rather than included from there: this file must not pull in the
// CUDA headers, whose comm layer is what it replaces.
using tensor_list2d = std::vector<std::vector<at::Tensor>>;

bool is_flagos(const at::Tensor& tensor) {
  return tensor.defined() && tensor.device().type() == kFlagos;
}

template <typename Tensors>
bool any_flagos(const Tensors& tensors) {
  return std::any_of(tensors.begin(), tensors.end(), [](const at::Tensor& t) {
    return is_flagos(t);
  });
}

// See SetScatterScope in the header.
thread_local bool scatter_scope = false;

// The attribute `name` currently holds on torch._C, or None when there is none.
//
// Stock torch registers the seven comm functions from its CUDA build only
// (torch/csrc/cuda/python_comm.cpp, reached from torch/csrc/Module.cpp under
// USE_CUDA), so a build without it -- the +cpu wheels the CUDA, Ascend and GCU
// pipelines of this project all install, and then hand an accelerator to
// through torch_fl's own dispatch -- has no such attribute at all. There the
// seven are published rather than replaced, which is the case torch_npu's
// initCommMethods() is written for too. Note that this is per name, not a
// decision to skip the install: skipping it would leave DataParallel with no
// comm layer whatsoever on those builds, which is the failure this file exists
// to fix.
py::object original_or_none(const py::module_& m, const char* name) {
  // Two returns rather than a ternary: py::object and py::none in one
  // conditional makes pybind11 convert the *object* branch to py::none, which
  // is a checked downcast and throws at runtime ("Object of type
  // 'builtin_function_or_method' is not an instance of 'none'") the moment a
  // stock implementation is actually present.
  if (py::hasattr(m, name)) {
    return m.attr(name);
  }
  return py::none();
}

// Whether a replacement has a stock implementation to fall back to for input it
// is not responsible for. False on a torch build with no CUDA comm layer, where
// the replacement is the only implementation of the name and therefore has to
// run for every call rather than delegate.
bool has_stock_fallback(const py::object& original) {
  return !original.is_none();
}

// Some operations can be performed more efficiently if we're handling tensors
// of a single type only. Adding this logic directly in the loop makes it a bit
// ugly, so here's a helper for it.
//
// Copied from torch/csrc/cuda/comm.cpp, which keeps it file-private (torch_npu
// copies it the same way). type_id() is
// backend * ScalarType::NumOptions + scalar_type, so it fits in a byte for any
// backend a PrivateUse1 tensor can have, and is only ever compared for
// equality.
struct unique_type_checker {
  void show(size_t type_id) {
    if (!unique) {
      return;
    }
    if (!type_id_) {
      type_id_ = type_id;
    }
    unique = type_id_.value() == type_id;
  }

  std::optional<size_t> type_id_;
  bool unique = true;
};

// ***************** Broadcast *******************
//
// Broadcast a source tensor (CPU or flagos) to a list of flagos devices, or to
// flagos tensors on one or more devices.

// no checks
std::vector<at::Tensor>& broadcast_out_impl(
    const at::Tensor& tensor,
    std::vector<at::Tensor>& out_tensors) {
  for (auto& out_tensor : out_tensors) {
    out_tensor.copy_(tensor, /*non_blocking=*/is_flagos(tensor));
  }
  return out_tensors;
}

std::vector<at::Tensor>& broadcast_out(
    const at::Tensor& tensor,
    std::vector<at::Tensor>& out_tensors) {
  for (const auto i : c10::irange(out_tensors.size())) {
    TORCH_CHECK(
        is_flagos(out_tensors[i]),
        "Expected all output tensors to be flagos tensors, but output tensor at index ",
        i,
        " has device '",
        out_tensors[i].device(),
        "'");
    TORCH_CHECK(
        out_tensors[i].sizes() == tensor.sizes(),
        "Expected all output tensors to have same shape as the source tensor ",
        tensor.sizes(),
        ", but output tensor at index ",
        i,
        " has shape ",
        out_tensors[i].sizes());
  }
  return broadcast_out_impl(tensor, out_tensors);
}

std::vector<at::Tensor> broadcast(
    const at::Tensor& tensor,
    at::IntArrayRef devices) {
  std::vector<at::Tensor> diff_device_dst_tensors;
  diff_device_dst_tensors.reserve(devices.size());
  for (auto device : devices) {
    TORCH_CHECK(
        device >= 0, "Expected non-negative device index, but got ", device);
    if (device != tensor.get_device()) {
      diff_device_dst_tensors.emplace_back(at::empty(
          tensor.sizes(),
          tensor.options().device(at::Device(
              kFlagos, static_cast<c10::DeviceIndex>(device)))));
    }
  }
  broadcast_out_impl(tensor, diff_device_dst_tensors);
  std::vector<at::Tensor> dst_tensors;
  dst_tensors.reserve(devices.size());
  auto it = diff_device_dst_tensors.begin();
  for (auto device : devices) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (device != tensor.get_device()) {
      dst_tensors.emplace_back(*it++);
    } else {
      dst_tensors.emplace_back(tensor);
    }
  }
  TORCH_INTERNAL_ASSERT(it == diff_device_dst_tensors.end());
  return dst_tensors;
}

// NOTE [ Version Counter in comm.*_coalesced ]
//
// broadcast_coalesced
// ~~~~~~~~~~~~~~~~~~~
//
// In broadcast_coalesced, multiple variables may be coalesced into a single
// large one, broadcast to other devices, and the get split according to the
// original shapes.
//
// When splitting, the view operations will make all Variables broadcast
// together to share a single version counter, because they are all views of the
// large Variable. However, that large Variable is immediately discarded and all
// these Variables do not share storage at all.
//
// For example, when two buffers are broadcast together in `DataParallel` and
// one of them is modified in-place during `forward` but the other is needed in
// backward, autograd engine will complain.
//
// We thus re-wrap these Variables after broadcasting (i.e., effectively doing
// what is equivalent to .data in Python), and give them individual version
// counters.
//
// NB: Just calling detach() on the variables is not sufficient
//
// NB: For `device[0]` in broadcast_coalesced, the input Variables are always
//     returned as-is, so **do not** re-wrap them.
tensor_list2d broadcast_coalesced(
    at::TensorList tensors,
    at::IntArrayRef devices,
    size_t buffer_size) {
  // Stock indexes devices[0] without checking; DataParallel always passes at
  // least one device, and a caller who passes none gets an out-of-bounds read
  // there rather than an error.
  TORCH_CHECK(!devices.empty(), "Expected at least one device to broadcast to");
  TORCH_CHECK(
      std::all_of(
          tensors.begin(),
          tensors.end(),
          [&](const at::Tensor& t) { return t.get_device() == devices[0]; }),
      "All tensors must be on devices[0]: ",
      devices[0]);

  tensor_list2d outputs(devices.size());
  outputs[0] = tensors.vec();
  for (auto& o : outputs) {
    o.reserve(tensors.size());
  }

  unique_type_checker type_checker;
  c10::OptionalDeviceGuard device_guard;
  device_guard.reset_device(
      c10::Device(kFlagos, static_cast<c10::DeviceIndex>(devices[0])));
  for (auto& chunk : torch::utils::take_tensors(tensors, buffer_size)) {
    const auto type_id = chunk.type_id();
    type_checker.show(type_id);
    if (chunk.options().is_sparse()) {
      auto flat_tuple = torch::utils::flatten_sparse_tensors(chunk.tensors);
      auto broadcast_indices = broadcast(flat_tuple.first, devices);
      auto broadcast_values = broadcast(flat_tuple.second, devices);
      for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        device_guard.reset_device(
            c10::Device(kFlagos, static_cast<c10::DeviceIndex>(devices[i])));
        auto& device_outputs = outputs[i];
        auto& inds = broadcast_indices[i];
        auto& vals = broadcast_values[i];
        for (const auto& var : torch::utils::unflatten_sparse_tensors(
                 inds, vals, chunk.tensors)) {
          // See NOTE [ Version Counter in comm.*_coalesced ]
          device_outputs.emplace_back(
              torch::autograd::make_variable(var.tensor_data(), false));
        }
      }
    } else {
      auto results = broadcast(
          torch::utils::flatten_dense_tensors(chunk.tensors), devices);
      for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        device_guard.reset_device(
            c10::Device(kFlagos, static_cast<c10::DeviceIndex>(devices[i])));
        auto& device_outputs = outputs[i];
        for (auto& var :
             torch::utils::unflatten_dense_tensors(results[i], chunk.tensors)) {
          // See NOTE [ Version Counter in comm.*_coalesced ]
          device_outputs.emplace_back(
              torch::autograd::make_variable(var.tensor_data(), false));
        }
      }
    }
  }

  // If we only saw a single tensor type, then we can skip expensive reordering
  if (!type_checker.unique) {
    for (auto& o : outputs) {
      torch::utils::reorder_tensors_like(o, tensors);
    }
  }
  return outputs;
}

// ***************** Scatter *******************
//
// Scatter a source tensor (CPU or flagos) to a list of flagos tensors on one or
// more devices. The streams argument is dropped; see the file comment.

std::vector<at::Tensor>& scatter_out(
    const at::Tensor& tensor,
    std::vector<at::Tensor>& out_tensors,
    int64_t dim) {
  TORCH_CHECK(
      !out_tensors.empty(),
      "Expected at least one output tensor to scatter to");
  dim = at::maybe_wrap_dim(dim, tensor);
  int64_t total_size = 0;
  std::vector<int64_t> chunk_sizes;
  chunk_sizes.reserve(out_tensors.size());
  for (const auto i : c10::irange(out_tensors.size())) {
    TORCH_CHECK(
        is_flagos(out_tensors[i]),
        "Expected all output tensors to be flagos tensors, but output tensor at index ",
        i,
        " has device '",
        out_tensors[i].device(),
        "'");
    auto out_sizes = out_tensors[i].sizes().vec();
    bool same_ndim = out_sizes.size() == static_cast<size_t>(tensor.dim());
    if (same_ndim) {
      total_size += out_sizes[dim];
      chunk_sizes.emplace_back(out_sizes[dim]);
      out_sizes[dim] = tensor.size(dim);
    }
    TORCH_CHECK(
        same_ndim && out_sizes == tensor.sizes(),
        "Output tensor at index ",
        i,
        " has incorrect shape: ",
        out_tensors[i].sizes(),
        ". Expected same "
        "shape except for scatter dim ",
        dim,
        " as the source tensor: ",
        at::IntArrayRef(tensor.sizes()));
  }
  TORCH_CHECK(
      total_size == tensor.size(dim),
      "Total size for output tensors along scatter dim ",
      dim,
      " does not match "
      "the source tensor size at dim ",
      dim,
      ". Expected ",
      tensor.size(dim),
      ", but got total size ",
      total_size);

  auto chunks =
      tensor.split_with_sizes(/*split_sizes=*/chunk_sizes, /*dim=*/dim);
  for (const auto i : c10::irange(chunks.size())) {
    // NB: We don't detect the case where `out_tensor` is already the correct
    //     view of `tensor` since that would be nontrivial and involve checking
    //     ptr, offset, and strides. So `scatter_out(src, src.chunk(...))` does
    //     more copying than `scatter(src)`.
    out_tensors[i].copy_(chunks[i], /*non_blocking=*/is_flagos(tensor));
  }
  return out_tensors;
}

std::vector<at::Tensor> scatter(
    const at::Tensor& tensor,
    at::IntArrayRef devices,
    const std::optional<std::vector<int64_t>>& chunk_sizes,
    int64_t dim) {
  TORCH_CHECK(!devices.empty(), "Expected at least one device to scatter to");
  if (chunk_sizes.has_value()) {
    TORCH_CHECK(
        chunk_sizes->size() == devices.size(),
        "Expected devices and chunk_sizes to be of same length, but got "
        "len(devices) = ",
        devices.size(),
        " and len(chunk_sizes) = ",
        chunk_sizes->size());
  }
  dim = at::maybe_wrap_dim(dim, tensor);
  std::vector<at::Tensor> chunks = chunk_sizes
      ? tensor.split_with_sizes(/*split_sizes=*/*chunk_sizes, /*dim=*/dim)
      : tensor.chunk(
            /*chunks=*/static_cast<int64_t>(devices.size()), /*dim=*/dim);
  for (const auto i : c10::irange(chunks.size())) {
    const auto device_index = static_cast<int16_t>(devices[i]);
    if (device_index != tensor.get_device()) {
      TORCH_CHECK(
          device_index >= 0,
          "Expected non-negative device index, but got ",
          device_index);
      chunks[i] = chunks[i].to(
          {kFlagos, device_index},
          /*non_blocking=*/is_flagos(tensor),
          /*copy=*/false,
          /*memory_format=*/at::MemoryFormat::Preserve);
    }
  }
  return chunks;
}

// ***************** Gather *******************
//
// Gather a list of flagos tensors on one or more devices to a target tensor or
// device, either CPU or flagos.

// no checks
at::Tensor& gather_out_impl(
    at::TensorList tensors,
    at::Tensor& out_tensor,
    int64_t dim) {
  std::vector<int64_t> chunk_sizes;
  chunk_sizes.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    chunk_sizes.emplace_back(tensor.size(dim));
  }
  auto chunks =
      out_tensor.split_with_sizes(/*split_sizes=*/chunk_sizes, /*dim=*/dim);
  for (const auto i : c10::irange(tensors.size())) {
    chunks[i].copy_(tensors[i], /*non_blocking=*/is_flagos(out_tensor));
  }
  return out_tensor;
}

at::Tensor& gather_out(
    at::TensorList tensors,
    at::Tensor& out_tensor,
    int64_t dim) {
  TORCH_CHECK(!tensors.empty(), "Expected at least one tensor to gather from");
  int64_t total_size = 0;
  auto& first = tensors.front();
  const auto first_size = first.sizes();
  dim = at::maybe_wrap_dim(dim, first);
  std::vector<int64_t> expected_size(first_size.begin(), first_size.end());
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    TORCH_CHECK(
        is_flagos(tensor),
        "Expected all input tensors to be flagos tensors, but "
        "tensor at index ",
        i,
        " has device '",
        tensor.device(),
        "'");
    TORCH_CHECK(
        tensor.ndimension() == static_cast<int64_t>(expected_size.size()),
        "Expected all input tensors to have the same number of dimensions, but ",
        "tensor at index ",
        i,
        "has ",
        tensor.ndimension(),
        " dimensions, (expected ",
        expected_size.size(),
        ")");
    expected_size[dim] = tensor.size(dim);
    for (const auto dimension : c10::irange(expected_size.size())) {
      TORCH_CHECK(
          expected_size[dimension] == tensor.size(dimension),
          "Input tensor at index ",
          i,
          " has invalid shape ",
          tensor.sizes(),
          ", but expected ",
          at::IntArrayRef(expected_size));
    }
    total_size += tensor.size(dim);
  }
  expected_size[dim] = total_size;
  TORCH_CHECK(
      out_tensor.sizes() == expected_size,
      "Expected out tensor to have shape ",
      at::IntArrayRef(expected_size),
      ", but got ",
      out_tensor.sizes());

  return gather_out_impl(tensors, out_tensor, dim);
}

at::Tensor gather(
    at::TensorList tensors,
    int64_t dim,
    std::optional<int32_t> destination_index) {
  TORCH_CHECK(!tensors.empty(), "Expected at least one tensor to gather from");
  int64_t total_size = 0;
  auto& first = tensors.front();
  const auto first_size = first.sizes();
  dim = at::maybe_wrap_dim(dim, first);
  std::vector<int64_t> expected_size(first_size.begin(), first_size.end());
  auto memory_format = first.suggest_memory_format();
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    TORCH_CHECK(
        is_flagos(tensor),
        "Expected all input tensors to be flagos tensors, but "
        "tensor at index ",
        i,
        " has device ",
        tensor.device());
    TORCH_CHECK(
        tensor.ndimension() == static_cast<int64_t>(expected_size.size()),
        "Expected all input tensors to have the same number of dimensions, but ",
        "tensor at index ",
        i,
        "has ",
        tensor.ndimension(),
        " dimensions, (expected ",
        expected_size.size(),
        ")");
    expected_size[dim] = tensor.size(dim);
    for (const auto dimension : c10::irange(expected_size.size())) {
      TORCH_CHECK(
          expected_size[dimension] == tensor.size(dimension),
          "Input tensor at index ",
          i,
          " has invalid shape ",
          tensor.sizes(),
          ", but expected ",
          at::IntArrayRef(expected_size));
    }
    total_size += tensor.size(dim);
    if (memory_format != at::MemoryFormat::Contiguous &&
        tensor.suggest_memory_format() != memory_format) {
      memory_format = at::MemoryFormat::Contiguous;
    }
  }
  expected_size[dim] = total_size;
  // -1 is how comm.gather spells "CPU" (it is what _get_device_index returns
  // for torch.device("cpu")), and a missing destination means this device. The
  // -1 reaches at::Device rather than being resolved here, because the class
  // has no meaningful index for a tensor: c10::Device(PrivateUse1, -1) is the
  // one spelling of "current flagos device" that the guard impl resolves.
  at::Device device(c10::DeviceType::CPU);
  if (!destination_index || *destination_index != -1) {
    device = at::Device(
        kFlagos,
        destination_index ? static_cast<c10::DeviceIndex>(*destination_index)
                          : c10::DeviceIndex(-1));
  }

  at::Tensor result =
      at::empty(expected_size, first.options().device(device), memory_format);
  return gather_out_impl(tensors, result, dim);
}

} // namespace

bool SetScatterScope(bool active) {
  const bool previous = scatter_scope;
  scatter_scope = active;
  return previous;
}

void InitDataParallelComm() {
  static bool installed = false;
  if (installed) {
    return;
  }
  installed = true;

  // torch._C is imported before this runs -- the whole point is to replace
  // attributes on the module the caller is already using, which is how
  // torch_npu reaches it from a separate extension module too.
  THPObjectPtr torch_C(PyImport_ImportModule("torch._C"));
  if (!torch_C) {
    throw python_error();
  }
  auto m = py::handle(torch_C).cast<py::module>();

  // Captured before the rebinding below, and held by the cpp_function
  // closures, so they live exactly as long as the replacements do. None when
  // this torch build has no CUDA comm layer to capture; see
  // original_or_none().
  const py::object orig_broadcast_coalesced =
      original_or_none(m, "_broadcast_coalesced");
  const py::object orig_broadcast = original_or_none(m, "_broadcast");
  const py::object orig_broadcast_out = original_or_none(m, "_broadcast_out");
  const py::object orig_scatter = original_or_none(m, "_scatter");
  const py::object orig_scatter_out = original_or_none(m, "_scatter_out");
  const py::object orig_gather = original_or_none(m, "_gather");
  const py::object orig_gather_out = original_or_none(m, "_gather_out");

  // Set the attributes, do not m.def them. module_::def passes whatever is
  // already under the name as a pybind11 *sibling*, which turns the
  // replacement into one more overload of the very function it replaces -- and
  // the original is then still tried first. Measured on PPU before this was a
  // setattr: torch._C._scatter came back as "Overloaded function. 1. <the CUDA
  // signature> 2. <ours>", a flagos tensor was still scattered to [flagos:0,
  // cuda:1] by overload 1, and _gather still raised its CUDA-only check, since
  // a TORCH_CHECK inside a matching overload is a RuntimeError, not the
  // TypeError pybind11 would fall through on. Setting the attribute outright
  // leaves exactly one implementation behind: this one.
  //
  // No call_guard<gil_scoped_release> on any of them either: the delegating
  // branch calls back into Python and needs the GIL. The flagos branch releases
  // it around the C++ work instead.
  py::cpp_function py_broadcast_coalesced(
      [orig_broadcast_coalesced](
          std::vector<at::Tensor>& tensors,
          const std::vector<int64_t>& devices,
          size_t buffer_size) -> py::object {
        if (has_stock_fallback(orig_broadcast_coalesced) &&
            !any_flagos(tensors)) {
          return orig_broadcast_coalesced(tensors, devices, buffer_size);
        }
        tensor_list2d outputs;
        {
          py::gil_scoped_release no_gil;
          outputs = broadcast_coalesced(tensors, devices, buffer_size);
        }
        return py::cast(std::move(outputs));
      },
      py::arg("tensors"),
      py::arg("devices"),
      py::arg("buffer_size"),
      py::name("_broadcast_coalesced"));

  py::cpp_function py_broadcast(
      [orig_broadcast](
          at::Tensor& tensor, const std::vector<int64_t>& devices) -> py::object {
        if (has_stock_fallback(orig_broadcast) && !is_flagos(tensor)) {
          return orig_broadcast(tensor, devices);
        }
        std::vector<at::Tensor> dst_tensors;
        {
          py::gil_scoped_release no_gil;
          dst_tensors = broadcast(tensor, devices);
        }
        return py::cast(std::move(dst_tensors));
      },
      py::arg("tensor"),
      py::arg("devices"),
      py::name("_broadcast"));

  py::cpp_function py_broadcast_out(
      [orig_broadcast_out](
          at::Tensor& tensor, std::vector<at::Tensor>& out_tensors) -> py::object {
        if (has_stock_fallback(orig_broadcast_out) && !any_flagos(out_tensors)) {
          return orig_broadcast_out(tensor, out_tensors);
        }
        {
          py::gil_scoped_release no_gil;
          broadcast_out(tensor, out_tensors);
        }
        return py::cast(out_tensors);
      },
      py::arg("tensor"),
      py::arg("out"),
      py::name("_broadcast_out"));

  py::cpp_function py_scatter(
      [orig_scatter](
          at::Tensor& tensor,
          std::vector<int64_t>& devices,
          const std::optional<std::vector<int64_t>>& chunk_sizes,
          int64_t dim,
          std::optional<py::object> py_streams) -> py::object {
        // A CPU source carries no device type of its own, so the scope is the
        // only thing that can still say the target devices are flagos ones;
        // see SetScatterScope.
        if (has_stock_fallback(orig_scatter) && !is_flagos(tensor) &&
            !scatter_scope) {
          return orig_scatter(
              tensor,
              devices,
              chunk_sizes,
              dim,
              py_streams ? *py_streams : py::none());
        }
        std::vector<at::Tensor> chunks;
        {
          py::gil_scoped_release no_gil;
          chunks = scatter(tensor, devices, chunk_sizes, dim);
        }
        return py::cast(std::move(chunks));
      },
      py::arg("tensor"),
      py::arg("devices"),
      py::arg("chunk_sizes"),
      py::arg("dim"),
      py::arg("streams"),
      py::name("_scatter"));

  py::cpp_function py_scatter_out(
      [orig_scatter_out](
          at::Tensor& tensor,
          std::vector<at::Tensor>& out_tensors,
          int64_t dim,
          std::optional<py::object> py_streams) -> py::object {
        if (has_stock_fallback(orig_scatter_out) && !any_flagos(out_tensors) &&
            !scatter_scope) {
          return orig_scatter_out(
              tensor,
              out_tensors,
              dim,
              py_streams ? *py_streams : py::none());
        }
        {
          py::gil_scoped_release no_gil;
          scatter_out(tensor, out_tensors, dim);
        }
        return py::cast(out_tensors);
      },
      py::arg("tensor"),
      py::arg("out"),
      py::arg("dim"),
      py::arg("streams"),
      py::name("_scatter_out"));

  py::cpp_function py_gather(
      [orig_gather](
          std::vector<at::Tensor>& tensors,
          int64_t dim,
          std::optional<int32_t> destination_index) -> py::object {
        if (has_stock_fallback(orig_gather) && !any_flagos(tensors)) {
          return orig_gather(tensors, dim, destination_index);
        }
        at::Tensor result;
        {
          py::gil_scoped_release no_gil;
          result = gather(tensors, dim, destination_index);
        }
        return py::cast(std::move(result));
      },
      py::arg("tensors"),
      py::arg("dim"),
      py::arg("destination_index"),
      py::name("_gather"));

  py::cpp_function py_gather_out(
      [orig_gather_out](
          std::vector<at::Tensor>& tensors,
          at::Tensor& out_tensor,
          int64_t dim) -> py::object {
        if (has_stock_fallback(orig_gather_out) && !any_flagos(tensors) &&
            !is_flagos(out_tensor)) {
          return orig_gather_out(tensors, out_tensor, dim);
        }
        {
          py::gil_scoped_release no_gil;
          gather_out(tensors, out_tensor, dim);
        }
        return py::cast(out_tensor);
      },
      py::arg("tensors"),
      py::arg("out"),
      py::arg("dim"),
      py::name("_gather_out"));

  py::setattr(m, "_broadcast_coalesced", py_broadcast_coalesced);
  py::setattr(m, "_broadcast", py_broadcast);
  py::setattr(m, "_broadcast_out", py_broadcast_out);
  py::setattr(m, "_scatter", py_scatter);
  py::setattr(m, "_scatter_out", py_scatter_out);
  py::setattr(m, "_gather", py_gather);
  py::setattr(m, "_gather_out", py_gather_out);
}

} // namespace torch_fl::comm
