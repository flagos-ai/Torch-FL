// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

#include <algorithm>

#include "aten/backends/apple/metal.h"
#include "runtime/functions.h"

namespace py = pybind11;
namespace mps = at::native::flagos::mps;

namespace {

mps::MetalGrid Grid(const std::vector<uint64_t>& dims) {
  TORCH_CHECK(!dims.empty() && dims.size() <= 3, "Metal grids have 1 to 3 dimensions");
  mps::MetalGrid result{1, 1, 1};
  std::copy(dims.begin(), dims.end(), result.begin());
  return result;
}

void Run(const mps::MetalKernel& kernel, const py::args& values,
         const std::vector<uint64_t>& threads,
         const std::optional<std::vector<uint64_t>>& group_size) {
  std::vector<mps::MetalArg> args;
  args.reserve(values.size());
  for (const auto& value : values) {
    if (THPVariable_Check(value.ptr())) {
      args.emplace_back(THPVariable_Unpack(value.ptr()));
    } else {
      args.emplace_back(py::cast<int64_t>(value));
    }
  }
  std::optional<mps::MetalGrid> group;
  if (group_size) group = Grid(*group_size);
  kernel.run(args, Grid(threads), group);
}

} // namespace

int InitMetalBindings(PyObject* mod) {
  HANDLE_TH_ERRORS
  auto module = py::reinterpret_borrow<py::module_>(mod);
  py::class_<mps::MetalKernel>(module, "_MetalKernel")
      .def(py::init<const std::string&>())
      .def("generated_kernel", &Run, py::arg("threads"), py::arg("group_size") = py::none());

  auto runtime = module.def_submodule("_native_host");
  runtime.def("current_stream", &c10::flagos::CurrentStreamHandle);
  runtime.def("capture_enqueues", [](const py::function& run) {
    py::object result;
    auto events = mps::CaptureEnqueues([&] { result = run(); });
    return py::make_tuple(result, events);
  });
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}
