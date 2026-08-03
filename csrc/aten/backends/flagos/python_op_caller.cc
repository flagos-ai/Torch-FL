// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/Generator.h>
#include <ATen/core/Generator.h>
#include <ATen/core/ivalue.h>
#include <c10/core/DeviceGuard.h>

#include "runtime/functions.h"

#include <mutex>
#include <unordered_map>

namespace py = pybind11;
using namespace pybind11::literals;

namespace at::native::flagos {

namespace {

struct PythonOpCache {
  py::module_ ops_module;
  std::unordered_map<std::string, py::object> func_cache;
  std::mutex init_mutex;

  // Kept for call-site symmetry; the flag_gems.ops import it used to do eagerly
  // now happens on first bare-name lookup (see GetFunc). This caller is shared
  // with TileOPs, whose targets are dotted qualnames in torch_fl and need no
  // flag_gems at all -- importing it up front made every TileOPs call fail on a
  // host that has TileOPs but not FlagGems.
  void EnsureInitialized() {}

  // flag_gems.ops, imported once on demand. Only bare-name lookups need it.
  py::module_& OpsModule() {
    if (!ops_module) {
      ops_module = py::module_::import("flag_gems.ops");
    }
    return ops_module;
  }

  py::object GetFunc(const char* name) {
    std::lock_guard<std::mutex> lock(init_mutex);
    auto it = func_cache.find(name);
    if (it != func_cache.end()) return it->second;
    std::string qual(name);
    py::object func;
    auto dot = qual.rfind('.');
    if (dot == std::string::npos) {
      // Bare name: resolve from the flag_gems.ops module (legacy typed callers).
      func = OpsModule().attr(name);
    } else {
      // Dotted qualname "module.submodule.func": import the module, getattr func.
      // This is what the auto-discovered generic kernels pass, taken from
      // fn.__module__ + "." + fn.__name__, so it locates the exact callable in
      // _FULL_CONFIG regardless of whether flag_gems.ops re-exports it.
      std::string module_path = qual.substr(0, dot);
      std::string func_name = qual.substr(dot + 1);
      try {
        py::module_ mod = py::module_::import(module_path.c_str());
        func = mod.attr(func_name.c_str());
      } catch (const py::error_already_set&) {
        // The qualname is recorded on whichever box ran the discovery codegen, so
        // a vendor-private override module gets frozen into the generated file:
        // 28 of these say "_hygon.ops.<op>" (mul, mm, silu, sort, ...), which
        // does not exist on an Ascend host -> ModuleNotFoundError: No module
        // named '_hygon.ops.mul' the first time FlagGems mode multiplied two
        // tensors. flag_gems.ops re-exports the vendor-resolved callable for
        // every one of them, so retry there; the dotted path stays the preferred
        // lookup because it pins the exact _FULL_CONFIG entry when it does exist.
        PyErr_Clear();
        func = ops_module.attr(func_name.c_str());
      }
    }
    func_cache[name] = func;
    return func;
  }
};

PythonOpCache& GetCache() {
  // Intentionally leaked: prevent destructor from running after Python
  // interpreter finalization, which would crash on Py_DECREF with no GIL.
  static PythonOpCache* cache = new PythonOpCache();
  return *cache;
}

// Resolve the flagos device a factory call should produce on.
//
// Mirrors aten factory semantics (and csrc/aten/empty.cc): an absent device, or
// one carrying no index, means "the current device". Hardcoding index 0 here
// used to allocate the output on device 0 while gems' Triton kernel launched on
// whatever the current device was -- on a multi-GPU run that is a cross-device
// write, which faults the GPU (VMFault / "Invalid address access") instead of
// raising, and it also made torch.ones(..., device="flagos:1") report device 0.
c10::Device ResolveFactoryDevice(std::optional<at::Device> device) {
  if (device.has_value() && device->has_index()) {
    return *device;
  }
  return c10::Device(c10::DeviceType::PrivateUse1, c10::flagos::CurrentDevice());
}

// Resolve the device a *non-factory* Python call must run on: the device of its
// first indexed flagos tensor operand, or nullopt to leave the current device
// alone.
//
// Same hazard the factory callers guard against, reached from the other side.
// gems allocates its own intermediates with `device=input.device` but launches
// Triton on the *current* device, so calling an op on a tensor that lives on
// device N while the current device is M reads and writes across devices. That
// does not raise -- it faults the GPU (segfault / VMFault / "Invalid address
// access"), which is how multinomial on flagos:1 died before this guard.
//
// Guarding here also fixes TensorToPython's CPU-scalar hop, which resolves to
// the current device: under the guard that is now the operand's device rather
// than whatever was current at entry.
std::optional<c10::Device> DeviceOfTensor(const at::Tensor& t) {
  if (t.defined() && t.device().is_privateuseone() && t.device().has_index()) {
    return t.device();
  }
  return std::nullopt;
}

// Same, for the boxed-argument callers: first tensor (or first tensor in a
// tensor list) that names a device wins. Non-tensor leading args are skipped,
// so ops like `topk(values, k)` still resolve correctly.
std::optional<c10::Device> DeviceOfArgs(const std::vector<c10::IValue>& args) {
  for (const auto& arg : args) {
    if (arg.isTensor()) {
      if (auto dev = DeviceOfTensor(arg.toTensor())) return dev;
    } else if (arg.isTensorList()) {
      for (const at::Tensor& t : arg.toTensorList()) {
        if (auto dev = DeviceOfTensor(t)) return dev;
      }
    }
  }
  return std::nullopt;
}

// Same, for the fixed-arity callers: first tensor operand naming a device wins.
template <typename... Ts>
std::optional<c10::Device> DeviceOfFirst(const at::Tensor& t, const Ts&... rest) {
  if (auto dev = DeviceOfTensor(t)) return dev;
  if constexpr (sizeof...(rest) > 0) {
    return DeviceOfFirst(rest...);
  }
  return std::nullopt;
}

// Convert at::Tensor to Python THPVariable.
// CPU scalar tensors are moved to the flagos device since FlagGems kernels
// cannot access CPU memory.
py::object TensorToPython(const at::Tensor& t) {
  if (!t.defined()) return py::none();
  if (t.device().is_cpu() && t.dim() == 0) {
    // Current device, not index 0: a scalar operand feeding a computation on
    // device N must not drag that computation back to device 0.
    auto dev_t = t.to(ResolveFactoryDevice(std::nullopt));
    PyObject* obj = THPVariable_Wrap(dev_t);
    return py::reinterpret_steal<py::object>(obj);
  }
  PyObject* obj = THPVariable_Wrap(t);
  return py::reinterpret_steal<py::object>(obj);
}

// Convert Python THPVariable back to at::Tensor.
at::Tensor PythonToTensor(const py::object& obj) {
  if (obj.is_none()) return at::Tensor();
  PyObject* raw = obj.ptr();
  TORCH_CHECK(THPVariable_Check(raw), "Expected a Tensor from Python op");
  return THPVariable_Unpack(raw);
}

// Convert at::Scalar to Python
py::object ScalarToPython(const at::Scalar& s) {
  if (s.isFloatingPoint()) {
    return py::float_(s.toDouble());
  } else if (s.isIntegral(/*includeBool=*/false)) {
    return py::int_(s.toLong());
  } else if (s.isBoolean()) {
    return py::bool_(s.toBool());
  } else if (s.isComplex()) {
    auto c = s.toComplexDouble();
    return py::cast(c);
  }
  return py::float_(s.toDouble());
}

// Convert IntArrayRef to Python tuple
py::object IntArrayRefToPython(at::IntArrayRef arr) {
  py::tuple t(arr.size());
  for (size_t i = 0; i < arr.size(); ++i) {
    t[i] = py::int_(arr[i]);
  }
  return t;
}

// Convert OptionalIntArrayRef to Python
// Empty dim list means "reduce all dims" which maps to None in FlagGems.
py::object OptionalIntArrayRefToPython(at::OptionalIntArrayRef arr) {
  if (!arr.has_value()) return py::none();
  return IntArrayRefToPython(*arr);
}

// Convert optional<ScalarType> to Python
py::object OptionalDtypeToPython(std::optional<at::ScalarType> dtype) {
  if (!dtype.has_value()) return py::none();
  // Import torch and get the dtype object
  static py::module_ torch_mod = py::module_::import("torch");
  switch (*dtype) {
    case at::ScalarType::Float:   return torch_mod.attr("float32");
    case at::ScalarType::Double:  return torch_mod.attr("float64");
    case at::ScalarType::Half:    return torch_mod.attr("float16");
    case at::ScalarType::BFloat16: return torch_mod.attr("bfloat16");
    case at::ScalarType::Int:     return torch_mod.attr("int32");
    case at::ScalarType::Long:    return torch_mod.attr("int64");
    case at::ScalarType::Short:   return torch_mod.attr("int16");
    case at::ScalarType::Byte:    return torch_mod.attr("uint8");
    case at::ScalarType::Char:    return torch_mod.attr("int8");
    case at::ScalarType::Bool:    return torch_mod.attr("bool");
    default: return py::none();
  }
}

// Convert a single IValue to a Python object, covering the argument types the
// codegen'd generic FlagGems kernels can produce. Note: ScalarType is
// deliberately NOT handled here -- an IValue stores ScalarType as a plain int
// (see c10::IValue(ScalarType)), so it is indistinguishable from an ordinary
// int at runtime. Ops that pass a dtype to the FlagGems function are excluded
// from the generic path (kept in the codegen skip list) instead.
py::object IValueToPython(const c10::IValue& val, const char* func_name) {
  if (val.isTensor()) {
    return TensorToPython(val.toTensor());
  } else if (val.isInt()) {
    return py::int_(val.toInt());
  } else if (val.isDouble()) {
    return py::float_(val.toDouble());
  } else if (val.isBool()) {
    return py::bool_(val.toBool());
  } else if (val.isNone()) {
    return py::none();
  } else if (val.isString()) {
    return py::str(val.toStringRef());
  } else if (val.isScalar()) {
    return ScalarToPython(val.toScalar());
  } else if (val.isIntList()) {
    auto list = val.toIntList();
    py::tuple t(list.size());
    for (size_t j = 0; j < list.size(); ++j) {
      t[j] = py::int_(static_cast<int64_t>(list[j]));
    }
    return std::move(t);
  } else if (val.isDoubleList()) {
    auto list = val.toDoubleList();
    py::tuple t(list.size());
    for (size_t j = 0; j < list.size(); ++j) {
      t[j] = py::float_(static_cast<double>(list[j]));
    }
    return std::move(t);
  } else if (val.isBoolList()) {
    auto list = val.toBoolList();
    py::tuple t(list.size());
    for (size_t j = 0; j < list.size(); ++j) {
      t[j] = py::bool_(static_cast<bool>(list[j]));
    }
    return std::move(t);
  } else if (val.isTensorList()) {
    auto list = val.toTensorList();
    py::list t;
    for (size_t j = 0; j < list.size(); ++j) {
      t.append(TensorToPython(list.get(j)));
    }
    return std::move(t);
  }
  TORCH_CHECK(false, "Unsupported IValue type in generic FlagGems caller for op: ",
              func_name);
}

// Build the Python positional-arg tuple from a vector of IValues.
py::tuple BuildPyArgs(const std::vector<c10::IValue>& args, const char* func_name) {
  py::tuple py_args(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    py_args[i] = IValueToPython(args[i], func_name);
  }
  return py_args;
}

} // namespace

at::Tensor CallPythonOp_T(const char* func_name, const at::Tensor& self) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(self));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(self));
  return PythonToTensor(result);
}

at::Tensor& CallPythonOp_T_inplace(const char* func_name, at::Tensor& self) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(self));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  func(TensorToPython(self));
  return self;
}

at::Tensor CallPythonOp_TT(const char* func_name, const at::Tensor& a, const at::Tensor& b) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(a, b));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(a), TensorToPython(b));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_TTS(const char* func_name, const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(a, b));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(a), TensorToPython(b), "alpha"_a = ScalarToPython(alpha));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_TS(const char* func_name, const at::Tensor& self, const at::Scalar& other) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(self));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(self), ScalarToPython(other));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_TIB(const char* func_name, const at::Tensor& self, int64_t dim, bool flag) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(self));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(self), py::int_(dim), py::bool_(flag));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_TOIB(const char* func_name, const at::Tensor& self,
                              at::OptionalIntArrayRef dim, bool keepdim,
                              std::optional<at::ScalarType> dtype) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(self));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(
      TensorToPython(self),
      OptionalIntArrayRefToPython(dim),
      py::bool_(keepdim),
      "dtype"_a = OptionalDtypeToPython(dtype));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_TTT(const char* func_name, const at::Tensor& a, const at::Tensor& b, const at::Tensor& c) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(a, b, c));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(a), TensorToPython(b), TensorToPython(c));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_TD(const char* func_name, const at::Tensor& self,
                            std::optional<at::ScalarType> dtype) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(self));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(self), "dtype"_a = OptionalDtypeToPython(dtype));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_ListI(const char* func_name,
                              const at::ITensorListRef& tensors, int64_t dim) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  std::optional<c10::Device> list_dev;
  for (const auto& t : tensors) {
    if ((list_dev = DeviceOfTensor(t))) break;
  }
  const c10::OptionalDeviceGuard device_guard(list_dev);
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::list py_tensors;
  for (const auto& t : tensors) {
    py_tensors.append(TensorToPython(t));
  }
  py::object result = func(py_tensors, py::int_(dim));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_Embedding(const char* func_name, const at::Tensor& weight,
                                  const at::Tensor& indices, int64_t padding_idx,
                                  bool scale_grad_by_freq, bool sparse) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(weight, indices));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(
      TensorToPython(weight), TensorToPython(indices),
      py::int_(padding_idx), py::bool_(scale_grad_by_freq), py::bool_(sparse));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_Generic(const char* func_name, const std::vector<c10::IValue>& args) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfArgs(args));

  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::tuple py_args = BuildPyArgs(args, func_name);
  return PythonToTensor(func(*py_args));
}

namespace {

// Build the **kwargs dict for a keyword-arg call. `is_dtype` kwargs convert the
// int ScalarType payload to a torch.dtype (see PyKwarg); `is_none` kwargs pass
// Python None (an absent optional the IValue can't otherwise represent).
py::dict BuildPyKwargs(const std::vector<PyKwarg>& kwargs, const char* func_name) {
  py::dict d;
  for (const auto& kw : kwargs) {
    if (kw.is_none) {
      d[kw.name] = py::none();
    } else if (kw.is_dtype) {
      d[kw.name] = OptionalDtypeToPython(
          static_cast<at::ScalarType>(kw.value.toInt()));
    } else {
      d[kw.name] = IValueToPython(kw.value, func_name);
    }
  }
  return d;
}

} // namespace

at::Tensor CallPythonOp_GenericKw(const char* func_name,
                                  const std::vector<c10::IValue>& args,
                                  const std::vector<PyKwarg>& kwargs) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfArgs(args));

  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::tuple py_args = BuildPyArgs(args, func_name);
  py::dict py_kwargs = BuildPyKwargs(kwargs, func_name);
  return PythonToTensor(func(*py_args, **py_kwargs));
}

at::Tensor CallPythonOp_Factory(const char* func_name,
                                const std::vector<c10::IValue>& args,
                                std::optional<at::ScalarType> dtype,
                                std::optional<at::Device> device) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  // Set the device before entering Python: gems' internal torch.empty and the
  // Triton launch both read the *current* device, so they must see the one we
  // are about to name in the device kwarg.
  const auto target = ResolveFactoryDevice(device);
  const c10::DeviceGuard device_guard(target);
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::tuple py_args = BuildPyArgs(args, func_name);

  static py::module_ torch_mod = py::module_::import("torch");
  // device=flagos:<idx> -> gems' internal torch.empty(...) allocates on
  // PrivateUse1 via our own allocator (no recursion, no CUDA copy). Uses the
  // registered PrivateUse1 backend name so it stays correct if the alias changes.
  py::object flagos_dev = torch_mod.attr("device")(
      torch_mod.attr("_C").attr("_get_privateuse1_backend_name")(),
      static_cast<int>(target.index()));
  py::object result = func(
      *py_args,
      "dtype"_a = OptionalDtypeToPython(dtype),
      "layout"_a = torch_mod.attr("strided"),
      "device"_a = flagos_dev,
      "pin_memory"_a = py::none());
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_LikeFactory(const char* func_name,
                                    const std::vector<c10::IValue>& args,
                                    std::optional<at::ScalarType> dtype,
                                    std::optional<at::Device> device) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  // For *_like, an absent device means "same as self" (not "current device"), so
  // fall back to args[0]'s device before ResolveFactoryDevice's current-device
  // default. args[0] is the source tensor by construction -- see the header.
  std::optional<at::Device> device_hint = device;
  if ((!device_hint.has_value() || !device_hint->has_index()) && !args.empty() &&
      args[0].isTensor()) {
    const auto& self = args[0].toTensor();
    if (self.defined() && self.device().is_privateuseone() &&
        self.device().has_index()) {
      device_hint = self.device();
    }
  }
  const auto target = ResolveFactoryDevice(device_hint);
  const c10::DeviceGuard device_guard(target);
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::tuple py_args = BuildPyArgs(args, func_name);

  static py::module_ torch_mod = py::module_::import("torch");
  // device=flagos:<idx> -> gems' internal torch.empty_like(self, device=...)
  // stays on PrivateUse1 (no CUDA round-trip). dtype=None means "same as self".
  py::object flagos_dev = torch_mod.attr("device")(
      torch_mod.attr("_C").attr("_get_privateuse1_backend_name")(),
      static_cast<int>(target.index()));
  py::object result = func(
      *py_args,
      "dtype"_a = OptionalDtypeToPython(dtype),
      "layout"_a = torch_mod.attr("strided"),
      "device"_a = flagos_dev,
      "pin_memory"_a = py::none(),
      "memory_format"_a = py::none());
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_RandomInplace(const char* func_name,
                                      const std::vector<c10::IValue>& args) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfArgs(args));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::tuple py_args = BuildPyArgs(args, func_name);
  // generator=None: gems reads its seed+offset from the per-device CUDA
  // generators the compat shim installs as torch.cuda.default_generators (same
  // path as rand/randn/multinomial). Passing an explicit generator object here
  // raced with triton's cold-cache first-compile.
  py::object result = func(*py_args);
  return PythonToTensor(result);
}

std::vector<at::Tensor> CallPythonOp_GenericKwTuple(
    const char* func_name, const std::vector<c10::IValue>& args,
    const std::vector<PyKwarg>& kwargs, int64_t n) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfArgs(args));

  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::tuple py_args = BuildPyArgs(args, func_name);
  py::dict py_kwargs = BuildPyKwargs(kwargs, func_name);
  py::object result = func(*py_args, **py_kwargs);

  py::sequence seq = py::reinterpret_borrow<py::sequence>(result);
  TORCH_CHECK(static_cast<int64_t>(py::len(seq)) == n,
              "Expected ", n, " return values from FlagGems op ", func_name,
              ", got ", py::len(seq));
  std::vector<at::Tensor> out;
  out.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    out.push_back(PythonToTensor(py::reinterpret_borrow<py::object>(seq[i])));
  }
  return out;
}

std::vector<at::Tensor> CallPythonOp_GenericTuple(
    const char* func_name, const std::vector<c10::IValue>& args, int64_t n) {
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfArgs(args));

  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::tuple py_args = BuildPyArgs(args, func_name);
  py::object result = func(*py_args);

  // FlagGems tuple-returning ops give back a tuple/list of Tensors (some may be
  // None, e.g. an optional running-stats output); map None -> undefined Tensor.
  py::sequence seq = py::reinterpret_borrow<py::sequence>(result);
  TORCH_CHECK(static_cast<int64_t>(py::len(seq)) == n,
              "Expected ", n, " return values from FlagGems op ", func_name,
              ", got ", py::len(seq));
  std::vector<at::Tensor> out;
  out.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    out.push_back(PythonToTensor(py::reinterpret_borrow<py::object>(seq[i])));
  }
  return out;
}

at::Generator GetFlagosDefaultCudaGenerator(int64_t device_index) {
  static std::mutex cache_mu;
  static std::unordered_map<int64_t, at::Generator> gen_cache;
  {
    std::lock_guard<std::mutex> lk(cache_mu);
    auto it = gen_cache.find(device_index);
    if (it != gen_cache.end()) {
      return it->second;
    }
  }
  py::gil_scoped_acquire gil;
  py::module_ torch_cuda = py::module_::import("torch.cuda");
  if (py::len(torch_cuda.attr("default_generators")) == 0) {
    torch_cuda.attr("init")();
  }
  py::object generators = torch_cuda.attr("default_generators");
  at::Generator generator =
      generators[py::cast(device_index)].cast<at::Generator>();
  {
    std::lock_guard<std::mutex> lk(cache_mu);
    gen_cache[device_index] = generator;
  }
  return generator;
}

std::pair<uint64_t, uint64_t> GetFlagosPhiloxState(
    int64_t device_index, uint64_t increment) {
  py::gil_scoped_acquire gil;
  py::module_ torch_cuda = py::module_::import("torch.cuda");
  py::object generators = torch_cuda.attr("default_generators");
  py::object generator = generators[py::cast(device_index)];
  py::object state = generator.attr("get_state")();
  auto values = state.cast<at::Tensor>().contiguous().view(at::kLong);
  TORCH_CHECK(values.numel() == 2,
              "GCU philox generator state must contain seed and offset");
  auto seed = static_cast<uint64_t>(values[0].item<int64_t>());
  auto offset = static_cast<uint64_t>(values[1].item<int64_t>());
  // FlagGems rounds each philox draw up to a multiple of 4; matching that keeps
  // native and FlagGems draws on non-overlapping segments of one stream.
  increment = (increment + 3) / 4 * 4;
  values[1] = static_cast<int64_t>(offset + increment);
  generator.attr("set_state")(values.view(at::kByte));
  return {seed, offset};
}

} // namespace at::native::flagos
