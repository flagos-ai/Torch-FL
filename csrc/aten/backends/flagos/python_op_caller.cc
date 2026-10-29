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

// soft_lowp::Fp8PythonDtype, for the keyword dtypes a decoded FP8 pair carries:
// python_dtype_for_fp8 on the generic callers below.
#include "../soft_lowp/format.h"
#include "runtime/functions.h"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace at::native::flagos {

namespace {

// One counter for every Python op call this process makes; see
// PythonOpCallsStarted in the header. Atomic because a caller runs on whatever
// thread dispatched the op.
std::atomic<uint64_t> python_op_calls{0};

} // namespace

uint64_t PythonOpCallsStarted() {
  return python_op_calls.load(std::memory_order_relaxed);
}

void NotePythonOpCall() {
  python_op_calls.fetch_add(1, std::memory_order_relaxed);
}

namespace {

struct PythonOpCache {
  py::module_ gems_module;
  py::module_ ops_module;
  std::unordered_map<std::string, py::object> func_cache;
  std::mutex init_mutex;

  // Kept for call-site symmetry; the flag_gems.ops import it used to do eagerly
  // now happens on first bare-name lookup (see GetFunc). This caller is shared
  // with TileOPs, whose targets are dotted qualnames in torch_fl and need no
  // flag_gems at all -- importing it up front made every TileOPs call fail on a
  // host that has TileOPs but not FlagGems.
  void EnsureInitialized() {}

  // flag_gems package, imported once on demand. Its top level is where the
  // SpecOpRegistrar installs the vendor-resolved callable for the active
  // backend, so it is the authoritative source (see VendorOpByName).
  py::module_& GemsModule() {
    if (!gems_module) {
      gems_module = py::module_::import("flag_gems");
    }
    return gems_module;
  }

  // flag_gems.ops, imported once on demand. The fallback for bare-name lookups.
  py::module_& OpsModule() {
    if (!ops_module) {
      ops_module = py::module_::import("flag_gems.ops");
    }
    return ops_module;
  }

  // The active backend's callable for `func_name`, obtained through FlagGems'
  // own resolver: the SpecOpRegistrar in flag_gems/__init__.py rebinds the
  // package-level name for every op the vendor overrides, while the generic
  // flag_gems.ops submodule keeps the stock implementation. Vendors whose
  // tune_configs.yaml declares a different META schema than the generic kernel
  // (tsingmicro's mm adds SPLIT_K; TILE_M/N/K for bmm) rely on exactly this:
  // the vendor mm takes SPLIT_K, the generic one does not, so resolving to the
  // generic callable turns the tuned config into
  // "KeyError: Keyword argument SPLIT_K was specified but unrecognised".
  //
  // Returns an empty object when the name is absent or not callable at the
  // package top level, so the caller can fall back to the dotted path.
  py::object VendorOpByName(const char* func_name) {
    py::object candidate;
    try {
      candidate = GemsModule().attr(func_name);
    } catch (const py::error_already_set&) {
      // AttributeError: not every spec op is re-exported (a few private
      // helpers exist only under flag_gems.ops).
      PyErr_Clear();
      return py::object();
    }
    if (!py::isinstance<py::function>(candidate)) {
      // Do not let a module/type/constant shadow a real kernel. Every name in
      // the generated tables is a function in the cohort codegen ran against.
      return py::object();
    }
    return candidate;
  }

  // `name` is a bare function name or a dotted "module.submodule.func", and it
  // is also the cache key. Callers that reach one function through two
  // spellings use GetFuncBare below, which keys it under the bare name.
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
      // Dotted qualname "module.submodule.func". codegen freezes the *generic*
      // package path here -- _normalize_flaggems_qualname maps every discovery
      // alias (including vendor-private "_tsingmicro.ops.mm") to
      // flag_gems.ops.* so the generated artifact stays hardware-portable.
      //
      // Portability in the artifact must not mean "run the generic kernel".
      // Discovery measured arity/category/kwargs against the *vendor* callable
      // that _FULL_CONFIG held, so calling the generic one mismatches on
      // exactly the ops the vendor tuned: a tsingmicro mm invocation reaches
      // mm_kernel_general through flag_gems.ops.mm.mm while the tuned config
      // it is handed carries SPLIT_K, which only _tsingmicro.ops.mm accepts.
      //
      // So re-resolve through the package top level first. That keeps the
      // artifact portable (the frozen path is still what codegen pinned) while
      // letting FlagGems' registrar pick the vendor implementation for whatever
      // backend is actually active.
      std::string module_path = qual.substr(0, dot);
      std::string func_name = qual.substr(dot + 1);
      func = VendorOpByName(func_name.c_str());
      if (!func) {
        try {
          py::module_ mod = py::module_::import(module_path.c_str());
          func = mod.attr(func_name.c_str());
        } catch (const py::error_already_set&) {
          // The qualname is recorded on whichever box ran the discovery codegen,
          // so a vendor-private override module gets frozen into the generated
          // file: 28 of these say "_hygon.ops.<op>" (mul, mm, silu, sort, ...),
          // which does not exist on an Ascend host -> ModuleNotFoundError: No
          // module named '_hygon.ops.mul' the first time FlagGems mode
          // multiplied two tensors. flag_gems.ops re-exports the vendor-resolved
          // callable for every one of them, so retry there.
          PyErr_Clear();
          func = OpsModule().attr(func_name.c_str());
        }
      }
    }
    func_cache[name] = func;
    return func;
  }

  // The callable for a keyword-arg call (CallPythonOp_GenericKw and the tuple
  // form), keyed by the bare function name whichever spelling asks for it. The
  // generated tables and a hand-written caller can name the same kernel more
  // than one way ("flag_gems.mm_out" and "flag_gems.ops.mm.mm_out"), and GetFunc
  // would cache those as separate entries that must be the same callable. Every
  // caller in this tree passes a qualified name, and both qualifying spellings
  // of a flag_gems kernel start with "flag_gems", so they take the same path
  // below and the one cache entry stays right. A bare name carries no qualifier
  // to classify -- this cache is shared with the TileOPs shims, which must not
  // import flag_gems on a host that lacks it -- so it resolves from flag_gems.ops
  // as before.
  //
  // A flag_gems target resolves through the active backend first, for the same
  // reason GetFunc's dotted path does. The generated keyword-arg wrappers were
  // discovered against the callable `_FULL_CONFIG` held, which is the
  // vendor-resolved one, so their parameter names are the vendor's -- and the
  // tuned META their op is handed is the vendor's too. Resolving to the stock
  // flag_gems.ops implementation is then not "the generic kernel with the same
  // arguments": it is the generic kernel with a config schema it does not
  // declare. That is how `mm.out` dies on Ascend (CI 40656617119 at 0a2a74b3):
  // the Ascend tune config for `mm` carries BLOCK_*/GROUP_M/SPLIT_K/EVEN_K, the
  // vendor kernel `_ascend.ops.mm` declares them, and
  // flag_gems/ops/mm.py:mm_kernel_general does not -- so the tuner's own
  // benchmark run raises "KeyError: 'Keyword argument SPLIT_K was specified but
  // unrecognised'" from triton's _pack_args before any kernel is compiled.
  // `mm` itself is unaffected only because it is a dotted call and goes
  // through GetFunc, which already asks VendorOpByName; the `.out` spelling is
  // a keyword call and did not.
  // It is not Ascend-specific: tsingmicro routes mm.out and bmm.out to FlagGems
  // over the same kind of divergent kernel.
  //
  // flag_gems.ops stays the fallback (a few private helpers exist only there),
  // and a non-flag_gems qualname -- the TileOPs shims -- is left alone, so a
  // host with TileOPs but no FlagGems still never imports flag_gems here.
  py::object GetFuncBare(const char* name) {
    std::lock_guard<std::mutex> lock(init_mutex);
    std::string qual(name);
    auto dot = qual.rfind('.');
    std::string bare = dot == std::string::npos ? qual : qual.substr(dot + 1);
    auto it = func_cache.find(bare);
    if (it != func_cache.end()) return it->second;
    py::object func;
    if (qual.rfind("flag_gems", 0) == 0) {
      func = VendorOpByName(bare.c_str());
    }
    if (!func) func = OpsModule().attr(bare.c_str());
    func_cache[bare] = func;
    return func;
  }
};

// torch.dtype objects for the four FP8 scalar types: the resolver the software
// low-precision path installs (soft_lowp/format.h). It lives here because this
// file is compiled only with the FlagGems route, is the file that has pybind11,
// and holds the function cache a keyword-arg call is made through.
//
// The path handling exists because importing torch from C++ starts a fresh
// interpreter into which torch has NOT been imported: `torch` is then whatever
// sits first on the path, not the torch this process is running on. Its own
// modules fail to import (`No module named 'torch._strobelight'`) or it is a
// different build's copy. The sibling directories of torch_fl -- the
// site-packages holding it and one level above, plus the interpreter home --
// reach the same install the rest of the process uses. An entry already on the
// path is left where it is, and the additions are taken back if the import
// fails.
namespace {

std::vector<std::string> SiblingDirs(const std::string& module_file) {
  std::vector<std::string> dirs;
  auto end = module_file.size();
  for (int up = 1; up <= 2; ++up) {
    end = module_file.rfind('/', end == 0 ? 0 : end - 1);
    if (end == std::string::npos) return dirs;
    dirs.push_back(module_file.substr(0, end));
  }
  if (const char* home = std::getenv("PYTHONHOME")) {
    dirs.push_back(std::string(home) + "/lib");
  }
  return dirs;
}

void* Fp8DtypeObjectImpl(c10::ScalarType dtype) {
  const char* name = nullptr;
  switch (dtype) {
    case c10::kFloat8_e4m3fn:   name = "float8_e4m3fn"; break;
    case c10::kFloat8_e5m2:     name = "float8_e5m2"; break;
    case c10::kFloat8_e4m3fnuz: name = "float8_e4m3fnuz"; break;
    case c10::kFloat8_e5m2fnuz: name = "float8_e5m2fnuz"; break;
    default:                    return nullptr;
  }
  // Resolution of a dtype object is not a routed op, so it takes the GIL but
  // does not count as a Python op call.
  py::gil_scoped_acquire gil;
  static py::object cached;
  if (!cached) {
    py::object torch_fl = py::module_::import("torch_fl");
    py::object sys_path = py::module_::import("sys").attr("path");
    std::size_t added = 0;
    for (const auto& dir :
         SiblingDirs(torch_fl.attr("__file__").cast<std::string>())) {
      if (!sys_path.attr("__contains__")(dir).cast<bool>()) {
        sys_path.attr("insert")(0, dir);
        ++added;
      }
    }
    try {
      cached = py::module_::import("torch").attr(name);
    } catch (...) {
      sys_path.attr("__delitem__")(py::slice(0, static_cast<py::ssize_t>(added), 1));
      throw;
    }
  }
  return cached.ptr();
}

} // namespace

PythonOpCache& GetCache() {
  // Intentionally leaked: prevent destructor from running after Python
  // interpreter finalization, which would crash on Py_DECREF with no GIL.
  static PythonOpCache* cache = new PythonOpCache();
  return *cache;
}

// Every caller below constructs one, which is what keeps PythonOpCallsStarted a
// count of calls: a call can resolve a FlagGems callable (the lookup is cached
// per process, on first use), so the count has to start before anything is
// looked up, and it does not matter which caller did the looking.
class CallScope {
 public:
  CallScope() { NotePythonOpCall(); }
};

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

py::object ScalarToPython(const at::Scalar& s);

// Convert at::Tensor to Python THPVariable.
// CPU scalar tensors are moved to the flagos device since FlagGems kernels
// cannot access CPU memory.
py::object TensorToPython(const at::Tensor& t) {
  if (!t.defined()) return py::none();
  // A number written in Python for an op ATen only declares with a Tensor
  // parameter -- `tensor + 0.1` is the common case -- reaches this caller as a
  // *wrapped-number* 0-dim CPU tensor, because ATen boxes the scalar that way
  // before dispatching onto the .Tensor overload. is_wrapped_number() is what
  // tells ATen's type promotion to keep the operand at its category instead of
  // promoting with the full tensor dtype (ATen/native/TypeProperties.h), and
  // FlagGems does not implement that flag: it sees a plain 0-dim tensor and
  // promotes fp32 + fp64 -> fp64, which the Ascend backend cannot compile
  // ("[fp8, fp64] is unsupported on Ascend for now"). Hand the number over as
  // the Python scalar it came from instead; FlagGems' scalar overloads then
  // follow ATen's rule and the kernel stays in the tensor operand's dtype.
  if (t.unsafeGetTensorImpl()->is_wrapped_number()) {
    return ScalarToPython(t.item());
  }
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
//
// Every ScalarType this build can build a FlagGems call for has an arm here.
// The `default` is reserved for a dtype this switch genuinely does not name,
// and it must NOT silently become Python `None`: a factory op reads `None` as
// "use torch.get_default_dtype()", so an unnamed dtype came back as float32
// rather than failing (issue #540).
//
// complex64/complex128 are named here so the requested dtype reaches FlagGems
// intact. Whether FlagGems can then serve it is a separate question, asked
// before the call rather than by lossily dropping the dtype here: a
// CUDA-ABI boxing build (MetaX, DCU) reroutes the complex call to its vendor
// kernels, and a wheel with no complex-capable route now fails loudly inside
// FlagGems instead of returning a real tensor for a complex request.
py::object OptionalDtypeToPython(std::optional<at::ScalarType> dtype) {
  if (!dtype.has_value()) return py::none();
  // A decoded low-precision pair carries one of the four FP8 dtypes to every
  // route that takes its dtype -- a FlagGems keyword and the dedicated
  // dtype-conversion callers alike -- so the FP8 objects are named here rather
  // than at each of those calls.
  if (void* fp8 = soft_lowp::Fp8PythonDtype(*dtype)) {
    return py::reinterpret_borrow<py::object>(static_cast<PyObject*>(fp8));
  }
  // Import torch and get the dtype object
  static py::module_ torch_mod = py::module_::import("torch");
  switch (*dtype) {
    case at::ScalarType::Float:        return torch_mod.attr("float32");
    case at::ScalarType::Double:       return torch_mod.attr("float64");
    case at::ScalarType::Half:         return torch_mod.attr("float16");
    case at::ScalarType::BFloat16:     return torch_mod.attr("bfloat16");
    case at::ScalarType::Int:          return torch_mod.attr("int32");
    case at::ScalarType::Long:         return torch_mod.attr("int64");
    case at::ScalarType::Short:        return torch_mod.attr("int16");
    case at::ScalarType::Byte:         return torch_mod.attr("uint8");
    case at::ScalarType::Char:         return torch_mod.attr("int8");
    case at::ScalarType::Bool:         return torch_mod.attr("bool");
    case at::ScalarType::ComplexFloat: return torch_mod.attr("complex64");
    case at::ScalarType::ComplexDouble: return torch_mod.attr("complex128");
    default:
      TORCH_CHECK(false,
                  "OptionalDtypeToPython has no Python dtype for ScalarType ",
                  static_cast<int>(*dtype),
                  "; add it rather than letting the caller fall back to the "
                  "default dtype");
      return py::none();  // unreachable: TORCH_CHECK above always throws
  }
}

// Registered at load rather than linked directly, so a build that compiles the
// software path without this module still links: it keeps its ordinary dtypes,
// and handing an FP8 dtype to Python is the error it was before.
const bool fp8_dtype_resolver_registered = [] {
  soft_lowp::RegisterFp8DtypeResolver(&Fp8DtypeObjectImpl);
  return true;
}();

// The FP8 keyword dtype for a call that opted in (python_dtype_for_fp8): the
// resolver above when it has one, the strict conversion otherwise.
py::object KeywordDtype(at::ScalarType dtype, bool python_dtype_for_fp8) {
  void* fp8 = python_dtype_for_fp8 ? soft_lowp::Fp8PythonDtype(dtype) : nullptr;
  return fp8 != nullptr ? py::reinterpret_borrow<py::object>(
                             static_cast<PyObject*>(fp8))
                        : OptionalDtypeToPython(dtype);
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
  const CallScope call;
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(self));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(self));
  return PythonToTensor(result);
}

at::Tensor& CallPythonOp_T_inplace(const char* func_name, at::Tensor& self) {
  const CallScope call;
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(self));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  func(TensorToPython(self));
  return self;
}

at::Tensor CallPythonOp_TT(const char* func_name, const at::Tensor& a, const at::Tensor& b) {
  const CallScope call;
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(a, b));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(a), TensorToPython(b));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_TTS(const char* func_name, const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  const CallScope call;
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(a, b));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(a), TensorToPython(b), "alpha"_a = ScalarToPython(alpha));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_TS(const char* func_name, const at::Tensor& self, const at::Scalar& other) {
  const CallScope call;
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfFirst(self));
  py::gil_scoped_acquire gil;
  auto func = cache.GetFunc(func_name);
  py::object result = func(TensorToPython(self), ScalarToPython(other));
  return PythonToTensor(result);
}

at::Tensor CallPythonOp_TIB(const char* func_name, const at::Tensor& self, int64_t dim, bool flag) {
  const CallScope call;
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
  const CallScope call;
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
  const CallScope call;
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
  const CallScope call;
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
  const CallScope call;
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
  const CallScope call;
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
  const CallScope call;
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
//
// `python_dtype_for_fp8` sends the four FP8 types to soft_lowp::Fp8PythonDtype
// instead of raising; no other dtype is affected, and nothing else turns it on.
py::dict BuildPyKwargs(const std::vector<PyKwarg>& kwargs, const char* func_name,
                       bool python_dtype_for_fp8) {
  py::dict d;
  for (const auto& kw : kwargs) {
    if (kw.is_none) {
      d[kw.name] = py::none();
    } else if (kw.is_dtype) {
      d[kw.name] = KeywordDtype(
          static_cast<at::ScalarType>(kw.value.toInt()), python_dtype_for_fp8);
    } else {
      d[kw.name] = IValueToPython(kw.value, func_name);
    }
  }
  return d;
}

} // namespace

at::Tensor CallPythonOp_GenericKw(const char* func_name,
                                  const std::vector<c10::IValue>& args,
                                  const std::vector<PyKwarg>& kwargs,
                                  bool python_dtype_for_fp8) {
  const CallScope call;
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfArgs(args));

  py::gil_scoped_acquire gil;
  auto func = cache.GetFuncBare(func_name);
  py::tuple py_args = BuildPyArgs(args, func_name);
  py::dict py_kwargs = BuildPyKwargs(kwargs, func_name, python_dtype_for_fp8);
  return PythonToTensor(func(*py_args, **py_kwargs));
}

at::Tensor CallPythonOp_Factory(const char* func_name,
                                const std::vector<c10::IValue>& args,
                                std::optional<at::ScalarType> dtype,
                                std::optional<at::Device> device) {
  const CallScope call;
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
  const CallScope call;
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
  const CallScope call;
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
    const std::vector<PyKwarg>& kwargs, int64_t n, bool python_dtype_for_fp8) {
  const CallScope call;
  auto& cache = GetCache();
  cache.EnsureInitialized();
  const c10::OptionalDeviceGuard device_guard(DeviceOfArgs(args));

  py::gil_scoped_acquire gil;
  auto func = cache.GetFuncBare(func_name);
  py::tuple py_args = BuildPyArgs(args, func_name);
  py::dict py_kwargs = BuildPyKwargs(kwargs, func_name, python_dtype_for_fp8);
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
  const CallScope call;
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
