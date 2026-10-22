// Copyright (c) 2026, BAAI. All rights reserved.
//
// Python op caller: bridge from C++ stub layer to FlagGems Python wrapper ops
// via pybind11 embedded interpreter.

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>
#include <c10/util/ArrayRef.h>

#include <utility>

namespace at::native::flagos {

// Call a FlagGems Python op by function name.
// All variants acquire the GIL internally and look up the function
// from the cached flag_gems.ops module.

// (Tensor) -> Tensor
at::Tensor CallPythonOp_T(const char* func_name, const at::Tensor& self);

// (Tensor) -> Tensor  [inplace: returns self]
at::Tensor& CallPythonOp_T_inplace(const char* func_name, at::Tensor& self);

// (Tensor, Tensor) -> Tensor
at::Tensor CallPythonOp_TT(const char* func_name, const at::Tensor& a, const at::Tensor& b);

// (Tensor, Tensor, Scalar) -> Tensor
at::Tensor CallPythonOp_TTS(const char* func_name, const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha);

// (Tensor, Scalar) -> Tensor
at::Tensor CallPythonOp_TS(const char* func_name, const at::Tensor& self, const at::Scalar& other);

// (Tensor, int64_t, bool) -> Tensor
at::Tensor CallPythonOp_TIB(const char* func_name, const at::Tensor& self, int64_t dim, bool flag);

// (Tensor, OptionalIntArrayRef, bool, optional<ScalarType>) -> Tensor
at::Tensor CallPythonOp_TOIB(const char* func_name, const at::Tensor& self,
                              at::OptionalIntArrayRef dim, bool keepdim,
                              std::optional<at::ScalarType> dtype);

// (Tensor, Tensor, Tensor) -> Tensor
at::Tensor CallPythonOp_TTT(const char* func_name, const at::Tensor& a, const at::Tensor& b, const at::Tensor& c);

// (Tensor, optional<ScalarType>) -> Tensor  [keyword: dtype=...]
at::Tensor CallPythonOp_TD(const char* func_name, const at::Tensor& self,
                            std::optional<at::ScalarType> dtype);

// (TensorList, int64_t) -> Tensor   [e.g. cat(tensors, dim)]
at::Tensor CallPythonOp_ListI(const char* func_name,
                              const at::ITensorListRef& tensors, int64_t dim);

// (Tensor, Tensor, int64_t, bool, bool) -> Tensor
// [embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)]
at::Tensor CallPythonOp_Embedding(const char* func_name, const at::Tensor& weight,
                                  const at::Tensor& indices, int64_t padding_idx,
                                  bool scale_grad_by_freq, bool sparse);

// Generic variadic caller using Python *args/**kwargs.
// For ops with complex signatures not covered above.
// Arguments are passed as a vector of IValues.
at::Tensor CallPythonOp_Generic(const char* func_name, const std::vector<c10::IValue>& args);

// A keyword argument to forward to the FlagGems Python op by name. Used for the
// aten trailing args that gems declares keyword-only (dtype/alpha/correction/...).
// `is_dtype` flags a ScalarType payload: an IValue stores ScalarType as a plain
// int, so the caller can't tell it apart from an ordinary int at runtime -- the
// codegen sets this per-arg from the schema so the value is converted to a
// torch.dtype (or None) instead of an int. `is_none` carries an absent optional
// (e.g. dtype=None / correction=None) since IValue can't distinguish "missing".
struct PyKwarg {
  const char* name;
  c10::IValue value;
  bool is_dtype = false;
  bool is_none = false;
};

// Like CallPythonOp_Generic, but also forwards `kwargs` by name. Positional
// `args` are the aten args gems takes positionally; `kwargs` are the trailing
// aten args gems declares keyword-only. Covers functional_pure/inplace via the
// single-tensor return.
at::Tensor CallPythonOp_GenericKw(const char* func_name,
                                  const std::vector<c10::IValue>& args,
                                  const std::vector<PyKwarg>& kwargs);

// Like CallPythonOp_GenericTuple, but with keyword args (e.g. sort.stable,
// var_mean.correction). Returns the N tensors in order.
std::vector<at::Tensor> CallPythonOp_GenericKwTuple(
    const char* func_name, const std::vector<c10::IValue>& args,
    const std::vector<PyKwarg>& kwargs, int64_t n);

// Factory caller (arange/eye/full/ones/zeros/linspace/...). `args` are the
// shape/scalar positionals; the tensor-options are injected as kwargs:
// device=flagos (so gems' internal torch.empty hits OUR allocator and produces
// a PrivateUse1 tensor -- no CUDA round-trip, no recursion), layout=strided
// (gems eye/randperm validate layout, None is rejected), pin_memory=None, and
// dtype forwarded from the aten call (nullopt -> None, else the ScalarType).
//
// `device` is the aten call's device argument, forwarded so the result lands on
// the requested index. Following aten factory semantics, nullopt (or a device
// with no index) means "the current device". The injected kwarg carries the
// resolved index and the Python call runs under a DeviceGuard on it, so gems'
// internal torch.empty and its Triton launch agree -- a mismatch allocates on
// one device and launches on another, which faults the GPU rather than erroring.
at::Tensor CallPythonOp_Factory(const char* func_name,
                                const std::vector<c10::IValue>& args,
                                std::optional<at::ScalarType> dtype,
                                std::optional<at::Device> device = std::nullopt);

// Like-factory caller (zeros_like/ones_like/full_like/...). `args` are the
// non-TensorOptions positionals -- the input tensor `self` (whose shape/device
// gems reads via torch.empty_like) plus any value positional (full_like's
// fill_value). The tensor-options are injected as kwargs: device=flagos (so
// gems' internal empty_like stays on PrivateUse1), layout=strided,
// memory_format=None, pin_memory=None, and dtype forwarded (nullopt -> None,
// meaning "same as self"). Distinct from CallPythonOp_Factory whose first arg
// is a shape array; here it's the source tensor.
//
// `device` behaves as in CallPythonOp_Factory: forwarded from the aten call,
// nullopt/index-less means the current device.
at::Tensor CallPythonOp_LikeFactory(const char* func_name,
                                    const std::vector<c10::IValue>& args,
                                    std::optional<at::ScalarType> dtype,
                                    std::optional<at::Device> device = std::nullopt);

// Random in-place caller (uniform_/exponential_/bernoulli_.float). `args` are
// the aten positionals with the trailing `Generator?` arg dropped (self, plus
// scalar params like from/to/lambd/p). We call gems with generator=None; gems'
// philox_backend_seed_offset then reads torch_device_fn.default_generators
// (the per-device CUDA generators the compat shim installs -- see
// torch_fl/accelerator/cuda/_cuda_compat.py), the SAME source the generator-less
// rng ops (rand/randn/multinomial/...) use. We do NOT inject an explicit
// generator object here: that raced with triton's first-compile of
// exponential_/uniform_ on a cold kernel cache ("Unable to cast <int> to '?'"),
// and keeping generator=None routes all rng ops through one seedable source.
at::Tensor CallPythonOp_RandomInplace(const char* func_name,
                                      const std::vector<c10::IValue>& args);

// Fetch the vendor compat shim's per-device CUDA generator for CUDA-compatible
// kernels. The returned handle is cached while manual_seed mutates it in place.
at::Generator GetFlagosDefaultCudaGenerator(int64_t device_index);

// Fetch the vendor compat shim's per-device philox generator state. This is
// the SAME object FlagGems reads via philox_backend_seed_offset, so native GCU
// kernels and FlagGems advance one seed/offset stream under torch.manual_seed.
std::pair<uint64_t, uint64_t> GetFlagosPhiloxState(
    int64_t device_index, uint64_t increment);

// Like CallPythonOp_Generic, but the Python op returns a tuple/list of N tensors
// (e.g. sort -> (values, indices), var_mean -> (var, mean)). Returns the N
// tensors in order. Used by the codegen tuple_return kernels.
std::vector<at::Tensor> CallPythonOp_GenericTuple(
    const char* func_name, const std::vector<c10::IValue>& args, int64_t n);

} // namespace at::native::flagos
