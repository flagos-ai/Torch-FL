// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Copied into generated host translation units by the build-time compiler.

#include <ATen/Context.h>
#include <ATen/EmptyTensor.h>
#include <ATen/Operators.h>
#include <ATen/ops/empty_ops.h>
#include <ATen/ops/contiguous_ops.h>
#include <c10/core/GradMode.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <c10/util/SmallVector.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <sstream>

#include "host_abi.h"
namespace py = pybind11;

inline at::Tensor HostContiguous(const at::Tensor& tensor, at::MemoryFormat format) {
#if defined(FLAGOS_HYGON_NATIVE_HOST_FAST_ALLOC)
  if (!c10::GradMode::is_enabled() && tensor.device().is_privateuseone() &&
      tensor.layout() == at::kStrided && !tensor.has_names() &&
      !tensor.unsafeGetTensorImpl()->has_symbolic_sizes_strides() &&
      !tensor.key_set().has(c10::DispatchKey::Python) &&
      !tensor.key_set().has(c10::DispatchKey::Functionalize) &&
      format != at::MemoryFormat::Preserve && tensor.is_contiguous(format))
    return tensor;
#endif
  return at::_ops::contiguous::call(tensor, format);
}

inline at::Tensor HostEmpty(c10::SymIntArrayRef size, std::optional<at::ScalarType> dtype,
                            std::optional<at::Layout> layout, std::optional<at::Device> device,
                            std::optional<bool> pinned, std::optional<at::MemoryFormat> format) {
#if defined(FLAGOS_HYGON_NATIVE_HOST_FAST_ALLOC)
  if (device && device->is_privateuseone() && device->has_index() &&
      (!layout || *layout == at::kStrided) && !pinned.value_or(false)) {
    c10::impl::VirtualGuardImpl implementation(at::kPrivateUse1);
    c10::OptionalDeviceGuard guard;
    if (implementation.getDevice() != *device) guard.reset_device(*device);
    return at::detail::empty_generic(C10_AS_INTARRAYREF_SLOW(size),
        at::GetAllocator(at::kPrivateUse1), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())), format);
  }
#endif
  return at::_ops::empty_memory_format::call(size, dtype, layout, device, pinned, format);
}

struct HostIntList : c10::SmallVector<int64_t, 8> {
  using Base = c10::SmallVector<int64_t, 8>;
  HostIntList() = default;
  HostIntList(std::initializer_list<int64_t> value) : Base(value) {}
  HostIntList(at::IntArrayRef value) : Base(value.begin(), value.end()) {}
  HostIntList(const std::vector<int64_t>& value) : Base(value.begin(), value.end()) {}
  HostIntList(const c10::List<int64_t>& value) {
    for (auto item : value) push_back(item);
  }
  HostIntList copy() const { return *this; }
  int64_t get(int64_t i) const {
    TORCH_CHECK(i >= 0 && i < int64_t(size()), "list index out of range");
    return (*this)[i];
  }
  std::vector<int64_t> vec() const { return {begin(), end()}; }
  operator c10::List<int64_t>() const { return c10::List<int64_t>(vec()); }
};
inline int64_t Get(const HostIntList& values, int64_t i) {
  if (i < 0) i += values.size();
  return values.get(i);
}
inline c10::IValue Box(const HostIntList& values) { return c10::IValue(values.vec()); }
inline bool Truth(const HostIntList& values) { return !values.empty(); }
inline bool Contains(const HostIntList& values, int64_t x) {
  return std::find(values.begin(), values.end(), x) != values.end();
}
inline int64_t Sum(const HostIntList& values) {
  int64_t result = 0;
  for (auto x : values) result += x;
  return result;
}
inline bool operator==(const HostIntList& a, const HostIntList& b) {
  return at::IntArrayRef(a) == at::IntArrayRef(b);
}
inline bool operator!=(const HostIntList& a, const HostIntList& b) { return !(a == b); }
inline bool operator==(const HostIntList& a, const c10::List<int64_t>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b.get(i)) return false;
  return true;
}
inline bool operator==(const c10::List<int64_t>& a, const HostIntList& b) { return b == a; }
inline bool operator!=(const HostIntList& a, const c10::List<int64_t>& b) { return !(a == b); }
inline bool operator!=(const c10::List<int64_t>& a, const HostIntList& b) { return !(a == b); }
inline HostIntList operator+(HostIntList a, const HostIntList& b) {
  a.Base::append(b.begin(), b.end());
  return a;
}
inline HostIntList operator+(const HostIntList& a, const c10::List<int64_t>& b) {
  return a + HostIntList(b);
}
inline HostIntList operator+(const c10::List<int64_t>& a, const HostIntList& b) {
  return HostIntList(a) + b;
}
inline HostIntList Repeat(const HostIntList& a, int64_t n) {
  HostIntList result;
  for (int64_t i = 0; i < n; ++i) result.Base::append(a.begin(), a.end());
  return result;
}
inline HostIntList Slice(const HostIntList& values, std::optional<int64_t> start,
                         std::optional<int64_t> stop, int64_t step) {
  TORCH_CHECK(step != 0, "slice step cannot be zero");
  int64_t n = values.size();
  auto normalize = [&](int64_t x) {
    if (x < 0) x += n;
    return std::clamp(x, step < 0 ? int64_t(-1) : int64_t(0), step < 0 ? n - 1 : n);
  };
  int64_t first = start ? normalize(*start) : (step < 0 ? n - 1 : 0),
          last = stop ? normalize(*stop) : (step < 0 ? -1 : n);
  HostIntList result;
  for (int64_t i = first; step < 0 ? i > last : i < last; i += step)
    result.push_back(values.get(i));
  return result;
}
std::vector<std::string> dtype_names;
std::vector<c10::IValue> module_globals;
std::vector<int64_t> kernel_slots, pointwise_slots, foreign_slots;
using LaunchApi = flagos::native_host::Launch;
LaunchApi launch_api = nullptr;
using PointwiseApi = flagos::native_host::Pointwise;
PointwiseApi pointwise_api = nullptr;
flagos::native_host::PointwiseAt pointwise_at_api = nullptr;
flagos::native_host::PointwiseLaunch pointwise_launch_api = nullptr;
flagos::native_host::Foreign foreign_api = nullptr;
inline c10::IValue ForeignPacked(int64_t slot, std::initializer_list<c10::IValue> args,
                                 std::initializer_list<flagos::native_host::Keyword> kwargs) {
  TORCH_CHECK(foreign_api, "foreign host ABI is not bound");
  return foreign_api(slot, {c10::ArrayRef<c10::IValue>(args)},
                     {c10::ArrayRef<flagos::native_host::Keyword>(kwargs)});
}
inline c10::IValue Foreign(int64_t slot, const c10::List<c10::IValue>& args,
                           const c10::impl::GenericDict& kwargs) {
  c10::SmallVector<c10::IValue, 16> values;
  for (const auto& value : args) values.push_back(value);
  c10::SmallVector<flagos::native_host::Keyword, 8> keywords;
  for (const auto& item : kwargs) keywords.push_back({item.key().toStringRef(), item.value()});
  TORCH_CHECK(foreign_api, "foreign host ABI is not bound");
  return foreign_api(slot, {values}, {keywords});
}
inline void LaunchPointwisePacked(int64_t slot, int64_t rank,
                                  std::initializer_list<c10::IValue> args,
                                  std::initializer_list<at::Tensor> outputs) {
  TORCH_CHECK(pointwise_launch_api, "pointwise launch ABI is not bound");
  pointwise_launch_api(slot, rank, {c10::ArrayRef<c10::IValue>(args)},
                       {c10::ArrayRef<at::Tensor>(outputs)});
}
inline at::Tensor ApplyPointwiseAtPacked(int64_t slot, std::initializer_list<c10::IValue> args,
                                         std::initializer_list<flagos::native_host::Keyword> kwargs,
                                         int64_t index) {
  TORCH_CHECK(pointwise_at_api, "pointwise projection ABI is not bound");
  return pointwise_at_api(slot, {c10::ArrayRef<c10::IValue>(args)},
                          {c10::ArrayRef<flagos::native_host::Keyword>(kwargs)}, index);
}
inline at::Tensor ApplyPointwiseAt(int64_t slot, const c10::List<c10::IValue>& args,
                                   const c10::impl::GenericDict& kwargs, int64_t index) {
  c10::SmallVector<c10::IValue, 16> values;
  for (const auto& value : args) values.push_back(value);
  c10::SmallVector<flagos::native_host::Keyword, 8> keywords;
  for (const auto& item : kwargs) keywords.push_back({item.key().toStringRef(), item.value()});
  TORCH_CHECK(pointwise_at_api, "pointwise projection ABI is not bound");
  return pointwise_at_api(slot, {values}, {keywords}, index);
}
inline void LaunchPacked(int64_t slot, std::initializer_list<c10::IValue> args,
                         std::initializer_list<flagos::native_host::Keyword> kwargs,
                         const std::vector<int64_t>& grid, int64_t function) {
  TORCH_CHECK(launch_api, "native host ABI is not bound");
  launch_api(slot, {c10::ArrayRef<c10::IValue>(args)},
             {c10::ArrayRef<flagos::native_host::Keyword>(kwargs)}, grid, function);
}
inline void LaunchPacked(int64_t slot, std::initializer_list<c10::IValue> args,
                         std::initializer_list<flagos::native_host::Keyword> kwargs,
                         const c10::List<int64_t>& grid, int64_t function) {
  LaunchPacked(slot, args, kwargs, grid.vec(), function);
}
inline std::vector<at::Tensor> ApplyPointwisePacked(
    int64_t slot, std::initializer_list<c10::IValue> args,
    std::initializer_list<flagos::native_host::Keyword> kwargs) {
  TORCH_CHECK(pointwise_api, "pointwise ABI is not bound");
  return pointwise_api(slot, {c10::ArrayRef<c10::IValue>(args)},
                       {c10::ArrayRef<flagos::native_host::Keyword>(kwargs)});
}
inline std::vector<at::Tensor> ApplyPointwise(int64_t slot, const c10::List<c10::IValue>& args,
                                              const c10::impl::GenericDict& kwargs) {
  c10::SmallVector<c10::IValue, 16> values;
  for (const auto& value : args) values.push_back(value);
  c10::SmallVector<flagos::native_host::Keyword, 8> keywords;
  for (const auto& item : kwargs) keywords.push_back({item.key().toStringRef(), item.value()});
  TORCH_CHECK(pointwise_api, "pointwise ABI is not bound");
  return pointwise_api(slot, {values}, {keywords});
}
inline void Launch(int64_t slot, const c10::List<c10::IValue>& args,
                   const c10::impl::GenericDict& kwargs, const std::vector<int64_t>& grid,
                   int64_t function) {
  c10::SmallVector<c10::IValue, 16> values;
  for (const auto& value : args) values.push_back(value);
  c10::SmallVector<flagos::native_host::Keyword, 8> keywords;
  for (const auto& item : kwargs) keywords.push_back({item.key().toStringRef(), item.value()});
  TORCH_CHECK(launch_api, "native host ABI is not bound");
  launch_api(slot, {values}, {keywords}, grid, function);
}
inline void Launch(int64_t slot, const c10::List<c10::IValue>& args,
                   const c10::impl::GenericDict& kwargs, const c10::List<int64_t>& grid,
                   int64_t function) {
  Launch(slot, args, kwargs, grid.vec(), function);
}
inline c10::IValue MakeDescriptor(const at::Tensor& base, const c10::List<int64_t>& shape,
                                  const c10::List<int64_t>& strides,
                                  const c10::List<int64_t>& block, const std::string& padding) {
  return flagos::native_host::Descriptor::Create(base, shape, strides, block, padding);
}
inline c10::IValue TupleValue(const c10::IValue& value) {
  return c10::ivalue::Tuple::create(value.toList().vec());
}
inline std::vector<int64_t> BroadcastShapes(const c10::List<c10::List<int64_t>>& shapes) {
  size_t rank = 0;
  for (const auto& item : shapes) {
    const auto shape = static_cast<c10::List<int64_t>>(item);
    rank = std::max(rank, shape.size());
  }
  std::vector<int64_t> result(rank, 1);
  for (const auto& item : shapes) {
    const auto shape = static_cast<c10::List<int64_t>>(item);
    for (size_t i = 0; i < shape.size(); ++i) {
      auto& dst = result[rank - shape.size() + i];
      auto src = shape.get(i);
      TORCH_CHECK(dst == src || dst == 1 || src == 1, "incompatible broadcast shapes");
      if (dst == 1) dst = src;
    }
  }
  return result;
}
inline std::string DeviceName(c10::Device device) {
  return device.is_privateuseone() ? c10::get_privateuse1_backend()
                                   : c10::DeviceTypeName(device.type(), true);
}
inline at::TensorOptions TensorOptions(std::optional<int64_t> dtype,
                                       std::optional<c10::Device> device, bool requires_grad) {
  auto options = at::TensorOptions().requires_grad(requires_grad);
  if (dtype) options = options.dtype(static_cast<at::ScalarType>(*dtype));
  if (device) options = options.device(*device);
  return options;
}
template <class T>
int64_t PythonScalarDtype() {
  if constexpr (std::is_same_v<T, bool>)
    return int64_t(at::kBool);
  else if constexpr (std::is_integral_v<T>)
    return int64_t(at::kLong);
  else
    return int64_t(c10::get_default_dtype_as_scalartype());
}
template <class T>
at::Tensor TensorValue(const T& value, std::optional<int64_t> dtype,
                       std::optional<c10::Device> device, bool requires_grad) {
  if (!dtype) dtype = PythonScalarDtype<T>();
  return at::scalar_tensor(at::Scalar(value), TensorOptions(dtype, device, requires_grad));
}
template <class T>
at::Tensor TensorValue(const std::vector<T>& value, std::optional<int64_t> dtype,
                       std::optional<c10::Device> device, bool requires_grad) {
  if (!dtype)
    dtype =
        value.empty() ? int64_t(c10::get_default_dtype_as_scalartype()) : PythonScalarDtype<T>();
  return at::tensor(value, TensorOptions(dtype, device, requires_grad));
}
template <class T>
at::Tensor TensorValue(const c10::List<T>& value, std::optional<int64_t> dtype,
                       std::optional<c10::Device> device, bool requires_grad) {
  return TensorValue(value.vec(), dtype, device, requires_grad);
}
inline at::Tensor TensorValue(const HostIntList& value, std::optional<int64_t> dtype,
                              std::optional<c10::Device> device, bool requires_grad) {
  return TensorValue(value.vec(), dtype, device, requires_grad);
}
using HostGridMeta = flagos::native_host::GridMetadata;
inline int64_t Get(const HostGridMeta& meta, const std::string& key) {
  if (meta.constants->contains(key)) return meta.constants->at(key);
  auto integer = [](const c10::IValue& value) {
    return value.isBool() ? int64_t(value.toBool()) : value.toInt();
  };
  if (meta.kwargs.contains(key)) return integer(meta.kwargs.at(key));
  if (meta.names)
    for (size_t i = 0; i < meta.args.size(); ++i)
      if (meta.names->at(i) == key) return integer(meta.args.get(i));
  TORCH_CHECK(false, "missing grid metadata: ", key);
}
template <class T>
c10::IValue Box(const T& value) {
  return c10::IValue(value);
}
inline c10::IValue Box(std::nullopt_t) { return c10::IValue(); }
template <class T>
c10::IValue Box(const std::vector<T>& v) {
  return c10::IValue(v);
}
inline c10::List<c10::IValue> AnyList(std::initializer_list<c10::IValue> values) {
  c10::List<c10::IValue> out(c10::AnyType::get());
  for (const auto& v : values) out.push_back(v);
  return out;
}
inline c10::IValue Invoke(const std::string& name, std::vector<c10::IValue> stack,
                          const std::string& overload = "") {
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name.c_str(), overload.c_str());
  op.callBoxed(&stack);
  return stack.empty() ? c10::IValue() : stack.back();
}
template <class T>
T Get(const c10::List<T>& values, int64_t index) {
  if (index < 0) index += values.size();
  return values.get(index);
}
template <class T>
T Get(const std::vector<T>& values, int64_t index) {
  if (index < 0) index += values.size();
  return values.at(index);
}
template <class T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b) {
  auto result = a;
  result.insert(result.end(), b.begin(), b.end());
  return result;
}
template <class T>
c10::List<T> operator+(const c10::List<T>& a, const c10::List<T>& b) {
  auto result = a.copy();
  result.append(b);
  return result;
}
template <class T>
c10::List<T> Repeat(const c10::List<T>& value, int64_t count) {
  auto result = value.copy();
  result.clear();
  for (int64_t i = 0; i < count; ++i) result.append(value);
  return result;
}
inline int64_t RangeLength(int64_t start, int64_t stop, int64_t step) {
  TORCH_CHECK(step != 0, "range step cannot be zero");
  return step > 0 ? (stop <= start ? 0 : 1 + (stop - 1 - start) / step)
                  : (stop >= start ? 0 : 1 + (start - 1 - stop) / (-step));
}
template <class T>
c10::List<T> SetItem(c10::List<T> values, int64_t index, const T& value) {
  if (index < 0) index += values.size();
  values.set(index, value);
  return values;
}
template <class T>
T Pop(c10::List<T> values, int64_t index) {
  if (index < 0) index += values.size();
  auto result = values.get(index);
  values.erase(values.begin() + index);
  return result;
}
template <class T>
c10::List<T> Slice(const c10::List<T>& values, std::optional<int64_t> start,
                   std::optional<int64_t> stop, int64_t step) {
  TORCH_CHECK(step != 0, "slice step cannot be zero");
  int64_t n = values.size();
  auto normalize = [&](int64_t x) {
    if (x < 0) x += n;
    return std::clamp(x, step < 0 ? int64_t(-1) : int64_t(0), step < 0 ? n - 1 : n);
  };
  int64_t first = start ? normalize(*start) : (step < 0 ? n - 1 : 0);
  int64_t last = stop ? normalize(*stop) : (step < 0 ? -1 : n);
  auto result = values.copy();
  result.clear();
  for (int64_t i = first; step < 0 ? i > last : i < last; i += step)
    result.push_back(values.get(i));
  return result;
}
template <class T>
bool Contains(const c10::List<T>& values, const T& value) {
  for (size_t i = 0; i < values.size(); ++i)
    if (values.get(i) == value) return true;
  return false;
}
template <class T>
T Sum(const c10::List<T>& values) {
  T result{};
  for (size_t i = 0; i < values.size(); ++i) result += values.get(i);
  return result;
}
template <class T>
bool Truth(const c10::List<T>& value) {
  return !value.empty();
}
template <class T>
bool Truth(const std::vector<T>& value) {
  return !value.empty();
}
template <class T>
bool Truth(const T& value) {
  return bool(value);
}
inline std::vector<std::string> Split(const std::string& value, const std::string& sep,
                                      int64_t maxsplit) {
  TORCH_CHECK(!sep.empty(), "empty separator");
  std::vector<std::string> result;
  size_t offset = 0;
  while (maxsplit < 0 || int64_t(result.size()) < maxsplit) {
    auto at = value.find(sep, offset);
    if (at == std::string::npos) break;
    result.push_back(value.substr(offset, at - offset));
    offset = at + sep.size();
  }
  result.push_back(value.substr(offset));
  return result;
}
template <class T>
T Get(const c10::Dict<std::string, T>& values, const std::string& key) {
  return values.at(key);
}
inline int64_t FloorDiv(int64_t a, int64_t b) {
  TORCH_CHECK(b != 0, "division by zero");
  return a / b - ((a % b) != 0 && ((a < 0) != (b < 0)));
}
inline int64_t Remainder(int64_t a, int64_t b) { return a - FloorDiv(a, b) * b; }
inline std::string Format(std::string text, std::vector<c10::IValue> values) {
  size_t start = 0;
  for (auto& value : values) {
    auto p = text.find("{}", start);
    if (p == std::string::npos) break;
    std::ostringstream s;
    s << value;
    auto part = s.str();
    text.replace(p, 2, part);
    start = p + part.size();
  }
  return text;
}
