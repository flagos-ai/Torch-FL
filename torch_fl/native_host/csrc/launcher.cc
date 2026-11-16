// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// The launch ABI is data. No signature-specific C/C++ is generated at runtime.
#include <ATen/EmptyTensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <Python.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <c10/util/SmallVector.h>
#include <dlfcn.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "host_abi.h"
#include "backend_api.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

namespace backend = flagos::native_host::backend;
const backend::Api* backend_api = nullptr;
using backend::CheckedProduct;

const backend::Api& Backend() { return *backend_api; }

bool IsCapturing(uint64_t stream) {
  if (!Backend().is_capturing)
    throw std::runtime_error("backend cannot query stream capture state");
  return Backend().is_capturing(stream);
}

using CurrentStream = void* (*)(int);
CurrentStream current_stream = nullptr;
std::vector<std::string> dtype_names;

py::object& descriptor_factory = *new py::object();
constexpr const char* descriptor_tag = "__flagos_tensor_descriptor_v1__";

bool IsDescriptor(const c10::IValue& value) { return flagos::native_host::Descriptor::Is(value); }

c10::IValue FromPython(py::handle value) {
  if (descriptor_factory && !descriptor_factory.is_none() &&
      py::isinstance(value, descriptor_factory)) {
    c10::impl::GenericDict result(c10::StringType::get(), c10::AnyType::get());
    result.insert(descriptor_tag, true);
    result.insert("base", torch::jit::toIValue(value.attr("base"), c10::AnyType::get()));
    result.insert("padding", py::cast<std::string>(value.attr("padding")));
    for (const char* name : {"shape", "strides", "block_shape"})
      result.insert(name, c10::IValue(py::cast<std::vector<int64_t>>(value.attr(name))));
    return result;
  }
  return torch::jit::toIValue(value, c10::AnyType::get());
}

class PythonArguments {
 public:
  py::object Convert(const c10::IValue& value) {
    const bool descriptor = IsDescriptor(value);
    if (!descriptor && !value.isTuple() && !value.isList() && !value.isGenericDict())
      return torch::jit::toPyObject(value);
    const auto* identity = value.internalToPointer();
    auto found = objects_.find(identity);
    if (found != objects_.end()) return found->second;
    py::object result;
    if (descriptor) {
      flagos::native_host::Descriptor data(value);
      result = descriptor_factory(Convert(data.at("base")), Convert(data.at("shape")),
                                  Convert(data.at("strides")), Convert(data.at("block_shape")),
                                  Convert(data.at("padding")));
    } else if (value.isTuple()) {
      const auto& values = value.toTupleRef().elements();
      py::tuple items(values.size());
      Remember(value, items);
      for (size_t i = 0; i < values.size(); ++i) {
        auto item = Convert(values[i]);
        // The memo owns a second reference while this new tuple is populated.
        // Each previously empty slot receives exactly one owned reference.
        PyTuple_SET_ITEM(items.ptr(), i, item.release().ptr());
      }
      return items;
    } else if (value.isList()) {
      const auto values = value.toList();
      py::list items(values.size());
      Remember(value, items);
      for (size_t i = 0; i < values.size(); ++i) items[i] = Convert(values.get(i));
      return items;
    } else {
      py::dict items;
      Remember(value, items);
      for (const auto& item : value.toGenericDict())
        items[Convert(item.key())] = Convert(item.value());
      return items;
    }
    Remember(value, result);
    return result;
  }
  c10::IValue Restore(py::handle object) {
    const auto found = values_.find(object.ptr());
    if (found != values_.end()) return found->second;
    auto value = FromPython(object);
    if (value.isTuple()) {
      std::vector<c10::IValue> children;
      for (auto child : py::reinterpret_borrow<py::sequence>(object))
        children.push_back(Restore(child));
      value = c10::ivalue::Tuple::create(std::move(children));
    }
    if (IsDescriptor(value) || value.isTuple() || value.isList() || value.isGenericDict())
      Remember(value, py::reinterpret_borrow<py::object>(object));
    return value;
  }
  void Sync() {
    // Restore may append newly assigned containers. Process them too, keeping
    // shared list/dictionary identities across input arguments and the result.
    for (size_t i = 0; i < bindings_.size(); ++i) {
      const auto item = bindings_[i];
      if (IsDescriptor(item.first)) {
        flagos::native_host::Descriptor data(item.first);
        for (const char* field : {"base", "shape", "strides", "block_shape", "padding"})
          data.insert_or_assign(field, Restore(item.second.attr(field)));
      } else if (item.first.isList()) {
        auto target = item.first.toList();
        std::vector<c10::IValue> children;
        for (auto child : py::reinterpret_borrow<py::sequence>(item.second))
          children.push_back(Restore(child));
        target.clear();
        for (auto& child : children) target.push_back(std::move(child));
      } else if (item.first.isGenericDict()) {
        auto target = item.first.toGenericDict();
        std::vector<std::pair<c10::IValue, c10::IValue>> children;
        for (auto child : py::reinterpret_borrow<py::dict>(item.second))
          children.emplace_back(Restore(child.first), Restore(child.second));
        target.clear();
        for (auto& child : children) target.insert(std::move(child.first), std::move(child.second));
      }
    }
  }

 private:
  void Remember(const c10::IValue& value, py::object object) {
    objects_.emplace(value.internalToPointer(), object);
    values_.emplace(object.ptr(), value);
    bindings_.emplace_back(value, std::move(object));
  }
  std::unordered_map<const void*, py::object> objects_;
  std::unordered_map<PyObject*, c10::IValue> values_;
  std::vector<std::pair<c10::IValue, py::object>> bindings_;
};

py::object ToPython(const c10::IValue& value) {
  PythonArguments context;
  return context.Convert(value);
}

using NativeHook = void (*)(const c10::Dict<std::string, c10::IValue>&);
struct HookRegistration {
  NativeHook entry;
  flagos::native_host::MetadataHook packed;
  py::object owner;
};
std::unordered_map<PyObject*, HookRegistration>& NativeHooks() {
  static auto* hooks = new std::unordered_map<PyObject*, HookRegistration>();
  return *hooks;
}

uint64_t Stream(int device) {
  if (!current_stream) throw std::runtime_error("FlagOS stream provider is not initialized");
  return reinterpret_cast<uint64_t>(current_stream(device));
}

class CurrentDeviceScope {
 public:
  explicit CurrentDeviceScope(c10::Device device) {
    c10::impl::VirtualGuardImpl implementation(device.type());
    if (implementation.getDevice() != device) guard_.reset_device(device);
  }

 private:
  c10::OptionalDeviceGuard guard_;
};

struct HostGridMeta {
  const c10::Dict<std::string, int64_t>* constants;
  const std::vector<std::string>* names;
  const c10::List<c10::IValue>* args;
  const c10::Dict<std::string, c10::IValue>* kwargs;
};

struct StridedBuffer;
PyTypeObject* buffer_type = nullptr;
uint64_t Pointer(py::handle value);

// Native storage for the wheel's metadata-only StridedBuffer abstraction.
// It retains the current base object; it never owns a cached output buffer.
struct StridedBuffer {
  py::object base, dtype, device;
  py::tuple shape, strides;
  std::vector<int64_t> sizes, steps;
  int64_t offset, elements = 1;
  uint64_t pointer;
  StridedBuffer(py::object base_value, py::object shape_value, py::object stride_value,
                py::object dtype_value, int64_t offset_value)
      : base(base_value), offset(offset_value) {
    const bool tensor = THPVariable_CheckExact(base.ptr());
    dtype = dtype_value.is_none() ? py::object(base.attr("dtype")) : dtype_value;
    shape = py::tuple(shape_value.is_none() ? py::object(base.attr("shape")) : shape_value);
    strides = py::tuple(stride_value.is_none() ? base.attr("stride")() : stride_value);
    sizes = py::cast<std::vector<int64_t>>(shape);
    steps = py::cast<std::vector<int64_t>>(strides);
    for (auto size : sizes) elements *= size;
    device = base.attr("device");
    pointer = tensor ? reinterpret_cast<uint64_t>(THPVariable_Unpack(base.ptr()).const_data_ptr())
                     : Pointer(base);
    pointer += offset * py::cast<int64_t>(dtype.attr("itemsize"));
  }
  py::object Clone() const {
    return py::cast(StridedBuffer(base.attr("clone")(), shape, strides, dtype, offset));
  }
  StridedBuffer& Copy(py::object source) {
    if (Py_TYPE(source.ptr()) != buffer_type) {
      auto wrapped = py::cast(StridedBuffer(source, py::none(), py::none(), py::none(), 0));
      return Copy(wrapped);
    }
    const auto& other = py::cast<const StridedBuffer&>(source);
    base.attr("copy_")(other.base);
    shape = other.shape;
    strides = other.strides;
    sizes = other.sizes;
    steps = other.steps;
    dtype = other.dtype;
    device = other.device;
    offset = other.offset;
    elements = other.elements;
    return *this;
  }
};

uint64_t Pointer(py::handle value) {
  if (value.is_none()) return 0;
  if (PyLong_Check(value.ptr())) return py::cast<uint64_t>(value);
  if (Py_TYPE(value.ptr()) == buffer_type) {
    const auto& buffer = py::cast<const StridedBuffer&>(value);
    if (THPVariable_CheckExact(buffer.base.ptr()) &&
        THPVariable_Unpack(buffer.base.ptr()).device().is_cpu())
      throw std::invalid_argument("CPU tensor cannot be passed to a FlagGems device kernel");
    return buffer.pointer;
  }
  if (THPVariable_CheckExact(value.ptr())) {
    const auto& tensor = THPVariable_Unpack(value.ptr());
    if (tensor.device().is_cpu())
      throw std::invalid_argument("CPU tensor cannot be passed to a FlagGems device kernel");
    return reinterpret_cast<uint64_t>(tensor.const_data_ptr());
  }
  if (py::hasattr(value, "device") &&
      py::cast<std::string>(value.attr("device").attr("type")) == "cpu")
    throw std::invalid_argument("CPU tensor cannot be passed to a FlagGems device kernel");
  return py::cast<uint64_t>(value.attr("data_ptr")());
}

using Slot = backend::Slot;

struct Argument {
  enum class Kind {
    Pointer, Descriptor, Float64, Float32, Bool, Int8, Int16, Int32, Int64, Other
  };
  std::vector<size_t> path;
  std::string type;
  std::shared_ptr<const backend::Descriptor> descriptor;
  Kind kind = Kind::Other;
};

class Launcher {
 public:
  Launcher(py::object source, py::object metadata)
      : metadata_(metadata), kernel_(Backend().kernel(metadata.ptr())) {
    slot_capacity_ = kernel_->scratch_slots;
    const auto signatures = py::cast<py::dict>(source.attr("signature"));
    py::object descriptors = py::getattr(metadata, "tensordesc_meta", py::none());
    size_t index = 0, descriptor = 0;
    for (auto item : signatures) {
      Add(item.second, {index++}, descriptors, descriptor);
    }
    if (!descriptors.is_none() && descriptor != py::len(descriptors))
      throw std::invalid_argument("Triton descriptor metadata arity mismatch");
    num_args_ = index;

  }

  // Matches CompiledKernel.run; callers pass metadata and optional profiler hooks.
  void Call(unsigned x, unsigned y, unsigned z, uint64_t stream, uint64_t function,
            py::object packed, py::object launch_metadata, py::object enter, py::object leave,
            py::args args) const {
    std::vector<py::handle> values;
    values.reserve(args.size());
    for (auto value : args) values.push_back(value);
    CallValues(x, y, z, stream, function, values, launch_metadata, enter, leave);
  }

  void CallValues(unsigned x, unsigned y, unsigned z, uint64_t stream, uint64_t function,
                  const std::vector<py::handle>& args, py::object launch_metadata = py::none(),
                  py::object enter = py::none(), py::object leave = py::none()) const {
    if (args.size() != num_args_)
      throw std::invalid_argument("Triton launch argument arity mismatch");
    if (!enter.is_none()) enter(launch_metadata);
    std::vector<Slot> slots;
    slots.reserve(arguments_.size() * 22 + kernel_->scratch_slots);
    for (const auto& argument : arguments_) {
      PyObject* value = args[argument.path.front()].ptr();
      for (size_t depth = 1; depth < argument.path.size(); ++depth) {
        size_t index = argument.path[depth];
        if (PyTuple_Check(value))
          value = PyTuple_GetItem(value, index);
        else if (PyList_Check(value))
          value = PyList_GetItem(value, index);
        else
          throw std::invalid_argument("nested Triton argument must be a tuple or list");
        if (!value) throw py::error_already_set();
      }
      Pack(argument, py::handle(value), slots);
    }

    const uint64_t grid = CheckedProduct(CheckedProduct(x, y), z);
    const auto programs = CheckedProduct(grid, kernel_->scratch_multiplier);
    const auto global_bytes = CheckedProduct(programs, kernel_->global.bytes);
    const auto profile_bytes = CheckedProduct(programs, kernel_->profile.bytes);
    py::object global = Scratch(global_bytes, kernel_->global.alignment);
    py::object profile = Scratch(profile_bytes, kernel_->profile.alignment);
    const uint64_t global_ptr = Pointer(global), profile_ptr = Pointer(profile);
    PackScratch(global_ptr, profile_ptr, slots);
    std::vector<void*> pointers;
    pointers.reserve(slots.size());
    for (auto& slot : slots) pointers.push_back(slot.bytes.data());
    if (grid)
      kernel_->Enqueue({{x, y, z}, stream, function, pointers.data(), global_ptr,
                        global_bytes, true});
    if (!leave.is_none()) leave(launch_metadata);
  }

  py::list Abi() const {
    py::list result;
    for (const auto& a : arguments_) result.append(py::make_tuple(a.path, a.type));
    return result;
  }

  bool NativeSafe() const {
    if (kernel_->global.bytes || kernel_->profile.bytes) return false;
    for (const auto& arg : arguments_)
      if (arg.type == "fp16" || arg.type == "bf16") return false;
    return true;
  }
  bool RuntimeArgument(size_t position) const {
    for (const auto& arg : arguments_)
      if (arg.path.front() == position) return true;
    return false;
  }
  template <class GetValue>
  void CallNative(size_t argument_count, const GetValue& get_value,
                  const std::vector<int64_t>& grid, int device, uint64_t function) const {
    TORCH_CHECK(argument_count == num_args_, "native launch argument arity mismatch");
    c10::SmallVector<Slot, 32> slots;
    slots.reserve(slot_capacity_);
    for (const auto& arg : arguments_) {
      // The caller owns this value through packing. A copied legacy argument
      // also lives here until this iteration ends; nested leaves are borrowed.
      const auto& root = get_value(arg.path.front());
      const auto* leaf = &root;
      for (size_t depth = 1; depth < arg.path.size(); ++depth)
        leaf = leaf->isTuple() ? &leaf->toTupleRef().elements().at(arg.path[depth])
                              : &leaf->toListRef().at(arg.path[depth]);
      const auto& value = *leaf;
      if (arg.kind == Argument::Kind::Descriptor) {
        PackDescriptorNative(arg, value, slots);
        continue;
      }
      slots.emplace_back();
      auto& slot = slots.back();
      switch (arg.kind) {
      case Argument::Kind::Pointer: {
        uint64_t pointer = 0;
        if (value.isTensor()) {
          const auto& tensor = value.toTensor();
          TORCH_CHECK(!tensor.device().is_cpu(),
                      "CPU tensor cannot be passed to a FlagGems device kernel");
          pointer = reinterpret_cast<uint64_t>(tensor.const_data_ptr());
        } else if (!value.isNone())
          pointer = static_cast<uint64_t>(value.toInt());
        slot.Set(pointer);
        break;
      }
      case Argument::Kind::Float64:
        slot.Set(value.toDouble());
        break;
      case Argument::Kind::Float32:
        slot.Set(static_cast<float>(value.toDouble()));
        break;
      case Argument::Kind::Bool:
        slot.Set(static_cast<uint8_t>(value.toBool()));
        break;
      case Argument::Kind::Int8:
        slot.Set(static_cast<uint8_t>(value.toInt()));
        break;
      case Argument::Kind::Int16:
        slot.Set(static_cast<uint16_t>(value.toInt()));
        break;
      case Argument::Kind::Int32:
        slot.Set(static_cast<uint32_t>(value.toInt()));
        break;
      case Argument::Kind::Int64:
        slot.Set(static_cast<uint64_t>(value.toInt()));
        break;
      default:
        throw std::invalid_argument("unsupported native scalar ABI");
      }
    }
    PackScratch(0, 0, slots);
    c10::SmallVector<void*, 32> pointers;
    for (auto& slot : slots) pointers.push_back(slot.bytes.data());
    TORCH_CHECK(!grid.empty() && grid.size() <= 3, "invalid launch grid");
    std::array<uint64_t, 3> dims{1, 1, 1};
    for (size_t i = 0; i < grid.size(); ++i) {
      TORCH_CHECK(grid[i] >= 0, "negative launch dimension");
      dims[i] = grid[i];
    }
    if (!CheckedProduct(CheckedProduct(dims[0], dims[1]), dims[2])) return;
    kernel_->Enqueue({dims, Stream(device), function, pointers.data()});
  }

 private:
  template <class Slots>
  void PackScratch(uint64_t global, uint64_t profile, Slots& slots) const {
    if (!kernel_->scratch_slots) return;
    const auto start = slots.size();
    slots.resize(start + kernel_->scratch_slots);
    kernel_->PackScratch(global, profile, slots.data() + start);
  }

  static py::object Scratch(uint64_t size, uint64_t alignment) {
    if (!size) return py::none();
    if (alignment > Backend().scratch_alignment)
      throw std::invalid_argument("scratch alignment exceeds allocator guarantee");
    auto torch = py::module_::import("torch");
    return torch.attr("empty")(py::make_tuple(size), "dtype"_a = torch.attr("uint8"),
                               "device"_a = "flagos");
  }
  void Add(py::handle type, std::vector<size_t> path, py::object descriptors, size_t& descriptor) {
    if (py::isinstance<py::tuple>(type)) {
      size_t i = 0;
      for (auto child : py::reinterpret_borrow<py::tuple>(type)) {
        auto next = path;
        next.push_back(i++);
        Add(child, std::move(next), descriptors, descriptor);
      }
      return;
    }
    const std::string name = py::cast<std::string>(type);
    if (name == "constexpr") return;
    Argument argument{std::move(path), name, nullptr};
    if (name.rfind("tensordesc", 0) == 0) {
      argument.kind = Argument::Kind::Descriptor;
      auto metadata = descriptors.is_none() ? py::none() : py::object(descriptors[py::int_(descriptor)]);
      ++descriptor;
      if (!Backend().descriptor)
        throw std::invalid_argument("backend does not support tensor descriptors");
      argument.descriptor = Backend().descriptor(metadata.ptr());
    } else if (!name.empty() && name[0] == '*')
      argument.kind = Argument::Kind::Pointer;
    else if (name == "fp64") argument.kind = Argument::Kind::Float64;
    else if (name == "fp32" || name == "f32") argument.kind = Argument::Kind::Float32;
    else if (name == "i1" || name == "u1") argument.kind = Argument::Kind::Bool;
    else if (name == "i8" || name == "u8") argument.kind = Argument::Kind::Int8;
    else if (name == "i16" || name == "u16") argument.kind = Argument::Kind::Int16;
    else if (name == "i32" || name == "u32") argument.kind = Argument::Kind::Int32;
    else if (name == "i64" || name == "u64") argument.kind = Argument::Kind::Int64;
    slot_capacity_ += argument.kind == Argument::Kind::Descriptor
                          ? argument.descriptor->Capacity()
                          : 1;
    arguments_.push_back(std::move(argument));
  }
  static void Pack(const Argument& a, py::handle value, std::vector<Slot>& slots) {
    if (a.type.rfind("tensordesc", 0) == 0) {
      Descriptor(a, value, slots);
      return;
    }
    slots.emplace_back();
    Slot& slot = slots.back();
    if (a.type[0] == '*') {
      slot.Set(Pointer(value));
      return;
    }
    if (a.type == "fp64") {
      slot.Set(py::cast<double>(value));
      return;
    }
    if (a.type == "fp32" || a.type == "f32") {
      slot.Set(py::cast<float>(value));
      return;
    }
    if (a.type == "fp16") {
      if (PyFloat_Pack2(py::cast<double>(value), reinterpret_cast<char*>(slot.bytes.data()), 1) < 0)
        throw py::error_already_set();
      return;
    }
    if (a.type == "bf16") {
      const float f = py::cast<float>(value);
      uint32_t bits;
      std::memcpy(&bits, &f, sizeof(bits));
      slot.Set(static_cast<uint16_t>(bits >> 16));
      return;
    }
    if (a.type == "i1" || a.type == "u1") {
      slot.Set(static_cast<uint8_t>(py::cast<bool>(value)));
      return;
    }
    if (a.type == "i8") {
      slot.Set(py::cast<int8_t>(value));
      return;
    }
    if (a.type == "i16") {
      slot.Set(py::cast<int16_t>(value));
      return;
    }
    if (a.type == "i32") {
      slot.Set(py::cast<int32_t>(value));
      return;
    }
    if (a.type == "i64") {
      slot.Set(py::cast<int64_t>(value));
      return;
    }
    if (a.type == "u8") {
      slot.Set(py::cast<uint8_t>(value));
      return;
    }
    if (a.type == "u16") {
      slot.Set(py::cast<uint16_t>(value));
      return;
    }
    if (a.type == "u32") {
      slot.Set(py::cast<uint32_t>(value));
      return;
    }
    if (a.type == "u64") {
      slot.Set(py::cast<uint64_t>(value));
      return;
    }
    throw std::invalid_argument("unsupported Triton parameter ABI: " + a.type);
  }

  template <class Slots>
  static void PackDescriptor(const Argument& argument, uint64_t pointer,
                             c10::ArrayRef<uint64_t> shape,
                             c10::ArrayRef<uint64_t> strides, bool nan, Slots& slots) {
    if (shape.empty() || shape.size() != strides.size())
      throw std::invalid_argument("invalid tensor descriptor dimensions");
    const size_t start = slots.size(), capacity = argument.descriptor->Capacity();
    slots.resize(start + capacity);
    auto count = argument.descriptor->Pack(
        {pointer, shape.size(), shape.data(), strides.data(), nan}, slots.data() + start);
    if (count > capacity) throw std::runtime_error("backend descriptor exceeds argument capacity");
    slots.resize(start + count);
  }

  static void PackDescriptorNative(const Argument& argument, const c10::IValue& value,
                                   c10::SmallVectorImpl<Slot>& slots) {
    TORCH_CHECK(IsDescriptor(value), "native tensor descriptor was not admitted");
    const flagos::native_host::Descriptor data(value);
    const auto tensor = data.at("base").toTensor();
    TORCH_CHECK(!tensor.device().is_cpu(), "CPU descriptor cannot launch on the device");
    const auto dimensions = data.at("shape").toIntList();
    const auto steps = data.at("strides").toIntList();
    c10::SmallVector<uint64_t, 8> shape(dimensions.begin(), dimensions.end());
    c10::SmallVector<uint64_t, 8> strides(steps.begin(), steps.end());
    PackDescriptor(argument, reinterpret_cast<uint64_t>(tensor.const_data_ptr()), shape, strides,
                   data.at("padding").toStringRef() == "nan", slots);
  }

  static void Descriptor(const Argument& argument, py::handle value, std::vector<Slot>& slots) {
    PackDescriptor(argument, Pointer(value.attr("base")),
                   py::cast<std::vector<uint64_t>>(value.attr("shape")),
                   py::cast<std::vector<uint64_t>>(value.attr("strides")),
                   py::cast<std::string>(value.attr("padding")) == "nan", slots);
  }

  std::vector<Argument> arguments_;
  size_t slot_capacity_ = 0;
  py::object metadata_;
  size_t num_args_;
  std::unique_ptr<backend::Kernel> kernel_;
};

// Device-specialization metadata only. Every hit still binds this call's tensors,
// evaluates its grid and prehooks, and fetches the current stream.
class KernelCache {
  struct Binding {
    int position = -1;
    py::object keyword = py::none();
    py::object constant = py::none();
    std::string keyword_name;
    c10::IValue native_constant;
  };
  struct Variant {
    struct Hook {
      NativeHook entry;
      flagos::native_host::MetadataHook packed;
      c10::Dict<std::string, c10::IValue> constants;
      std::vector<flagos::native_host::Keyword> constant_views;
    };
    std::vector<Hook> native_hooks;
    std::string native_key;
    py::object owner;
    py::object result;
    py::dict constants;
    py::tuple hooks;
    Launcher* launcher;
    uint64_t function;
    std::vector<Binding> bindings;
    bool native = false;
    c10::Dict<std::string, int64_t> native_meta;
  };

 public:
  KernelCache(py::tuple names, py::dict defaults, bool returns_constants)
      : names_(names), defaults_(defaults), returns_constants_(returns_constants) {
    for (auto name : names) native_names_.push_back(py::cast<std::string>(name));
  }

  py::object Run(py::tuple args, py::dict kwargs, py::object grid, int device) {
    std::string key;
    if (!Key(args, kwargs, device, key)) return py::none();
    const Variant* selected;
    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      auto found = variants_.find(key);
      if (found == variants_.end()) return py::bytes(key);
      selected = found->second.get();
    }
    const auto& variant = *selected;
    py::dict named;
    const bool grid_callable = PyCallable_Check(grid.ptr());
    if (grid_callable || !variant.hooks.empty()) {
      for (size_t i = 0; i < args.size(); ++i) named[names_[i]] = args[i];
      named.attr("update")(kwargs);
    }
    if (grid_callable) {
      py::dict metadata(named);
      metadata = py::reinterpret_steal<py::dict>(PyDict_Copy(named.ptr()));
      metadata.attr("update")(variant.constants);
      grid = grid(metadata);
    }
    for (auto item : variant.hooks) {
      auto hook = py::cast<py::tuple>(item);
      auto values = py::reinterpret_steal<py::dict>(PyDict_Copy(named.ptr()));
      values.attr("update")(hook[1]);
      hook[0](values);
    }
    auto dimensions = py::cast<py::sequence>(grid);
    if (dimensions.empty()) throw std::invalid_argument("Triton grid must not be empty");
    unsigned x = py::cast<unsigned>(dimensions[0]);
    unsigned y = dimensions.size() > 1 ? py::cast<unsigned>(dimensions[1]) : 1;
    unsigned z = dimensions.size() > 2 ? py::cast<unsigned>(dimensions[2]) : 1;
    std::vector<py::handle> values;
    values.reserve(variant.bindings.size());
    for (const auto& binding : variant.bindings) {
      if (binding.position >= 0)
        values.push_back(args[binding.position]);
      else if (!binding.keyword.is_none()) {
        PyObject* value = PyDict_GetItemWithError(kwargs.ptr(), binding.keyword.ptr());
        if (!value) throw std::invalid_argument("missing current kernel keyword");
        values.emplace_back(value);
      } else
        values.push_back(binding.constant);
    }
    variant.launcher->CallValues(x, y, z, Stream(device), variant.function, values);
    return variant.result;
  }

  void Prepare(py::bytes original_key, py::tuple args, py::dict kwargs, py::tuple record) {
    std::string key = original_key;
    auto variant = std::make_unique<Variant>();
    variant->owner = record;
    variant->launcher = py::cast<Launcher*>(record[1]);
    variant->function = py::cast<uint64_t>(record[2]);
    variant->constants = py::cast<py::dict>(record[4]);
    variant->hooks = py::cast<py::tuple>(record[7]);
    variant->native = variant->launcher->NativeSafe();
    for (auto item : variant->hooks) {
      auto hook = py::cast<py::tuple>(item);
      auto found = NativeHooks().find(hook[0].ptr());
      if (found == NativeHooks().end()) {
        variant->native = false;
        break;
      }
      auto constants = c10::impl::toTypedDict<std::string, c10::IValue>(
          c10::impl::GenericDict(c10::StringType::get(), c10::AnyType::get()));
      try {
        for (auto entry : py::cast<py::dict>(hook[1]))
          constants.insert(py::cast<std::string>(entry.first), FromPython(entry.second));
      } catch (const std::exception&) {
        variant->native = false;
        PyErr_Clear();
        break;
      }
      std::vector<flagos::native_host::Keyword> views;
      for (const auto& item : constants) views.push_back({item.key(), item.value()});
      variant->native_hooks.push_back(
          {found->second.entry, found->second.packed, std::move(constants), std::move(views)});
    }
    for (auto item : variant->constants)
      if (PyLong_Check(item.second.ptr()))
        variant->native_meta.insert(py::cast<std::string>(item.first),
                                    py::cast<int64_t>(item.second));
    variant->result = returns_constants_ ? py::object(py::make_tuple(record[0], record[4]))
                                         : py::reinterpret_borrow<py::object>(record[0]);
    for (size_t i = 0; i < names_.size(); ++i) {
      Binding binding;
      py::handle name = names_[i];
      if (i < args.size())
        binding.position = static_cast<int>(i);
      else if (kwargs.contains(name)) {
        binding.keyword = py::reinterpret_borrow<py::object>(name);
        binding.keyword_name = py::cast<std::string>(name);
      } else {
        bool present = false;
        for (auto mapping : {py::cast<py::dict>(record[5]), py::cast<py::dict>(record[6]),
                             variant->constants, defaults_}) {
          if (mapping.contains(name)) {
            binding.constant = mapping[name];
            present = true;
            break;
          }
        }
        if (!present) throw std::invalid_argument("missing specialization parameter");
        if (variant->launcher->RuntimeArgument(i)) {
          try {
            binding.native_constant = torch::jit::toIValue(binding.constant, c10::AnyType::get());
          } catch (const std::exception&) {
            variant->native = false;
            PyErr_Clear();
          }
        }
      }
      variant->bindings.push_back(std::move(binding));
    }
    variant->native_key = key;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto inserted = variants_.emplace(std::move(key), std::move(variant));
    if (inserted.second && inserted.first->second->native) {
      auto* item = inserted.first->second.get();
      native_variants_[std::hash<std::string_view>{}(item->native_key)].push_back(item);
    }
  }

  template <class Args, class Kwargs>
  bool RunNative(const Args& args, const Kwargs& kwargs, const std::vector<int64_t>& grid,
                 int device, int64_t grid_function = 0) {
    static_assert(std::is_same_v<Kwargs, flagos::native_host::Keywords> ||
                      std::is_lvalue_reference_v<decltype((*kwargs.begin()).key())>,
                  "native keyword views require dictionary-owned strings");
    StackKey key;
    Append(key, device);
    key.push_back('q');
    Append(key, args.size());
    for (const auto& value : args)
      if (!NativeValue(key, value)) return false;
    c10::SmallVector<std::pair<std::string_view, const c10::IValue*>, 8> ordered;
    c10::SmallVector<c10::IValue, 8> owned_keywords;
    if constexpr (!std::is_lvalue_reference_v<decltype((*kwargs.begin()).value())>)
      owned_keywords.reserve(kwargs.size());
    for (const auto& item : kwargs) {
      if constexpr (std::is_lvalue_reference_v<decltype(item.value())>)
        ordered.emplace_back(item.key(), &item.value());
      else {
        owned_keywords.push_back(item.value());
        ordered.emplace_back(item.key(), &owned_keywords.back());
      }
    }
    const c10::IValue warmup(false);
    if (!kwargs.contains("warmup")) ordered.emplace_back("warmup", &warmup);
    std::sort(ordered.begin(), ordered.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    Append(key, ordered.size());
    for (const auto& item : ordered) {
      Text(key, item.first.data(), item.first.size());
      if (!NativeValue(key, *item.second)) return false;
    }
    const Variant* selected = nullptr;
    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      auto found = native_variants_.find(std::hash<std::string_view>{}(key.view()));
      if (found == native_variants_.end()) return false;
      for (auto* candidate : found->second)
        if (candidate->native_key == key.view()) {
          selected = candidate;
          break;
        }
      if (!selected) return false;
    }
    const auto& variant = *selected;
    auto get_value = [&](size_t position) -> decltype(auto) {
      const auto& binding = variant.bindings.at(position);
      if constexpr (std::is_same_v<Args, flagos::native_host::Arguments>) {
        if (binding.position >= 0) return args.get(binding.position);
        if (!binding.keyword_name.empty()) return kwargs.at(binding.keyword_name);
        return (binding.native_constant);
      } else {
        if (binding.position >= 0) return c10::IValue(args.get(binding.position));
        if (!binding.keyword_name.empty())
          return c10::IValue(kwargs.at(binding.keyword_name));
        return c10::IValue(binding.native_constant);
      }
    };
    std::vector<int64_t> actual_grid;
    if (grid_function) {
      if constexpr (std::is_same_v<Args, flagos::native_host::Arguments>) {
        flagos::native_host::GridMetadata metadata{&variant.native_meta, &native_names_, args,
                                                   kwargs};
        using Grid = std::vector<int64_t> (*)(const flagos::native_host::GridMetadata&,
                                              std::vector<int64_t>);
        actual_grid = reinterpret_cast<Grid>(grid_function)(metadata, grid);
      } else {
        HostGridMeta metadata{&variant.native_meta, &native_names_, &args, &kwargs};
        using Grid = std::vector<int64_t> (*)(const HostGridMeta&, std::vector<int64_t>);
        actual_grid = reinterpret_cast<Grid>(grid_function)(metadata, grid);
      }
    }
    for (const auto& hook : variant.native_hooks) {
      if (hook.packed) {
        if constexpr (std::is_same_v<Args, flagos::native_host::Arguments>) {
          hook.packed({&native_names_, args, kwargs, {hook.constant_views}});
        } else {
          c10::SmallVector<c10::IValue, 32> current_args;
          for (const auto& value : args) current_args.push_back(value);
          c10::SmallVector<flagos::native_host::Keyword, 16> current_kwargs;
          for (const auto& item : kwargs) current_kwargs.push_back({item.key(), item.value()});
          hook.packed({&native_names_, {current_args}, {current_kwargs}, {hook.constant_views}});
        }
      } else {
        auto current = c10::impl::toTypedDict<std::string, c10::IValue>(
            c10::impl::GenericDict(c10::StringType::get(), c10::AnyType::get()));
        for (size_t i = 0; i < args.size(); ++i) current.insert(native_names_.at(i), args.get(i));
        for (const auto& item : kwargs)
          current.insert_or_assign(std::string(item.key()), item.value());
        for (const auto& item : hook.constants) current.insert_or_assign(item.key(), item.value());
        hook.entry(current);
      }
    }
    variant.launcher->CallNative(variant.bindings.size(), get_value,
                                 grid_function ? actual_grid : grid, device, variant.function);
    return true;
  }

 private:
  struct StackKey {
    c10::SmallVector<char, 1024> bytes;
    void push_back(char value) { bytes.push_back(value); }
    void append(const char* data, size_t size) { bytes.append(data, data + size); }
    void append(const char* text) { append(text, std::strlen(text)); }
    std::string_view view() const { return {bytes.data(), bytes.size()}; }
  };
  template <class Buffer, class T>
  static void Append(Buffer& key, const T& value) {
    key.append(reinterpret_cast<const char*>(&value), sizeof(value));
  }
  template <class Buffer>
  static void Text(Buffer& key, const char* value, size_t size) {
    Append(key, size);
    key.append(value, size);
  }
  static bool Value(std::string& key, py::handle value) {
    PyObject* object = value.ptr();
    if (object == Py_None) {
      key.push_back('n');
      return true;
    }
    if (PyBool_Check(object)) {
      key.push_back(object == Py_True ? 't' : 'f');
      return true;
    }
    if (PyLong_CheckExact(object)) {
      key.push_back('i');
      int overflow = 0;
      auto number = PyLong_AsLongLongAndOverflow(object, &overflow);
      if (!overflow) {
        key.push_back('s');
        Append(key, number);
      } else {
        key.push_back('b');
        auto text = py::str(value);
        Text(key, PyUnicode_AsUTF8(text.ptr()), py::len(text));
      }
      return true;
    }
    if (PyFloat_CheckExact(object)) {
      key.push_back('d');
      Append(key, PyFloat_AS_DOUBLE(object));
      return true;
    }
    if (PyUnicode_Check(object)) {
      key.push_back('s');
      Py_ssize_t size = 0;
      const char* data = PyUnicode_AsUTF8AndSize(object, &size);
      if (!data) throw py::error_already_set();
      Text(key, data, size);
      return true;
    }
    if (THPDtype_Check(object)) {
      key.push_back('y');
      Append(key, reinterpret_cast<THPDtype*>(object)->scalar_type);
      return true;
    }
    if (Py_TYPE(object) == buffer_type) {
      key.push_back('b');
      const auto& buffer = py::cast<const StridedBuffer&>(value);
      Value(key, buffer.dtype);
      Append(key, buffer.pointer % 16 == 0);
      Append(key, buffer.sizes.size());
      for (auto size : buffer.sizes) Append(key, size);
      for (auto stride : buffer.steps) Append(key, stride);
      return true;
    }
    if (THPVariable_CheckExact(object)) {
      key.push_back('T');
      const auto& tensor = THPVariable_Unpack(object);
      Append(key, tensor.scalar_type());
      Append(key, tensor.device().index());
      Append(key, tensor.dim());
      Append(key, reinterpret_cast<uint64_t>(tensor.const_data_ptr()) % 16 == 0);
      for (auto size : tensor.sizes()) Append(key, size);
      for (auto stride : tensor.strides()) Append(key, stride);
      return true;
    }
    if (PyTuple_Check(object) || PyList_Check(object)) {
      key.push_back('q');
      auto items = py::reinterpret_borrow<py::sequence>(value);
      Append(key, items.size());
      for (auto item : items)
        if (!Value(key, item)) return false;
      return true;
    }
    if (py::hasattr(value, "data_ptr")) {
      key.push_back('B');
      return Value(key, value.attr("dtype")) &&
             Value(key, py::bool_(py::cast<uint64_t>(value.attr("data_ptr")()) % 16 == 0)) &&
             Value(key, value.attr("shape")) && Value(key, value.attr("stride")());
    }
    if (py::hasattr(value, "base") && py::hasattr(value, "block_shape")) {
      key.push_back('D');
      for (const char* name : {"base", "shape", "strides", "block_shape", "padding"})
        if (!Value(key, value.attr(name))) return false;
      return true;
    }
    return false;
  }
  template <class Buffer>
  static bool NativeValue(Buffer& key, const c10::IValue& value) {
    if (IsDescriptor(value)) {
      key.push_back('D');
      if (value.isList()) {
        for (const auto& field : value.toListRef().slice(1))
          if (!NativeValue(key, field)) return false;
        return true;
      }
      const flagos::native_host::Descriptor descriptor(value);
      for (const char* name : {"base", "shape", "strides", "block_shape", "padding"})
        if (!NativeValue(key, descriptor.at(name))) return false;
      return true;
    }
    if (value.isNone()) {
      key.push_back('n');
      return true;
    }
    if (value.isBool()) {
      key.push_back(value.toBool() ? 't' : 'f');
      return true;
    }
    if (value.isInt()) {
      key.append("is");
      Append(key, value.toInt());
      return true;
    }
    if (value.isDouble()) {
      key.push_back('d');
      Append(key, value.toDouble());
      return true;
    }
    if (value.isString()) {
      key.push_back('s');
      const auto& text = value.toStringRef();
      Text(key, text.data(), text.size());
      return true;
    }
    if (value.isTensor()) {
      key.push_back('T');
      const auto& tensor = value.toTensor();
      Append(key, tensor.scalar_type());
      Append(key, tensor.device().index());
      Append(key, tensor.dim());
      Append(key, reinterpret_cast<uint64_t>(tensor.const_data_ptr()) % 16 == 0);
      if (tensor.dim()) {
        key.append(reinterpret_cast<const char*>(tensor.sizes().data()),
                   tensor.dim() * sizeof(int64_t));
        key.append(reinterpret_cast<const char*>(tensor.strides().data()),
                   tensor.dim() * sizeof(int64_t));
      }
      return true;
    }
    if (value.isList()) {
      key.push_back('q');
      const auto list = value.toListRef();
      Append(key, list.size());
      for (const auto& item : list)
        if (!NativeValue(key, item)) return false;
      return true;
    }
    if (value.isTuple()) {
      key.push_back('q');
      const auto& tuple = value.toTupleRef().elements();
      Append(key, tuple.size());
      for (const auto& item : tuple)
        if (!NativeValue(key, item)) return false;
      return true;
    }
    return false;
  }
  static bool Key(py::tuple args, py::dict kwargs, int device, std::string& key) {
    key.reserve(512);
    Append(key, device);
    if (!Value(key, args)) return false;
    std::vector<std::pair<std::string, py::handle>> keywords;
    keywords.reserve(kwargs.size());
    for (auto item : kwargs) keywords.emplace_back(py::cast<std::string>(item.first), item.second);
    std::sort(keywords.begin(), keywords.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    Append(key, keywords.size());
    for (const auto& item : keywords) {
      Text(key, item.first.data(), item.first.size());
      if (!Value(key, item.second)) return false;
    }
    return true;
  }
  py::tuple names_;
  std::vector<std::string> native_names_;
  py::dict defaults_;
  bool returns_constants_;
  std::unordered_map<std::string, std::unique_ptr<Variant>> variants_;
  std::unordered_map<size_t, std::vector<const Variant*>> native_variants_;
  std::shared_mutex mutex_;
};

void ExecutePointwise(py::object owner, size_t rank, py::tuple args,
                      const std::vector<py::object>& outputs, const std::vector<int64_t>& shape,
                      bool collapse);

class PointwiseSpec {
 public:
  struct Prepared {
    c10::SmallVector<at::Tensor, 4> outputs;
    c10::SmallVector<int64_t, 8> shape;
    bool collapse;
  };
  PointwiseSpec(py::object schema, py::object promotion, py::object config)
      : promotion_(promotion), config_(config) {
    const size_t count = py::cast<size_t>(schema.attr("num_inputs")());
    for (size_t i = 0; i < count; ++i)
      inputs_.push_back(py::cast<bool>(schema.attr("is_tensor")(i)));
    const size_t outputs = py::cast<size_t>(schema.attr("num_output_tensors")());
    for (size_t i = 0; i < outputs; ++i) keys_.emplace_back("out" + std::to_string(i));
    promotions_ = py::tuple(schema.attr("_promotion_methods"));
    for (size_t i = 0; i < outputs; ++i) {
      output_names_.push_back("out" + std::to_string(i));
      auto rule = py::cast<py::tuple>(promotions_[i]);
      std::vector<size_t> indices;
      for (size_t j = 0; j + 1 < rule.size(); ++j) indices.push_back(py::cast<size_t>(rule[j]));
      promotion_indices_.push_back(std::move(indices));
    }
  }
  template <class Args, class Kwargs>
  std::optional<Prepared> PrepareNative(const Args& args, const Kwargs& kwargs) {
    if (args.size() != inputs_.size()) return std::nullopt;
    Prepared prepared;
    prepared.outputs.resize(output_names_.size());
    c10::SmallVector<at::Tensor, 8> tensors, inputs;
    for (size_t i = 0; i < output_names_.size(); ++i)
      if (kwargs.contains(output_names_[i]) && !kwargs.at(output_names_[i]).isNone()) {
        const auto value = kwargs.at(output_names_[i]);
        if (!value.isTensor()) return std::nullopt;
        prepared.outputs[i] = value.toTensor();
        tensors.push_back(prepared.outputs[i]);
      }
    for (size_t i = 0; i < args.size(); ++i) {
      auto value = args.get(i);
      if (inputs_[i]) {
        if (!value.isTensor()) return std::nullopt;
        inputs.push_back(value.toTensor());
        tensors.push_back(inputs.back());
      } else if (!value.isInt() && !value.isDouble() && !value.isBool())
        return std::nullopt;
    }
    if (inputs.empty()) return std::nullopt;
    const auto& first = tensors.front();
    if (first.device().type() != c10::DeviceType::PrivateUse1) return std::nullopt;
    CurrentDeviceScope device_scope(first.device());
    if (IsCapturing(Stream(first.device().index()))) return std::nullopt;
    bool same_shape = true, same_stride = true, contiguous = true;
    for (const auto& tensor : tensors) {
      if (tensor.device().type() != c10::DeviceType::PrivateUse1 || tensor.is_complex() ||
          (tensor.key_set().has(c10::DispatchKey::Python) ||
           tensor.key_set().has(c10::DispatchKey::Functionalize)))
        return std::nullopt;
      same_shape &= tensor.sizes() == first.sizes();
      same_stride &= tensor.strides() == first.strides();
      contiguous &= tensor.is_contiguous();
    }
    if (first.numel() > INT32_MAX) {
      py::gil_scoped_acquire gil;
      config_.attr("prefer_block_pointer") = false;
    }
    prepared.collapse =
        same_shape && (contiguous || (same_stride && first.is_non_overlapping_and_dense()));
    if (prepared.collapse)
      prepared.shape = {first.numel()};
    else {
      size_t rank = 0;
      for (const auto& tensor : inputs) rank = std::max(rank, static_cast<size_t>(tensor.dim()));
      prepared.shape.assign(rank, 1);
      for (const auto& tensor : inputs)
        for (int64_t i = 0; i < tensor.dim(); ++i) {
          auto& target = prepared.shape[rank - tensor.dim() + i];
          auto size = tensor.size(i);
          if (target != size && target != 1 && size != 1) return std::nullopt;
          if (target == 1) target = size;
        }
      for (const auto& output : prepared.outputs)
        if (output.defined() && (output.sizes() != at::IntArrayRef(prepared.shape) ||
                                 !output.is_non_overlapping_and_dense()))
          return std::nullopt;
    }
    const at::Tensor* like = nullptr;
    if (prepared.collapse)
      like = &first;
    else
      for (const auto& tensor : tensors)
        if (tensor.sizes() == at::IntArrayRef(prepared.shape)) {
          like = &tensor;
          break;
        }
    for (size_t i = 0; i < prepared.outputs.size(); ++i)
      if (!prepared.outputs[i].defined()) {
        // Promotion depends on types and scalar rank, not tensor values. Keep
        // the complete existing key without formatting integers on every call.
        std::string key(reinterpret_cast<const char*>(&i), sizeof(i));
        key.push_back(static_cast<char>(c10::get_default_dtype_as_scalartype()));
        for (auto index : promotion_indices_[i]) {
          const auto value = args.get(index);
          if (value.isTensor()) {
            const auto& tensor = value.toTensor();
            key.push_back('T');
            key.push_back(static_cast<char>(tensor.scalar_type()));
            key.push_back(tensor.dim() == 0);
          } else
            key.push_back(value.isBool() ? 'b' : value.isInt() ? 'i' : 'f');
        }
        std::optional<at::ScalarType> dtype;
        {
          std::lock_guard<std::mutex> lock(dtype_mutex_);
          auto found = native_dtypes_.find(key);
          if (found != native_dtypes_.end()) dtype = found->second;
        }
        if (!dtype) {
          py::gil_scoped_acquire gil;
          py::tuple values(promotion_indices_[i].size());
          for (size_t j = 0; j < values.size(); ++j)
            values[j] = torch::jit::toPyObject(args.get(promotion_indices_[i][j]));
          auto rule = py::cast<py::tuple>(promotions_[i]);
          py::dict options;
          options["type_promotion"] = rule[rule.size() - 1];
          auto result = py::cast<py::tuple>(promotion_(*values, **options));
          TORCH_CHECK(THPDtype_Check(result[1].ptr()), "invalid pointwise promotion dtype");
          dtype = reinterpret_cast<THPDtype*>(result[1].ptr())->scalar_type;
          std::lock_guard<std::mutex> lock(dtype_mutex_);
          native_dtypes_.emplace(key, *dtype);
        }
        // The surrounding device scope already selected the allocation device.
        // Use the same constructors as FlagOS empty/empty_strided for ordinary
        // dense results; empty_like retains the general Preserve/names rules.
        auto* allocator = at::GetAllocator(at::kPrivateUse1);
        constexpr c10::DispatchKeySet keys(c10::DispatchKey::PrivateUse1);
        if (like && like->is_non_overlapping_and_dense() && !like->has_names())
          prepared.outputs[i] = at::detail::empty_strided_generic(
              like->sizes(), like->strides(), allocator, keys, *dtype);
        else if (like)
          prepared.outputs[i] =
              at::empty_like(*like, like->options().dtype(*dtype), at::MemoryFormat::Preserve);
        else
          prepared.outputs[i] = at::detail::empty_generic(
              prepared.shape, allocator, keys, *dtype, at::MemoryFormat::Contiguous);
      }
    return prepared;
  }
  py::object Prepare(py::tuple args, py::dict kwargs, int device, py::object owner = py::none()) {
    if (args.size() != inputs_.size()) return py::none();
    if (IsCapturing(Stream(device))) return py::none();
    std::vector<py::object> outputs(keys_.size(), py::none());
    std::vector<py::object> tensors;
    for (size_t i = 0; i < keys_.size(); ++i) {
      if (kwargs.contains(keys_[i])) {
        outputs[i] = kwargs[keys_[i]];
        if (!THPVariable_CheckExact(outputs[i].ptr())) return py::none();
        tensors.push_back(outputs[i]);
      }
    }
    std::vector<const at::Tensor*> inputs;
    for (size_t i = 0; i < args.size(); ++i) {
      if (inputs_[i]) {
        if (!THPVariable_CheckExact(args[i].ptr())) return py::none();
        tensors.push_back(py::reinterpret_borrow<py::object>(args[i]));
        inputs.push_back(&THPVariable_Unpack(args[i].ptr()));
      } else if (THPVariable_Check(args[i].ptr()))
        return py::none();
      else if (!owner.is_none() && !PyLong_Check(args[i].ptr()) && !PyFloat_Check(args[i].ptr()) &&
               args[i].ptr() != Py_None)
        return py::none();
    }
    if (tensors.empty() || inputs.empty()) return py::none();
    const auto& first = THPVariable_Unpack(tensors.front().ptr());
    bool same_shape = true, same_stride = true, contiguous = true;
    for (const auto& item : tensors) {
      const auto& tensor = THPVariable_Unpack(item.ptr());
      same_shape &= tensor.sizes() == first.sizes();
      same_stride &= tensor.strides() == first.strides();
      contiguous &= tensor.is_contiguous();
    }
    if (first.numel() > INT32_MAX) config_.attr("prefer_block_pointer") = false;
    const bool collapse =
        same_shape && (contiguous || (same_stride && first.is_non_overlapping_and_dense()));
    std::vector<int64_t> task;
    if (collapse)
      task = {first.numel()};
    else {
      size_t rank = 0;
      for (auto tensor : inputs) rank = std::max(rank, static_cast<size_t>(tensor->dim()));
      task.assign(rank, 1);
      for (auto tensor : inputs) {
        const size_t leading = rank - tensor->dim();
        for (size_t i = 0; i < static_cast<size_t>(tensor->dim()); ++i) {
          const auto size = tensor->size(i);
          auto& target = task[leading + i];
          if (target != size && target != 1 && size != 1) return py::none();
          if (target == 1) target = size;
        }
      }
      for (const auto& output : outputs)
        if (!output.is_none()) {
          const auto& tensor = THPVariable_Unpack(output.ptr());
          if (tensor.sizes() != at::IntArrayRef(task) || !tensor.is_non_overlapping_and_dense())
            return py::none();
        }
    }
    const at::Tensor* like = nullptr;
    if (collapse)
      like = &first;
    else
      for (const auto& item : tensors) {
        const auto& tensor = THPVariable_Unpack(item.ptr());
        if (tensor.sizes() == at::IntArrayRef(task)) {
          like = &tensor;
          break;
        }
      }
    for (size_t i = 0; i < outputs.size(); ++i)
      if (outputs[i].is_none()) {
        auto rule = py::cast<py::tuple>(promotions_[i]);
        py::tuple promote(rule.size() - 1);
        for (size_t j = 0; j < promote.size(); ++j) promote[j] = args[py::cast<size_t>(rule[j])];
        std::string key =
            std::to_string(i) + ":" + std::to_string(int(c10::get_default_dtype_as_scalartype()));
        bool supported = true;
        for (auto value : promote) {
          if (THPVariable_CheckExact(value.ptr())) {
            const auto& tensor = THPVariable_Unpack(value.ptr());
            key += ":T" + std::to_string(int(tensor.scalar_type())) + ":" +
                   std::to_string(tensor.dim() == 0);
          } else if (PyBool_Check(value.ptr()))
            key += ":b";
          else if (PyLong_CheckExact(value.ptr()))
            key += ":i";
          else if (PyFloat_CheckExact(value.ptr()))
            key += ":f";
          else
            supported = false;
        }
        auto found = supported ? dtypes_.find(key) : dtypes_.end();
        at::ScalarType dtype;
        if (found != dtypes_.end())
          dtype = found->second;
        else {
          py::dict options;
          options["type_promotion"] = rule[rule.size() - 1];
          auto result = py::cast<py::tuple>(promotion_(*promote, **options));
          if (!THPDtype_Check(result[1].ptr())) return py::none();
          dtype = reinterpret_cast<THPDtype*>(result[1].ptr())->scalar_type;
          if (supported) dtypes_.emplace(key, dtype);
        }
        auto tensor =
            like ? at::empty_like(*like, like->options().dtype(dtype), at::MemoryFormat::Preserve)
                 : at::empty(task, first.options().dtype(dtype));
        outputs[i] = py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor));
      }
    if (!owner.is_none()) {
      ExecutePointwise(owner, task.size(), args, outputs, task, collapse);
      if (outputs.size() == 1) return outputs.front();
      py::tuple result(outputs.size());
      for (size_t i = 0; i < outputs.size(); ++i) result[i] = outputs[i];
      return result;
    }
    auto shape = py::cast(task);
    auto wrap = [&](py::object object) {
      const auto& tensor = THPVariable_Unpack(object.ptr());
      std::vector<int64_t> strides(task.size(), 0);
      if (collapse)
        strides[0] = 1;
      else {
        const size_t leading = task.size() - tensor.dim();
        for (size_t i = 0; i < static_cast<size_t>(tensor.dim()); ++i)
          strides[leading + i] =
              tensor.size(i) == 1 && task[leading + i] != 1 ? 0 : tensor.stride(i);
      }
      return py::cast(StridedBuffer(object, shape, py::cast(strides), py::none(), 0));
    };
    py::tuple values(args.size());
    for (size_t i = 0; i < args.size(); ++i)
      values[i] = inputs_[i] ? wrap(py::reinterpret_borrow<py::object>(args[i]))
                             : py::reinterpret_borrow<py::object>(args[i]);
    py::dict out;
    for (size_t i = 0; i < outputs.size(); ++i) out[keys_[i]] = wrap(outputs[i]);
    return py::make_tuple(task.size(), values, out);
  }

 private:
  std::vector<bool> inputs_;
  std::vector<py::str> keys_;
  py::tuple promotions_;
  py::object promotion_, config_;
  std::unordered_map<std::string, at::ScalarType> dtypes_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<size_t>> promotion_indices_;
  std::unordered_map<std::string, at::ScalarType> native_dtypes_;
  std::mutex dtype_mutex_;
};

class PointwiseSchedule {
  struct View {
    std::vector<int64_t> shape, strides;
    int64_t elements;
    int device;
  };
  static bool Read(py::handle object, View& view) {
    if (Py_TYPE(object.ptr()) == buffer_type) {
      const auto& b = py::cast<const StridedBuffer&>(object);
      view.shape = b.sizes;
      view.strides = b.steps;
      view.elements = b.elements;
      view.device = py::cast<int>(b.device.attr("index"));
      return true;
    }
    if (THPVariable_CheckExact(object.ptr())) {
      const auto& t = THPVariable_Unpack(object.ptr());
      view.shape = t.sizes().vec();
      view.strides = t.strides().vec();
      view.elements = t.numel();
      view.device = t.device().index();
      return true;
    }
    return false;
  }

 public:
  PointwiseSchedule(py::object owner, size_t rank, py::object kernel, py::tuple helpers,
                    py::tuple device_api)
      : rank_(rank), kernel_(kernel), helpers_(helpers), device_api_(device_api) {
    auto schema = owner.attr("fx");
    auto config = owner.attr("config");
    for (size_t i = 0, n = py::cast<size_t>(schema.attr("num_inputs")()); i < n; ++i)
      if (py::cast<bool>(schema.attr("is_tensor")(i))) tensors_.push_back(i);
    for (size_t i = 0, n = py::cast<size_t>(schema.attr("num_output_tensors")()); i < n; ++i)
      outputs_.emplace_back("out" + std::to_string(i));
    for (size_t i = 0; i < rank_; ++i) tile_names_.push_back("tile_size" + std::to_string(i));
    one_dim_ = py::cast<bool>(config.attr("prefer_1d_tile"));
    block_pointer_ = py::cast<bool>(config.attr("prefer_block_pointer"));
    balanced_ = py::cast<bool>(config.attr("balance_grid"));
    max_grid_ = py::cast<int64_t>(config.attr("max_grid_size")[py::int_(0)]);
    max_tile_ = py::cast<int64_t>(config.attr("max_tile_size")) /
                (py::cast<bool>(helpers[4](schema)) ? 2 : 1);
    const auto name = py::cast<std::string>(owner.attr("_scalar_fn").attr("__name__"));
    fill_ = name.find("fill_scalar") != std::string::npos && py::cast<int>(helpers[5]) >= 9;
    cache_ = py::cast<KernelCache*>(kernel.attr("native_cache"));
    warps_.resize(std::max<int64_t>(max_tile_, 1024) + 1);
    for (size_t i = 1; i < warps_.size(); ++i) warps_[i] = py::cast<int64_t>(helpers_[1](i));
  }
  template <class Args, class Outputs>
  void CallNative(const Args& args, const Outputs& outputs,
                  std::optional<at::IntArrayRef> task = std::nullopt, bool collapse = false) {
    c10::SmallVector<at::Tensor, 8> tensors;
    for (auto i : tensors_) tensors.push_back(args.get(i).toTensor());
    for (const auto& out : outputs) tensors.push_back(out);
    TORCH_CHECK(!tensors.empty() && outputs.size() == outputs_.size(),
                "pointwise output arity mismatch");
    const auto& output = tensors[tensors_.size()];
    if (!task)
      for (const auto& tensor : tensors)
        TORCH_CHECK(tensor.sizes() == output.sizes(), "pointwise inputs must be broadcasted");
    const at::IntArrayRef task_shape = task ? *task : output.sizes();
    const int64_t tasks = output.numel();
    if (rank_ && !tasks) return;
    c10::SmallVector<c10::IValue, 64> values;
    for (const auto& value : args) values.push_back(value);
    for (size_t i = 0; i < outputs.size(); ++i) values.push_back(outputs.get(i));
    c10::SmallVector<flagos::native_host::Keyword, 16> options;
    int64_t ctas = 1, warps = 1;
    if (rank_) {
      c10::SmallVector<int64_t, 8> shape;
      if (one_dim_)
        shape.push_back(tasks);
      else
        shape.append(task_shape.begin(), task_shape.end());
      c10::SmallVector<int64_t, 8> tiles(shape.size(), 1);
      int64_t budget = max_tile_;
      if (fill_)
        tiles = {one_dim_ ? 1024 : 64};
      else
        for (size_t r = shape.size(); r > 0; --r) {
          uint64_t value = static_cast<uint64_t>(shape[r - 1]);
          uint64_t next = value <= 1 ? 1 : uint64_t(1) << (64 - __builtin_clzll(value - 1));
          tiles[r - 1] = std::min<int64_t>(budget, next);
          budget = std::max<int64_t>(1, budget / tiles[r - 1]);
        }
      int64_t tile_size = 1, count = 1;
      for (auto tile : tiles) tile_size *= tile;
      if (one_dim_)
        count = (tasks + tile_size - 1) / tile_size;
      else
        for (size_t i = 0; i < tiles.size(); ++i)
          count *= (task_shape[i] + tiles[i] - 1) / tiles[i];
      ctas = fill_ ? count : std::min(max_grid_, count);
      int64_t per_cta = (count + ctas - 1) / ctas;
      if (balanced_) ctas = (count + per_cta - 1) / per_cta;
      warps = warps_.at(tile_size);
      for (const auto& tensor : tensors) {
        c10::SmallVector<int64_t, 8> strides(task_shape.size(), 0);
        if (collapse)
          strides[0] = 1;
        else {
          size_t leading = task_shape.size() - tensor.dim();
          for (int64_t i = 0; i < tensor.dim(); ++i)
            strides[leading + i] =
                tensor.size(i) == 1 && task_shape[leading + i] != 1 ? 0 : tensor.stride(i);
        }
        for (auto stride : strides) values.push_back(stride);
        if (block_pointer_ && !one_dim_) {
          c10::SmallVector<int64_t, 8> order(strides.size());
          for (size_t i = 0; i < strides.size(); ++i) order[i] = i;
          std::sort(order.begin(), order.end(), [&](auto a, auto b) {
            auto left = std::abs(strides[a]), right = std::abs(strides[b]);
            return left == right ? a < b : left < right;
          });
          for (auto dim : order) values.push_back(dim);
        }
      }
      for (auto size : task_shape) values.push_back(size);
      values.push_back(tasks);
      options.push_back({"tiles_per_cta", per_cta});
      options.push_back({"one_tile_per_cta", per_cta == 1});
      if (one_dim_)
        options.push_back({"tile_size", tile_size});
      else
        for (size_t i = 0; i < tiles.size(); ++i) options.push_back({tile_names_.at(i), tiles[i]});
    }
    options.push_back({"num_warps", warps});
    const int device = tensors.front().device().index();
    CurrentDeviceScope guard(tensors.front().device());
    const std::vector<int64_t> grid{ctas, 1, 1};
    if (cache_->RunNative(flagos::native_host::Arguments{values},
                          flagos::native_host::Keywords{options}, grid, device))
      return;
    py::gil_scoped_acquire gil;
    py::tuple pyargs(values.size());
    for (size_t i = 0; i < values.size(); ++i) pyargs[i] = torch::jit::toPyObject(values[i]);
    py::dict pykwargs;
    for (const auto& item : options)
      pykwargs[py::str(std::string(item.key()))] = torch::jit::toPyObject(item.value());
    pykwargs["grid"] = py::tuple(py::cast(grid));
    kernel_.attr("run")(*pyargs, **pykwargs);
  }
  py::object Call(py::tuple args, py::dict kwargs) {
    std::vector<View> views;
    for (auto i : tensors_) {
      View v;
      if (!Read(args[i], v)) return py::none();
      views.push_back(std::move(v));
    }
    py::tuple results(outputs_.size());
    for (size_t i = 0; i < outputs_.size(); ++i) {
      if (!kwargs.contains(outputs_[i])) return py::none();
      results[i] = kwargs[outputs_[i]];
      View v;
      if (!Read(results[i], v)) return py::none();
      views.push_back(std::move(v));
    }
    const auto& output = views[tensors_.size()];
    for (const auto& view : views)
      if (view.shape != output.shape) return py::none();
    py::object result =
        results.size() == 1 ? py::reinterpret_borrow<py::object>(results[0]) : py::object(results);
    if (rank_ && !output.elements) return result;
    py::list parameters;
    for (auto value : args) parameters.append(value);
    for (auto value : results) parameters.append(value);
    int64_t ctAs = 1, warps = 1;
    py::dict options;
    options["warmup"] = false;
    if (rank_) {
      std::vector<int64_t> task = one_dim_ ? std::vector<int64_t>{output.elements} : output.shape;
      std::vector<int64_t> tiles;
      if (fill_)
        tiles = {one_dim_ ? 1024 : 64};
      else {
        py::tuple sizes(task.size() + 1);
        sizes[0] = max_tile_;
        for (size_t i = 0; i < task.size(); ++i) sizes[i + 1] = task[i];
        tiles = py::cast<std::vector<int64_t>>(helpers_[0](*sizes));
      }
      int64_t tile_size = 1, num_tiles = 1;
      for (auto tile : tiles) tile_size *= tile;
      if (one_dim_)
        num_tiles = (output.elements + tile_size - 1) / tile_size;
      else
        for (size_t i = 0; i < tiles.size(); ++i)
          num_tiles *= (output.shape[i] + tiles[i] - 1) / tiles[i];
      int64_t per_cta;
      if (balanced_) {
        auto partition = py::cast<py::tuple>(helpers_[3](num_tiles, max_grid_));
        ctAs = py::cast<int64_t>(partition[0]);
        per_cta = py::cast<int64_t>(partition[1]);
      } else {
        ctAs = fill_ ? num_tiles : std::min(max_grid_, num_tiles);
        per_cta = (num_tiles + ctAs - 1) / ctAs;
      }
      warps = py::cast<int64_t>(helpers_[1](tile_size));
      for (const auto& view : views) {
        for (auto stride : view.strides) parameters.append(stride);
        if (block_pointer_ && !one_dim_) {
          auto order =
              rank_ >= 2 ? helpers_[2](py::cast(view.strides)) : py::object(py::make_tuple(0));
          for (auto index : py::cast<py::sequence>(order)) parameters.append(index);
        }
      }
      for (auto size : output.shape) parameters.append(size);
      parameters.append(output.elements);
      options["tiles_per_cta"] = per_cta;
      options["one_tile_per_cta"] = per_cta == 1;
      if (one_dim_)
        options["tile_size"] = tile_size;
      else
        for (size_t i = 0; i < tiles.size(); ++i)
          options[py::str("tile_size" + std::to_string(i))] = tiles[i];
    }
    options["num_warps"] = warps;
    const int previous = py::cast<int>(device_api_[0](views.front().device));
    try {
      auto values = py::tuple(parameters);
      auto grid = py::make_tuple(ctAs, 1, 1);
      auto hit = cache_->Run(values, options, grid, views.front().device);
      if (hit.is_none() || PyBytes_Check(hit.ptr())) {
        options["grid"] = grid;
        kernel_.attr("run")(*values, **options);
      }
    } catch (...) {
      if (previous != views.front().device) device_api_[1](previous);
      throw;
    }
    if (previous != views.front().device) device_api_[1](previous);
    return result;
  }

 private:
  size_t rank_;
  std::vector<size_t> tensors_;
  std::vector<py::str> outputs_;
  py::object kernel_;
  py::tuple helpers_, device_api_;
  KernelCache* cache_;
  std::vector<std::string> tile_names_;
  bool one_dim_, block_pointer_, balanced_, fill_;
  int64_t max_grid_, max_tile_;
  std::vector<int64_t> warps_;
};

void ExecutePointwise(py::object owner, size_t rank, py::tuple args,
                      const std::vector<py::object>& outputs, const std::vector<int64_t>& shape,
                      bool collapse) {
  auto overload = owner.attr("instantiate")(rank);
  auto* schedule = py::cast<PointwiseSchedule*>(overload.attr("native"));
  c10::List<c10::IValue> values(c10::AnyType::get());
  for (auto value : args) values.push_back(torch::jit::toIValue(value, c10::AnyType::get()));
  c10::List<at::Tensor> tensors;
  for (const auto& output : outputs) tensors.push_back(THPVariable_Unpack(output.ptr()));
  schedule->CallNative(values, tensors, at::IntArrayRef(shape), collapse);
}

struct NativePointwiseSlot {
  py::object owner;
  std::unique_ptr<PointwiseSpec> spec;
  std::mutex mutex;
  std::unordered_map<PointwiseSchedule*, bool> schedules;
  std::vector<py::object> keepalive;
  std::atomic<uint64_t> version{0};
  std::atomic<bool> watched{false};
  PyObject* overload_dictionary = nullptr;
  uint64_t cached_version = 0;
  std::unordered_map<size_t, PointwiseSchedule*> ranks;
  std::vector<py::object> watched_objects;
};
std::vector<NativePointwiseSlot*>& NativePointwiseSlots() {
  static auto* slots = new std::vector<NativePointwiseSlot*>();
  return *slots;
}
std::mutex pointwise_slot_mutex;
#if PY_VERSION_HEX >= 0x030C0000 && !defined(Py_GIL_DISABLED)
std::unordered_map<PyObject*, std::vector<NativePointwiseSlot*>>& PointwiseWatchers() {
  static auto* watchers = new std::unordered_map<PyObject*, std::vector<NativePointwiseSlot*>>();
  return *watchers;
}
int InvalidatePointwise(PyDict_WatchEvent event, PyObject* dictionary, PyObject*, PyObject*) {
  auto found = PointwiseWatchers().find(dictionary);
  if (found != PointwiseWatchers().end())
    for (auto* entry : found->second)
      if (event != PyDict_EVENT_ADDED || dictionary != entry->overload_dictionary)
        entry->version.fetch_add(1, std::memory_order_release);
  return 0;
}
bool WatchPointwise(NativePointwiseSlot* entry, py::handle overload) {
  static const int watcher = PyDict_AddWatcher(&InvalidatePointwise);
  if (watcher < 0) {
    PyErr_Clear();
    return false;
  }
  try {
    const auto config = entry->owner.attr("config");
    entry->overload_dictionary = entry->owner.attr("overloads").ptr();
    for (py::object dictionary : {entry->owner.attr("__dict__"), config.attr("__dict__"),
                                  entry->owner.attr("overloads"), overload.attr("__dict__")}) {
      if (!PyDict_CheckExact(dictionary.ptr())) return false;
      auto& entries = PointwiseWatchers()[dictionary.ptr()];
      if (std::find(entries.begin(), entries.end(), entry) != entries.end()) continue;
      if (PyDict_Watch(watcher, dictionary.ptr()) < 0) {
        PyErr_Clear();
        return false;
      }
      entries.push_back(entry);
      entry->watched_objects.push_back(std::move(dictionary));
    }
    return true;
  } catch (const py::error_already_set&) {
    PyErr_Clear();
    return false;
  }
}
#else
bool WatchPointwise(NativePointwiseSlot*, py::handle) { return false; }
#endif

PointwiseSchedule* PointwiseForRank(NativePointwiseSlot* entry, size_t rank) {
  {
    std::lock_guard<std::mutex> lock(entry->mutex);
    if (entry->watched.load(std::memory_order_acquire) &&
        entry->cached_version == entry->version.load(std::memory_order_acquire)) {
      auto found = entry->ranks.find(rank);
      if (found != entry->ranks.end()) return found->second;
    }
  }
  py::gil_scoped_acquire gil;
  const auto before = entry->version.load(std::memory_order_acquire);
  auto overload = entry->owner.attr("instantiate")(rank);
  py::object native = overload.attr("native");
  auto* schedule = py::cast<PointwiseSchedule*>(native);
  const bool watched = WatchPointwise(entry, overload);
  std::lock_guard<std::mutex> lock(entry->mutex);
  if (!entry->schedules.count(schedule)) {
    entry->keepalive.push_back(overload);
    entry->keepalive.push_back(native);
    entry->schedules.emplace(schedule, true);
  }
  const auto version = entry->version.load(std::memory_order_acquire);
  if (entry->cached_version != version) entry->ranks.clear();
  entry->cached_version = version;
  if (before == version) entry->ranks.insert_or_assign(rank, schedule);
  entry->watched.store(watched, std::memory_order_release);
  return schedule;
}
struct ForeignHost {
  py::object function;
  py::tuple prefix;
};
std::vector<ForeignHost*>& ForeignHosts() {
  static auto* entries = new std::vector<ForeignHost*>();
  return *entries;
}
std::mutex foreign_host_mutex;
template <class Args, class Kwargs>
c10::IValue CallForeign(int64_t slot, const Args& args, const Kwargs& kwargs) {
  ForeignHost* entry;
  {
    std::lock_guard<std::mutex> lock(foreign_host_mutex);
    entry = ForeignHosts().at(slot);
  }
  PythonArguments context;
  py::tuple values(entry->prefix.size() + args.size());
  for (size_t i = 0; i < entry->prefix.size(); ++i) values[i] = entry->prefix[i];
  for (size_t i = 0; i < args.size(); ++i)
    values[entry->prefix.size() + i] = context.Convert(args.get(i));
  py::dict keywords;
  for (const auto& item : kwargs)
    keywords[py::str(std::string(item.key()))] = context.Convert(item.value());
  py::object result;
  try {
    result = entry->function(*values, **keywords);
  } catch (...) {
    auto error = std::current_exception();
    try {
      context.Sync();
    } catch (...) {
      PyErr_Clear();
    }
    std::rethrow_exception(error);
  }
  auto value = context.Restore(result);
  context.Sync();
  return value;
}
c10::IValue CallForeignPacked(int64_t slot, flagos::native_host::Arguments args,
                              flagos::native_host::Keywords kwargs) {
  py::gil_scoped_acquire gil;
  return CallForeign(slot, args, kwargs);
}
at::Tensor CallForeignTensor(int64_t slot, const c10::List<c10::IValue>& args,
                             const c10::Dict<std::string, c10::IValue>& kwargs) {
  py::gil_scoped_acquire gil;
  auto result = CallForeign(slot, args, kwargs);
  TORCH_CHECK(result.isTensor(), "host helper must return a tensor");
  return result.toTensor();
}
void LaunchPointwiseNative(int64_t slot, int64_t rank, const c10::List<c10::IValue>& args,
                           const c10::List<at::Tensor>& outputs) {
  NativePointwiseSlot* entry;
  {
    std::lock_guard<std::mutex> lock(pointwise_slot_mutex);
    entry = NativePointwiseSlots().at(slot);
  }
  auto* schedule = PointwiseForRank(entry, rank);
  schedule->CallNative(args, outputs);
}

void LaunchPointwisePacked(int64_t slot, int64_t rank, flagos::native_host::Arguments args,
                           flagos::native_host::TensorValues outputs) {
  NativePointwiseSlot* entry;
  {
    std::lock_guard<std::mutex> lock(pointwise_slot_mutex);
    entry = NativePointwiseSlots().at(slot);
  }
  PointwiseForRank(entry, rank)->CallNative(args, outputs);
}

template <class Args, class Kwargs>
c10::SmallVector<at::Tensor, 4> ApplyPointwiseImpl(int64_t slot, const Args& args,
                                                   const Kwargs& kwargs) {
  NativePointwiseSlot* entry;
  {
    std::lock_guard<std::mutex> lock(pointwise_slot_mutex);
    entry = NativePointwiseSlots().at(slot);
  }
  auto prepared = entry->spec->PrepareNative(args, kwargs);
  if (prepared) {
    auto* schedule = PointwiseForRank(entry, prepared->shape.size());
    schedule->CallNative(args, flagos::native_host::TensorValues{prepared->outputs},
                         at::IntArrayRef(prepared->shape), prepared->collapse);
    return std::move(prepared->outputs);
  }
  py::gil_scoped_acquire gil;
  py::tuple values(args.size());
  for (size_t i = 0; i < args.size(); ++i) values[i] = torch::jit::toPyObject(args.get(i));
  py::dict keywords;
  for (const auto& item : kwargs)
    keywords[py::str(std::string(item.key()))] = torch::jit::toPyObject(item.value());
  auto result = entry->owner(*values, **keywords);
  if (THPVariable_Check(result.ptr())) return {THPVariable_Unpack(result.ptr())};
  c10::SmallVector<at::Tensor, 4> tensors;
  for (auto value : py::cast<py::sequence>(result)) {
    TORCH_CHECK(THPVariable_Check(value.ptr()), "pointwise output must be a tensor");
    tensors.push_back(THPVariable_Unpack(value.ptr()));
  }
  return tensors;
}

std::vector<at::Tensor> ApplyPointwiseNative(int64_t slot, const c10::List<c10::IValue>& args,
                                             const c10::Dict<std::string, c10::IValue>& kwargs) {
  auto outputs = ApplyPointwiseImpl(slot, args, kwargs);
  return {outputs.begin(), outputs.end()};
}
std::vector<at::Tensor> ApplyPointwisePacked(int64_t slot, flagos::native_host::Arguments args,
                                             flagos::native_host::Keywords kwargs) {
  auto outputs = ApplyPointwiseImpl(slot, args, kwargs);
  return {outputs.begin(), outputs.end()};
}
at::Tensor ApplyPointwiseAt(int64_t slot, flagos::native_host::Arguments args,
                            flagos::native_host::Keywords kwargs, int64_t index) {
  auto outputs = ApplyPointwiseImpl(slot, args, kwargs);
  if (index < 0) index += outputs.size();
  TORCH_CHECK_INDEX(index >= 0 && index < int64_t(outputs.size()),
                    "pointwise output index out of range");
  return std::move(outputs[index]);
}

struct ScriptKernel {
  py::object entry;
  KernelCache* cache;
};
std::vector<ScriptKernel*>& ScriptKernels() {
  static auto* slots = new std::vector<ScriptKernel*>();
  return *slots;
}
std::mutex script_kernel_mutex;
std::optional<c10::Device> KernelDevice(const c10::IValue& value) {
  if (value.isTensor()) {
    const auto& tensor = value.toTensor();
    if (tensor.defined() && tensor.device().is_privateuseone()) return tensor.device();
  } else if (value.isTuple()) {
    for (const auto& item : value.toTupleRef().elements())
      if (auto found = KernelDevice(item)) return found;
  } else if (value.isList()) {
    for (const auto& item : value.toList())
      if (auto found = KernelDevice(item)) return found;
  } else if (value.isGenericDict()) {
    for (const auto& item : value.toGenericDict())
      if (auto found = KernelDevice(item.value())) return found;
  }
  return std::nullopt;
}
template <class Args, class Kwargs>
void LaunchOriginalKernel(ScriptKernel* entry, const Args& args, const Kwargs& kwargs,
                          const std::vector<int64_t>& grid, int64_t grid_function, bool packed) {
  py::gil_scoped_acquire gil;
  PythonArguments context;
  py::tuple values(args.size());
  for (size_t i = 0; i < args.size(); ++i) values[i] = context.Convert(args.get(i));
  py::dict keywords;
  for (const auto& item : kwargs)
    keywords[py::str(std::string(item.key()))] = context.Convert(item.value());
  if (grid_function) {
    keywords["grid"] = py::cpp_function([grid_function, grid, packed](py::dict meta) {
      c10::Dict<std::string, int64_t> constants;
      for (auto item : meta)
        if (PyLong_Check(item.second.ptr()))
          constants.insert(py::cast<std::string>(item.first), py::cast<int64_t>(item.second));
      if (packed) {
        flagos::native_host::GridMetadata metadata{&constants, nullptr, {}, {}};
        using Grid = std::vector<int64_t> (*)(const flagos::native_host::GridMetadata&,
                                              std::vector<int64_t>);
        return py::tuple(py::cast(reinterpret_cast<Grid>(grid_function)(metadata, grid)));
      }
      HostGridMeta metadata{&constants, nullptr, nullptr, nullptr};
      using Grid = std::vector<int64_t> (*)(const HostGridMeta&, std::vector<int64_t>);
      return py::tuple(py::cast(reinterpret_cast<Grid>(grid_function)(metadata, grid)));
    });
  } else
    keywords["grid"] = py::tuple(py::cast(grid));
  try {
    entry->entry.attr("run")(*values, **keywords);
  } catch (...) {
    auto error = std::current_exception();
    try {
      context.Sync();
    } catch (...) {
      PyErr_Clear();
    }
    std::rethrow_exception(error);
  }
  context.Sync();
}

void LaunchScriptKernelImpl(int64_t slot, const c10::List<c10::IValue>& args,
                            const c10::Dict<std::string, c10::IValue>& kwargs,
                            const std::vector<int64_t>& grid, int64_t grid_function) {
  ScriptKernel* entry;
  {
    std::lock_guard<std::mutex> lock(script_kernel_mutex);
    entry = ScriptKernels().at(slot);
  }
  int device = -1;
  std::optional<CurrentDeviceScope> guard;
  std::optional<c10::Device> target;
  for (const auto& item : args)
    if ((target = KernelDevice(item))) break;
  if (!target)
    for (const auto& item : kwargs)
      if ((target = KernelDevice(item.value()))) break;
  if (target) {
    guard.emplace(*target);
    device = target->index();
  }
  if (device >= 0 && entry->cache->RunNative(args, kwargs, grid, device, grid_function)) return;
  LaunchOriginalKernel(entry, args, kwargs, grid, grid_function, false);
}

void LaunchScriptKernelPacked(int64_t slot, flagos::native_host::Arguments args,
                              flagos::native_host::Keywords kwargs,
                              const std::vector<int64_t>& grid, int64_t grid_function) {
  ScriptKernel* entry;
  {
    std::lock_guard<std::mutex> lock(script_kernel_mutex);
    entry = ScriptKernels().at(slot);
  }
  std::optional<c10::Device> target;
  for (const auto& item : args)
    if ((target = KernelDevice(item))) break;
  if (!target)
    for (const auto& item : kwargs)
      if ((target = KernelDevice(item.value()))) break;
  std::optional<CurrentDeviceScope> guard;
  if (target) {
    guard.emplace(*target);
    if (entry->cache->RunNative(args, kwargs, grid, target->index(), grid_function)) return;
  }
  LaunchOriginalKernel(entry, args, kwargs, grid, grid_function, true);
}

TORCH_LIBRARY_FRAGMENT(flagos_source_native, m) {
  m.def(torch::schema("pointwise_apply(int slot, Any[] args, Dict(str, Any) kwargs) -> Tensor[]",
                      c10::AliasAnalysisKind::CONSERVATIVE),
        &ApplyPointwiseNative);
  m.def(torch::schema("foreign_value(int slot, Any[] args, Dict(str, Any) kwargs) -> Any",
                      c10::AliasAnalysisKind::CONSERVATIVE),
        [](int64_t slot, const c10::List<c10::IValue>& args,
           const c10::Dict<std::string, c10::IValue>& kwargs) -> c10::IValue {
          py::gil_scoped_acquire gil;
          return CallForeign(slot, args, kwargs);
        });
  m.def("tuple_value(Any values) -> Any", [](const c10::IValue& value) -> c10::IValue {
    return c10::ivalue::Tuple::create(value.toList().vec());
  });
  m.def("dtype_str(int dtype) -> str", [](int64_t dtype) { return dtype_names.at(dtype); });
  m.def(torch::schema("foreign_bool(int slot, Any[] args, Dict(str, Any) kwargs) -> bool",
                      c10::AliasAnalysisKind::CONSERVATIVE),
        [](int64_t slot, const c10::List<c10::IValue>& args,
           const c10::Dict<std::string, c10::IValue>& kwargs) {
          py::gil_scoped_acquire gil;
          return CallForeign(slot, args, kwargs).toBool();
        });
  m.def(torch::schema("foreign_tensor(int slot, Any[] args, Dict(str, Any) kwargs) -> Tensor",
                      c10::AliasAnalysisKind::CONSERVATIVE),
        &CallForeignTensor);
  m.def("is_alias(Tensor a, Tensor b) -> bool",
        [](const at::Tensor& a, const at::Tensor& b) { return a.is_alias_of(b); });
  m.def("broadcast_shapes(int[][] shapes) -> int[]",
        [](const std::vector<std::vector<int64_t>>& shapes) {
          size_t rank = 0;
          for (const auto& shape : shapes) rank = std::max(rank, shape.size());
          std::vector<int64_t> result(rank, 1);
          for (const auto& shape : shapes)
            for (size_t i = 0; i < shape.size(); ++i) {
              auto& dst = result[rank - shape.size() + i];
              auto src = shape[i];
              TORCH_CHECK(dst == src || dst == 1 || src == 1, "incompatible broadcast shapes");
              if (dst == 1) dst = src;
            }
          return result;
        });
  m.def(torch::schema("pointwise_launch(int slot, int rank, Any[] args, Tensor[] outputs) -> ()",
                      c10::AliasAnalysisKind::CONSERVATIVE),
        &LaunchPointwiseNative);
  m.def(torch::schema("launch(int slot, Any[] args, Dict(str, Any) kwargs, int[] grid) -> ()",
                      c10::AliasAnalysisKind::CONSERVATIVE),
        [](int64_t slot, const c10::List<c10::IValue>& args,
           const c10::Dict<std::string, c10::IValue>& kwargs, const std::vector<int64_t>& grid) {
          LaunchScriptKernelImpl(slot, args, kwargs, grid, 0);
        });
  m.def(torch::schema("launch_grid(int slot, Any[] args, Dict(str, Any) kwargs, int grid_function, "
                      "int[] captures) -> ()",
                      c10::AliasAnalysisKind::CONSERVATIVE),
        [](int64_t slot, const c10::List<c10::IValue>& args,
           const c10::Dict<std::string, c10::IValue>& kwargs, int64_t grid_function,
           const std::vector<int64_t>& captures) {
          LaunchScriptKernelImpl(slot, args, kwargs, captures, grid_function);
        });
}

}  // namespace

PYBIND11_MODULE(_launcher, module) {
  backend_api = &backend::Validate(backend::CompiledBackend());
  module.def("backend_contract", [] {
    const auto& api = Backend();
    return py::dict("abi_version"_a = api.abi_version, "vendor"_a = api.vendor,
                    "compiler_backend"_a = api.compiler_backend,
                    "capabilities"_a = api.capabilities,
                    "scratch_alignment"_a = api.scratch_alignment,
                    "slot_bytes"_a = backend::kSlotBytes,
                    "slot_alignment"_a = backend::kSlotAlignment,
                    "native_source_sha256"_a = FLAGOS_NATIVE_HOST_BACKEND_SHA256);
  });
  module.attr("foreign_packed_api") =
      py::capsule(reinterpret_cast<void*>(&CallForeignPacked), "flagos.native_host.foreign.v1");
  module.attr("pointwise_launch_api") = py::capsule(reinterpret_cast<void*>(&LaunchPointwisePacked),
                                                    "flagos.native_host.pointwise_launch.v1");
  module.attr("pointwise_at_api") =
      py::capsule(reinterpret_cast<void*>(&ApplyPointwiseAt), "flagos.native_host.pointwise_at.v1");
  module.attr("launch_packed_api") = py::capsule(reinterpret_cast<void*>(&LaunchScriptKernelPacked),
                                                 "flagos.native_host.launch.v2");
  module.attr("pointwise_packed_api") = py::capsule(reinterpret_cast<void*>(&ApplyPointwisePacked),
                                                    "flagos.native_host.pointwise.v2");
  module.attr("launch_api") =
      py::capsule(reinterpret_cast<void*>(&LaunchScriptKernelImpl), "flagos.native_host.launch.v1");
  module.attr("pointwise_api") = py::capsule(reinterpret_cast<void*>(&ApplyPointwiseNative),
                                             "flagos.native_host.pointwise.v1");
  dtype_names.resize(static_cast<size_t>(at::ScalarType::NumOptions));
  for (size_t i = 0; i < dtype_names.size(); ++i) {
    if (i == static_cast<size_t>(at::ScalarType::Undefined)) continue;
    auto* dtype = torch::getTHPDtype(static_cast<at::ScalarType>(i));
    if (dtype) dtype_names[i] = py::str(reinterpret_cast<PyObject*>(dtype));
  }
  module.def("configure_descriptor",
             [](py::object factory) { descriptor_factory = std::move(factory); });
  module.def("register_native_hook_v2", [](py::object function, uint64_t address) {
    TORCH_CHECK(address, "native hook address is null");
    NativeHooks().insert_or_assign(
        function.ptr(),
        HookRegistration{nullptr, reinterpret_cast<flagos::native_host::MetadataHook>(address),
                         function});
  });
  module.def("register_native_hook", [](py::object function, uint64_t address) {
    TORCH_CHECK(address, "native hook address is null");
    NativeHooks().insert_or_assign(
        function.ptr(), HookRegistration{reinterpret_cast<NativeHook>(address), nullptr, function});
  });
  module.def("register_foreign", [](py::object function, py::tuple prefix) {
    std::lock_guard<std::mutex> lock(foreign_host_mutex);
    auto index = ForeignHosts().size();
    ForeignHosts().push_back(new ForeignHost{function, prefix});
    return index;
  });
  module.def("register_pointwise", [](py::object owner) {
    std::lock_guard<std::mutex> lock(pointwise_slot_mutex);
    auto* entry = new NativePointwiseSlot();
    entry->owner = owner;
    auto provider =
        py::module_::import(py::cast<std::string>(py::type::of(owner).attr("__module__")).c_str());
    entry->spec = std::make_unique<PointwiseSpec>(owner.attr("fx"), provider.attr("type_promotion"),
                                                  owner.attr("config"));
    auto index = NativePointwiseSlots().size();
    NativePointwiseSlots().push_back(entry);
    return index;
  });
  module.def("register_script_kernel", [](py::object kernel) {
    std::lock_guard<std::mutex> lock(script_kernel_mutex);
    const auto id = ScriptKernels().size();
    ScriptKernels().push_back(
        new ScriptKernel{kernel, py::cast<KernelCache*>(kernel.attr("native_cache"))});
    return id;
  });
  auto buffer =
      py::class_<StridedBuffer>(module, "StridedBuffer")
          .def(py::init<py::object, py::object, py::object, py::object, int64_t>(), "base"_a,
               "shape"_a = py::none(), "strides"_a = py::none(), "dtype"_a = py::none(),
               "offset"_a = 0)
          .def_readonly("_base", &StridedBuffer::base)
          .def_readonly("dtype", &StridedBuffer::dtype)
          .def_readonly("device", &StridedBuffer::device)
          .def_readonly("offset", &StridedBuffer::offset)
          .def_readonly("_data_ptr", &StridedBuffer::pointer)
          .def_readonly("shape", &StridedBuffer::shape)
          .def_readonly("_strides", &StridedBuffer::strides)
          .def_property_readonly("ndim", [](const StridedBuffer& b) { return b.sizes.size(); })
          .def("stride", [](const StridedBuffer& b) { return b.strides; })
          .def("size", [](const StridedBuffer& b) { return b.shape; })
          .def("element_size", [](const StridedBuffer& b) { return b.dtype.attr("itemsize"); })
          .def("numel", [](const StridedBuffer& b) { return b.elements; })
          .def("dim", [](const StridedBuffer& b) { return b.sizes.size(); })
          .def("unwrap", [](const StridedBuffer& b) { return b.base; })
          .def("data_ptr", [](const StridedBuffer& b) { return b.pointer; })
          .def("untyped_storage",
               [](const StridedBuffer& b) { return b.base.attr("untyped_storage")(); })
          .def("clone", &StridedBuffer::Clone)
          .def("copy_", &StridedBuffer::Copy, py::return_value_policy::reference_internal);
  buffer_type = reinterpret_cast<PyTypeObject*>(buffer.ptr());
  module.def("configure_runtime", [](const std::string& path) {
    void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!library) throw std::runtime_error(dlerror());
    current_stream = reinterpret_cast<CurrentStream>(dlsym(library, "GetCurrentStreamForDevice"));
    if (!current_stream) throw std::runtime_error("missing FlagOS stream provider");
  });
  module.def("current_stream", &Stream);
  module.def("is_capturing", [](int device) {
    return IsCapturing(Stream(device));
  });
  py::class_<Launcher>(module, "Launcher")
      .def(py::init<py::object, py::object>())
      .def("__call__", &Launcher::Call)
      .def_property_readonly("abi", &Launcher::Abi);
  py::class_<KernelCache>(module, "KernelCache")
      .def(py::init<py::tuple, py::dict, bool>())
      .def("run", &KernelCache::Run)
      .def("prepare", &KernelCache::Prepare);
  py::class_<PointwiseSpec>(module, "PointwiseSpec")
      .def(py::init<py::object, py::object, py::object>())
      .def("prepare", &PointwiseSpec::Prepare, "args"_a, "kwargs"_a, "device"_a,
           "owner"_a = py::none());
  py::class_<PointwiseSchedule>(module, "PointwiseSchedule")
      .def(py::init<py::object, size_t, py::object, py::tuple, py::tuple>())
      .def("__call__", &PointwiseSchedule::Call);
}
