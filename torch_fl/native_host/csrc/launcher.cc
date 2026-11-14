// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// The launch ABI is data. No signature-specific C/C++ is generated at runtime.
#include <Python.h>
#include <cuda.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

using CurrentStream = void* (*)(int);
CurrentStream current_stream = nullptr;

uint64_t Stream(int device) {
  if (!current_stream) throw std::runtime_error("FlagOS stream provider is not initialized");
  return reinterpret_cast<uint64_t>(current_stream(device));
}

void Check(CUresult status) {
  if (status != CUDA_SUCCESS) {
    const char* error = nullptr;
    cuGetErrorString(status, &error);
    throw std::runtime_error(error ? error : "CUDA driver operation failed");
  }
}

uint64_t Pointer(py::handle value) {
  if (value.is_none()) return 0;
  if (PyLong_Check(value.ptr())) return py::cast<uint64_t>(value);
  if (py::hasattr(value, "device") &&
      py::cast<std::string>(value.attr("device").attr("type")) == "cpu")
    throw std::invalid_argument("CPU tensor cannot be passed to a FlagGems CUDA kernel");
  return py::cast<uint64_t>(value.attr("data_ptr")());
}

struct alignas(128) Slot {
  std::array<unsigned char, 128> bytes{};
  template <typename T> void Set(T value) { std::memcpy(bytes.data(), &value, sizeof(T)); }
};

struct Argument {
  std::vector<size_t> path;
  std::string type;
  py::object descriptor_metadata;
};

class Launcher {
 public:
  Launcher(py::object source, py::object metadata) : metadata_(metadata) {
    const auto signatures = py::cast<py::dict>(source.attr("signature"));
    py::object descriptors = py::getattr(metadata, "tensordesc_meta", py::none());
    size_t index = 0, descriptor = 0;
    for (auto item : signatures) {
      Add(item.second, {index++}, descriptors, descriptor);
    }
    if (!descriptors.is_none() && descriptor != py::len(descriptors))
      throw std::invalid_argument("Triton descriptor metadata arity mismatch");
    num_args_ = index;
    num_warps_ = py::cast<unsigned>(metadata.attr("num_warps"));
    num_ctas_ = py::cast<unsigned>(py::getattr(metadata, "num_ctas", py::int_(1)));
    shared_ = py::cast<unsigned>(metadata.attr("shared"));
    cluster_ = py::cast<std::array<unsigned, 3>>(
        py::getattr(metadata, "cluster_dims", py::make_tuple(1, 1, 1)));
    cooperative_ = py::cast<bool>(py::getattr(metadata, "launch_cooperative_grid", py::bool_(false)));
    pdl_ = py::cast<bool>(py::getattr(metadata, "launch_pdl", py::bool_(false)));
    global_size_ = py::cast<uint64_t>(metadata.attr("global_scratch_size"));
    global_align_ = py::cast<uint64_t>(metadata.attr("global_scratch_align"));
    profile_size_ = py::cast<uint64_t>(metadata.attr("profile_scratch_size"));
    profile_align_ = py::cast<uint64_t>(metadata.attr("profile_scratch_align"));
  }

  // Matches CompiledKernel.run; callers pass metadata and optional profiler hooks.
  void Call(unsigned x, unsigned y, unsigned z, uint64_t stream, uint64_t function,
            py::object packed, py::object launch_metadata, py::object enter,
            py::object leave, py::args args) const {
    if (args.size() != num_args_)
      throw std::invalid_argument("Triton launch argument arity mismatch");
    if (!enter.is_none()) enter(launch_metadata);
    std::vector<Slot> slots;
    slots.reserve(arguments_.size() * 22 + 2);
    for (const auto& argument : arguments_) {
      PyObject* value = args.ptr();
      for (size_t index : argument.path) {
        if (PyTuple_Check(value)) value = PyTuple_GetItem(value, index);
        else if (PyList_Check(value)) value = PyList_GetItem(value, index);
        else throw std::invalid_argument("nested Triton argument must be a tuple or list");
        if (!value) throw py::error_already_set();
      }
      Pack(argument, py::handle(value), slots);
    }

    const uint64_t grid = CheckedProduct(CheckedProduct(x, y), z);
    py::object global = Scratch(CheckedProduct(CheckedProduct(grid, num_ctas_), global_size_), global_align_);
    py::object profile = Scratch(CheckedProduct(CheckedProduct(grid, num_ctas_), profile_size_), profile_align_);
    const uint64_t global_ptr = Pointer(global), profile_ptr = Pointer(profile);
    slots.emplace_back(); slots.back().Set(global_ptr);
    slots.emplace_back(); slots.back().Set(profile_ptr);
    std::vector<void*> pointers;
    pointers.reserve(slots.size());
    for (auto& slot : slots) pointers.push_back(slot.bytes.data());
    if (grid) {
      auto handle = reinterpret_cast<CUfunction>(function);
      auto cuda_stream = reinterpret_cast<CUstream>(stream);
      if (cooperative_ && global_ptr)
        Check(cuMemsetD8Async(global_ptr, 0, CheckedProduct(CheckedProduct(grid, num_ctas_), global_size_), cuda_stream));
      CUlaunchAttribute attributes[4]{};
      unsigned count = 0;
      if (pdl_) {
        attributes[count].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attributes[count++].value.programmaticStreamSerializationAllowed = 1;
      }
      if (cooperative_) {
        attributes[count].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
        attributes[count++].value.cooperative = 1;
      }
      const uint64_t cluster_size = CheckedProduct(CheckedProduct(cluster_[0], cluster_[1]), cluster_[2]);
      if (cluster_size != 1) {
        attributes[count].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attributes[count].value.clusterDim.x = cluster_[0];
        attributes[count].value.clusterDim.y = cluster_[1];
        attributes[count++].value.clusterDim.z = cluster_[2];
        attributes[count].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
        attributes[count++].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      }
      if (num_ctas_ == 16 || cluster_size == 16)
        Check(cuFuncSetAttribute(handle, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
      CUlaunchConfig config{};
      config.gridDimX = GridSize(CheckedProduct(CheckedProduct(x, num_ctas_), cluster_[0]));
      config.gridDimY = GridSize(CheckedProduct(y, cluster_[1]));
      config.gridDimZ = GridSize(CheckedProduct(z, cluster_[2]));
      config.blockDimX = GridSize(CheckedProduct(num_warps_, 32));
      config.blockDimY = config.blockDimZ = 1;
      config.sharedMemBytes = shared_;
      config.hStream = cuda_stream;
      config.attrs = attributes;
      config.numAttrs = count;
      Check(cuLaunchKernelEx(&config, handle, pointers.data(), nullptr));
    }
    if (!leave.is_none()) leave(launch_metadata);
  }

  py::list Abi() const {
    py::list result;
    for (const auto& a : arguments_) result.append(py::make_tuple(a.path, a.type));
    return result;
  }

 private:
  static uint64_t CheckedProduct(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
      throw std::overflow_error("Triton launch size overflow");
    return a * b;
  }
  static unsigned GridSize(uint64_t value) {
    if (value > std::numeric_limits<unsigned>::max())
      throw std::overflow_error("Triton grid dimension overflow");
    return static_cast<unsigned>(value);
  }
  static py::object Scratch(uint64_t size, uint64_t alignment) {
    if (!size) return py::none();
    if (alignment > 256) throw std::invalid_argument("scratch alignment exceeds allocator guarantee");
    auto torch = py::module_::import("torch");
    return torch.attr("empty")(py::make_tuple(size), "dtype"_a=torch.attr("uint8"), "device"_a="flagos");
  }
  void Add(py::handle type, std::vector<size_t> path, py::object descriptors, size_t& descriptor) {
    if (py::isinstance<py::tuple>(type)) {
      size_t i = 0;
      for (auto child : py::reinterpret_borrow<py::tuple>(type)) {
        auto next = path; next.push_back(i++);
        Add(child, std::move(next), descriptors, descriptor);
      }
      return;
    }
    const std::string name = py::cast<std::string>(type);
    if (name == "constexpr") return;
    Argument argument{std::move(path), name, py::none()};
    if (name.rfind("tensordesc", 0) == 0) {
      if (!descriptors.is_none()) argument.descriptor_metadata = descriptors[py::int_(descriptor)];
      ++descriptor;
    }
    arguments_.push_back(std::move(argument));
  }
  static void Pack(const Argument& a, py::handle value, std::vector<Slot>& slots) {
    if (a.type.rfind("tensordesc", 0) == 0) {
      Descriptor(a, value, slots);
      return;
    }
    slots.emplace_back();
    Slot& slot = slots.back();
    if (a.type[0] == '*') { slot.Set(Pointer(value)); return; }
    if (a.type == "fp64") { slot.Set(py::cast<double>(value)); return; }
    if (a.type == "fp32" || a.type == "f32") { slot.Set(py::cast<float>(value)); return; }
    if (a.type == "fp16") {
      if (PyFloat_Pack2(py::cast<double>(value), reinterpret_cast<char*>(slot.bytes.data()), 1) < 0)
        throw py::error_already_set();
      return;
    }
    if (a.type == "bf16") {
      const float f = py::cast<float>(value); uint32_t bits;
      std::memcpy(&bits, &f, sizeof(bits)); slot.Set(static_cast<uint16_t>(bits >> 16)); return;
    }
    if (a.type == "i1" || a.type == "u1") { slot.Set(static_cast<uint8_t>(py::cast<bool>(value))); return; }
    if (a.type == "i8") { slot.Set(py::cast<int8_t>(value)); return; }
    if (a.type == "i16") { slot.Set(py::cast<int16_t>(value)); return; }
    if (a.type == "i32") { slot.Set(py::cast<int32_t>(value)); return; }
    if (a.type == "i64") { slot.Set(py::cast<int64_t>(value)); return; }
    if (a.type == "u8") { slot.Set(py::cast<uint8_t>(value)); return; }
    if (a.type == "u16") { slot.Set(py::cast<uint16_t>(value)); return; }
    if (a.type == "u32") { slot.Set(py::cast<uint32_t>(value)); return; }
    if (a.type == "u64") { slot.Set(py::cast<uint64_t>(value)); return; }
    throw std::invalid_argument("unsupported Triton parameter ABI: " + a.type);
  }
  static void Descriptor(const Argument& a, py::handle value, std::vector<Slot>& slots) {
    auto shape = py::cast<std::vector<uint64_t>>(value.attr("shape"));
    const auto strides = py::cast<std::vector<uint64_t>>(value.attr("strides"));
    const bool nan = py::cast<std::string>(value.attr("padding")) == "nan";
    const size_t rank = shape.size();
    if (rank < 1 || rank > 5 || strides.size() != rank || strides.back() != 1)
      throw std::invalid_argument("invalid Triton tensor descriptor dimensions");
    if (a.descriptor_metadata.is_none()) {
      slots.emplace_back(); slots.back().Set(Pointer(value.attr("base")));
      for (uint64_t s : shape) { slots.emplace_back(); slots.back().Set(s); }
      for (uint64_t s : strides) { slots.emplace_back(); slots.back().Set(s); }
      slots.emplace_back(); slots.back().Set(static_cast<uint8_t>(nan));
    } else {
      auto meta = py::cast<py::dict>(a.descriptor_metadata);
      if (py::cast<bool>(meta["fp4_padded"])) shape.back() = CheckedProduct(shape.back(), 2);
      auto block = py::cast<std::vector<uint32_t>>(meta["block_size"]);
      if (block.size() != rank) throw std::invalid_argument("TMA block rank mismatch");
      const auto elem_size = py::cast<uint64_t>(meta["elem_size"]);
      int elem_type = py::cast<int>(meta["elem_type"]);
      if (elem_type == 8) elem_type = 10;
      else if (elem_type == 9) elem_type = 8;
      else if (elem_type == 10) elem_type = 9;
      uint64_t reversed_shape[5]{}, reversed_stride[5]{};
      uint32_t reversed_block[5]{}, element_stride[5]{1, 1, 1, 1, 1};
      for (size_t i = 0; i < rank; ++i) {
        reversed_shape[rank - i - 1] = shape[i];
        reversed_block[rank - i - 1] = block[i];
        if (i + 1 < rank) reversed_stride[rank - i - 2] = CheckedProduct(strides[i], elem_size);
      }
      slots.emplace_back();
      Check(cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap*>(slots.back().bytes.data()),
          static_cast<CUtensorMapDataType>(elem_type), rank,
          reinterpret_cast<void*>(Pointer(value.attr("base"))), reversed_shape, reversed_stride,
          reversed_block, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
          static_cast<CUtensorMapSwizzle>(py::cast<int>(meta["swizzle"])),
          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
          nan ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }
    for (uint64_t s : shape) {
      if (s > INT32_MAX) throw std::overflow_error("descriptor dimension exceeds kernel ABI");
      slots.emplace_back(); slots.back().Set(static_cast<int32_t>(s));
    }
    for (uint64_t s : strides) { slots.emplace_back(); slots.back().Set(s); }
  }

  std::vector<Argument> arguments_;
  py::object metadata_;
  size_t num_args_;
  unsigned num_warps_, num_ctas_, shared_;
  std::array<unsigned, 3> cluster_;
  bool cooperative_, pdl_;
  uint64_t global_size_, global_align_, profile_size_, profile_align_;
};

}  // namespace

PYBIND11_MODULE(_launcher, module) {
  module.def("configure_runtime", [](const std::string& path) {
    void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!library) throw std::runtime_error(dlerror());
    current_stream = reinterpret_cast<CurrentStream>(dlsym(library, "GetCurrentStreamForDevice"));
    if (!current_stream) throw std::runtime_error("missing FlagOS stream provider");
  });
  module.def("current_stream", &Stream);
  module.def("is_capturing", [](int device) {
    CUstreamCaptureStatus status;
    Check(cuStreamIsCapturing(reinterpret_cast<CUstream>(Stream(device)), &status));
    return status != CU_STREAM_CAPTURE_STATUS_NONE;
  });
  py::class_<Launcher>(module, "Launcher")
      .def(py::init<py::object, py::object>())
      .def("__call__", &Launcher::Call)
      .def_property_readonly("abi", &Launcher::Abi);
}
