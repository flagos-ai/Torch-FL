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

#include <ATen/Context.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include <memory>

#ifdef USE_MUSA
#include <musa_runtime.h>
#include <runtime/accelerator/musa/musa_stream.h>
#endif

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/Generator.h>

#include <runtime/functions.h>
#include <runtime/generator.h>
#include <macros.h>
#include <runtime/allocator/caching_device_allocator.h>

namespace {

// Forward declarations
at::Tensor flagos_to_cuda_view_impl(const at::Tensor& self);
at::Tensor cuda_to_flagos_view_impl(const at::Tensor& self, int device_index);


// Zero-copy view conversion: flagos -> CUDA
// This version keeps a reference to the original flagos tensor to prevent memory from being freed
at::Tensor flagos_to_cuda_view_impl(const at::Tensor& self) {
  TORCH_CHECK(self.is_privateuseone(), "Expected flagos tensor, got ", self.device());

  int device_index = self.device().index();
  if (device_index < 0) device_index = 0;

  // Get storage info from the original tensor directly (no contiguous call to avoid recursion)
  auto storage_impl = self.storage().unsafeGetStorageImpl();
  auto data_ptr = storage_impl->data_ptr().get();
  auto nbytes = storage_impl->nbytes();

  // Keep a copy of the original tensor to prevent its storage from being freed
  auto flagos_tensor_holder = std::make_shared<at::Tensor>(self);

  // Create CUDA storage sharing the same data pointer
  // The deleter captures the shared_ptr to keep the flagos tensor alive
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      nbytes,
      c10::DataPtr(
          data_ptr,
          new std::shared_ptr<at::Tensor>(flagos_tensor_holder),
          [](void* ctx) {
            // Release the reference to the flagos tensor
            delete static_cast<std::shared_ptr<at::Tensor>*>(ctx);
          },
          c10::Device(c10::kCUDA, device_index)
      ),
      nullptr,
      false
  );

  // Get the correct TypeMeta from the source tensor
  auto dtype = self.scalar_type();
  auto meta = c10::scalarTypeToTypeMeta(dtype);

  // Use at::detail::make_tensor_base to create a tensor directly with storage
  // This avoids the set_() call which has issues on flagos device
  auto cuda_tensor = at::detail::make_tensor_base<c10::TensorImpl>(
      std::move(storage),
      c10::DispatchKeySet(c10::DispatchKey::CUDA),
      meta
  );

  // Set the correct offset, sizes and strides
  cuda_tensor.unsafeGetTensorImpl()->set_storage_offset(self.storage_offset());
  cuda_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
      self.sizes().vec(), self.strides().vec());

  return cuda_tensor;
}

// Zero-copy view conversion: CUDA -> flagos
// This version keeps a reference to the original CUDA tensor to prevent memory from being freed
at::Tensor cuda_to_flagos_view_impl(const at::Tensor& self, int device_index) {
  TORCH_CHECK(self.is_cuda(), "Expected CUDA tensor, got ", self.device());

  if (device_index < 0) {
    device_index = self.device().index();
    if (device_index < 0) device_index = 0;
  }

  // Get storage info from the original tensor directly (no contiguous call)
  auto storage_impl = self.storage().unsafeGetStorageImpl();
  auto data_ptr = storage_impl->data_ptr().get();
  auto nbytes = storage_impl->nbytes();

  // Keep a copy of the original tensor to prevent its storage from being freed
  // When the flagos tensor is destroyed, this shared_ptr will be decremented,
  // and only when it reaches 0 will the CUDA memory be freed
  auto cuda_tensor_holder = std::make_shared<at::Tensor>(self);

  // Create flagos storage sharing the same data pointer
  // The deleter captures the shared_ptr to keep the CUDA tensor alive
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      nbytes,
      c10::DataPtr(
          data_ptr,
          new std::shared_ptr<at::Tensor>(cuda_tensor_holder),
          [](void* ctx) {
            // Release the reference to the CUDA tensor
            delete static_cast<std::shared_ptr<at::Tensor>*>(ctx);
          },
          c10::Device(c10::kPrivateUse1, device_index)
      ),
      nullptr,
      false
  );

  // Get the correct TypeMeta from the source tensor
  auto dtype = self.scalar_type();
  auto meta = c10::scalarTypeToTypeMeta(dtype);

  // Use at::detail::make_tensor_base to create a tensor directly with storage
  // This avoids the set_() call which has issues on flagos device
  auto flagos_tensor = at::detail::make_tensor_base<c10::TensorImpl>(
      std::move(storage),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      meta
  );

  // Set the correct offset, sizes and strides
  flagos_tensor.unsafeGetTensorImpl()->set_storage_offset(self.storage_offset());
  flagos_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
      self.sizes().vec(), self.strides().vec());

  // Handle autograd: share the autograd metadata from the CUDA tensor
  // Since the tensors share the same memory, gradients will flow correctly
  auto* src_autograd_meta = torch::autograd::impl::get_autograd_meta(self);
  if (src_autograd_meta && src_autograd_meta->grad_fn_) {
    // Share the same grad_fn - gradients computed on CUDA will be accumulated
    // on the CUDA tensor, which shares memory with the flagos tensor
    auto new_meta = std::make_unique<torch::autograd::AutogradMeta>(
        flagos_tensor.unsafeGetTensorImpl(),
        true  // requires_grad
    );
    new_meta->grad_fn_ = src_autograd_meta->grad_fn_;
    new_meta->output_nr_ = src_autograd_meta->output_nr_;

    flagos_tensor.unsafeGetTensorImpl()->set_autograd_meta(std::move(new_meta));
  } else if (self.requires_grad()) {
    // Leaf tensor with requires_grad=True
    auto new_meta = std::make_unique<torch::autograd::AutogradMeta>(
        flagos_tensor.unsafeGetTensorImpl(),
        true  // requires_grad
    );
    flagos_tensor.unsafeGetTensorImpl()->set_autograd_meta(std::move(new_meta));
  }

  return flagos_tensor;
}

} // anonymous namespace

static PyObject* _initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _getDefaultGenerator(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_get_default_generator expects an int, but got ",
      THPUtils_typename(arg));
  auto idx = static_cast<int>(THPUtils_unpackLong(arg));

  return THPGenerator_initDefaultGenerator(
      at::globalContext().defaultGenerator(
          c10::Device(c10::DeviceType::PrivateUse1, idx)));

  END_HANDLE_TH_ERRORS
}

static PyObject* _reserveRngSeed(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  int device_index;
  PyObject* generator_obj = Py_None;
  if (!PyArg_ParseTuple(args, "i|O", &device_index, &generator_obj)) {
    return nullptr;
  }

  std::optional<at::Generator> generator;
  if (generator_obj != Py_None) {
    TORCH_CHECK(
        THPGenerator_Check(generator_obj),
        "_reserve_rng_seed expects a torch.Generator or None, but got ",
        THPUtils_typename(generator_obj));
    generator = THPGenerator_Unwrap(generator_obj);
  }
  auto seed = c10::flagos::ReserveSeed(generator, device_index);
  return PyLong_FromUnsignedLongLong(seed);
  END_HANDLE_TH_ERRORS
}

static PyObject* _getMusaDeviceProperties(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
#ifdef USE_MUSA
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_get_musa_device_properties expects an int, but got ",
      THPUtils_typename(arg));
  auto idx = static_cast<int>(THPUtils_unpackLong(arg));
  TORCH_CHECK(
      idx >= 0 && idx < c10::flagos::DeviceCount(),
      "MUSA device index ", idx, " is out of range");

  musaDeviceProp prop{};
  auto err = musaGetDeviceProperties(&prop, idx);
  TORCH_CHECK(
      err == musaSuccess,
      "musaGetDeviceProperties failed for device ",
      idx,
      ": ",
      musaGetErrorString(err));

  // The trailing fields are what Inductor's Triton heuristics and its
  // benchmarking buffer read. Report the SDK's own values rather than letting
  // Python invent them: a fabricated zero L2 size produces an empty flush
  // buffer, which mudnn's Fill rejects with NOT_SUPPORTED.
  return Py_BuildValue(
      "{s:s,s:K,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:K}",
      "name",
      prop.name,
      "total_memory",
      static_cast<unsigned long long>(prop.totalGlobalMem),
      "multi_processor_count",
      prop.multiProcessorCount,
      "major",
      prop.major,
      "minor",
      prop.minor,
      "warp_size",
      prop.warpSize,
      "l2_cache_size",
      prop.l2CacheSize,
      "max_threads_per_multi_processor",
      prop.maxThreadsPerMultiProcessor,
      "regs_per_block",
      prop.regsPerBlock,
      "shared_memory_per_multiprocessor",
      static_cast<unsigned long long>(prop.sharedMemPerMultiprocessor));
#else
  TORCH_CHECK(false, "MUSA device properties are unavailable in this build");
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject* _getMusaCurrentRawStream(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
#ifdef USE_MUSA
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_get_musa_current_raw_stream expects an int, but got ",
      THPUtils_typename(arg));
  auto idx = static_cast<int>(THPUtils_unpackLong(arg));
  auto current_device = c10::flagos::CurrentDevice();
  TORCH_CHECK(
      idx == current_device,
      "MUSA stream requested for device ",
      idx,
      ", but the current device is ",
      current_device);
  auto stream = at::native::flagos::musa::GetDefaultMusaStream();
  return PyLong_FromUnsignedLongLong(
      reinterpret_cast<unsigned long long>(stream));
#else
  TORCH_CHECK(false, "MUSA streams are unavailable in this build");
#endif
  END_HANDLE_TH_ERRORS
}

PyObject* _setDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to setDevice");
  auto device = THPUtils_unpackLong(arg);

  torch::utils::device_lazy_init(at::kPrivateUse1);
  c10::flagos::SetCurrentDevice(static_cast<c10::DeviceIndex>(device));

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _exchangeDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto current_device = c10::flagos::ExchangeDevice(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* _getDevice(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto device = static_cast<int32_t>(c10::flagos::CurrentDevice());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* _getDeviceCount(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(c10::flagos::DeviceCount());
  END_HANDLE_TH_ERRORS
}

PyObject* _synchronize(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  DeviceSynchronize();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _flagos_to_cuda_view(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPVariable_Check(arg), "Expected a tensor argument");
  auto tensor = THPVariable_Unpack(arg);
  auto result = flagos_to_cuda_view_impl(tensor);
  return THPVariable_Wrap(result);
  END_HANDLE_TH_ERRORS
}

PyObject* _cuda_to_flagos_view(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* tensor_obj;
  int device_index = -1;
  if (!PyArg_ParseTuple(args, "O|i", &tensor_obj, &device_index)) {
    return nullptr;
  }
  TORCH_CHECK(THPVariable_Check(tensor_obj), "Expected a tensor argument");
  auto tensor = THPVariable_Unpack(tensor_obj);
  auto result = cuda_to_flagos_view_impl(tensor, device_index);
  return THPVariable_Wrap(result);
  END_HANDLE_TH_ERRORS
}

PyObject* _flagos_identity_view(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPVariable_Check(arg), "Expected a tensor argument");
  // Identity view: return the tensor unchanged. This allows FlagCX's MUSA
  // adaptor to receive privateuseone tensors directly, bypassing the need for
  // a zero-copy flagos->musa storage view. FlagCX extracts data_ptr() and the
  // musaStream_t from the tensor without validating device().type(), so a
  // privateuseone tensor works as-is. Verified with FlagCX main branch
  // (flagcx/adaptor/device/musa_adaptor.cc + plugin/torch/flagcx/src/).
  Py_INCREF(arg);
  return arg;
  END_HANDLE_TH_ERRORS
}

// --- Caching Allocator APIs ---

PyObject* _empty_cache(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (c10::flagos::CachingDeviceAllocator::is_enabled()) {
    c10::flagos::GetCachingAllocator()->empty_cache();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _memory_stats(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  int device = THPUtils_unpackInt(arg);
  PyObject* dict = PyDict_New();
  if (c10::flagos::CachingDeviceAllocator::is_enabled()) {
    auto stats = c10::flagos::GetCachingAllocator()->get_stats(device);
    PyDict_SetItemString(dict, "allocated_bytes",
        PyLong_FromSize_t(stats.bytes_allocated));
    PyDict_SetItemString(dict, "reserved_bytes",
        PyLong_FromSize_t(stats.bytes_reserved));
    PyDict_SetItemString(dict, "peak_allocated_bytes",
        PyLong_FromSize_t(stats.peak_allocated));
    PyDict_SetItemString(dict, "peak_reserved_bytes",
        PyLong_FromSize_t(stats.peak_reserved));
    PyDict_SetItemString(dict, "num_alloc_calls",
        PyLong_FromSize_t(stats.num_alloc_calls));
    PyDict_SetItemString(dict, "num_free_calls",
        PyLong_FromSize_t(stats.num_free_calls));
    PyDict_SetItemString(dict, "num_device_malloc",
        PyLong_FromSize_t(stats.num_device_malloc));
    PyDict_SetItemString(dict, "num_device_free",
        PyLong_FromSize_t(stats.num_device_free));
    PyDict_SetItemString(dict, "num_alloc_retries",
        PyLong_FromSize_t(stats.num_alloc_retries));
  }
  return dict;
  END_HANDLE_TH_ERRORS
}

PyObject* _memory_allocated(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  int device = THPUtils_unpackInt(arg);
  size_t result = 0;
  if (c10::flagos::CachingDeviceAllocator::is_enabled()) {
    auto stats = c10::flagos::GetCachingAllocator()->get_stats(device);
    result = stats.bytes_allocated;
  }
  return PyLong_FromSize_t(result);
  END_HANDLE_TH_ERRORS
}

PyObject* _memory_reserved(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  int device = THPUtils_unpackInt(arg);
  size_t result = 0;
  if (c10::flagos::CachingDeviceAllocator::is_enabled()) {
    auto stats = c10::flagos::GetCachingAllocator()->get_stats(device);
    result = stats.bytes_reserved;
  }
  return PyLong_FromSize_t(result);
  END_HANDLE_TH_ERRORS
}

PyObject* _reset_peak_memory_stats(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  int device = THPUtils_unpackInt(arg);
  if (c10::flagos::CachingDeviceAllocator::is_enabled()) {
    c10::flagos::GetCachingAllocator()->reset_stats(device);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyMethodDef methods[] = {
    {"_init", _initExtension, METH_NOARGS, nullptr},
    {"_get_default_generator", _getDefaultGenerator, METH_O, nullptr},
    {"_reserve_rng_seed", _reserveRngSeed, METH_VARARGS, nullptr},
#ifdef USE_MUSA
    {"_get_musa_device_properties", _getMusaDeviceProperties, METH_O, nullptr},
    {"_get_musa_current_raw_stream", _getMusaCurrentRawStream, METH_O, nullptr},
#endif
    {"_get_device", _getDevice, METH_NOARGS, nullptr},
    {"_set_device", _setDevice, METH_O, nullptr},
    {"_exchangeDevice", _exchangeDevice, METH_O, nullptr},
    {"_get_device_count", _getDeviceCount, METH_NOARGS, nullptr},
    {"_synchronize", _synchronize, METH_NOARGS, nullptr},
    {"_flagos_to_cuda_view", _flagos_to_cuda_view, METH_O, nullptr},
    {"_cuda_to_flagos_view", _cuda_to_flagos_view, METH_VARARGS, nullptr},
    {"_flagos_identity_view", _flagos_identity_view, METH_O, nullptr},
    {"_empty_cache", _empty_cache, METH_NOARGS, nullptr},
    {"_memory_stats", _memory_stats, METH_O, nullptr},
    {"_memory_allocated", _memory_allocated, METH_O, nullptr},
    {"_memory_reserved", _memory_reserved, METH_O, nullptr},
    {"_reset_peak_memory_stats", _reset_peak_memory_stats, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}};

extern "C" FLAGOS_EXPORT PyObject* initFlagosModule(void) {
  static struct PyModuleDef flagos_C_module = {
      PyModuleDef_HEAD_INIT, "torch_fl._C", nullptr, -1, methods};
  PyObject* mod = PyModule_Create(&flagos_C_module);

  return mod;
}
