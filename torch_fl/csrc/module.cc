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
#include <ATen/ops/from_blob.h>
#include <vector>
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
#include <runtime/accelerator/mthreads/musa_stream.h>
#endif

#ifdef USE_ILUVATAR
// corex ships the CUDA-shaped C runtime, and this build links the CPU torch
// wheel, so the device properties, the current stream and the stream-priority
// range below come from the runtime itself rather than from torch.cuda.
#include <cuda_runtime.h>
#include <flagos.h>
#endif

#ifdef USE_KUNLUNXIN
// The XPU runtime's per-attribute query, which is the whole of what this build
// can ask a P800 about itself: see _getKunlunxinDeviceProperties below.
#include <cstdio>

#include <xpu/runtime.h>
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

#include <aten/common.h>

#include "dataparallel_comm.h"

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

// Iluvatar's counterpart of the MUSA getter above, and for the same reason: the
// backend links the CPU torch wheel, so there is no `_cuda_getCurrentRawStream`
// and no torch stream to read. The handle comes from the platform's own stream
// registry (libflagos.so, csrc/runtime/accelerator/iluvatar/stream.cc), which is
// the queue the eager iluvatar kernels submit to, so a FlagTree/FlagGems kernel
// launched on it is ordered against eager work.
//
// Unlike MUSA the index is not rejected when it is not the current device. The
// registry is keyed by device and a current-stream entry is therefore per
// device, while MUSA's single shared stream exists only for the current one --
// so a wrong index there describes the wrong stream, but here it names the
// stream that device would use.
static PyObject* _getIluvatarCurrentRawStream(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
#ifdef USE_ILUVATAR
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_get_iluvatar_current_raw_stream expects an int, but got ",
      THPUtils_typename(arg));
  const auto idx = static_cast<int>(THPUtils_unpackLong(arg));
  auto stream = GetCurrentStreamForDevice(idx);
  return PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(stream));
#else
  TORCH_CHECK(false, "Iluvatar streams are unavailable in this build");
#endif
  END_HANDLE_TH_ERRORS
}

// Iluvatar device properties, in the same dict shape as the MUSA getter so
// torch.flagos._DeviceProperties can read either. Corex ships a complete
// cuda_runtime.h, so this is cudaGetDeviceProperties rather than the stand-in
// the Ascend-shaped arms fall back to: inductor derives warp size and the
// launch-grid limits from it, and a falsely narrow device produces kernels that
// cannot be launched.
static PyObject* _getIluvatarDeviceProperties(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
#ifdef USE_ILUVATAR
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_get_iluvatar_device_properties expects an int, but got ",
      THPUtils_typename(arg));
  const auto idx = static_cast<int>(THPUtils_unpackLong(arg));
  cudaDeviceProp prop;
  TORCH_CHECK(
      cudaGetDeviceProperties(&prop, idx) == cudaSuccess,
      "cudaGetDeviceProperties failed for device ",
      idx);
  return Py_BuildValue(
      "{s:s,s:i,s:K,s:i,s:i,s:i,s:K,s:i,s:i,s:K}",
      "name",
      prop.name,
      "multi_processor_count",
      prop.multiProcessorCount,
      "total_memory",
      static_cast<unsigned long long>(prop.totalGlobalMem),
      "major",
      prop.major,
      "minor",
      prop.minor,
      "warp_size",
      prop.warpSize,
      "l2_cache_size",
      static_cast<unsigned long long>(prop.l2CacheSize),
      "max_threads_per_multi_processor",
      prop.maxThreadsPerMultiProcessor,
      "regs_per_block",
      prop.regsPerBlock,
      "shared_memory_per_multiprocessor",
      static_cast<unsigned long long>(prop.sharedMemPerMultiprocessor));
#else
  TORCH_CHECK(false, "Iluvatar device properties are unavailable in this build");
#endif
  END_HANDLE_TH_ERRORS
}

#ifdef USE_KUNLUNXIN
// One device attribute, as the XPU runtime answers it. The SDK has no
// per-device property struct -- xpu_device_get_attr is the entire query surface
// -- and no free-memory or device-name entry point (names live in XPUML, which
// this wheel does not link), so the properties below are assembled from the
// individual attributes the runtime does report.
static uint64_t KunlunxinDeviceAttr(XPUDeviceAttr attr, int idx) {
  uint64_t value = 0;
  const int err = xpu_device_get_attr(&value, attr, idx);
  TORCH_CHECK(
      err == 0,
      "xpu_device_get_attr(",
      static_cast<int>(attr),
      ") failed for device ",
      idx,
      " with error ",
      err);
  return value;
}
#endif

// Kunlunxin device properties, in the same dict shape as the MUSA and iluvatar
// getters. What the XPU runtime reports is much narrower than a cudaDeviceProp,
// so this is a genuine subset rather than a stand-in: before it existed,
// torch.flagos._DeviceProperties fell through to the Ascend-shaped arm and
// described a P800 as an "Ascend NPU" with 0 bytes -- which is what Inductor
// saw when it sized its workspace.
static PyObject* _getKunlunxinDeviceProperties(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
#ifdef USE_KUNLUNXIN
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_get_kunlunxin_device_properties expects an int, but got ",
      THPUtils_typename(arg));
  auto idx = static_cast<int>(THPUtils_unpackLong(arg));
  TORCH_CHECK(
      idx >= 0 && idx < c10::flagos::DeviceCount(),
      "Kunlunxin device index ",
      idx,
      " is out of range");

  const auto total_memory = KunlunxinDeviceAttr(XPUATTR_MEM_MAIN_CAPACITY, idx);
  const auto l3_capacity = KunlunxinDeviceAttr(XPUATTR_MEM_L3_CAPACITY, idx);
  const auto cluster_count = KunlunxinDeviceAttr(XPUATTR_NUM_CLUSTER, idx);
  // The board product number. It is the only identity libxpurt reports, so the
  // name is built from it rather than invented; it reaches Inductor's cache key
  // purely as a string.
  const auto model = KunlunxinDeviceAttr(XPUATTR_MODEL, idx);

  char name[64];
  std::snprintf(
      name,
      sizeof(name),
      "Kunlunxin XPU %llu",
      static_cast<unsigned long long>(model));

  // The trailing fields are what Inductor's Triton heuristics and its
  // benchmarking buffer read.
  return Py_BuildValue(
      "{s:s,s:K,s:i,s:i,s:i}",
      "name",
      name,
      "total_memory",
      static_cast<unsigned long long>(total_memory),
      "multi_processor_count",
      static_cast<int>(cluster_count),
      // A fabricated zero leaves that flush buffer empty. The XPU reports the
      // on-die L3 (96 MiB on a P800), the cache level these kernels' working
      // set lives in; the SDK has no separate L2 query.
      "l2_cache_size",
      static_cast<int>(l3_capacity),
      "model",
      static_cast<int>(model));
#else
  TORCH_CHECK(false, "Kunlunxin device properties are unavailable in this build");
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

// Adopt memory the caller already owns as a flagos tensor, without copying.
//
// The BPU kernel only accepts addresses inside a hbUCPMallocCached allocation
// (a plain buffer is rejected with rc=-100001), and hbm_runtime hands its
// output back as exactly such a mapping. Without this entry point the eager
// path has to copy that output into a fresh device tensor before the next op
// can read it -- one full tensor-width memcpy per operator, which measured
// 5.6 ms for a 3.2 MB activation against 0.23 ms for the copy-free wrap.
//
// Borrowed storage is not owned by the tensor: the caller keeps the bytes alive
// for as long as the tensor is used. `borrowed` registers the block with the
// runtime allocator so cache maintenance and PointerGetAttributes see it, and
// so a Free of this pointer never releases memory it did not allocate.
PyObject* _flagos_from_blob(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  void* ptr = nullptr;
  PyObject* sizes_obj = nullptr;
  int dtype = 0;
  int borrowed = 1;
  if (!PyArg_ParseTuple(
          args, "KO!i|p", &ptr, &PyTuple_Type, &sizes_obj, &dtype, &borrowed)) {
    return nullptr;
  }
  TORCH_CHECK(ptr != nullptr, "from_blob: null pointer");

  std::vector<int64_t> sizes;
  const Py_ssize_t ndim = PyTuple_Size(sizes_obj);
  int64_t numel = 1;
  for (Py_ssize_t i = 0; i < ndim; ++i) {
    const int64_t dim = PyLong_AsLongLong(PyTuple_GetItem(sizes_obj, i));
    TORCH_CHECK(dim >= 0, "from_blob: negative size");
    sizes.push_back(dim);
    numel *= dim;
  }
  const auto scalar_type = static_cast<c10::ScalarType>(dtype);

  if (borrowed) {
    Borrow(ptr, static_cast<size_t>(numel) * c10::elementSize(scalar_type), 0);
  }

  auto options = at::TensorOptions()
                     .dtype(scalar_type)
                     .device(c10::Device(c10::DeviceType::PrivateUse1, 0));
  auto tensor = at::from_blob(ptr, at::IntArrayRef(sizes), options);
  return THPVariable_Wrap(tensor);
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

PyObject* _shutdown_allocator(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (c10::flagos::CachingDeviceAllocator::is_enabled()) {
    c10::flagos::GetCachingAllocator()->shutdown();
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
    c10::flagos::GetCachingAllocator()->reset_peak_stats(device);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// torch_fl._select_backend_config() resolves the conf path at import time and
// hands it over here instead of writing os.environ["FLAGOS_BACKEND_CONFIG"].
// The environment write is what made "the user overrode the conf" and "the
// wheel chose its own conf" the same observable state for the rest of the
// process. Called before the first dispatch builds the routing table.
PyObject* _set_backend_config_path(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!PyUnicode_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, "_set_backend_config_path expects a str");
    return nullptr;
  }
  const char* path = PyUnicode_AsUTF8(arg);
  if (path == nullptr) return nullptr;
  at::native::flagos::SetBackendConfigPath(path);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// Publishes the flagos implementations of torch._C's DataParallel comm
// primitives. See torch_fl/csrc/dataparallel_comm.cc for what they are and why
// they cannot be left to the CUDA ones. Safe to call more than once.
PyObject* _init_data_parallel_comm(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch_fl::comm::InitDataParallelComm();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// Marks this thread as scattering on behalf of a flagos-placed DataParallel and
// returns the value it replaced, so a wrapper can restore it in a `finally`.
PyObject* _set_scatter_scope(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  const bool previous = torch_fl::comm::SetScatterScope(THPUtils_unpackBool(arg));
  if (previous) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
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
#ifdef USE_ILUVATAR
    {"_get_iluvatar_device_properties", _getIluvatarDeviceProperties, METH_O,
     nullptr},
    {"_get_iluvatar_current_raw_stream", _getIluvatarCurrentRawStream, METH_O,
     nullptr},
#endif
#ifdef USE_KUNLUNXIN
    {"_get_kunlunxin_device_properties", _getKunlunxinDeviceProperties, METH_O,
     nullptr},
#endif
    {"_get_device", _getDevice, METH_NOARGS, nullptr},
    {"_set_device", _setDevice, METH_O, nullptr},
    {"_exchangeDevice", _exchangeDevice, METH_O, nullptr},
    {"_get_device_count", _getDeviceCount, METH_NOARGS, nullptr},
    {"_synchronize", _synchronize, METH_NOARGS, nullptr},
    {"_flagos_to_cuda_view", _flagos_to_cuda_view, METH_O, nullptr},
    {"_cuda_to_flagos_view", _cuda_to_flagos_view, METH_VARARGS, nullptr},
    {"_flagos_identity_view", _flagos_identity_view, METH_O, nullptr},
    {"_flagos_from_blob", _flagos_from_blob, METH_VARARGS, nullptr},
    {"_empty_cache", _empty_cache, METH_NOARGS, nullptr},
    {"_shutdown_allocator", _shutdown_allocator, METH_NOARGS, nullptr},
    {"_memory_stats", _memory_stats, METH_O, nullptr},
    {"_memory_allocated", _memory_allocated, METH_O, nullptr},
    {"_memory_reserved", _memory_reserved, METH_O, nullptr},
    {"_reset_peak_memory_stats", _reset_peak_memory_stats, METH_O, nullptr},
    {"_set_backend_config_path", _set_backend_config_path, METH_O, nullptr},
    {"_init_dataparallel_comm", _init_data_parallel_comm, METH_NOARGS, nullptr},
    {"_set_scatter_scope", _set_scatter_scope, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}};

extern "C" FLAGOS_EXPORT PyObject* initFlagosModule(void) {
  static struct PyModuleDef flagos_C_module = {
      PyModuleDef_HEAD_INIT, "torch_fl._C", nullptr, -1, methods};
  PyObject* mod = PyModule_Create(&flagos_C_module);

  return mod;
}
