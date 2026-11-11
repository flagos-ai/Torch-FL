// Copyright (c) 2026, BAAI. All rights reserved.
//
// Fast replay of a captured FlagGems Triton kernel. See fast_plan.h for the contract.

#include "csrc/aten/backends/flagos/fast_plan.h"

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/Storage.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(FLAGOS_HAVE_FAST_PLAN)

// The driver entry points are resolved at runtime rather than linked: the plugin links the
// CUDA runtime, not the driver, and resolving keeps this file compiling on platforms whose
// toolchain has no driver library. A platform where they do not resolve simply never
// registers a plan and keeps the Python route.
#include <dlfcn.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>

namespace at::native::flagos {
namespace {

using CuModuleLoadDataFn = CUresult (*)(CUmodule*, const void*);
using CuModuleGetFunctionFn = CUresult (*)(CUfunction*, CUmodule, const char*);
using CuLaunchKernelFn = CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                                      unsigned, unsigned, unsigned, CUstream, void**,
                                      void**);

struct DriverApi {
  CuModuleLoadDataFn module_load_data = nullptr;
  CuModuleGetFunctionFn module_get_function = nullptr;
  CuLaunchKernelFn launch_kernel = nullptr;

  bool ok() const {
    return module_load_data != nullptr && module_get_function != nullptr &&
           launch_kernel != nullptr;
  }
};

void* DriverHandle() {
  // The CUDA runtime loads libcuda with RTLD_LOCAL, so RTLD_DEFAULT does not see its
  // symbols; open the driver explicitly. Both names are tried because the unversioned
  // soname exists only where a development package installed the linker script.
  static void* handle = [] {
    for (const char* name : {"libcuda.so.1", "libcuda.so"}) {
      if (void* opened = dlopen(name, RTLD_NOW | RTLD_GLOBAL)) {
        return opened;
      }
    }
    return static_cast<void*>(nullptr);
  }();
  return handle;
}

const DriverApi& Driver() {
  static const DriverApi api = [] {
    DriverApi resolved;
    void* handle = DriverHandle();
    if (handle == nullptr) {
      return resolved;
    }
    resolved.module_load_data =
        reinterpret_cast<CuModuleLoadDataFn>(dlsym(handle, "cuModuleLoadData"));
    resolved.module_get_function =
        reinterpret_cast<CuModuleGetFunctionFn>(dlsym(handle, "cuModuleGetFunction"));
    resolved.launch_kernel =
        reinterpret_cast<CuLaunchKernelFn>(dlsym(handle, "cuLaunchKernel"));
    return resolved;
  }();
  return api;
}

// A pooled result buffer. The pool hands a slot back only when the last reference to the
// tensor wrapping it drops, so two live results can never alias; when every slot is busy
// the pool grows instead of reusing.
struct Pool {
  std::vector<at::Tensor> buffers;
  std::vector<bool> busy;

  size_t acquire(int64_t numel, c10::ScalarType dtype, const at::Device& device) {
    for (size_t index = 0; index < buffers.size(); ++index) {
      if (!busy[index]) {
        busy[index] = true;
        return index;
      }
    }
    buffers.push_back(
        at::empty({numel}, at::TensorOptions().dtype(dtype).device(device)));
    busy.push_back(true);
    return buffers.size() - 1;
  }

  void release(size_t index) { busy[index] = false; }
};

// Ownership of a pooled slot travels with the result: the deleter returns the slot to the
// pool when the last reference to the tensor drops, which is what keeps two live results
// from sharing a buffer without needing any bookkeeping at the call site.
struct SlotState {
  Pool* pool = nullptr;
  size_t index = 0;
};

at::Tensor WrapSlot(Pool* pool, size_t slot, const std::vector<int64_t>& shape,
                    c10::ScalarType dtype, const c10::DispatchKeySet& key_set,
                    const at::Device& device) {
  auto* state = new SlotState{pool, slot};
  const int64_t bytes = pool->buffers[slot].nbytes();
  c10::DataPtr data(pool->buffers[slot].data_ptr(), state,
                    [](void* context) {
                      auto* owned = static_cast<SlotState*>(context);
                      owned->pool->release(owned->index);
                      delete owned;
                    },
                    device);
  c10::Storage storage(c10::Storage::use_byte_size_t(), bytes, std::move(data),
                       /*allocator=*/nullptr, /*resizable=*/false);
  auto tensor = at::detail::make_tensor<c10::TensorImpl>(std::move(storage), key_set,
                                                         c10::scalarTypeToTypeMeta(dtype));
  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
  return tensor;
}

struct Plan {
  std::string kernel;
  CUfunction fn = nullptr;
  int grid = 0;
  int threads = 0;
  int shared = 0;
  std::vector<std::string> kinds;
  std::vector<uint64_t> values;  // argument storage the kernel reads
  std::vector<std::vector<int64_t>> operand_shapes;
  std::vector<c10::ScalarType> operand_dtypes;
  std::vector<int64_t> scalar_values;  // the call's non-tensor arguments
  std::vector<int64_t> out_shape;
  int64_t out_numel = 0;
  c10::ScalarType out_dtype = c10::ScalarType::Float;
  Pool pool;
  std::atomic<int64_t> replays{0};
};

std::mutex& RegistryMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, std::shared_ptr<Plan>>& Registry() {
  static std::unordered_map<std::string, std::shared_ptr<Plan>> registry;
  return registry;
}

// The kernel is launched with each parameter as its own slot, so the values live in a
// per-launch array of pointers into the plan's storage.
CUresult Launch(const Plan& plan, CUstream stream, void** kernel_params) {
  return Driver().launch_kernel(plan.fn, static_cast<unsigned>(plan.grid), 1, 1,
                                static_cast<unsigned>(plan.threads), 1, 1,
                                static_cast<unsigned>(plan.shared), stream, kernel_params,
                                nullptr);
}

// The plan is only valid for the call it was captured from: same operand shapes, types
// and layout, and the same scalar arguments.
bool CallMatches(const Plan& plan, const std::vector<at::Tensor>& operands,
                 const std::vector<int64_t>& scalars) {
  if (operands.size() != plan.operand_shapes.size()) {
    return false;
  }
  if (scalars != plan.scalar_values) {
    return false;
  }
  for (size_t index = 0; index < operands.size(); ++index) {
    const at::Tensor& operand = operands[index];
    if (operand.scalar_type() != plan.operand_dtypes[index]) {
      return false;
    }
    if (operand.sizes() != at::IntArrayRef(plan.operand_shapes[index])) {
      return false;
    }
    if (!operand.is_contiguous()) {
      return false;
    }
  }
  return true;
}

int64_t ScalarPayload(const c10::IValue& value) {
  if (value.isBool()) {
    return value.toBool() ? 1 : 0;
  }
  if (value.isInt()) {
    return value.toInt();
  }
  if (value.isDouble()) {
    return static_cast<int64_t>(value.toDouble());
  }
  return 0;
}

}  // namespace

bool RegisterFastPlan(const std::string& op, const std::string& kernel, const void* cubin,
                      size_t cubin_size, int64_t grid, int64_t threads, int64_t shared,
                      const std::vector<std::string>& kinds,
                      const std::vector<int64_t>& ints,
                      const std::vector<std::vector<int64_t>>& operand_shapes,
                      const std::vector<int64_t>& operand_dtypes,
                      const std::vector<int64_t>& scalar_values,
                      const std::vector<int64_t>& out_shape, int64_t out_dtype) {
  const DriverApi& driver = Driver();
  if (!driver.ok() || cubin == nullptr || cubin_size == 0) {
    return false;
  }
  CUmodule module = nullptr;
  if (driver.module_load_data(&module, cubin) != CUDA_SUCCESS) {
    return false;
  }
  auto plan = std::make_shared<Plan>();
  if (driver.module_get_function(&plan->fn, module, kernel.c_str()) != CUDA_SUCCESS) {
    return false;
  }
  plan->kernel = kernel;
  plan->grid = static_cast<int>(grid);
  plan->threads = static_cast<int>(threads);
  plan->shared = static_cast<int>(shared);
  plan->kinds = kinds;
  plan->values.assign(kinds.size(), 0);

  // Scalar parameters are baked from the captured launch, in kernel order. They describe
  // the shape the plan was captured for, which is why the plan is also matched on shapes.
  size_t int_index = 0;
  for (size_t slot = 0; slot < plan->kinds.size(); ++slot) {
    if (plan->kinds[slot] == "i32") {
      if (int_index >= ints.size()) {
        return false;
      }
      const int32_t value = static_cast<int32_t>(ints[int_index++]);
      plan->values[slot] = 0;
      std::memcpy(&plan->values[slot], &value, sizeof(value));
    }
  }
  if (operand_dtypes.size() != operand_shapes.size()) {
    return false;
  }
  plan->operand_shapes = operand_shapes;
  plan->operand_dtypes.reserve(operand_dtypes.size());
  for (int64_t dtype : operand_dtypes) {
    plan->operand_dtypes.push_back(static_cast<c10::ScalarType>(dtype));
  }
  plan->scalar_values = scalar_values;
  plan->out_shape = out_shape;
  plan->out_dtype = static_cast<c10::ScalarType>(out_dtype);
  plan->out_numel = 1;
  for (int64_t extent : plan->out_shape) {
    plan->out_numel *= extent;
  }

  std::lock_guard<std::mutex> guard(RegistryMutex());
  Registry()[op] = std::move(plan);
  return true;
}

bool HasFastPlan(const char* op) {
  if (op == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> guard(RegistryMutex());
  return Registry().find(op) != Registry().end();
}

int64_t FastPlanCount() {
  std::lock_guard<std::mutex> guard(RegistryMutex());
  return static_cast<int64_t>(Registry().size());
}

int64_t FastPlanReplays(const char* op) {
  if (op == nullptr) {
    return 0;
  }
  std::lock_guard<std::mutex> guard(RegistryMutex());
  auto found = Registry().find(op);
  return found == Registry().end() ? 0 : found->second->replays.load();
}

at::Tensor TryFastReplay(const char* op, const std::vector<c10::IValue>& args) {
  if (op == nullptr) {
    return at::Tensor();
  }
  std::vector<at::Tensor> operands;
  std::vector<int64_t> scalars;
  operands.reserve(args.size());
  for (const c10::IValue& arg : args) {
    if (arg.isTensor()) {
      operands.push_back(arg.toTensor());
    } else {
      scalars.push_back(ScalarPayload(arg));
    }
  }
  if (operands.empty()) {
    return at::Tensor();
  }
  std::shared_ptr<Plan> plan;
  {
    std::lock_guard<std::mutex> guard(RegistryMutex());
    auto found = Registry().find(op);
    if (found == Registry().end()) {
      return at::Tensor();
    }
    plan = found->second;
  }
  if (!CallMatches(*plan, operands, scalars)) {
    // A different shape or layout is what the plan was not captured for; the caller falls
    // back rather than launching with the captured geometry.
    return at::Tensor();
  }

  const at::Device device = operands[0].device();
  const size_t slot = plan->pool.acquire(plan->out_numel, plan->out_dtype, device);
  size_t operand_index = 0;
  for (size_t position = 0; position < plan->kinds.size(); ++position) {
    const std::string& kind = plan->kinds[position];
    if (kind == "in") {
      plan->values[position] = reinterpret_cast<uint64_t>(operands[operand_index].data_ptr());
      ++operand_index;
    } else if (kind == "out") {
      plan->values[position] =
          reinterpret_cast<uint64_t>(plan->pool.buffers[slot].data_ptr());
    }
  }

  // The driver reads each parameter through its slot for the duration of the call, so the
  // pointer array is built on the stack from the plan's storage.
  uint64_t local[16];
  void* kernel_params[16];
  const size_t count =
      std::min(plan->kinds.size(), sizeof(kernel_params) / sizeof(kernel_params[0]));
  for (size_t index = 0; index < count; ++index) {
    local[index] = plan->values[index];
    kernel_params[index] = &local[index];
  }

  CUstream stream = reinterpret_cast<CUstream>(c10::cuda::getCurrentCUDAStream().stream());
  if (Launch(*plan, stream, kernel_params) != CUDA_SUCCESS) {
    plan->pool.release(slot);
    return at::Tensor();
  }

  plan->replays.fetch_add(1, std::memory_order_relaxed);
  return WrapSlot(&plan->pool, slot, plan->out_shape, plan->out_dtype,
                  operands[0].key_set(), device);
}

}  // namespace at::native::flagos

#else  // FLAGOS_HAVE_FAST_PLAN

namespace at::native::flagos {

bool RegisterFastPlan(const std::string&, const std::string&, const void*, size_t, int64_t,
                      int64_t, int64_t, const std::vector<std::string>&,
                      const std::vector<int64_t>&,
                      const std::vector<std::vector<int64_t>>&,
                      const std::vector<int64_t>&, const std::vector<int64_t>&,
                      const std::vector<int64_t>&, int64_t) {
  return false;
}

bool HasFastPlan(const char*) { return false; }

at::Tensor TryFastReplay(const char*, const std::vector<c10::IValue>&) {
  return at::Tensor();
}

int64_t FastPlanCount() { return 0; }

int64_t FastPlanReplays(const char*) { return 0; }

}  // namespace at::native::flagos

#endif  // FLAGOS_HAVE_FAST_PLAN
#if defined(FLAGOS_HAVE_FAST_PLAN)

// Registration surface for Python, which owns the compiled kernel. A custom operator
// rather than a pybind module: the plugin builds one shared library and registers its
// entry points through torch, so this needs no new module, no new build target and no
// packaging change.
TORCH_LIBRARY(flagos_fast_plan, m) {
  m.def(
      "register_fast_plan(str op, str kernel, Tensor cubin, int grid, int threads, "
      "int shared, str[] kinds, int[] ints, int[][] operand_shapes, "
      "int[] operand_dtypes, int[] scalars, int[] out_shape, int out_dtype) -> bool",
      [](const std::string& op, const std::string& kernel, const at::Tensor& cubin,
         int64_t grid, int64_t threads, int64_t shared, std::vector<std::string> kinds,
         std::vector<int64_t> ints,
         std::vector<std::vector<int64_t>> operand_shapes,
         std::vector<int64_t> operand_dtypes, std::vector<int64_t> scalars,
         std::vector<int64_t> out_shape, int64_t out_dtype) {
        return at::native::flagos::RegisterFastPlan(
            op, kernel, cubin.data_ptr(), static_cast<size_t>(cubin.numel()), grid,
            threads, shared, kinds, ints, operand_shapes, operand_dtypes, scalars,
            out_shape, out_dtype);
      });
  m.def("fast_plan_count() -> int", []() { return at::native::flagos::FastPlanCount(); });
}

#endif  // FLAGOS_HAVE_FAST_PLAN
