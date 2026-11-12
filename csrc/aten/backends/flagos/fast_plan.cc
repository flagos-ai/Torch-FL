// Copyright (c) 2026, BAAI. All rights reserved.
//
// Fast replay of a captured FlagGems Triton kernel. See fast_plan.h for the contract.

#include "csrc/aten/backends/flagos/fast_plan.h"

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <torch/library.h>

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

#include <c10/core/DeviceGuard.h>
#include <ATen/cuda/CUDAEvent.h>
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

// A pooled result buffer.
//
// Reuse is gated twice, because two different things keep a buffer alive: a host tensor
// that still references it (`busy`), and the GPU work that produced or consumes it.
//
// The second gate is where the cost lives, so it is only paid where the hardware does not
// already provide it. Reusing a buffer on the stream that wrote it needs no event: the
// stream orders the new launch after the old one by itself. Reuse across streams does need
// proof, so a slot written on another stream is only reused once an event recorded at that
// launch has completed - and until such an event exists it is not reused at all.
struct Slot {
  at::Tensor buffer;
  bool busy = false;
  int64_t stream_id = -1;    // the stream that last wrote it
  at::cuda::CUDAEvent done;  // recorded only once a plan has been replayed on a second stream
  bool has_event = false;
  bool queried_free = false;  // last query said the event had completed
};

struct Pool {
  std::mutex mutex;
  std::vector<Slot> slots;
  int64_t numel = 0;
  c10::ScalarType dtype = c10::ScalarType::Float;
  at::Device device = at::Device(at::kCPU);
  bool has_device = false;

  size_t Acquire(const at::Device& wanted, int64_t stream_id) {
    std::lock_guard<std::mutex> guard(mutex);
    if (!has_device) {
      device = wanted;
      has_device = true;
    }
    for (size_t index = 0; index < slots.size(); ++index) {
      Slot& slot = slots[index];
      if (slot.busy) {
        continue;  // a live tensor references it
      }
      if (slot.stream_id == stream_id) {
        // Same stream: the stream itself orders this launch after whatever wrote it, so
        // no event is needed and none is recorded on this path.
        slot.busy = true;
        slot.has_event = false;
        return index;
      }
      // Another stream wrote it. Only reuse it with proof that the work drained; without
      // an event there is no proof, so the caller gets a new buffer instead.
      if (slot.has_event && slot.done.query()) {
        slot.busy = true;
        slot.stream_id = stream_id;
        slot.has_event = false;
        return index;
      }
      continue;
    }
    Slot fresh;
    fresh.buffer = at::empty({numel}, at::TensorOptions().dtype(dtype).device(device));
    fresh.busy = true;
    fresh.stream_id = stream_id;
    slots.push_back(std::move(fresh));
    return slots.size() - 1;
  }

  void Release(size_t index) {
    std::lock_guard<std::mutex> guard(mutex);
    slots[index].busy = false;
  }

  // Records an event only for plans that have been replayed on more than one stream. A
  // plan that stays on one stream never pays for the gate at all.
  void RecordLaunch(size_t index, const c10::cuda::CUDAStream& stream, bool need_event) {
    std::lock_guard<std::mutex> guard(mutex);
    Slot& slot = slots[index];
    slot.stream_id = static_cast<int64_t>(stream.id());
    if (need_event) {
      slot.done.record(stream);
      slot.has_event = true;
    }
  }

  void* DataPtr(size_t index) const { return slots[index].buffer.data_ptr(); }
};

// Ownership of a pooled slot travels with the result: the deleter releases the slot when
// the last reference to the tensor drops, and the shared ownership of the pool keeps the
// buffers alive even if the plan is unregistered or replaced while results are outstanding.
struct SlotState {
  std::shared_ptr<Pool> pool;
  size_t index = 0;
};

struct Plan {
  std::string kernel;
  CUfunction fn = nullptr;
  unsigned grid[3] = {1, 1, 1};
  unsigned threads = 1;
  unsigned shared = 0;
  int device_index = 0;
  std::vector<std::string> kinds;
  // The scalars the plan was captured with, at their kernel positions. Pointer positions
  // stay zero: they are filled per call into a local copy, so a replay never mutates the
  // plan and two calls cannot race on it.
  std::vector<uint64_t> template_values;
  std::vector<std::vector<int64_t>> operand_shapes;
  std::vector<c10::ScalarType> operand_dtypes;
  std::vector<int64_t> scalar_values;
  std::vector<int64_t> out_shape;
  int64_t out_numel = 0;
  c10::ScalarType out_dtype = c10::ScalarType::Float;
  std::shared_ptr<Pool> pool;
  std::atomic<int64_t> replays{0};
  // The first stream the plan ran on, and whether a second one has been seen. Until it
  // has, no event is recorded: a single-stream plan is ordered by its stream alone.
  std::atomic<int64_t> first_stream{-1};
  std::atomic<bool> multi_stream{false};
};

at::Tensor WrapSlot(const std::shared_ptr<Pool>& pool, size_t slot,
                    const std::vector<int64_t>& shape, c10::ScalarType dtype,
                    const c10::DispatchKeySet& key_set, const at::Device& device) {
  auto* state = new SlotState{pool, slot};
  const int64_t bytes = pool->slots[slot].buffer.nbytes();
  c10::DataPtr data(pool->DataPtr(slot), state,
                    [](void* context) {
                      auto* owned = static_cast<SlotState*>(context);
                      owned->pool->Release(owned->index);
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

std::mutex& RegistryMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, std::shared_ptr<Plan>>& Registry() {
  static std::unordered_map<std::string, std::shared_ptr<Plan>> registry;
  return registry;
}

std::shared_ptr<Plan> Lookup(const char* op) {
  if (op == nullptr) {
    return nullptr;
  }
  std::lock_guard<std::mutex> guard(RegistryMutex());
  auto found = Registry().find(op);
  return found == Registry().end() ? nullptr : found->second;
}

CUresult Launch(Plan& plan, CUstream stream, void** kernel_params) {
  return Driver().launch_kernel(plan.fn, plan.grid[0], plan.grid[1], plan.grid[2],
                                plan.threads, 1, 1, plan.shared, stream, kernel_params,
                                nullptr);
}

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
    // A CUfunction belongs to the context it was loaded in, and a pooled buffer belongs to
    // one device: replaying either on another device would compute on the wrong memory.
    if (operand.device().index() != plan.device_index) {
      return false;
    }
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
                      size_t cubin_size, int64_t grid_x, int64_t grid_y, int64_t grid_z,
                      int64_t threads, int64_t shared, int64_t device_index,
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
  if (grid_x <= 0 || grid_y <= 0 || grid_z <= 0 || device_index < 0) {
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
  plan->grid[0] = static_cast<unsigned>(grid_x);
  plan->grid[1] = static_cast<unsigned>(grid_y);
  plan->grid[2] = static_cast<unsigned>(grid_z);
  plan->threads = static_cast<unsigned>(threads);
  plan->shared = static_cast<unsigned>(shared);
  plan->device_index = static_cast<int>(device_index);
  plan->kinds = kinds;
  plan->template_values.assign(kinds.size(), 0);

  // Scalar parameters are baked from the captured launch, in kernel order. They describe
  // the call the plan was captured for, which is why the plan is also matched on them.
  size_t int_index = 0;
  for (size_t slot = 0; slot < plan->kinds.size(); ++slot) {
    if (plan->kinds[slot] == "i32") {
      if (int_index >= ints.size()) {
        return false;
      }
      const int32_t value = static_cast<int32_t>(ints[int_index++]);
      std::memcpy(&plan->template_values[slot], &value, sizeof(value));
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
  plan->pool = std::make_shared<Pool>();
  plan->pool->numel = plan->out_numel;
  plan->pool->dtype = plan->out_dtype;

  std::lock_guard<std::mutex> guard(RegistryMutex());
  Registry()[op] = std::move(plan);
  return true;
}

bool UnregisterFastPlan(const std::string& op) {
  std::shared_ptr<Plan> removed;
  {
    std::lock_guard<std::mutex> guard(RegistryMutex());
    auto found = Registry().find(op);
    if (found == Registry().end()) {
      return false;
    }
    removed = std::move(found->second);
    Registry().erase(found);
  }
  // `removed` goes out of scope here, but any result still in flight holds its own
  // reference to the pool, so outstanding tensors stay valid.
  return true;
}

bool HasFastPlan(const char* op) { return Lookup(op) != nullptr; }

int64_t FastPlanCount() {
  std::lock_guard<std::mutex> guard(RegistryMutex());
  return static_cast<int64_t>(Registry().size());
}

int64_t FastPlanReplays(const char* op) {
  auto plan = Lookup(op);
  return plan == nullptr ? 0 : plan->replays.load();
}

at::Tensor TryFastReplay(const char* op, const std::vector<c10::IValue>& args) {
  auto plan = Lookup(op);
  if (plan == nullptr) {
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
  if (operands.empty() || !CallMatches(*plan, operands, scalars)) {
    return at::Tensor();
  }

  // The device comes from the operands: this file is not where the plugin's device type is
  // named, and the pointer the plan replays with belongs to whatever device they sit on.
  const at::Device device = operands[0].device();
  const std::shared_ptr<Pool> pool = plan->pool;

  // The replay runs before the caller's device guard, and both the CUfunction and the
  // stream belong to a device, so select it here. A generic guard, not a CUDA one: the
  // operands live on the plugin's own device type, which the CUDA guard rejects.
  const c10::OptionalDeviceGuard guard(device);
  const c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(device.index());
  const int64_t stream_id = static_cast<int64_t>(stream.id());
  // Observe the stream before acquiring: once a plan has run on two streams every launch
  // records an event, so a buffer written on one can be proven safe to reuse on another.
  int64_t expected = -1;
  if (plan->first_stream.compare_exchange_strong(expected, stream_id)) {
    // First stream seen; nothing to compare against yet.
  } else if (expected != stream_id) {
    plan->multi_stream.store(true, std::memory_order_relaxed);
  }
  const size_t slot = pool->Acquire(device, stream_id);

  // The arguments are assembled per call from the plan's template, so a replay never
  // writes to the plan and concurrent calls cannot overwrite each other's pointers.
  std::vector<uint64_t> values = plan->template_values;
  size_t operand_index = 0;
  for (size_t position = 0; position < plan->kinds.size(); ++position) {
    const std::string& kind = plan->kinds[position];
    if (kind == "in") {
      values[position] = reinterpret_cast<uint64_t>(operands[operand_index].data_ptr());
      ++operand_index;
    } else if (kind == "out") {
      values[position] = reinterpret_cast<uint64_t>(pool->DataPtr(slot));
    }
  }
  std::vector<void*> kernel_params(values.size());
  for (size_t index = 0; index < values.size(); ++index) {
    kernel_params[index] = &values[index];
  }

  if (Launch(*plan, stream.stream(), kernel_params.data()) != CUDA_SUCCESS) {
    pool->Release(slot);
    return at::Tensor();
  }
  pool->RecordLaunch(slot, stream, plan->multi_stream.load(std::memory_order_relaxed));
  plan->replays.fetch_add(1, std::memory_order_relaxed);
  return WrapSlot(pool, slot, plan->out_shape, plan->out_dtype, operands[0].key_set(),
                  device);
}

}  // namespace at::native::flagos

// Registration surface for Python, which owns the compiled kernel. A custom operator
// rather than a pybind module: the plugin builds one shared library and registers its
// entry points through torch, so this needs no new module, no new build target and no
// packaging change.
TORCH_LIBRARY(flagos_fast_plan, m) {
  m.def(
      "register_fast_plan(str op, str kernel, Tensor cubin, int grid_x, int grid_y, "
      "int grid_z, int threads, int shared, int device, str[] kinds, int[] ints, "
      "int[][] operand_shapes, int[] operand_dtypes, int[] scalars, int[] out_shape, "
      "int out_dtype) -> bool",
      [](const std::string& op, const std::string& kernel, const at::Tensor& cubin,
         int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t threads, int64_t shared,
         int64_t device, std::vector<std::string> kinds, std::vector<int64_t> ints,
         std::vector<std::vector<int64_t>> operand_shapes,
         std::vector<int64_t> operand_dtypes, std::vector<int64_t> scalars,
         std::vector<int64_t> out_shape, int64_t out_dtype) {
        return at::native::flagos::RegisterFastPlan(
            op, kernel, cubin.data_ptr(), static_cast<size_t>(cubin.numel()), grid_x,
            grid_y, grid_z, threads, shared, device, kinds, ints, operand_shapes,
            operand_dtypes, scalars, out_shape, out_dtype);
      });
  m.def("unregister_fast_plan(str op) -> bool", [](const std::string& op) {
    return at::native::flagos::UnregisterFastPlan(op);
  });
  m.def("fast_plan_count() -> int", []() { return at::native::flagos::FastPlanCount(); });
}

#else  // FLAGOS_HAVE_FAST_PLAN

namespace at::native::flagos {

bool RegisterFastPlan(const std::string&, const std::string&, const void*, size_t, int64_t,
                      int64_t, int64_t, int64_t, int64_t, int64_t,
                      const std::vector<std::string>&, const std::vector<int64_t>&,
                      const std::vector<std::vector<int64_t>>&,
                      const std::vector<int64_t>&, const std::vector<int64_t>&,
                      const std::vector<int64_t>&, int64_t) {
  return false;
}

bool UnregisterFastPlan(const std::string&) { return false; }

bool HasFastPlan(const char*) { return false; }

at::Tensor TryFastReplay(const char*, const std::vector<c10::IValue>&) {
  return at::Tensor();
}

int64_t FastPlanCount() { return 0; }

int64_t FastPlanReplays(const char*) { return 0; }

}  // namespace at::native::flagos

#endif  // FLAGOS_HAVE_FAST_PLAN