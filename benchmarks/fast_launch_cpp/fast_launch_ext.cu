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

// Pooled C++ replay for a captured FlagGems Triton kernel (prototype).
//
// The per-call path is: three pointer stores into a prebuilt argument array, then
// cuLaunchKernel. No Python, no per-call allocation (the output comes from a pool),
// no dict lookup and no argument-list rebuild. Python registers a plan once.
//
// The pooled output is *diagnostic* here: production needs the buffer returned to the
// pool when the result tensor's refcount drops (a custom DataPtr deleter), otherwise a
// caller that keeps two results sees them alias.

#include <torch/extension.h>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>


struct Plan {
  CUfunction fn = nullptr;
  int grid = 0;
  int threads = 0;
  int shared = 0;
  std::vector<std::string> kinds;      // "in" (per input, in order), "out", "i32", "zero"
  std::vector<uint64_t> values;        // argument storage the kernel reads
  std::vector<void*> args;             // args[i] -> &values[i]
  std::vector<at::Tensor> buffers;     // one backing store per pool slot
  std::vector<bool> busy;              // slot is referenced by a live result
  at::Tensor scratch;                  // Triton global scratch, when required
  int64_t numel = 0;
  std::vector<int64_t> shape;          // the result shape, so the buffer keeps it
  at::TensorOptions options;
  size_t cursor = 0;

  void bake() {
    args.resize(values.size());
    for (size_t i = 0; i < values.size(); i++) args[i] = &values[i];
  }

  // A slot may only be handed out while no live result references it: reusing one
  // would make two results alias, which is the silent-corruption case this design has
  // to avoid. Growing instead keeps the hot path allocation-free in steady state.
  size_t acquire() {
    for (size_t i = 0; i < buffers.size(); i++) {
      if (!busy[i]) { busy[i] = true; return i; }
    }
    buffers.push_back(at::empty(shape, options));
    busy.push_back(true);
    return buffers.size() - 1;
  }

  void release(size_t index) { busy[index] = false; }
};

// The result tensor owns a pooled buffer until its refcount drops, at which point the
// slot returns to the plan. The deleter is the only place pool lifetime is decided.
struct SlotState {
  Plan* plan;
  size_t index;
};

at::Tensor WrapSlot(Plan& plan, size_t index) {
  auto state = std::make_shared<SlotState>(SlotState{&plan, index});
  return at::from_blob(
      plan.buffers[index].data_ptr(), plan.shape,
      [state](void*) { state->plan->release(state->index); }, plan.options);
}

std::unordered_map<std::string, std::unique_ptr<Plan>> g_plans;

Plan& MustGet(const std::string& key) {
  auto it = g_plans.find(key);
  TORCH_CHECK(it != g_plans.end(), "no such plan: ", key);
  return *it->second;
}

CUresult Launch(Plan& plan, CUstream stream) {
  // Argument storage on the stack, exactly like the launch that was verified earlier:
  // the driver reads each parameter through the pointer for the duration of the call.
  uint64_t local[8];
  void* stack_args[8];
  size_t count = plan.values.size() < 8 ? plan.values.size() : 8;
  for (size_t i = 0; i < count; i++) {
    local[i] = plan.values[i];
    stack_args[i] = &local[i];
  }
  return cuLaunchKernel(plan.fn, (unsigned)plan.grid, 1, 1, (unsigned)plan.threads, 1, 1,
                        (unsigned)plan.shared, stream, stack_args, nullptr);
}

void FillIn(Plan& plan, const std::vector<at::Tensor>& inputs, size_t slot) {
  size_t next_input = 0;
  for (size_t i = 0; i < plan.kinds.size(); i++) {
    if (plan.kinds[i] == "in") {
      TORCH_CHECK(next_input < inputs.size(), "plan expects more inputs");
      plan.values[i] = (uint64_t)inputs[next_input++].data_ptr();
    } else if (plan.kinds[i] == "out") {
      plan.values[i] = (uint64_t)plan.buffers[slot].data_ptr();
    } else if (plan.kinds[i] == "scratch") {
      plan.values[i] = (uint64_t)plan.scratch.data_ptr();
    } else if (plan.kinds[i] == "zero") {
      plan.values[i] = 0;  // a scratch pointer the kernel is allowed to ignore
    }
    // "i32" positions were baked at registration
  }
  TORCH_CHECK(next_input == inputs.size(), "plan expects ", next_input, " inputs, got ",
              inputs.size());
}

void register_kernel(std::string key, py::bytes cubin_bytes, std::string kernel_name,
                     int64_t grid, int64_t threads, int64_t shared,
                     std::vector<std::string> kinds, std::vector<int64_t> int_args,
                     int64_t pool_size, std::vector<int64_t> out_shape,
                     int64_t out_dtype_code) {
  std::string data = cubin_bytes;
  CUmodule module = nullptr;
  TORCH_CHECK(cuModuleLoadData(&module, data.data()) == CUDA_SUCCESS, "cuModuleLoadData failed");
  auto plan = std::make_unique<Plan>();
  TORCH_CHECK(cuModuleGetFunction(&plan->fn, module, kernel_name.c_str()) == CUDA_SUCCESS,
              "cuModuleGetFunction failed: ", kernel_name);
  plan->grid = (int)grid;
  plan->threads = (int)threads;
  plan->shared = (int)shared;
  plan->kinds = std::move(kinds);
  plan->values.assign(plan->kinds.size(), 0);

  size_t int_index = 0;
  for (size_t i = 0; i < plan->kinds.size(); i++) {
    if (plan->kinds[i] == "i32") {
      TORCH_CHECK(int_index < int_args.size(), "missing int arg");
      int32_t value = (int32_t)int_args[int_index++];
      plan->values[i] = 0;
      memcpy(&plan->values[i], &value, sizeof(value));  // low 4 bytes, little endian
    }
  }
  plan->bake();

  auto options = at::TensorOptions()
                     .dtype(at::ScalarType(out_dtype_code))
                     .device(at::Device(at::kCUDA, at::cuda::current_device()));
  plan->shape = std::move(out_shape);
  plan->numel = 1;
  for (int64_t extent : plan->shape) plan->numel *= extent;
  plan->options = options;
  for (int64_t i = 0; i < pool_size; i++) {
    plan->buffers.push_back(at::empty(plan->shape, options));
    plan->busy.push_back(false);
  }
  plan->scratch = at::empty({1 << 20}, options);  // over-allocated, never read by us
  g_plans[key] = std::move(plan);
}

at::Tensor launch(std::string key, std::vector<at::Tensor> inputs) {
  Plan& plan = MustGet(key);
  const size_t slot = plan.acquire();
  FillIn(plan, inputs, slot);
  CUstream stream = at::cuda::getCurrentCUDAStream();
  CUresult status = Launch(plan, stream);
  if (status != CUDA_SUCCESS) {
    plan.release(slot);
    TORCH_CHECK(false, "cuLaunchKernel failed: ", (int)status);
  }
  return WrapSlot(plan, slot);
}

py::dict bench(std::string key, std::vector<at::Tensor> inputs, int64_t iters) {
  Plan& plan = MustGet(key);
  CUstream stream = at::cuda::getCurrentCUDAStream();

  for (int i = 0; i < 20; i++) {
    size_t slot = plan.acquire();
    FillIn(plan, inputs, slot);
    Launch(plan, stream);
    plan.release(slot);
  }
  cudaStreamSynchronize(stream);

  // Batched to keep the launch queue bounded; the sync is outside the timed region per
  // batch, and batches are large enough that its cost is amortised.
  const int64_t batch = 256;
  auto t0 = std::chrono::steady_clock::now();
  int64_t done = 0;
  while (done < iters) {
    int64_t chunk = std::min(batch, iters - done);
    for (int64_t i = 0; i < chunk; i++) {
      size_t slot = plan.acquire();
      FillIn(plan, inputs, slot);
      Launch(plan, stream);
      plan.release(slot);
    }
    cudaStreamSynchronize(stream);
    done += chunk;
  }
  auto t1 = std::chrono::steady_clock::now();
  double pooled_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

  py::dict result;
  result["pooled_replay_us"] = pooled_us;
  result["pool_size"] = (int64_t)plan.buffers.size();
  return result;
}
