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

// Operator-library runtime for the BPU. See bpu_oplib.h for the design.
//
// The board ships libdnn.so and libhbucp.so without headers, so the handful of
// entry points this layer needs are declared here, and the libraries are
// opened with dlopen rather than linked: a build host without them still links,
// and the first Lookup() on a board without FLAGOS_BPU_OPLIB answers
// "unsupported" without ever touching them.
//
// The struct layouts below mirror the ones the Python side builds with ctypes
// in torch_fl/accelerator/bpu/submit.py, which were checked against the
// runtime: DNNTensor.properties sits at offset 24, tensorType at 44 and
// alignedByteSize at 88, and UCPSchedParam is 32 bytes with backend at 16.

#include "bpu_oplib.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace flagos {
namespace bpu {
namespace oplib {
namespace {

constexpr int kMaxDims = 8;      // HB_DNN_MAX_DIMS
constexpr int kF32 = 7;          // hbDNNDataType::HB_DNN_TENSOR_TYPE_F32
constexpr int kCacheClean = 2;
constexpr int kCacheInvalidate = 1;
constexpr int kCoreAny = 0x80;

// Layouts mirror what the Python side builds with ctypes in
// torch_fl/accelerator/bpu/submit.py, which was checked field by field
// against the runtime: DNNTensorShape puts dimensionSize BEFORE numDimensions,
// strides are int64, and the quantization block is two counted pointers, not
// arrays. Getting these wrong corrupts the stack rather than failing a call.
struct DNNShape {
  int32_t dimensionSize[kMaxDims];
  int32_t numDimensions;
};

struct DNNQuantiScale {
  int32_t scaleLen;
  float* scaleData;
  int32_t zeroPointLen;
  int32_t* zeroPointData;
};

struct DNNTensorProperties {
  DNNShape validShape;
  int32_t tensorType;
  DNNQuantiScale scale;
  int32_t quantiType;
  int32_t quantizeAxis;
  int64_t alignedByteSize;
  int64_t stride[kMaxDims];
};

struct UCPSysMem {
  uint64_t phyAddr;
  void* virAddr;
  uint64_t memSize;
};

struct DNNTensor {
  UCPSysMem sysMem;
  DNNTensorProperties properties;
};

struct UCPSchedParam {
  int32_t priority;
  int64_t customId;
  uint64_t backend;
  uint32_t deviceId;
};

// Pinned to the sizes the Python side asserts, so a layout change here breaks
// the build instead of corrupting a call.
static_assert(sizeof(DNNShape) == 36, "DNNShape layout drifted");
static_assert(sizeof(UCPSysMem) == 24, "UCPSysMem layout drifted");
static_assert(sizeof(UCPSchedParam) == 32, "UCPSchedParam layout drifted");

using InitFromFilesFn = int (*)(void**, const char* const*, int);
using GetModelHandleFn = int (*)(void**, void*, const char*);
using ReleaseFn = int (*)(void*);
using GetPropsFn = int (*)(DNNTensorProperties*, const void*, int);
using InferV2Fn = int (*)(void**, DNNTensor*, DNNTensor*, const void*);
using MallocFn = int (*)(UCPSysMem*, uint64_t, int);
using FreeFn = int (*)(UCPSysMem*);
using MemFlushFn = int (*)(UCPSysMem*, int);
using SubmitFn = int (*)(void*, const UCPSchedParam*);
using WaitFn = int (*)(void*, int);
using ReleaseTaskFn = int (*)(void*);

struct Libs {
  InitFromFilesFn init_from_files = nullptr;
  GetModelHandleFn get_model_handle = nullptr;
  ReleaseFn dnn_release = nullptr;
  GetPropsFn get_in_props = nullptr;
  GetPropsFn get_out_props = nullptr;
  InferV2Fn infer_v2 = nullptr;
  MallocFn malloc_cached = nullptr;
  FreeFn free_mem = nullptr;
  MemFlushFn mem_flush = nullptr;
  SubmitFn submit = nullptr;
  WaitFn wait = nullptr;
  ReleaseTaskFn release_task = nullptr;

  bool ok() const {
    return init_from_files && get_model_handle && get_in_props && get_out_props &&
           infer_v2 && malloc_cached && free_mem && mem_flush && submit && wait &&
           release_task;
  }
};

Libs& Libraries() {
  static Libs libs;
  static std::once_flag once;
  std::call_once(once, [] {
    void* dnn = dlopen("libdnn.so", RTLD_NOW | RTLD_GLOBAL);
    void* ucp = dlopen("libhbucp.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dnn || !ucp) {
      return;
    }
    auto sym = [](void* h, const char* n) { return dlsym(h, n); };
    libs.init_from_files = reinterpret_cast<InitFromFilesFn>(sym(dnn, "hbDNNInitializeFromFiles"));
    libs.get_model_handle = reinterpret_cast<GetModelHandleFn>(sym(dnn, "hbDNNGetModelHandle"));
    libs.dnn_release = reinterpret_cast<ReleaseFn>(sym(dnn, "hbDNNRelease"));
    libs.get_in_props =
        reinterpret_cast<GetPropsFn>(sym(dnn, "hbDNNGetInputTensorProperties"));
    libs.get_out_props =
        reinterpret_cast<GetPropsFn>(sym(dnn, "hbDNNGetOutputTensorProperties"));
    libs.infer_v2 = reinterpret_cast<InferV2Fn>(sym(dnn, "hbDNNInferV2"));
    libs.malloc_cached = reinterpret_cast<MallocFn>(sym(ucp, "hbUCPMallocCached"));
    libs.free_mem = reinterpret_cast<FreeFn>(sym(ucp, "hbUCPFree"));
    libs.mem_flush = reinterpret_cast<MemFlushFn>(sym(ucp, "hbUCPMemFlush"));
    libs.submit = reinterpret_cast<SubmitFn>(sym(ucp, "hbUCPSubmitTask"));
    libs.wait = reinterpret_cast<WaitFn>(sym(ucp, "hbUCPWaitTaskDone"));
    libs.release_task = reinterpret_cast<ReleaseTaskFn>(sym(ucp, "hbUCPReleaseTask"));
  });
  return libs;
}

// One artifact with its device buffers held open.
//
// Keeping them open is the whole point of this class: allocating both buffers
// per call costs 0.025 ms on the board, against a submit/wait floor near
// 0.024 ms. Eager dispatch calls the same artifact thousands of times at one
// bucket width, so the allocation is pure overhead.
struct Kernel {
  void* packed = nullptr;
  void* model = nullptr;
  UCPSysMem in_mem{};
  UCPSysMem out_mem{};
  int64_t bytes = 0;
  bool ready = false;

  ~Kernel() {
    Libs& libs = Libraries();
    if (in_mem.virAddr && libs.free_mem) {
      libs.free_mem(&in_mem);
    }
    if (out_mem.virAddr && libs.free_mem) {
      libs.free_mem(&out_mem);
    }
    if (packed && libs.dnn_release) {
      libs.dnn_release(packed);
    }
  }
};

std::mutex g_mutex;
std::unordered_map<std::string, Kernel*> g_kernels;
long g_hits = 0;
long g_misses = 0;
std::string g_summary;

// The filename carries the dtype, the model name inside it does not: the
// artifact at relu_1x512_f32.hbm declares a model called "relu_1x512".
// hbDNNGetModelHandle looks the name up, so it is built separately.
std::string ArtifactStem(const char* op, int64_t width) {
  return std::string(op) + "_1x" + std::to_string(width);
}

std::string ArtifactFile(const char* op, int64_t width) {
  return ArtifactStem(op, width) + "_f32";
}

Kernel* LoadKernel(const std::string& stem) {
  Libs& libs = Libraries();
  if (!libs.ok()) {
    return nullptr;
  }
  const char* root = Root();
  if (root == nullptr || root[0] == '\0') {
    return nullptr;
  }

  std::string path = std::string(root) + "/" + stem + ".hbm";
  FILE* probe = fopen(path.c_str(), "rb");
  if (!probe) {
    return nullptr;
  }
  fclose(probe);

  auto kernel = std::make_unique<Kernel>();
  const char* paths[1] = {path.c_str()};
  if (libs.init_from_files(&kernel->packed, paths, 1) != 0) {
    return nullptr;
  }
  std::string model_name = stem;
  const std::string suffix = "_f32";
  if (model_name.size() > suffix.size() &&
      model_name.compare(model_name.size() - suffix.size(), suffix.size(), suffix) == 0) {
    model_name.resize(model_name.size() - suffix.size());
  }
  if (libs.get_model_handle(&kernel->model, kernel->packed, model_name.c_str()) != 0) {
    return nullptr;
  }

  DNNTensorProperties in_props{};
  DNNTensorProperties out_props{};
  if (libs.get_in_props(&in_props, kernel->model, 0) != 0 ||
      libs.get_out_props(&out_props, kernel->model, 0) != 0) {
    return nullptr;
  }

  // One bucket per Fortran-order f32 tensor: bytes = width * 4.
  int64_t width = 1;
  for (int d = 0; d < in_props.validShape.numDimensions; ++d) {
    width *= in_props.validShape.dimensionSize[d];
  }
  kernel->bytes = width * static_cast<int64_t>(sizeof(float));
  if (kernel->bytes <= 0) {
    return nullptr;
  }

  if (libs.malloc_cached(&kernel->in_mem, kernel->bytes, 0) != 0) {
    return nullptr;
  }
  if (libs.malloc_cached(&kernel->out_mem, kernel->bytes, 0) != 0) {
    return nullptr;
  }
  kernel->ready = true;
  return kernel.release();
}

Kernel* Cached(const std::string& stem) {
  std::lock_guard<std::mutex> guard(g_mutex);
  auto it = g_kernels.find(stem);
  if (it != g_kernels.end()) {
    return it->second;
  }
  Kernel* kernel = LoadKernel(stem);
  if (kernel != nullptr) {
    g_kernels[stem] = kernel;
  }
  return kernel;
}

} // namespace

const char* Root() {
  const char* root = std::getenv("FLAGOS_BPU_OPLIB");
  return root ? root : "";
}

int64_t BucketFor(const char* op, int nargs, int64_t width) {
  if (op == nullptr || width <= 0) {
    return 0;
  }
  // The ladder mirrors DEFAULT_BUCKETS in build_bpu_oplib.py. Consulting the
  // files rather than a table means a library built with --shapes is honoured.
  static const int64_t kLadder[] = {8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512};
  for (int64_t bucket : kLadder) {
    if (bucket < width) {
      continue;
    }
    std::string file = ArtifactFile(op, bucket);
    std::string path = std::string(Root()) + "/" + file + ".hbm";
    FILE* probe = fopen(path.c_str(), "rb");
    if (probe) {
      fclose(probe);
      return bucket;
    }
  }
  return 0;
}

bool Run(const char* op, int nargs, const float* const* in, int64_t width,
         float* out) {
  if (op == nullptr || in == nullptr || out == nullptr || width <= 0) {
    return false;
  }
  const int64_t bucket = BucketFor(op, nargs, width);
  if (bucket == 0) {
    std::lock_guard<std::mutex> guard(g_mutex);
    ++g_misses;
    return false;
  }

  Kernel* kernel = Cached(ArtifactFile(op, bucket));
  Libs& libs = Libraries();
  if (kernel == nullptr || !kernel->ready || !libs.ok()) {
    std::lock_guard<std::mutex> guard(g_mutex);
    ++g_misses;
    return false;
  }

  const int64_t elements = bucket;
  const int64_t bytes = elements * static_cast<int64_t>(sizeof(float));
  auto* dst = static_cast<float*>(kernel->in_mem.virAddr);
  // Zero first: every library op is elementwise, so the padded lanes cannot
  // influence the ones inside the request, but they must not be stale garbage.
  std::memset(dst, 0, static_cast<size_t>(bytes));
  std::memcpy(dst, in[0], static_cast<size_t>(width * sizeof(float)));
  if (nargs == 2) {
    // A two-input artifact takes its operands as consecutive buffers.
    std::memcpy(dst + elements, in[1], static_cast<size_t>(width * sizeof(float)));
    std::memset(dst + width, 0, static_cast<size_t>((elements - width) * sizeof(float)));
  }

  if (libs.mem_flush(&kernel->in_mem, kCacheClean) != 0) {
    std::lock_guard<std::mutex> guard(g_mutex);
    ++g_misses;
    return false;
  }

  DNNTensor in_tensor{};
  DNNTensor out_tensor{};
  in_tensor.sysMem = kernel->in_mem;
  out_tensor.sysMem = kernel->out_mem;

  DNNTensorProperties in_props{};
  DNNTensorProperties out_props{};
  if (libs.get_in_props(&in_props, kernel->model, 0) != 0 ||
      libs.get_out_props(&out_props, kernel->model, 0) != 0) {
    return false;
  }
  in_props.tensorType = kF32;
  out_props.tensorType = kF32;
  in_tensor.properties = in_props;
  out_tensor.properties = out_props;

  void* handle = nullptr;
  if (libs.infer_v2(&handle, &out_tensor, &in_tensor, kernel->model) != 0) {
    std::lock_guard<std::mutex> guard(g_mutex);
    ++g_misses;
    return false;
  }
  UCPSchedParam sched{};
  sched.priority = 0;
  sched.customId = 0;
  sched.backend = kCoreAny;
  sched.deviceId = 0;
  if (libs.submit(handle, &sched) != 0) {
    libs.release_task(handle);
    std::lock_guard<std::mutex> guard(g_mutex);
    ++g_misses;
    return false;
  }
  const int rc = libs.wait(handle, -1);
  libs.release_task(handle);
  if (rc != 0) {
    std::lock_guard<std::mutex> guard(g_mutex);
    ++g_misses;
    return false;
  }
  libs.mem_flush(&kernel->out_mem, kCacheInvalidate);

  std::memcpy(out, kernel->out_mem.virAddr, static_cast<size_t>(width * sizeof(float)));
  std::lock_guard<std::mutex> guard(g_mutex);
  ++g_hits;
  return true;
}

const char* Summary() {
  std::lock_guard<std::mutex> guard(g_mutex);
  char buffer[160];
  std::snprintf(buffer, sizeof(buffer), "bpu oplib: %ld on device, %ld missed", g_hits,
                g_misses);
  g_summary.assign(buffer);
  return g_summary.c_str();
}

} // namespace oplib
} // namespace bpu
} // namespace flagos