// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <flagos.h>
#include "aten/common.h"
#include "runtime/accelerator/ascend/acl_stream.h"
#include "runtime/allocator/caching_device_allocator.h"

#include <ATen/ATen.h>
#include <dlfcn.h>
#include <flagos_env.h>
#include <stdexcept>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <functional>
#include <initializer_list>

#include <acl/acl_base_rt.h>
#include <acl/acl_rt.h>
#include <aclnn/acl_meta.h>

namespace at::native::flagos::ascend {

inline aclDataType ToAclDataType(at::ScalarType type) {
  switch (type) {
    case at::kDouble:  return ACL_DOUBLE;
    case at::kFloat:   return ACL_FLOAT;
    case at::kHalf:    return ACL_FLOAT16;
    case at::kBFloat16: return ACL_BF16;
    case at::kInt:     return ACL_INT32;
    case at::kLong:    return ACL_INT64;
    case at::kShort:   return ACL_INT16;
    case at::kChar:    return ACL_INT8;
    case at::kByte:    return ACL_UINT8;
    case at::kBool:    return ACL_BOOL;
    default:
      TORCH_CHECK(false, "Unsupported dtype for ACL: ", type);
  }
}

struct AclTensorWrapper {
  aclTensor* acl_tensor = nullptr;
  // aclCreateTensor stores *pointers* to the shape/stride/storage arrays rather
  // than copying them, but those pointers only need to stay valid until
  // aclnn<Op>GetWorkspaceSize has run: that call bakes the shapes into the
  // executor, after which the arrays are dead (verified empirically — freeing
  // and poisoning them between GetWorkspaceSize and the execute call still
  // yields correct output). EXEC_ASCEND_CMD calls GetWorkspaceSize while this
  // wrapper is still in scope, so the arrays can live INLINE in the wrapper
  // (stack) instead of the old heap-allocate-and-leak scheme. Inline storage
  // covers the common case (ndim <= kInlineDims); larger ranks fall back to a
  // heap buffer freed in the destructor. This removes 3 heap allocs per tensor
  // (9 per binary op) that previously leaked on every dispatch.
  static constexpr int kInlineDims = 8;
  int64_t sizes_inl_[kInlineDims];
  int64_t strides_inl_[kInlineDims];
  int64_t storage_dim_ = 0;
  int64_t* sizes_ = nullptr;    // -> sizes_inl_ or heap
  int64_t* strides_ = nullptr;  // -> strides_inl_ or heap
  uint64_t ndim_ = 0;
  bool heap_ = false;
  // Saved so a moved-into wrapper can rebuild its aclTensor pointing at its OWN
  // inline buffers (aclCreateTensor stores pointers into sizes_/strides_, which
  // move with the object; without rebuild they would dangle to the source's
  // buffers). Only used on the vector-storage path (cat/index).
  aclDataType dtype_ = ACL_FLOAT;
  int64_t offset_ = 0;
  aclFormat format_ = ACL_FORMAT_ND;
  void* storage_ptr_ = nullptr;

  // `fmt` overrides the aclFormat. Defaults to ACL_FORMAT_ND; pass e.g.
  // ACL_FORMAT_NCHW for ops (avg_pool2d, conv) that reject ND 4-D inputs.
  AclTensorWrapper(const at::Tensor& tensor, aclFormat fmt = ACL_FORMAT_ND) {
    if (!tensor.defined()) {
      acl_tensor = nullptr;
      return;
    }

    auto sz = tensor.sizes();
    auto st = tensor.strides();
    ndim_ = static_cast<uint64_t>(sz.size());
    if (ndim_ <= static_cast<uint64_t>(kInlineDims)) {
      sizes_ = sizes_inl_;
      strides_ = strides_inl_;
    } else {
      sizes_ = new int64_t[ndim_];
      strides_ = new int64_t[ndim_];
      heap_ = true;
    }
    std::copy(sz.begin(), sz.end(), sizes_);
    std::copy(st.begin(), st.end(), strides_);

    offset_ = tensor.storage_offset();
    dtype_ = ToAclDataType(tensor.scalar_type());
    format_ = fmt;
    storage_dim_ = static_cast<int64_t>(
        tensor.storage().nbytes() / tensor.element_size());
    storage_ptr_ = const_cast<void*>(tensor.storage().data());

    acl_tensor = aclCreateTensor(
        sizes_, ndim_, dtype_, strides_, offset_, format_,
        &storage_dim_, static_cast<uint64_t>(1), storage_ptr_);
  }

  // The shape/stride arrays are only consumed by GetWorkspaceSize (already run
  // by the time this destructor fires at end of the EXEC_ASCEND_CMD scope), so
  // it is safe to release everything here. Destroy the aclTensor and free the
  // heap fallback (inline storage needs no free).
  ~AclTensorWrapper() {
    if (acl_tensor) {
      aclDestroyTensor(acl_tensor);
    }
    if (heap_) {
      delete[] sizes_;
      delete[] strides_;
    }
  }

  // Move constructor: needed so wrappers can live in std::vector (cat/index).
  // The aclTensor stores pointers into sizes_/strides_; for inline storage those
  // buffers move WITH the object to a new address, so we rebuild the aclTensor
  // to point at the destination's own inline buffers. Heap storage can just
  // transfer the pointer. The source is left empty (acl_tensor=nullptr) so its
  // destructor is a no-op.
  AclTensorWrapper(AclTensorWrapper&& o) noexcept {
    ndim_ = o.ndim_;
    storage_dim_ = o.storage_dim_;
    dtype_ = o.dtype_; offset_ = o.offset_; format_ = o.format_;
    storage_ptr_ = o.storage_ptr_;
    heap_ = o.heap_;
    if (heap_) {
      sizes_ = o.sizes_;
      strides_ = o.strides_;
      acl_tensor = o.acl_tensor;  // still points at the (unmoved) heap buffers
    } else {
      std::copy(o.sizes_inl_, o.sizes_inl_ + ndim_, sizes_inl_);
      std::copy(o.strides_inl_, o.strides_inl_ + ndim_, strides_inl_);
      sizes_ = sizes_inl_;
      strides_ = strides_inl_;
      // Rebuild: the source's aclTensor referenced the source's inline buffers.
      if (o.acl_tensor) {
        aclDestroyTensor(o.acl_tensor);
        acl_tensor = aclCreateTensor(
            sizes_, ndim_, dtype_, strides_, offset_, format_,
            &storage_dim_, static_cast<uint64_t>(1), storage_ptr_);
      }
    }
    o.acl_tensor = nullptr;
    o.heap_ = false;
  }

  AclTensorWrapper(const AclTensorWrapper&) = delete;
  AclTensorWrapper& operator=(const AclTensorWrapper&) = delete;
  AclTensorWrapper& operator=(AclTensorWrapper&&) = delete;

  const aclTensor* get() const { return acl_tensor; }
};

// Defer reuse of an aclnn scratch-workspace block until the given stream has
// finished the op that consumes it. Records an event on the stream via the
// caching allocator so free_block holds the block back until the event fires.
// No-op when the caching allocator is disabled (passthrough alloc frees the
// block straight back to the device, which cannot be reused before the sync
// on the next host-visible read).
inline void RecordWorkspaceStream(const at::Tensor& workspace, aclrtStream stream) {
  if (!c10::flagos::CachingDeviceAllocator::is_enabled()) {
    // Passthrough allocator aclrtFree's the block on host immediately, so it
    // could be remalloc'd and overwritten before the kernel drains. Without a
    // caching pool to defer reuse, the only safe option is a full sync.
    aclrtSynchronizeStream(stream);
    return;
  }
  auto* alloc = c10::flagos::GetCachingAllocator();
  alloc->record_stream(workspace.storage().data_ptr(),
                       reinterpret_cast<Stream_t>(stream));
}

inline void* GetOpApiLibHandle() {
  static void* handle = []() -> void* {
    void* h = dlopen("libopapi.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
      const char* err = dlerror();
      throw std::runtime_error(
          std::string("Failed to load libopapi.so: ") + (err ? err : "unknown error"));
    }
    // Call Init() to initialize libopapi.so
    typedef void (*InitFunc)();
    InitFunc initFunc = reinterpret_cast<InitFunc>(dlsym(h, "Init"));
    if (initFunc) {
      initFunc();
    }
    return h;
  }();
  return handle;
}

inline void* GetOpBaseLibHandle() {
  static void* handle = []() -> void* {
    void* h = dlopen("libnnopbase.so", RTLD_LAZY);
    if (!h) {
      const char* err = dlerror();
      throw std::runtime_error(
          std::string("Failed to load libnnopbase.so: ") + (err ? err : "unknown error"));
    }
    return h;
  }();
  return handle;
}

inline void GetApiFunc(const char* api_name, const char* workspace_name,
                       void*& api_func, void*& workspace_func) {
  void* handle = GetOpApiLibHandle();
  if (!api_func) {
    api_func = dlsym(handle, api_name);
  }
  if (!workspace_func) {
    workspace_func = dlsym(handle, workspace_name);
  }
}


struct AclScalarWrapper {
  aclScalar* acl_scalar = nullptr;
  // Storage for the scalar value (must outlive aclScalar*)
  union {
    double d;
    float f;
    int64_t i64;
    int32_t i32;
    int16_t i16;
    int8_t i8;
    uint8_t u8;
    bool b;
    at::Half h;
    at::BFloat16 bf16;
  } value_storage;

  // Absent-optional form: leaves acl_scalar null. aclnn ops that take an
  // optional scalar (e.g. aclnnClamp's clipValueMin/Max) read a null pointer
  // as "not supplied", which is what an empty std::optional must map to.
  AclScalarWrapper() = default;

  AclScalarWrapper(const at::Scalar& scalar, at::ScalarType dtype) {
    // aclCreateScalar stores a pointer to the value, so we must keep it alive
    switch (dtype) {
      case at::kDouble:
        value_storage.d = scalar.toDouble();
        acl_scalar = aclCreateScalar(&value_storage.d, ACL_DOUBLE);
        break;
      case at::kFloat:
        value_storage.f = scalar.toFloat();
        acl_scalar = aclCreateScalar(&value_storage.f, ACL_FLOAT);
        break;
      case at::kHalf:
        value_storage.h = static_cast<at::Half>(scalar.toFloat());
        acl_scalar = aclCreateScalar(&value_storage.h, ACL_FLOAT16);
        break;
      case at::kBFloat16:
        value_storage.bf16 = static_cast<at::BFloat16>(scalar.toFloat());
        acl_scalar = aclCreateScalar(&value_storage.bf16, ACL_BF16);
        break;
      case at::kLong:
        value_storage.i64 = scalar.toLong();
        acl_scalar = aclCreateScalar(&value_storage.i64, ACL_INT64);
        break;
      case at::kInt:
        value_storage.i32 = static_cast<int32_t>(scalar.toLong());
        acl_scalar = aclCreateScalar(&value_storage.i32, ACL_INT32);
        break;
      case at::kShort:
        value_storage.i16 = static_cast<int16_t>(scalar.toLong());
        acl_scalar = aclCreateScalar(&value_storage.i16, ACL_INT16);
        break;
      case at::kChar:
        value_storage.i8 = static_cast<int8_t>(scalar.toLong());
        acl_scalar = aclCreateScalar(&value_storage.i8, ACL_INT8);
        break;
      case at::kByte:
        value_storage.u8 = static_cast<uint8_t>(scalar.toLong());
        acl_scalar = aclCreateScalar(&value_storage.u8, ACL_UINT8);
        break;
      case at::kBool:
        value_storage.b = scalar.toBool();
        acl_scalar = aclCreateScalar(&value_storage.b, ACL_BOOL);
        break;
      default:
        TORCH_CHECK(false, "Unsupported scalar type for ACL: ", dtype);
    }
  }

  ~AclScalarWrapper() = default;

  const aclScalar* get() const { return acl_scalar; }
};

struct AclIntArrayWrapper {
  aclIntArray* acl_array = nullptr;

  AclIntArrayWrapper(at::IntArrayRef arr) {
    acl_array = aclCreateIntArray(arr.data(), arr.size());
  }

  ~AclIntArrayWrapper() = default;

  const aclIntArray* get() const { return acl_array; }
};

struct AclBoolArrayWrapper {
  aclBoolArray* acl_array = nullptr;
  // aclCreateBoolArray stores a pointer to the data, so it must outlive the
  // aclBoolArray (bool[] cannot be borrowed from a temporary).
  std::vector<uint8_t> storage_;

  AclBoolArrayWrapper(at::ArrayRef<bool> arr) {
    storage_.assign(arr.begin(), arr.end());
    acl_array = aclCreateBoolArray(
        reinterpret_cast<const bool*>(storage_.data()), storage_.size());
  }

  ~AclBoolArrayWrapper() = default;

  const aclBoolArray* get() const { return acl_array; }
};

struct AclTensorListWrapper {
  aclTensorList* acl_list = nullptr;

  AclTensorListWrapper(const std::vector<const aclTensor*>& tensors) {
    acl_list = aclCreateTensorList(tensors.data(), tensors.size());
  }

  ~AclTensorListWrapper() = default;

  void release() { acl_list = nullptr; }

  const aclTensorList* get() const { return acl_list; }
};

// ==========================================================================
// Repeatable-executor cache (the torch_npu-parity fast path).
//
// For eager decode the op set and shapes are constant, so the aclOpExecutor
// built by aclnn<Op>GetWorkspaceSize can be reused across steps: cache it keyed
// by (op, tensor signatures, scalar bytes), then on a hit only rebind the
// tensor data addresses (aclSetInput/OutputTensorAddr) and execute -- skipping
// both GetWorkspaceSize and aclCreateTensor. Verified on CANN 9.0.0:
//   * aclSetAclOpExecutorRepeatable(ex) returns 0 for aclnnMul/aclnnAdd.
//   * input/output tensor addresses are indexed over TENSORS ONLY (interleaved
//     scalars consume no index) and inputs vs outputs index separately.
//   * REBIND REQUIRES THE ORIGINAL aclTensor OBJECTS: passing a freshly-created
//     aclTensor (even same shape) yields wrong output. So the cache OWNS the
//     aclTensors that built the executor and reuses them on every hit.
//   * scalars are baked into the executor at GetWorkspaceSize -> they are NOT
//     rebindable and MUST be part of the cache key (alpha, etc.).
//   * storage_offset is baked into the aclTensor -> also part of the key; the
//     rebind address is the storage BASE ptr (offset applied internally).
// ==========================================================================

// The op-specific GetWorkspaceSize entry, called through a variadic pointer
// (same ABI contract as EXEC_ASCEND_CMD: int64/bool/pointer args are fine,
// by-value float/double are NOT -- pass scalars as aclScalar*).
typedef int (*GwsFunc)(...);

inline void GetRebindFuncs(void*& set_repeatable, void*& set_in_addr,
                           void*& set_out_addr) {
  void* h = GetOpBaseLibHandle();
  if (!set_repeatable) set_repeatable = dlsym(h, "aclSetAclOpExecutorRepeatable");
  if (!set_in_addr)    set_in_addr    = dlsym(h, "aclSetInputTensorAddr");
  if (!set_out_addr)   set_out_addr   = dlsym(h, "aclSetOutputTensorAddr");
}

struct CachedExecKey {
  const char* api = nullptr;   // static per-call-site string ptr (unique id)
  uint64_t sig = 0;            // 64-bit hash of tensor sigs + scalar bytes
  int device = -1;             // device the executor was built on
  bool operator==(const CachedExecKey& o) const {
    return api == o.api && sig == o.sig && device == o.device;
  }
};

struct CachedExecKeyHash {
  size_t operator()(const CachedExecKey& k) const {
    return std::hash<const void*>()(static_cast<const void*>(k.api)) ^
           (static_cast<size_t>(k.sig) * 0x9E3779B97F4A7C15ULL) ^
           (static_cast<size_t>(k.device) * 0xC2B2AE3D27D4EB4FULL);
  }
};

struct CachedExecEntry {
  aclOpExecutor* executor = nullptr;
  uint64_t workspace_size = 0;
  // These OWN the aclTensors the executor is bound to. Never moved after the
  // executor is built (reserve() below prevents vector realloc), so the
  // aclTensor* the executor holds stay valid for the cache's lifetime. The
  // entry itself only ever moves via unordered_map node relocation, which
  // moves the vector's heap buffer pointer but not the elements, so the
  // AclTensorWrapper objects (and their inline buffers) never physically move.
  std::vector<AclTensorWrapper> in_tensors;
  std::vector<AclTensorWrapper> out_tensors;
  // Cached scratch workspace (allocated once on first use, reused on every
  // hit). Safe because all ops for a given entry execute on the same stream
  // serially -- by the time call N+1 reaches device, call N has already
  // consumed and released the workspace. Eliminates at::empty + record_stream
  // overhead (~6-25 us) on every cache hit for ops with workspace_size > 0.
  at::Tensor workspace_tensor;
};

// 64-bit FNV-1a over the shape/stride/dtype/offset of each tensor arg plus the
// raw bytes of any scalar args. Collisions would return a wrong executor, so a
// strong 64-bit hash is used (matches torch_npu's own hash-keyed PTA cache).
struct SigHasher {
  uint64_t h = 1469598103934665603ULL;
  void bytes(const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
  }
  template <typename T> void val(const T& v) { bytes(&v, sizeof(T)); }
  void tensor(const at::Tensor& t) {
    if (!t.defined()) { uint8_t z = 0; bytes(&z, 1); return; }
    int64_t nd = t.dim();
    val(nd);
    for (auto s : t.sizes()) val(s);
    for (auto s : t.strides()) val(s);
    int64_t off = t.storage_offset();
    val(off);
    int32_t dt = static_cast<int32_t>(t.scalar_type());
    val(dt);
  }
};

inline std::unordered_map<CachedExecKey, CachedExecEntry, CachedExecKeyHash>&
GetExecCache() {
  static thread_local
      std::unordered_map<CachedExecKey, CachedExecEntry, CachedExecKeyHash> cache;
  return cache;
}

// --- optional hit/miss stats (FLAGOS_LOG=op_cache) --------------------------
inline bool CacheStatsEnabled() {
  static const bool on = LogEnabled("op_cache");
  return on;
}

struct CacheStat { uint64_t hit = 0; uint64_t miss = 0; };

inline std::unordered_map<const char*, CacheStat>& GetCacheStats() {
  static std::unordered_map<const char*, CacheStat> s;
  return s;
}

inline void RecordCacheStat(const char* api, bool hit) {
  auto& s = GetCacheStats()[api];
  if (hit) ++s.hit; else ++s.miss;
  static bool registered = [] {
    std::atexit([] {
      fprintf(stderr, "\n=== FLAGOS exec-cache stats (api: hit/miss) ===\n");
      for (auto& kv : GetCacheStats()) {
        uint64_t tot = kv.second.hit + kv.second.miss;
        fprintf(stderr, "  %-28s hit=%-8llu miss=%-6llu (%.1f%% hit)\n",
                kv.first, (unsigned long long)kv.second.hit,
                (unsigned long long)kv.second.miss,
                tot ? 100.0 * kv.second.hit / tot : 0.0);
      }
    });
    return true;
  }();
  (void)registered;
}

// Cached-executor dispatch. `build` calls the op-specific GetWorkspaceSize via
// the variadic func ptr, interleaving any scalar args (captured by reference)
// in the correct positions, using the OWNED input/output aclTensors passed to
// it. On a miss the executor is built + marked repeatable + stored; on a hit
// only the tensor addresses are rebound. `inputs`/`outputs` list the at::Tensor
// args in tensor-index order (inputs and outputs indexed separately).
template <typename BuildFn>
void ExecAscendCached(const char* api_name, const char* ws_name,
                      void*& opApiFuncAddr, void*& getWsFuncAddr,
                      uint64_t sig,
                      std::initializer_list<const at::Tensor*> inputs,
                      std::initializer_list<const at::Tensor*> outputs,
                      BuildFn&& build) {
  GetApiFunc(api_name, ws_name, opApiFuncAddr, getWsFuncAddr);
  TORCH_CHECK(opApiFuncAddr && getWsFuncAddr,
      "Failed to load symbols for ", api_name, ": ", dlerror());

  static void* setRepeatableAddr = nullptr;
  static void* setInAddrAddr = nullptr;
  static void* setOutAddrAddr = nullptr;
  GetRebindFuncs(setRepeatableAddr, setInAddrAddr, setOutAddrAddr);

  typedef int (*SetRepeatableFunc)(aclOpExecutor*);
  typedef int (*SetAddrFunc)(aclOpExecutor*, size_t, aclTensor*, void*);
  auto setRepeatable = reinterpret_cast<SetRepeatableFunc>(setRepeatableAddr);
  auto setInAddr = reinterpret_cast<SetAddrFunc>(setInAddrAddr);
  auto setOutAddr = reinterpret_cast<SetAddrFunc>(setOutAddrAddr);

  // The executor and the cached workspace below are both device-bound: an
  // aclOpExecutor is built against the device current at GetWorkspaceSize time
  // and its workspace is allocated on that device. Keying the cache without the
  // device let a call on device 1 reuse an executor (and workspace) built on
  // device 0, which either failed outright or produced garbage.
  int device_index = 0;
  ::GetDevice(&device_index);
  auto acl_stream = GetCurrentAclStreamForDevice(device_index);
  auto& cache = GetExecCache();
  CachedExecKey key{api_name, sig, device_index};

  auto it = cache.find(key);
  // Optional hit/miss instrumentation (FLAGOS_LOG=op_cache): prints per-op
  // hit/miss counts at process exit. Zero cost when the env var is unset.
  if (CacheStatsEnabled()) RecordCacheStat(api_name, it != cache.end());
  // Cap resident executors: eager decode uses a bounded shape set, but a long
  // varying-shape workload (e.g. prefill over growing sequence lengths) could
  // otherwise accumulate executors without bound. Past the cap, run uncached
  // (the executor for this shape is built, used once, and destroyed) so the
  // resident set stays fixed while still serving the hot decode shapes.
  static constexpr size_t kMaxCachedExecutors = 4096;
  if (it == cache.end() && cache.size() >= kMaxCachedExecutors) {
    auto gws = reinterpret_cast<GwsFunc>(getWsFuncAddr);
    std::vector<AclTensorWrapper> in_t, out_t;
    in_t.reserve(inputs.size());
    for (const at::Tensor* t : inputs) in_t.emplace_back(*t);
    out_t.reserve(outputs.size());
    for (const at::Tensor* t : outputs) out_t.emplace_back(*t);
    uint64_t ws = 0; aclOpExecutor* ex = nullptr;
    int ret = build(gws, in_t, out_t, &ws, &ex);
    TORCH_CHECK(ret == 0, api_name, "GetWorkspaceSize failed, ret=", ret);
    void* wsa = nullptr; at::Tensor wst;
    if (ws > 0) {
      wst = at::empty({static_cast<int64_t>(ws)},
          at::TensorOptions().dtype(at::kByte).device(at::kPrivateUse1));
      wsa = wst.data_ptr();
    }
    typedef int (*ExecFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);
    auto ef = reinterpret_cast<ExecFunc>(opApiFuncAddr);
    int er = ef(wsa, ws, ex, acl_stream);
    TORCH_CHECK(er == 0, api_name, " execution failed, ret=", er);
    if (ws > 0) RecordWorkspaceStream(wst, acl_stream);
    return;
  }
  if (it == cache.end()) {
    // ---- miss: build executor bound to owned tensors, mark repeatable ----
    auto res = cache.emplace(key, CachedExecEntry{});
    CachedExecEntry& e = res.first->second;
    e.in_tensors.reserve(inputs.size());
    for (const at::Tensor* t : inputs) e.in_tensors.emplace_back(*t);
    e.out_tensors.reserve(outputs.size());
    for (const at::Tensor* t : outputs) e.out_tensors.emplace_back(*t);

    auto gws = reinterpret_cast<GwsFunc>(getWsFuncAddr);
    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    int ret = build(gws, e.in_tensors, e.out_tensors, &ws, &ex);
    TORCH_CHECK(ret == 0, api_name, "GetWorkspaceSize failed, ret=", ret);

    // Best-effort: mark reusable. If the op cannot be made repeatable, fall
    // back to running it once uncached (executor stays valid for this call).
    if (setRepeatable) {
      int rr = setRepeatable(ex);
      if (rr != 0) {
        // Not repeatable: run once, then drop from cache to avoid rebinding a
        // non-repeatable executor on a later hit.
        void* wsa = nullptr; at::Tensor wst;
        if (ws > 0) {
          wst = at::empty({static_cast<int64_t>(ws)},
              at::TensorOptions().dtype(at::kByte).device(at::kPrivateUse1));
          wsa = wst.data_ptr();
        }
        typedef int (*ExecFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);
        auto ef = reinterpret_cast<ExecFunc>(opApiFuncAddr);
        int er = ef(wsa, ws, ex, acl_stream);
        TORCH_CHECK(er == 0, api_name, " execution failed, ret=", er);
        if (ws > 0) RecordWorkspaceStream(wst, acl_stream);
        cache.erase(key);
        return;
      }
    }
    e.executor = ex;
    e.workspace_size = ws;
    it = res.first;
  } else {
    // ---- hit: rebind the owned tensors' data addresses to current storage ----
    size_t i = 0;
    for (const at::Tensor* t : inputs) {
      void* addr = const_cast<void*>(t->storage().data());
      int r = setInAddr(it->second.executor, i, it->second.in_tensors[i].acl_tensor, addr);
      TORCH_CHECK(r == 0, api_name, " aclSetInputTensorAddr failed, ret=", r);
      ++i;
    }
    size_t j = 0;
    for (const at::Tensor* t : outputs) {
      void* addr = const_cast<void*>(t->storage().data());
      int r = setOutAddr(it->second.executor, j, it->second.out_tensors[j].acl_tensor, addr);
      TORCH_CHECK(r == 0, api_name, " aclSetOutputTensorAddr failed, ret=", r);
      ++j;
    }
  }

  // ---- execute (both paths): cached workspace, no per-hit allocation ----
  // Allocate workspace once on the first (miss) call and reuse it on every
  // subsequent hit. Safety: all ops sharing this entry execute on the same
  // single default ACL stream. The stream guarantees serial device execution,
  // so call N's kernel has finished consuming the workspace before call N+1's
  // kernel starts -- no concurrent access, no stream-record needed on hits.
  CachedExecEntry& e = it->second;
  void* workspace_addr = nullptr;
  if (e.workspace_size > 0) {
    if (!e.workspace_tensor.defined()) {
      e.workspace_tensor = at::empty({static_cast<int64_t>(e.workspace_size)},
          at::TensorOptions().dtype(at::kByte).device(at::kPrivateUse1));
    }
    workspace_addr = e.workspace_tensor.data_ptr();
  }
  typedef int (*ExecFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);
  auto executeFunc = reinterpret_cast<ExecFunc>(opApiFuncAddr);
  int exec_ret = executeFunc(workspace_addr, e.workspace_size, e.executor, acl_stream);
  TORCH_CHECK(exec_ret == 0, api_name, " execution failed, ret=", exec_ret);
}

} // namespace at::native::flagos::ascend

// EXEC_ASCEND_CMD_SIG(aclnn_api, ws_sig, ...) is EXEC_ASCEND_CMD with an
// explicitly-typed GetWorkspaceSize pointer. `ws_sig` is the callee's full
// parenthesized parameter list, e.g.
//
//   EXEC_ASCEND_CMD_SIG(aclnnInplaceNormal,
//       (const aclTensor*, float, float, int64_t, int64_t,
//        uint64_t*, aclOpExecutor**),
//       acl_out.get(), 0.0f, 1.0f, seed, 0)
//
// Use this for ANY aclnn entry that declares a `float` (or any other narrow
// arithmetic) parameter. The default macro calls through `int (*)(...)`, so
// every argument goes through varargs default promotion: a `float` is passed as
// a 64-bit `double` in a v-register, and a callee prototyped `float std` reads
// only the low 32 bits of that double. For std=1.0 (0x3FF0000000000000) the low
// half is exactly 0, so aclnnInplaceNormal saw std=0 and torch.randn() silently
// returned all zeros. Passing `0.0`/`1.0` instead of `0.0f`/`1.0f` does NOT help
// -- the promotion makes both spellings identical. Only a correctly-typed
// pointer puts the bits where the callee looks. Most aclnn entries declare
// `double` (aclnnInplaceUniform, aclnnInplaceRandom), which is why the variadic
// path works for them and this variant is only needed for the exceptions.
#define EXEC_ASCEND_CMD_SIG(aclnn_api, ws_sig, ...)                           \
  EXEC_ASCEND_CMD_IMPL_(aclnn_api, int(*) ws_sig, __VA_ARGS__)

#define EXEC_ASCEND_CMD(aclnn_api, ...)                                       \
  EXEC_ASCEND_CMD_IMPL_(aclnn_api, int (*)(...), __VA_ARGS__)

#define EXEC_ASCEND_CMD_IMPL_(aclnn_api, ws_fn_type, ...)                     \
  do {                                                                        \
    static void* opApiFuncAddr = nullptr;                                     \
    static void* getWorkspaceSizeFuncAddr = nullptr;                          \
    at::native::flagos::ascend::GetApiFunc(                                   \
        #aclnn_api, #aclnn_api "GetWorkspaceSize",                            \
        opApiFuncAddr, getWorkspaceSizeFuncAddr);                              \
                                                                              \
    auto acl_stream = at::native::flagos::ascend::GetCurrentAclStream();      \
                                                                              \
    uint64_t workspace_size = 0;                                              \
    aclOpExecutor* executor = nullptr;                                        \
                                                                              \
    TORCH_CHECK(getWorkspaceSizeFuncAddr && opApiFuncAddr,                     \
        "Failed to load symbols for " #aclnn_api ": ", dlerror());            \
                                                                              \
    using GetWorkspaceSizeFunc = ws_fn_type;                                   \
    auto getWorkspaceSize =                                                   \
        reinterpret_cast<GetWorkspaceSizeFunc>(getWorkspaceSizeFuncAddr);      \
    int ws_ret = getWorkspaceSize(__VA_ARGS__, &workspace_size, &executor);    \
    TORCH_CHECK(ws_ret == 0,                                                  \
        #aclnn_api "GetWorkspaceSize failed, ret=", ws_ret);                  \
                                                                              \
    void* workspace_addr = nullptr;                                           \
    at::Tensor workspace_tensor;                                              \
    if (workspace_size > 0) {                                                 \
      workspace_tensor = at::empty({static_cast<int64_t>(workspace_size)},   \
          at::TensorOptions().dtype(at::kByte).device(at::kPrivateUse1));    \
      workspace_addr = workspace_tensor.data_ptr();                           \
    }                                                                         \
                                                                              \
    typedef int (*ExecFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);    \
    auto executeFunc = reinterpret_cast<ExecFunc>(opApiFuncAddr);             \
    int exec_ret = executeFunc(                                               \
        workspace_addr, workspace_size, executor, acl_stream);                \
    TORCH_CHECK(exec_ret == 0, #aclnn_api " execution failed, ret=",         \
        exec_ret);                                                            \
    /* No per-op aclrtSynchronizeStream: ops enqueue asynchronously on the    \
     * shared default stream (FIFO on-device), overlapping host dispatch with \
     * device compute. Correctness is preserved by (1) draining the default   \
     * stream before any host-visible read (D2H/H2D/D2D memcpy; see           \
     * runtime/accelerator/ascend/memory.cc), and (2) stream-ordering the     \
     * scratch workspace below.                                               \
     *                                                                        \
     * The workspace tensor is freed on the host as soon as this scope ends,  \
     * returning its block to the caching pool. Under async dispatch the      \
     * kernel may still be reading that scratch when a later op reuses the    \
     * block, corrupting results. record_stream defers the block's reuse      \
     * until the default stream has passed this point, which fixes the race   \
     * without a full sync. Inputs/outputs need no such guard: they stay      \
     * live (referenced by the producing/consuming ops) and are only read     \
     * back via the drained memcpy path. */                                   \
    if (workspace_size > 0) {                                                  \
      at::native::flagos::ascend::RecordWorkspaceStream(                       \
          workspace_tensor, acl_stream);                                      \
    }                                                                          \
  } while (false)

