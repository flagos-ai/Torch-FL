// Copyright (c) 2026, BAAI. All rights reserved.
//
// Marshalling helpers for the Iluvatar BI-V150 (corex) operator backend.
//
// This backend is *not* the CUDA-boxing path and is not a FlagGems shim: it
// calls the vendor's kernel libraries directly, the way Kunlunxin calls xdnn.
// The difference from every other native backend in this tree is that the
// vendor does not publish a vendor-specific API -- corex ships the CUDA-named C
// ABI. Its `/usr/local/corex/include/cudnn.h` is a thin umbrella that pulls in
// `IX/mapping_cudnn.h` (585 `#define`s renaming every CUDA symbol to its ix*
// counterpart) and the real `IX/ixdnn/ixdnn.h`; `cublas_v2.h` does the same for
// `IX/ixblas/`. So `cudnnConvolutionForward` compiles to a call to the 198
// exported `T ixdnn*` symbols in libcudnn.so.7 and `cublasSgemm` to the
// `ixblas*_v2` symbols in libcublas.so.10 -- the CUDA-named C API *is* the
// vendor implementation, backed by separate libraries, not by libcudart.
//
// WHY THERE IS NO BOXING HERE
// ---------------------------
// The obvious alternative -- put the tensor on DispatchKey::CUDA and let
// torch's CUDA kernels run on corex -- is what this backend deliberately avoids.
// corex's runtime is CUDA-shaped, so a CUDA kernel *can* run, but the op would
// then be answered by whatever libtorch the vendor shipped, outside our
// dispatcher's conf, its dtype escape (ResolveFn), its logging, and its
// per-op measurement. This backend instead keeps every op on PrivateUse1 and
// routes it, per the conf, to the kIluvatar slot; an op with no iluvatar kernel
// reaches the boxed cpu_fallback rather than silently acquiring a CUDA kernel
// the conf never named.
//
// WHY data_ptr() IS PASSED STRAIGHT THROUGH
// -----------------------------------------
// A flagos tensor's storage *is* corex device memory -- the same pointer a CUDA
// tensor would carry (verified on ub22: flagos->cuda is a same-pointer view,
// and ixdnn conv2d on flagos memory matches the CPU result bit-for-bit,
// max_diff 0.0). So a kernel hands the vendor `tensor.data_ptr()` with no
// retagging of device metadata: retagging would put the tensor on c10::cuda and
// hand the op to a CUDA kernel, which is exactly the boxing this backend exists
// to avoid. The only marshalling is building the vendor's own descriptors
// (cudnnTensorDescriptor_t etc.) from the aten tensor's sizes and strides.
//
// The build defines __ILUVATAR__ (matching the vendor torch's own ABI selection)
// and compiles with the pre-cxx11 ABI against that torch; USE_ILUVATAR selects
// the platform and this file. See cmake/flagos_platforms.json and
// CMakeLists.txt's iluvatar arm.

#pragma once

#ifdef USE_ILUVATAR

#include <ATen/ATen.h>
#include <ATen/ops/empty.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAStream.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace at::native::flagos::iluvatar {

// The flagos device index of `tensor`. flagos aliases the same physical device
// as torch's CUDA backend, so a flagos tensor's index is directly the corex
// device ordinal -- no translation table is needed the way a separate-runtime
// backend would require. An undefined or non-privateuseone tensor resolves to
// device 0, which is what the kernels pass for their (always-privateuseone)
// operands.
inline int IluvatarDeviceIndex(const at::Tensor& tensor) {
  if (!tensor.defined() || !tensor.device().is_privateuseone()) {
    return 0;
  }
  const int device = static_cast<int>(tensor.device().index());
  return device < 0 ? 0 : device;
}

// A device guard on the same pattern as KunlunxinDeviceGuard and
// MusaDeviceGuard: a vendor pointer resolves only against the *current* corex
// device, so an op on flagos:1 has to run with device 1 selected or the vendor
// library reads / writes the wrong card. corex's runtime is CUDA-shaped and its
// cudaSetDevice (the vendor ixSetDevice) is the same thread-local current-device
// torch's own CUDA guard moves, so this and torch agree -- the raw corex calls
// are used rather than the flagos ::GetDevice/::SetDevice wrappers, matching the
// sibling backends' direct use of their vendor runtime (xpu_set_device,
// musaSetDevice).
//
// The index is read from cudaGetDevice rather than a shadow copy for the reason
// runtime/accelerator/cuda/device.cc records: torch.cuda.set_device() moves the
// driver's index without going through us, so a cached value can only diverge.
class IluvatarDeviceGuard {
 public:
  explicit IluvatarDeviceGuard(const at::Tensor& tensor) {
    const int target = IluvatarDeviceIndex(tensor);
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current == target) {
      return;
    }
    if (cudaSetDevice(target) != cudaSuccess) {
      return;
    }
    previous_ = current;
  }

  ~IluvatarDeviceGuard() {
    if (previous_ >= 0) {
      cudaSetDevice(previous_);
    }
  }

  IluvatarDeviceGuard(const IluvatarDeviceGuard&) = delete;
  IluvatarDeviceGuard& operator=(const IluvatarDeviceGuard&) = delete;

 private:
  int previous_ = -1;
};

// The corex stream an op's work must be queued on.
//
// On a CUDA-compatible platform the flagos device shares torch's CUDA streams,
// so the current stream is readable from c10::cuda (this is why copy_ops.cc and
// runtime/guard.h keep `c10::cuda::getCurrentCUDAStream` for USE_ILUVATAR and
// exclude only the de-boxed / hipified platforms). corex's own stream type is
// aliased to cudaStream_t by the mapping headers, so the handle passes straight
// to cudnnSetStream / cublasSetStream with a cast.
//
// Triton (FlagGems) launches on the same torch current CUDA stream, so binding
// the vendor handle to it orders vendor work against the FlagGems path without
// an explicit barrier -- the property the MUSA backend has to reconstruct by
// hand with a single shared stream (musa_stream.h) but CUDA-compat platforms
// inherit from torch.
inline cudaStream_t IluvatarCurrentStream(int device) {
  return reinterpret_cast<cudaStream_t>(
      c10::cuda::getCurrentCUDAStream(device).stream());
}

// Drain the current stream of `device`. CUDA-family ops are asynchronous, so a
// kernel that must return a completed result (the dispatcher's synchronous
// contract) drains here; it also orders against work the FlagGems/Triton path
// queued on the same stream. Cheap in the steady state because the previous op
// already drained.
inline void IluvatarSynchronizeStream(int device) {
  c10::cuda::getCurrentCUDAStream(device).synchronize();
}

// The vendor handle, created once per device and bound to that device's current
// stream on every call (a handle carries a stream, and the current stream can
// change between ops, so rebinding each call is what keeps the handle on the
// caller's stream rather than on whichever one existed at creation).
//
// Must be called with `device` already current -- IluvatarDeviceGuard does
// that. Handles are never destroyed: destruction would have to run from a static
// destructor whose ordering against the vendor runtime's own teardown is
// unknown, and a process that exits without it reclaims the same memory one
// instruction later (the same trade Kunlunxin's context cache and GCU's
// EnsureTopsatenInit make).
inline cudnnHandle_t GetCudnnHandle(int device) {
  static std::mutex mutex;
  static std::unordered_map<int, cudnnHandle_t> handles;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = handles.find(device);
  if (it == handles.end()) {
    cudnnHandle_t handle = nullptr;
    TORCH_CHECK(
        cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS,
        "iluvatar: cudnnCreate failed on device ",
        device);
    it = handles.emplace(device, handle).first;
  }
  TORCH_CHECK(
      cudnnSetStream(it->second, IluvatarCurrentStream(device)) ==
          CUDNN_STATUS_SUCCESS,
      "iluvatar: cudnnSetStream failed on device ",
      device);
  return it->second;
}

inline cublasHandle_t GetCublasHandle(int device) {
  static std::mutex mutex;
  static std::unordered_map<int, cublasHandle_t> handles;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = handles.find(device);
  if (it == handles.end()) {
    cublasHandle_t handle = nullptr;
    TORCH_CHECK(
        cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS,
        "iluvatar: cublasCreate failed on device ",
        device);
    it = handles.emplace(device, handle).first;
  }
  TORCH_CHECK(
      cublasSetStream(it->second, IluvatarCurrentStream(device)) ==
          CUBLAS_STATUS_SUCCESS,
      "iluvatar: cublasSetStream failed on device ",
      device);
  return it->second;
}

// The base set of aten dtypes mapped to a vendor type below. Necessary but not
// sufficient: the per-op predicates narrow it, and the first cohort's vendor
// entry points were only exercised on the float types. Everything outside the
// mapped set has no vendor scalar type, and ToCudnnDataType / ToCudaDataType
// raise on it rather than reinterpreting the bytes as some neighbouring type.
inline bool IluvatarSupportsDtype(at::ScalarType type) {
  switch (type) {
    case at::kFloat:
    case at::kHalf:
    case at::kDouble:
    case at::kBFloat16:
      return true;
    default:
      return false;
  }
}

// Per-op dtype gates for the first cohort. The frozen scope is conv / bn / gemm
// on float16 and float32: ixdnn and ixblas both map those, and they are the
// dtypes the runtime-conformance dispatch contract and the common model paths
// exercise. Wider types (float64, the integer types) are deliberately left out
// of the cohort rather than being mapped on the strength of the enum alone --
// each is widened here only once a BI-V150 measurement says the vendor entry
// point accepts it, following the konlunxin precedent where the base map carries
// float64 but the per-op gates do not. A kernel tests its op's predicate first
// and takes the CPU round-trip for anything outside it, so the op stays correct
// and -- unlike the CUDA-boxing path -- nothing CUDA-backed answers for it.
inline bool IluvatarSupportsConvDtype(at::ScalarType type) {
  return type == at::kHalf || type == at::kFloat;
}

// Batch norm is float-only on the device path. ixdnn routes scale, bias, the
// saved statistics AND the running statistics through one
// bnScaleBiasMeanVar descriptor with a single element type, while running
// statistics are float32 by contract -- so a half input cannot be served
// without either widening the descriptor (loses the point of half) or
// mis-typing one of the pointers. Measured on ub22: a fp16 input with the
// required float32 running stats raises "expected scalar type Half but found
// Float" rather than producing a silently wrong result, but a raise is still a
// failed op, so half takes the host round-trip below and stays correct. (The
// convolution path is unaffected: ixdnn's conv forward takes one dtype and half
// conv2d is measured correct at 4.9e-04, so IluvatarSupportsConvDtype keeps it.)
inline bool IluvatarSupportsBatchNormDtype(at::ScalarType type) {
  return type == at::kFloat;
}

inline bool IluvatarSupportsGemmDtype(at::ScalarType type) {
  return type == at::kHalf || type == at::kFloat;
}

// Whether a `Tensor?` argument actually carries a tensor.
//
// `has_value()` alone is not the test. C++ composite code forwards its own
// default-constructed `at::Tensor()` through a `Tensor?` slot rather than
// passing std::nullopt, so the optional is *present but undefined* -- e.g.
// aten::conv2d hands `bias` straight to convolution when the caller supplied
// none. Dereferencing such a placeholder does not fail loudly at the source
// line: it reaches at::native::reshape_symint, whose symbol argument lands on
// the SymInt overload of aten::view, which this backend never registers, and
// the error surfaces there as "no tensor arguments ... no fallback function is
// registered for schema aten::view" (measured on ub22: bias-less aten::conv2d
// dies, the same call with a real bias succeeds). Every dereference of an
// optional tensor argument in this backend goes through here.
inline bool IluvatarDefined(const ::std::optional<at::Tensor>& tensor) {
  return tensor.has_value() && tensor->defined();
}

// Which aten dtypes have a vendor scalar type. An unmapped dtype raises rather
// than being silently reinterpreted. The vendor enum has the wider members
// (kint8, kdouble, kbfloat16, ...), so `IluvatarSupportsDtype` above and this
// map are the same list: complex is the notable absence, and no op in this
// backend is exercised on it, so a raise is the honest answer where an
// untested reinterpretation would be a silent one.
inline cudnnDataType_t ToCudnnDataType(at::ScalarType type) {
  switch (type) {
    case at::kFloat:    return CUDNN_DATA_FLOAT;
    case at::kDouble:   return CUDNN_DATA_DOUBLE;
    case at::kHalf:     return CUDNN_DATA_HALF;
    case at::kBFloat16: return CUDNN_DATA_BFLOAT16;
    default:
      TORCH_CHECK(
          false, "iluvatar: unsupported dtype for the corex backend: ", type);
  }
}

// The same mapping into the ixblas data type (aliased to cudaDataType_t by
// mapping_cublas.h). ixblasGemmEx takes its A/B/C type and its computeType as
// this enum, which is why gemm needs it in addition to cudnnDataType_t.
//
// The enum values are deliberately aligned with CUDA's for the common cases
// (CUDA_R_32F == IX_R_32F == 0, CUDA_R_16F == IX_R_16F == 2), but the names are
// used rather than raw integers so a future divergence does not silently pick
// the wrong type.
inline cudaDataType_t ToCudaDataType(at::ScalarType type) {
  switch (type) {
    case at::kFloat:    return CUDA_R_32F;
    case at::kDouble:   return CUDA_R_64F;
    case at::kHalf:     return CUDA_R_16F;
    case at::kBFloat16: return CUDA_R_16BF;
    default:
      TORCH_CHECK(
          false, "iluvatar: unsupported dtype for the corex backend: ", type);
  }
}

// The accumulate type for cublasGemmEx: float32 for the half/bfloat16 inputs
// (so the products are not accumulated in the input's own precision) and the
// input's type otherwise. torch's matmul keeps TF32 off by default, so this
// names the full-precision float path rather than a tensor-op variant; a
// kernel that wants TF32 has to ask for it explicitly, as it would on CUDA.
inline cudaDataType_t IluvatarGemmComputeType(at::ScalarType type) {
  switch (type) {
    case at::kHalf:
    case at::kBFloat16:
      return CUDA_R_32F;
    default:
      return ToCudaDataType(type);
  }
}

// Scratch space for the ops that need it (conv forward queries
// ixdnnGetConvolutionForwardWorkspaceSize). Backed by at::empty on the flagos
// device so the memory comes from our caching allocator and is freed when the
// returned tensor drops. Returns an empty tensor for a zero-byte request, which
// cudnn accepts as "no workspace".
inline at::Tensor IluvatarWorkspaceTensor(
    size_t nbytes,
    const at::Tensor& reference) {
  if (nbytes == 0) {
    return at::Tensor();
  }
  return at::empty(
      {static_cast<int64_t>(nbytes)}, reference.options().dtype(at::kByte));
}

// ixdnn and ixblas are separate libraries with separate status enums, so
// "success" has a per-library spelling. Overloading on the status value lets
// ILUVATAR_EXEC below take either without the caller having to pick a comparison.
inline bool IluvatarStatusOk(ixdnnStatus_t status) {
  return status == IXDNN_STATUS_SUCCESS;
}

inline bool IluvatarStatusOk(ixblasStatus_t status) {
  return status == IXBLAS_STATUS_SUCCESS;
}

// The vendor's own error string, plus the numeric code so an unnamed status is
// still reportable. ixdnnGetErrorString is a real function; ixblas' is the
// ixblasGetStatusString the mapping header aliases cublasGetStatusString to.
inline std::string IluvatarStatusName(ixdnnStatus_t status) {
  const char* name = ixdnnGetErrorString(status);
  return std::string(name != nullptr ? name : "IXDNN_STATUS_UNKNOWN") + " (" +
      std::to_string(static_cast<int>(status)) + ")";
}

inline std::string IluvatarStatusName(ixblasStatus_t status) {
  const char* name = ixblasGetStatusString(status);
  return std::string(name != nullptr ? name : "IXBLAS_STATUS_UNKNOWN") + " (" +
      std::to_string(static_cast<int>(status)) + ")";
}

} // namespace at::native::flagos::iluvatar

// Issues one vendor call and raises on failure with the status enum's name.
//
// `status` is the expression returning an ixdnnStatus_t or ixblasStatus_t and
// `opname` is a string literal naming the vendor entry point (adjacent-literal
// concatenated into the message). The status types are different enums, so the
// comparison and the stringification are dispatched by overload on the status
// value rather than spelled once -- a `== CUDNN_STATUS_SUCCESS` in the macro
// would not compile for a cublas status, and vice versa.
//
// The device guard is *not* part of this macro: it has to be opened before the
// output tensor is allocated (at::empty resolves the device from the current
// one, and the allocator would otherwise hand back memory on the wrong card), so
// kernels open it first and call this only for the call itself. Draining is done
// by the kernel after its last vendor call, not here, so a kernel that issues
// several calls (query algo, query workspace size, forward) pays one sync.
#define ILUVATAR_EXEC(status, opname)                                          \
  do {                                                                         \
    const auto _iluvatar_status = (status);                                    \
    TORCH_CHECK(                                                               \
        ::at::native::flagos::iluvatar::IluvatarStatusOk(_iluvatar_status),    \
        (opname),                                                              \
        " failed on the Iluvatar (corex) backend: ",                           \
        ::at::native::flagos::iluvatar::IluvatarStatusName(_iluvatar_status)); \
  } while (0)

#endif // USE_ILUVATAR