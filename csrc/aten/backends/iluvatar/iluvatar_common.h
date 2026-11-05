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
// and links only the CPU torch wheel: the platform has its own runtime sources
// (runtime/accelerator/{iluvatar,allocator/backends/iluvatar_memory.h}) that talk
// to corex's C runtime directly, so no c10::cuda symbol -- and therefore not
// libtorch_cuda.so / libc10_cuda.so -- is needed. USE_ILUVATAR selects the
// platform and this file. See cmake/flagos_platforms.json and CMakeLists.txt's
// iluvatar arm.

#pragma once

#ifdef USE_ILUVATAR

#include <ATen/ATen.h>
#include <ATen/ops/empty.h>
#include <c10/core/Scalar.h>

#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include "runtime/accelerator/iluvatar/iluvatar_context.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace at::native::flagos::iluvatar {

// The flagos device index of `tensor`. The flagos device ordinal is the corex
// device ordinal -- flagos is registered on the corex device set directly, so no
// translation table is needed. An undefined or non-privateuseone tensor resolves
// to device 0, which is what the kernels pass for their (always-privateuseone)
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
// the flagos runtime's own ::SetDevice moves, so the raw corex calls and the
// contract wrappers agree -- the raw calls are used, matching the sibling
// backends' direct use of their vendor runtime (xpu_set_device, musaSetDevice).
//
// The index is read from cudaGetDevice rather than a shadow copy for the reason
// runtime/accelerator/iluvatar/device.cc records: the driver's index can be
// moved without going through our ::SetDevice, so a cached value can only
// diverge.
class IluvatarDeviceGuard {
 public:
  explicit IluvatarDeviceGuard(const at::Tensor& tensor) {
    c10::flagos::iluvatar::EnsureIluvatarContext();
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
// This is torch_fl's own per-device stream, from the flagos runtime contract
// (runtime/accelerator/iluvatar/stream.cc), and NOT torch's. Reading torch's
// stream here -- `c10::cuda::getCurrentCUDAStream`, which is what this did --
// is what made every iluvatar build link libtorch_cuda.so and libc10_cuda.so,
// so no CPU torch wheel could load it; the platform therefore had no compliant
// test environment. Iluvatar is not a boxing platform, nothing in the build
// enters the CUDA dispatch key, and there is no torch stream to be ordered
// against in the first place.
//
// corex's own stream type is aliased to cudaStream_t by the mapping headers, so
// the handle passes straight to cudnnSetStream / cublasSetStream with a cast.
//
// The ordering property the old arrangement got from torch is kept, by the same
// mechanism the MUSA and Kunlunxin backends use: one default stream per device
// shared by every producer. The vendor kernels submit to it through this
// function, and FlagTree's Triton launcher is handed the same handle by
// flagtree_shim.get_iluvatar_current_raw_stream, so a FlagGems kernel and the
// vendor op consuming its output are ordered without an explicit barrier.
inline cudaStream_t IluvatarCurrentStream(int device) {
  return reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(device));
}

// Drain the current stream of `device`. CUDA-family ops are asynchronous, so a
// kernel that must return a completed result (the dispatcher's synchronous
// contract) drains here; it also orders against work the FlagGems/Triton path
// queued on the same stream. Cheap in the steady state because the previous op
// already drained.
inline void IluvatarSynchronizeStream(int device) {
  cudaStream_t stream = IluvatarCurrentStream(device);
  if (stream != nullptr) {
    cudaStreamSynchronize(stream);
  }
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
// mapped set has no vendor scalar type, and ToCudnnDataType / ToBlasDataType
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

// bfloat16 joins the gate on the strength of a BI-V150 measurement: corex's
// cublasGemmEx accepts CUDA_R_16BF operands with a CUDA_R_32F accumulate
// (`IX/mapping_ixsdk.h` aliases CUDA_R_16BF to IX_R_16BF, and ToBlasDataType /
// IluvatarGemmComputeType already carried that pair), and a 2048x2048 bf16 `mm`
// on flagos:0 goes from 0.41 TFLOPS to 79.9 TFLOPS once it stops taking the host
// round-trip below -- the same op is 80.2 TFLOPS in half, and the vendor's own
// torch reports 84.6. Without bf16 here every bf16 matmul in a bf16 model ran on
// the CPU: 40 ms per 2048^3 GEMM instead of 0.2 ms.
inline bool IluvatarSupportsGemmDtype(at::ScalarType type) {
  return type == at::kHalf || type == at::kFloat || type == at::kBFloat16;
}

// Whether `ixblasAxpyEx` serves this scalar type on a BI-V150.
//
// Measured on u180 (corex 4.5.0), `add.Tensor` shapes: float32 and float16
// return IXBLAS_STATUS_SUCCESS and match the CPU reference exactly; float64,
// bfloat16, complex64, complex128, int32 and int64 return status 15
// (IXBLAS_STATUS_NOT_SUPPORTED) for every alpha and execution type tried, and
// `executionType` must be 0 -- the enum's named values (16F, 32F, 64F, 32I)
// all return 15. So this is a deliberately narrow gate, not
// `IluvatarSupportsDtype`: a wider gate would hand the kernel a call that
// fails, and a kernel that caught the failure and took the host round trip
// instead would be the cpu_fallback this backend exists to avoid.
inline bool IluvatarSupportsAxpyDtype(at::ScalarType type) {
  return type == at::kFloat || type == at::kHalf;
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

// The same mapping into the ixblas data type. Deliberately *not* called
// cudaDataType_t: cublas_v2.h names its enum cudaDataType_t only because
// mapping_cublas.h #defines it to ixblasDataType_t, and cudnn.h's umbrella pulls
// mapping_cudnn.h into the same translation unit, so the two vendor mapping
// headers see each other's renames. The vendor type names are used directly
// instead, so this file does not depend on which mapping header was included
// first -- which is exactly how the aliased spelling failed to compile. ixblasGemmEx takes its A/B/C type and its computeType as
// this enum, which is why gemm needs it in addition to cudnnDataType_t.
//
// The enum values are the vendor's own (IX_R_32F, IX_R_16F, ...), used by their
// names rather than raw integers so a change in the vendor enum does not
// silently pick the wrong type.
inline ixblasDataType_t ToBlasDataType(at::ScalarType type) {
  switch (type) {
    case at::kFloat:    return IX_R_32F;
    case at::kDouble:   return IX_R_64F;
    case at::kHalf:     return IX_R_16F;
    case at::kBFloat16: return IX_R_16BF;
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
inline ixblasDataType_t IluvatarGemmComputeType(at::ScalarType type) {
  switch (type) {
    case at::kHalf:
    case at::kBFloat16:
      return IX_R_32F;
    default:
      return ToBlasDataType(type);
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

// `x` laid out one element per index of `target`'s shape, for an add whose
// operand broadcasts. It must not go through `x.expand(target).contiguous()`:
// both tensors are ventored on this backend, and `reshape` is deferred to
// `view`, which reads a broadcast operand back as a contiguous one and
// refuses the copy with "The expanded size of the tensor (64) must match the
// existing size (128)" (measured on u180). Materialising the strides here
// replaces `reshape` with ATen's strided clone, which is the copy this needs.
inline at::Tensor IluvatarBroadcastOperand(
    const at::Tensor& x, at::IntArrayRef target) {
  const at::Tensor xc = x.contiguous();
  std::vector<int64_t> strides(target.size(), 0);
  const int64_t lead = static_cast<int64_t>(target.size()) - xc.dim();
  for (const auto d : c10::irange(xc.dim())) {
    const int64_t i = lead + d;
    if (xc.size(d) == target[i] && xc.stride(d) != 0) {
      strides[i] = xc.stride(d);
    }
  }
  return xc.as_strided(target, strides).clone();
}

// y += alpha * other on every element, where `other` is a 0-dim tensor whose
// value is added everywhere. Kept separate from IluvatarAxpy because the two
// vendor calls differ: the tensor form takes `other`'s data pointer, and a
// 0-dim tensor has no length for the vendor to read per output.
//
// y += alpha * x on the device, for the four `add` schemas
// (codegen_iluvatar.py's add_* categories).
//
// ixblasAxpyEx takes alpha by pointer with its own type tag, and its
// `executionType` byte was measured on u180 (corex 4.5.0) to accept only 0 --
// the enum's named values (16F, 32F, 64F, 32I) all returned
// IXBLAS_STATUS_NOT_SUPPORTED. alpha is float32 for both operand types (see
// the comment on alpha_f below). Both were verified against the CPU reference;
// every other dtype is refused by IluvatarSupportsAxpyDtype before the call,
// because the vendor call returns status 15 for all of them.
inline void IluvatarAxpy(
    at::Tensor& y,
    const at::Tensor& x,
    const at::Scalar& alpha) {
  const int64_t n = x.numel();
  if (n == 0 || alpha.to<double>() == 0.0) {
    return;
  }
  at::Tensor xc = x.is_contiguous() ? x : x.contiguous();
  const int device = IluvatarDeviceIndex(y);
  cublasHandle_t handle = GetCublasHandle(device);
  // alpha is float32 in both arms: the vendor takes alpha in its own type tag
  // and was measured to accept a float32 alpha beside float16 operands
  // (IX_R_16F x/y, IX_R_32F alpha -> status 0, exact), while a float16 alpha
  // returns status 15 for every execution type tried. So one alpha width
  // serves both dtypes.
  const float alpha_f = static_cast<float>(alpha.to<double>());
  if (x.scalar_type() == at::kHalf) {
    ILUVATAR_EXEC(
        ixblasAxpyEx(
            handle,
            static_cast<int>(n),
            &alpha_f,
            IX_R_32F,
            xc.const_data_ptr(),
            IX_R_16F,
            1,
            y.data_ptr(),
            IX_R_16F,
            1,
            static_cast<ixblasDataType_t>(0)),
        "ixblasAxpyEx");
  } else {
    ILUVATAR_EXEC(
        ixblasAxpyEx(
            handle,
            static_cast<int>(n),
            &alpha_f,
            IX_R_32F,
            xc.const_data_ptr(),
            IX_R_32F,
            1,
            y.data_ptr(),
            IX_R_32F,
            1,
            static_cast<ixblasDataType_t>(0)),
        "ixblasAxpyEx");
  }
  IluvatarSynchronizeStream(device);
}

// y += alpha * s on every element of y, for `t + 1.0`. Built on ixblasAxpyEx
// rather than ixblasScal: corex exports only the typed Sscal/Dscal/Cscal/
// Zscal entry points, and every one of them is single precision or wider.
// y += alpha * s is the axpy y += (alpha * s) * ones, and that broadcast is
// not free: sizing that vector against the tensor route would need a fill
// kernel first (fused_ops has `fill_`, not `full`), so the scalar call's
// advantage over routing the 0-dim tensor through add.Tensor is that it skips
// the contiguous() copy, not the ones buffer.
inline void IluvatarAxpyScalar(
    at::Tensor& y,
    const at::Scalar& s,
    const at::Scalar& alpha) {
  const int64_t n = y.numel();
  if (n == 0 || alpha.to<double>() == 0.0) {
    return;
  }
  const int device = IluvatarDeviceIndex(y);
  cublasHandle_t handle = GetCublasHandle(device);
  const float alpha_f = static_cast<float>(alpha.to<double>());
  // `ones` stands in for the x operand: n elements of 1 in the output's type,
  // with alpha carrying `alpha * s`. The value is rounded to the output's type
  // rather than the multiply being narrowed, which is what ATen does for a
  // python number.
  at::Tensor ones;
  if (y.scalar_type() == at::kHalf) {
    ones = at::full({n}, 1, y.options());
    const float scaled =
        alpha_f * static_cast<float>(s.to<at::Half>());
    ILUVATAR_EXEC(
        ixblasAxpyEx(
            handle,
            static_cast<int>(n),
            &scaled,
            IX_R_32F,
            ones.const_data_ptr(),
            IX_R_16F,
            1,
            y.data_ptr(),
            IX_R_16F,
            1,
            static_cast<ixblasDataType_t>(0)),
        "ixblasAxpyEx");
  } else {
    ones = at::full({n}, 1, y.options());
    const float scaled = alpha_f * static_cast<float>(s.to<double>());
    ILUVATAR_EXEC(
        ixblasAxpyEx(
            handle,
            static_cast<int>(n),
            &scaled,
            IX_R_32F,
            ones.const_data_ptr(),
            IX_R_32F,
            1,
            y.data_ptr(),
            IX_R_32F,
            1,
            static_cast<ixblasDataType_t>(0)),
        "ixblasAxpyEx");
  }
  IluvatarSynchronizeStream(y.get_device());
}



} // namespace at::native::flagos::iluvatar

#endif  // USE_ILUVATAR
