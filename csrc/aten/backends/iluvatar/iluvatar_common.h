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
#include <ATen/ops/ones.h>
#include <ATen/ops/scalar_tensor.h>
#include <c10/core/Scalar.h>

#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include "runtime/accelerator/iluvatar/iluvatar_context.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
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
// Both overloads have to precede ILUVATAR_EXEC, which calls them.
inline std::string IluvatarStatusName(ixdnnStatus_t status) {
  const char* name = ixdnnGetErrorString(status);
  return std::string(name != nullptr ? name : "IXDNN_STATUS_UNKNOWN") + " (" +
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
//
// Defined here rather than at the end of the header: the axpy helpers below use
// it, and a macro must be declared before its first use.
#define ILUVATAR_EXEC(status, opname)                                          \
  do {                                                                         \
    const auto _iluvatar_status = (status);                                    \
    TORCH_CHECK(                                                               \
        IluvatarStatusOk(_iluvatar_status),    \
        (opname),                                                              \
        " failed on the Iluvatar (corex) backend: ",                           \
        IluvatarStatusName(_iluvatar_status)); \
  } while (0)

inline std::string IluvatarStatusName(ixblasStatus_t status) {
  const char* name = ixblasGetStatusString(status);
  return std::string(name != nullptr ? name : "IXBLAS_STATUS_UNKNOWN") + " (" +
      std::to_string(static_cast<int>(status)) + ")";
}

// The dtypes ixblasAxpyEx serves. Measured on BI-V150 (corex 4.5.0): the call
// accepts xType/yType IX_R_32F and IX_R_16F and returns
// IXBLAS_STATUS_NOT_SUPPORTED (15) for float64, bfloat16, complex and the
// integer types, for every execution type it has.
inline bool IluvatarSupportsAxpyDtype(at::ScalarType type) {
  return type == at::kFloat || type == at::kHalf;
}

// d += alpha * x on the device, for float32/float16 operands: one
// ixblasAxpyEx, the in-place term of an elementwise add. The other term is the
// destination's own value, which IluvatarAdd makes sure already holds what it
// should before this call -- there is no vendor call for `d = alpha*x + beta*y`
// itself.
//
// ixblasAxpyEx defers to the handle's stream, which is the flagos current
// stream (GetCublasHandle rebinds it on every call), so the caller's next op is
// ordered behind it without a device-wide drain.
//
// alpha is float32 for both operand types: measured on u180 (corex 4.5.0), the
// vendor accepts IX_R_16F x/y with IX_R_32F alpha (status 0, exact) and returns
// status 15 for a float16 alpha under every execution type, and
// `executionType` accepts only 0 (its 16F/32F/64F/32I values all return
// NOT_SUPPORTED). Strides are in elements; 1 is the only stride a multi-element
// contiguous tensor can have.
//
// `d` may alias `x` (the in-place add passes it as both) -- the vendor kernel
// reads element by element, so the read of x precedes the write to it.
//
// `groups` is 1 for the elementwise case and more when the operand is a
// broadcast vector (see IluvatarAdd): the destination is then the operand
// repeated `groups` times along its last dimension, and each repetition is one
// more call over the same operand bytes.
inline void IluvatarAxpy(
    at::Tensor& d,
    const at::Tensor& x,
    double alpha,
    int64_t groups = 1) {
  const int64_t n = d.numel();
  if (n == 0) {
    return;
  }
  const int device = IluvatarDeviceIndex(d);
  cublasHandle_t handle = GetCublasHandle(device);
  const float alpha_f = static_cast<float>(alpha);
  const auto type = ToBlasDataType(d.scalar_type());
  if (groups == 1) {
    ILUVATAR_EXEC(
        ixblasAxpyEx(
            handle,
            static_cast<int>(n),
            &alpha_f,
            IX_R_32F,
            x.const_data_ptr(),
            type,
            1,
            d.data_ptr(),
            type,
            1,
            static_cast<ixblasDataType_t>(0)),
        "ixblasAxpyEx");
    return;
  }
  const int64_t m = n / groups;
  TORCH_CHECK(
      m * groups == n,
      "iluvatar: ", n, " elements do not split into ", groups, " groups");
  char* base = static_cast<char*>(d.data_ptr());
  const int64_t bytes = m * d.element_size();
  for (int64_t g = 0; g < groups; ++g) {
    ILUVATAR_EXEC(
        ixblasAxpyEx(
            handle,
            static_cast<int>(m),
            &alpha_f,
            IX_R_32F,
            x.const_data_ptr(),
            type,
            1,
            base + g * bytes,
            type,
            1,
            static_cast<ixblasDataType_t>(0)),
        "ixblasAxpyEx");
  }
}

// The elementwise add the conf routes to `iluvatar`, in all three spellings
// (add.Tensor, add_.Tensor, add_.Scalar): one entry point, because they differ
// only in where the result goes and what the second addend is.
//
// These are the only kernels register.cc names by hand rather than taking from
// an `.inc`. They live here, not in iluvatar_kernels.cc, because register.cc
// includes that file inside its TORCH_LIBRARY_IMPL body (a function body),
// where torch and corex headers cannot be pulled in; this header is included at
// file scope instead.
//
// The conf routes this op to `iluvatar` for every dtype, so this is the one
// answer for all of them: float32/float16 contiguous operands go through the
// vendor kernel (IluvatarAxpy, one per operand-sized group), and every other
// dtype -- and the layouts the vendor call cannot express -- fall to
// at::add_out, the same registration register.cc used before this change.
// Neither arm is a CPU fallback.
//
// Why FlagGems is not the route: its add is a Python kernel that allocates and
// writes a full-size temporary, and on a 48 MB bf16 activation measured ~7 ms
// on u180 against ~0.5 ms of bandwidth. The elementwise add is called hundreds
// of times per training step, so the temporary is the whole cost.
inline at::Tensor& IluvatarAdd(
    at::Tensor& out,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Scalar& alpha) {
  iluvatar::IluvatarDeviceGuard guard(out);
  const int64_t n = out.numel();
  const int64_t m = b.numel();
  // The vendor call is a contiguous linear add: the destination and the a
  // operand must be contiguous and the same length, and the b operand either
  // the same length (elementwise) or an evenly repeated vector (broadcast --
  // a bias or a per-channel scale, the other common shape of an add).
  const bool vectorised = iluvatar::IluvatarSupportsAxpyDtype(out.scalar_type()) &&
      a.scalar_type() == out.scalar_type() &&
      b.scalar_type() == out.scalar_type() && a.numel() == n &&
      a.is_contiguous() && out.is_contiguous() && m > 0 &&
      n % m == 0 && b.is_contiguous() &&
      (b.sizes().equals(out.sizes()) || out.sizes().back() == m);
  if (!vectorised) {
    return at::add_out(out, a, b, alpha);
  }
  const double s = alpha.to<double>();
  const bool a_aliases_out = out.storage().is_alias_of(a.storage());
  const bool b_aliases_out = out.storage().is_alias_of(b.storage());
  // IluvatarAxpy adds into the destination (d += alpha * x), so `out`'s value
  // at the call has to be one of the result's terms already. The steady
  // state -- every add in a model -- is b_aliases_out with alpha 1, and that
  // is the single vendor call with nothing to prepare.
  const int64_t groups = n / m;
  if (b_aliases_out && s == 1.0) {
    IluvatarAxpy(out, a, 1.0, groups);
    return out;
  }
  if (a_aliases_out) {
    // Nothing may be written to `out` before `a` (which is `out`) is read.
    return at::add_out(out, a, b, alpha);
  }
  // Make `out` hold b (scaled when alpha is not 1), then add alpha * a. The
  // copy and the scale are elementwise kernels; at::add_out would be the same
  // two passes, so this is never slower than the registration it replaces.
  if (n == m) {
    out.copy_(b);
    if (s != 1.0) {
      out.mul_(s);
    }
  } else {
    // A broadcast operand is a view of the same bytes, so one-call copy and
    // scale are both broadcasts; only the fill itself repeats.
    if (s == 1.0) {
      out.view({groups, m}).copy_(b);
    } else {
      out.view({groups, m}).copy_(b).mul_(s);
    }
  }
  IluvatarAxpy(out, a, s, groups);
  return out;
}

// add_.Scalar / add.Scalar / add.Scalar_out: `b` is a host number. ATen's
// composite for these spellings boxes it into a 0-dim device tensor and calls
// the tensor form above, so there is one implementation and one result -- a
// number that does not fit the dtype's precision by more than one rounding step
// would need a variant that reads b as a float instead.
inline at::Tensor& IluvatarAddScalar(
    at::Tensor& out, const at::Scalar& b, const at::Scalar& alpha) {
  return IluvatarAdd(
      out, out, at::scalar_tensor(b, out.options()), alpha);
}

// add.Scalar_out, the out= spelling. Its schema names an out parameter, and
// registering a by-reference kernel under it aborts the library at load with
// "Mismatch in kernel C++ signatures" (measured on u180), so it is registered
// through the boxed kernel below, the same way add.out is.
inline at::Tensor& IluvatarAddScalarOutImpl(
    const at::Tensor& self, const at::Scalar& b, const at::Scalar& alpha,
    at::Tensor& out) {
  if (!out.sizes().equals(self.sizes())) {
    out.resize_(self.sizes());
  }
  out.copy_(self);
  return IluvatarAddScalar(out, b, alpha);
}

inline void IluvatarAddScalarOut(
    const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const size_t base = stack->size() - schema.arguments().size();
  const at::Tensor self = stack->at(base).toTensor();
  const at::Scalar& b = stack->at(base + 1).toScalar();
  const at::Scalar& alpha = stack->at(base + 2).toScalar();
  at::Tensor& out =
      stack->at(base + static_cast<size_t>(*schema.argumentIndexWithName("out")))
          .toTensor();
  IluvatarAddScalarOutImpl(self, b, alpha, out);
  stack->resize(base + 1);
  stack->at(base) = out;
}

// add.Tensor, the out-of-place tensor add. Nothing in ATen's own path names it
// -- `a + b` and `a.add(b)` reach add.out -- so this is the kernel for a call
// that comes straight to the op, and it also answers a plain `a + b`: measured
// on u180, remove the add.Tensor registration and `a + b` segfaults, because
// ATen's composite add.Tensor lowers to `add_(empty_like(self) + 0, ..)` and
// the empty destination it starts from is garbage, which segfaults inside
// copy_ before the add. IluvatarAdd reads its destination as one of the
// result's terms, so this entry point writes both of them first.
inline at::Tensor IluvatarAddTensor(
    const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  at::Tensor out = at::empty(self.sizes(), self.options());
  const int64_t m = other.numel();
  iluvatar::IluvatarDeviceGuard guard(out);
  const bool vectorised =
      iluvatar::IluvatarSupportsAxpyDtype(out.scalar_type()) &&
      self.scalar_type() == out.scalar_type() &&
      other.scalar_type() == out.scalar_type() &&
      self.numel() == out.numel() && self.is_contiguous() && m > 0 &&
      out.numel() % m == 0 && other.is_contiguous() &&
      (other.sizes().equals(out.sizes()) || out.sizes().back() == m);
  if (!vectorised) {
    return IluvatarAdd(out, self, other, alpha);
  }
  const double s = alpha.to<double>();
  const int64_t groups = out.numel() / m;
  out.view({groups, m}).copy_(other);
  if (s != 1.0) {
    out.mul_(s);
  }
  IluvatarAxpy(out, self, s, groups);
  return out;
}

// add.out, and therefore add_.Tensor. Its schema is
// `(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)`,
// and add_.Tensor is a structured delegate to it (`structured_delegate:
// add.out` in native_functions.yaml), so both spellings reach this kernel and a
// typed registration must carry the four-argument signature. It is a callable
// object rather than a plain function because torch::Library::impl takes the
// out argument by value, which a `Tensor&` parameter does not bind to; the
// library accepts functors, and calls theirs with the schema's arguments.
//
// `out` arrives empty in the `out=` spelling, so it must be written, not read:
// measured on u180, a kernel that read it answered garbage plus self. The
// in-place spelling reaches the same kernel with `out` aliasing `self`, which
// IluvatarAdd serves with one vendor call. Sizes that do not all match are the
// same split IluvatarAdd makes -- ATen resizes `out` to the broadcast shape,
// then the kernel runs at that shape.
//
// torch::Library::impl wraps the functor for every dispatch key the schema
// lists, and only the PrivateUse1 one has an out parameter to take by value;
// add.out's (catch all) registration is dispatched on operands alone. Measured
// on u180, a functor here aborted the whole library with "Mismatch in kernel
// C++ signatures" at registerKernel, so add.out goes in through the boxed
// kernel below, and this struct is the body that kernel calls.
struct IluvatarAddOut {
  at::Tensor& operator()(
      const at::Tensor& self,
      const at::Tensor& other,
      const at::Scalar& alpha,
      at::Tensor& out) const {
    const bool same_shape =
        other.sizes().equals(self.sizes()) && out.sizes().equals(self.sizes());
    if (!same_shape || out.device() != self.device()) {
      const auto shape = at::infer_size(self.sizes(), other.sizes());
      out.resize_(shape);
    }
    const double s = alpha.to<double>();
    // Only the same-shape, contiguous case goes to the vendor axpy. Frame plus
    // bias is (M, 1) plus (1, N), and a stride-0 expanded view of self is what
    // would make that one call -- measured on u180, running the axpy on a
    // resize alone segfaulted. Everything else is answered here with the
    // elementwise kernels instead of through at::add_out: that call
    // redispatches to whatever add.out resolves to for these operands, which on
    // this backend is the very registration that is running.
    if (out.is_contiguous() && self.sizes().equals(out.sizes()) &&
        other.sizes().equals(out.sizes())) {
      return IluvatarAdd(out, self, other, alpha);
    }
    // The broadcasting side is not one contiguous run of the destination, so
    // it is two elementwise passes: out = self (scaled), then out += other
    // (scaled). Both are broadcast copies, which is the only way a stride-0
    // operand is written down here.
    const auto scale = [&](const at::Tensor& t) {
      return s == 1.0 ? t : t.mul(s);
    };
    out.copy_(scale(self));
    out.add_(scale(other));
    return out;
  }
};

// add.out as a boxed kernel. The stack holds the op's own arguments, alpha and
// out among them, and the return is the Tensor the operator passes in as
// `out`, so self and other are the first two arguments the schema lists and
// the destination is the one named `out`.
inline void IluvatarAddOutBoxed(
    const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const size_t base = stack->size() - schema.arguments().size();
  at::Tensor self = stack->at(base).toTensor();
  at::Tensor other = stack->at(base + 1).toTensor();
  at::Tensor& out =
      stack->at(base + static_cast<size_t>(*schema.argumentIndexWithName("out")))
          .toTensor();
  const size_t alpha_at =
      base + static_cast<size_t>(*schema.argumentIndexWithName("alpha"));
  const double alpha = stack->at(alpha_at).toScalar().toDouble();
  IluvatarAddOut{}(self, other, at::Scalar(alpha), out);
  stack->resize(base + 1);
  stack->at(base) = out;
}

// add.Scalar, the out-of-place spelling `a + 1` / `a.add(1)`. Its schema has
// no out parameter, so it keeps the three-argument shape; the destination is a
// fresh tensor, and IluvatarAdd reads `out` as a term of the result, so it is
// written from `self` before the vendor call.
inline at::Tensor IluvatarAddScalarTensor(
    const at::Tensor& self, const at::Scalar& b, const at::Scalar& alpha) {
  at::Tensor out = at::empty(self.sizes(), self.options());
  out.copy_(self);
  return IluvatarAddScalar(out, b, alpha);
}

} // namespace at::native::flagos::iluvatar

#endif // USE_ILUVATAR
