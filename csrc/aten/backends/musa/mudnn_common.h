// Copyright (c) 2026, BAAI. All rights reserved.
//
// Marshalling helpers for the Moore Threads MUSA (mudnn) operator backend.
//
// mudnn is the vendor's torch-independent kernel library: it links against
// musart only and pulls in no torch symbols at all. That is the whole point of
// using it here -- the earlier route through torch_musa's flat `at::musa::*`
// API tied this plugin to one exact torch build, because the vendor .so embeds
// torch's C++ object layout (e.g. sizeof(c10::MessageLogger) changed 408 -> 400
// between 2.9.1 and 2.10, which corrupts the stack inside the vendor binary).
// Against mudnn the backend is version-agnostic, like Ascend's aclnn and GCU's
// topsaten.
//
// The mudnn call shape is: describe operands as `musa::dnn::Tensor`, configure
// an op object (mode/alpha/dims), then `Run(handle, out, in...)`. Unlike
// topsaten, a mudnn Tensor carries strides, so ops read non-contiguous inputs
// directly and kernels need fewer `.contiguous()` copies. A few ops (Reduce,
// MatMul) additionally want a workspace allocator.

#pragma once

#ifdef USE_MUSA

#include "runtime/accelerator/musa/musa_stream.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/ops/empty.h>
#include <c10/core/Scalar.h>

#include <mudnncxx/mudnn.h>
#include <musa_runtime.h>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace at::native::flagos::musa_ops {

namespace mudnn = ::musa::dnn;

inline const char* MudnnStatusName(mudnn::Status status) {
  switch (status) {
    case mudnn::Status::SUCCESS:           return "SUCCESS";
    case mudnn::Status::INVALID_PARAMETER: return "INVALID_PARAMETER";
    case mudnn::Status::NOT_INITIALIZED:   return "NOT_INITIALIZED";
    case mudnn::Status::ALLOC_FAILED:      return "ALLOC_FAILED";
    case mudnn::Status::NOT_SUPPORTED:     return "NOT_SUPPORTED";
    case mudnn::Status::INTERNAL_ERROR:    return "INTERNAL_ERROR";
    case mudnn::Status::ARCH_MISMATCH:     return "ARCH_MISMATCH";
    case mudnn::Status::EXECUTION_FAILED:  return "EXECUTION_FAILED";
    default:                               return "UNKNOWN";
  }
}

inline mudnn::Tensor::Type ToMudnnDataType(at::ScalarType type) {
  switch (type) {
    case at::kFloat:    return mudnn::Tensor::Type::FLOAT;
    case at::kDouble:   return mudnn::Tensor::Type::DOUBLE;
    case at::kHalf:     return mudnn::Tensor::Type::HALF;
    case at::kBFloat16: return mudnn::Tensor::Type::BFLOAT16;
    case at::kLong:     return mudnn::Tensor::Type::INT64;
    case at::kInt:      return mudnn::Tensor::Type::INT32;
    case at::kShort:    return mudnn::Tensor::Type::INT16;
    case at::kChar:     return mudnn::Tensor::Type::INT8;
    case at::kByte:     return mudnn::Tensor::Type::UINT8;
    case at::kBool:     return mudnn::Tensor::Type::BOOL;
    default:
      TORCH_CHECK(false, "Unsupported dtype for mudnn: ", type);
  }
}

// Every dtype in the table above has a mudnn Tensor::Type, including int64 --
// unlike topsaten, which has no int64 kernels at all. Kernels call this before
// running so that an exotic dtype (complex, quantized, the fp8 variants we do
// not map) takes the CPU fallback instead of raising from ToMudnnDataType.
inline bool MudnnSupportsDtype(at::ScalarType type) {
  switch (type) {
    case at::kFloat:
    case at::kDouble:
    case at::kHalf:
    case at::kBFloat16:
    case at::kLong:
    case at::kInt:
    case at::kShort:
    case at::kChar:
    case at::kByte:
    case at::kBool:
      return true;
    default:
      return false;
  }
}

// Bool is narrower than the rest: mudnn takes it for the logical and comparison
// modes and for IDENTITY/CAST, but rejects it for arithmetic
// ("Unsupported binary mode: MUL, with left data type: BOOL"). PyTorch *does*
// define bool arithmetic (bool * bool == and), so arithmetic kernels use this
// predicate and let bool operands take the CPU fallback.
inline bool MudnnSupportsArithmeticDtype(at::ScalarType type) {
  return type != at::kBool && MudnnSupportsDtype(type);
}

// A musa device pointer resolves only against the *current* device, so an op on
// flagos:1 must run with device 1 selected. Restores the previous device.
class MusaDeviceGuard {
 public:
  explicit MusaDeviceGuard(const at::Tensor& tensor) {
    if (!tensor.defined() || !tensor.device().is_privateuseone()) {
      return;
    }
    const int target = static_cast<int>(tensor.device().index());
    if (target < 0) {
      return;
    }
    int current = -1;
    if (musaGetDevice(&current) != musaSuccess || current == target) {
      return;
    }
    if (musaSetDevice(target) != musaSuccess) {
      return;
    }
    prev_device_ = current;
  }

  ~MusaDeviceGuard() {
    if (prev_device_ >= 0) {
      musaSetDevice(prev_device_);
    }
  }

  MusaDeviceGuard(const MusaDeviceGuard&) = delete;
  MusaDeviceGuard& operator=(const MusaDeviceGuard&) = delete;

 private:
  int prev_device_ = -1;
};

// `mudnn::Handle` is per-device, non-copyable, and not free to construct, so
// cache one per device and bind it to that device's shared stream. Must be
// called with the target device already current (EXEC_MUDNN_CMD's guard does
// that), since Handle() and GetDefaultMusaStream() both key off musaGetDevice.
//
// TF32 is refreshed on every call rather than only at construction: mudnn
// enables it by default, whereas torch defaults matmul TF32 *off*, and the user
// can flip torch.backends.cuda.matmul.allow_tf32 at any point. Left at mudnn's
// default, a 64x64 float mm drifts ~2e-2 from CPU (measured); following torch's
// flag makes it exact.
inline mudnn::Handle& GetMudnnHandle() {
  static std::mutex mutex;
  static std::unordered_map<int, std::unique_ptr<mudnn::Handle>> handles;

  int device = 0;
  if (musaGetDevice(&device) != musaSuccess) {
    device = 0;
  }

  std::lock_guard<std::mutex> lock(mutex);
  auto it = handles.find(device);
  if (it == handles.end()) {
    auto handle = std::make_unique<mudnn::Handle>(device);
    handle->SetStream(at::native::flagos::musa::GetDefaultMusaStream());
    it = handles.emplace(device, std::move(handle)).first;
  }
  it->second->SetAllowTF32(at::globalContext().allowTF32CuBLAS());
  return *it->second;
}

// Describes an aten tensor to mudnn. Sizes and strides are copied because
// SetNdInfo takes bare `const int64_t*` and mudnn may read them lazily; keeping
// the vectors in the wrapper guarantees they outlive the op call.
//
// Strides are always passed, so a non-contiguous input (a transpose, a slice)
// is handled on-device rather than through a materializing copy.
class MudnnTensorWrapper {
 public:
  explicit MudnnTensorWrapper(const at::Tensor& tensor)
      : sizes_(tensor.sizes().vec()), strides_(tensor.strides().vec()) {
    TORCH_CHECK(tensor.defined(), "mudnn: undefined tensor");
    // mudnn rejects a rank-0 shape, so a scalar tensor (a full reduction
    // result, a 0-dim operand) is described as the equivalent 1-element vector.
    if (sizes_.empty()) {
      sizes_.assign(1, 1);
      strides_.assign(1, 1);
    }
    // PyTorch's is_contiguous() ignores the stride of any size-1 dimension
    // (that dim contributes only index 0, so no read ever depends on it), so
    // a transpose like `torch.randn(2, 1).t()` reports contiguous while
    // leaving a degenerate stride behind -- e.g. shape (1, 2) stride (1, 1).
    // mudnn's own validation is stricter (MatMul's lda check wants the
    // leading dimension's stride >= the trailing extent) and rejects that
    // degenerate value. Rewrite every size-1 dim's stride to the value a
    // real C-contiguous tensor of this shape would carry there; the actual
    // number is unconstrained for correctness since the dim has one
    // element, so this only needs to satisfy mudnn's shape validation.
    int64_t contiguous_stride = 1;
    for (int64_t i = static_cast<int64_t>(sizes_.size()) - 1; i >= 0; --i) {
      if (sizes_[i] == 1) {
        strides_[i] = contiguous_stride;
      }
      contiguous_stride *= sizes_[i];
    }
    tensor_.SetType(ToMudnnDataType(tensor.scalar_type()));
    // data_ptr() already accounts for storage_offset.
    tensor_.SetAddr(tensor.const_data_ptr());
    tensor_.SetNdInfo(
        static_cast<int>(sizes_.size()), sizes_.data(), strides_.data());
  }

  // Non-const: mudnn output parameters are `Tensor&`.
  mudnn::Tensor& get() {
    return tensor_;
  }

  MudnnTensorWrapper(const MudnnTensorWrapper&) = delete;
  MudnnTensorWrapper& operator=(const MudnnTensorWrapper&) = delete;

 private:
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  mudnn::Tensor tensor_;
};

// Workspace allocator for the ops that need scratch space (Reduce, MatMul).
// Backed by at::empty on the flagos device so the memory comes from our caching
// allocator; the returned MemoryHandler's deleter drops the tensor, which
// releases it back to the pool.
inline mudnn::MemoryMaintainer MudnnWorkspaceFor(const at::Tensor& reference) {
  auto options = reference.options().dtype(at::kByte);
  return [options](size_t size_in_bytes) -> mudnn::MemoryHandler {
    if (size_in_bytes == 0) {
      return mudnn::MemoryHandler(nullptr, [](void*) {});
    }
    auto* holder = new at::Tensor(
        at::empty({static_cast<int64_t>(size_in_bytes)}, options));
    return mudnn::MemoryHandler(
        holder->data_ptr(), [holder](void*) { delete holder; });
  };
}

// True when mudnn's Reduce misbehaves on this input and it must be materialized
// first. The trigger is an input that is a *broadcast of a single element* --
// every dim with extent > 1 has stride 0 -- which reaches real code through bias
// gradients: `linear(x, w, b).sum()` backward reduces a grad_output that
// autograd produced as `ones.expand(...)`, whose storage is one float.
//
// Two distinct failures were measured on mudnn v3300, both silent in the status:
//
//   - Reducing over *more than one* dim raises SIGFPE inside the vendor library
//     -- an uncatchable crash, not a NOT_SUPPORTED status.
//   - Reducing over a *single* dim intermittently writes only out[0] and leaves
//     the remaining output elements untouched, so the answer is whatever the
//     caching allocator last left in that block. Observed as a wrong bias
//     gradient (`[4, 34, 38, 42, 46]` where elements 1.. were a previous op's
//     result); the same reduce on a materialized copy is always correct.
//
// A multi-dim reduce is fine as soon as any one stride is non-zero, but the
// single-dim partial write does not reproduce standalone, so this predicate does
// not try to be narrower than "fully 0-strided". The copy it forces is one
// element wide on input, so the cost is a single small materialization.
inline bool MudnnReduceNeedsContiguous(const at::Tensor& self) {
  if (self.is_contiguous()) {
    return false;
  }
  for (int64_t d = 0; d < self.dim(); ++d) {
    if (self.size(d) > 1 && self.stride(d) != 0) {
      return false;
    }
  }
  return true;
}

// Maps an aten reduction dim onto mudnn's `int` dim array.
inline std::vector<int> ToMudnnDims(at::IntArrayRef dims, int64_t ndim) {
  std::vector<int> out;
  out.reserve(dims.size());
  for (int64_t d : dims) {
    out.push_back(static_cast<int>(at::maybe_wrap_dim(d, ndim)));
  }
  return out;
}

// Copies `src` into `dst` on device, handling both non-contiguous layouts and
// dtype conversion in a single pass. This replaces torch_musa's `_copy_from`:
// mudnn Tensors carry strides, so IDENTITY gathers a strided source into a
// contiguous destination, and CAST does the same while converting dtype.
//
// Used by copy_ops.cc and contiguous_ops.cc, which otherwise would have to fall
// back to a CPU round-trip (there is no CUDA runtime on this platform to boxed
// dispatch into).
void MudnnCopy(const at::Tensor& src, at::Tensor& dst);

} // namespace at::native::flagos::musa_ops

// Issues a mudnn op on the shared stream of `guard_tensor`'s device and waits
// for it, mirroring EXEC_TOPSATEN_CMD's synchronous contract.
//
// `op_expr` is the full call, e.g. `op.Run(_mudnn_h, t_out.get(), t_in.get())`,
// with `_mudnn_h` naming the cached handle. Spelling it out (rather than taking
// the op and args separately like EXEC_TOPSATEN_CMD) keeps the macro usable for
// mudnn's varied Run/RunWithIndices/RunBwd entry points.
#define EXEC_MUDNN_CMD(tag, guard_tensor, op_expr)                            \
  do {                                                                        \
    at::native::flagos::musa_ops::MusaDeviceGuard _musa_guard(                \
        (guard_tensor));                                                      \
    auto& _mudnn_h = at::native::flagos::musa_ops::GetMudnnHandle();          \
    ::musa::dnn::Status _mudnn_status = (op_expr);                            \
    TORCH_CHECK(                                                              \
        _mudnn_status == ::musa::dnn::Status::SUCCESS,                        \
        tag, " failed: ",                                                     \
        at::native::flagos::musa_ops::MudnnStatusName(_mudnn_status));        \
    musaError_t _musa_sync = musaStreamSynchronize(_mudnn_h.GetStream());     \
    TORCH_CHECK(                                                              \
        _musa_sync == musaSuccess,                                           \
        tag, " stream sync failed: ", musaGetErrorString(_musa_sync));        \
  } while (0)

#endif // USE_MUSA
