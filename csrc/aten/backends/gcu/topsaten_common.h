// Copyright (c) 2026, BAAI. All rights reserved.
//
// Marshalling helpers for the Enflame GCU (topsaten) operator backend.
//
// Compared to Ascend's aclnn, the topsaten call shape is a single direct call:
// there is no workspace query / executor phase. A kernel therefore only needs
// to wrap its aten tensors into `topsatenTensor`, pick the stream, call the op
// and check the status.

#pragma once

#ifdef USE_GCU

#include "runtime/accelerator/gcu/tops_stream.h"

#include <ATen/ATen.h>
#include <ATen/ops/full.h>
#include <c10/core/Scalar.h>

#include <topsaten/topsaten.h>

#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace at::native::flagos::gcu {

inline const char* TopsatenStatusName(topsatenStatus_t status) {
  switch (status) {
    case TOPSATEN_STATUS_SUCCESS:        return "SUCCESS";
    case TOPSATEN_STATUS_ALLOC_FAILED:   return "ALLOC_FAILED";
    case TOPSATEN_STATUS_BAD_PARAM:      return "BAD_PARAM";
    case TOPSATEN_STATUS_NOT_SUPPORT:    return "NOT_SUPPORT";
    case TOPSATEN_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case TOPSATEN_STATUS_RUNTIME_ERROR:  return "RUNTIME_ERROR";
    case TOPSATEN_STATUS_EXECUTE_ERROR:  return "EXECUTE_ERROR";
    default:                             return "UNKNOWN";
  }
}

// topsaten requires a one-time global init before any op call.
inline void EnsureTopsatenInit() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    // Some ops (e.g. the tensor-with-scalar overloads) allocate a temporary
    // device buffer internally; point those allocations at the tops runtime.
    // Must be registered before init so the internal pools pick them up.
    topsatenMallocFuncRegister(
        [](void** p, size_t n) { return topsMalloc(p, n); });
    topsatenFreeFuncRegister([](void* p) { return topsFree(p); });
    topsatenMallocAsyncFuncRegister(
        [](void** p, size_t n, topsStream_t, uint64_t) {
          return topsMalloc(p, n);
        });
    topsatenFreeAsyncFuncRegister(
        [](void* p, topsStream_t) { return topsFree(p); });
    topsatenStatus_t status = topsatenInit();
    TORCH_CHECK(
        status == TOPSATEN_STATUS_SUCCESS,
        "topsatenInit failed: ", TopsatenStatusName(status));
  });
}

inline topsatenDataType_t ToTopsatenDataType(at::ScalarType type) {
  switch (type) {
    case at::kFloat:    return TOPSATEN_DATA_FP32;
    case at::kDouble:   return TOPSATEN_DATA_F64;
    case at::kHalf:     return TOPSATEN_DATA_FP16;
    case at::kBFloat16: return TOPSATEN_DATA_BF16;
    case at::kLong:     return TOPSATEN_DATA_I64;
    case at::kInt:      return TOPSATEN_DATA_I32;
    case at::kShort:    return TOPSATEN_DATA_I16;
    case at::kChar:     return TOPSATEN_DATA_I8;
    case at::kByte:     return TOPSATEN_DATA_U8;
    case at::kBool:     return TOPSATEN_DATA_PRED;
    default:
      TORCH_CHECK(false, "Unsupported dtype for topsaten: ", type);
  }
}

// topsaten has no int64, float64 or complex kernels: every op returns NOT_SUPPORT
// for an I64 operand (verified across add/mul/eq/abs/sum), and an F64 operand
// fails the same way (measured on S60 across add/abs/reciprocal, vendor log
// "datatype not support yet"). Complex is the same story one layer earlier --
// `ToTopsatenDataType` has no mapping for it at all, so an op on a complex
// operand raises before it reaches the vendor. The first caller is the complex
// rotary embedding of Qwen-Image: the freqs live on the device, `cat` on them
// raised `Unsupported dtype for topsaten: ComplexFloat`, and the transformer
// step could not start.
//
// Callers check this and run the op on CPU instead, which keeps int64 tensors
// (indices, masks, counters), float64 tensors and complex tensors working
// instead of raising. GradScaler depends on the float64 path: it computes the
// inverse scale as `scale.double().reciprocal().float()`.
inline bool TopsatenSupportsDtype(at::ScalarType type) {
  return type != at::kLong && type != at::kDouble && !c10::isComplexType(type);
}

// Which dtypes topsatenArange may be handed.
//
// Narrower than TopsatenSupportsDtype on purpose. `arange` is the one op whose
// vendor entry point takes no size: it fills an output the caller has already
// sized, so a dtype the kernel declines leaves the buffer unwritten instead of
// raising, and the whole point of the op is the values in that buffer. The set
// is measured on S60 rather than inferred from ToTopsatenDataType, which only
// says the dtype is *representable*.
//
// Bool is deliberately absent even though the vendor takes PRED: ATen has no
// arange kernel for it at all (`arange_cpu not implemented for 'Bool'`, and a
// GCU caller gets the same on a hosted tensor), so declining here is what keeps
// the host path's NotImplementedError instead of inventing a result CPU and
// CUDA both refuse to produce.
inline bool TopsatenArangeDtype(at::ScalarType type) {
  switch (type) {
    case at::kFloat:
    case at::kInt:
    case at::kShort:
    case at::kChar:
    case at::kByte:
    case at::kHalf:
    case at::kBFloat16:
      return true;
    default:
      return false;
  }
}

// Which dtypes topsatenIndexSelect accepts an *index* operand as.
//
// An index tensor is the one place an integer operand has to be braced for
// specially rather than declined: the coordinate data is consumed as integers,
// not as element data, so the "no int64 kernels" rule does not apply to it. Both
// dtypes are measured, not inferred -- index_select at dim 0/1/2 and with a
// 0-dim, empty or non-contiguous index returns the host result for an i32 index
// and for an i64 one. index_fill is the opposite case: the vendor rejects an i64
// index and only ever takes an i32 one, so it narrows its index through
// `TopsatenIndexFillIndex` below instead of using this.
inline bool TopsatenIndexDtype(at::ScalarType type) {
  return type == at::kInt || type == at::kLong;
}

// The two operand limits topsatenIndexSelect documents that its dtype table
// does not express: every dim of `in` and `out` must hold fewer than 2^24
// elements, and every operand buffer must stay under 4.0 GB (topsaten_ops.h).
// Neither is a performance threshold -- past either the kernel returns an
// error -- so a call that trips one is served by the host path rather than
// raising where ATen's own kernel would have produced a result.
inline bool TopsatenIndexSelectFits(at::TensorList operands) {
  constexpr int64_t kMaxDim = int64_t(1) << 24;
  constexpr int64_t kMaxBufferBytes = int64_t(4) << 30;
  for (const at::Tensor& t : operands) {
    if (t.numel() * t.element_size() >= kMaxBufferBytes) {
      return false;
    }
    for (const int64_t size : t.sizes()) {
      if (size >= kMaxDim) {
        return false;
      }
    }
  }
  return true;
}

// Whether every element of an index tensor is non-negative, i.e. whether ATen's
// `index` and the `index_select` this backend delegates it to agree on it.
//
// A negative index is the one place the two spellings differ: ATen's index wraps
// it (measured: -1 in dim 0 of a (2,3,4) selects the last row), while the vendor's
// index kernel is handed the value as a coordinate and resolves it outside the
// tensor. They agree for every non-negative index, so the readback is the test.
// It is O(num_indices) -- the same order as the narrowing in
// TopsatenIndexFillIndex below -- and `self` stays on the device.
inline bool TopsatenIndexNonNegative(const at::Tensor& index) {
  auto host = index.to(at::kCPU).contiguous().reshape({-1});
  const int64_t numel = host.numel();
  // Only reachable through TopsatenIndexDtype, so the index is i32 or i64.
  if (host.scalar_type() == at::kInt) {
    const int32_t* src = host.const_data_ptr<int32_t>();
    for (int64_t i = 0; i < numel; ++i) {
      if (src[i] < 0) {
        return false;
      }
    }
    return true;
  }
  const int64_t* src = host.const_data_ptr<int64_t>();
  for (int64_t i = 0; i < numel; ++i) {
    if (src[i] < 0) {
      return false;
    }
  }
  return true;
}

// The i32 index tensor topsatenIndexFill requires, built from ATen's own index.
//
// index_fill is the one index op whose operand types the two sides disagree on
// outright: ATen takes an int64 index and nothing else (`index_fill_(): Expected
// dtype int64 for index.`), and the vendor takes an int32 one and nothing else
// (BAD_PARAM for an i64 index at every dim, rank, self dtype and scalar tag
// measured on S60). The vendor's own bounds check is also not safe to lean on:
// an index equal to the dim size satisfies it and writes one element past the
// end of the tensor, and a larger index aborts the process outright
// (`Index out-of-bounds! bound=2, index=5`). ATen, meanwhile, accepts a negative
// index in this op and wraps it (measured: -1 in dim 0 of a (2,3,4) fills row 1,
// while -3 raises). So the index is read back once and wrapped and narrowed in
// the same pass. A value still out of range afterwards -- or one that cannot be
// represented as int32, which a truncating cast would silently fold back into
// range -- returns an undefined tensor, and the caller sends that call to the
// host path, where ATen raises its own message for it.
//
// The transfer is O(num_indices) and `self` stays on the device, which is the
// whole point: the host path this replaces copies the *tensor being filled* to
// the host and back.
inline at::Tensor TopsatenIndexFillIndex(const at::Tensor& index, int64_t size) {
  auto host = index.to(at::kCPU).contiguous().reshape({-1});
  auto narrowed = at::empty(host.sizes(), host.options().dtype(at::kInt));
  const int64_t* src = host.const_data_ptr<int64_t>();
  int32_t* dst = narrowed.data_ptr<int32_t>();
  const int64_t numel = host.numel();
  for (int64_t i = 0; i < numel; ++i) {
    int64_t value = src[i];
    if (value < 0) {
      value += size;
    }
    if (value < 0 || value >= size) {
      return at::Tensor();
    }
    dst[i] = static_cast<int32_t>(value);
  }
  return narrowed.to(index.device());
}

// The i32 index tensor topsatenEmbeddingDenseBackward requires, built from
// ATen's own index.
//
// The vendor's parameter table says int64 and its runtime disagrees: every int64
// index set comes back BAD_PARAM with `topsatenEmbeddingBackward error: index
// tensor must be of int32 data type`, and the same call with an int32 index
// succeeds and matches ATen's own kernel row for row -- measured on S60 through
// the vendor symbol itself, with padding_idx set, with scale_grad_by_freq set,
// and for a 1-D index as well as the [n, 1] the table asks for. ATen meanwhile
// hands this op an int64 index and nothing else, so the narrowing is the op's
// job.
//
// It cannot be a plain cast. ATen does not raise on an index outside
// [0, num_weights): it *ignores* that element of grad (measured -- 9, -2 and
// 2**32 + 1, each with num_weights 6, all leave the table holding only the
// in-range contributions). A truncating cast folds 2**32 + 1 back to 1, and the
// vendor then accumulates grad's row for it into row 1 where ATen accumulated
// nothing -- measured, and a silently wrong gradient rather than an ignored
// element. So the index is read back once and checked in the same pass: an entry
// outside [0, num_weights) returns an undefined tensor and the caller sends that
// call to the host path, where ATen's own kernel ignores it. A num_weights that
// does not fit in int32 takes the host path for the same reason -- a table that
// large cannot be addressed by the vendor at all.
//
// The transfer is O(num_indices) and `grad` -- num_indices * embed_dim
// elements, larger by the embedding width -- stays on the device, which is the
// whole point: the host path this replaces copies grad to the host and back.
inline at::Tensor TopsatenEmbeddingIndex(const at::Tensor& indices,
                                         int64_t num_weights,
                                         const at::Device& device) {
  if (num_weights <= 0 || num_weights > std::numeric_limits<int32_t>::max()) {
    return at::Tensor();
  }
  // int32 indices are legal for ATen's own kernel, so they are widened rather
  // than declined; both dtypes then take the same validation below.
  auto host = indices.to(at::kCPU).to(at::kLong).contiguous().reshape({-1});
  auto narrowed = at::empty(host.sizes(), host.options().dtype(at::kInt));
  const int64_t* src = host.const_data_ptr<int64_t>();
  int32_t* dst = narrowed.data_ptr<int32_t>();
  const int64_t numel = host.numel();
  for (int64_t i = 0; i < numel; ++i) {
    const int64_t value = src[i];
    if (value < 0 || value >= num_weights) {
      return at::Tensor();
    }
    dst[i] = static_cast<int32_t>(value);
  }
  return narrowed.to(device);
}

// Which dtypes topsatenEmbeddingDenseBackward may be handed. Its parameter
// table lists result/grad as float, fp16, bf16, i32, u32, i16, u16, i8, u8 --
// everything TopsatenSupportsDtype allows except PRED, which the vendor omits
// even though its arange and index kernels do take a bool. A bool embedding
// weight is legal in ATen, so declining here is what keeps the host path's
// result instead of a vendor status error.
inline bool TopsatenEmbeddingDenseBackwardDtype(at::ScalarType type) {
  return TopsatenSupportsDtype(type) && type != at::kBool;
}

// Which dtypes topsatenNonzero and topsatenCountNonzero may be handed.
//
// The same set for both calls, and it has to be: measured on S60 across the nine
// topsaten dtypes, f32, f16, u8, i16, i32 and PRED return SUCCESS with the right
// coordinates, while f64, bf16 and i64 are refused -- and the refusal is the
// dangerous kind. topsatenCountNonzero does not fail for those three, it returns
// SUCCESS with a wrong count (0, 2 and 0 for an input holding a single nonzero),
// and topsatenNonzero is handed an output whose numel has to be exactly
// rank * count. A kernel that leaned on the status would size its output from a
// wrong count and then write through it. So the dtype test is the guard, and it
// gates the count and the nonzero call together.
//
// The half dtypes are widened one level up, in TopsatenNonzeroOperand, so they
// never reach the vendor as halves; PRED is here and in the list above because a
// bool mask is the natural operand for this op.
inline bool TopsatenNonzeroDtype(at::ScalarType type) {
  switch (type) {
    case at::kFloat:
    case at::kByte:
    case at::kShort:
    case at::kInt:
    case at::kBool:
      return true;
    default:
      return false;
  }
}

// The operand to hand topsatenNonzero for `self`, or an undefined tensor when
// this call has to take the host path.
//
// The op reads the input's zero pattern and nothing else, and both half types
// widen to f32 exactly -- every half value, subnormal included, is representable
// in f32 -- so a half operand is converted rather than declined: the pattern
// cannot change. That matters here because the Qwen-Image transformer runs in
// bf16, and the alternative for a dtype the vendor refuses is a full host round
// trip of the same tensor. f64 and the wide integers are *not* widened: a double
// can flush to zero on the way to f32, and 2**32 folds to zero on the way to
// i32, either of which would turn a real element into an apparent zero and shift
// every coordinate after it.
inline at::Tensor TopsatenNonzeroOperand(const at::Tensor& self) {
  const at::ScalarType type = self.scalar_type();
  if (type == at::kHalf || type == at::kBFloat16) {
    return self.to(at::kFloat).contiguous();
  }
  if (!TopsatenNonzeroDtype(type)) {
    return at::Tensor();
  }
  return self.contiguous();
}

// Which dtypes topsatenAll may be handed.
//
// Measured on S60 with a (1,) PRED output, all-true and one-false inputs:
// f16, bf16, f32, i8, u8, i16 and i32 return SUCCESS with the right answer, and
// f64 and i64 return NOT_SUPPORT. PRED is not in that sweep but is measured
// separately and is the interesting entry -- `prompt_embeds_mask.all()`
// (pipeline_qwenimage.py) is a bool mask, so the one call site worth converting
// is a bool operand.
//
// Same shape as TopsatenNonzeroDtype: the *representable* dtypes are a superset,
// because ToTopsatenDataType maps i64 and f64 and would hand the vendor a dtype
// it refuses. Here that refusal is harmless (a status rather than a wrong
// answer), but declining is still what makes the host path run instead of the
// call failing.
inline bool TopsatenAllDtype(at::ScalarType type) {
  switch (type) {
    case at::kFloat:
    case at::kHalf:
    case at::kBFloat16:
    case at::kByte:
    case at::kBool:
    case at::kChar:
    case at::kShort:
    case at::kInt:
      return true;
    default:
      return false;
  }
}

// The operand to hand topsatenAll for `self`, or an undefined tensor when this
// call has to take the host path.
//
// The op asks whether every element is non-zero, which is a property of the zero
// pattern, so both half types widen to f32 exactly and are converted rather than
// declined -- same reasoning as TopsatenNonzeroOperand, and it matters for the
// same reason: the Qwen-Image transformer runs in bf16.
inline at::Tensor TopsatenAllOperand(const at::Tensor& self) {
  const at::ScalarType type = self.scalar_type();
  if (type == at::kHalf || type == at::kBFloat16) {
    return self.to(at::kFloat).contiguous();
  }
  if (!TopsatenAllDtype(type)) {
    return at::Tensor();
  }
  return self.contiguous();
}

// Which dtypes topsatenWhere may be handed, for the two *value* operands.
//
// This is the widest set in this file, and it is the one that could not be
// inferred from the others: measured on S60 with a PRED condition, every dtype
// topsaten can represent works -- f16, bf16, f32, f64, i64, i32, i16, i8, u8 and
// PRED itself -- including the two, f64 and i64, that TopsatenSupportsDtype
// excludes and that every other entry point here refuses. So where is the one op
// that must *not* be gated on TopsatenSupportsDtype, and its kernel has no
// operand guard beyond this test. The three dtypes it is still wrong to pass are
// the ones ToTopsatenDataType has no mapping for: complex, which would raise
// from the wrapper rather than fall back, and the float8 pair.
//
// The condition operand is not covered here; it has to be PRED. A u8 condition
// is BAD_PARAM at the vendor (measured) where ATen accepts it and casts, so the
// kernel tests the condition against at::kBool separately and sends anything
// else to the host.
inline bool TopsatenWhereDtype(at::ScalarType type) {
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

// Which device a multi-operand kernel should read its operands on, when one of
// them may be a host scalar.
//
// A 0-dim CPU tensor is how ATen spells a scalar inside a C++ composite -- the
// eager `_safe_softmax` builds its zero operand with `at::scalar_tensor(0, ...)`
// and no device -- and it is a valid operand of a device op, not a reason to
// compute on the host. A kernel that takes `self.device()` as the compute device
// therefore *silently* picks the host for such a call: the operands are then
// handed to the vendor as device pointers, which for where is a RUNTIME_ERROR out
// of topsatenWhere and, at a vendor that does not validate, would be wrong
// numbers instead.
//
// Returns the first operand that is actually on a card, so a kernel can hoist the
// CPU operand onto it before expanding. When none of them is, the call really is
// a host call and kCPU comes back, which every caller's fallback path takes.
inline at::Device TopsatenComputeDevice(
    const at::Tensor& first,
    const at::Tensor& second,
    const at::Tensor& third) {
  for (const at::Tensor* candidate : {&first, &second, &third}) {
    if (candidate->defined() && candidate->device().is_privateuseone()) {
      return candidate->device();
    }
  }
  return at::Device(at::kCPU);
}

// Brings an operand to the common shape of an elementwise call without
// materialising it when it does not have to be.
//
// The generated kernels used to do this with `tensor.expand(shape).contiguous()`
// on the premise that "topsaten does not broadcast for us". The vendor disagrees
// in its own documentation:
//
//   "1. Load input (L3->L1) or (L3->L2->L1 for broadcast), if lhs or rhs needs to
//    broadcast, it will be processed in this step."
//                    -- /opt/tops/include/gcu/topsaten/topsaten_ops.h:491
//
// and the materialisation is not free: `.contiguous()` on an expanded view
// allocates and fills the whole common shape, so `x * 2.0` -- whose operand is a
// 0-dim tensor -- pays an allocator round trip plus a full elementwise copy
// before the multiply starts. Measured on S60 at (4096, 2560) bf16, per call:
// `a * a` 0.212 ms, `a * 0-dim device` 0.322 ms, `a * 0-dim host` 0.416 ms, while
// `.contiguous()` on an already-contiguous tensor is 0.004 ms. The gap between
// the first two is the materialisation of a 21 MB copy.
//
// Handing over the view is safe because `expand()` over contiguous storage
// produces zero strides only on the dims it actually broadcasts, and a torch-free
// probe at the rank-4 attention shapes has checked every spelling of interest --
// all-zero strides, the mixed (0,0,1,0) per-token form, a rank-1 (1) with stride
// 0, a rank-2/rank-3 operand against a rank-4 one, and the small base buffer such
// a view really points into -- against the same op run on the materialised
// operand: bit-identical output in fp32 and in bf16, no form refused. A stride
// walk that ignored the descriptor would have landed on a different value,
// because the base buffer is filled per-offset rather than with a repeated
// constant.
//
// A non-contiguous operand still goes through `.contiguous()`: that is not a
// broadcast of contiguous storage, and the vendor's handling of an arbitrary
// strided layout is a separate question from broadcast. It costs nothing for the
// operands that are contiguous already, which is every operand these kernels
// receive after the `.to(device, dtype)` casts in their prologues.
inline at::Tensor BroadcastTo(const at::Tensor& tensor, at::IntArrayRef shape) {
  if (tensor.is_contiguous()) {
    return tensor.expand(shape);
  }
  return tensor.expand(shape).contiguous();
}

// A tops device pointer resolves only against the *current* device, so an op
// on flagos:1 must run with device 1 selected. Restores the previous device.
class TopsDeviceGuard {
 public:
  explicit TopsDeviceGuard(const at::Tensor& tensor) {
    if (!tensor.defined() || !tensor.device().is_privateuseone()) {
      return;
    }
    const int target = static_cast<int>(tensor.device().index());
    if (target < 0) {
      return;
    }
    int current = -1;
    if (topsGetDevice(&current) != topsSuccess || current == target) {
      return;
    }
    if (topsSetDevice(target) != topsSuccess) {
      return;
    }
    prev_device_ = current;
  }

  ~TopsDeviceGuard() {
    if (prev_device_ >= 0) {
      topsSetDevice(prev_device_);
    }
  }

  TopsDeviceGuard(const TopsDeviceGuard&) = delete;
  TopsDeviceGuard& operator=(const TopsDeviceGuard&) = delete;

 private:
  int prev_device_ = -1;
};

// `topsatenSize_t` only holds a `const int64_t*`, so the shape/stride arrays
// must outlive the topsatenTensor. Copy them into the wrapper.
class TopsatenTensorWrapper {
 public:
  explicit TopsatenTensorWrapper(const at::Tensor& tensor)
      : sizes_(tensor.sizes().vec()), strides_(tensor.strides().vec()) {
    TORCH_CHECK(tensor.defined(), "topsaten: undefined tensor");
    // topsaten rejects a rank-0 shape ("dims/strides length is invalid"), so a
    // scalar tensor (e.g. the result of a full reduction) is described as the
    // equivalent 1-element vector.
    if (sizes_.empty()) {
      sizes_.assign(1, 1);
      strides_.assign(1, 1);
    }
    // data_ptr() already accounts for storage_offset, so no SetOffset here.
    tops_tensor_ = topsatenTensor(
        topsatenSize_t(sizes_.data(), static_cast<int64_t>(sizes_.size())),
        topsatenSize_t(strides_.data(), static_cast<int64_t>(strides_.size())),
        ToTopsatenDataType(tensor.scalar_type()),
        const_cast<void*>(tensor.const_data_ptr()));
  }

  // Non-const: topsaten output parameters are `topsatenTensor&`.
  topsatenTensor& get() {
    return tops_tensor_;
  }

  TopsatenTensorWrapper(const TopsatenTensorWrapper&) = delete;
  TopsatenTensorWrapper& operator=(const TopsatenTensorWrapper&) = delete;

 private:
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  topsatenTensor tops_tensor_;
};

// Holds the int64 array backing a `topsatenSize_t` argument (dims lists etc).
class TopsatenSizeWrapper {
 public:
  explicit TopsatenSizeWrapper(at::IntArrayRef values)
      : values_(values.vec()) {}

  explicit TopsatenSizeWrapper(std::vector<int64_t> values)
      : values_(std::move(values)) {}

  topsatenSize_t get() const {
    return topsatenSize_t(values_.data(), static_cast<int64_t>(values_.size()));
  }

  TopsatenSizeWrapper(const TopsatenSizeWrapper&) = delete;
  TopsatenSizeWrapper& operator=(const TopsatenSizeWrapper&) = delete;

 private:
  std::vector<int64_t> values_;
};

// A topsatenTensor describing the first `rows` rows of a contiguous (n, rank)
// buffer.
//
// `topsatenNonzero`'s output numel has to be input_rank * input_num exactly: an
// (nnz, rank) output over an input with a different nonzero count is BAD_PARAM,
// and the call writes nothing at all when it refuses -- measured on S60, where a
// (2, 3) output over a rank-3 input holding four nonzeros failed while (4, 3)
// and (6, 3) both succeeded. `nonzero_static`'s truncation therefore cannot be
// expressed by handing the op a smaller output: the coordinates always go into
// an (nnz, rank) buffer and it is the read-back that gets shortened. That needs
// a description shorter than the buffer it points into, in both directions -- an
// (nnz, rank) view of a (size, rank) buffer when size > nnz, and a (size, rank)
// view of an (nnz, rank) buffer when size < nnz. A topsaten tensor carries no
// allocation size, only a pointer, a shape and strides, so the same base pointer
// with a smaller first dim is a legal description of either, and both directions
// are measured on S60.
//
// The obvious alternative, `at::Tensor::narrow`, is a dispatcher call and GCU
// has no `narrow` kernel, so the view would round-trip the buffer through the
// host -- which is the whole cost this chain exists to avoid.
class TopsatenRowView {
 public:
  TopsatenRowView(const at::Tensor& buffer, int64_t rows, int64_t rank)
      : sizes_{rows, rank}, strides_{rank, 1} {
    TORCH_CHECK(buffer.defined(), "topsaten: undefined tensor");
    TORCH_CHECK(
        buffer.is_contiguous(), "topsaten: a row view needs a contiguous buffer");
    // Rank 1 describes an (n, 1) buffer, which is what a contiguous 1-D tensor
    // of n elements already is, so the strides hold for every rank.
    tops_tensor_ = topsatenTensor(
        topsatenSize_t(sizes_.data(), 2),
        topsatenSize_t(strides_.data(), 2),
        ToTopsatenDataType(buffer.scalar_type()),
        const_cast<void*>(buffer.const_data_ptr()));
  }

  // Non-const: topsaten output parameters are `topsatenTensor&`.
  topsatenTensor& get() {
    return tops_tensor_;
  }

  TopsatenRowView(const TopsatenRowView&) = delete;
  TopsatenRowView& operator=(const TopsatenRowView&) = delete;

 private:
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  topsatenTensor tops_tensor_;
};

// Materializes a Scalar as a device tensor of `sizes`.
//
// topsaten's add/sub/mul/div tensor-with-scalar overloads stage the scalar into
// a host buffer that the driver refuses inside this process ("Cannot create
// memory object for kernel parameter 2"), so those ops go through the
// tensor-with-tensor overload instead. topsaten does not broadcast, hence the
// full-size tensor.
//
// The fill is done on the device. Staging it through the host instead --
// `at::full(sizes, scalar, options.device(at::kCPU)).to(options.device())`, which
// is what this used to do -- costs a full-size host allocation, a full-size CPU
// fill, a full-size H2D transfer and a second full-size device allocation, once
// per scalar operand, and the pipeline's scalar operands are not small. Measured
// on S60 at the (1, 24, 4114, 4114) fp32 attention shape: `a + 1.0` through the
// host is 630.1 ms against 3.7 ms for the device fill alone, and at the
// (1, 4096, 3072) bf16 activation shape it is 17.2 ms against 0.3 ms. Every
// codegen'd binary_scalar_as_tensor kernel stages its scalar this way, so a
// forward pays it once per `x + 1.0`-style expression in the model, not once per
// attention layer.
inline at::Tensor ScalarToDeviceTensor(
    const at::Scalar& scalar,
    at::IntArrayRef sizes,
    const at::TensorOptions& options) {
  return at::full(sizes, scalar, options);
}

// `topsatenScalar_t` is a plain {dtype, union{double fval; int64_t ival;}}, so
// the union member has to match the dtype the op will read it back as.
inline topsatenScalar_t ToTopsatenScalar(
    const at::Scalar& scalar,
    at::ScalarType type) {
  topsatenScalar_t out{};
  out.dtype = ToTopsatenDataType(type);
  if (at::isFloatingType(type)) {
    out.fval = scalar.to<double>();
  } else if (type == at::kBool) {
    out.ival = scalar.to<bool>() ? 1 : 0;
  } else {
    out.ival = scalar.to<int64_t>();
  }
  return out;
}

// Marshals an at::TensorList into the std::vector<topsatenTensor> the foreach
// ops take, owning one wrapper per tensor so every sizes/strides array stays
// alive for the duration of the call.
//
// The foreach kernels write through the strides they are given and accept an
// output that aliases an input (verified on hardware), so an in-place foreach
// can hand the same list as both operand and destination -- but only when each
// tensor is contiguous, since a non-contiguous destination cannot be written
// back through a temporary. Callers gate on IsForeachEligible() for that.
class TopsatenTensorList {
 public:
  explicit TopsatenTensorList(at::TensorList tensors) {
    wrappers_.reserve(tensors.size());
    tops_.reserve(tensors.size());
    for (const at::Tensor& t : tensors) {
      wrappers_.push_back(std::make_unique<TopsatenTensorWrapper>(t));
      tops_.push_back(wrappers_.back()->get());
    }
  }

  std::vector<topsatenTensor>& get() {
    return tops_;
  }

  TopsatenTensorList(const TopsatenTensorList&) = delete;
  TopsatenTensorList& operator=(const TopsatenTensorList&) = delete;

 private:
  std::vector<std::unique_ptr<TopsatenTensorWrapper>> wrappers_;
  std::vector<topsatenTensor> tops_;
};

// A foreach list goes to topsaten only if every tensor is a contiguous,
// zero-offset, topsaten-representable tensor on the same device. Anything else
// (an int64 list, a sliced parameter view, a mixed-device list) takes the
// generic path, which is still correct, just slower.
inline bool IsForeachEligible(at::TensorList tensors) {
  if (tensors.empty()) {
    return false;
  }
  auto device = tensors[0].device();
  if (device.type() != c10::kPrivateUse1) {
    return false;
  }
  for (const at::Tensor& t : tensors) {
    if (!t.defined() || !TopsatenSupportsDtype(t.scalar_type()) ||
        !t.is_contiguous() || t.storage_offset() != 0 || t.numel() == 0 ||
        t.device() != device) {
      return false;
    }
  }
  return true;
}

// Bounds used to fill in an absent clamp limit: clamping against the dtype's
// own extreme leaves that side untouched. Floating types use -inf/+inf so that
// NaN handling matches an unbounded clamp.
//
// The integral extreme has to be of the tensor's own width. The value reaches
// the vendor in the int64 member of topsatenScalar_t and is read back as the
// tensor's dtype, so a wider constant truncates: INT64_MAX handed to an int32
// clamp arrives as -1 (its low 32 bits), which turns `clamp(min=0)` on an int32
// tensor into a clamp to -1.
inline int64_t IntegralExtreme(at::ScalarType type, bool lowest) {
  switch (type) {
    case at::kByte:
      return lowest ? 0 : int64_t(std::numeric_limits<uint8_t>::max());
    case at::kChar:
      return lowest ? int64_t(std::numeric_limits<int8_t>::lowest())
                    : int64_t(std::numeric_limits<int8_t>::max());
    case at::kShort:
      return lowest ? int64_t(std::numeric_limits<int16_t>::lowest())
                    : int64_t(std::numeric_limits<int16_t>::max());
    case at::kInt:
      return lowest ? int64_t(std::numeric_limits<int32_t>::lowest())
                    : int64_t(std::numeric_limits<int32_t>::max());
    default:
      return lowest ? std::numeric_limits<int64_t>::lowest()
                    : std::numeric_limits<int64_t>::max();
  }
}

inline at::Scalar DtypeLowest(at::ScalarType type) {
  if (at::isFloatingType(type)) {
    return at::Scalar(-std::numeric_limits<double>::infinity());
  }
  if (type == at::kBool) {
    return at::Scalar(false);
  }
  return at::Scalar(IntegralExtreme(type, /*lowest=*/true));
}

inline at::Scalar DtypeHighest(at::ScalarType type) {
  if (at::isFloatingType(type)) {
    return at::Scalar(std::numeric_limits<double>::infinity());
  }
  if (type == at::kBool) {
    return at::Scalar(true);
  }
  return at::Scalar(IntegralExtreme(type, /*lowest=*/false));
}

// The dtype a clamp with scalar bounds computes in.
//
// ATen's clamp meta promotes its bounds into the result, so a float bound moves
// the whole op to the default float dtype: `clamp_min(0.5)` on an int32 tensor
// answers f32 [0.5, 2.0], while `clamp_min(0)` on the same tensor stays int32.
// Converting the bounds into self's dtype instead truncates them, and
// `clamp_min(0.5)` would then clamp at 0 for every value in [0, 1). The bounds
// are converted to the promoted type and self is cast to it.
inline at::ScalarType ClampComputeDtype(
    const at::Tensor& self,
    const ::std::optional<at::Scalar>& min,
    const ::std::optional<at::Scalar>& max) {
  auto dtype = self.scalar_type();
  if (min.has_value()) {
    dtype = c10::promoteTypes(dtype, at::result_type(self, min.value()));
  }
  if (max.has_value()) {
    dtype = c10::promoteTypes(dtype, at::result_type(self, max.value()));
  }
  return dtype;
}

} // namespace at::native::flagos::gcu

// Issues a topsaten op on the shared stream of `guard_tensor`'s device and
// waits for it, mirroring EXEC_ASCEND_CMD's synchronous contract.
//
// "Synchronous" has to mean both directions. The trailing synchronise waits for
// the op just issued; it does not wait for work another producer already queued,
// and on this backend there is another producer. `mean.dim`, `pow.Scalar` and
// the other reduction/pointwise ops that carry `= flaggems` in
// configs/backends_gcu.conf run through triton, and topsaten submits to the same
// tops stream triton launches on -- but an op that is merely *submitted* behind
// a queued kernel does not read that kernel's output on this hardware. Measured
// on an S60 at `transformer_blocks.0.attn.norm_q` -- the RMSNorm every attention
// block of the Qwen-Image transformer runs -- with the producer
// `variance = hidden_states.to(float32).pow(2).mean(-1, keepdim=True)`
// (FlagGems) and the consumer `rsqrt(variance + eps)` (topsaten), compared
// against a float64 reference on the real 1024x1024 activation: 40 of 40 rounds
// wrong at 2.371e-02 with no barrier, 0 of 40 wrong at 6.941e-09 when the stream
// was drained before the consumer was issued. A stale `variance` is not a
// small numerical difference: `variance + eps` can come out negative, `rsqrt`
// then returns NaN, and the instrumented run recorded this module's output NaN
// in exactly 131072 places -- 1024 rows of 128 lanes -- with finite values as
// large as 5.66e7 beside them. That is one poisoned row per attention head, in
// every block, on every step, and the image the pipeline decoded after 50 such
// steps was black (mean=0.0000 std=0.0000, against mean=0.3334 std=0.2581 for
// the vendor leg from the same initial latents).
//
// The pre-op drain is close to free in the steady state, because the post-op
// drain below has already emptied the stream: only work issued since the last
// topsaten op can be in flight, which is exactly the FlagGems work that needs
// waiting for.
#define EXEC_TOPSATEN_CMD(op, guard_tensor, ...)                              \
  do {                                                                        \
    at::native::flagos::gcu::EnsureTopsatenInit();                            \
    at::native::flagos::gcu::TopsDeviceGuard _tops_guard((guard_tensor));     \
    topsStream_t _tops_stream =                                               \
        at::native::flagos::gcu::GetCurrentTopsStream();                      \
    topsError_t _tops_pre = topsStreamSynchronize(_tops_stream);              \
    TORCH_CHECK(                                                              \
        _tops_pre == topsSuccess,                                             \
        #op, " pre-stream sync failed: ", topsGetErrorString(_tops_pre));     \
    topsatenStatus_t _tops_status = topsaten::op(__VA_ARGS__, _tops_stream);   \
    TORCH_CHECK(                                                              \
        _tops_status == TOPSATEN_STATUS_SUCCESS,                              \
        #op, " failed: ",                                                     \
        at::native::flagos::gcu::TopsatenStatusName(_tops_status));           \
    topsError_t _tops_sync = topsStreamSynchronize(_tops_stream);             \
    TORCH_CHECK(                                                              \
        _tops_sync == topsSuccess,                                            \
        #op, " stream sync failed: ", topsGetErrorString(_tops_sync));        \
  } while (0)

#endif // USE_GCU
