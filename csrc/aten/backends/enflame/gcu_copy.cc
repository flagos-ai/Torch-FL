// Copyright (c) 2026, BAAI. All rights reserved.

#include "gcu_copy.h"

#if defined(USE_GCU) && defined(FLAGOS_GCU_KERNEL)

#include <ATen/ATen.h>

#include "topsaten_common.h"

namespace at::native::flagos::gcu {

namespace {

// The dtypes topsatenCopy / topsatenToCopy may carry, measured rather than
// inferred.
//
// `/tmp/topsaten_cast_matrix.cc` runs every ordered (source, destination) pair
// over {fp32, bf16, fp16, i32, i16, i8, u8, pred, i64, f64} on S60 and compares
// the result against a host reference: every pair drawn from the eight dtypes
// listed below converts exactly, and every pair involving i64 or f64 either
// returns NOT_SUPPORT or -- worse, in the i64 case -- returns SUCCESS while
// writing a wrong result. That is the same set TopsatenSupportsDtype admits,
// which is why the only thing added here is the exclusion of the dtypes
// ToTopsatenDataType has no mapping for (ATen's UInt16/32/64, which that
// predicate does not catch and which the wrapper's own switch would raise on).
static bool TopsatenCopyDtype(at::ScalarType type) {
  if (!TopsatenSupportsDtype(type)) {
    return false;
  }
  switch (type) {
    case at::kFloat:
    case at::kHalf:
    case at::kBFloat16:
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

// A topsatenCopy-family call, bracketed the way EXEC_TOPSATEN_CMD brackets
// every other topsaten op.
//
// The submission guard preserves producer ordering. Native-host builds drain
// only at a change between the Triton and topsaten producers with ABI 1;
// ABI 2 orders the transition with a device event. Ordinary builds retain the
// legacy pre/post synchronization. Unlike
// EXEC_TOPSATEN_CMD this reports a vendor refusal as a status instead of
// raising, because a refusal is exactly what the caller's host fallback is for:
// only a stream error, which no fallback can serve, is fatal here.
class TopsCopyCall {
 public:
  explicit TopsCopyCall(const at::Tensor& device_tensor)
      : guard_(device_tensor) {
    EnsureTopsatenInit();
    stream_ = GetCurrentTopsStream();
    submission_ = std::make_unique<TopsSubmission>(stream_);
  }

  topsatenStatus_t finish(topsatenStatus_t status) {
    const topsError_t post = submission_->Finish();
    TORCH_CHECK(
        post == topsSuccess, "topsaten copy stream sync failed: ",
        topsGetErrorString(post));
    return status;
  }

  topsStream_t stream() const {
    return stream_;
  }

 private:
  TopsDeviceGuard guard_;
  topsStream_t stream_ = nullptr;
  std::unique_ptr<TopsSubmission> submission_;
};

}  // namespace

at::Tensor Int64Words(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.scalar_type() == at::kLong, "expected int64 storage");
  auto sizes = tensor.sizes().vec(), strides = tensor.strides().vec();
  for (auto& stride : strides) stride *= 2;
  sizes.push_back(2);
  strides.push_back(1);
  auto alias = at::empty({0}, tensor.options().dtype(at::kInt));
  alias.set_(tensor.storage(), tensor.storage_offset() * 2, sizes, strides);
  return alias;
}

bool StridedCopy(const at::Tensor& dst, const at::Tensor& src) {
  if (!dst.defined() || !src.defined()) {
    return false;
  }
  if (!dst.is_privateuseone() || !src.is_privateuseone()) {
    return false;
  }
  // Both operands are addressed through the *current* device, so a copy across
  // devices is not something this call can express.
  if (dst.device() != src.device()) {
    return false;
  }
  // topsatenCopy casts the source into the destination dtype as well, but only
  // the same-dtype case is measured for a strided source; a dtype change goes
  // through DtypeCast below, where the matrix covers it.
  if (dst.scalar_type() != src.scalar_type()) {
    return false;
  }
  if (src.scalar_type() == at::kLong) {
    // Copy both 32-bit words, preserving every int64 bit and arbitrary strides.
    // The vendor has no int64 arithmetic kernel; this is a storage alias only.
    return StridedCopy(Int64Words(dst), Int64Words(src));
  }
  if (!TopsatenCopyDtype(dst.scalar_type())) {
    return false;
  }
  if (dst.numel() == 0) {
    return true;  // nothing to copy
  }
  // A destination with a repeated element (a size>1 dim of stride 0) has no
  // well-defined value to write, and ATen never builds one here: dst comes from
  // at::empty or from the caller's own tensor, which copy_ has already
  // validated.
  const auto dst_sizes = dst.sizes();
  const auto dst_strides = dst.strides();
  for (size_t i = 0; i < dst_strides.size(); ++i) {
    if (dst_strides[i] == 0 && dst_sizes[i] > 1) {
      return false;
    }
  }

  TopsatenTensorWrapper dst_wrap(dst);
  TopsatenTensorWrapper src_wrap(src);
  // A zero stride in src is the shape `.expand` produces for a broadcast
  // operand, and it is the exact case `contiguous()` and `clone()` hand here --
  // measured to work as-is (/tmp/topsaten_copy_probe2.cc, case A), so the
  // description is passed through unnormalised. The vendor's own contract is
  // that src need only be broadcastable with dst, and it reports a violation as
  // BAD_PARAM, which lands in the fallback below rather than in a silent
  // truncation.
  TopsCopyCall call(dst);
  const topsatenStatus_t status = topsaten::topsatenCopy(
      dst_wrap.get(), src_wrap.get(), /*non_blocking=*/false, call.stream());
  return call.finish(status) == TOPSATEN_STATUS_SUCCESS;
}

at::Tensor DtypeCast(const at::Tensor& src, at::ScalarType dtype) {
  if (!src.defined() || !src.is_privateuseone()) {
    return {};
  }
  if (src.scalar_type() == at::kInt && dtype == at::kLong) {
    auto out = at::empty(src.sizes(), src.options().dtype(dtype));
    if (src.numel() == 0) return out;
    auto words = Int64Words(out);
    words.select(-1, 0).copy_(src);
    words.select(-1, 1).copy_(-at::lt(src, 0).to(at::kInt));
    return out;
  }
  if (src.scalar_type() == at::kLong && TopsatenCopyDtype(dtype)) {
    if (src.numel() == 0) return at::empty(src.sizes(), src.options().dtype(dtype));
    // A truncating vendor int64 cast silently changes large indices. Check the
    // sign extension on the device before using the measured int32 cast. Only
    // the scalar validity flag reaches the host; tensor arithmetic stays here.
    auto words = Int64Words(src);
    auto low = words.select(-1, 0), high = words.select(-1, 1);
    auto sign = -at::lt(low, 0).to(at::kInt);
    if (!at::all(at::eq(high, sign)).item<bool>()) return {};
    if (dtype == at::kInt) return low.contiguous();
    return DtypeCast(low, dtype);
  }
  if (!TopsatenCopyDtype(src.scalar_type()) || !TopsatenCopyDtype(dtype)) {
    return {};
  }
  if (src.scalar_type() == dtype) {
    return {};
  }
  // topsatenToCopy reads the source through its sizes and strides, but the
  // callers here always pass a contiguous operand and the cast is the point, so
  // the (rare) strided caller is normalised first.
  at::Tensor src_c = src.is_contiguous() ? src : src.contiguous();

  // The device is selected before the output is allocated: a tops pointer
  // resolves only against the current device, so allocating first would put the
  // result on whichever card happened to be current.
  TopsCopyCall call(src_c);
  at::Tensor out = at::empty(src_c.sizes(), src_c.options().dtype(dtype));
  if (src_c.numel() == 0) {
    return out;
  }

  TopsatenTensorWrapper out_wrap(out);
  TopsatenTensorWrapper src_wrap(src_c);
  const topsatenStatus_t status = topsaten::topsatenToCopy(
      out_wrap.get(), src_wrap.get(), ToTopsatenDataType(dtype),
      TOPSATEN_LAYOUT_STRIDED, TOPSATEN_DEVICE_GCU, TOPSATEN_MEMORY_CONTIGUOUS,
      /*non_blocking=*/false, call.stream());
  if (call.finish(status) != TOPSATEN_STATUS_SUCCESS) {
    return {};
  }
  return out;
}

}  // namespace at::native::flagos::gcu

#endif  // USE_GCU && FLAGOS_GCU_KERNEL
