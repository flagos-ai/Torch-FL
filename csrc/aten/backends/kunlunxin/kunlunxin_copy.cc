// Copyright (c) 2026, BAAI. All rights reserved.
//
// On-device copy primitives for the Kunlunxin XPU backend. kunlunxin_copy.h
// explains why these exist and what each one declines; this is the body.

#include "kunlunxin_copy.h"

#if defined(USE_KUNLUNXIN) && defined(FLAGOS_KUNLUNXIN_KERNEL)

#include <ATen/ATen.h>

#include "kunlunxin_common.h"

namespace at::native::flagos::kunlunxin {

namespace {

// Which dtypes these two primitives hand to the vendor. The XPU op library's
// copy and cast wrappers are exercised on the dtype pairs this backend's
// operators need -- the float types, and the sub-int64 integer types a mask or
// an index tensor passes through. int64, double and complex are declined: they
// have not been measured through this path on this hardware, and declining is
// free, because both callers keep their host round-trip as the fallback. An
// unmapped pair therefore costs a memcpy pair rather than risking a wrong
// answer.
bool KunlunxinCopyDtype(at::ScalarType type) {
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

// Opens the device guard and drains the stream on both sides of one vendor
// call -- the same bracket KUNLUNXIN_EXEC builds for an operator kernel,
// hoisted into an object because these two functions have to keep the guard
// alive across the destination's allocation. See DrainXpuStream for why both
// drains are there.
class KunlunxinCopyCall {
 public:
  explicit KunlunxinCopyCall(const at::Tensor& device_tensor)
      : guard_(device_tensor) {
    DrainXpuStream();
  }

  int finish(int status) const {
    DrainXpuStream();
    return status;
  }

 private:
  KunlunxinDeviceGuard guard_;
};

} // namespace

bool StridedCopy(const at::Tensor& dst, const at::Tensor& src) {
  if (!dst.defined() || !src.defined()) {
    return false;
  }
  if (!dst.is_privateuseone() || !src.is_privateuseone()) {
    return false;
  }
  if (dst.device() != src.device()) {
    return false;
  }
  if (dst.scalar_type() != src.scalar_type()) {
    return false;
  }
  if (!KunlunxinCopyDtype(dst.scalar_type())) {
    return false;
  }
  if (dst.numel() == 0) {
    return true;
  }

  // A destination dim of stride 0 with extent > 1 aliases every element of that
  // dim onto one address. The vendor descriptor carries strides, so it would
  // happily write one element's worth and report success.
  for (int64_t dim = 0; dim < dst.dim(); ++dim) {
    if (dst.size(dim) > 1 && dst.stride(dim) == 0) {
      return false;
    }
  }

  // The vendor's strided copy is `contiguous`: it materialises the source
  // through its own strides into the destination descriptor. That is not a
  // general broadcast copy, so a source broadcasting against a differently
  // shaped destination takes the caller's host round-trip instead. A rank-0
  // pair still compares equal here, because XdnnTensorWrapper rewrites a
  // scalar operand to the equivalent 1-element vector on both sides.
  if (dst.sizes() != src.sizes()) {
    return false;
  }

  KunlunxinCopyCall call(dst);
  xpytorch::xpu::api::Context* const context = GetKunlunxinContext(dst);
  XdnnTensorWrapper dst_wrapper(dst);
  XdnnTensorWrapper src_wrapper(src);
  const int status = xdnn_pytorch::contiguous(
      context,
      src_wrapper.get(),
      xdnn_pytorch::MemoryFormat::Default,
      dst_wrapper.get());
  return call.finish(status) == 0;
}

at::Tensor DtypeCast(const at::Tensor& src, at::ScalarType dtype) {
  if (!src.defined()) {
    return at::Tensor();
  }
  if (!src.is_privateuseone()) {
    return at::Tensor();
  }
  if (src.scalar_type() == dtype) {
    return at::Tensor();
  }
  if (!KunlunxinCopyDtype(src.scalar_type()) || !KunlunxinCopyDtype(dtype)) {
    return at::Tensor();
  }

  at::Tensor src_contig = src.is_contiguous() ? src : src.contiguous();

  // The device is bound before the destination exists. at::empty resolves the
  // device from the current one, so allocating first would put the result on
  // whichever card happened to be selected.
  KunlunxinCopyCall call(src_contig);
  at::Tensor out =
      at::empty(src_contig.sizes(), src_contig.options().dtype(dtype));
  if (src_contig.numel() == 0) {
    return out;
  }

  xpytorch::xpu::api::Context* const context = GetKunlunxinContext(out);
  XdnnTensorWrapper src_wrapper(src_contig);
  XdnnTensorWrapper out_wrapper(out);
  // `cast` takes no dtype argument: the destination descriptor's declared type
  // *is* the target, which is why `out` was allocated at `dtype` above.
  const int status =
      xdnn_pytorch::cast(context, src_wrapper.get(), out_wrapper.get());
  if (call.finish(status) != 0) {
    return at::Tensor();
  }
  return out;
}

} // namespace at::native::flagos::kunlunxin

#endif // USE_KUNLUNXIN && FLAGOS_KUNLUNXIN_KERNEL
