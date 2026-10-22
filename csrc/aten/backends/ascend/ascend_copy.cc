// Copyright (c) 2026, BAAI. All rights reserved.

#include "ascend_copy.h"

#include <ATen/core/Tensor.h>
#include "op_api_common.h"

namespace at::native::flagos::ascend {

bool StridedCopy(const at::Tensor& dst, const at::Tensor& src) {
  if (!dst.defined() || !src.defined()) {
    return false;
  }
  if (!dst.is_privateuseone() || !src.is_privateuseone()) {
    return false;
  }
  if (dst.numel() == 0) {
    return true;  // nothing to copy
  }

  // Issue #326: `contiguous()` / `copy_` reach this from platform-shared code
  // that carries no device, so the ambient current device is whatever ran last.
  // `EXEC_ASCEND_CMD` submits to that device's stream and allocates any scratch
  // from an index-less `at::kPrivateUse1` TensorOptions, i.e. also on it. Unlike
  // DtypeCast below this one is not cached, so the executor itself is rebuilt
  // from the operands' own descriptors every call -- measured on CANN 9.0.0,
  // that is enough to keep the result correct across every combination of
  // operand and ambient device (including a cross-device `copy_`), which
  // `test_contiguous_matches_same_device_result` pins. The guard is kept for
  // the stream (and any future workspace), the two ambient reads left here.
  OpDeviceGuard device_guard_(DeviceOf(dst));

  // aclnnInplaceCopy(selfRef, src): writes src into selfRef, honoring the
  // strides/offset recorded on each aclTensor. AclTensorWrapper preserves the
  // tensor's sizes/strides/offset, so a non-contiguous src is copied correctly
  // into the (contiguous) dst without a host round-trip.
  AclTensorWrapper dst_wrap(dst);
  AclTensorWrapper src_wrap(src);
  EXEC_ASCEND_CMD(aclnnInplaceCopy,
                  const_cast<aclTensor*>(dst_wrap.get()),
                  src_wrap.get());
  return true;
}

at::Tensor DtypeCast(const at::Tensor& src, at::ScalarType dtype) {
  if (!src.defined() || !src.is_privateuseone()) {
    return {};
  }
  // Issue #326: `_to_copy` reaches this from platform-shared code that carries
  // no device, so the ambient current device is whatever ran last. That matters
  // more here than in StridedCopy above, because the aclnn call goes through
  // `ExecAscendCached`: the cached executor and its scratch are both *keyed* on
  // `::GetDevice`, so an unguarded call builds them on the other device and
  // reuses them for every later call it is handed. Measured on CANN 9.0.0: a
  // `randperm(50, device="flagos:1")` cast to float32 with flagos:0 current
  // came back holding the *previous* draw -- the allocator had recycled the
  // output block, so the stale read looked like a plausible answer rather than
  // like garbage. That is the failure
  // `test_dtype_cast_matches_same_device_result` reproduces.
  OpDeviceGuard device_guard_(DeviceOf(src));
  // aclnnCast expects a dense input; make src contiguous first (cheap, and the
  // callers in _to_copy already pass a contiguous tensor).
  at::Tensor src_c = src.is_contiguous() ? src : src.contiguous();
  at::Tensor out = at::empty(src_c.sizes(), src_c.options().dtype(dtype));
  if (src_c.numel() == 0) {
    return out;
  }

  // aclnnCast(self, dtype, out): converts self to the given aclDataType
  // on-device. Route through the repeatable-executor cache: RMSNorm emits two
  // fp16<->fp32 casts per layer (285/step) at fixed decode shapes, so the
  // GetWorkspaceSize + aclCreateTensor build cost is paid once per shape. The
  // target aclDataType is baked into the executor at build time, so it must be
  // part of the cache key (folded in via SigHasher::val below).
  const aclDataType acl_dtype = ToAclDataType(dtype);
  static void* opApiFuncAddr = nullptr;
  static void* getWsFuncAddr = nullptr;
  SigHasher hsh;
  hsh.tensor(src_c);
  hsh.tensor(out);
  hsh.val(static_cast<int32_t>(acl_dtype));
  ExecAscendCached(
      "aclnnCast", "aclnnCastGetWorkspaceSize",
      opApiFuncAddr, getWsFuncAddr, hsh.h,
      {&src_c}, {&out},
      [&](GwsFunc gws,
          std::vector<AclTensorWrapper>& in,
          std::vector<AclTensorWrapper>& out_t,
          uint64_t* pws, aclOpExecutor** pex) {
        return gws(in[0].acl_tensor, acl_dtype, out_t[0].acl_tensor, pws, pex);
      });
  return out;
}

} // namespace at::native::flagos::ascend
