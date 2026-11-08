// Copyright (c) 2026, BAAI. All rights reserved.

#include "cambricon_copy.h"

#if defined(USE_CAMBRICON)

#include "cambricon_common.h"

namespace at::native::flagos::cambricon {

bool StridedCopy(const at::Tensor& dst, const at::Tensor& src) {
  if (!dst.is_contiguous()) {
    return false;
  }
  if (dst.sizes() != src.sizes() || dst.scalar_type() != src.scalar_type()) {
    return false;
  }
  if (dst.numel() == 0) {
    return true;
  }
  // The dtype set the generated kernels serve, spelled here rather than shared:
  // those helpers are internal to cambricon_kernels.cc.
  const auto dt = src.scalar_type();
  if (dt != at::kFloat && dt != at::kHalf && dt != at::kBFloat16) {
    return false;
  }

  CambriconDeviceGuard guard(src);
  const auto handle = HandleForDevice(static_cast<int>(src.device().index()));
  TensorDesc in_desc, out_desc;
  // The source keeps its strides; that is the whole point of the call.
  // cnnlSetTensorDescriptorEx requires the innermost stride to be 1, which a
  // view of a dense tensor always has.
  in_desc.SetStrided(src);
  out_desc.Set(dst);
  CheckCnnl(
      cnnlCopy_v2(
          handle, in_desc.get(), src.const_data_ptr(), out_desc.get(),
          dst.mutable_data_ptr(), nullptr, 0),
      "cnnlCopy_v2 (strided copy)");
  return true;
}

} // namespace at::native::flagos::cambricon

#endif
