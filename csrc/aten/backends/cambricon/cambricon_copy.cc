// Copyright (c) 2026, BAAI. All rights reserved.

#include "cambricon_copy.h"

#if defined(USE_CAMBRICON)

#include "cambricon_common.h"

#include <algorithm>

namespace at::native::flagos::cambricon {

bool StridedCopy(const at::Tensor& dst, const at::Tensor& src) {
  if (dst.device() != src.device() || dst.sizes() != src.sizes() ||
      dst.scalar_type() != src.scalar_type()) {
    return false;
  }
  if (dst.numel() == 0) {
    return true;
  }
  // The dtype set the generated kernels serve, spelled here rather than shared:
  // those helpers are internal to cambricon_kernels.cc.
  const auto dt = src.scalar_type();
  if (dt != at::kFloat && dt != at::kHalf && dt != at::kBFloat16 &&
      dt != at::kLong) {
    return false;
  }

  CambriconDeviceGuard guard(src);
  const auto handle = HandleForDevice(static_cast<int>(src.device().index()));
  TensorDesc in_desc, out_desc;
  if (!dst.is_contiguous()) {
    out_desc.SetStrided(dst);
  } else {
    out_desc.Set(dst);
  }
  const auto strides = src.strides();
  if (dst.is_contiguous() &&
      std::find(strides.begin(), strides.end(), 0) != strides.end()) {
    auto shape = src.sizes().vec();
    for (const auto axis : c10::irange(shape.size())) {
      if (strides[axis] == 0) shape[axis] = 1;
    }
    const auto base = src.as_strided(shape, strides);
    if (base.is_contiguous()) {
      in_desc.Set(base);
      CheckCnnl(
          cnnlExpand(handle, in_desc.get(), base.const_data_ptr(),
                     out_desc.get(), dst.mutable_data_ptr()),
          "cnnlExpand (strided copy)");
      return true;
    }
  }
  // The source keeps its strides; that is the whole point of the call.
  in_desc.SetStrided(src);
  CheckCnnl(
      cnnlCopy_v2(
          handle, in_desc.get(), src.const_data_ptr(), out_desc.get(),
          dst.mutable_data_ptr(), nullptr, 0),
      "cnnlCopy_v2 (strided copy)");
  return true;
}

} // namespace at::native::flagos::cambricon

#endif
