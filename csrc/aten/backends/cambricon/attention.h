// Copyright (c) 2026, BAAI. All rights reserved.
//
// Entry points of the Cambricon CNNL operator layer.

#pragma once

#ifdef USE_CAMBRICON

#include <ATen/ATen.h>

namespace at::native::flagos::cambricon {

// `mul.Tensor` through cnnlOpTensor. See mul.cc.
at::Tensor MulTensor(const at::Tensor& self, const at::Tensor& other);

#ifdef FLAGOS_CAMBRICON_CNNL_EXTRA
// Fused scaled-dot-product attention through cnnlFlashAttentionForward_v2.
// See attention.cc.
at::Tensor ScaledDotProductAttention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    double scale, bool is_causal);
#endif // FLAGOS_CAMBRICON_CNNL_EXTRA

} // namespace at::native::flagos::cambricon

#endif // USE_CAMBRICON