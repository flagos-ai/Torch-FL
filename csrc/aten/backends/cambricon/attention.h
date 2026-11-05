// Copyright (c) 2026, BAAI. All rights reserved.
//
// Entry points of the Cambricon CNNL operator layer.

#pragma once

#ifdef USE_CAMBRICON

#include <ATen/ATen.h>

#include <array>

namespace at::native::flagos::cambricon {

// `mul.Tensor` through cnnlOpTensor, and the out= spelling that forwards to it.
// See mul.cc.
at::Tensor MulTensor(const at::Tensor& self, const at::Tensor& other);
at::Tensor& MulOut(
    const at::Tensor& self, const at::Tensor& other, at::Tensor& out);

// `embedding_dense_backward` through cnnlEmbeddingBackward. See embedding.cc.
at::Tensor EmbeddingDenseBackward(
    const at::Tensor& grad_output, const at::Tensor& indices,
    int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq);

#ifdef FLAGOS_CAMBRICON_CNNL_EXTRA
// Fused scaled-dot-product attention through CNNL's flash attention. See
// attention.cc.
//
// `out` is what a caller wants; `lse` is the log-sum-exp the backward needs.
// Returning both from one call keeps the descriptor work in a single place
// instead of a second entry point repeating it to get at the LSE.
struct FusedAttentionOutput {
  at::Tensor out;
  at::Tensor lse;
};

FusedAttentionOutput Forward(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    double scale, bool is_causal);

std::array<at::Tensor, 3> Backward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& out, const at::Tensor& lse,
    double scale, bool is_causal);

// Forward wrapped in an autograd Function, so a training call gets the fused
// kernel for both passes.
at::Tensor ScaledDotProductAttention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    double scale, bool is_causal);
#endif // FLAGOS_CAMBRICON_CNNL_EXTRA

} // namespace at::native::flagos::cambricon

#endif // USE_CAMBRICON