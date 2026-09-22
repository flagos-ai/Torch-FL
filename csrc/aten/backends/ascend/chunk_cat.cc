// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include "device_guard.h"

#include <ATen/ops/cat.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/reshape.h>

namespace at::native::flagos {
namespace {

std::vector<at::Tensor> PrepareChunkCatInputs(
    at::TensorList tensors,
    int64_t dim,
    int64_t num_chunks) {
  TORCH_CHECK(num_chunks >= 1, "_chunk_cat expects positive num_chunks");
  TORCH_CHECK(!tensors.empty(), "_chunk_cat expects a non-empty input tensor list");

  const auto dtype = tensors[0].scalar_type();
  const auto device = tensors[0].device();
  bool same_ndim = true;
  std::vector<int64_t> leading_sizes;
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.numel() > 0, "_chunk_cat expects non-empty tensor");
    TORCH_CHECK(
        tensor.scalar_type() == dtype,
        "_chunk_cat expects all input tensors with the same dtype");
    TORCH_CHECK(
        tensor.device() == device,
        "_chunk_cat expects all input tensors on the same device");
    same_ndim = same_ndim && tensor.dim() == tensors[0].dim();
  }

  if (same_ndim) {
    dim = c10::maybe_wrap_dim(dim, tensors[0].dim());
  } else {
    TORCH_CHECK(
        dim >= 0,
        "_chunk_cat expects non-negative dim when input tensors have different ndims");
    for (const auto& tensor : tensors) {
      TORCH_CHECK(dim < tensor.dim(), "_chunk_cat expects dim < ndim");
    }
  }

  leading_sizes.assign(tensors[0].sizes().begin(), tensors[0].sizes().begin() + dim);
  for (const auto& tensor : tensors) {
    TORCH_CHECK(
        std::equal(
            leading_sizes.begin(),
            leading_sizes.end(),
            tensor.sizes().begin()),
        "_chunk_cat expects leading dimensions to match");
  }

  std::vector<at::Tensor> prepared;
  prepared.reserve(tensors.size());
  for (const auto& input : tensors) {
    at::Tensor tensor = input;
    const int64_t dim_size = tensor.size(dim);
    const int64_t padded_size =
        ((dim_size + num_chunks - 1) / num_chunks) * num_chunks;
    if (padded_size != dim_size) {
      std::vector<int64_t> pad(2 * (tensor.dim() - dim), 0);
      pad.back() = padded_size - dim_size;
      tensor = at::constant_pad_nd(tensor, pad, 0);
    }

    std::vector<int64_t> shape;
    shape.reserve(dim + 2);
    for (int64_t i = 0; i < dim; ++i) {
      shape.push_back(tensor.size(i));
    }
    shape.push_back(num_chunks);
    shape.push_back(-1);
    prepared.push_back(at::reshape(tensor, shape));
  }
  return prepared;
}

} // namespace

at::Tensor PrivChunkCatKernelAscend(
    at::TensorList tensors,
    int64_t dim,
    int64_t num_chunks) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(tensors));
  auto prepared = PrepareChunkCatInputs(tensors, dim, num_chunks);
  return at::cat(prepared, dim + 1);
}

at::Tensor& PrivChunkCatOutKernelAscend(
    at::TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    at::Tensor& out) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(tensors));
  auto prepared = PrepareChunkCatInputs(tensors, dim, num_chunks);
  // Concatenate straight into `out`. Going through the functional kernel and
  // then out.copy_() would allocate a second full-size buffer and re-copy every
  // byte, and this is FSDP2's per-backward gradient-packing path.
  at::cat_out(out, prepared, dim + 1);
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(
    PrivChunkCatFn,
    priv_chunk_cat_dispatcher,
    Backend::kAscend,
    PrivChunkCatKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(
    PrivChunkCatOutFn,
    priv_chunk_cat_out_dispatcher,
    Backend::kAscend,
    PrivChunkCatOutKernelAscend)

} // namespace at::native::flagos
