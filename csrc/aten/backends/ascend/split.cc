// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include "device_guard.h"
#include <ATen/core/Tensor.h>
#include <ATen/ops/narrow.h>

namespace at::native::flagos {

// split.Tensor(Tensor(a) self, SymInt split_size, int dim=0) -> Tensor(a)[]
// split_with_sizes(Tensor(a) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]
//
// Pure view ops: they only produce narrowed views over `self`'s existing
// storage, so there is no kernel to call and nothing device-specific to do. We
// build them from at::narrow, whose Ascend route (slice.Tensor) is already
// registered, keeping the results on the flagos device without any copy.
//
// Without these, FSDP2 fails with "split.Tensor: backend not registered" --
// fully_shard() splits each flat parameter across the mesh, and DTensor's
// sharding propagation calls split/chunk on the local shard.

::std::vector<at::Tensor> SplitTensorKernelAscend(
    const at::Tensor& self,
    int64_t split_size,
    int64_t dim) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  TORCH_CHECK(
      self.dim() != 0, "split: cannot split a 0-dimensional tensor");
  TORCH_CHECK(split_size > 0, "split: split_size must be > 0, got ", split_size);

  const int64_t ndim = self.dim();
  const int64_t d = c10::maybe_wrap_dim(dim, ndim);
  const int64_t dim_size = self.size(d);

  // Matches ATen: the last chunk is short when dim_size is not a multiple of
  // split_size, and a 0-sized dim yields exactly one empty chunk.
  const int64_t num_splits =
      dim_size == 0 ? 1 : (dim_size + split_size - 1) / split_size;

  ::std::vector<at::Tensor> splits;
  splits.reserve(num_splits);
  for (int64_t i = 0; i < num_splits; ++i) {
    const int64_t start = i * split_size;
    const int64_t length = std::min(split_size, dim_size - start);
    splits.push_back(at::narrow(self, d, start, length));
  }
  return splits;
}

::std::vector<at::Tensor> SplitWithSizesKernelAscend(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  TORCH_CHECK(
      self.dim() != 0, "split_with_sizes: cannot split a 0-dimensional tensor");

  const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
  const int64_t dim_size = self.size(d);

  ::std::vector<at::Tensor> splits;
  splits.reserve(split_sizes.size());
  int64_t start = 0;
  for (const int64_t length : split_sizes) {
    TORCH_CHECK(
        length >= 0,
        "split_with_sizes: split_sizes must be non-negative, got ",
        length);
    splits.push_back(at::narrow(self, d, start, length));
    start += length;
  }
  TORCH_CHECK(
      start == dim_size,
      "split_with_sizes: split sizes sum to ",
      start,
      " but tensor has ",
      dim_size,
      " elements along dim ",
      d);
  return splits;
}

// split_with_sizes_copy.out is FSDP2's all-gather copy-out entry point. Each
// output is a separate destination buffer, so there is nothing to fuse: the
// per-split at::copy_ (one aclnnInplaceCopy each) is the whole kernel. CUDA's
// generated version does one fused pass; CANN has no equivalent primitive, so
// the per-split launch overhead is inherent here.
void SplitWithSizesCopyOutKernelAscend(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim,
    at::TensorList out) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  auto splits = SplitWithSizesKernelAscend(self, split_sizes, dim);
  TORCH_CHECK(
      splits.size() == out.size(),
      "split_with_sizes_copy: expected ",
      splits.size(),
      " outputs but got ",
      out.size());
  for (size_t i = 0; i < splits.size(); ++i) {
    out[i].copy_(splits[i]);
  }
}

REGISTER_IMPL_TO_DISPATCHER(SplitTensorFn, split_tensor_dispatcher, Backend::kAscend, SplitTensorKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(SplitWithSizesFn, split_with_sizes_dispatcher, Backend::kAscend, SplitWithSizesKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(
    SplitWithSizesCopyOutFn,
    split_with_sizes_copy_out_dispatcher,
    Backend::kAscend,
    SplitWithSizesCopyOutKernelAscend)

} // namespace at::native::flagos
