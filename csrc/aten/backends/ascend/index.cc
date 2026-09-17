// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <set>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

namespace {

at::Tensor IndexBoolMaskAscend(const at::Tensor& self,
                               const at::Tensor& mask,
                               int64_t indexed_dim) {
  namespace ascend = at::native::flagos::ascend;

  TORCH_CHECK(mask.scalar_type() == at::kBool,
              "index: boolean mask must have dtype bool");
  TORCH_CHECK(mask.dim() > 0,
              "index: boolean mask must have at least one dimension");
  TORCH_CHECK(indexed_dim + mask.dim() <= self.dim(),
              "too many indices for tensor of dimension ", self.dim());
  for (int64_t i = 0; i < mask.dim(); ++i) {
    TORCH_CHECK(
        mask.size(i) == self.size(indexed_dim + i),
        "The shape of the mask ", mask.sizes(), " at index ", i,
        " does not match the shape of the indexed tensor ", self.sizes(),
        " at index ", indexed_dim + i);
  }

  auto contiguous_mask = mask.contiguous();
  int64_t selected = contiguous_mask.to(at::kLong).sum().item<int64_t>();

  // aclnnIndex consumes indexed dimensions from the front. Move the dimensions
  // covered by the boolean mask there, followed by the untouched leading and
  // trailing dimensions.
  std::vector<int64_t> perm;
  perm.reserve(self.dim());
  for (int64_t i = 0; i < mask.dim(); ++i) {
    perm.push_back(indexed_dim + i);
  }
  for (int64_t i = 0; i < indexed_dim; ++i) {
    perm.push_back(i);
  }
  for (int64_t i = indexed_dim + mask.dim(); i < self.dim(); ++i) {
    perm.push_back(i);
  }

  auto src = indexed_dim == 0 ? self : self.permute(perm).contiguous();
  std::vector<int64_t> acl_out_sizes{selected};
  for (int64_t i = mask.dim(); i < src.dim(); ++i) {
    acl_out_sizes.push_back(src.size(i));
  }
  auto acl_out = ascend::OpPreparation::apply_tensor_without_format(
      acl_out_sizes, self.options());

  std::vector<ascend::AclTensorWrapper> acl_indices_storage;
  acl_indices_storage.emplace_back(contiguous_mask);
  std::vector<const aclTensor*> acl_indices_ptrs{
      acl_indices_storage.front().get()};
  ascend::AclTensorListWrapper acl_indices(acl_indices_ptrs);
  ascend::AclTensorWrapper acl_src(src);
  ascend::AclTensorWrapper acl_result(acl_out);

  EXEC_ASCEND_CMD(
      aclnnIndex, acl_src.get(), acl_indices.get(), acl_result.get());
  acl_indices.release();

  if (indexed_dim == 0) {
    return acl_out;
  }

  // aclnnIndex returns [selected, leading..., trailing...]. Restore the
  // PyTorch order [leading..., selected, trailing...].
  std::vector<int64_t> out_perm;
  out_perm.reserve(acl_out.dim());
  for (int64_t i = 1; i <= indexed_dim; ++i) {
    out_perm.push_back(i);
  }
  out_perm.push_back(0);
  for (int64_t i = indexed_dim + 1; i < acl_out.dim(); ++i) {
    out_perm.push_back(i);
  }
  return acl_out.permute(out_perm).contiguous();
}

} // namespace

at::Tensor IndexTensorKernelAscend(const at::Tensor& self,
                                   const c10::List<::std::optional<at::Tensor>>& indices) {
  namespace ascend = at::native::flagos::ascend;

  // Collect defined index tensors and their positions
  std::vector<at::Tensor> defined_indices;
  std::vector<int64_t> defined_positions;
  for (size_t i = 0; i < indices.size(); ++i) {
    const auto& idx = indices.get(i);
    if (idx.has_value()) {
      defined_indices.push_back(idx.value());
      defined_positions.push_back(static_cast<int64_t>(i));
    }
  }

  TORCH_CHECK(!defined_indices.empty(), "index: expected at least one index tensor");

  size_t bool_index_count = 0;
  size_t bool_index_offset = 0;
  for (size_t i = 0; i < defined_indices.size(); ++i) {
    if (defined_indices[i].scalar_type() == at::kBool) {
      ++bool_index_count;
      bool_index_offset = i;
    }
  }
  if (bool_index_count > 0) {
    TORCH_CHECK(
        defined_indices.size() == 1 && bool_index_count == 1,
        "index: mixing a boolean mask with other advanced indices is not yet "
        "supported by the Ascend kernel");
    return IndexBoolMaskAscend(
        self, defined_indices[bool_index_offset],
        defined_positions[bool_index_offset]);
  }

  // Compute broadcast shape of all index tensors
  std::vector<int64_t> broadcast_shape = defined_indices[0].sizes().vec();
  for (size_t i = 1; i < defined_indices.size(); ++i) {
    broadcast_shape = at::infer_size(broadcast_shape, defined_indices[i].sizes().vec());
  }

  // Expand all index tensors to the broadcast shape and make contiguous
  std::vector<at::Tensor> expanded_indices;
  expanded_indices.reserve(defined_indices.size());
  for (auto& idx : defined_indices) {
    expanded_indices.push_back(idx.expand(broadcast_shape).contiguous());
  }

  // Determine if indexed dimensions are contiguous
  bool contiguous_indexed = true;
  for (size_t i = 1; i < defined_positions.size(); ++i) {
    if (defined_positions[i] != defined_positions[i-1] + 1) {
      contiguous_indexed = false;
      break;
    }
  }

  // Compute output shape following PyTorch advanced indexing semantics:
  // - If indexed dims are contiguous: [leading_dims..., broadcast_shape..., trailing_dims...]
  // - If not contiguous: [broadcast_shape..., non_indexed_dims...]
  std::vector<int64_t> out_sizes;
  if (contiguous_indexed) {
    int64_t first_indexed = defined_positions.front();
    int64_t last_indexed = defined_positions.back();
    // Leading non-indexed dims
    for (int64_t i = 0; i < first_indexed; ++i) {
      out_sizes.push_back(self.size(i));
    }
    // Broadcast shape replaces the indexed dims
    out_sizes.insert(out_sizes.end(), broadcast_shape.begin(), broadcast_shape.end());
    // Trailing non-indexed dims
    for (int64_t i = last_indexed + 1; i < self.dim(); ++i) {
      out_sizes.push_back(self.size(i));
    }
  } else {
    // Broadcast shape goes first
    out_sizes = broadcast_shape;
    // Then all non-indexed dims in order
    std::set<int64_t> indexed_set(defined_positions.begin(), defined_positions.end());
    for (int64_t i = 0; i < self.dim(); ++i) {
      if (indexed_set.find(i) == indexed_set.end()) {
        out_sizes.push_back(self.size(i));
      }
    }
  }

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_sizes, self.options());

  // For aclnnIndex, we pass only the defined indices (no nullptr).
  // aclnnIndex interprets the tensor list as indexing the first N dimensions.
  // If the indexed dims are not the leading dims, we need to permute self
  // so that indexed dims come first, then permute the output back.

  at::Tensor src = self;
  at::Tensor result_tensor;

  if (contiguous_indexed && defined_positions.front() == 0) {
    // Simple case: indexed dims are leading — pass indices directly
    std::vector<ascend::AclTensorWrapper> acl_indices_storage;
    acl_indices_storage.reserve(expanded_indices.size());
    std::vector<const aclTensor*> acl_indices_ptrs;
    acl_indices_ptrs.reserve(expanded_indices.size());

    for (auto& idx : expanded_indices) {
      acl_indices_storage.emplace_back(idx);
      acl_indices_ptrs.push_back(acl_indices_storage.back().get());
    }

    ascend::AclTensorListWrapper acl_indices(acl_indices_ptrs);
    ascend::AclTensorWrapper acl_self(src);
    ascend::AclTensorWrapper acl_out(out);

    EXEC_ASCEND_CMD(aclnnIndex, acl_self.get(), acl_indices.get(), acl_out.get());
    acl_indices.release();
    result_tensor = out;
  } else {
    // Complex case: indexed dims are not leading.
    // Permute self so indexed dims come first, then index, then permute back.
    std::vector<int64_t> perm;
    std::vector<int64_t> unperm;
    perm.reserve(self.dim());
    unperm.resize(self.dim());

    // Indexed dims first
    for (auto pos : defined_positions) {
      perm.push_back(pos);
    }
    // Then non-indexed dims
    std::set<int64_t> indexed_set(defined_positions.begin(), defined_positions.end());
    for (int64_t i = 0; i < self.dim(); ++i) {
      if (indexed_set.find(i) == indexed_set.end()) {
        perm.push_back(i);
      }
    }
    // Compute inverse permutation
    for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
      unperm[perm[i]] = i;
    }

    src = self.permute(perm).contiguous();

    // Now indexed dims are leading — compute output for permuted tensor
    std::vector<int64_t> perm_out_sizes = broadcast_shape;
    for (int64_t i = static_cast<int64_t>(defined_positions.size()); i < src.dim(); ++i) {
      perm_out_sizes.push_back(src.size(i));
    }

    auto perm_out = ascend::OpPreparation::apply_tensor_without_format(
        perm_out_sizes, src.options());

    std::vector<ascend::AclTensorWrapper> acl_indices_storage;
    acl_indices_storage.reserve(expanded_indices.size());
    std::vector<const aclTensor*> acl_indices_ptrs;
    acl_indices_ptrs.reserve(expanded_indices.size());

    for (auto& idx : expanded_indices) {
      acl_indices_storage.emplace_back(idx);
      acl_indices_ptrs.push_back(acl_indices_storage.back().get());
    }

    ascend::AclTensorListWrapper acl_indices(acl_indices_ptrs);
    ascend::AclTensorWrapper acl_src(src);
    ascend::AclTensorWrapper acl_perm_out(perm_out);

    EXEC_ASCEND_CMD(aclnnIndex, acl_src.get(), acl_indices.get(), acl_perm_out.get());
    acl_indices.release();

    // For contiguous indexed dims not at front, we need to move the broadcast
    // dims back to the correct position. The output of aclnnIndex has shape:
    // [broadcast_shape..., non_indexed_dims...]
    // We need: [leading_non_indexed..., broadcast_shape..., trailing_non_indexed...]
    if (contiguous_indexed) {
      // Permute output: broadcast dims should be at position first_indexed
      int64_t first_indexed = defined_positions.front();
      int64_t n_broadcast = static_cast<int64_t>(broadcast_shape.size());
      int64_t n_non_indexed_before = first_indexed;

      std::vector<int64_t> out_perm;
      // First: non-indexed dims that were before the indexed block
      for (int64_t i = n_broadcast; i < n_broadcast + n_non_indexed_before; ++i) {
        out_perm.push_back(i);
      }
      // Then: broadcast dims
      for (int64_t i = 0; i < n_broadcast; ++i) {
        out_perm.push_back(i);
      }
      // Then: remaining non-indexed dims (after the indexed block)
      for (int64_t i = n_broadcast + n_non_indexed_before; i < perm_out.dim(); ++i) {
        out_perm.push_back(i);
      }
      result_tensor = perm_out.permute(out_perm).contiguous();
    } else {
      result_tensor = perm_out;
    }
  }

  return result_tensor;
}

REGISTER_IMPL_TO_DISPATCHER(IndexTensorFn, index_tensor_dispatcher, Backend::kAscend, IndexTensorKernelAscend)

} // namespace at::native::flagos
