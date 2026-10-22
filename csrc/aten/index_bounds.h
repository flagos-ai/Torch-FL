// Copyright (c) 2026, BAAI. All rights reserved.
//
// Index-value range validation for the index family: index.Tensor,
// index_select, index_fill, index_add, index_copy and index_reduce.
//
// Every one of these takes an index tensor whose *values* address `self`
// directly, and ATen defines out-of-range values as an error. The CPU kernels
// enforce that by validating each value against the indexed dimension and
// raising (Indexer::get, ATen/native/cpu/IndexKernelUtils.h). Some vendor
// device kernels drop the check, and then an out-of-range value reads or writes
// outside `self`: silently wrong data, or a VMFault that aborts the process
// (issue #381). The check cannot go back into ATen -- the torch wheel is
// prebuilt -- so it runs in the generated PrivateUse1 wrapper instead. That
// wrapper is the one layer every route passes through (CUDA boxing, FlagGems,
// vendor-native), so the index tensor is validated before any backend kernel
// sees it.
//
// The check is one device reduction plus two host reads per call. That is not
// free -- it measures ~70 us on DCU whatever the index size, against ~20 us for
// the index_select it guards -- so it is compiled in only on the platforms whose
// vendor ATen is known to be unchecked. See FLAGOS_INDEX_BOUNDS_CHECK in
// csrc/CMakeLists.txt. Where the macro is absent both functions are empty
// inlines and the emitted call sites cost nothing.
//
// The cost is deliberately flat in the index size: the reduction runs on the
// device and only its two 0-dim results are read back, so the index tensor is
// never transferred. Copying the index to the host and reducing it there is
// cheaper below ~20k elements (~29 us against ~70 us) and much more expensive
// above them (~2.2 ms against ~70 us at 1M), because this host's device-to-host
// bandwidth is low; see the PR that added this for both measurements.

#pragma once

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/aminmax.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <c10/util/string_utils.h>

#include <string>
#include <tuple>

namespace at::native::flagos {

#if defined(FLAGOS_INDEX_BOUNDS_CHECK)

// The message ATen's CPU kernels raise for an out-of-range index value, shared
// by every op in the family so the value and the dimension are named even where
// the op's own kernel would not have named them.
inline std::string IndexBoundsMessage(
    int64_t value,
    int64_t message_dim,
    int64_t dim_size) {
  return c10::str(
      "index ",
      value,
      " is out of bounds for dimension ",
      message_dim,
      " with size ",
      dim_size);
}

// Whether `index` is the kind of argument the range check applies to.
//
// A bool tensor is a mask, not a list of positions: ATen validates its shape
// against the indexed dimensions before any kernel runs (both on CPU and
// through this plugin), and a mask holds only 0/1, so it cannot carry an
// out-of-range value. A non-integral dtype is rejected by ATen's own dtype
// check, which must keep its place in the error ordering -- checking values
// first would replace "tensors used as indices must be long, int, byte or bool
// tensors" with a bounds error for the same call. An empty index addresses
// nothing and is valid on every dimension.
inline bool IndexNeedsRangeCheck(const at::Tensor& index) {
  if (!index.defined() || index.numel() == 0) {
    return false;
  }
  return at::isIntegralType(index.scalar_type(), /*includeBool=*/false);
}

// Raise when any value of `index` falls outside
// [-self.size(dim), self.size(dim)).
//
// `dim` is the dimension the values address; `message_dim` is the number to
// name in the message. They are the same for the family ops -- index_fill_ is
// the one whose CPU kernel calls the value check with its own `dim` argument --
// but not for index.Tensor: its kernel counts only the advanced indices, so a
// nullopt slice does not advance the count and `x[:, j]` names dimension 0
// while validating against dimension 1. See CheckIndexListInRange.
//
// One reduction gives both ends, so an in-range index costs a single device
// reduction plus the two read-backs that reading the pair costs, and nothing
// proportional to the index size.
//
// The message is ATen's shared index wording, "index <value> is out of bounds
// for dimension <dim> with size <size>". The other family ops raise the same
// condition from their own CPU kernels with their own text (`index_select` and
// `index_reduce` say only "index out of range in self", `index_copy_` prefixes
// the op name), so the value and dimension are named here in every case rather
// than reproduced per op.
//
// The exception is always IndexError, the spelling `Indexer::get` uses and the
// one the CPU kernel of every op in this family raises but two fast paths:
// `index_select` on dimension 1 of a contiguous result, and `index_add_` on
// dimension 0 or the last one of a tensor with more than one dimension, which
// forward to `check_indexarray_range` and `scatter_add_` respectively and raise
// RuntimeError there. Neither is reproducible here for free. In particular
// index_add_'s is not even a property of the op: the forward is gated on an
// int64 index and on `alpha == 1.0`, so `alpha=2` and an int32 index take the
// generic path and raise IndexError for the same out-of-range value. That makes
// it a property of which CPU kernel ran, so the check keeps the one spelling the
// majority of the family uses.
inline void CheckIndexTensorInRangeImpl(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    int64_t message_dim) {
  // An entry past self.dim() is not an addressable dimension; ATen rejects the
  // call itself ("too many indices for tensor of dimension N"), and it does so
  // on every route, so there is nothing to add here. A `dim` outside
  // [-self.dim(), self.dim()) is the same kind of mistake and is likewise
  // ATen's to report -- calling self.size() on it here would replace ATen's
  // dimension error with this file's.
  if (!IndexNeedsRangeCheck(index) || dim >= self.dim() || dim < -self.dim()) {
    return;
  }
  // A negative `dim` counts from the end, and the size has to be read by
  // position.
  const int64_t resolved_dim = dim < 0 ? dim + self.dim() : dim;
  const int64_t dim_size = self.size(resolved_dim);
  // One reduction gives both ends. The public `aminmax` is used rather than the
  // `_aminmax` it dispatches to: the two cost the same here -- measured
  // interleaved in one process on DCU, 66.6 us against 66.4 us for a 4096-element
  // index, both dominated by the two read-backs below -- and `_aminmax` carries
  // ATen's per-process deprecation warning.
  const auto bounds = at::aminmax(index);
  const int64_t index_min = std::get<0>(bounds).item<int64_t>();
  const int64_t index_max = std::get<1>(bounds).item<int64_t>();
  // Negative values are relative to the end of the dimension, so the lower
  // bound is -dim_size; a zero-sized dimension admits no index at all, and
  // both comparisons reject every value as intended.
  if (index_min < -dim_size || index_max >= dim_size) {
    TORCH_CHECK_INDEX(
        false,
        IndexBoundsMessage(
            index_max >= dim_size ? index_max : index_min, message_dim, dim_size));
  }
}

inline void CheckIndexTensorInRange(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index) {
  CheckIndexTensorInRangeImpl(
      self, dim, index, dim < 0 ? dim + self.dim() : dim);
}

// The list form used by index.Tensor. Entry i addresses dimension i of `self`;
// a nullopt entry is a full slice on that dimension (TensorIndex expands
// `x[:, j]` and `x[..., j]` to [None, j], and `x[j, :]` drops the trailing
// slice), so the *position* advances the dimension counter, not the number of
// defined entries.
//
// A boolean entry is the one entry whose width is not one dimension: a mask
// consumes as many dimensions as it has axes, which is what makes
// `x[mask2d, j]` on a (2, 3, 4) tensor index dimension 2 with `j`. Advancing by
// index.dim() reproduces that, including the 0-dim mask that consumes none.
//
// The number the message names is not that counter. ATen's TensorIndexing
// counts only the advanced entries, so `x[:, j]` raises "index 9 is out of
// bounds for dimension 0 with size 6" on the CPU: the number is the sum of the
// widths of the preceding advanced entries, and the size is the real
// dimension's. `message_dim` tracks it separately.
inline void CheckIndexListInRange(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices) {
  int64_t dim = 0;
  int64_t message_dim = 0;
  for (const auto i : c10::irange(indices.size())) {
    const auto entry = indices.get(i);
    if (!entry.has_value() || !entry->defined()) {
      dim += 1;
      continue;
    }
    const at::Tensor& index = *entry;
    if (index.scalar_type() == at::kBool) {
      dim += index.dim();
      message_dim += index.dim();
      continue;
    }
    if (!IndexNeedsRangeCheck(index)) {
      dim += 1;
      message_dim += 1;
      continue;
    }
    // An empty dimension cannot be addressed by any index, so the value is
    // never the interesting part and ATen has a dedicated message for it --
    // libtorch_cpu.so carries the literal, with no value and no dimension
    // number in it. Only this entry point mirrors that: the family ops each
    // treat an empty dimension differently (`index_add_` does not report it at
    // all, `index_select` reports "self indexing axis dim should be positive",
    // `index_fill_` fails inside `setStorage`), so there is no shared message
    // to be faithful to.
    if (dim < self.dim() && self.size(dim) == 0) {
      TORCH_CHECK_INDEX(
          false, "index is out of bounds for dimension with size 0");
    }
    CheckIndexTensorInRangeImpl(self, dim, index, message_dim);
    dim += 1;
    message_dim += 1;
  }
}

#else

inline bool IndexNeedsRangeCheck(const at::Tensor&) {
  return false;
}

inline void CheckIndexTensorInRange(
    const at::Tensor&,
    int64_t,
    const at::Tensor&) {}

inline void CheckIndexListInRange(
    const at::Tensor&,
    const c10::List<c10::optional<at::Tensor>>&) {}

#endif  // FLAGOS_INDEX_BOUNDS_CHECK

}  // namespace at::native::flagos
