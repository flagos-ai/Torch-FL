# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate the dense two-input concatenation route for MACA Native Host."""

import argparse
from pathlib import Path


HEADER = """\
#pragma once

#include <mcr/mc_runtime_api.h>

namespace at::native::flagos::metax_cat {
mcError_t Concatenate(void*, const void*, const void*, unsigned, unsigned,
                      unsigned, mcStream_t);
}
"""

DEVICE = """\
#include "cat.h"

#include <mcr/mc_runtime.h>
#include <cstdint>

namespace at::native::flagos::metax_cat {
namespace {

__global__ void CopyRows(uint64_t* out, const uint64_t* left,
                         const uint64_t* right, unsigned left_words,
                         unsigned right_words, unsigned rows) {
  const bool second = blockIdx.y == 1;
  const unsigned width = second ? right_words : left_words;
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < width * rows) {
    const unsigned row = index / width;
    const unsigned column = index % width + (second ? left_words : 0);
    out[static_cast<uint64_t>(row) * (left_words + right_words) + column] =
        (second ? right : left)[index];
  }
}

}  // namespace

mcError_t Concatenate(void* output, const void* left, const void* right,
                      unsigned left_words, unsigned right_words,
                      unsigned rows, mcStream_t stream) {
  constexpr unsigned threads = 256;
  const unsigned wide = left_words > right_words ? left_words : right_words;
  const dim3 grid((wide * rows + threads - 1) / threads, 2, 1);
  CopyRows<<<grid, threads, 0, stream>>>(static_cast<uint64_t*>(output),
      static_cast<const uint64_t*>(left), static_cast<const uint64_t*>(right),
      left_words, right_words, rows);
  return mcGetLastError();
}

}  // namespace at::native::flagos::metax_cat
"""

HOST = """\
#if defined(FLAGOS_METAX_NATIVE_CAT)

#include "cat.h"
#include "../../../generated/ops.h"
#include <ATen/ops/empty.h>
#include <c10/core/DeviceGuard.h>
#include <flagos.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>

namespace at::native::flagos::metax {
namespace {

at::Tensor DenseCat(const at::ITensorListRef& tensors, int64_t dim) {
  if (tensors.size() != 2) return {};
  auto it = tensors.begin();
  const auto& left = *it;
  ++it;
  const auto& right = *it;
  if (!left.device().is_privateuseone() || left.dim() != right.dim() || !left.dim() ||
      left.device() != right.device() || left.scalar_type() != right.scalar_type() ||
      !left.is_contiguous() || !right.is_contiguous() ||
      left.is_neg() || right.is_neg() || left.is_conj() || right.is_conj() ||
      !left.numel() || !right.numel()) return {};

  if (dim < 0) dim += left.dim();
  if (dim < 0 || dim >= left.dim()) return {};
  for (int64_t i = 0; i < left.dim(); ++i) {
    if (i != dim && left.size(i) != right.size(i)) return {};
  }

  const auto sizes = left.sizes();
  const int64_t rows = std::accumulate(
      sizes.begin(), sizes.begin() + dim, int64_t{1}, std::multiplies<int64_t>());
  constexpr auto max_words = std::numeric_limits<int>::max();
  if (std::max(left.nbytes(), right.nbytes()) >
      static_cast<size_t>(max_words) * sizeof(uint64_t)) return {};
  const auto left_bytes = left.nbytes() / rows;
  const auto right_bytes = right.nbytes() / rows;
  constexpr auto word = sizeof(uint64_t);
  if (left_bytes % word || right_bytes % word ||
      reinterpret_cast<uintptr_t>(left.const_data_ptr()) % word ||
      reinterpret_cast<uintptr_t>(right.const_data_ptr()) % word) return {};

  auto shape = sizes.vec();
  shape[dim] += right.size(dim);
  c10::DeviceGuard guard(left.device());
  auto result = at::empty(shape, left.options());
  const auto stream =
      reinterpret_cast<mcStream_t>(GetCurrentStreamForDevice(left.device().index()));
  const auto status = metax_cat::Concatenate(result.mutable_data_ptr(),
      left.const_data_ptr(), right.const_data_ptr(), left_bytes / word,
      right_bytes / word, rows, stream);
  TORCH_CHECK(status == mcSuccess, "MACA cat: ", static_cast<int>(status));
  return result;
}

at::Tensor CatKernel(const at::ITensorListRef& tensors, int64_t dim) {
  auto result = DenseCat(tensors, dim);
  if (result.defined()) return result;
  result = cat_dispatcher.DispatchBackend(Backend::kFlagGems, tensors, dim);
  for (const auto& input : tensors) {
    if (result.is_alias_of(input)) return result.clone();
  }
  return result;
}

}  // namespace

REGISTER_IMPL_TO_DISPATCHER(CatFn, cat_dispatcher, Backend::kMetax, CatKernel);

}  // namespace at::native::flagos::metax

#endif  // FLAGOS_METAX_NATIVE_CAT
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    directory = (
        Path(__file__).resolve().parents[2] / "csrc/aten/backends/metax/generated"
    )
    banner = (
        "// Copyright 2026 FlagOS Contributors\n"
        "// SPDX-License-Identifier: Apache-2.0\n"
        "// Generated by scripts/codegen/codegen_metax_cat.py. Do not edit.\n\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in (("cat.h", HEADER), ("cat.mc", DEVICE), ("cat_host.cc", HOST)):
        dest = directory / name
        content = banner + source
        if args.check:
            if not dest.is_file() or dest.read_text() != content:
                raise SystemExit(f"stale generated MetaX cat source: {dest}")
        else:
            dest.write_text(content)


if __name__ == "__main__":
    main()
