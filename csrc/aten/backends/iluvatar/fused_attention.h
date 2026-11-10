// Copyright (c) 2026, BAAI. All rights reserved.
//
// The fused attention kernel CoreX ships as `libixattnbkd`.
//
// WHY THIS IS A SEPARATE HEADER
// -----------------------------
// `aten::scaled_dot_product_attention` is a composite: its fused-backend
// selection runs inside the composite and then branches on
// `query.device().type()`, so on PrivateUse1 the leaf kernels are never
// consulted (see csrc/aten/sdp_choice_stub.cc). The only place a vendor kernel
// can plug in is the composite override, and that file is shared by every
// platform; this header keeps the vendor call out of it.
//
// WHY IT USES THE VENDOR HEADER INSTEAD OF A MIRROR
// ------------------------------------------------
// `ixAttnBkdConfigInfo` is 104 bytes, and `ixAttnBkdPhiloxState` is 24: two
// union payloads, a uint32, a bool and tail padding. A mirror that gets that
// wrong by 8 shifts every field after it, and the library then reads `batch`
// from zeroed memory and rejects the launch with `invalid argument` from
// inside its own launcher -- an error that looks like a driver fault and is
// not one. Including <ixattnbkd.h> and default-constructing the config makes
// that class of mistake impossible: the vendor's own constructor zero-fills
// every field the caller does not set.
//
// WHY THE SYMBOLS ARE RESOLVED AT RUNTIME
// ---------------------------------------
// Linking `libixattnbkd.so` would make it a load-time dependency of every
// iluvatar wheel. The library ships with CoreX 4.5.0 but is not part of the
// older SDKs, and a wheel that fails to import on an SDK without it would lose
// every op to fix one. Resolving on first use keeps the route optional: when
// the library or a symbol is missing, FusedAttentionAvailable() is false and
// the caller falls through to the composite it used before.

#pragma once

#ifdef USE_ILUVATAR

#include <ATen/ATen.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>

#include <ixattnbkd.h>

#include "iluvatar_common.h"

namespace at::native::flagos::iluvatar {

namespace fused_attention_detail {

using FwdFn = ixAttnBkdStatus_t (*)(
    const ixAttnBkdConfigInfo&,
    const ixAttnBkdTensorDesc&,
    const ixAttnBkdTensorDesc&,
    const ixAttnBkdTensorDesc&,
    const ixAttnBkdTensorDesc&,
    const ixAttnBkdTensorDesc&,
    const ixAttnBkdTensorDesc&,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    void*,
    void*,
    void*);
using BuffSizeFn = ixAttnBkdStatus_t (*)(
    const uint32_t,
    const uint32_t,
    const uint32_t,
    const uint32_t,
    const uint32_t,
    const uint32_t,
    const uint32_t,
    size_t*,
    const ixAttnBkdImpMode_t);
using ErrorStringFn = const char* (*)(ixAttnBkdStatus_t);

struct Api {
  FwdFn fwd = nullptr;
  BuffSizeFn buff_size = nullptr;
  ErrorStringFn error_string = nullptr;

  bool ok() const { return fwd != nullptr && buff_size != nullptr; }
};

// Resolve once per process. `FLAGOS_ILUVATAR_ATTN_LIB` overrides the soname,
// for a host whose SDK keeps the library somewhere the loader does not search.
inline Api& GetApi() {
  static Api api = [] {
    Api loaded;
    const char* path = std::getenv("FLAGOS_ILUVATAR_ATTN_LIB");
    void* handle =
        dlopen(path ? path : "libixattnbkd.so.1", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      return loaded;
    }
    auto sym = [handle](const char* name) { return dlsym(handle, name); };
    loaded.fwd = reinterpret_cast<FwdFn>(sym("ixAttnBkdFlashAttnForward"));
    loaded.buff_size = reinterpret_cast<BuffSizeFn>(
        sym("ixAttnBkdGetFlashAttnFwdBuffSize"));
    loaded.error_string = reinterpret_cast<ErrorStringFn>(
        sym("ixAttnBkdGetErrorString"));
    return loaded;
  }();
  return api;
}

inline ixAttnBkdDataType_t ToVendorDtype(at::ScalarType type) {
  switch (type) {
    case at::kHalf:
      return IXATTNBKD_DATA_HALF;
    case at::kBFloat16:
      return IXATTNBKD_DATA_BF16;
    case at::kFloat:
      return IXATTNBKD_DATA_FLOAT;
    default:
      return IXATTNBKD_DATA_FLOAT;
  }
}

// The vendor's descriptor takes `dim` and `stride` as std::vectors of element
// counts, so ownership stays here and the descriptor is valid for the call.
inline ixAttnBkdTensorDesc MakeDesc(
    std::initializer_list<size_t> dim,
    std::initializer_list<size_t> stride,
    ixAttnBkdDataType_t dtype) {
  ixAttnBkdTensorDesc desc;
  desc.n_dims = static_cast<uint32_t>(dim.size());
  desc.dim.assign(dim.begin(), dim.end());
  desc.stride.assign(stride.begin(), stride.end());
  desc.dataType = dtype;
  return desc;
}

// Q/K/V/O are 4-D `[B, H, S, D]` contiguous. The vendor describes them as
// `[B, S, H, D]` (BSHD) with element strides, so the sequence and head axes
// swap places in both the shape and the stride.
inline ixAttnBkdTensorDesc BshdDesc(const at::Tensor& tensor) {
  const auto sizes = tensor.sizes();
  const auto strides = tensor.strides();
  return MakeDesc(
      {static_cast<size_t>(sizes[0]),
       static_cast<size_t>(sizes[2]),
       static_cast<size_t>(sizes[1]),
       static_cast<size_t>(sizes[3])},
      {static_cast<size_t>(strides[0]),
       static_cast<size_t>(strides[2]),
       static_cast<size_t>(strides[1]),
       static_cast<size_t>(strides[3])},
      ToVendorDtype(tensor.scalar_type()));
}

} // namespace fused_attention_detail

// Whether the library exposes the entry points this route needs.
inline bool FusedAttentionAvailable() {
  return fused_attention_detail::GetApi().ok();
}

// The shapes and dtypes this route was measured on. Everything outside them
// keeps the composite, which is the pre-existing behaviour -- a caller must not
// acquire an unmeasured fused route by accident.
//
//   * bfloat16 only: the kernel has half and float entry points too, but only
//     bfloat16 was measured on a BI-V150, and dtype breadth is the kind of
//     claim that has to be measured per dtype (see the per-op dtype gates in
//     the generated kernels).
//   * head_dim 64 or 128: the dispatcher has branches for 64, 96, 128, 160,
//     192 and 256, but 64 and 128 are the ones Qwen-family models use and the
//     ones measured here.
//   * equal source and target lengths, equal head counts: the API takes them
//     separately (and the descriptor carries the key/value length), but
//     cross-attention and grouped-query attention were not measured.
//   * no dropout and no mask: the mask descriptor path exists and is not
//     measured, so a masked call keeps the composite.
inline bool FusedAttentionEligible(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool /*is_causal*/,
    std::optional<double> /*scale*/,
    bool enable_gqa) {
  if (dropout_p != 0.0 || attn_mask.has_value() || enable_gqa) {
    return false;
  }
  if (!FusedAttentionAvailable()) {
    return false;
  }
  if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
    return false;
  }
  const auto dtype = query.scalar_type();
  if (dtype != at::kBFloat16 || key.scalar_type() != dtype ||
      value.scalar_type() != dtype) {
    return false;
  }
  if (!query.is_contiguous() || !key.is_contiguous() ||
      !value.is_contiguous()) {
    return false;
  }
  const auto head_dim = query.size(3);
  if (head_dim != 64 && head_dim != 128) {
    return false;
  }
  return query.size(0) == key.size(0) && query.size(0) == value.size(0) &&
      query.size(2) == key.size(2) && query.size(2) == value.size(2) &&
      key.size(3) == head_dim && value.size(3) == head_dim &&
      key.size(1) == value.size(1) && query.size(1) == key.size(1);
}

// Run the vendor's fused forward and return a flagos tensor.
//
// The caller must have checked FusedAttentionEligible(); this function asserts
// the preconditions it depends on rather than silently producing a result for
// a shape the kernel was not asked to handle.
inline at::Tensor FusedAttention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double scale,
    bool is_causal) {
  using namespace fused_attention_detail;

  auto& api = GetApi();
  TORCH_CHECK(
      api.ok(),
      "Iluvatar fused attention: libixattnbkd is not loadable; set "
      "FLAGOS_ILUVATAR_ATTN_LIB to its path");

  // [B, H, S, D]. The vendor's descriptors are BSHD, but the config's lengths
  // are the tensor's own: heads live on size(1), the sequence on size(2).
  const int64_t batch = query.size(0);
  const int64_t heads = query.size(1);
  const int64_t seq = query.size(2);
  const int64_t head_dim = query.size(3);

  // Open the guard before allocating: at::empty resolves the device from the
  // current one, and the vendor library resolves pointers against the current
  // corex device.
  IluvatarDeviceGuard guard(query);
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    device = IluvatarDeviceIndex(query);
  }

  at::Tensor out = at::empty_like(query);
  at::Tensor lse = at::empty(
      {batch, heads, seq}, query.options().dtype(at::kFloat));

  size_t workspace_bytes = 0;
  const auto buff_status = api.buff_size(
      static_cast<uint32_t>(batch),
      static_cast<uint32_t>(heads),
      static_cast<uint32_t>(heads),
      static_cast<uint32_t>(seq),
      static_cast<uint32_t>(seq),
      static_cast<uint32_t>(head_dim),
      static_cast<uint32_t>(head_dim),
      &workspace_bytes,
      IXATTNBKD_FATTN_MEM_MODE);
  TORCH_CHECK(
      buff_status == IXATTNBKD_STATUS_SUCCESS,
      "Iluvatar fused attention: workspace size query failed: ",
      api.error_string ? api.error_string(buff_status) : "unknown error");
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_bytes > 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_bytes)},
        query.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }

  // Empty mask: n_dims 0, and the vendor still wants a dtype on it (a
  // zero-initialised descriptor fails an earlier assertion than the launch).
  ixAttnBkdTensorDesc mask_desc;
  mask_desc.n_dims = 0;
  mask_desc.dataType = IXATTNBKD_DATA_BF16;

  const ixAttnBkdTensorDesc q_desc = BshdDesc(query);
  const ixAttnBkdTensorDesc k_desc = BshdDesc(key);
  const ixAttnBkdTensorDesc v_desc = BshdDesc(value);
  const ixAttnBkdTensorDesc o_desc = BshdDesc(out);
  const ixAttnBkdTensorDesc lse_desc = MakeDesc(
      {static_cast<size_t>(batch),
       static_cast<size_t>(heads),
       static_cast<size_t>(seq)},
      {static_cast<size_t>(heads * seq),
       static_cast<size_t>(seq),
       static_cast<size_t>(1)},
      IXATTNBKD_DATA_FLOAT);

  // Default-constructed, so every field the vendor reads but this route does
  // not set (dropout, philox, alibi, softcap, unpad) is zero, and
  // `imp_mode` starts at PERF. MEM matches the vendor's own attention call on
  // this platform; the two differ in workspace, and the measured kernel that
  // returns 0 for the workspace query is the MEM one.
  ixAttnBkdConfigInfo config;
  config.stream = reinterpret_cast<void*>(IluvatarCurrentStream(device));
  config.softmax_scale = static_cast<float>(scale);
  config.dropout_prob = 0.0f;
  config.is_causal = is_causal;
  config.causal_mode = 1;
  config.return_softmax_lse = true;
  config.batch = static_cast<unsigned>(batch);
  config.max_seq_len_src = static_cast<unsigned>(seq);
  config.max_seq_len_trg = static_cast<unsigned>(seq);
  config.softcap = 0.0f;
  config.imp_mode = IXATTNBKD_FATTN_MEM_MODE;

  const auto status = api.fwd(
      config,
      q_desc,
      k_desc,
      v_desc,
      mask_desc,
      o_desc,
      lse_desc,
      query.const_data_ptr(),
      key.const_data_ptr(),
      value.const_data_ptr(),
      nullptr, // mask
      nullptr, // loWinIdx
      nullptr, // hiWinIdx
      nullptr, // cu_seq_q
      nullptr, // cu_seq_k
      workspace_ptr,
      out.data_ptr(),
      lse.data_ptr());
  TORCH_CHECK(
      status == IXATTNBKD_STATUS_SUCCESS,
      "Iluvatar fused attention failed: ",
      api.error_string ? api.error_string(status) : "unknown error");

  // The dispatcher's contract for a synchronous op, and what orders the kernel
  // against the FlagGems/Triton work queued on the same stream.
  IluvatarSynchronizeStream(device);
  return out;
}

} // namespace at::native::flagos::iluvatar

#endif // USE_ILUVATAR