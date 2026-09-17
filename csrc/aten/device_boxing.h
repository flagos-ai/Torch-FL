// Copyright (c) 2026, BAAI. All rights reserved.
//
// Utilities for boxing/unboxing tensor device metadata between
// flagos (PrivateUse1) and CUDA. Since flagos and CUDA share the same
// GPU memory, temporarily changing device type in tensor metadata allows
// calling native PyTorch CUDA kernels that have device type assertions.

#pragma once

#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/SmallVector.h>

#include <flagos.h>

#include <type_traits>

namespace at::native::flagos {

// Change a tensor's device type in-place (metadata only, no data copy).
// Modifies dispatch key set, DataPtr device, and device_opt_.
// Operates directly on a TensorImpl* so callers holding a raw impl pointer
// (e.g. the boxing guards, which record impls to avoid refcount churn) don't
// need to reconstruct an owning Tensor to unbox.
inline void SetTensorImplDevice(c10::TensorImpl* impl, c10::DeviceType type) {
  auto idx = impl->device().index();
  auto new_device = c10::Device(type, idx);
  impl->_change_backend_component_keys(new_device);
  impl->unsafe_storage().unsafeGetStorageImpl()
      ->_mutable_data_ptr_no_checks().unsafe_set_device(new_device);
  // CRITICAL: Also update device_opt_ so device() returns the new device.
  // device_opt_ is protected, but we can access it via pointer offset.
  // TensorImpl layout: device_opt_ is at a known offset from the base.
  // Safer approach: use reinterpret_cast to access the field directly.
  struct TensorImplAccessor : public c10::TensorImpl {
    void set_device_opt(c10::Device d) { this->device_opt_ = d; }
  };
  static_cast<TensorImplAccessor*>(impl)->set_device_opt(new_device);
}

inline void SetTensorDevice(const at::Tensor& t, c10::DeviceType type) {
  // Undefined tensors (e.g. an unrequested grad in a *_backward output tuple,
  // like the bias grad of convolution_backward when output_mask[2]==false) have
  // no TensorImpl; touching device() would dereference null -> "tensor does not
  // have a device". Nothing to rebox, so skip.
  if (!t.defined()) return;
  SetTensorImplDevice(t.unsafeGetTensorImpl(), type);
}

inline void BoxToCuda(const at::Tensor& t) {
  SetTensorDevice(t, c10::DeviceType::CUDA);
}

inline void UnboxToFlagos(const at::Tensor& t) {
  SetTensorDevice(t, c10::DeviceType::PrivateUse1);
}

// The device half of the boxing contract.
//
// A boxed call runs through c10's CUDA machinery, so the *device* it leaves
// behind is c10::cuda's business rather than the flagos runtime's, and the two
// drift apart. c10's device guards switch with `cudaSetDevice`, and they only
// switch back when the previous device has a primary context: otherwise
// `c10::cuda::MaybeSetDevice` parks the index in its `targetDeviceIndex` TLS
// and trusts a later `SetTargetDevice()` to apply it, which nothing on the
// boxing path performs. A boxed call is entered with the caller's device
// current (flagos:0 in a single-device workload) while its tensors claim the
// accelerator index they were allocated on, so the switch really happens -- and
// it sticks.
//
// That is not harmless, because "current device" is what the flagos runtime
// keys its own decisions on: `empty_memory_format` emits a device guard only
// when the requested index differs from the one it reads back, and
// `DeviceAllocator` tags every block with the index read back at allocation
// time. Once the runtime reports N for the rest of the process, its guards stop
// binding N, allocations for an N-device tensor follow whatever context happens
// to be current instead, and the per-device pool ends up holding segments from
// two contexts. Measured on PPU (issue #313): with a 1 GiB tensor already
// allocated, `m.to(torch.bool)` on `flagos:15` leaves both
// `torch.cuda.current_device()` and `torch.flagos.current_device()` reporting
// 15 with a second CUDA context current, and a later FlagGems `sum` obtains its
// `mid` scratch block from one context while its operand lives in the other --
// Triton's launcher then rejects `mid`'s pointer
// (`cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_DEVICE_POINTER)` returning
// CUDA_ERROR_INVALID_VALUE) as "Pointer argument (at N) cannot be accessed from
// Triton (cpu tensor?)".
//
// Binding the boxed tensors' device with the runtime's own `SetDevice` for the
// lifetime of the boxed call keeps the two in step: c10's guards find that
// device already current and leave the process alone, and the caller's device
// is put back when the guard is destroyed, so a boxed op cannot silently
// relocate the process. A no-op when the boxed call's device already is the
// current one, i.e. for every flagos:0 workload.
class BoxingDeviceScope {
 public:
  BoxingDeviceScope() = default;

  // Called with the index of the first flagos tensor the boxed call touches,
  // before that tensor's metadata is rewritten. Later calls are no-ops: a boxed
  // call's tensors share one accelerator index (cross-device copies box only
  // the operand that must match the other side's device type, and the native
  // copy handles the two indices itself).
  void bind(int device_index) {
    if (bound_ || device_index < 0) {
      return;
    }
    bound_ = true;
    int current = -1;
    if (::GetDevice(&current) != Success) {
      // No readable previous device to come back to. Still bind the boxed
      // device so the call runs where its tensors live, and skip the restore
      // rather than guess at a device to return to.
      ::SetDevice(device_index);
      return;
    }
    if (current == device_index) {
      // The common case, and the only one for a single-device workload: the
      // boxed call is already where its tensors live, so leave the process
      // alone entirely -- no switch, and nothing to put back.
      return;
    }
    saved_ = current;
    ::SetDevice(device_index);
    restore_ = true;
  }

  ~BoxingDeviceScope() {
    if (restore_) {
      ::SetDevice(saved_);
    }
  }

  BoxingDeviceScope(const BoxingDeviceScope&) = delete;
  BoxingDeviceScope& operator=(const BoxingDeviceScope&) = delete;

 private:
  int saved_{-1};
  bool bound_{false};
  bool restore_{false};
};

// RAII guard: boxes flagos (PrivateUse1) tensors to CUDA, unboxes on destruction.
// CPU/CUDA inputs are left unchanged (e.g. mul/add with a CPU scalar).
//
// Boxed tensors are tracked by raw TensorImpl* rather than an owning Tensor:
// the box target always outlives the guard (callers pass named tensors, never
// temporaries), so no ownership is needed here and we avoid an atomic
// refcount inc/dec per boxed tensor on both box and unbox.
class DeviceBoxingGuard {
 public:
  template <typename... Tensors>
  explicit DeviceBoxingGuard(Tensors&&... tensors) {
    // Recording raw TensorImpl* is only safe if every boxed tensor outlives
    // the guard. Binding an rvalue (temporary) Tensor here would dangle after
    // the full expression, so reject temporaries at compile time -- callers
    // must pass named lvalue tensors.
    static_assert(
        (std::is_lvalue_reference_v<Tensors> && ...),
        "DeviceBoxingGuard must not box temporary (rvalue) tensors: "
        "the guard records raw TensorImpl* and does not extend lifetime");
    (maybe_box(tensors), ...);
  }
  ~DeviceBoxingGuard() {
    for (auto* impl : boxed_) {
      SetTensorImplDevice(impl, c10::DeviceType::PrivateUse1);
    }
  }
  DeviceBoxingGuard(const DeviceBoxingGuard&) = delete;
  DeviceBoxingGuard& operator=(const DeviceBoxingGuard&) = delete;
 private:
  void maybe_box(const at::Tensor& t) {
    if (t.defined() && t.is_privateuseone()) {
      device_.bind(t.device().index());
      auto* impl = t.unsafeGetTensorImpl();
      SetTensorImplDevice(impl, c10::DeviceType::CUDA);
      boxed_.push_back(impl);
    }
  }
  BoxingDeviceScope device_;
  c10::SmallVector<c10::TensorImpl*, 4> boxed_;
};

// Box/unbox all tensors in a TensorList (for _foreach_* ops).
inline void BoxTensorListToCuda(at::TensorList tensors) {
  for (const auto& t : tensors) {
    if (t.defined() && t.is_privateuseone()) {
      BoxToCuda(t);
    }
  }
}

inline void UnboxTensorListToFlagos(at::TensorList tensors) {
  for (const auto& t : tensors) {
    if (t.defined() && t.device().type() == c10::DeviceType::CUDA) {
      UnboxToFlagos(t);
    }
  }
}

// Materialize an ITensorListRef into a std::vector<at::Tensor>.
// The Tensor handles share the same TensorImpl as the originals, so boxing
// them (device metadata rewrite) affects the underlying tensors in place.
// The returned vector converts implicitly to at::TensorList (ArrayRef) for
// passing to PyTorch's public at:: API, which expects TensorList not IListRef.
inline std::vector<at::Tensor> MaterializeToTensorVec(
    const at::ITensorListRef& list) {
  std::vector<at::Tensor> out;
  out.reserve(list.size());
  for (const auto& t : list) {
    out.push_back(t);
  }
  return out;
}

// Drop "legacy empty" tensors (1-D with size 0) from a cat input list, matching
// ATen's native cat `should_skip` rule. maca's forked libtorch_cuda cat kernel
// takes a vectorized fast path when the non-empty tensor's numel is a multiple
// of 128 that does not honor this legacy skip, so it applies the cat dim against
// the empty tensor's 1-D rank and raises "Dimension out of range". Filtering
// here reproduces stock PyTorch semantics (e.g. transformers' KV-cache
// `torch.cat([torch.tensor([]), key_states], dim=-2)` on the first decode step).
// If every tensor is legacy-empty the list is returned unchanged so at::cat
// preserves its own empty-input behavior.
inline std::vector<at::Tensor> DropLegacyEmptyForCat(
    const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> kept;
  kept.reserve(tensors.size());
  for (const auto& t : tensors) {
    if (t.defined() && t.dim() == 1 && t.sym_size(0) == 0) {
      continue;
    }
    kept.push_back(t);
  }
  if (kept.empty()) {
    return tensors;
  }
  return kept;
}

// Box/unbox a vector of Tensors returned by non-inplace _foreach ops.
inline void UnboxTensorVecToFlagos(std::vector<at::Tensor>& tensors) {
  for (auto& t : tensors) {
    if (t.defined() && t.device().type() == c10::DeviceType::CUDA) {
      UnboxToFlagos(t);
    }
  }
}

// RAII guard for TensorList boxing (multiple lists).
// Only unboxes tensors that were actually boxed (originally PrivateUse1),
// leaving genuine CUDA tensors untouched. Boxed tensors are tracked by raw
// TensorImpl* (the list elements outlive the guard) to avoid refcount churn.
class TensorListBoxingGuard {
 public:
  TensorListBoxingGuard() = default;

  void box(at::TensorList tensors) {
    for (const auto& t : tensors) {
      if (t.defined() && t.is_privateuseone()) {
        device_.bind(t.device().index());
        auto* impl = t.unsafeGetTensorImpl();
        SetTensorImplDevice(impl, c10::DeviceType::CUDA);
        boxed_.push_back(impl);
      }
    }
  }

  // Track a tensor that was already boxed (for ITensorListRef iteration)
  void track(const at::Tensor& t) {
    if (t.defined()) {
      device_.bind(t.device().index());
      boxed_.push_back(t.unsafeGetTensorImpl());
    }
  }

  ~TensorListBoxingGuard() {
    for (auto* impl : boxed_) {
      SetTensorImplDevice(impl, c10::DeviceType::PrivateUse1);
    }
  }

  TensorListBoxingGuard(const TensorListBoxingGuard&) = delete;
  TensorListBoxingGuard& operator=(const TensorListBoxingGuard&) = delete;

 private:
  BoxingDeviceScope device_;
  c10::SmallVector<c10::TensorImpl*, 4> boxed_;
};

} // namespace at::native::flagos
