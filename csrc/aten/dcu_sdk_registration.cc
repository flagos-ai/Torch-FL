// Copyright (c) 2026, BAAI. All rights reserved.

// Core-side half of the DCU SDK-native plugin registration ABI.
//
// Lives directly under csrc/aten/ (not backends/dcu_sdk/) on purpose: everything
// under backends/dcu_sdk/ is compiled into the separate plugin artifact and is
// filtered out of libtorch_fl.so's source glob, whereas this bridge must be IN
// libtorch_fl.so -- it is the symbol the plugin resolves against.
//
// Only the header is shared between the two sides.
//
// This file performs no casts of its own. Every void* -> typed-function-pointer
// conversion happens in the generated slot table
// (generated/dtk_slot_table.cc), which is the single audited place where the
// core's knowledge of each dispatcher's real signature lives.

#include "backends/dcu_sdk/dcu_sdk_abi.h"

#include "common.h"
#include "generated/ops.h"

#include <torch/headeronly/version.h>

#include <atomic>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <string>

namespace at::native::flagos::dtk {
// Defined by the generated slot table. Resolves a schema name to its typed
// Backend::kDcuSdk dispatcher slot and installs `fn` there, but only if
// `type_tag` matches the fn-type this core compiled for that operator.
//
// Returns kFlagosDcuSdkOk, kFlagosDcuSdkUnknownOperator (no such slot in this
// build) or kFlagosDcuSdkSignatureMismatch (slot found, signature disagrees).
FlagosDcuSdkStatus InstallKernel(
    const char* op_name, void* fn, const char* type_tag);

// Upper bound on installable operators in this core build.
size_t KnownSlotCount();

// Resolution-only probe: same lookup and type check as InstallKernel but without
// installing. Lets the bridge validate the whole table before mutating any
// dispatcher, so a bad row leaves nothing half-registered.
FlagosDcuSdkStatus CheckKernel(
    const char* op_name, void* fn, const char* type_tag);
} // namespace at::native::flagos::dtk

namespace {

// Set only after a full successful registration, so a rejected plugin cannot
// make the runtime believe SDK kernels are available.
std::atomic<bool> g_registered{false};

// Number of kernels the successful registration installed.
std::atomic<size_t> g_registered_count{0};

// op_name from the most recent per-operator rejection. Points into the caller's
// registration array, which the ABI documents as living for the process
// lifetime, so holding the pointer is safe.
std::atomic<const char*> g_last_rejected{nullptr};

} // namespace

extern "C" {

const char* FlagosDcuSdkStatusString(FlagosDcuSdkStatus status) {
  switch (status) {
    case kFlagosDcuSdkOk:
      return "ok";
    case kFlagosDcuSdkAbiVersionMismatch:
      return "plugin was built against a different FLAGOS_DCU_SDK_ABI_VERSION "
             "than this torch_fl build; rebuild libdcu_aten_ops.so";
    case kFlagosDcuSdkStructSizeMismatch:
      return "FlagosDcuSdkRegistration size differs between plugin and "
             "torch_fl; rebuild libdcu_aten_ops.so against this torch_fl";
    case kFlagosDcuSdkTorchVersionMismatch:
      return "plugin was built against a different PyTorch version than the one "
             "loaded; rebuild libdcu_aten_ops.so against the active torch";
    case kFlagosDcuSdkCxxAbiMismatch:
      return "plugin and torch_fl disagree on _GLIBCXX_USE_CXX11_ABI; rebuild "
             "libdcu_aten_ops.so with the same libstdc++ ABI as torch";
    case kFlagosDcuSdkMissingKernel:
      return "a kernel record carries a null op_name or a null function pointer";
    case kFlagosDcuSdkInvalidArgument:
      return "null registration struct, or a null kernel array with a nonzero "
             "kernel_count";
    case kFlagosDcuSdkUnknownOperator:
      return "plugin offers an operator that has no dispatcher slot in this "
             "torch_fl build; plugin and core were generated from different "
             "operator sets -- rerun scripts/codegen_dtk.py and rebuild both";
    case kFlagosDcuSdkSignatureMismatch:
      return "plugin and torch_fl disagree on an operator's signature; rerun "
             "scripts/codegen_dtk.py and rebuild both";
    case kFlagosDcuSdkEmptyRegistration:
      return "plugin registered zero kernels; this is a build accident, not a "
             "valid configuration";
  }
  return "unknown status";
}

FlagosDcuSdkStatus FlagosDcuSdkRegisterKernels(
    const FlagosDcuSdkRegistration* reg) {
  if (reg == nullptr) {
    return kFlagosDcuSdkInvalidArgument;
  }

  // --- compatibility header, checked before anything else is examined -------

  if (reg->abi_version != FLAGOS_DCU_SDK_ABI_VERSION) {
    return kFlagosDcuSdkAbiVersionMismatch;
  }
  // Checked after abi_version, since struct_size is only meaningful once we know
  // both sides agree on which struct this is.
  if (reg->struct_size != sizeof(FlagosDcuSdkRegistration)) {
    return kFlagosDcuSdkStructSizeMismatch;
  }
  // Exact match, patch included: at::Tensor and at::Scalar cross this boundary
  // by value/reference, and torch offers no stability guarantee for their layout
  // even across patch releases.
  if (reg->torch_major != TORCH_VERSION_MAJOR ||
      reg->torch_minor != TORCH_VERSION_MINOR ||
      reg->torch_patch != TORCH_VERSION_PATCH) {
    return kFlagosDcuSdkTorchVersionMismatch;
  }
  if (reg->cxx11_abi != FLAGOS_DCU_SDK_CXX11_ABI) {
    return kFlagosDcuSdkCxxAbiMismatch;
  }

  // --- kernel table --------------------------------------------------------

  if (reg->kernel_count == 0) {
    return kFlagosDcuSdkEmptyRegistration;
  }
  if (reg->kernels == nullptr) {
    return kFlagosDcuSdkInvalidArgument;
  }

  g_last_rejected.store(nullptr, std::memory_order_relaxed);

  // Pass 1: validate every row without touching a dispatcher. With 500+ rows in
  // a mature plugin, installing as we go would leave a partially-routed
  // dispatcher behind on the first bad row -- some operators pointing into a
  // plugin we then decided to reject. Validate-then-install keeps registration
  // all-or-nothing.
  for (size_t i = 0; i < reg->kernel_count; ++i) {
    const FlagosDcuSdkKernel& k = reg->kernels[i];
    if (k.op_name == nullptr || k.fn == nullptr) {
      g_last_rejected.store(k.op_name, std::memory_order_relaxed);
      return kFlagosDcuSdkMissingKernel;
    }
    FlagosDcuSdkStatus st =
        at::native::flagos::dtk::CheckKernel(k.op_name, k.fn, k.type_tag);
    if (st != kFlagosDcuSdkOk) {
      g_last_rejected.store(k.op_name, std::memory_order_relaxed);
      return st;
    }
  }

  // Pass 2: install. Every row already resolved and type-checked above, so this
  // cannot fail partway.
  size_t installed = 0;
  for (size_t i = 0; i < reg->kernel_count; ++i) {
    const FlagosDcuSdkKernel& k = reg->kernels[i];
    FlagosDcuSdkStatus st =
        at::native::flagos::dtk::InstallKernel(k.op_name, k.fn, k.type_tag);
    if (st != kFlagosDcuSdkOk) {
      // Unreachable unless the slot table is nondeterministic between the two
      // passes. Report rather than silently miscount.
      g_last_rejected.store(k.op_name, std::memory_order_relaxed);
      return st;
    }
    ++installed;
  }

  g_registered_count.store(installed, std::memory_order_release);
  g_registered.store(true, std::memory_order_release);

  if (const char* v = std::getenv("FLAGOS_LOG_DISPATCH");
      v && std::string(v) == "1") {
    fprintf(
        stderr,
        "[flagos dcu_sdk] registered %zu/%zu kernels from %s (sdk %s)\n",
        installed,
        at::native::flagos::dtk::KnownSlotCount(),
        reg->plugin_name ? reg->plugin_name : "<unnamed>",
        reg->sdk_version ? reg->sdk_version : "<unknown>");
  }

  return kFlagosDcuSdkOk;
}

const char* FlagosDcuSdkLastRejectedOperator(void) {
  const char* op = g_last_rejected.load(std::memory_order_relaxed);
  return op ? op : "";
}

int FlagosDcuSdkKernelsRegistered(void) {
  return g_registered.load(std::memory_order_acquire) ? 1 : 0;
}

size_t FlagosDcuSdkRegisteredCount(void) {
  return g_registered_count.load(std::memory_order_acquire);
}

size_t FlagosDcuSdkKnownSlotCount(void) {
  return at::native::flagos::dtk::KnownSlotCount();
}

} // extern "C"
