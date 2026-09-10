// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

// Registration ABI between libtorch_fl.so and the DCU SDK-native operator
// plugin (libdcu_aten_ops.so).
//
// Why an explicit ABI instead of a second TORCH_LIBRARY_IMPL: csrc/aten/register.cc
// owns the one and only TORCH_LIBRARY_IMPL(aten, PrivateUse1) for this backend, and
// a second one for the same key would either be rejected or silently shadow the
// first depending on load order. The plugin therefore does not touch torch's
// dispatcher at all -- it hands typed function pointers to the internal
// Dispatcher slots (Backend::kDcuSdk) through the single entry point below.
//
// The plugin is built separately (DTK hipcc + the DTK math/DNN libraries) and
// links only the official libc10/libtorch_cpu plus the SDK -- never DTK's forked
// libtorch_hip.so/libc10_hip.so. Because plugin and core are compiled as
// separate artifacts and can be shipped independently, every field that affects
// binary compatibility is checked at registration time and a mismatch is
// rejected BEFORE any pointer is installed. Installing first and validating
// after would leave a half-registered dispatcher behind on failure.
//
// at::Tensor / at::Scalar cross this boundary, which is only sound because both
// sides are compiled against the same official Torch headers with the same
// libstdc++ ABI. That is precisely what torch_major/minor/patch and cxx11_abi
// exist to enforce; keep them mandatory.
//
// ---------------------------------------------------------------------------
// Why the kernel list is a table and not a struct field per kernel
// ---------------------------------------------------------------------------
// ABI v1 carried one typed field per kernel (mm, mm_out, bmm, ...). That shape
// cannot express the target coverage: 1554 operators currently reach DTK's
// libtorch_hip.so on this platform, and even after PyTorch's own composite
// layers absorb the derivable ones, ~539 distinct kernels remain. A struct with
// 539 named fields would have to be edited -- and its ABI version bumped -- for
// every single operator added, and every consumer recompiled in lockstep.
//
// v2 therefore passes a counted array of {op_name, fn, type_tag} records. The
// ABI becomes independent of WHICH operators a given plugin implements: a plugin
// covering 12 operators and one covering 539 use the identical struct layout, so
// growing coverage never bumps the ABI version. The core resolves each record to
// its typed Dispatcher slot by op_name through a generated table, and `type_tag`
// makes a signature disagreement a clean rejection instead of undefined
// behavior at the first call.

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/macros/Macros.h>
// TORCH_VERSION_MAJOR/MINOR/PATCH; not pulled in transitively by the above.
#include <torch/headeronly/version.h>

#include <cstddef>
#include <cstdint>

// libstdc++ defines this; be explicit rather than relying on it so the field is
// always populated with a comparable value on both sides of the boundary.
#ifndef _GLIBCXX_USE_CXX11_ABI
#define FLAGOS_DCU_SDK_CXX11_ABI (-1)
#else
#define FLAGOS_DCU_SDK_CXX11_ABI _GLIBCXX_USE_CXX11_ABI
#endif

// Bump on ANY layout change to FlagosDcuSdkRegistration or
// FlagosDcuSdkKernel -- adding a field, reordering fields, or changing the
// meaning of one. The plugin embeds the value it was compiled against and the
// bridge requires an exact match, so an old plugin against a new core fails
// loudly instead of reading garbage pointers.
//
// Note what does NOT require a bump: adding operators. That is the entire point
// of the table layout (see the header comment).
#define FLAGOS_DCU_SDK_ABI_VERSION 2u

extern "C" {

// Status returned by FlagosDcuSdkRegisterKernels. Distinct codes (rather than a
// bool) so the Python loader can turn each failure into an actionable message
// instead of a generic "registration failed".
typedef enum FlagosDcuSdkStatus {
  kFlagosDcuSdkOk = 0,
  // registration->abi_version != FLAGOS_DCU_SDK_ABI_VERSION.
  kFlagosDcuSdkAbiVersionMismatch = 1,
  // registration->struct_size disagrees with sizeof(FlagosDcuSdkRegistration).
  kFlagosDcuSdkStructSizeMismatch = 2,
  // Plugin was built against a different Torch than the one now loaded.
  kFlagosDcuSdkTorchVersionMismatch = 3,
  // _GLIBCXX_USE_CXX11_ABI differs between plugin and core.
  kFlagosDcuSdkCxxAbiMismatch = 4,
  // A kernel record carries a null op_name or a null fn.
  kFlagosDcuSdkMissingKernel = 5,
  // registration itself is null, or kernel_count > 0 with a null array.
  kFlagosDcuSdkInvalidArgument = 6,
  // An op_name has no Backend::kDcuSdk slot in this core build. Usually means
  // plugin and core were generated from different operator sets.
  kFlagosDcuSdkUnknownOperator = 7,
  // op_name resolved, but the plugin's type_tag disagrees with the core's
  // dispatcher signature. Installing would be undefined behavior.
  kFlagosDcuSdkSignatureMismatch = 8,
  // kernel_count is 0. An empty plugin is a build accident, not a valid state.
  kFlagosDcuSdkEmptyRegistration = 9,
} FlagosDcuSdkStatus;

// One operator implementation offered by the plugin.
//
// `fn` is deliberately void*: this struct is shared by a core that knows all
// signatures and a plugin that only knows the ones it implements, so the array
// cannot be statically typed. The cast back to the real signature happens in the
// core's generated slot table, guarded by `type_tag` -- never blind.
typedef struct FlagosDcuSdkKernel {
  // Schema name exactly as it appears in native_functions.yaml and in the
  // backend conf files, e.g. "mm", "mm.out", "add.Tensor", "abs_".
  const char* op_name;
  // The kernel, to be reinterpret_cast to the dispatcher's FnPtr by the core.
  void* fn;
  // The generated dispatcher fn-type name for this operator, e.g. "MmFn",
  // "AddTensorFn". Both sides derive it from the same schema via
  // scripts/codegen_ops.py:schema_to_cpp_name, so a mismatch means the two
  // artifacts were generated against different PyTorch headers.
  const char* type_tag;
} FlagosDcuSdkKernel;

// Filled in by the plugin, validated then consumed by the bridge.
//
// Append-only: new HEADER fields go at the end and the ABI version gets bumped.
// New OPERATORS need neither -- they are rows in `kernels`.
typedef struct FlagosDcuSdkRegistration {
  // --- compatibility header (validated before anything is installed) ---
  uint32_t abi_version; // FLAGOS_DCU_SDK_ABI_VERSION at plugin build time
  size_t struct_size; // sizeof(FlagosDcuSdkRegistration) at plugin build time
  uint32_t torch_major; // TORCH_VERSION_MAJOR the plugin compiled against
  uint32_t torch_minor; // TORCH_VERSION_MINOR
  uint32_t torch_patch; // TORCH_VERSION_PATCH
  int32_t cxx11_abi; // _GLIBCXX_USE_CXX11_ABI (1 on official wheels)

  // --- provenance, diagnostics only; never used for control flow ---
  const char* plugin_name; // e.g. "libdcu_aten_ops.so"
  const char* sdk_version; // DTK version string the plugin was built on

  // --- coverage ---
  // Borrowed, not owned: the array lives in the plugin's static storage for the
  // lifetime of the process, so the bridge does not copy it. The plugin must not
  // free or mutate it after calling in.
  size_t kernel_count;
  const FlagosDcuSdkKernel* kernels;
} FlagosDcuSdkRegistration;

// Exported by libtorch_fl.so; called by the plugin loader after torch_fl._C is
// imported (so the dispatchers exist).
//
// All-or-nothing: every record is resolved and type-checked before the first
// pointer is installed, so a plugin with one bad row installs nothing rather
// than leaving a partially-routed dispatcher behind. Idempotent, which keeps a
// double dlopen from being an error.
//
// Not marked C10_EXPORT-conditional on purpose: this symbol must be visible in
// libtorch_fl.so regardless of the visibility default, otherwise the plugin's
// dlsym lookup fails at load with no useful diagnostic.
__attribute__((visibility("default"))) FlagosDcuSdkStatus
FlagosDcuSdkRegisterKernels(const FlagosDcuSdkRegistration* registration);

// Human-readable form of a status code, for the loader's error message.
__attribute__((visibility("default"))) const char* FlagosDcuSdkStatusString(
    FlagosDcuSdkStatus status);

// On a kFlagosDcuSdkUnknownOperator / kFlagosDcuSdkSignatureMismatch rejection,
// the op_name that caused it; "" otherwise. Lets the loader name the offending
// operator instead of making the user bisect a 539-row table. Points into the
// caller's own registration array, so read it before freeing that.
__attribute__((visibility("default"))) const char*
FlagosDcuSdkLastRejectedOperator(void);

// True once a registration has succeeded. The loader uses this to distinguish
// "plugin never loaded" from "plugin loaded but routing sent the op elsewhere".
__attribute__((visibility("default"))) int FlagosDcuSdkKernelsRegistered(void);

// How many kernels the successful registration installed; 0 if none succeeded.
// Reported by the loader and by the runtime check, so coverage is observable at
// runtime rather than inferred from the manifest.
__attribute__((visibility("default"))) size_t FlagosDcuSdkRegisteredCount(void);

// Number of Backend::kDcuSdk slots this core build knows about, i.e. the upper
// bound a plugin could possibly fill. Together with
// FlagosDcuSdkRegisteredCount() this gives the honest coverage ratio.
__attribute__((visibility("default"))) size_t FlagosDcuSdkKnownSlotCount(void);

} // extern "C"

// Convenience initializer for the plugin side: fills the compatibility header
// from the headers the plugin is being compiled against, so the plugin never
// hand-writes the version numbers it claims.
#define FLAGOS_DCU_SDK_INIT_HEADER(reg)                       \
  do {                                                        \
    (reg).abi_version = FLAGOS_DCU_SDK_ABI_VERSION;           \
    (reg).struct_size = sizeof(FlagosDcuSdkRegistration);     \
    (reg).torch_major = TORCH_VERSION_MAJOR;                  \
    (reg).torch_minor = TORCH_VERSION_MINOR;                  \
    (reg).torch_patch = TORCH_VERSION_PATCH;                  \
    (reg).cxx11_abi = FLAGOS_DCU_SDK_CXX11_ABI;               \
  } while (0)
