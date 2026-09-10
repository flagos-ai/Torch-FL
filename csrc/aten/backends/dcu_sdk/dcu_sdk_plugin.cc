// Copyright (c) 2026, BAAI. All rights reserved.

// Entry point of libdcu_aten_ops.so, the DCU SDK-native operator plugin.
//
// The plugin hands libtorch_fl.so a table of {schema name, function pointer,
// fn-type tag} records through the ABI in dcu_sdk_abi.h. It never touches
// torch's dispatcher and never links DTK's forked libtorch_hip.so/libc10_hip.so
// -- only the official libc10/libtorch_cpu plus the DTK SDK. That is what makes
// an install possible against an official CPU torch wheel.
//
// The kernel rows are generated (scripts/codegen_dtk.py), so adding operator
// coverage does not touch this file or the ABI version.

#include "dcu_sdk_abi.h"

#include "generated/dtk_kernels_decl.h"

#include <hip/hip_runtime.h>

#include <cstdio>

namespace {

// DTK version string for diagnostics. Reported to the loader so a coverage or
// numerics bug can be tied to the exact SDK the plugin was built against.
const char* SdkVersionString() {
  static char buf[64];
  static const bool init = []() {
    int major = 0, minor = 0, patch = 0;
#if defined(HIP_VERSION_MAJOR)
    major = HIP_VERSION_MAJOR;
    minor = HIP_VERSION_MINOR;
    patch = HIP_VERSION_PATCH;
#endif
    std::snprintf(buf, sizeof(buf), "hip %d.%d.%d", major, minor, patch);
    return true;
  }();
  (void)init;
  return buf;
}

using at::native::flagos::dcu_sdk::kDtkKernels;
using at::native::flagos::dcu_sdk::kDtkKernelCount;

} // namespace

extern "C" {

// Returns kFlagosDcuSdkOk on success; any other value is a compatibility
// rejection whose meaning the loader renders via FlagosDcuSdkStatusString.
__attribute__((visibility("default"))) FlagosDcuSdkStatus
FlagosDcuSdkPluginInit(void) {
  FlagosDcuSdkRegistration reg{};
  FLAGOS_DCU_SDK_INIT_HEADER(reg);

  reg.plugin_name = "libdcu_aten_ops.so";
  reg.sdk_version = SdkVersionString();

  // Borrowed by the bridge, not copied: the generated array has static storage
  // duration, so it outlives every use.
  reg.kernel_count = kDtkKernelCount;
  reg.kernels = kDtkKernels;

  return FlagosDcuSdkRegisterKernels(&reg);
}

// Exposed so the loader can print the exact ABI version this plugin carries when
// reporting a mismatch, without having to parse the error string.
__attribute__((visibility("default"))) unsigned FlagosDcuSdkPluginAbiVersion(
    void) {
  return FLAGOS_DCU_SDK_ABI_VERSION;
}

// Operator count this plugin offers, for the loader's coverage report. Reading
// it does not require a successful registration, so the loader can report
// "offered N, installed 0" when a rejection happens.
__attribute__((visibility("default"))) size_t FlagosDcuSdkPluginKernelCount(
    void) {
  return kDtkKernelCount;
}

} // extern "C"
