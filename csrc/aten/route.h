// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <flagos.h>

#include <cstdint>
#include <string>
#include <vector>

namespace at::native::flagos {

// Backend selector for unified op wrappers.
// Determines which physical backend impl() dispatches to.
// kUncached is a reserved sentinel, never stored in the BackendTable or
// returned by GetBackendForOp. Dispatcher caches a complete immutable entry.
// Keep it last so the real backends stay contiguous.
enum class Backend {
  kFlagGemsCpp,    // FlagGems C++ runtime (liboperators.so, conf key: flaggems_cpp)
  kFlagGems,       // FlagGems Python/Triton path (conf key: flaggems)
  // One value per vendor, spelled the way its conf key is (see AGENTS.md,
  // "Platform names"). These three used to be named after the SDK rather than
  // the vendor.
  kAscend,
  kMthreads,       // conf key: mthreads
  kMetax,
  kTsingMicro,
  kEnflame,        // conf key: enflame
  kKunlunxin,
  kIluvatar,
  kCambricon,      // conf key: cambricon
  kTileOps,
  kApple,          // Apple Metal / MPSGraph (conf key: apple)
  // D-Robotics RDK BPU. The kernels submit compiled operator artifacts from
  // the operator library named by FLAGOS_BPU_OPLIB; without it every kernel
  // takes the CPU round-trip. Conf key: drobotics.
  kDrobotics,
  kNvidia,         // Published vendor-native precision entries; never boxing.
  kHygon,          // Published HCU precision entries on PrivateUse1 storage.
  kThead,          // Published T-Head SDK precision entries.
  // No accelerated implementation on this platform. The op is deliberately not
  // registered on PrivateUse1, so it reaches the boxed cpu_fallback instead of
  // arriving here with an empty kernel slot. Recorded explicitly (rather than
  // by omission) so a vendor conf states coverage for every op and support can
  // be counted from the file.
  kNone,
  kUncached
};

struct ConditionalBackend {
  uint64_t dtype_mask;
  Backend backend;
};

struct BackendConfigEntry {
  Backend backend = Backend::kFlagGems;
  std::vector<ConditionalBackend> dtypes;
};

// The immutable config entry for an op. Unlisted keys use the FlagGems
// default. Dtype clauses apply when a matching kernel is registered.
const BackendConfigEntry& GetBackendConfigForOp(const std::string& op_name);
Backend GetBackendForOp(const std::string& op_name);
bool HasBackendForOp(const std::string& op_name);

// Set by the Python binding before the first lookup. The explicit path
// precedes FLAGOS_BACKEND_CONFIG and the native library's default.
FLAGOS_EXPORT void SetBackendConfigPath(const std::string& path);

// Conf keys accept legacy aliases and are matched case-insensitively.
bool ParseBackendName(const std::string& name, Backend* backend);
const char* BackendLogName(Backend backend);

bool LogEnabled(const char* item);
void ThrowIfForcedBackendMissing(const std::string& op_name, Backend backend);
std::string DispatchFailureMessage(Backend backend);

} // namespace at::native::flagos
