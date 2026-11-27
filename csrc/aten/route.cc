// Copyright (c) 2026, BAAI. All rights reserved.

#include "route.h"

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <flagos_env.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace at::native::flagos {
namespace {

std::string& BackendConfigPathOverride() {
  static std::string path;
  return path;
}

std::string DefaultConfigPath() {
#ifndef _WIN32
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(GetBackendForOp), &info) && info.dli_fname) {
    std::string lib_path(info.dli_fname);
    auto pos = lib_path.rfind('/');
    if (pos != std::string::npos) {
      std::string dir = lib_path.substr(0, pos);

      // Try platform-specific config first (e.g. backends_tsingmicro.conf)
      const char* platform = nullptr;
#if defined(USE_TSINGMICRO)
      platform = "tsingmicro";
#elif defined(USE_GCU)
      platform = "enflame";
#elif defined(USE_KUNLUNXIN)
      platform = "kunlunxin";
#elif defined(USE_ASCEND)
      platform = "ascend";
#elif defined(USE_MUSA)
      platform = "mthreads";
#elif defined(USE_BPU)
      // The vendor name, matching backends_drobotics.conf: AGENTS.md spells a
      // conf after the vendor, and the BPU is D-Robotics' part, not a vendor.
      platform = "drobotics";
#elif defined(USE_ILUVATAR)
      platform = "iluvatar";
#elif defined(USE_MPS)
      platform = "apple";
#elif defined(USE_CAMBRICON)
      platform = "cambricon";
#endif
      if (platform) {
        // dir is <prefix>/torch_fl/lib, configs are at <prefix>/torch_fl/configs/
        std::string candidate =
            dir + "/../configs/backends_" + platform + ".conf";
        std::ifstream test(candidate);
        if (test.is_open()) return candidate;
      }

      // No platform-specific conf (metax, or a CUDA-ABI vendor whose conf was
      // not found): the nvidia conf is the base list those builds inherit.
      // Python normally sets the path via SetBackendConfigPath(); this is only
      // the last-resort default.
      //
      // A conf is named after the build's vendor (see cmake/flagos_platforms.json);
      // only platforms that compile a vendor op library have one, which is why
      // this arm names the ones with a USE_* macro.
      std::string candidate = dir + "/../configs/backends_nvidia.conf";
      std::ifstream test(candidate);
      if (test.is_open()) return candidate;
      // Try: <dir>/configs/backends_nvidia.conf
      candidate = dir + "/configs/backends_nvidia.conf";
      test.open(candidate);
      if (test.is_open()) return candidate;
    }
  }
#endif
  // Fallback to build-time path
  return FLAGOS_SOURCE_ROOT "/torch_fl/configs/backends_nvidia.conf";
}

std::string TrimStr(std::string s) {
  size_t l = s.find_first_not_of(" \t\r\n");
  size_t r = s.find_last_not_of(" \t\r\n");
  return (l == std::string::npos) ? "" : s.substr(l, r - l + 1);
}

// Backend names are matched case-insensitively so a per-op override can be
// spelled the way it reads in docs (FLAGOS_OP_mm=FLAGGEMS) as well as the way
// the conf files write it (mm = flaggems).
std::string LowerStr(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
  }
  return s;
}

struct BackendInfo {
  Backend backend;
  const char* name;
  const char* kernel_set;
  const char* legacy_log_name = nullptr;
};

// Indexed by Backend. A legacy log spelling is also an accepted conf key.
constexpr BackendInfo kBackends[] = {
    {Backend::kFlagGemsCpp, "flaggems_cpp", "flaggems_cpp", "flagos"},
    {Backend::kFlagGems, "flaggems", "flaggems", "flagos_python"},
    {Backend::kAscend, "ascend", "vendor"},
    {Backend::kMthreads, "mthreads", "vendor", "musa"},
    {Backend::kMetax, "metax", "vendor"},
    {Backend::kTsingMicro, "tsingmicro", "vendor"},
    {Backend::kEnflame, "enflame", "vendor", "gcu"},
    {Backend::kKunlunxin, "kunlunxin", "vendor"},
    {Backend::kIluvatar, "iluvatar", "vendor"},
    {Backend::kCambricon, "cambricon", "vendor"},
    {Backend::kTileOps, "tileops", "tileops"},
    {Backend::kApple, "apple", "vendor", "mps"},
    {Backend::kDrobotics, "drobotics", "vendor"},
    {Backend::kNvidia, "nvidia", "vendor"},
    {Backend::kHygon, "hygon", "vendor"},
    {Backend::kThead, "thead", "vendor"},
    {Backend::kNone, "none", nullptr},
};
static_assert([] {
  for (size_t i = 0; i < std::size(kBackends); ++i) {
    if (static_cast<size_t>(kBackends[i].backend) != i) return false;
  }
  return std::size(kBackends) == static_cast<size_t>(Backend::kUncached);
}(), "Backend metadata must follow enum order");

struct BackendAlias {
  const char* name;
  Backend backend;
};

constexpr BackendAlias kAliases[] = {
    {"bpu", Backend::kDrobotics},
    {"flaggems_python", Backend::kFlagGems},
};

const BackendInfo* FindBackend(Backend backend) {
  const auto index = static_cast<size_t>(backend);
  return index < std::size(kBackends) ? &kBackends[index] : nullptr;
}

const char* BackendName(Backend backend) {
  const auto* info = FindBackend(backend);
  return info ? info->name : "unknown";
}

bool ParseConfiguredBackend(const std::string& text, BackendConfigEntry* result) {
  const auto first = text.find(';');
  if (!ParseBackendName(TrimStr(text.substr(0, first)), &result->backend)) return false;
  result->dtypes.clear();
  uint64_t seen = 0;
  auto pos = first;
  while (pos != std::string::npos) {
    const auto end = text.find(';', pos + 1);
    const auto clause = TrimStr(text.substr(pos + 1, end - pos - 1));
    const auto colon = clause.find(':');
    TORCH_CHECK(colon != std::string::npos && clause.find(':', colon + 1) == std::string::npos,
                "flagos: invalid dtype route '", clause, "'");
    Backend backend;
    TORCH_CHECK(ParseBackendName(TrimStr(clause.substr(colon + 1)), &backend),
                "flagos: unknown dtype-route backend in '", clause, "'");
    TORCH_CHECK(backend != Backend::kNone,
                "flagos: a dtype route requires a kernel backend, not none");
    uint64_t mask = 0;
    const auto type_names = TrimStr(clause.substr(0, colon));
    TORCH_CHECK(!type_names.empty() && type_names.back() != ',',
                "flagos: empty route dtype in '", clause, "'");
    std::istringstream names(type_names);
    std::string name;
    while (std::getline(names, name, ',')) {
      name = LowerStr(TrimStr(name));
      uint64_t bit = 0;
      for (unsigned i = 0; i < static_cast<unsigned>(c10::ScalarType::NumOptions) && i < 64; ++i) {
        if (i != static_cast<unsigned>(c10::ScalarType::Undefined) &&
            LowerStr(c10::toString(static_cast<c10::ScalarType>(i))) == name) {
          bit = uint64_t{1} << i;
          break;
        }
      }
      TORCH_CHECK(bit && !(mask & bit) && !(seen & bit),
                  "flagos: unknown or repeated route dtype '", name, "'");
      mask |= bit;
    }
    TORCH_CHECK(mask, "flagos: empty route dtype in '", clause, "'");
    seen |= mask;
    result->dtypes.push_back({mask, backend});
    pos = end;
  }
  return true;
}

// Includes are relative to their parent conf and are read in place. Later
// assignments win. A trailing backend annotation records an alternative for
// FLAGOS_FORCE_BACKEND.
void ParseConfigInto(const std::string& path,
                     std::unordered_map<std::string, BackendConfigEntry>& table,
                     std::unordered_map<std::string, Backend>& alt,
                     int depth = 0) {
  if (depth > 8) {
    fprintf(stderr, "[flagos] include nesting too deep at %s, skipping\n",
            path.c_str());
    return;
  }

  std::ifstream f(path);
  if (!f.is_open()) {
    fprintf(stderr, "[flagos] cannot open backend config %s\n", path.c_str());
    return;
  }

  fprintf(stderr, "[flagos] loading backend config from %s\n", path.c_str());

  std::string line;
  while (std::getline(f, line)) {
    // strip comments, keeping the text for the `# <backend>` annotation below
    std::string comment_text;
    auto comment = line.find('#');
    if (comment != std::string::npos) {
      comment_text = TrimStr(line.substr(comment + 1));
      line = line.substr(0, comment);
    }

    auto eq = line.find('=');
    if (eq == std::string::npos) {
      // `include <path>` -- the only non-assignment directive. Anything else
      // without an '=' stays silently ignored, as before.
      std::string t = TrimStr(line);
      if (t.rfind("include", 0) == 0 && t.size() > 7 &&
          (t[7] == ' ' || t[7] == '\t')) {
        std::string inc = TrimStr(t.substr(7));
        if (inc.empty()) continue;
        if (inc[0] != '/') {
          auto slash = path.rfind('/');
          if (slash != std::string::npos) {
            inc = path.substr(0, slash + 1) + inc;
          }
        }
        ParseConfigInto(inc, table, alt, depth + 1);
      }
      continue;
    }

    std::string op = TrimStr(line.substr(0, eq));
    std::string val = TrimStr(line.substr(eq + 1));

    if (op.empty() || val.empty()) continue;

    BackendConfigEntry parsed;
    if (ParseConfiguredBackend(val, &parsed)) {
      table[op] = parsed;
    } else {
      fprintf(stderr, "[flagos] unknown backend '%s' for op '%s', using flaggems\n", val.c_str(), op.c_str());
      table[op] = {};
    }

    Backend annotated;
    if (!comment_text.empty() && ParseBackendName(comment_text, &annotated)) {
      alt[op] = annotated;
    } else {
      alt.erase(op);
    }
  }
}

const std::string& ForcedBackendMode() {
  // Resolved once: the routing table is built once and the environment cannot
  // change under a running process, and the dispatch-miss path must not pay for
  // a getenv per op.
  static const std::string mode = [] {
    std::string out;
    flagos_env::EnvChoice("FLAGOS_FORCE_BACKEND", {"flaggems", "vendor", "tileops"},
                          "", &out);
    return out;
  }();
  return mode;
}

bool IsFlagGems(Backend b) {
  return b == Backend::kFlagGemsCpp || b == Backend::kFlagGems;
}

// Re-pin an op only when its configured backend or annotation supplies
// that family. TileOPs only moves annotated ops.
void ApplyForcedBackend(std::unordered_map<std::string, BackendConfigEntry>& table,
                        const std::unordered_map<std::string, Backend>& alt) {
  const std::string& mode = ForcedBackendMode();
  if (mode.empty()) return;

  const bool to_tileops = mode == "tileops";
  const bool to_flaggems = mode == "flaggems";

  std::vector<std::string> unsupported;
  size_t moved = 0;
  for (auto& [op, route] : table) {
    auto& backend = route.backend;
    auto it = alt.find(op);
    Backend annotated = it != alt.end() ? it->second : Backend::kNone;

    if (to_tileops) {
      if (annotated != Backend::kTileOps || backend == Backend::kTileOps) continue;
      backend = Backend::kTileOps;
      route.dtypes.clear();
      ++moved;
      continue;
    }

    // The op's candidates are its routed backend and its annotation.
    Backend target = Backend::kNone;
    if (to_flaggems) {
      if (IsFlagGems(backend)) target = backend;
      else if (IsFlagGems(annotated)) target = annotated;
    } else {
      if (!IsFlagGems(backend) && backend != Backend::kNone) target = backend;
      else if (!IsFlagGems(annotated) && annotated != Backend::kNone) target = annotated;
    }

    if (target == Backend::kNone) {
      if (!to_flaggems && std::any_of(route.dtypes.begin(), route.dtypes.end(),
                                    [](const auto& candidate) {
          return !IsFlagGems(candidate.backend) && candidate.backend != Backend::kNone;
        })) continue;  // Keep measured dtype limits instead of widening them.
      unsupported.push_back(op);
      continue;
    }
    if (to_flaggems) route.dtypes.clear();
    if (target != backend) {
      backend = target;
      route.dtypes.clear();
      ++moved;
    }
  }

  if (to_tileops) {
    fprintf(stderr, "[flagos] FLAGOS_FORCE_BACKEND=tileops: %zu ops repinned\n",
            moved);
    return;
  }

  const char* which = to_flaggems ? "flaggems" : "vendor";
  fprintf(stderr,
          "[flagos] FLAGOS_FORCE_BACKEND=%s: %zu ops repinned, %zu left as "
          "configured\n",
          which, moved, unsupported.size());
  if (!unsupported.empty()) {
    std::sort(unsupported.begin(), unsupported.end());
    fprintf(stderr, "[flagos] FLAGOS_FORCE_BACKEND=%s: no implementation for",
            which);
    for (size_t i = 0; i < unsupported.size() && i < 12; ++i) {
      fprintf(stderr, " %s", unsupported[i].c_str());
    }
    if (unsupported.size() > 12) {
      fprintf(stderr, " ... (+%zu more)", unsupported.size() - 12);
    }
    fprintf(stderr, "\n");
  }
}

std::string ResolveBackendConfigPath() {
  std::string path = BackendConfigPathOverride();
  if (path.empty()) path = flagos_env::EnvValue("FLAGOS_BACKEND_CONFIG");
  if (path.empty()) path = DefaultConfigPath();
  return path;
}

std::unordered_map<std::string, BackendConfigEntry> LoadBackendConfig() {
  std::unordered_map<std::string, BackendConfigEntry> table;

  std::string path = ResolveBackendConfigPath();

  // The top-level conf must be readable. Missing legacy includes still
  // warn and allow parsing to continue.
  {
    std::ifstream probe(path);
    TORCH_CHECK(probe.good(), "flagos: cannot open backend config '", path,
                "'. Unset FLAGOS_BACKEND_CONFIG to let torch_fl select the conf "
                "for this build, or point it at a file that exists.");
  }

  std::unordered_map<std::string, Backend> alt;
  ParseConfigInto(path, table, alt);
  ApplyForcedBackend(table, alt);

  // Per-op env var overrides, e.g. FLAGOS_OP_mm=flaggems or
  // FLAGOS_OP_mm__out=flaggems.
  // Dots in op names are replaced with double underscores to avoid ambiguity
  // with ops that already contain underscores (e.g. mm_out vs mm.out).
  for (auto& [op, _] : table) {
    std::string key = "FLAGOS_OP_";
    for (char c : op) {
      if (c == '.') key += "__";
      else key += c;
    }
    std::string override_val = flagos_env::EnvValue(key.c_str());
    if (override_val.empty()) continue;
    BackendConfigEntry parsed;
    if (ParseConfiguredBackend(override_val, &parsed)) {
      table[op] = parsed;
      fprintf(stderr, "[flagos] env override: %s -> %s\n", op.c_str(),
              BackendName(parsed.backend));
    } else {
      fprintf(stderr, "[flagos] env override: unknown backend '%s' for op '%s', ignored\n",
              override_val.c_str(), op.c_str());
    }
  }

  return table;
}

const std::unordered_map<std::string, BackendConfigEntry>& BackendTable() {
  // Dispatchers retain entry pointers. Keep the immutable table alive through
  // native-library teardown rather than depending on static destructor order.
  static const auto* table =
      new std::unordered_map<std::string, BackendConfigEntry>(LoadBackendConfig());
  return *table;
}

} // namespace

bool ParseBackendName(const std::string& raw, Backend* backend) {
  const auto name = LowerStr(raw);
  TORCH_CHECK(name != "boxing" && name != "cuda",
              "flagos: the '", raw, "' boxing route has been removed. "
              "Use the platform's default backend configuration or select a registered backend.");
  for (const auto& info : kBackends) {
    if (name == info.name ||
        (info.legacy_log_name && name == info.legacy_log_name)) {
      *backend = info.backend;
      return true;
    }
  }
  for (const auto& alias : kAliases) {
    if (name == alias.name) {
      *backend = alias.backend;
      return true;
    }
  }
  return false;
}

const char* BackendLogName(Backend backend) {
  const auto* info = FindBackend(backend);
  if (!info) return "unknown";
  return info->legacy_log_name ? info->legacy_log_name : info->name;
}

FLAGOS_EXPORT void SetBackendConfigPath(const std::string& path) {
  BackendConfigPathOverride() = path;
}

constexpr const char* kLogItems[] = {"dispatch", "fallback", "op_cache"};

bool LogEnabled(const char* item) {
  // Parse and validate the logging flags on first use.
  static const std::string list = [] {
    const std::string raw = flagos_env::EnvValue("FLAGOS_LOG");
    size_t pos = 0;
    while (pos <= raw.size()) {
      const size_t comma = raw.find(',', pos);
      const size_t end = comma == std::string::npos ? raw.size() : comma;
      const std::string word = TrimStr(raw.substr(pos, end - pos));

      bool known = word.empty();
      for (const char* candidate : kLogItems) {
        if (flagos_env::EnvIs(word.c_str(), candidate)) {
          known = true;
          break;
        }
      }
      if (!known) {
        flagos_env::EnvWarn("FLAGOS_LOG=\"" + raw + "\": \"" + word +
                            "\" is not one of dispatch, fallback, op_cache");
      }

      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    return raw;
  }();

  return flagos_env::ListedIn(list.c_str(), item);
}

Backend GetBackendForOp(const std::string& op_name) {
  return GetBackendConfigForOp(op_name).backend;
}

const BackendConfigEntry& GetBackendConfigForOp(const std::string& op_name) {
  const auto& table = BackendTable();
  auto it = table.find(op_name);
  // Keep the default entry alive for the per-op cache.
  static const auto* unlisted = new BackendConfigEntry;
  return it != table.end() ? it->second : *unlisted;
}

bool HasBackendForOp(const std::string& op_name) {
  const auto& table = BackendTable();
  return table.find(op_name) != table.end();
}

// Explicit FlagGems and vendor requests report a missing kernel.
void ThrowIfForcedBackendMissing(const std::string& op_name, Backend backend) {
  const std::string& forced = ForcedBackendMode();
  if (forced.empty()) return;

  if (forced == "flaggems" && IsFlagGems(backend)) {
    throw std::runtime_error(
        op_name +
        ": FLAGOS_FORCE_BACKEND=flaggems but no FlagGems implementation is "
        "compiled into this wheel (rebuild with FLAGOS_BUILD_FLAGGEMS_CPP=1 "
        "or FLAGOS_BUILD_FLAGGEMS=1)");
  }
  if (forced == "vendor" && !IsFlagGems(backend) && backend != Backend::kNone &&
      backend != Backend::kTileOps) {
    throw std::runtime_error(
        op_name +
        ": FLAGOS_FORCE_BACKEND=vendor but no vendor kernel is registered "
        "for this op");
  }
}

std::string DispatchFailureMessage(Backend backend) {
  if (backend == Backend::kNone) {
    return ": routed to 'none' (no accelerated impl on this platform) but the "
           "op is registered on PrivateUse1 -- regenerate the vendor conf so "
           "registration and routing agree";
  }
  std::string message = ": backend not registered";
#if defined(FLAGOS_BUILTIN_KERNELS)
  const auto* info = FindBackend(backend);
  if (info && info->kernel_set &&
      !flagos_env::ListedIn(FLAGOS_BUILTIN_KERNELS, info->kernel_set)) {
    message += std::string(" (the '") + info->kernel_set +
               "' kernel set was not compiled into this wheel)";
  }
#endif
  return message;
}

} // namespace at::native::flagos
