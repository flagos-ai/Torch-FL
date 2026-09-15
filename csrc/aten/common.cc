// Copyright (c) 2026, BAAI. All rights reserved.

#include "common.h"

#include <algorithm>
#include <cstdio>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace at::native::flagos {

namespace {

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
      platform = "gcu";
#elif defined(USE_ASCEND)
      platform = "ascend";
#elif defined(USE_MUSA)
      platform = "musa";
#elif defined(USE_BPU)
      platform = "bpu";
#endif
      if (platform) {
        // dir is <prefix>/torch_fl/lib, configs are at <prefix>/torch_fl/configs/
        std::string candidate =
            dir + "/../configs/backends_" + platform + ".conf";
        std::ifstream test(candidate);
        if (test.is_open()) return candidate;
      }

      // Try package-relative: <dir>/../configs/backends.conf
      std::string candidate = dir + "/../configs/backends.conf";
      std::ifstream test(candidate);
      if (test.is_open()) return candidate;
      // Try: <dir>/configs/backends.conf
      candidate = dir + "/configs/backends.conf";
      test.open(candidate);
      if (test.is_open()) return candidate;
    }
  }
#endif
  // Fallback to build-time path
  return FLAGOS_SOURCE_ROOT "/torch_fl/configs/backends.conf";
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

// Maps a conf/env backend name to its enum slot. Returns false for an
// unrecognized name so the caller can warn and pick its own default.
//
// The five names a vendor conf uses are "flaggems_cpp", "flaggems", "tileops",
// "<vendor>" and "none". Legacy spellings "flagos" (for flaggems_cpp),
// "flaggems_python" and "flagos_python" (for flaggems) stay accepted so an
// out-of-tree conf or a pinned script keeps working.
bool ParseBackendName(const std::string& raw, Backend* out) {
  const std::string val = LowerStr(raw);
  if (val == "cuda") {
    *out = Backend::kCuda;
  } else if (val == "metax") {
    *out = Backend::kMetax;
  } else if (val == "tsingmicro") {
    *out = Backend::kTsingMicro;
  } else if (val == "gcu") {
    *out = Backend::kGcu;
  } else if (val == "ascend") {
    *out = Backend::kAscend;
  } else if (val == "musa") {
    *out = Backend::kMusa;
  } else if (val == "tileops") {
    *out = Backend::kTileOps;
  } else if (val == "flaggems_cpp" || val == "flagos") {
    *out = Backend::kFlagGemsCpp;
  } else if (val == "flaggems" || val == "flaggems_python" ||
             val == "flagos_python") {
    *out = Backend::kFlagGems;
  } else if (val == "none") {
    *out = Backend::kNone;
  } else {
    return false;
  }
  return true;
}

const char* BackendName(Backend b) {
  switch (b) {
    case Backend::kCuda:         return "cuda";
    case Backend::kFlagGemsCpp:  return "flaggems_cpp";
    case Backend::kFlagGems:     return "flaggems";
    case Backend::kAscend:       return "ascend";
    case Backend::kMusa:         return "musa";
    case Backend::kMetax:        return "metax";
    case Backend::kTsingMicro:   return "tsingmicro";
    case Backend::kGcu:          return "gcu";
    case Backend::kTileOps:      return "tileops";
    case Backend::kNone:         return "none";
    default:                     return "unknown";
  }
}

// Parse one conf file into `table`. Later assignments win, so a caller that
// wants to override an inherited route just restates the op after the
// `include`.
//
// A line of the form `include <path>` splices another conf in at that point.
// Relative paths resolve against the including file's directory. This lets a
// conf inherit another's baseline and state only its own overrides, instead of
// duplicating every op. Duplicating was the previous arrangement and it
// silently rotted: the Ascend baseline grew to 223 ops via codegen while the
// hand-maintained hybrid conf stayed at 55, so 168 ops fell through to
// Backend::kFlagGemsCpp -- which has no kernel registered in an Ascend build,
// surfacing as "<op>: backend not registered" at runtime. The generated vendor
// confs now state all four keys per op directly and need no include.
//
// `depth` bounds include recursion so a cyclic include can't hang import.
// `alt` collects trailing `# <backend>` annotations: a generated vendor conf
// writes `abs = flaggems  # musa` to record that the vendor also implements the
// op even though priority routed it to FlagGems. Runtime routing ignores the
// annotation; ALL_USE_VENDOR needs it to know which ops may legally be pinned
// back to the vendor kernel.
void ParseConfigInto(const std::string& path,
                     std::unordered_map<std::string, Backend>& table,
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

    auto trim = [](std::string s) { return TrimStr(std::move(s)); };

    std::string op = trim(line.substr(0, eq));
    std::string val = trim(line.substr(eq + 1));

    if (op.empty() || val.empty()) continue;

    Backend parsed;
    if (ParseBackendName(val, &parsed)) {
      table[op] = parsed;
    } else {
      fprintf(stderr, "[flagos] unknown backend '%s' for op '%s', using flaggems\n", val.c_str(), op.c_str());
      table[op] = Backend::kFlagGems;
    }

    Backend annotated;
    if (!comment_text.empty() && ParseBackendName(comment_text, &annotated)) {
      alt[op] = annotated;
    } else {
      alt.erase(op);
    }
  }
}

bool EnvIsOn(const char* name) {
  const char* v = std::getenv(name);
  if (!v || !*v) return false;
  std::string s(v);
  return s != "0" && s != "off" && s != "OFF" && s != "false" && s != "FALSE";
}

bool IsFlagGems(Backend b) {
  return b == Backend::kFlagGemsCpp || b == Backend::kFlagGems;
}

// ALL_USE_FLAGGEMS=1 / ALL_USE_VENDOR=1 collapse the whole table onto one
// backend, for A/B measurement against the default mixed routing. Each op is
// only movable if that backend actually implements it -- known from the routed
// value plus the `# <backend>` annotation. Ops that are not are reported and
// left on their configured backend; nothing is silently rerouted to a kernel
// that does not exist, which would surface much later as "backend not
// registered" mid-model.
void ApplyAllUseOverride(std::unordered_map<std::string, Backend>& table,
                         const std::unordered_map<std::string, Backend>& alt) {
  const bool all_flaggems = EnvIsOn("ALL_USE_FLAGGEMS");
  const bool all_vendor = EnvIsOn("ALL_USE_VENDOR");
  if (!all_flaggems && !all_vendor) return;
  if (all_flaggems && all_vendor) {
    fprintf(stderr,
            "[flagos] ALL_USE_FLAGGEMS and ALL_USE_VENDOR are mutually "
            "exclusive, ignoring both\n");
    return;
  }

  std::vector<std::string> unsupported;
  size_t moved = 0;
  for (auto& [op, backend] : table) {
    auto it = alt.find(op);
    Backend annotated = it != alt.end() ? it->second : Backend::kNone;

    // The op's candidates are its routed backend and its annotation.
    Backend target = Backend::kNone;
    if (all_flaggems) {
      if (IsFlagGems(backend)) target = backend;
      else if (IsFlagGems(annotated)) target = annotated;
    } else {
      if (!IsFlagGems(backend) && backend != Backend::kNone) target = backend;
      else if (!IsFlagGems(annotated) && annotated != Backend::kNone) target = annotated;
    }

    if (target == Backend::kNone) {
      unsupported.push_back(op);
      continue;
    }
    if (target != backend) {
      backend = target;
      ++moved;
    }
  }

  const char* which = all_flaggems ? "ALL_USE_FLAGGEMS" : "ALL_USE_VENDOR";
  fprintf(stderr, "[flagos] %s=1: %zu ops repinned, %zu left as configured\n",
          which, moved, unsupported.size());
  if (!unsupported.empty()) {
    std::sort(unsupported.begin(), unsupported.end());
    fprintf(stderr, "[flagos] %s: no implementation for", which);
    for (size_t i = 0; i < unsupported.size() && i < 12; ++i) {
      fprintf(stderr, " %s", unsupported[i].c_str());
    }
    if (unsupported.size() > 12) {
      fprintf(stderr, " ... (+%zu more)", unsupported.size() - 12);
    }
    fprintf(stderr, "\n");
  }
}

// FLAGOS_USE_TILEOPS=1 repins every op the conf annotates `# tileops` onto the
// TileOPs backend. TileOPs is an opt-in path: it needs the `tileops` package
// plus an SM90 device, and its kernels are only built when TILEOPS_KERNEL=ON,
// so the default routing must not name it. Keeping the op set in the conf as an
// annotation -- rather than in a separate backends_tileops.conf -- means the
// per-op default and the TileOPs candidate are stated on one line, and the same
// `alt` map ALL_USE_* already reads carries it.
//
// If a table-collapse flag is also set it wins: ALL_USE_* exists to measure one
// backend against the mixed default, and silently leaving 60 ops on a third
// backend would corrupt exactly that measurement.
void ApplyTileOpsOptIn(std::unordered_map<std::string, Backend>& table,
                       const std::unordered_map<std::string, Backend>& alt) {
  if (!EnvIsOn("FLAGOS_USE_TILEOPS")) return;
  if (EnvIsOn("ALL_USE_FLAGGEMS") || EnvIsOn("ALL_USE_VENDOR")) {
    fprintf(stderr,
            "[flagos] FLAGOS_USE_TILEOPS ignored: ALL_USE_FLAGGEMS/"
            "ALL_USE_VENDOR pin the whole table\n");
    return;
  }

  size_t moved = 0;
  for (auto& [op, backend] : table) {
    auto it = alt.find(op);
    if (it == alt.end() || it->second != Backend::kTileOps) continue;
    if (backend == Backend::kTileOps) continue;
    backend = Backend::kTileOps;
    ++moved;
  }
  fprintf(stderr, "[flagos] FLAGOS_USE_TILEOPS=1: %zu ops repinned to tileops\n",
          moved);
}

std::unordered_map<std::string, Backend> LoadBackendConfig() {
  std::unordered_map<std::string, Backend> table;

  const char* env = std::getenv("FLAGOS_BACKEND_CONFIG");
  std::string path = env ? env : DefaultConfigPath();

  std::unordered_map<std::string, Backend> alt;
  ParseConfigInto(path, table, alt);
  ApplyAllUseOverride(table, alt);
  ApplyTileOpsOptIn(table, alt);

  // Per-op env var overrides: FLAGOS_OP_<op_name>=cuda|metax|flaggems|tileops
  // e.g. FLAGOS_OP_mm=cuda  or  FLAGOS_OP_mm__out=cuda
  // Dots in op names are replaced with double underscores to avoid ambiguity
  // with ops that already contain underscores (e.g. mm_out vs mm.out).
  for (auto& [op, _] : table) {
    std::string key = "FLAGOS_OP_";
    for (char c : op) {
      if (c == '.') key += "__";
      else key += c;
    }
    const char* override_val = std::getenv(key.c_str());
    if (!override_val) continue;
    Backend parsed;
    if (ParseBackendName(override_val, &parsed)) {
      table[op] = parsed;
      fprintf(stderr, "[flagos] env override: %s -> %s\n", op.c_str(),
              BackendName(parsed));
    } else {
      fprintf(stderr, "[flagos] env override: unknown backend '%s' for op '%s', ignored\n",
              override_val, op.c_str());
    }
  }

  return table;
}

const std::unordered_map<std::string, Backend>& BackendTable() {
  static const auto table = LoadBackendConfig();
  return table;
}

} // namespace

Backend GetBackendForOp(const std::string& op_name) {
  const auto& table = BackendTable();
  auto it = table.find(op_name);
  return it != table.end() ? it->second : Backend::kFlagGems;
}

bool FlagGemsRejectsDtype(at::ScalarType dtype) {
#if defined(USE_ASCEND)
  // See the declaration in common.h for the measurement. float64 is the only
  // dtype FlagGems' Ascend route could not serve across the pointwise family;
  // fp32/fp16/bf16/int64/bool all compile and run.
  return dtype == at::kDouble;
#else
  // No vendor route is known to be dtype-limited in this way. MUSA's
  // FlagGems gaps are per-op (a Python-float operand against a bf16 tensor)
  // and are already recorded in NATIVE_TRITON_GAPS, so they are not reached
  // from here.
  (void)dtype;
  return false;
#endif
}

} // namespace at::native::flagos