// Copyright (c) 2026, BAAI. All rights reserved.

#include "common.h"

#include <cstdio>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

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

// Parse one conf file into `table`. Later assignments win, so a caller that
// wants to override an inherited route just restates the op after the
// `include`.
//
// A line of the form `include <path>` splices another conf in at that point.
// Relative paths resolve against the including file's directory. This lets a
// hybrid conf (e.g. backends_ascend_flagos_py.conf) inherit the full vendor
// baseline and state only its own overrides, instead of duplicating every op.
// Duplicating was the previous arrangement and it silently rotted: the Ascend
// baseline grew to 223 ops via codegen while the hybrid conf stayed at 55, so
// 168 ops fell through to Backend::kFlagOs -- which has no kernel registered in
// an Ascend build, surfacing as "<op>: backend not registered" at runtime.
//
// `depth` bounds include recursion so a cyclic include can't hang import.
void ParseConfigInto(const std::string& path,
                     std::unordered_map<std::string, Backend>& table,
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
    // strip comments
    auto comment = line.find('#');
    if (comment != std::string::npos) line = line.substr(0, comment);

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
        ParseConfigInto(inc, table, depth + 1);
      }
      continue;
    }

    auto trim = [](std::string s) { return TrimStr(std::move(s)); };

    std::string op = trim(line.substr(0, eq));
    std::string val = trim(line.substr(eq + 1));

    if (op.empty() || val.empty()) continue;

    if (val == "cuda") {
      table[op] = Backend::kCuda;
    } else if (val == "metax") {
      table[op] = Backend::kMetax;
    } else if (val == "tsingmicro") {
      table[op] = Backend::kTsingMicro;
    } else if (val == "gcu") {
      table[op] = Backend::kGcu;
    } else if (val == "ascend") {
      table[op] = Backend::kAscend;
    } else if (val == "musa") {
      table[op] = Backend::kMusa;
    } else if (val == "flagos" || val == "flaggems") {
      table[op] = Backend::kFlagOs;
    } else if (val == "flagos_python" || val == "flaggems_python") {
      table[op] = Backend::kFlagOsPython;
    } else if (val == "tileops") {
      table[op] = Backend::kTileOps;
    } else {
      fprintf(stderr, "[flagos] unknown backend '%s' for op '%s', using flagos\n", val.c_str(), op.c_str());
      table[op] = Backend::kFlagOs;
    }
  }
}

std::unordered_map<std::string, Backend> LoadBackendConfig() {
  std::unordered_map<std::string, Backend> table;

  const char* env = std::getenv("FLAGOS_BACKEND_CONFIG");
  std::string path = env ? env : DefaultConfigPath();

  ParseConfigInto(path, table);

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
    std::string v(override_val);
    if (v == "cuda") {
      table[op] = Backend::kCuda;
      fprintf(stderr, "[flagos] env override: %s -> cuda\n", op.c_str());
    } else if (v == "metax") {
      table[op] = Backend::kMetax;
      fprintf(stderr, "[flagos] env override: %s -> metax\n", op.c_str());
    } else if (v == "tsingmicro") {
      table[op] = Backend::kTsingMicro;
      fprintf(stderr, "[flagos] env override: %s -> tsingmicro\n", op.c_str());
    } else if (v == "gcu") {
      table[op] = Backend::kGcu;
      fprintf(stderr, "[flagos] env override: %s -> gcu\n", op.c_str());
    } else if (v == "ascend") {
      table[op] = Backend::kAscend;
      fprintf(stderr, "[flagos] env override: %s -> ascend\n", op.c_str());
    } else if (v == "musa") {
      table[op] = Backend::kMusa;
      fprintf(stderr, "[flagos] env override: %s -> musa\n", op.c_str());
    } else if (v == "flagos" || v == "flaggems") {
      table[op] = Backend::kFlagOs;
      fprintf(stderr, "[flagos] env override: %s -> flaggems\n", op.c_str());
    } else if (v == "flagos_python" || v == "flaggems_python") {
      table[op] = Backend::kFlagOsPython;
      fprintf(stderr, "[flagos] env override: %s -> flaggems_python\n", op.c_str());
    } else if (v == "tileops") {
      table[op] = Backend::kTileOps;
      fprintf(stderr, "[flagos] env override: %s -> tileops\n", op.c_str());
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
  return it != table.end() ? it->second : Backend::kFlagOs;
}

} // namespace at::native::flagos