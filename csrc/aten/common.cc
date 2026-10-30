// Copyright (c) 2026, BAAI. All rights reserved.

#include "common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <flagos_env.h>

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

// Maps a conf/env backend name to its enum slot. Returns false for an
// unrecognized name so the caller can warn and pick its own default.
//
// The five names a vendor conf uses are "flaggems_cpp", "flaggems", "tileops",
// "<vendor>" and "none". Legacy spellings "flagos" (for flaggems_cpp),
// "flaggems_python" and "flagos_python" (for flaggems) stay accepted so an
// out-of-tree conf or a pinned script keeps working.
bool ParseBackendName(const std::string& raw, Backend* out) {
  const std::string val = LowerStr(raw);
  if (val == "boxing" || val == "nvidia") {
    // `nvidia` is the key the generator emits for the NVIDIA kernel set: on a
    // CUDA wheel the boxing kernels are that vendor's own kernels, routed
    // through kBoxing. Without this arm the conf is read as an unknown backend
    // for every op it routes that way, and they silently take FlagGems.
    *out = Backend::kBoxing;
  } else if (val == "cuda") {
    // Legacy spelling: the boxing kernel used to be named for the CUDA runtime
    // it borrows. Still accepted so an older conf or FLAGOS_OP_override keeps
    // working; BackendName() reports the canonical "boxing".
    *out = Backend::kBoxing;
  } else if (val == "metax") {
    *out = Backend::kMetax;
  } else if (val == "hygon" || val == "thead") {
    // This CUDA-ABI vendor's kernel slot was retired with the mxcc backend, so
    // there is no Backend for it. Before the vendor rename a conf could not name
    // it either; the two spellings are accepted only so an out-of-tree conf (or a
    // FLAGOS_OP_* override written earlier) does not fail to parse. They resolve
    // to kNone, the same answer the tree used to give the op.
    *out = Backend::kNone;
  } else if (val == "tsingmicro") {
    *out = Backend::kTsingMicro;
  } else if (val == "enflame" || val == "gcu") {  // gcu: the SDK-era spelling
    *out = Backend::kEnflame;
  } else if (val == "kunlunxin") {
    *out = Backend::kKunlunxin;
  } else if (val == "iluvatar") {
    *out = Backend::kIluvatar;
  } else if (val == "ascend") {
    *out = Backend::kAscend;
  } else if (val == "mthreads" || val == "musa") {  // musa: the SDK-era spelling
    *out = Backend::kMthreads;
  } else if (val == "tileops") {
    *out = Backend::kTileOps;
  } else if (val == "apple" || val == "mps") {  // mps: the SDK-era spelling
    *out = Backend::kApple;
  } else if (val == "drobotics" || val == "bpu") {  // bpu: the platform id
    *out = Backend::kDrobotics;
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
    case Backend::kBoxing:       return "boxing";
    case Backend::kFlagGemsCpp:  return "flaggems_cpp";
    case Backend::kFlagGems:     return "flaggems";
    case Backend::kAscend:       return "ascend";
    case Backend::kMthreads:     return "mthreads";
    case Backend::kMetax:        return "metax";
    case Backend::kTsingMicro:   return "tsingmicro";
    case Backend::kEnflame:      return "enflame";
    case Backend::kKunlunxin:    return "kunlunxin";
    case Backend::kIluvatar:     return "iluvatar";
    case Backend::kTileOps:      return "tileops";
    case Backend::kApple:        return "apple";
    case Backend::kDrobotics:    return "drobotics";
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
// annotation; FLAGOS_FORCE_BACKEND=vendor needs it to know which ops may
// legally be pinned back to the vendor kernel.
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

// The last boolean reader in this file was EnvIsOn, a thin forwarder onto
// flagos_env::EnvFlag that existed only to keep the truth table in one place
// (csrc/include/flagos_env.h) while its call sites were migrated. It had two --
// both table-collapse switches -- and FLAGOS_FORCE_BACKEND is an enum rather
// than a flag, so the forwarder is gone and every read now names flagos_env
// directly.

bool IsFlagGems(Backend b) {
  return b == Backend::kFlagGemsCpp || b == Backend::kFlagGems;
}

// FLAGOS_FORCE_BACKEND=<flaggems|vendor|tileops> collapses the whole table onto
// one backend family, for A/B measurement against the default mixed routing.
// One name rather than the three it replaces (ALL_USE_FLAGGEMS, ALL_USE_VENDOR,
// FLAGOS_USE_TILEOPS), because "two of them at once" was a state the old code
// had to detect and reject at runtime -- here it is unrepresentable.
//
// An op is only movable if the target backend actually implements it -- known
// from the routed value plus the `# <backend>` annotation. Ops that are not are
// reported and left on their configured backend; nothing is silently rerouted to
// a kernel that does not exist, which would surface much later as "backend not
// registered" mid-model. That is also why the flaggems and vendor modes raise on
// a miss inside Dispatcher (dispatcher.h) rather than falling through: the user
// asked for that backend across the whole table.
//
// The vendor mode is partial by nature -- a vendor implements far fewer ops than
// FlagGems -- and an op routed to tileops already counts as a vendor-side
// target, so it is left where it is.
//
// The tileops mode is an opt-in path in a different sense: it needs the
// `tileops` package, an SM90 device and a FLAGOS_BUILD_TILEOPS=ON build, so the
// default routing must not name it. Its op set is the one the conf annotates
// `# tileops` -- keeping the set in the conf as an annotation rather than in a
// separate backends_tileops.conf means the per-op default and the TileOPs
// candidate are stated on one line, and the same `alt` map the other modes read
// carries it. Ops it misses are simply not TileOPs candidates; there is nothing
// to report.
void ApplyForcedBackend(std::unordered_map<std::string, Backend>& table,
                        const std::unordered_map<std::string, Backend>& alt) {
  const std::string& mode = ForcedBackendMode();
  if (mode.empty()) return;

  const bool to_tileops = mode == "tileops";
  const bool to_flaggems = mode == "flaggems";

  std::vector<std::string> unsupported;
  size_t moved = 0;
  for (auto& [op, backend] : table) {
    auto it = alt.find(op);
    Backend annotated = it != alt.end() ? it->second : Backend::kNone;

    if (to_tileops) {
      if (annotated != Backend::kTileOps || backend == Backend::kTileOps) continue;
      backend = Backend::kTileOps;
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
      unsupported.push_back(op);
      continue;
    }
    if (target != backend) {
      backend = target;
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

// One decision point, three sources in order: the path Python resolved at
// import time, then an explicitly set FLAGOS_BACKEND_CONFIG, then the path
// derived from where this library was loaded from.
//
// The first used to be an environment write -- _select_backend_config() set
// FLAGOS_BACKEND_CONFIG so this function would read it back. That made the
// wheel's own choice indistinguishable from a user's: once written, a test
// asking "is the user overriding the conf?" got the answer "yes" on every
// install, and a stale export in the shell of a CI job could not be told apart
// from the build's own selection. The setter keeps the value here, in the one
// place that consumes it.
//
// An explicitly set but empty FLAGOS_BACKEND_CONFIG means "unset", so a shell
// idiom like FLAGOS_BACKEND_CONFIG=$EXTRA_CONF falls back to auto-detection
// rather than failing to open "".
std::string ResolveBackendConfigPath() {
  std::string path = BackendConfigPathOverride();
  if (path.empty()) path = flagos_env::EnvValue("FLAGOS_BACKEND_CONFIG");
  if (path.empty()) path = DefaultConfigPath();
  return path;
}

std::unordered_map<std::string, Backend> LoadBackendConfig() {
  std::unordered_map<std::string, Backend> table;

  std::string path = ResolveBackendConfigPath();

  // A conf that will not open is fatal, because the failure is otherwise
  // indistinguishable from a working configuration. ParseConfigInto leaves the
  // table empty, and an empty table makes GetBackendForOp answer
  // Backend::kFlagGems for every op -- so one stderr line is all that separates
  // a typo from a run where the whole workload takes the slowest route. Measured
  // on A100: FLAGOS_BACKEND_CONFIG=/nonexistent/nope.conf sends every dispatch
  // to flagos_python, including the ~1600 ops the conf routes to CUDA boxing,
  // and it also silences the Python-side FlagGems setup that reads this same
  // file (_conf_routes_to_flaggems in torch_fl/__init__.py).
  //
  // Only the top-level file is fatal. A missing `include` inside a conf still
  // warns and continues: the generated confs state every op directly and need
  // no include, so that path is legacy, and a failed include drops fewer routes
  // than the file the caller named.
  {
    std::ifstream probe(path);
    TORCH_CHECK(probe.good(), "flagos: cannot open backend config '", path,
                "'. Unset FLAGOS_BACKEND_CONFIG to let torch_fl select the conf "
                "for this build, or point it at a file that exists.");
  }

  std::unordered_map<std::string, Backend> alt;
  ParseConfigInto(path, table, alt);
  ApplyForcedBackend(table, alt);

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
    std::string override_val = flagos_env::EnvValue(key.c_str());
    if (override_val.empty()) continue;
    Backend parsed;
    if (ParseBackendName(override_val, &parsed)) {
      table[op] = parsed;
      fprintf(stderr, "[flagos] env override: %s -> %s\n", op.c_str(),
              BackendName(parsed));
    } else {
      fprintf(stderr, "[flagos] env override: unknown backend '%s' for op '%s', ignored\n",
              override_val.c_str(), op.c_str());
    }
  }

  return table;
}

const std::unordered_map<std::string, Backend>& BackendTable() {
  static const auto table = LoadBackendConfig();
  return table;
}

} // namespace

// Set once by torch_fl._select_backend_config() through
// torch_fl._C._set_backend_config_path(), before the first op dispatch builds
// the table. Called from Python rather than written to os.environ so the
// wheel's own choice stays distinguishable from a user's -- see
// ResolveBackendConfigPath() above.
FLAGOS_EXPORT void SetBackendConfigPath(const std::string& path) {
  BackendConfigPathOverride() = path;
}

// Declared in common.h and read from dispatcher.h's inline dispatch path, so it
// lives outside the anonymous namespace above.
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

// The items FLAGOS_LOG honors. Kept beside the reader rather than in
// flagos_env.h because this is the only translation unit that reads the
// variable, and EnvListed/ListedIn there stay generic.
constexpr const char* kLogItems[] = {"dispatch", "fallback", "op_cache"};

bool LogEnabled(const char* item) {
  // The list is parsed once per process, on the first lookup, which is also
  // where it is validated: every item the list names must be one we honor.
  // Anything else would turn its diagnostic off with no sign that the setting
  // was wrong, so it is reported instead. The three booleans this replaced each
  // warned about a bad value; FLAGOS_LOG must not be the one that does not.
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
  const auto& table = BackendTable();
  auto it = table.find(op_name);
  // kFlagGems is the answer for an op the conf does not mention. In a healthy
  // process that never happens: registration and routing are both generated from
  // the same conf, so an op absent from the table has no m.impl either and the
  // call reaches cpu_fallback without ever arriving here. Reaching this line
  // therefore means the running conf and the compiled registration disagree --
  // most often because FLAGOS_BACKEND_CONFIG named a file other than the one the
  // build was generated from. FlagGems is the deliberate choice for that case
  // (it is the broadest route, so the op usually still runs), and
  // LoadBackendConfig above makes the one disagreement it can detect -- an
  // unreadable file -- fatal instead of silent.
  return it != table.end() ? it->second : Backend::kFlagGems;
}

bool HasBackendForOp(const std::string& op_name) {
  const auto& table = BackendTable();
  return table.find(op_name) != table.end();
}

bool FlagGemsRejectsDtype(at::ScalarType dtype) {
#if defined(FLAGOS_ACCEL_METAX)
  // Triton's MetaX backend cannot lower complex pointers. FlagGems kernels
  // therefore fail before launch for every complex-valued op measured here
  // (the first FlagQuantum failure is flip):
  //
  //   KeyError: 'complex64'
  //
  // MetaX is CUDA ABI compatible, and boxing the same calls into the vendor
  // torch kernels preserves both the complex dtype and device residency.
  return dtype == at::kComplexFloat || dtype == at::kComplexDouble;
#elif defined(FLAGOS_ACCEL_DCU)
  // Same gap, same shape as MetaX, and measured separately in the shipped DCU
  // environment (torch 2.10.0+dtk2604, HIP 6.3.26113, DTK, `backends_hygon.conf`,
  // boxing ON). The released wheel predates this change, so it reproduces the
  // two defects below; the fix was measured on a libtorch_fl.so rebuilt from
  // this source and copied into that same environment.
  //
  // Without this arm a complex factory never failed loudly: it silently lost
  // its dtype before FlagGems was reached, because the wheel's dtype converter
  // had no complex arm (issue #540). The visible result was
  //
  //   torch.zeros(4, dtype=torch.complex64, device="flagos:0").dtype
  //   # -> torch.float32
  //
  // FlagGems' HCU Triton backend has no complex pointer type either, so simply
  // preserving the dtype in the converter is not enough -- the call must leave
  // the FlagGems route. With this escape the complex call reroutes to the DCU
  // CUDA boxing kernels, measured on the rebuilt build (FLAGOS_LOG=dispatch,
  // and values checked against the CPU reference):
  //
  //   zeros  -> cuda   complex64/128 preserved (was silent float32)
  //   ones   -> cuda   complex64/128 preserved (was silent float32)
  //   randn  -> cuda   complex64/128 preserved (was silent float32)
  //   abs    -> cuda   correct real result    (was KeyError: 'complex64')
  //   fill_  -> cuda   correct               (composed through the same escape)
  //
  // the last one because OnesKernelCuda and ZeroKernelCuda are compositional
  // (`at::empty(...).fill_(1)` / `.zero_()`), and the inner in-place call is
  // itself diverted -- `fill_` and `zero_` are registered on the FlagGems
  // route and the DCU codegen has a kBoxing kernel for each. The escape is
  // dtype-scoped, so the same op over float is untouched: measured
  // `ones(float32)` still logs `ones -> flagos_python`.
  //
  // ResolveFn returns the FlagGems route unchanged when no vendor kernel
  // exists, so an op outside this intersection keeps its own outcome (loud
  // failure) rather than changing route.
  return dtype == at::kComplexFloat || dtype == at::kComplexDouble;
#elif defined(USE_ASCEND)
  // See the declaration in common.h for the measurement. float64 is the only
  // dtype FlagGems' Ascend route could not serve across the pointwise family;
  // fp32/fp16/bf16/int64/bool all compile and run.
  return dtype == at::kDouble;
#elif defined(USE_GCU)
  // See the declaration in common.h for the measurement. int64 is the dtype
  // that matters: it fails inside the compiler for every flag_gems pointwise
  // kernel that takes one, and for remainder.Tensor it does not fail at all --
  // it returns int32. float64 fails the same way but is deliberately left
  // alone, because one FlagGems float64 route in the same seven-op
  // intersection (remainder.Tensor) is correct today.
  return dtype == at::kLong;
#else
  // No vendor route is known to be dtype-limited in this way. MUSA's
  // FlagGems gaps are per-op (a Python-float operand against a bf16 tensor)
  // and are already recorded in NATIVE_TRITON_GAPS, so they are not reached
  // from here.
  (void)dtype;
  return false;
#endif
}

bool FlagGemsRejectsOpDtype(const char* op_name, at::ScalarType dtype) {
  if (FlagGemsRejectsDtype(dtype)) return true;
#if defined(USE_ASCEND)
  // The intersection this table exists for: a FlagGems kernel that does not
  // guard the element type it code-generates, for a dtype the reference
  // implementation of that op rejects outright. See the declaration in
  // common.h for why neither a conf entry nor the dtype-wide predicate can
  // state it.
  //
  // `neg` over bool. flag_gems/ops/neg.py is a bare pointwise `-x`, so a bool
  // operand is code-generated like any other element type and lowers to
  // `hivm.hir.vadd` over `i1`, which BiShengIR refuses to verify:
  //
  //   MLIRCompilationError: 'hivm.hir.vadd' op failed to verify that operand
  //   at idx 0 and 1 should have element type 16-bit signless integer or
  //   32-bit signless integer or 16-bit float or 32-bit float or 64-bit
  //   signless integer
  //
  // (Ascend910, CANN 9.0.0, FlagTree 0.6.2a1+ascend3.5.) Reached through the
  // vendor slot instead, the aclnn kernel falls back to at::neg on a CPU copy
  // for a dtype IsUnaryDtypeSupported does not cover -- which is where the
  // reference behaviour
  //
  //   RuntimeError: Negation, the `-` operator, on a bool tensor is not
  //   supported. If you are trying to invert a mask, use the `~` or
  //   `logical_not()` operator instead.
  //
  // is raised. tests/integration/test_dtype_coverage.py pins that contract.
  //
  // `neg_` is deliberately absent: its vendor template (T_INPLACE_UNARY) has
  // no dtype guard at all, so routing bool there reaches aclnnInplaceNeg and
  // fails with "aclnnInplaceNegGetWorkspaceSize failed, ret=161002" rather
  // than with the reference message. That is a separate defect in the
  // in-place codegen and is left visible rather than papered over here.
  //
  // `mm` and `mm.out` over the integral dtypes are the second entry, and the
  // only one where the op stays on the FlagGems route while part of its dtype
  // set leaves it. The FlagGems matmul kernel builds its tl.dot from the
  // operand element type, and the Ascend dot intrinsic takes floating point
  // only, so int64/int32/int16/uint8 operands die inside the compiler:
  //
  //   triton.compiler.errors.CompilationError: at 56:15:
  //     acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
  //   AssertionError('Unsupported lhs dtype int64')
  //
  // (Ascend910C, CANN 9.0.0, shipped backends_ascend.conf.) int8 is the
  // exception and is left alone: tl.dot accepts it and the FlagGems route is
  // correct for it. The aclnn kernel behind the vendor slot serves all four --
  // integer matmul is exact there, so this is not a precision trade. Measured
  // on the same conf: int64/int32/int16/uint8 `mm` on the vendor route agree
  // with the CPU reference exactly.
  //
  // A route cannot state this split, which is why the table has to. `mm` was
  // moved off the vendor route because the aclnn cube path downcasts fp32 to
  // HF32 across the whole float family -- 4.7e-3 against a CPU reference, where
  // tests/integration/runtime asserts 1e-3 and the FlagGems kernel answers
  // 8.6e-6 against that same reference (see NATIVE_TRITON_GAPS in
  // scripts/codegen/gen_vendor_confs.py). Leaving mm on the vendor route keeps
  // the float failure; leaving every dtype of it on FlagGems turns working
  // integral matmul into a compile error.
  struct OpDtypeGap {
    at::ScalarType dtype;
    const char* op_name;
  };
  static constexpr OpDtypeGap kGaps[] = {
      {at::kBool, "neg"},
      // Integral matmul: the four dtypes the Ascend dot intrinsic refuses.
      // "mm.out" is its own conf key, so it needs its own entries.
      {at::kLong, "mm"},
      {at::kInt, "mm"},
      {at::kShort, "mm"},
      {at::kByte, "mm"},
      {at::kLong, "mm.out"},
      {at::kInt, "mm.out"},
      {at::kShort, "mm.out"},
      {at::kByte, "mm.out"},
  };

  for (const OpDtypeGap& gap : kGaps) {
    // dtype first, so the common path costs one enum compare per entry before
    // any strcmp -- this runs on every FlagGems dispatch.
    if (dtype == gap.dtype && std::strcmp(op_name, gap.op_name) == 0) {
      return true;
    }
  }
  return false;
#else
  // No vendor route is known to have a gap that is specific to one op and
  // dtype in this way. See the dtype-wide case above for what MUSA records
  // instead.
  (void)op_name;
  return false;
#endif
}

int64_t TritonMinDotContractExtent() {
#if defined(FLAGOS_ACCEL_METAX)
  // 16, measured. FlagTree's adaptive block size (AABS) shrinks a block that
  // is wider than the tensor -- `if bs > ts: updated_bs = next_power_of_2(ts)`
  // in triton/runtime/adjust_kernel_param.py -- so FlagGems' BLOCK_K of 32
  // becomes next_power_of_2(K) for a small K. Its `metax` arm then restores M
  // and N to >= 16 and leaves K where the shrink put it, while the `iluvatar`
  // and `sunrise` arms next to it restore all three. The dot intrinsic then
  // refuses the tile:
  //
  //   triton/spec/metax/triton/language/semantic.py: in dot
  //     assert lhs.shape[-2].value >= min_dot_size[0] and ...
  //   AssertionError: Input shapes should have M >= 1, N >= 1 and K >= 16
  //
  // Measured on a MetaX C550 (torch 2.10.0+metax3.8.1.0, flag_gems 5.4.0),
  // calling FlagGems' mm directly with a cold FLAGGEMS_CACHE_DIR:
  //
  //   (4, 4) x (4, 4)    CompilationError, the assertion above
  //   (4, 8) x (8, 4)    CompilationError, the assertion above
  //   (4, 16) x (16, 4)  OK, max_abs 4.4e-07 against a CPU fp64 reference
  //   (4, 32) x (32, 4)  OK, max_abs 1.8e-06
  //
  // The floor is 16 and not `next_power_of_2(K) < 16`: AABS only shrinks when
  // the block exceeds the extent, so a config that already carries a small
  // BLOCK_K trips the same assertion at a K the power-of-two rule would pass
  // (the thead measurement in scripts/codegen/gen_vendor_confs.py records
  // K = 4/8/8 failing while 8/8/15 passes -- which config the tuner draws
  // decides it). Padding every K below 16 costs a copy of a tensor with fewer
  // than 16 columns and removes the dependence on that draw.
  //
  // This is a shim for a vendor-wheel gap, and it has an expiry: FlagGems pads
  // K itself from 5.6, and FlagTree's metax arm should restore K the way its
  // iluvatar arm does. When either lands in the pinned cohort, delete this
  // value and the padding that reads it.
  return 16;
#else
  // No other platform that compiles a soft-lowp decode is known to impose a
  // contracted-extent floor. Hygon DCU's FlagTree arm also restores only M and
  // N, but its HIP dot intrinsic has not been measured to refuse a small K --
  // if its lane ever raises the assertion above, that is this same shim with
  // its own measurement, not a reason to widen this one on a guess.
  return 0;
#endif
}

int64_t MatmulContractDim(const at::Tensor& operand, bool is_rhs) {
  return is_rhs ? operand.size(-2) : operand.size(-1);
}

at::Tensor PadMatmulContractDim(const at::Tensor& operand, bool is_rhs,
                                int64_t target) {
  auto shape = operand.sizes().vec();
  const auto axis = is_rhs ? shape.size() - 2 : shape.size() - 1;
  const auto extent = shape[axis];
  shape[axis] = target;
  auto padded = at::empty(shape, operand.options());
  padded.zero_();
  padded.narrow(static_cast<int64_t>(axis), 0, extent).copy_(operand);
  return padded;
}

bool PadMatmulPairForDot(at::Tensor& lhs, at::Tensor& rhs) {
  const auto floor = TritonMinDotContractExtent();
  if (floor <= 0) {
    return false;
  }
  // Early out on the path that matters: a real GEMM has a contracted extent in
  // the hundreds or thousands and pays one integer compare here.
  const auto extent = MatmulContractDim(lhs, false);
  if (extent >= floor || extent == 0) {
    return false;
  }
  if (MatmulContractDim(rhs, true) != extent) {
    // Mismatched operands are the caller's error, and the kernel below reports
    // it with the shapes in the message. Padding them to a common target here
    // would turn that into a wrong answer.
    return false;
  }
  lhs = PadMatmulContractDim(lhs, false, floor);
  rhs = PadMatmulContractDim(rhs, true, floor);
  return true;
}

} // namespace at::native::flagos
