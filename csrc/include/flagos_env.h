// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// The C++ mirror of torch_fl/_env.py: the one place a FLAGOS_* environment
// variable is read. Header-only and free of any ATen dependency, so csrc/aten/,
// csrc/runtime/ and csrc/profiler/ can all include it -- including the
// standalone tracer shims, which are dlopen'd rather than linked into
// libtorch_fl.so.
//
// Before this header each translation unit parsed booleans its own way. Three
// different truth tables were in use: anything-not-"0", literal '"1"' only, and
// existence (getenv != nullptr, which made FLAGOS_CUPTI_SHIM_DEBUG=0 turn the
// logging *on*). The table below is stated once and matches the Python side
// line for line.
//
// Deliberate carve-out: csrc/runtime/accelerator/metax/cudart_shim.c. setup.py
// builds it standalone into libcudart_shim.so for LD_PRELOAD, so it cannot link
// libtorch_fl.so and keeps its own getenv. It reads only vendor *paths*
// (MACA_PATH/MACA_HOME), never a boolean, so it is unaffected by the table.

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>

namespace flagos {

// Emit one "[flagos] ..." line to stderr. Warnings name the variable and the
// value that was rejected, so a misconfiguration is diagnosable from the log
// alone. Matches torch_fl._env.warn.
inline void EnvWarn(const std::string& message) {
  std::fprintf(stderr, "[flagos] %s\n", message.c_str());
}

// The raw value, trimmed. Returns false when the variable is unset, empty, or
// only whitespace -- all three mean "not set", which is what a shell idiom like
// FLAGOS_LOG=${EXTRA_LOG} produces. Matches _env.py's _raw().
inline bool EnvRaw(const char* name, std::string* out) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) return false;
  size_t begin = 0;
  size_t end = std::strlen(raw);
  while (begin < end && std::isspace(static_cast<unsigned char>(raw[begin]))) ++begin;
  while (end > begin && std::isspace(static_cast<unsigned char>(raw[end - 1]))) --end;
  if (begin == end) return false;
  out->assign(raw + begin, end - begin);
  return true;
}

// Case-insensitive compare against a lowercase literal. Names in this header
// are always lowercase ASCII.
inline bool EnvIs(const char* raw, const char* lowered) {
  for (; *raw != '\0' && *lowered != '\0'; ++raw, ++lowered) {
    char c = *raw;
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    if (c != *lowered) return false;
  }
  return *raw == '\0' && *lowered == '\0';
}

// Read a boolean switch. 1/true/on/yes are true, 0/false/off/no and an unset or
// empty value are false. Anything else warns and returns default_value rather
// than being treated as truthy.
inline bool EnvFlag(const char* name, bool default_value = false) {
  std::string raw;
  if (!EnvRaw(name, &raw)) return default_value;
  const char* v = raw.c_str();
  if (EnvIs(v, "1") || EnvIs(v, "true") || EnvIs(v, "on") || EnvIs(v, "yes")) {
    return true;
  }
  if (EnvIs(v, "0") || EnvIs(v, "false") || EnvIs(v, "off") || EnvIs(v, "no")) {
    return false;
  }
  EnvWarn(name + (": \"" + raw + "\" is not a boolean; using the default"));
  return default_value;
}

// Read a string value, with an unset or empty variable meaning `def`.
inline std::string EnvValue(const char* name, const char* def = "") {
  std::string raw;
  return EnvRaw(name, &raw) ? raw : std::string(def);
}

// Read one of `allowed` (case-insensitively). Returns false and writes `def`
// when the value is unset, and returns false after warning when it is present
// but not listed -- so a caller can distinguish "not set" from "set wrong".
inline bool EnvChoice(const char* name,
                      std::initializer_list<const char*> allowed,
                      const char* def,
                      std::string* out) {
  std::string raw;
  if (!EnvRaw(name, &raw)) {
    *out = def;
    return false;
  }
  for (const char* candidate : allowed) {
    if (EnvIs(raw.c_str(), candidate)) {
      *out = candidate;
      return true;
    }
  }
  EnvWarn(std::string(name) + "=\"" + raw + "\" is not a recognised value; using " + def);
  *out = def;
  return true;
}

// True when the comma-separated list in `name` contains `item`
// (FLAGOS_LOG=dispatch,fallback). Whitespace around entries is ignored, so a
// trailing comma is harmless.
inline bool EnvListed(const char* name, const char* item) {
  std::string haystack;
  if (!EnvRaw(name, &haystack)) return false;
  const size_t item_len = std::strlen(item);
  size_t pos = 0;
  while (pos <= haystack.size()) {
    const size_t comma = haystack.find(',', pos);
    const size_t end = comma == std::string::npos ? haystack.size() : comma;
    size_t begin = pos;
    size_t stop = end;
    while (begin < stop && std::isspace(static_cast<unsigned char>(haystack[begin]))) ++begin;
    while (stop > begin && std::isspace(static_cast<unsigned char>(haystack[stop - 1]))) --stop;
    if (stop - begin == item_len && EnvIs(haystack.substr(begin, stop - begin).c_str(), item)) {
      return true;
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return false;
}

}  // namespace flagos
