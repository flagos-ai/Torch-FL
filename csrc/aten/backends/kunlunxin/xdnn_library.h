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

// Runtime resolution of the vendor xdnn op library's own dependencies.
//
// libtorch_fl.so links libxdnn_pytorch.so by DT_NEEDED, so the dynamic loader
// has resolved it before any of this code runs. That resolution needs more than
// the op library itself: its DT_NEEDED names the vendor's device-side kernels
//
//     libxpu_blas.so  libxpu_dnn.so  libxpu_flash_attention.so  libcudart.so.12
//
// and the vendor torch package keeps them *inside its Python package directory*
// (``<site-packages>/torch_xmlir/*.so``) rather than next to a library the XPU
// SDK installs. Those files carry no torch ABI dependency and importing the
// ``torch_xmlir`` Python package is NOT required -- but the directory has to be
// on the loader path, and when it is not the failure is
//
//     ImportError: libxpu_blas.so: cannot open shared object file
//
// which names neither the missing piece nor where it lives.
//
// This header closes that gap from the runtime side, without the wheel needing
// to know the build host's layout:
//
//   * ``FLAGOS_KUNLUNXIN_XDNN_LIB_DIR`` -- directory holding the component
//     libraries (colon-separated for more than one). When set, the XPU SDK's
//     CUDA-compat runtime is loaded first, then the component libraries from
//     that directory, so an installed wheel can be pointed at a venv it was not
//     built in.
//   * Otherwise the environment is left alone and the wheel's RUNPATH decides.
//
// What this can and cannot do is set by the loader's own order of operations.
// libtorch_fl.so's NEEDED list is satisfied *before* any initializer in this
// library runs, so when the loader cannot find the component libraries the
// import fails and this code never gets a chance. The load therefore is a
// readiness fallback for the case where it *can* run -- libtorch_fl.so loaded
// from a process that pre-mapped the pieces (or a future lazy import) -- and a
// diagnostic: the message names the variable, the missing libraries and where
// they live. Making the failing case work needs the variable to be on the
// loader's search path *before* `import torch_fl`; that part is the caller's
// (see the kunlunxin section of the install docs).

#if defined(USE_KUNLUNXIN) && !defined(_WIN32)

#include <dlfcn.h>

#include <string>
#include <vector>

#include <flagos_env.h>

namespace at::native::flagos::kunlunxin {

// The XPU SDK libraries that have to be in the global scope *before* the
// vendor's component libraries, in this order. `libxpu_blas.so` carries an
// undefined, unversioned `cudaGetDriverEntryPoint` but no DT_NEEDED on any CUDA
// library, so that reference is satisfied only by whatever the process has
// already mapped RTLD_GLOBAL: measured on a P800, loading it standalone fails
// with `undefined symbol: cudaGetDriverEntryPoint`, and it succeeds as soon as
// `libxpucuda.so` + `libcudart.so.12` + `libxpurt.so.2` are loaded first.
//
// `libxpucuda` is the CUDA-compat runtime the XPU SDK installs; `libcudart.so.12`
// is its SONAME-compatible shim; `libxpurt` is the native runtime the backend
// itself links. They come from the XPU SDK, so they are looked up on the
// loader's default path (which the install RPATH augments) rather than in the
// vendor directory the variable names.
inline const std::vector<const char*>& XpuCudaRuntimeLibraries() {
  static const std::vector<const char*> libs = {
      "libxpucuda.so",
      "libcudart.so.12",
      "libxpurt.so.2",
  };
  return libs;
}

// The component libraries libxdnn_pytorch.so NEEDs that ship with the vendor
// torch package. `libxpuml.so.1` is deliberately absent: it is part of the XPU
// SDK (the install RPATH already covers it), so listing it here would report it
// missing for every correctly configured install.
inline const std::vector<const char*>& XdnnComponentLibraries() {
  static const std::vector<const char*> libs = {
      "libxpu_blas.so",
      "libxpu_dnn.so",
      "libxpu_flash_attention.so",
  };
  return libs;
}

inline std::vector<std::string> SplitSearchPath(const std::string& raw) {
  std::vector<std::string> dirs;
  std::string current;
  for (char c : raw) {
    if (c == ':') {
      if (!current.empty()) dirs.push_back(current);
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) dirs.push_back(current);
  return dirs;
}

// Resolve one library against a search path, load it RTLD_GLOBAL (so a later
// library's relocations can find it), and return false when no directory holds
// it. A library that is already loaded -- the normal case when this runs after
// the loader resolved the NEEDED list -- is a no-op success.
inline bool LoadFromDirs(const std::vector<std::string>& dirs, const char* name) {
  for (const std::string& dir : dirs) {
    const std::string path = dir + "/" + name;
    if (::dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL) != nullptr) {
      return true;
    }
  }
  return false;
}

// Load by SONAME on the loader's own search path (LD_LIBRARY_PATH, RPATH).
inline bool LoadBySoname(const char* name) {
  return ::dlopen(name, RTLD_NOW | RTLD_GLOBAL) != nullptr;
}

// Called once, from a static initializer in xdnn_library.cc. Returns the number
// of component libraries it managed to load; a caller that wants to report a
// partial load can check it, and the ordinary path ignores it.
inline int PreloadXdnnComponentsFromEnv() {
  std::string raw;
  if (!flagos_env::EnvRaw("FLAGOS_KUNLUNXIN_XDNN_LIB_DIR", &raw)) {
    return 0;
  }
  const std::vector<std::string> dirs = SplitSearchPath(raw);
  if (dirs.empty()) {
    flagos_env::EnvWarn(
        "FLAGOS_KUNLUNXIN_XDNN_LIB_DIR is set but names no directory; ignoring "
        "it and leaving the vendor op library to the loader's search path");
    return 0;
  }

  // Order matters: blas resolves cudaGetDriverEntryPoint out of the global
  // scope, so the CUDA-compat runtime has to be there first.
  for (const char* name : XpuCudaRuntimeLibraries()) {
    LoadBySoname(name);
  }

  int loaded = 0;
  std::vector<const char*> missing;
  for (const char* name : XdnnComponentLibraries()) {
    if (LoadFromDirs(dirs, name)) {
      ++loaded;
    } else {
      missing.push_back(name);
    }
  }

  if (!missing.empty()) {
    std::string names;
    for (const char* name : missing) {
      if (!names.empty()) names += ", ";
      names += name;
    }
    flagos_env::EnvWarn(
        "FLAGOS_KUNLUNXIN_XDNN_LIB_DIR=" + raw +
        " does not contain " + names +
        "; loading libtorch_fl.so may still fail with \"cannot open shared "
        "object file\". These ship with the vendor torch package (a directory "
        "named torch_xmlir/), not the XPU SDK.");
  }
  return loaded;
}

} // namespace at::native::flagos::kunlunxin

#endif // USE_KUNLUNXIN && !_WIN32