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

// Applies FLAGOS_KUNLUNXIN_XDNN_LIB_DIR at library load time.
//
// This has to be a static initializer rather than a call site, and it cannot
// rescue the loader's own NEEDED resolution: DT_NEEDED entries are satisfied
// before any initializer runs, so a build with no usable RUNPATH for
// libxdnn_pytorch.so's components fails at `import torch_fl` and never reaches
// this code. What the initializer does is (a) honour the variable when the
// process still gets this far and (b) replace a bare "cannot open shared object
// file" with a message naming the variable and the libraries it is missing.
//
// The pre-load order is not cosmetic: libxpu_blas.so has an undefined,
// unversioned cudaGetDriverEntryPoint and no DT_NEEDED on a CUDA library, so it
// only relocates once the XPU SDK's CUDA-compat runtime is already global.
//
// See xdnn_library.h for what the variable means and why importing the
// torch_xmlir Python package is not required.

#include "xdnn_library.h"

#if defined(USE_KUNLUNXIN) && !defined(_WIN32)

namespace {

// A namespace-scope object, so the pre-load happens as part of dlopen of
// libtorch_fl.so. Nothing reads the result: the point is that the component
// libraries are mapped before the dynamic loader needs to satisfy any later
// reference to them.
const int kXdnnPreloaded = ::at::native::flagos::kunlunxin::
    PreloadXdnnComponentsFromEnv();

} // namespace

#endif // USE_KUNLUNXIN && !_WIN32