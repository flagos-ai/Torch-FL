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

#include <tops_runtime_api.h>

namespace c10::flagos::enflame {

// TOPS 4 uses the CUDA-shaped runtime API. Older TopsRider releases expose
// a different pointer attribute type and pinned host allocation/free pair.
#if defined(TOPSRT_VERSION) && TOPSRT_VERSION >= 4000
using TopsPointerAttributes = topsPointerAttributes;

inline topsError_t AllocatePinnedHost(void** ptr, size_t size) {
  return topsHostAlloc(ptr, size, topsHostAllocDefault);
}

inline topsError_t FreePinnedHost(void* ptr) {
  return topsFreeHost(ptr);
}
#else
using TopsPointerAttributes = topsPointerAttribute_t;

inline topsError_t AllocatePinnedHost(void** ptr, size_t size) {
  return topsHostMalloc(ptr, size, topsHostMallocDefault);
}

inline topsError_t FreePinnedHost(void* ptr) {
  return topsHostFree(ptr);
}
#endif

} // namespace c10::flagos::enflame
