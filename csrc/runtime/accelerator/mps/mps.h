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

// Apple Metal runtime extensions beyond the flagos C ABI.
//
// The ABI in csrc/include/flagos.h has no capacity query, but Metal does
// (MTLDevice.recommendedMaxWorkingSetSize / currentAllocatedSize), so the
// allocator's get_memory_info reports real numbers instead of the zeros a
// backend with no query at all has to settle for (see bpu_memory.h). Declaring
// it here rather than widening the flagos ABI keeps the cross-platform contract
// unchanged; only the allocator (csrc/) and this runtime ever name it.

#pragma once

#include <stddef.h>

#include <flagos.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bytes the Metal device can still hand out and its recommended working-set
// ceiling. Either pointer may be NULL, in which case that value is skipped.
// Reports zeros when no MPS-capable device is present.
FLAGOS_EXPORT void FlagosMPSDeviceMemoryInfo(size_t* freeBytes, size_t* totalBytes);

#ifdef __cplusplus
} // extern "C"
#endif
