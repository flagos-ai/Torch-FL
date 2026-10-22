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

// BPU-specific extensions beyond the flagos runtime contract.

#pragma once

#include <stdint.h>

#include <macros.h>

#ifdef __cplusplus
extern "C" {
#endif

// Physical address backing a device pointer returned by Malloc, or 0 if the
// pointer is not BPU device memory.
//
// The BPU addresses tensors by physical address (hbUCPSysMem.phyAddr) while
// torch only tracks the host virtual pointer, so bridging the two is what lets
// the compiled .hbm read a torch tensor's storage in place instead of going
// through a numpy copy at each partition boundary.
FLAGOS_EXPORT uint64_t FlagosBPUPhysicalAddress(const void* ptr);

#ifdef __cplusplus
} // extern "C"
#endif
