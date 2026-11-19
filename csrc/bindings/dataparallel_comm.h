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

namespace torch_fl::comm {

/// Publishes the flagos implementations of torch._C's DataParallel
/// communication primitives. See dataparallel_comm.cc for the whole story.
///
/// Idempotent: the first call captures torch._C's originals and rebinds them,
/// every later call is a no-op. Rebinding twice would capture the previous
/// replacement as the "original" and recurse, so the guard is load bearing.
void InitDataParallelComm();

/// Marks this thread as scattering on behalf of a flagos-placed DataParallel.
/// Returns the value it replaced, so the caller can restore it in a `finally`.
///
/// It exists because a CPU input tensor carries no device type, and the integer
/// device ids DataParallel hands to comm.scatter are ambiguous exactly where
/// this layer is needed: on PPU "cuda" and "flagos" name the same silicon, so
/// [0, 1] could mean either list. Only the wrapper knows which one it is
/// driving, so the wrapper says so. Thread local, because two DataParallel
/// wrappers in two threads must not see each other's scope; the setter and the
/// reader are always the same thread, since the scatter is called synchronously
/// from DataParallel.forward.
bool SetScatterScope(bool active);

} // namespace torch_fl::comm
