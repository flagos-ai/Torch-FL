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

// Eager primary-context creation for the Iluvatar (corex) runtime.
//
// corex only creates a device's primary context from `cudaFree(nullptr)` (the
// classic "cudaFree(0) to initialise" idiom). Every *other* runtime entry point
// that would do it implicitly -- `cudaSetDevice` followed by `cudaMalloc` in
// particular -- instead spins inside the driver and never returns, and the spin
// is machine-wide once it has started: `ixsmi` keeps answering, `cudaSetDevice`
// keeps returning cudaSuccess, and only context creation hangs. Measured on
// u101/u180 with a three-line C program:
//
//     cudaFree(0)                                  -> cudaSuccess, 0.05s
//     cudaGetDevice, cudaSetDevice(0), cudaMalloc  -> hangs
//     cudaFree(0), then set/malloc on devices 0-2  -> all cudaSuccess
//
// torch's first op on the device does not begin with `cudaFree(0)`: it begins
// with a DeviceGuard (`cudaGetDevice` + `cudaSetDevice`) and then the
// allocator's `cudaMalloc`, which is exactly the hanging order. So the very
// first operation on the device hung in every probe, and killing that process
// wedged the machine for everybody afterwards -- including plain C programs run
// on the host. Three fleet hosts (ub22, u180, u211) had to be given back to
// infra because of it.
//
// Calling `cudaFree(nullptr)` once, before any other runtime call in the
// process, is therefore not an optimisation: it is what makes the runtime
// usable at all. Every entry point that can be the process's first call invokes
// EnsureIluvatarContext(), so the order cannot depend on which one torch
// happens to reach first.
//
// The call is idempotent and context-free on a healthy runtime: after the first
// context exists `cudaFree(nullptr)` is a no-op that returns an error code,
// which is deliberately ignored -- a runtime that refuses it has already
// created the context some other way, and there is nothing left to do.

#pragma once

#include <cuda_runtime.h>

#include <mutex>

namespace c10::flagos::iluvatar {

inline void EnsureIluvatarContext() {
  static std::once_flag once;
  std::call_once(once, [] {
    // Nothing else may run before this: cudaGetDevice/cudaSetDevice are cheap
    // and do not create a context, but a context has to exist before
    // cudaMalloc, and cudaFree(nullptr) is the only call corex accepts for that
    // role (see the file comment).
    (void)cudaFree(nullptr);
  });
}

} // namespace c10::flagos::iluvatar
