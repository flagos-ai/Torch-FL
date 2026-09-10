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

// Per-device "current stream" tracking for DCU.
//
// On CUDA proper, csrc/runtime/guard.h forwards getStream()/exchangeStream() to
// c10::cuda::getCurrentCUDAStream/setCurrentCUDAStream, so torch owns the notion
// of a current stream and flagos never needs its own. DCU cannot do that: the
// DTK wheel is hipified and exports c10::hip:: symbols with zero c10::cuda ones,
// so FLAGOS_GUARD_HAS_CUDA_STREAM is 0 there and the guard's fallback returned a
// stream id of 0 for every device unconditionally.
//
// A constant id 0 is harmless as long as nothing reads it -- the boxing path
// lands inside libtorch_hip.so, which consults torch's own HIP current stream --
// but the SDK-native operators are outside torch entirely and have no other way
// to learn which stream to submit on. Without a registry, rocblas_set_stream
// would always get the null (default) stream and any user code inside a
// `with torch.Stream(...)` block would silently run on the wrong stream.
//
// Semantics deliberately mirror c10::cuda's:
//   * thread-local, because CUDA's current stream is thread-local and code that
//     sets a stream on one thread must not perturb another;
//   * nullptr means the device's default (legacy) stream, which is what
//     GetDefaultStreamForDevice returns;
//   * the handle is a raw device stream pointer, matching the Stream_t
//     convention used everywhere else in this ABI (see the reinterpret_cast in
//     csrc/aten/register.cc's WrapperRecordStream).
//
// Built only for ACCELERATOR=dcu. The CUDA build intentionally leaves these
// three entry points undefined: nothing on that platform calls them, and
// defining them would create a second, competing source of truth alongside
// c10::cuda's.

#include <flagos.h>
#include <cuda_runtime.h>

#include <cstdio>

namespace {

// Matches DTK's supported device count ceiling with room to spare. A fixed array
// keeps the lookup a plain load with no allocation and no lock, which matters
// because every SDK GEMM reads it.
constexpr int kMaxDevices = 64;

// nullptr entry == "use the default stream", so zero-initialization is already
// the correct initial state and no lazy setup is needed.
thread_local Stream_t t_current_streams[kMaxDevices] = {};

// Resolve a possibly-negative device index the way the rest of the runtime does:
// a negative index means "whatever is current".
int ResolveDevice(int device) {
  if (device >= 0) {
    return device;
  }
  int current = 0;
  if (cudaGetDevice(&current) != cudaSuccess) {
    return 0;
  }
  return current;
}

} // namespace

Stream_t GetDefaultStreamForDevice(int device) {
  (void)device;
  // The legacy default stream is spelled as the null handle by both the CUDA and
  // HIP runtimes, so there is nothing per-device to look up.
  return nullptr;
}

Stream_t GetCurrentStreamForDevice(int device) {
  int resolved = ResolveDevice(device);
  if (resolved < 0 || resolved >= kMaxDevices) {
    return nullptr;
  }
  return t_current_streams[resolved];
}

Error_t SetCurrentStreamForDevice(int device, Stream_t stream) {
  int resolved = ResolveDevice(device);
  if (resolved < 0 || resolved >= kMaxDevices) {
    fprintf(
        stderr,
        "[flagos] SetCurrentStreamForDevice: device %d out of range [0, %d)\n",
        resolved,
        kMaxDevices);
    return ErrorInvalidDevice;
  }
  t_current_streams[resolved] = stream;
  return Success;
}
