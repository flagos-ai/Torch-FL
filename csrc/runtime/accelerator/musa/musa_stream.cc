// Copyright (c) 2026, BAAI. All rights reserved.
//
// The one definition of the MUSA default stream. This file is compiled into
// libflagos.so -- the accelerator runtime library, built from
// csrc/runtime/accelerator/musa/*.cc -- rather than into libtorch_fl.so or
// libtorch_bindings.so, because those are two separate shared objects and a
// function-local static inside an `inline` function gives each of them its own
// copy. See musa_stream.h for the measurements that made this the headline bug
// it is: two independently created "default" streams meant the mudnn ops and the
// Triton kernels were never ordered against each other.
//
// libflagos.so is the right owner because every other MUSA shared object links
// it, directly (libtorch_fl.so) or through it (libtorch_bindings.so, i.e.
// torch_fl._C), so all of them resolve this single symbol.
//
// musa_stream.h is gated on USE_MUSA, which this target does not define -- it
// compiles the plain C runtime and has no reason to know about the backend's
// compile-time flags. Defining it around this one include keeps the declaration
// and the definition in the same translation unit, so a signature drift between
// them is a compile error instead of a second silently distinct symbol.

#include <musa_runtime.h>

#include <mutex>
#include <unordered_map>

#define USE_MUSA 1
#include "musa_stream.h"

namespace at::native::flagos::musa {

musaStream_t GetDefaultMusaStream() {
  static std::mutex mutex;
  static std::unordered_map<int, musaStream_t> streams;

  int device = 0;
  if (musaGetDevice(&device) != musaSuccess) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex);
  auto it = streams.find(device);
  if (it != streams.end()) {
    return it->second;
  }
  musaStream_t stream = nullptr;
  if (musaStreamCreate(&stream) != musaSuccess) {
    // Fall back to the null (legacy default) stream; ops still execute, just
    // without a dedicated queue.
    stream = nullptr;
  }
  streams.emplace(device, stream);
  return stream;
}

} // namespace at::native::flagos::musa
