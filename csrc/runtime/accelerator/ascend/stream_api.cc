// Copyright (c) 2026, BAAI. All rights reserved.
//
// Public C API for FlagGems to obtain the current ACL stream managed by torch_fl.
// FlagGems calls this from backend_utils.h::getRawStream() to get the aclrtStream
// for passing to TritonJIT kernels.
//
// Only compiled on Ascend builds (USE_ASCEND=1).

#ifdef USE_ASCEND

#include "acl_stream.h"

#include <flagos.h>

#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace at::native::flagos::ascend {
namespace {

thread_local std::unordered_map<int, aclrtStream> current_streams;

// One default stream per device, plus the lock guarding the map. See
// acl_stream.h for why a single process-wide stream is not enough on a
// multi-device host.
std::mutex default_streams_mutex;
std::unordered_map<int, aclrtStream> default_streams;

// Devices whose aclrtCreateStream has already failed, so the diagnostic below
// is printed once per device rather than once per op.
std::unordered_map<int, bool> default_stream_failures;

} // namespace

// Single process-wide definition of the shared default ACL stream. Declared in
// acl_stream.h with default visibility so every shared object (libflagos.so and
// libtorch_fl.so) resolves to THIS one instance.
FLAGOS_EXPORT aclrtStream GetDefaultAclStreamForDevice(int device_index) {
  std::lock_guard<std::mutex> lock(default_streams_mutex);
  auto it = default_streams.find(device_index);
  if (it != default_streams.end()) {
    return it->second;
  }

  // aclrtCreateStream binds the stream to whichever device is current, so the
  // requested device has to be made current around the call -- and the
  // caller's device restored afterwards, since this is reached from deep
  // inside dispatch on the calling thread.
  int previous_device = -1;
  bool restore = ::GetDevice(&previous_device) == Success &&
                 previous_device != device_index;
  if (restore) {
    ::SetDevice(device_index);
  }

  aclrtStream stream = nullptr;
  aclError err = aclrtCreateStream(&stream);
  if (err != ACL_SUCCESS) {
    stream = nullptr;
    // Do not cache a failure: a transient one (a device still busy at init)
    // recovers on the next op, while caching nullptr would pin the process to
    // an unordered stream for the rest of its life -- DrainDefaultAclStreams
    // skips null entries, so nothing would ever drain the work these ops left
    // on the runtime's default stream. Print once per device either way.
    if (!default_stream_failures[device_index]) {
      default_stream_failures[device_index] = true;
      fprintf(stderr,
              "[flagos-ascend] aclrtCreateStream on device %d failed: %d; "
              "running on the runtime default stream, which is not drained\n",
              device_index, static_cast<int>(err));
    }
  } else {
    default_streams[device_index] = stream;
  }

  if (restore) {
    ::SetDevice(previous_device);
  }

  return stream;
}

FLAGOS_EXPORT aclrtStream GetDefaultAclStream() {
  // ::GetDevice() reads the cached current device (device.cc's gCurrentDevice)
  // instead of round-tripping through aclrtGetDevice. This sits on the
  // dispatch path of every aclnn op, so the runtime call is not affordable
  // here.
  int device_index = 0;
  ::GetDevice(&device_index);
  return GetDefaultAclStreamForDevice(device_index);
}

FLAGOS_EXPORT void DrainDefaultAclStreams() {
  std::vector<std::pair<int, aclrtStream>> snapshot;
  {
    std::lock_guard<std::mutex> lock(default_streams_mutex);
    snapshot.reserve(default_streams.size());
    for (const auto& entry : default_streams) {
      if (entry.second != nullptr) {
        snapshot.emplace_back(entry.first, entry.second);
      }
    }
  }

  // aclrtSynchronizeStream is resolved against the current device's context,
  // so each stream is synchronized with its own device made current.
  int previous_device = -1;
  ::GetDevice(&previous_device);
  for (const auto& entry : snapshot) {
    if (entry.first != previous_device) {
      ::SetDevice(entry.first);
    }
    aclrtSynchronizeStream(entry.second);
    if (entry.first != previous_device) {
      ::SetDevice(previous_device);
    }
  }
}

FLAGOS_EXPORT aclrtStream GetCurrentAclStreamForDevice(int device_index) {
  auto it = current_streams.find(device_index);
  if (it != current_streams.end() && it->second != nullptr) {
    return it->second;
  }
  return GetDefaultAclStreamForDevice(device_index);
}

FLAGOS_EXPORT aclrtStream GetCurrentAclStream() {
  int device_index = 0;
  ::GetDevice(&device_index);
  return GetCurrentAclStreamForDevice(device_index);
}

FLAGOS_EXPORT void SetCurrentAclStreamForDevice(
    int device_index,
    aclrtStream stream) {
  if (stream == nullptr || stream == GetDefaultAclStreamForDevice(device_index)) {
    current_streams.erase(device_index);
  } else {
    current_streams[device_index] = stream;
  }
}

FLAGOS_EXPORT void SetCurrentAclStream(aclrtStream stream) {
  int device_index = 0;
  ::GetDevice(&device_index);
  SetCurrentAclStreamForDevice(device_index, stream);
}

} // namespace at::native::flagos::ascend

extern "C" {

__attribute__((visibility("default")))
void* GetCurrentStream(int device_index) {
  return (void*)at::native::flagos::ascend::GetCurrentAclStreamForDevice(
      device_index);
}

__attribute__((visibility("default")))
void SetCurrentStream(int device_index, void* stream) {
  at::native::flagos::ascend::SetCurrentAclStreamForDevice(
      device_index, (aclrtStream)stream);
}

} // extern "C"

#endif // USE_ASCEND
