// Copyright (c) 2026, BAAI. All rights reserved.

#include <flagos.h>

#include <algorithm>
#include <array>
#include <mutex>
#include <unordered_map>

#include "musa_stream.h"

namespace {

struct StreamPool {
  std::array<musaStream_t, 32> queues{};
  size_t next = 0;
};

struct DeviceStreams {
  musaStream_t default_stream = nullptr;
  std::unordered_map<int, StreamPool> pools;
};

// libflagos owns the streams shared by the bindings and operator libraries.
// Selection is per thread and per device.
std::mutex stream_mutex;
std::unordered_map<int, DeviceStreams> streams;
thread_local std::unordered_map<int, Stream_t> current_streams;

class ScopedDevice {
 public:
  explicit ScopedDevice(int device) : device_(device) {
    valid_ = musaGetDevice(&previous_) == musaSuccess &&
        (previous_ == device || musaSetDevice(device) == musaSuccess);
  }

  ~ScopedDevice() {
    if (valid_ && previous_ != device_) musaSetDevice(previous_);
  }

  explicit operator bool() const { return valid_; }

 private:
  int previous_ = -1;
  int device_;
  bool valid_;
};

}  // namespace

Stream_t GetDefaultStreamForDevice(int device) {
  std::lock_guard<std::mutex> lock(stream_mutex);
  auto it = streams.find(device);
  if (it != streams.end() && it->second.default_stream)
    return reinterpret_cast<Stream_t>(it->second.default_stream);
  ScopedDevice guard(device);
  musaStream_t stream = nullptr;
  if (!guard || musaStreamCreate(&stream) != musaSuccess) return nullptr;
  streams[device].default_stream = stream;
  return reinterpret_cast<Stream_t>(stream);
}

Stream_t GetCurrentStreamForDevice(int device) {
  auto it = current_streams.find(device);
  return it == current_streams.end() ? GetDefaultStreamForDevice(device) : it->second;
}

Error_t SetCurrentStreamForDevice(int device, Stream_t stream) {
  if (stream) current_streams[device] = stream;
  else current_streams.erase(device);
  return Success;
}

namespace at::native::flagos::musa {

musaStream_t GetCurrentMusaStream() {
  int device = 0;
  if (musaGetDevice(&device) != musaSuccess) return nullptr;
  return reinterpret_cast<musaStream_t>(GetCurrentStreamForDevice(device));
}

musaStream_t GetMusaStreamFromPool(int device, int priority) {
  ScopedDevice guard(device);
  if (!guard) return nullptr;
  int least = 0;
  int greatest = 0;
  priority = musaDeviceGetStreamPriorityRange(&least, &greatest) == musaSuccess
      ? std::clamp(priority, greatest, least)
      : 0;
  std::lock_guard<std::mutex> lock(stream_mutex);
  auto& pool = streams[device].pools[priority];
  auto& stream = pool.queues[pool.next];
  if (!stream && musaStreamCreateWithPriority(&stream, musaStreamNonBlocking, priority) != musaSuccess)
    return nullptr;
  pool.next = (pool.next + 1) % pool.queues.size();
  return stream;
}

}  // namespace at::native::flagos::musa
