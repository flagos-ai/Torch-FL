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

#include <acl/acl_base_rt.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace at::native::flagos::ascend {
namespace {

thread_local bool eager_worker = false;
thread_local std::unordered_map<int, aclrtStream> current_streams;
std::atomic<bool> eager_queue_created{false};

class EagerLaunchQueue {
 public:
  EagerLaunchQueue() : worker_([this] { Run(); }) {}
  ~EagerLaunchQueue() {
    {
      std::lock_guard<std::mutex> lock(ready_mutex_);
      exiting_ = true;
    }
    ready_.notify_one();
    worker_.join();
    eager_queue_created.store(false, std::memory_order_release);
  }

  void Submit(AscendAclnnCall call) {
    std::lock_guard<std::mutex> producer(producer_mutex_);
    const auto index = submitted_.load();
    WaitForProgress([&] { return index - consumed_.load() < kCapacity || failed_.load(); });
    CheckFailure();
    slots_[index % kCapacity] = std::move(call);
    submitted_.store(index + 1);
    if (waiting_for_work_.load()) {
      std::lock_guard<std::mutex> lock(ready_mutex_);
      ready_.notify_one();
    }
  }

  void WaitUntilSubmitted() {
    const auto target = submitted_.load();
    WaitForProgress([&] { return completed_.load() >= target; });
  }

  void Flush() {
    WaitUntilSubmitted();
    CheckFailure();
  }

 private:
  static constexpr size_t kCapacity = 512;
  using Task = AscendAclnnCall;

  template <class Ready>
  void WaitForProgress(Ready ready) {
    if (ready()) return;
    std::unique_lock<std::mutex> lock(progress_mutex_);
    ++progress_waiters_;
    progress_.wait(lock, ready);
    --progress_waiters_;
  }

  void NotifyProgress() {
    if (progress_waiters_.load() == 0) return;
    std::lock_guard<std::mutex> lock(progress_mutex_);
    progress_.notify_all();
  }

  void CheckFailure() {
    if (failed_.load()) std::rethrow_exception(failure_);
  }

  void Run() {
    eager_worker = true;
    int device = -1;
    for (uint64_t index = 0; ; ++index) {
      if (submitted_.load() == index) {
        std::unique_lock<std::mutex> lock(ready_mutex_);
        waiting_for_work_.store(true);
        ready_.wait(lock, [&] { return exiting_ || submitted_.load() != index; });
        waiting_for_work_.store(false);
        if (submitted_.load() == index) return;
      }
      Task task = std::move(slots_[index % kCapacity]);
      consumed_.store(index + 1);
      NotifyProgress();
      if (!failed_.load()) {
        try {
          if (device != task.device) {
            if (::SetDevice(task.device) != Success) {
              throw std::runtime_error("Unable to bind the Ascend submission worker");
            }
            device = task.device;
          }
          const int result = task.execute(task.workspace, task.workspace_size,
                                          task.executor, task.stream);
          if (result != 0) {
            throw std::runtime_error(std::string(task.api) + " failed: " +
                                     std::to_string(result));
          }
        } catch (...) {
          failure_ = std::current_exception();
          failed_.store(true);
        }
      }
      task.descriptors.reset();
      completed_.store(index + 1);
      NotifyProgress();
    }
  }

  std::array<Task, kCapacity> slots_;
  alignas(64) std::atomic<uint64_t> submitted_{0};
  alignas(64) std::atomic<uint64_t> consumed_{0};
  alignas(64) std::atomic<uint64_t> completed_{0};
  std::mutex producer_mutex_;
  std::mutex ready_mutex_;
  std::condition_variable ready_;
  bool exiting_ = false;
  std::atomic<bool> waiting_for_work_{false};
  std::mutex progress_mutex_;
  std::condition_variable progress_;
  std::atomic<size_t> progress_waiters_{0};
  std::atomic<bool> failed_{false};
  std::exception_ptr failure_;
  std::thread worker_;
};

EagerLaunchQueue& EagerQueue() {
  static EagerLaunchQueue queue;
  return queue;
}

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

FLAGOS_EXPORT bool AscendEagerQueueEnabled() {
  static const bool enabled = [] {
    const char* mode = std::getenv("TASK_QUEUE_ENABLE");
    const char* blocking = std::getenv("ASCEND_LAUNCH_BLOCKING");
    return (mode == nullptr || std::strcmp(mode, "0") != 0) &&
           (blocking == nullptr || std::strcmp(blocking, "1") != 0);
  }();
  return enabled;
}

FLAGOS_EXPORT void SubmitAscendEager(AscendAclnnCall call) {
  auto& queue = EagerQueue();
  eager_queue_created.store(true, std::memory_order_release);
  queue.Submit(std::move(call));
}

FLAGOS_EXPORT void FlushAscendEager() {
  if (!eager_worker && eager_queue_created.load(std::memory_order_acquire))
    EagerQueue().Flush();
}

FLAGOS_EXPORT void WaitForAscendEagerSubmissions() {
  if (!eager_worker && eager_queue_created.load(std::memory_order_acquire))
    EagerQueue().WaitUntilSubmitted();
}

FLAGOS_EXPORT void DrainDefaultAclStreams() {
  FlushAscendEager();
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
  FlushAscendEager();
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

FLAGOS_EXPORT std::string GetSocName() {
  // aclrtGetSocName initializes ACL and answers for whichever device is
  // current, so one has to be selected first. The result is cached: this is
  // queried on every matmul dispatch, and a host reports a single SoC for all
  // of its devices.
  static const std::string soc_name = []() {
    int device_index = 0;
    ::GetDevice(&device_index);
    const char* name = aclrtGetSocName();
    return name == nullptr ? std::string() : std::string(name);
  }();
  return soc_name;
}

} // namespace at::native::flagos::ascend

extern "C" {

__attribute__((visibility("default")))
void* GetCurrentStream(int device_index) {
  at::native::flagos::ascend::WaitForAscendEagerSubmissions();
  return (void*)at::native::flagos::ascend::GetCurrentAclStreamForDevice(
      device_index);
}

__attribute__((visibility("default")))
void SetCurrentStream(int device_index, void* stream) {
  at::native::flagos::ascend::SetCurrentAclStreamForDevice(
      device_index, (aclrtStream)stream);
}

} // extern "C"

extern "C" FLAGOS_EXPORT void FlagosAscendDrainHostSubmissions() {
  at::native::flagos::ascend::FlushAscendEager();
}

extern "C" FLAGOS_EXPORT int FlagosAscendFlushHostSubmissions() noexcept {
  try {
    at::native::flagos::ascend::FlushAscendEager();
    return Success;
  } catch (...) {
    return ErrorUnknown;
  }
}

#endif // USE_ASCEND
