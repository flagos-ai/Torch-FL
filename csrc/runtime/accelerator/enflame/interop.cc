// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "interop.h"
#include <tops_runtime_api.h>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {
std::atomic<uint32_t> enabled{0};
std::recursive_mutex mutex;
struct Producer {
  uint32_t id = 0;
  topsEvent_t event = nullptr;
};
std::unordered_map<void*, Producer> producers;
thread_local std::vector<bool> submissions;
}

int FlagosEnflameEnableInterop(uint32_t abi) {
  if (abi != 1 && abi != 2) return topsErrorInvalidValue;
  std::lock_guard<std::recursive_mutex> lock(mutex);
  const auto current = enabled.load(std::memory_order_acquire);
  if (current) return current == abi ? topsSuccess : topsErrorInvalidValue;
  auto status = topsDeviceSynchronize();
  if (status != topsSuccess) return status;
  producers.clear();
  enabled.store(abi, std::memory_order_release);
  return topsSuccess;
}

int FlagosEnflameBeginLaunch(void* stream, uint32_t producer) {
  if (producer != 1 && producer != 2) return topsErrorInvalidValue;
  const auto abi = enabled.load(std::memory_order_acquire);
  if (!abi) {
    auto status = topsStreamSynchronize(reinterpret_cast<topsStream_t>(stream));
    if (status == topsSuccess) submissions.push_back(false);
    return status;
  }
  mutex.lock();
  auto& previous = producers[stream];
  if (previous.id && previous.id != producer) {
    const auto handle = reinterpret_cast<topsStream_t>(stream);
    auto status = topsSuccess;
    if (abi == 2) {
      if (!previous.event)
        status = topsEventCreateWithFlags(&previous.event, topsEventDisableTiming);
      if (status == topsSuccess) status = topsEventRecord(previous.event, handle);
      if (status == topsSuccess) status = topsStreamWaitEvent(handle, previous.event, 0);
    } else {
      status = topsStreamSynchronize(handle);
    }
    if (status != topsSuccess) {
      mutex.unlock();
      return status;
    }
  }
  previous.id = producer;
  submissions.push_back(true);
  return topsSuccess;
}

int FlagosEnflameEndLaunch(void* stream) {
  if (submissions.empty()) return topsErrorInvalidValue;
  bool tracked = submissions.back();
  submissions.pop_back();
  if (tracked) {
    mutex.unlock();
    return topsSuccess;
  }
  return topsStreamSynchronize(reinterpret_cast<topsStream_t>(stream));
}

void FlagosEnflameForgetStream(void* stream) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  auto found = producers.find(stream);
  if (found != producers.end()) {
    if (found->second.event) topsEventDestroy(found->second.event);
    producers.erase(found);
  }
}
