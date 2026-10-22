// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "../device_memory_interface.h"

#include <tops_runtime_api.h>
#include <cstdio>

namespace c10::flagos {

namespace {

// The tops runtime resolves a device pointer only against the *current* device;
// touching memory owned by another device fails with "not allocated in current
// device". Switch to the owning device for the operation and restore after.
class TopsPointerDeviceGuard {
 public:
  explicit TopsPointerDeviceGuard(const void* dev_ptr) {
    if (!dev_ptr)
      return;
    topsPointerAttribute_t attr{};
    if (topsPointerGetAttributes(&attr, dev_ptr) != topsSuccess)
      return;
    if (attr.device < 0)
      return;
    int current = -1;
    if (topsGetDevice(&current) != topsSuccess || current == attr.device)
      return;
    if (topsSetDevice(attr.device) != topsSuccess)
      return;
    prev_device_ = current;
  }

  ~TopsPointerDeviceGuard() {
    if (prev_device_ >= 0)
      topsSetDevice(prev_device_);
  }

  TopsPointerDeviceGuard(const TopsPointerDeviceGuard&) = delete;
  TopsPointerDeviceGuard& operator=(const TopsPointerDeviceGuard&) = delete;

 private:
  int prev_device_ = -1;
};

// Reports the two cards involved when this is a copy between different cards,
// and returns false for a host-side copy, a copy inside one card, or a pointer
// the runtime does not recognise.
//
// A cross-card copy has to go through `topsMemcpyPeer`. `topsMemcpy` with
// `topsMemcpyDeviceToDevice` returns an error and leaves the destination
// untouched for every candidate current device, so routing one through it moves
// no bytes and reports nothing to a caller that ignores the status.
// csrc/runtime/accelerator/gcu/memory.cc carries the measurement and the
// reasoning; this is the same fix on the DeviceMemoryInterface path.
bool TopsCrossDeviceEnds(
    const void* dst,
    const void* src,
    int* dst_device,
    int* src_device) {
  topsPointerAttribute_t dst_attr{};
  topsPointerAttribute_t src_attr{};
  if (topsPointerGetAttributes(&dst_attr, dst) != topsSuccess)
    return false;
  if (topsPointerGetAttributes(&src_attr, src) != topsSuccess)
    return false;
  if (dst_attr.device < 0 || src_attr.device < 0)
    return false;
  if (dst_attr.device == src_attr.device)
    return false;
  *dst_device = dst_attr.device;
  *src_device = src_attr.device;
  return true;
}

} // namespace

// Enflame GCU (tops runtime) implementation of DeviceMemoryInterface.
// Directly calls tops runtime APIs for raw memory operations.
class GcuDeviceMemory final : public DeviceMemoryInterface {
 public:
  Error_t device_malloc(void** ptr, size_t size) override {
    topsError_t err = topsMalloc(ptr, size);
    if (err != topsSuccess) {
      fprintf(stderr, "[flagos-gcu] topsMalloc(%zu bytes) failed: %s\n",
              size, topsGetErrorString(err));
      *ptr = nullptr;
      return ErrorMemoryAllocation;
    }
    return Success;
  }

  Error_t device_free(void* ptr) override {
    TopsPointerDeviceGuard guard(ptr);
    topsError_t err = topsFree(ptr);
    if (err != topsSuccess) {
      fprintf(stderr, "[flagos-gcu] topsFree(%p) failed: %s\n",
              ptr, topsGetErrorString(err));
      return ErrorUnknown;
    }
    return Success;
  }

  Error_t get_device_index(int* device) override {
    topsError_t err = topsGetDevice(device);
    return (err == topsSuccess) ? Success : ErrorUnknown;
  }

  Error_t set_device(int device) override {
    topsError_t err = topsSetDevice(device);
    return (err == topsSuccess) ? Success : ErrorInvalidDevice;
  }

  Error_t get_memory_info(size_t* free, size_t* total) override {
    topsError_t err = topsMemGetInfo(free, total);
    return (err == topsSuccess) ? Success : ErrorUnknown;
  }

  Error_t event_create(Event_t* event) override {
    topsError_t err = topsEventCreate(reinterpret_cast<topsEvent_t*>(event));
    return (err == topsSuccess) ? Success : ErrorUnknown;
  }

  Error_t event_destroy(Event_t event) override {
    topsError_t err = topsEventDestroy(reinterpret_cast<topsEvent_t>(event));
    return (err == topsSuccess) ? Success : ErrorUnknown;
  }

  Error_t event_record(Event_t event, Stream_t stream) override {
    topsError_t err = topsEventRecord(
        reinterpret_cast<topsEvent_t>(event),
        reinterpret_cast<topsStream_t>(stream));
    return (err == topsSuccess) ? Success : ErrorUnknown;
  }

  Error_t event_query(Event_t event) override {
    topsError_t err = topsEventQuery(reinterpret_cast<topsEvent_t>(event));
    if (err == topsSuccess) {
      return Success;
    } else if (err == topsErrorNotReady) {
      return ErrorNotReady;
    }
    return ErrorUnknown;
  }

  Error_t memcpy(
      void* dst,
      const void* src,
      size_t count,
      MemcpyKind kind) override {
    int dst_device = -1;
    int src_device = -1;
    if (kind == MemcpyDeviceToDevice &&
        TopsCrossDeviceEnds(dst, src, &dst_device, &src_device)) {
      TopsPointerDeviceGuard guard(src);
      topsError_t err = topsMemcpyPeer(dst, dst_device, src, src_device, count);
      return (err == topsSuccess) ? Success : ErrorUnknown;
    }

    topsMemcpyKind tops_kind;
    const void* dev_ptr = nullptr;
    switch (kind) {
      case MemcpyHostToHost:
        tops_kind = topsMemcpyHostToHost;
        break;
      case MemcpyHostToDevice:
        tops_kind = topsMemcpyHostToDevice;
        dev_ptr = dst;
        break;
      case MemcpyDeviceToHost:
        tops_kind = topsMemcpyDeviceToHost;
        dev_ptr = src;
        break;
      case MemcpyDeviceToDevice:
        tops_kind = topsMemcpyDeviceToDevice;
        dev_ptr = src;
        break;
      default:
        return ErrorUnknown;
    }
    TopsPointerDeviceGuard guard(dev_ptr);
    topsError_t err = topsMemcpy(dst, src, count, tops_kind);
    return (err == topsSuccess) ? Success : ErrorUnknown;
  }
};

} // namespace c10::flagos
