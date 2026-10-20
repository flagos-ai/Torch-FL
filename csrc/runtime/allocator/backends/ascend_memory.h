// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "../device_memory_interface.h"

#include <acl/acl_rt.h>

namespace c10::flagos {

// Ascend implementation of DeviceMemoryInterface.
// Wraps the CANN ACL runtime for raw device memory operations.
//
// Unlike CUDA, Ascend has no mature drop-in caching allocator to delegate to,
// so provides_caching() stays false (the default) and CachingDeviceAllocator
// uses its own built-in block pool on top of these raw device_malloc/free ops.
class AscendDeviceMemory final : public DeviceMemoryInterface {
 public:
  Error_t device_malloc(void** ptr, size_t size) override {
    // The accelerator contract initializes ACL and selects device 0 on first
    // use; a fresh ACL context has no current device to read directly.
    int device = 0;
    if (::GetDevice(&device) != Success || ::SetDevice(device) != Success) {
      *ptr = nullptr;
      return ErrorMemoryAllocation;
    }
    aclError err = aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_SUCCESS || *ptr == nullptr) {
      *ptr = nullptr;
      return ErrorMemoryAllocation;
    }
    return Success;
  }

  Error_t device_free(void* ptr) override {
    aclError err = aclrtFree(ptr);
    return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
  }

  Error_t get_device_index(int* device) override {
    // The caching allocator relies on this probe to establish the initial
    // device. GetDevice's default (0) and SetDevice's validation share the
    // gCurrentDevice state, so a raw ACL lookup would fail before that state
    // has been initialized.
    int current = 0;
    Error_t err = ::GetDevice(&current);
    if (err != Success) {
      return err;
    }
    err = ::SetDevice(current);
    if (err != Success) {
      return err;
    }
    *device = current;
    return Success;
  }

  Error_t set_device(int device) override {
    return ::SetDevice(device);
  }

  Error_t get_memory_info(size_t* free, size_t* total) override {
    aclError err = aclrtGetMemInfo(ACL_HBM_MEM, free, total);
    return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
  }

  Error_t event_create(Event_t* event) override {
    aclrtEvent acl_event = nullptr;
    // Disable timing to match CUDA's cudaEventDisableTiming — these events are
    // used only for stream-ordering / deferred-free safety.
    aclError err = aclrtCreateEventWithFlag(&acl_event, ACL_EVENT_SYNC);
    if (err != ACL_SUCCESS) {
      return ErrorUnknown;
    }
    *event = reinterpret_cast<Event_t>(acl_event);
    return Success;
  }

  Error_t event_destroy(Event_t event) override {
    aclError err = aclrtDestroyEvent(reinterpret_cast<aclrtEvent>(event));
    return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
  }

  Error_t event_record(Event_t event, Stream_t stream) override {
    aclError err = aclrtRecordEvent(
        reinterpret_cast<aclrtEvent>(event),
        reinterpret_cast<aclrtStream>(stream));
    return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
  }

  Error_t event_query(Event_t event) override {
    aclrtEventRecordedStatus status;
    aclError err = aclrtQueryEventStatus(
        reinterpret_cast<aclrtEvent>(event), &status);
    if (err != ACL_SUCCESS) {
      return ErrorUnknown;
    }
    return (status == ACL_EVENT_RECORDED_STATUS_COMPLETE) ? Success
                                                          : ErrorNotReady;
  }

  Error_t memcpy(void* dst, const void* src, size_t count, MemcpyKind kind)
      override {
    aclrtMemcpyKind acl_kind;
    switch (kind) {
      case MemcpyHostToHost:
        acl_kind = ACL_MEMCPY_HOST_TO_HOST;
        break;
      case MemcpyHostToDevice:
        acl_kind = ACL_MEMCPY_HOST_TO_DEVICE;
        break;
      case MemcpyDeviceToHost:
        acl_kind = ACL_MEMCPY_DEVICE_TO_HOST;
        break;
      case MemcpyDeviceToDevice:
        acl_kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
        break;
      default:
        return ErrorUnknown;
    }
    aclError err = aclrtMemcpy(dst, count, src, count, acl_kind);
    return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
  }
};

} // namespace c10::flagos
