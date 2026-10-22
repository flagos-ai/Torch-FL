// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "../device_memory_interface.h"

#include <cstdio>

namespace c10::flagos {

// D-Robotics RDK BPU (Horizon BPU) implementation of DeviceMemoryInterface.
//
// Unlike the other backends this one does not call a vendor runtime directly:
// the flagos contract functions in csrc/runtime/accelerator/bpu/ already wrap
// the UCP allocator and own the virtual/physical address registry that the
// zero-copy inference path reads. Going through them keeps a single source of
// truth for which pointers are device memory -- a second, independent
// registry here would disagree the moment either side changed.
class BPUDeviceMemory final : public DeviceMemoryInterface {
 public:
  Error_t device_malloc(void** ptr, size_t size) override {
    Error_t err = Malloc(ptr, size);
    if (err != Success) {
      fprintf(
          stderr, "[flagos-bpu] device_malloc(%zu bytes) failed\n", size);
      *ptr = nullptr;
    }
    return err;
  }

  Error_t device_free(void* ptr) override {
    return Free(ptr);
  }

  Error_t get_device_index(int* device) override {
    return GetDevice(device);
  }

  Error_t set_device(int device) override {
    return SetDevice(device);
  }

  Error_t get_memory_info(size_t* free, size_t* total) override {
    // The UCP allocator exposes no capacity query, and BPU memory is carved out
    // of system DRAM by the ION/CMA pool rather than being a separate device
    // heap. Reporting zeroes is the honest answer; the caching allocator only
    // uses this for stats, not for allocation decisions.
    if (free) {
      *free = 0;
    }
    if (total) {
      *total = 0;
    }
    return Success;
  }

  Error_t event_create(Event_t* event) override {
    return EventCreate(event);
  }

  Error_t event_destroy(Event_t event) override {
    return EventDestroy(event);
  }

  Error_t event_record(Event_t event, Stream_t stream) override {
    return EventRecord(event, stream);
  }

  Error_t event_query(Event_t event) override {
    return EventQuery(event);
  }

  Error_t memcpy(void* dst, const void* src, size_t count, MemcpyKind kind)
      override {
    return Memcpy(dst, src, count, kind);
  }
};

} // namespace c10::flagos
