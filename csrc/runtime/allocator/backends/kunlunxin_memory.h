// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "../device_memory_interface.h"

namespace c10::flagos {

// Kunlunxin P800 XPU implementation of DeviceMemoryInterface.
//
// Every operation is forwarded to the flagos runtime contract functions in
// runtime/accelerator/kunlunxin rather than to the raw xpu_* entry points. That
// is deliberate: the accelerator layer keeps the block registry that answers
// PointerGetAttributes, since the SDK's own xpu_pointer_get_attrs /
// xpu_host_pointer_get_attrs are declared but not exported. Allocating through
// the contract keeps caching-allocator blocks in that same registry, which a
// direct xpu_malloc would bypass.
//
// provides_caching() stays at its false default: the XPU SDK ships no caching
// allocator of its own to delegate to, so flagos's own block pool is the only
// allocator in play.
class KunlunxinDeviceMemory final : public DeviceMemoryInterface {
 public:
  Error_t device_malloc(void** ptr, size_t size) override {
    return ::Malloc(ptr, size);
  }

  Error_t device_free(void* ptr) override {
    return ::Free(ptr);
  }

  Error_t get_device_index(int* device) override {
    return ::GetDevice(device);
  }

  Error_t set_device(int device) override {
    return ::SetDevice(device);
  }

  Error_t get_memory_info(size_t* free, size_t* total) override {
    // The XPU SDK has no free-memory query of any kind. The only capacity API is
    // xpu_device_get_attr(&v, XPUATTR_MEM_MAIN_CAPACITY, devid), which reports
    // the card's total main memory (a fixed per-model constant) and says
    // nothing about what is currently in use. Reporting that as `total` next to
    // a `free` this process would have to invent would be a misleading pair, so
    // both fields stay zero -- the same answer, for the same reason, as the BPU
    // backend gives. Nothing in the caching allocator reads these values for
    // allocation decisions; they feed stats only.
    if (free) {
      *free = 0;
    }
    if (total) {
      *total = 0;
    }
    return Success;
  }

  Error_t event_create(Event_t* event) override {
    // Disable timing: these events only order work and guard deferred frees,
    // and the XPU SDK exposes no timestamp query anyway.
    return ::EventCreateWithFlags(event, EventDisableTiming);
  }

  Error_t event_destroy(Event_t event) override {
    return ::EventDestroy(event);
  }

  Error_t event_record(Event_t event, Stream_t stream) override {
    return ::EventRecord(event, stream);
  }

  Error_t event_query(Event_t event) override {
    // ::EventQuery reports ErrorNotReady for every unresolved non-zero
    // xpu_event_query result (a completed event reports 0). That is exactly
    // what this call site needs: the caching allocator recycles a block on any
    // status other than ErrorNotReady, so a "not ready" code it does not
    // recognize must never be reported as a completion. See EventQuery in
    // runtime/accelerator/kunlunxin/stream.cc.
    return ::EventQuery(event);
  }

  Error_t memcpy(
      void* dst,
      const void* src,
      size_t count,
      MemcpyKind kind) override {
    // Also covers MemcpyHostToHost, which the XPU runtime cannot express, and
    // drains the default streams before a transfer that touches device memory.
    return ::Memcpy(dst, src, count, kind);
  }
};

} // namespace c10::flagos
