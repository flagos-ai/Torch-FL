// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "../device_memory_interface.h"

namespace c10::flagos {

// Cambricon MLU implementation of DeviceMemoryInterface.
//
// Every operation is forwarded to the flagos runtime contract functions in
// runtime/accelerator/cambricon rather than to the raw cnrt entry points. That
// is what the other native backends do, and here it matters for one concrete
// reason: PointerGetAttributes in that layer answers from the driver
// (cnrtPointerGetAttributes), so a block allocated through the contract is
// recognised by hooks.h and by any later isPinnedPtr/pinned-memory decision. A
// direct cnrtMalloc would bypass the same bookkeeping path the rest of the
// backend uses.
//
// provides_caching() stays at its false default: the Neuware toolkit ships no
// caching allocator to delegate to, and torch_mlu's MLUCachingAllocator is not
// part of this build (torch_mlu is deliberately not installed), so flagos's own
// block pool is the only allocator in play.
class CambriconDeviceMemory final : public DeviceMemoryInterface {
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
    // cnrt offers no free-memory query; the only capacity information is
    // cnrtDeviceProp_t::totalMem, which is a per-model constant and says nothing
    // about what is in use. Reporting it as `total` beside a `free` this process
    // would have to invent would be a misleading pair, so both fields stay zero
    // -- the same answer, for the same reason, as the Kunlunxin and BPU
    // backends give. Nothing in the caching allocator reads these values for
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
    // and cnrtNotifierElapsedTime requires a device-side timing setup the
    // allocator never asks for.
    return ::EventCreateWithFlags(event, EventDisableTiming);
  }

  Error_t event_destroy(Event_t event) override {
    return ::EventDestroy(event);
  }

  Error_t event_record(Event_t event, Stream_t stream) override {
    return ::EventRecord(event, stream);
  }

  Error_t event_query(Event_t event) override {
    // ::EventQuery maps cnrtErrorNotReady to ErrorNotReady and everything else
    // that is not a success to ErrorUnknown. That distinction is exactly what
    // this call site needs: the caching allocator recycles a block on any status
    // other than ErrorNotReady, so "not ready" must not be reported as a
    // completion. See EventQuery in runtime/accelerator/cambricon/stream.cc.
    return ::EventQuery(event);
  }

  Error_t memcpy(
      void* dst,
      const void* src,
      size_t count,
      MemcpyKind kind) override {
    // Also covers MemcpyHostToHost, which cnrt expresses as its own direction
    // tag rather than inferring it from the pointers.
    return ::Memcpy(dst, src, count, kind);
  }
};

} // namespace c10::flagos