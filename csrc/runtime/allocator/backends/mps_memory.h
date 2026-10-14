// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "../device_memory_interface.h"

#include <cstdio>

// The non-contract memory-info helper lives with the Metal runtime. Both the
// runtime sources and this header build into the same wheel, and the helper is
// an extern "C" symbol in libflagos.so, which torch_fl links; the relative path
// avoids depending on the include-directory order of the two CMake targets.
#include "../../accelerator/mps/mps.h"

namespace c10::flagos {

// Apple Silicon (M-series) Metal implementation of DeviceMemoryInterface.
//
// Like the BPU backend this delegates to the flagos contract functions in
// csrc/runtime/accelerator/mps/ rather than calling Metal directly: those own
// the side table that says which pointers are device memory, and a second,
// independent registry here would disagree the moment either side changed.
//
// provides_caching() is left at the default (false). Metal ships no allocator
// that claims the PrivateUse1 slot flagos registers, but there is also none
// worth delegating to -- the built-in block pool is the right owner of an
// MTLBuffer's lifetime here, and MTLResourceStorageModeShared buffers are cheap
// to create.
class MpsDeviceMemory final : public DeviceMemoryInterface {
 public:
  Error_t device_malloc(void** ptr, size_t size) override {
    Error_t err = Malloc(ptr, size);
    if (err != Success) {
      fprintf(
          stderr, "[flagos-mps] device_malloc(%zu bytes) failed\n", size);
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
    // Metal reports a recommended working-set ceiling and a running total of
    // created resource bytes; the difference is a conservative "free". Reporting
    // it beats the zeros a backend with no query at all must settle for, and the
    // caching allocator only uses this for stats, not allocation decisions.
    FlagosMPSDeviceMemoryInfo(free, total);
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
    // Must return ErrorNotReady (not a generic error) while the recorded signal
    // is outstanding: the allocator's block-reuse path reads that value
    // specifically. EventQuery already does.
    return EventQuery(event);
  }

  Error_t memcpy(void* dst, const void* src, size_t count, MemcpyKind kind)
      override {
    return Memcpy(dst, src, count, kind);
  }
};

} // namespace c10::flagos
