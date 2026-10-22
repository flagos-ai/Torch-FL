// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "../device_memory_interface.h"

#include <tx_runtime.h>
#include <cstdio>

namespace c10::flagos {

// TsingMicro (TX API) implementation of DeviceMemoryInterface.
// Directly calls TX runtime APIs for raw memory operations.
class TsingMicroDeviceMemory final : public DeviceMemoryInterface {
 public:
  Error_t device_malloc(void** ptr, size_t size) override {
    txError_t err = txMalloc(ptr, static_cast<uint64_t>(size));
    if (err != TX_SUCCESS) {
      fprintf(stderr, "[flagos-tsingmicro] txMalloc(%zu bytes) failed: %s\n",
              size, txGetErrorString(err));
      *ptr = nullptr;
      return ErrorMemoryAllocation;
    }
    return Success;
  }

  Error_t device_free(void* ptr) override {
    txError_t err = txFree(ptr);
    if (err != TX_SUCCESS) {
      fprintf(stderr, "[flagos-tsingmicro] txFree(%p) failed: %s\n",
              ptr, txGetErrorString(err));
      return ErrorUnknown;
    }
    return Success;
  }

  Error_t get_device_index(int* device) override {
    uint32_t tx_device = 0;
    txError_t err = txGetDevice(&tx_device);
    if (err != TX_SUCCESS) {
      return ErrorUnknown;
    }
    *device = static_cast<int>(tx_device);
    return Success;
  }

  Error_t set_device(int device) override {
    txError_t err = txSetDevice(static_cast<uint32_t>(device));
    if (err != TX_SUCCESS) {
      return ErrorInvalidDevice;
    }
    return Success;
  }

  Error_t get_memory_info(size_t* free, size_t* total) override {
    uint64_t tx_free = 0;
    uint64_t tx_total = 0;
    txError_t err = txMemGetInfo(&tx_free, &tx_total);
    if (err != TX_SUCCESS) {
      return ErrorUnknown;
    }
    *free = static_cast<size_t>(tx_free);
    *total = static_cast<size_t>(tx_total);
    return Success;
  }

  Error_t event_create(Event_t* event) override {
    txError_t err = txEventCreate(reinterpret_cast<txEvent_t*>(event));
    if (err != TX_SUCCESS) {
      return ErrorUnknown;
    }
    return Success;
  }

  Error_t event_destroy(Event_t event) override {
    txError_t err = txEventDestroy(reinterpret_cast<txEvent_t>(event));
    return (err == TX_SUCCESS) ? Success : ErrorUnknown;
  }

  Error_t event_record(Event_t event, Stream_t stream) override {
    txError_t err = txEventRecord(
        reinterpret_cast<txEvent_t>(event),
        reinterpret_cast<txStream_t>(stream));
    return (err == TX_SUCCESS) ? Success : ErrorUnknown;
  }

  Error_t event_query(Event_t event) override {
    txError_t err = txEventQuery(reinterpret_cast<txEvent_t>(event));
    if (err == TX_SUCCESS) {
      return Success;
    } else if (err == TX_ERROR_NOT_READY) {
      return ErrorNotReady;
    }
    return ErrorUnknown;
  }

  Error_t memcpy(
      void* dst,
      const void* src,
      size_t count,
      MemcpyKind kind) override {
    txMemcpyKind tx_kind;
    switch (kind) {
      case MemcpyHostToHost:
        tx_kind = txMemcpyHostToHost;
        break;
      case MemcpyHostToDevice:
        tx_kind = txMemcpyHostToDevice;
        break;
      case MemcpyDeviceToHost:
        tx_kind = txMemcpyDeviceToHost;
        break;
      case MemcpyDeviceToDevice:
        tx_kind = txMemcpyDeviceToDevice;
        break;
      default:
        return ErrorUnknown;
    }
    txError_t err = txMemcpy(dst, src, static_cast<uint64_t>(count), tx_kind);
    return (err == TX_SUCCESS) ? Success : ErrorUnknown;
  }
};

} // namespace c10::flagos