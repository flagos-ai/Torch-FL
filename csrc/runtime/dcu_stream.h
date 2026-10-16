// Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#ifdef FLAGOS_DCU_VENDOR_STREAMS
#include <c10/core/Allocator.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <flagos.h>
#include <macros.h>
namespace c10::flagos::dcu {
FLAGOS_EXPORT c10::Stream Current(c10::DeviceIndex device = -1);
FLAGOS_EXPORT c10::Stream Default(c10::DeviceIndex device = -1);
FLAGOS_EXPORT c10::Stream New(c10::DeviceIndex device, int priority);
FLAGOS_EXPORT c10::Stream Exchange(c10::Stream stream);
FLAGOS_EXPORT c10::Stream External(Stream_t raw, c10::DeviceIndex device);
FLAGOS_EXPORT Stream_t Raw(c10::Stream stream);
FLAGOS_EXPORT bool Query(c10::Stream stream);
FLAGOS_EXPORT void Synchronize(c10::Stream stream);
FLAGOS_EXPORT void Record(void** event, c10::Stream stream, c10::DeviceIndex device, c10::EventFlag flag);
FLAGOS_EXPORT void Block(void* event, c10::Stream stream);
FLAGOS_EXPORT bool QueryEvent(void* event);
FLAGOS_EXPORT void SynchronizeEvent(void* event);
FLAGOS_EXPORT void DestroyEvent(void* event, c10::DeviceIndex device) noexcept;
FLAGOS_EXPORT double Elapsed(void* first, void* last, c10::DeviceIndex device);
// Blocking contract on explicitly selected current streams. No device-wide
// barrier: destination-before-copy and copy-before-destination are both covered
// by stream completion. Callers retain both allocations until this returns.
FLAGOS_EXPORT void CopyDeviceBytes(void* dst, c10::DeviceIndex dst_device,
    const void* src, c10::DeviceIndex src_device, size_t count);
FLAGOS_EXPORT void CopyBytesOnCurrentStream(
    void* dst, const void* src, size_t count, MemcpyKind kind);
FLAGOS_EXPORT void CopyData(void* dst, const void* src, size_t count);
FLAGOS_EXPORT c10::DataPtr Allocate(size_t size);
FLAGOS_EXPORT c10::DeleterFnPtr RawDeleter();
FLAGOS_EXPORT void RecordDataPtr(const c10::DataPtr& ptr, c10::Stream stream);
} // namespace c10::flagos::dcu
#endif
