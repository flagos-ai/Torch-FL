// Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0
#include "dcu_stream.h"

#ifdef FLAGOS_DCU_VENDOR_STREAMS
#include <ATen/Context.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/DeviceGuard.h>
// Only this TU consumes matching DTK HIP headers. Public headers remain Torch
// vendor-neutral; libflagos remains independent of Torch symbols.
#include <c10/hip/HIPStream.h>

namespace c10::flagos::dcu {
namespace {
const c10::impl::DeviceGuardImplInterface* Vendor() {
  return c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA);
}

c10::Stream Retag(c10::Stream stream, c10::DeviceType type) {
  return c10::Stream(
      c10::Stream::UNSAFE,
      c10::Device(type, stream.device_index()),
      stream.id());
}

c10::Stream ToVendor(c10::Stream stream) {
  TORCH_CHECK(
      stream.device_type() == c10::DeviceType::PrivateUse1,
      "Expected a flagos stream, got ", stream.device());
  TORCH_CHECK(stream.device_index() >= 0 &&
              stream.device_index() < Vendor()->deviceCount(),
              "Invalid DCU stream device index: ", stream.device_index());
  return Retag(stream, c10::DeviceType::CUDA);
}

c10::Device Device(c10::DeviceIndex device) {
  if (device < 0) {
    device = Vendor()->getDevice().index();
  }
  TORCH_CHECK(
      device >= 0 && device < Vendor()->deviceCount(),
      "Invalid DCU device index: ", device);
  return c10::Device(c10::DeviceType::CUDA, device);
}
} // namespace

c10::Stream Current(c10::DeviceIndex device) {
  return Retag(Vendor()->getStream(Device(device)), c10::DeviceType::PrivateUse1);
}

c10::Stream Default(c10::DeviceIndex device) {
  return Retag(Vendor()->getDefaultStream(Device(device)), c10::DeviceType::PrivateUse1);
}

c10::Stream New(c10::DeviceIndex device, int priority) {
  return Retag(Vendor()->getNewStream(Device(device), priority), c10::DeviceType::PrivateUse1);
}

c10::Stream Exchange(c10::Stream stream) {
  return Retag(Vendor()->exchangeStream(ToVendor(stream)), c10::DeviceType::PrivateUse1);
}

c10::Stream External(Stream_t raw, c10::DeviceIndex device) {
  const auto resolved = Device(device);
  auto stream = c10::hip::getStreamFromExternal(
      reinterpret_cast<hipStream_t>(raw), resolved.index());
  return Retag(stream.unwrap(), c10::DeviceType::PrivateUse1);
}

Stream_t Raw(c10::Stream stream) {
  ToVendor(stream); // Validate the public tag before unpacking the opaque ID.
  auto hip_stream = c10::hip::HIPStream(
      c10::hip::HIPStream::UNCHECKED, Retag(stream, c10::DeviceType::HIP));
  return reinterpret_cast<Stream_t>(hip_stream.stream());
}

bool Query(c10::Stream stream) {
  return Vendor()->queryStream(ToVendor(stream));
}

void Synchronize(c10::Stream stream) {
  Vendor()->synchronizeStream(ToVendor(stream));
}

void Record(void** event, c10::Stream stream, c10::DeviceIndex device, c10::EventFlag flag) {
  Vendor()->record(event, ToVendor(stream), device, flag);
}

void Block(void* event, c10::Stream stream) {
  Vendor()->block(event, ToVendor(stream));
}

bool QueryEvent(void* event) {
  return !event || Vendor()->queryEvent(event);
}

void SynchronizeEvent(void* event) {
  if (event) {
    Vendor()->synchronizeEvent(event);
  }
}

void DestroyEvent(void* event, c10::DeviceIndex device) noexcept {
  if (event) {
    Vendor()->destroyEvent(event, device);
  }
}

double Elapsed(void* first, void* last, c10::DeviceIndex device) {
  return Vendor()->elapsedTime(first, last, device);
}

void CopyDeviceBytes(void* dst, c10::DeviceIndex dst_device,
                     const void* src, c10::DeviceIndex src_device, size_t count) {
  if (count == 0) return;
  TORCH_CHECK(dst && src, "DCU copy requires non-null device pointers");
  Device(src_device);
  Device(dst_device);
  const c10::DeviceGuard guard(c10::Device(c10::DeviceType::PrivateUse1, src_device));
  const auto producer = Current(src_device);
  if (dst_device != src_device) {
    // Native side streams are non-blocking; a default-stream memcpy does not
    // order against the destination's previous work. Only drain its current
    // stream, not unrelated streams or the whole device.
    Synchronize(Current(dst_device));
  }
  // Submit on the producer's actual stream, not the legacy/default stream.
  // Keep the public operation blocking: neither old Storage nor a temporary
  // can be freed/reused until this transfer has finished.
  const auto error = ::MemcpyAsync(dst, src, count, MemcpyDeviceToDevice, Raw(producer));
  TORCH_CHECK(error == Success, "DCU device copy failed (error ", int(error), ")");
  Synchronize(producer);
}

void CopyBytesOnCurrentStream(
    void* dst, const void* src, size_t count, MemcpyKind kind) {
  if (count == 0) return;
  if (kind == MemcpyDeviceToDevice) {
    CopyData(dst, src, count);
    return;
  }
  const auto stream = Current();
  const auto error = ::MemcpyAsync(dst, src, count, kind, Raw(stream));
  TORCH_CHECK(error == Success, "DCU blocking copy failed (error ", int(error), ")");
  // A host-blocking contract must not accidentally submit to the legacy stream
  // and return before a non-default consumer can see the transfer.
  Synchronize(stream);
}

void CopyData(void* dst, const void* src, size_t count) {
  if (count == 0) return;
  // flagos's Malloc registry does NOT contain vendor allocator pointers.
  // Resolve allocation ownership using HIP, including interior pointers.
  hipPointerAttribute_t dst_attr{}, src_attr{};
  const auto dst_error = hipPointerGetAttributes(&dst_attr, dst);
  const auto src_error = hipPointerGetAttributes(&src_attr, src);
  TORCH_CHECK(dst_error == hipSuccess && src_error == hipSuccess,
              "DCU allocator copy_data requires valid device pointers");
  TORCH_CHECK(dst_attr.memoryType == hipMemoryTypeDevice &&
              src_attr.memoryType == hipMemoryTypeDevice,
              "DCU allocator copy_data requires device allocations");
  CopyDeviceBytes(dst, dst_attr.device, src, src_attr.device, count);
}

c10::DataPtr Allocate(size_t size) {
  // Stream creation alone does not initialize the vendor caching allocator.
  // Use the same device-generic lazy init as DcuDeviceMemory::allocator().
  // Context serializes init; do not swap official Torch core DSOs for DAS ones.
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  return c10::getDeviceAllocator(c10::DeviceType::CUDA)->allocate(size);
}

c10::DeleterFnPtr RawDeleter() {
  Vendor();
  return c10::getDeviceAllocator(c10::DeviceType::CUDA)->raw_deleter();
}

void RecordDataPtr(const c10::DataPtr& ptr, c10::Stream stream) {
  if (ptr.get()) {
    Vendor()->recordDataPtrOnStream(ptr, ToVendor(stream));
  }
}
} // namespace c10::flagos::dcu
#endif
