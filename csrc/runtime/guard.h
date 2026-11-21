// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <flagos.h>

// The platform's own USE_* macro (FLAGOS_PLATFORM_USE_MACRO in the platform
// table). Only the runtime library target is given it on the command line; this
// header also compiles as part of csrc/, where nothing defines it, so on the
// platforms whose csrc/ arms this header's calls mirror, the FLAGOS_ACCEL_<ID>
// marker csrc/ does get stands in. Without the define those arms silently fall
// back to the generic ones -- on DCU that is a device-wide synchronize where a
// per-stream wait was wanted, and on Kunlunxin a device-wide one where the
// platform's own handle belongs.
#if !defined(USE_DCU) && defined(FLAGOS_ACCEL_DCU)
#define USE_DCU 1
#endif
#if !defined(USE_KUNLUNXIN) && defined(FLAGOS_ACCEL_KUNLUNXIN)
#define USE_KUNLUNXIN 1
#endif
#if !defined(USE_ILUVATAR) && defined(FLAGOS_ACCEL_ILUVATAR)
#define USE_ILUVATAR 1
#endif
#if !defined(USE_CAMBRICON) && defined(FLAGOS_ACCEL_CAMBRICON)
#define USE_CAMBRICON 1
#endif

#include <cstdint>

#ifdef USE_ASCEND
#include <c10/core/DeviceGuard.h>
#include "runtime/accelerator/ascend/acl_stream.h"
#include "runtime/allocator/caching_device_allocator.h"
#endif

#if defined(FLAGOS_ACCEL_CUDA) || defined(USE_CUDA) || defined(USE_MACA)
#define FLAGOS_GUARD_HAS_NATIVE_CUDA_STREAM 1
#include <c10/core/DeviceGuard.h>
#include "runtime/allocator/caching_device_allocator.h"
#else
#define FLAGOS_GUARD_HAS_NATIVE_CUDA_STREAM 0
#endif

#if defined(USE_DCU)
#define FLAGOS_GUARD_HAS_NATIVE_HYGON_STREAM 1
#include <c10/core/DeviceGuard.h>
#include "runtime/allocator/caching_device_allocator.h"
#else
#define FLAGOS_GUARD_HAS_NATIVE_HYGON_STREAM 0
#endif

#define FLAGOS_GUARD_HAS_OWNED_STREAM \
  (FLAGOS_GUARD_HAS_NATIVE_CUDA_STREAM || FLAGOS_GUARD_HAS_NATIVE_HYGON_STREAM)

#if defined(USE_ILUVATAR) || defined(USE_CAMBRICON) || FLAGOS_GUARD_HAS_OWNED_STREAM
#define FLAGOS_GUARD_HAS_PLATFORM_STREAM 1
#else
#define FLAGOS_GUARD_HAS_PLATFORM_STREAM 0
#endif

namespace c10::flagos {

#if defined(USE_DCU)
// Two entry points this interface needs on DCU are declared here rather than
// reached through flagos.h, and both are defined in
// runtime/accelerator/nvidia/ under their Dcu* names: DTK's headers declare
// PointerGetAttributes as a template, so the flat function a caller that only
// sees flagos.h resolves has no symbol in the library; and the stream handle
// this interface synchronizes has to be encoded from the c10 stream id, exactly
// as the Kunlunxin and Iluvatar arms encode theirs. Every other flagos.h call on
// DCU is reached by its own name.
__attribute__((visibility("default"))) Stream_t DcuStreamHandleOf(uintptr_t stream_id);
__attribute__((visibility("default"))) Error_t
DcuPointerGetAttributes(PointerAttributes* attributes, const void* ptr);
#endif

#ifdef USE_ASCEND
inline int ResolveAscendDevice(c10::Device d) {
  if (d.index() >= 0) {
    return d.index();
  }
  int device = 0;
  ::GetDevice(&device);
  return device;
}

inline c10::Stream MakeAscendStream(int device, aclrtStream stream) {
  return c10::Stream(
      c10::Stream::UNSAFE,
      c10::Device(c10::DeviceType::PrivateUse1, device),
      static_cast<c10::StreamId>(reinterpret_cast<uintptr_t>(stream)));
}

inline aclrtStream AscendStreamHandle(const c10::Stream& stream) {
  return reinterpret_cast<aclrtStream>(
      static_cast<uintptr_t>(stream.id()));
}
#endif

#if defined(USE_KUNLUNXIN) || FLAGOS_GUARD_HAS_PLATFORM_STREAM
// These native backends need no extra stream header here: the contract's
// GetDefaultStreamForDevice / GetCurrentStreamForDevice /
// SetCurrentStreamForDevice already return and take the `Stream_t` this
// interface traffics in, and each platform's stream.cc backs them with its own
// per-device registry (runtime/accelerator/{kunlunxin,iluvatar}/stream.cc). A
// `c10::Stream` is a device/index pair plus a StreamId, and StreamId is a
// uint64, so the handle rides in the id exactly as the Ascend arm does it.
inline int ResolvePlatformStreamDevice(c10::Device d) {
  if (d.index() >= 0) {
    return d.index();
  }
  int device = 0;
  ::GetDevice(&device);
  return device;
}

inline c10::Stream MakePlatformDeviceStream(int device, Stream_t stream) {
  return c10::Stream(
      c10::Stream::UNSAFE,
      c10::Device(c10::DeviceType::PrivateUse1, device),
      static_cast<c10::StreamId>(reinterpret_cast<uintptr_t>(stream)));
}

inline Stream_t PlatformStreamHandle(const c10::Stream& stream) {
  return reinterpret_cast<Stream_t>(static_cast<uintptr_t>(stream.id()));
}
#endif

#ifdef USE_KUNLUNXIN
inline int ResolveKunlunxinDevice(c10::Device d) {
  if (d.index() >= 0) {
    return d.index();
  }
  int device = 0;
  ::GetDevice(&device);
  return device;
}

inline c10::Stream MakeKunlunxinStream(int device, Stream_t stream) {
  return c10::Stream(
      c10::Stream::UNSAFE,
      c10::Device(c10::DeviceType::PrivateUse1, device),
      static_cast<c10::StreamId>(reinterpret_cast<uintptr_t>(stream)));
}

inline Stream_t KunlunxinStreamHandle(const c10::Stream& stream) {
  return reinterpret_cast<Stream_t>(static_cast<uintptr_t>(stream.id()));
}
#endif

struct GuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  GuardImpl() = default;
  explicit GuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
  }

  c10::DeviceType type() const override {
    return c10::DeviceType::PrivateUse1;
  }

  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    auto old_device_index = exchangeDeviceIndex(d.index());
    return c10::Device(c10::DeviceType::PrivateUse1, old_device_index);
  }

  c10::DeviceIndex exchangeDeviceIndex(c10::DeviceIndex device_index) const {
    int prev_device = -1;
    ::GetDevice(&prev_device);
    if (prev_device != device_index) {
      ::SetDevice(device_index);
    }
    return static_cast<c10::DeviceIndex>(prev_device);
  }

  c10::Device getDevice() const override {
    int device = -1;
    ::GetDevice(&device);
    return c10::Device(c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(device));
  }

  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    ::SetDevice(d.index());
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    ::SetDevice(d.index());
  }

  c10::Stream getStream(c10::Device d) const noexcept override {
#if   defined(USE_ASCEND)
    auto device = ResolveAscendDevice(d);
    return MakeAscendStream(
        device,
        at::native::flagos::ascend::GetCurrentAclStreamForDevice(device));
#elif defined(USE_KUNLUNXIN)
    auto device = ResolveKunlunxinDevice(d);
    return MakeKunlunxinStream(device, GetCurrentStreamForDevice(device));
#elif FLAGOS_GUARD_HAS_PLATFORM_STREAM
    auto device = ResolvePlatformStreamDevice(d);
    return MakePlatformDeviceStream(device, GetCurrentStreamForDevice(device));
#else
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
#endif
  }

  c10::Stream getDefaultStream(c10::Device d) const override {
#if   defined(USE_ASCEND)
    auto device = ResolveAscendDevice(d);
    return MakeAscendStream(
        device,
        at::native::flagos::ascend::GetDefaultAclStreamForDevice(device));
#elif defined(USE_KUNLUNXIN)
    auto device = ResolveKunlunxinDevice(d);
    return MakeKunlunxinStream(device, GetDefaultStreamForDevice(device));
#elif FLAGOS_GUARD_HAS_PLATFORM_STREAM
    auto device = ResolvePlatformStreamDevice(d);
    return MakePlatformDeviceStream(device, GetDefaultStreamForDevice(device));
#else
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
#endif
  }

  c10::Stream getStreamFromGlobalPool(c10::Device d, bool isHighPriority = false) const override {
#if   defined(USE_ASCEND)
    auto device = ResolveAscendDevice(d);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    return MakeAscendStream(device, stream);
#elif defined(USE_KUNLUNXIN)
    auto device = ResolveKunlunxinDevice(d);
    // The XPU runtime has no stream pool, so "from the global pool" is a fresh
    // stream. StreamCreateWithPriority would drop the priority anyway (the SDK
    // exposes none), and `isHighPriority` is dropped here for the same reason.
    Stream_t stream = nullptr;
    ::StreamCreate(&stream);
    return MakeKunlunxinStream(device, stream);
#elif FLAGOS_GUARD_HAS_PLATFORM_STREAM
    auto device = ResolvePlatformStreamDevice(d);
#if FLAGOS_GUARD_HAS_OWNED_STREAM
    c10::OptionalDeviceGuard device_guard;
    if (getDevice().index() != device) device_guard.reset_device(c10::Device(static_type, device));
#endif
    // As on Kunlunxin: the contract exposes no pool of its own, so "from the
    // global pool" is a fresh stream. Unlike Kunlunxin, corex's runtime does
    // take a priority (cudaStreamCreateWithPriority), and CUDA's convention is
    // the one this must follow: the numerically LOWEST value is the highest
    // priority, so ask the runtime for its range rather than guessing a number.
    // If the range query fails there is no ordering to honour and the neutral
    // default (0) is used.
    int least = 0;
    int greatest = 0;
    int priority = 0;
    if (::DeviceGetStreamPriorityRange(&least, &greatest) == Success) {
      priority = isHighPriority ? greatest : least;
    }
    Stream_t stream = nullptr;
#if FLAGOS_GUARD_HAS_OWNED_STREAM
    TORCH_CHECK(::StreamCreateWithPriority(&stream, 0, priority) == Success,
                "accelerator stream creation failed");
#else
    ::StreamCreateWithPriority(&stream, 0, priority);
#endif
    return MakePlatformDeviceStream(device, stream);
#else
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
#endif
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
#if   defined(USE_ASCEND)
    auto device = ResolveAscendDevice(s.device());
    auto old = at::native::flagos::ascend::GetCurrentAclStreamForDevice(device);
    at::native::flagos::ascend::SetCurrentAclStreamForDevice(
        device, AscendStreamHandle(s));
    return MakeAscendStream(device, old);
#elif defined(USE_KUNLUNXIN)
    auto device = ResolveKunlunxinDevice(s.device());
    auto old = GetCurrentStreamForDevice(device);
    SetCurrentStreamForDevice(device, KunlunxinStreamHandle(s));
    return MakeKunlunxinStream(device, old);
#elif FLAGOS_GUARD_HAS_PLATFORM_STREAM
    auto device = ResolvePlatformStreamDevice(s.device());
    auto old = GetCurrentStreamForDevice(device);
    SetCurrentStreamForDevice(device, PlatformStreamHandle(s));
    return MakePlatformDeviceStream(device, old);
#else
    return c10::Stream(c10::Stream::UNSAFE, s.device(), 0);
#endif
  }

  c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
#if   defined(USE_ASCEND)
    auto device = ResolveAscendDevice(d);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    return MakeAscendStream(device, stream);
#elif defined(USE_KUNLUNXIN)
    auto device = ResolveKunlunxinDevice(d);
    Stream_t stream = nullptr;
    ::StreamCreateWithPriority(&stream, 0, priority);
    return MakeKunlunxinStream(device, stream);
#elif FLAGOS_GUARD_HAS_PLATFORM_STREAM
    auto device = ResolvePlatformStreamDevice(d);
#if FLAGOS_GUARD_HAS_OWNED_STREAM
    c10::OptionalDeviceGuard device_guard;
    if (getDevice().index() != device) device_guard.reset_device(c10::Device(static_type, device));
    Stream_t stream = nullptr;
    TORCH_CHECK(::StreamCreateWithPriority(&stream, 0, priority) == Success,
                "accelerator stream creation failed");
    return MakePlatformDeviceStream(device, stream);
#else
    // `priority` is the c10 convention (a linear index, where 0 is the default
    // band), while corex -- like CUDA -- orders by the numeric value with the
    // lowest being highest priority. Map the index onto the runtime's range
    // rather than passing it through, which would silently invert the request.
    int least = 0;
    int greatest = 0;
    int corex_priority = 0;
    if (::DeviceGetStreamPriorityRange(&least, &greatest) == Success &&
        priority > 0) {
      corex_priority = greatest;
    }
    Stream_t stream = nullptr;
    ::StreamCreateWithPriority(&stream, 0, corex_priority);
    return MakePlatformDeviceStream(device, stream);
#endif
#else
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
#endif
  }

  bool queryStream(const c10::Stream& stream) const override {
#ifdef USE_ASCEND
    aclrtStream handle = AscendStreamHandle(stream);
    aclrtStreamStatus status;
    if (aclrtStreamQuery(handle, &status) != ACL_SUCCESS) {
      return false;
    }
    return status == ACL_STREAM_STATUS_COMPLETE;
#elif defined(USE_KUNLUNXIN)
    // The XPU runtime ships no stream-status query, so StreamQuery answers
    // ErrorNotSupported (runtime/accelerator/kunlunxin/stream.cc). Waiting on
    // the stream and reporting completion is therefore the only state this
    // interface can honestly report, and it is a stronger guarantee than the
    // answer requires -- never "not done" for work that is done.
    return ::StreamSynchronize(KunlunxinStreamHandle(stream)) == Success;
#elif FLAGOS_GUARD_HAS_PLATFORM_STREAM
    // corex does ship a stream query, so unlike Kunlunxin this can answer
    // without draining the stream. ErrorNotReady, and any other error, mean the
    // stream has not provably finished -- reported as not done.
    return ::StreamQuery(PlatformStreamHandle(stream)) == Success;
#else
    ::DeviceSynchronize();
    return true;
#endif
  }

  void synchronizeStream(const c10::Stream& stream) const override {
#ifdef USE_ASCEND
    aclrtSynchronizeStream(AscendStreamHandle(stream));
#elif defined(USE_KUNLUNXIN)
    ::StreamSynchronize(KunlunxinStreamHandle(stream));
#elif FLAGOS_GUARD_HAS_PLATFORM_STREAM
    // corex has a real per-stream wait, so ask for the stream the caller named
    // rather than falling through to the device-wide one: same answer, and it
    // does not also drain unrelated work on other streams.
    ::StreamSynchronize(PlatformStreamHandle(stream));
#elif defined(USE_DCU)
    // DCU has a real per-stream wait; the handle comes from the encoded
    // c10::Stream::id() (see DcuStreamHandleOf in nvidia/stream.cc).
    ::StreamSynchronize(DcuStreamHandleOf(static_cast<uintptr_t>(stream.id())));
#else
    ::DeviceSynchronize();
#endif
  }

  // The accelerator API -- torch.accelerator.synchronize(), and through it
  // every caller of torch.utils.benchmark.Timer, which synchronises that way --
  // reaches this method, and the interface's default refuses with "Backend
  // doesn't support synchronizing all streams on device." Delegating to the
  // stream primitive is not the same request: this one is for the whole device.
  //
  // A negative index means "the current device", which is what c10 passes when
  // the caller named none, so it is left alone. Otherwise the device is selected
  // for the duration and the previous one restored, matching CUDAGuardImpl: the
  // caller asked about one device and must not come back having switched.
  void synchronizeDevice(const c10::DeviceIndex device_index) const override {
    const c10::DeviceIndex previous =
        device_index < 0 ? -1 : exchangeDeviceIndex(device_index);
#ifdef USE_ASCEND
    aclrtSynchronizeDevice();
#else
    ::DeviceSynchronize();
#endif
    if (previous >= 0) {
      ::SetDevice(previous);
    }
  }

  void synchronizeEvent(void* event) const override {
    if (event) {
      ::EventSynchronize((Event_t)event);
    }
  }

  void recordDataPtrOnStream(
      const c10::DataPtr& data_ptr,
      const c10::Stream& stream) const override {
#if defined(USE_ASCEND)
    if (CachingDeviceAllocator::is_enabled()) {
      c10::OptionalDeviceGuard device_guard;
      if (getDevice() != stream.device()) device_guard.reset_device(stream.device());
      GetCachingAllocator()->record_stream(
          data_ptr, reinterpret_cast<Stream_t>(AscendStreamHandle(stream)));
    }
#elif FLAGOS_GUARD_HAS_OWNED_STREAM
    if (CachingDeviceAllocator::is_enabled()) {
      c10::OptionalDeviceGuard device_guard;
      if (getDevice() != stream.device()) device_guard.reset_device(stream.device());
      GetCachingAllocator()->record_stream(data_ptr, PlatformStreamHandle(stream));
    }
#endif
  }

  double elapsedTime(
      void* event1,
      void* event2,
      const c10::DeviceIndex device_index) const override {
    float ms = 0.0f;
    if (event1 && event2) {
      ::EventElapsedTime(&ms, (Event_t)event1, (Event_t)event2);
    }
    return static_cast<double>(ms);
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    int count = 0;
    ::GetDeviceCount(&count);
    return static_cast<c10::DeviceIndex>(count);
  }

  c10::DeviceCapability getDeviceCapability(c10::Device d) const override {
    // Return a default device capability struct with all scalar types enabled
    // This is called by autograd profiler to determine device properties
    // The default constructor already enables all capabilities
    return c10::DeviceCapability();
  }

  void record(
      void** event,
      const c10::Stream& stream,
      const c10::DeviceIndex device_index,
      const c10::EventFlag flag) const override {
#if FLAGOS_GUARD_HAS_OWNED_STREAM
    c10::OptionalDeviceGuard device_guard;
    if (getDevice() != stream.device()) device_guard.reset_device(stream.device());
#endif
    if (!*event) {
      ::EventCreate((Event_t*)event);
    }
#if   defined(USE_ASCEND)
    ::EventRecord(*(Event_t*)event, (Stream_t)AscendStreamHandle(stream));
#elif defined(USE_KUNLUNXIN)
    ::EventRecord(*(Event_t*)event, KunlunxinStreamHandle(stream));
#elif FLAGOS_GUARD_HAS_PLATFORM_STREAM
    ::EventRecord(*(Event_t*)event, PlatformStreamHandle(stream));
#else
    ::EventRecord(*(Event_t*)event, nullptr);
#endif
  }

  void block(void* event, const c10::Stream& stream) const override {
#ifdef USE_ASCEND
    ::StreamWaitEvent(
        (Stream_t)AscendStreamHandle(stream), (Event_t)event, 0);
#elif defined(USE_KUNLUNXIN)
    ::StreamWaitEvent(KunlunxinStreamHandle(stream), (Event_t)event, 0);
#elif FLAGOS_GUARD_HAS_PLATFORM_STREAM
    // The stream has to be named here. On the CUDA platform the #else arm's
    // nullptr means stream 0, which is where torch's work actually runs; on
    // iluvatar the kernels run on the registry's own default stream, so a wait
    // armed on stream 0 would order against nothing.
    ::StreamWaitEvent(PlatformStreamHandle(stream), (Event_t)event, 0);
#else
    ::StreamWaitEvent(nullptr, (Event_t)event, 0);
#endif
  }

  bool queryEvent(void* event) const override {
    return ::EventQuery((Event_t)event) == Success;
  }

  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    ::EventDestroy((Event_t)event);
  }
};

} // namespace c10::flagos
