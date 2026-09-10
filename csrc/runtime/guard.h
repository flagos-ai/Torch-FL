// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <flagos.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>

#ifdef USE_ASCEND
#include "runtime/accelerator/ascend/acl_stream.h"
#endif

// Real c10::cuda:: stream/event plumbing. Gated the same way as
// runtime/allocator/caching_device_allocator.cc's delegation to
// c10::cuda::CUDACachingAllocator (NOT the looser cuda_runtime.h guard used
// in hooks.h/contiguous_ops.cc/copy_ops.cc): the DCU wheel is hipified and
// exports c10::hip::HIPCachingAllocator with zero c10::cuda symbols (see
// backends/dcu_memory.h), so c10::cuda::getCurrentCUDAStream would fail to
// resolve there even though DCU's CUDA-compat runtime satisfies plain
// cudaStream_t calls. MUSA is excluded for the plainer reason that the Moore
// Threads toolkit ships no CUDA runtime at all, so the header itself is absent
// -- same exclusion as hooks.h and copy_ops.cc already carry.
#if !defined(USE_ASCEND) && !defined(USE_TSINGMICRO) && !defined(USE_DCU) && \
    !defined(USE_GCU) && !defined(USE_MUSA) && !defined(USE_BPU)
#define FLAGOS_GUARD_HAS_CUDA_STREAM 1
#include <c10/cuda/CUDAStream.h>
#else
#define FLAGOS_GUARD_HAS_CUDA_STREAM 0
#endif

namespace c10::flagos {

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

#ifdef USE_DCU
// DCU has no usable c10::cuda:: stream API (see the guard above), so the current
// stream is tracked by flagos itself in
// runtime/accelerator/dcu/stream_registry.cc. These helpers translate between
// that raw handle and c10::Stream using the same id-is-the-pointer convention as
// the Ascend pair above and as WrapperRecordStream in csrc/aten/register.cc.
//
// Without this, the #else fallback below reported stream id 0 for every device,
// which the SDK-native operators would read as "default stream" no matter what
// stream the caller had made current.
inline int ResolveDcuDevice(c10::Device d) {
  if (d.index() >= 0) {
    return d.index();
  }
  int device = 0;
  ::GetDevice(&device);
  return device;
}

inline c10::Stream MakeDcuStream(int device, Stream_t stream) {
  return c10::Stream(
      c10::Stream::UNSAFE,
      c10::Device(c10::DeviceType::PrivateUse1, device),
      static_cast<c10::StreamId>(reinterpret_cast<uintptr_t>(stream)));
}

inline Stream_t DcuStreamHandle(const c10::Stream& stream) {
  return reinterpret_cast<Stream_t>(static_cast<uintptr_t>(stream.id()));
}

// Fixed pool of side streams, handed out round-robin -- the same shape as
// c10::cuda's stream pool and for the same reason: creating a stream per
// getNewStream() call would leak one on every `torch.Stream(device="flagos")`,
// while returning the default stream would silently serialize code that asked
// for a side stream. Streams are created once and intentionally never destroyed;
// they live for the process, so there is no teardown ordering hazard against
// pending work.
inline Stream_t DcuStreamFromPool(int device) {
  constexpr int kPoolSize = 32;
  struct Pool {
    Stream_t streams[kPoolSize] = {};
    std::atomic<unsigned> next{0};
  };
  // Per-device pools, initialized on first use for that device.
  static std::array<std::once_flag, 64> once;
  static std::array<Pool, 64> pools;
  if (device < 0 || device >= static_cast<int>(pools.size())) {
    return nullptr;
  }
  std::call_once(once[device], [device]() {
    // Create on the target device: stream creation is device-scoped.
    int prev = -1;
    ::GetDevice(&prev);
    if (prev != device) {
      ::SetDevice(device);
    }
    for (auto& s : pools[device].streams) {
      if (::StreamCreate(&s) != Success) {
        s = nullptr; // fall back to the default stream for this slot
      }
    }
    if (prev >= 0 && prev != device) {
      ::SetDevice(prev);
    }
  });
  unsigned i = pools[device].next.fetch_add(1, std::memory_order_relaxed);
  return pools[device].streams[i % kPoolSize];
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

  // NB on all the FLAGOS_GUARD_HAS_CUDA_STREAM branches below: `d.index()`
  // may be -1 (unresolved / "current device"), matching CUDAGuardImpl's own
  // contract. c10::cuda resolves -1 to the real current device internally,
  // but that resolution lives in the *returned* CUDAStream/DeviceIndex, not
  // in `d`. Reusing the caller's original (possibly still -1) Device when
  // building the returned c10::Stream would leak an unresolved index back
  // into torch; a later setCurrentCUDAStream() on that leaked -1 corrupts
  // CUDA's fixed-size per-device stream-pool bookkeeping (reproduced
  // locally as a `free(): invalid pointer` crash from a plain
  // `with torch.Stream(device="flagos"): ...`). Always reconstruct the
  // returned Device from the resolved index.

  c10::Stream getStream(c10::Device d) const noexcept override {
#if FLAGOS_GUARD_HAS_CUDA_STREAM
    auto s = c10::cuda::getCurrentCUDAStream(d.index());
    return c10::Stream(
        c10::Stream::UNSAFE,
        c10::Device(c10::DeviceType::PrivateUse1, s.device_index()),
        s.id());
#elif defined(USE_ASCEND)
    auto device = ResolveAscendDevice(d);
    return MakeAscendStream(
        device,
        at::native::flagos::ascend::GetCurrentAclStreamForDevice(device));
#elif defined(USE_DCU)
    auto device = ResolveDcuDevice(d);
    return MakeDcuStream(device, ::GetCurrentStreamForDevice(device));
#else
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
#endif
  }

  c10::Stream getDefaultStream(c10::Device d) const override {
#if FLAGOS_GUARD_HAS_CUDA_STREAM
    auto s = c10::cuda::getDefaultCUDAStream(d.index());
    return c10::Stream(
        c10::Stream::UNSAFE,
        c10::Device(c10::DeviceType::PrivateUse1, s.device_index()),
        s.id());
#elif defined(USE_ASCEND)
    auto device = ResolveAscendDevice(d);
    return MakeAscendStream(
        device,
        at::native::flagos::ascend::GetDefaultAclStream());
#elif defined(USE_DCU)
    auto device = ResolveDcuDevice(d);
    return MakeDcuStream(device, ::GetDefaultStreamForDevice(device));
#else
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
#endif
  }

  c10::Stream getStreamFromGlobalPool(c10::Device d, bool isHighPriority = false) const override {
#if FLAGOS_GUARD_HAS_CUDA_STREAM
    auto s = c10::cuda::getStreamFromPool(isHighPriority, d.index());
    return c10::Stream(
        c10::Stream::UNSAFE,
        c10::Device(c10::DeviceType::PrivateUse1, s.device_index()),
        s.id());
#elif defined(USE_ASCEND)
    auto device = ResolveAscendDevice(d);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    return MakeAscendStream(device, stream);
#elif defined(USE_DCU)
    // DTK's runtime exposes no stream priorities through the compat layer, so
    // isHighPriority cannot be honored; a normal-priority pool stream is the
    // closest correct answer.
    (void)isHighPriority;
    auto device = ResolveDcuDevice(d);
    return MakeDcuStream(device, DcuStreamFromPool(device));
#else
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
#endif
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
#if FLAGOS_GUARD_HAS_CUDA_STREAM
    // CUDAStream::UNCHECKED does not itself validate device_type, but
    // setCurrentCUDAStream() stores the wrapped c10::Stream verbatim in its
    // per-device slot, and every later getCurrentCUDAStream() unpacks that
    // slot assuming DeviceType::CUDA -- wrapping with s's PrivateUse1
    // device (as the brief's snippet does) corrupts that slot. Re-tag as
    // CUDA before handing to c10::cuda.
    auto cs = c10::cuda::CUDAStream(
        c10::cuda::CUDAStream::UNCHECKED,
        c10::Stream(
            c10::Stream::UNSAFE,
            c10::Device(c10::DeviceType::CUDA, s.device_index()),
            s.id()));
    auto old = c10::cuda::getCurrentCUDAStream(cs.device_index());
    c10::cuda::setCurrentCUDAStream(cs);
    return c10::Stream(
        c10::Stream::UNSAFE,
        c10::Device(c10::DeviceType::PrivateUse1, old.device_index()),
        old.id());
#elif defined(USE_ASCEND)
    auto device = ResolveAscendDevice(s.device());
    auto old = at::native::flagos::ascend::GetCurrentAclStreamForDevice(device);
    at::native::flagos::ascend::SetCurrentAclStreamForDevice(
        device, AscendStreamHandle(s));
    return MakeAscendStream(device, old);
#elif defined(USE_DCU)
    auto device = ResolveDcuDevice(s.device());
    auto old = ::GetCurrentStreamForDevice(device);
    ::SetCurrentStreamForDevice(device, DcuStreamHandle(s));
    return MakeDcuStream(device, old);
#else
    return c10::Stream(c10::Stream::UNSAFE, s.device(), 0);
#endif
  }

  c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
#if FLAGOS_GUARD_HAS_CUDA_STREAM
    auto s = c10::cuda::getStreamFromPool(priority, d.index());
    return c10::Stream(
        c10::Stream::UNSAFE,
        c10::Device(c10::DeviceType::PrivateUse1, s.device_index()),
        s.id());
#elif defined(USE_ASCEND)
    auto device = ResolveAscendDevice(d);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    return MakeAscendStream(device, stream);
#elif defined(USE_DCU)
    (void)priority; // no priority control through DTK's compat runtime
    auto device = ResolveDcuDevice(d);
    return MakeDcuStream(device, DcuStreamFromPool(device));
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
#elif defined(USE_DCU)
    // Query the one stream instead of draining the device: the ids are real
    // handles here (see the registry above), and a device-wide sync would both
    // over-synchronize and report "complete" for work on other streams.
    return ::StreamQuery(DcuStreamHandle(stream)) == Success;
#else
    ::DeviceSynchronize();
    return true;
#endif
  }

  void synchronizeStream(const c10::Stream& stream) const override {
#ifdef USE_ASCEND
    aclrtSynchronizeStream(AscendStreamHandle(stream));
#elif defined(USE_DCU)
    ::StreamSynchronize(DcuStreamHandle(stream));
#else
    ::DeviceSynchronize();
#endif
  }

  void synchronizeEvent(void* event) const override {
    if (event) {
      ::EventSynchronize((Event_t)event);
    }
  }

  void recordDataPtrOnStream(
      const c10::DataPtr& data_ptr,
      const c10::Stream& stream) const override {
    // No-op: flagos uses CUDA memory which is already tracked
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
    if (!*event) {
      ::EventCreate((Event_t*)event);
    }
#if FLAGOS_GUARD_HAS_CUDA_STREAM
    // Re-tag as CUDA (see exchangeStream above) before wrapping -- CUDAStream
    // only holds the c10::Stream, it does not re-validate device_type, but
    // cs.stream()'s internal unpack does assume CUDA.
    auto cs = c10::cuda::CUDAStream(
        c10::cuda::CUDAStream::UNCHECKED,
        c10::Stream(
            c10::Stream::UNSAFE,
            c10::Device(c10::DeviceType::CUDA, stream.device_index()),
            stream.id()));
    // cs.stream() returns cudaStream_t; Stream_t is flagos's opaque
    // `struct Stream*` ABI handle. Both conventions alias a raw
    // cudaStream_t under the hood (see the (cudaStream_t)stream casts in
    // accelerator/cuda/stream.cc), so this reinterpret is the established
    // flagos<->CUDA stream conversion, not a new pattern.
    ::EventRecord(*(Event_t*)event, (Stream_t)cs.stream());
#elif defined(USE_ASCEND)
    ::EventRecord(*(Event_t*)event, (Stream_t)AscendStreamHandle(stream));
#elif defined(USE_DCU)
    // Record on the stream actually passed in. The nullptr fallback below would
    // pin every event to the default stream, so an event recorded inside a side
    // stream would order against the wrong queue.
    ::EventRecord(*(Event_t*)event, DcuStreamHandle(stream));
#else
    ::EventRecord(*(Event_t*)event, nullptr);
#endif
  }

  void block(void* event, const c10::Stream& stream) const override {
#ifdef USE_ASCEND
    ::StreamWaitEvent(
        (Stream_t)AscendStreamHandle(stream), (Event_t)event, 0);
#elif defined(USE_DCU)
    ::StreamWaitEvent(DcuStreamHandle(stream), (Event_t)event, 0);
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
