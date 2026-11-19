// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

struct _object;

namespace flagos::native_host::backend {

constexpr uint32_t kAbiVersion = 2;
constexpr size_t kSlotBytes = 128;
constexpr size_t kSlotAlignment = 128;

enum Capability : uint64_t {
  SoftwareDescriptor = 1 << 0,
  HardwareDescriptor = 1 << 1,
  GlobalScratch = 1 << 2,
  ProfileScratch = 1 << 3,
  CooperativeLaunch = 1 << 4,
  ClusterLaunch = 1 << 5,
  ProgrammaticLaunch = 1 << 6,
  StreamCaptureQuery = 1 << 7,
};

struct alignas(kSlotAlignment) Slot {
  Slot() {}
  std::array<unsigned char, kSlotBytes> bytes;
  template <class T>
  void Set(T value) {
    static_assert(sizeof(T) <= kSlotBytes);
    std::memcpy(bytes.data(), &value, sizeof(T));
  }
};

inline uint64_t CheckedProduct(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::overflow_error("Triton launch size overflow");
  return a * b;
}

struct Scratch {
  uint64_t bytes = 0;
  uint64_t alignment = 1;
};

// Handles and argument storage are borrowed for this enqueue. The original
// CompiledKernel owns the device module; its cache entry retains that owner.
struct Launch {
  std::array<uint64_t, 3> grid;
  uint64_t stream;
  uint64_t function;
  void** arguments;
  uint64_t global_scratch = 0;
  uint64_t global_bytes = 0;
  bool extended = false;
};

class Kernel {
 public:
  virtual ~Kernel() = default;
  virtual void Enqueue(const Launch&) const = 0;
  virtual void PackScratch(uint64_t, uint64_t, Slot*) const {
    if (scratch_slots) throw std::runtime_error("backend scratch ABI is not implemented");
  }
  Scratch global, profile;
  uint64_t scratch_multiplier = 1;
  size_t scratch_slots = 0;
};

struct DescriptorValue {
  uint64_t pointer;
  size_t rank;
  const uint64_t* shape;
  const uint64_t* strides;
  bool nan_padding;
};

class Descriptor {
 public:
  virtual ~Descriptor() = default;
  virtual size_t Capacity() const = 0;
  // Writes at most Capacity() slots; no device work or cached input/output.
  virtual size_t Pack(const DescriptorValue&, Slot*) const = 0;
};

// Compiled once per wheel. Python metadata is consumed only while constructing
// these immutable backend objects, never by the warm native launch path.
struct Api {
  uint32_t abi_version;
  size_t struct_size;
  const char* vendor;
  const char* compiler_backend;
  uint64_t capabilities;
  uint64_t scratch_alignment;
  std::unique_ptr<Kernel> (*kernel)(_object*);
  std::shared_ptr<const Descriptor> (*descriptor)(_object*, _object* kernel_metadata);
  bool (*is_capturing)(uint64_t);
};

inline const Api& Validate(const Api& api) {
  if (api.abi_version != kAbiVersion || api.struct_size != sizeof(Api))
    throw std::runtime_error("native-host backend ABI mismatch");
  if (!api.vendor || !api.compiler_backend || !api.kernel ||
      ((api.capabilities & (SoftwareDescriptor | HardwareDescriptor)) && !api.descriptor) ||
      ((api.capabilities & StreamCaptureQuery) && !api.is_capturing) ||
      !api.scratch_alignment || (api.scratch_alignment & (api.scratch_alignment - 1)))
    throw std::runtime_error("incomplete native-host backend capabilities");
  return api;
}

const Api& CompiledBackend();

}  // namespace flagos::native_host::backend
