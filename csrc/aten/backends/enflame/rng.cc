// Copyright (c) 2026, BAAI. All rights reserved.
//
// Native random-number kernels for Enflame GCU (topsaten).
//
// topsaten's RNG API accepts an explicit {seed, offset} value rather than an
// ATen generator. Generator-less calls consume the same seed/offset stream as
// FlagGems; an explicitly supplied generator remains isolated from that stream.

#include "../../generated/ops.h"
#include "../flagos/python_op_caller.h"
#include "topsaten_common.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Generator.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/randn.h>
#include <ATen/ops/randperm.h>
#include <ATen/ops/normal.h>
#include <ATen/ops/bernoulli.h>
#include <ATen/ops/binomial.h>
#include <ATen/ops/exponential.h>
#include <ATen/ops/multinomial.h>
#include <ATen/ops/native_dropout.h>
#include <ATen/ops/native_dropout_backward.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/poisson.h>
#include <ATen/ops/_sample_dirichlet.h>
#include <ATen/ops/_standard_gamma.h>
#include <ATen/InferSize.h>
#include <c10/core/Device.h>

#include "runtime/generator.h"

#include <limits>
#include <mutex>

namespace at::native::flagos {
namespace {

// One 64-bit seed per stochastic operation, drawn from the flagos PrivateUse1
// generator -- the same contract as the Ascend and MUSA backends.
//
// This must not be an `at::check_generator<at::CPUGeneratorImpl>` call even
// though the flagos generator derives from CPUGeneratorImpl: check_generator
// compares `device_type()`, which is kCPU there, so an explicit
// `torch.Generator(device="flagos")` was rejected with "Expected a 'cpu' device
// type for generator but found 'flagos'". Routing through ReserveSeed also puts
// generator-less draws on the generator that torch.flagos.manual_seed,
// get_rng_state and set_rng_state actually own, instead of a second philox
// state those entry points never touch.
uint64_t next_seed(
    const at::Tensor& tensor,
    const ::std::optional<at::Generator>& generator) {
  // A flagos generator on a CPU tensor is a dispatch-key error, not a request
  // to run the op on the device: `torch.empty(8).normal_(generator=flagos_gen)`
  // must raise rather than silently redispatch to PrivateUse1.
  TORCH_CHECK(
      tensor.device().is_privateuseone(),
      "Expected a matching device type for generator and tensor, but found a "
      "flagos generator with ", tensor.device(), " tensor");
  return c10::flagos::ReserveSeed(
      generator, static_cast<c10::DeviceIndex>(tensor.device().index()));
}

// A CPU generator seeded from the reserved flagos seed. The CPU-fallback paths
// below must not hand their flagos generator straight to a CPU ATen kernel --
// that hits the same device-type check -- but they still have to consume from
// the flagos stream so a fallback and a native draw stay ordered.
at::Generator cpu_generator_for(
    const at::Tensor& tensor,
    const ::std::optional<at::Generator>& generator) {
  return at::detail::createCPUGenerator(next_seed(tensor, generator));
}

// Same, for the factory fallbacks: there is no device tensor to key on yet,
// only the requested target device.
at::Generator cpu_generator_for_device(
    const ::std::optional<at::Device>& device,
    const ::std::optional<at::Generator>& generator) {
  auto index = device.has_value() ? device->index() : 0;
  return at::detail::createCPUGenerator(c10::flagos::ReserveSeed(
      generator, static_cast<c10::DeviceIndex>(index)));
}

topsatenGenerator_t make_generator(
    const ::std::optional<at::Generator>& generator,
    const at::Tensor& output) {
  // Offset zero: each operation starts its own philox stream from a freshly
  // reserved seed, matching how FlagGems consumes `(seed, 0)` per launch.
  return topsatenGenerator_t{next_seed(output, generator), 0};
}

at::Tensor cpu_to_device(const at::Tensor& cpu, const at::Device& device) {
  return cpu.to(device);
}

at::Tensor make_empty(
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  return at::empty(
      size, at::TensorOptions()
                .dtype(dtype.value_or(at::kFloat))
                .layout(layout.value_or(at::kStrided))
                .device(device.value_or(at::Device(at::kPrivateUse1, 0)))
                .pinned_memory(pin_memory.value_or(false)));
}

bool supported(const at::Tensor& tensor) {
  return tensor.defined() && tensor.numel() > 0 &&
      gcu::TopsatenSupportsDtype(tensor.scalar_type());
}

at::Tensor randn_cpu(
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    const ::std::optional<at::Generator>& generator) {
  auto target = device.value_or(at::Device(at::kPrivateUse1, 0));
  auto cpu = at::empty(
      size, at::TensorOptions()
                .dtype(dtype.value_or(at::kFloat))
                .layout(layout.value_or(at::kStrided))
                .device(at::kCPU)
                .pinned_memory(pin_memory.value_or(false)));
  cpu.normal_(0.0, 1.0, cpu_generator_for_device(target, generator));
  return cpu_to_device(cpu, target);
}

at::Tensor randint_cpu(
    int64_t low,
    int64_t high,
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    const ::std::optional<at::Generator>& generator) {
  auto target = device.value_or(at::Device(at::kPrivateUse1, 0));
  auto cpu = at::empty(
      size, at::TensorOptions()
                .dtype(dtype.value_or(at::kLong))
                .layout(layout.value_or(at::kStrided))
                .device(at::kCPU)
                .pinned_memory(pin_memory.value_or(false)));
  cpu.random_(low, high, cpu_generator_for_device(target, generator));
  return cpu_to_device(cpu, target);
}

// Exclusive upper bound for random_ when the caller omits `to`, mirroring
// ATen's per-dtype contract (and the Ascend table).
//
// topsatenRandom's no-bound overload does not honour it: on an int8 tensor it
// fills the full signed range, while ATen documents [0, iinfo(int8).max]. The
// bounded overload is correct, so the default overloads pass an explicit range
// rather than forwarding the bare call.
int64_t default_upper_bound(at::ScalarType dtype) {
  switch (dtype) {
    case at::kBool: return 2;
    case at::kByte: return 256;
    case at::kChar: return 128;
    case at::kShort: return 32768;
    case at::kInt:
      return int64_t(std::numeric_limits<int32_t>::max()) + 1;
    case at::kLong:
      // int64_t cannot represent max + 1, so the exclusive upper bound loses
      // one value out of 2^63, matching the Ascend limitation.
      return std::numeric_limits<int64_t>::max();
    case at::kHalf: return int64_t{1} << 11;
    case at::kBFloat16: return int64_t{1} << 8;
    case at::kFloat: return int64_t{1} << 24;
    case at::kDouble: return int64_t{1} << 53;
    default:
      TORCH_CHECK(false, "unsupported random_ dtype: ", dtype);
  }
}

} // namespace

at::Tensor RandnKernelGcu(
    at::IntArrayRef size, ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout, ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  if (!supported(out)) return randn_cpu(size, dtype, layout, device, pin_memory, {});
  auto gen = make_generator({}, out);
  gcu::TopsatenSizeWrapper shape(size);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(topsatenRandn, out, t_out.get(), shape.get(),
                    gcu::ToTopsatenDataType(out.scalar_type()),
                    TOPSATEN_LAYOUT_STRIDED, pin_memory.value_or(false), gen);
  return out;
}

at::Tensor RandnGeneratorKernelGcu(
    at::IntArrayRef size, ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  if (!supported(out)) {
    return randn_cpu(size, dtype, layout, device, pin_memory, generator);
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenSizeWrapper shape(size);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(topsatenRandn, out, t_out.get(), shape.get(),
                    gcu::ToTopsatenDataType(out.scalar_type()),
                    TOPSATEN_LAYOUT_STRIDED, pin_memory.value_or(false), gen);
  return out;
}

at::Tensor RandnLikeGeneratorKernelGcu(
    const at::Tensor& self, ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  auto out = at::empty_like(self, dtype, layout, device, pin_memory, memory_format);
  if (!supported(out)) {
    auto cpu = at::empty_like(self.cpu(), dtype, layout, at::kCPU, pin_memory,
                               memory_format);
    cpu.normal_(0.0, 1.0, cpu_generator_for(out, generator));
    return cpu_to_device(cpu, self.device());
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out), t_self(self);
  EXEC_TOPSATEN_CMD(topsatenRandnLike, out, t_out.get(), t_self.get(),
                    gcu::ToTopsatenDataType(out.scalar_type()),
                    TOPSATEN_LAYOUT_STRIDED, pin_memory.value_or(false),
                    TOPSATEN_MEMORY_CONTIGUOUS, gen);
  return out;
}

at::Tensor& RandnLikeGeneratorOutKernelGcu(
    const at::Tensor& self, ::std::optional<at::Generator> generator,
    ::std::optional<at::MemoryFormat> memory_format, at::Tensor& out) {
  if (!supported(out)) {
    auto cpu = out.cpu();
    cpu.normal_(0.0, 1.0, cpu_generator_for(out, generator));
    out.copy_(cpu);
    return out;
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out), t_self(self);
  EXEC_TOPSATEN_CMD(topsatenRandnLike, out, t_out.get(), t_self.get(),
                    gcu::ToTopsatenDataType(out.scalar_type()),
                    TOPSATEN_LAYOUT_STRIDED, false,
                    TOPSATEN_MEMORY_CONTIGUOUS, gen);
  return out;
}

at::Tensor RandintLowGeneratorKernelGcu(
    int64_t low, int64_t high, at::IntArrayRef size,
    ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout, ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory);

at::Tensor RandintGeneratorKernelGcu(
    int64_t high, at::IntArrayRef size, ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  return RandintLowGeneratorKernelGcu(0, high, size, generator, dtype, layout,
                                      device, pin_memory);
}

at::Tensor RandintLowGeneratorKernelGcu(
    int64_t low, int64_t high, at::IntArrayRef size,
    ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout, ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  auto out = make_empty(size, dtype.value_or(at::kLong), layout, device, pin_memory);
  if (!supported(out)) {
    return randint_cpu(low, high, size, dtype, layout, device, pin_memory, generator);
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenSizeWrapper shape(size);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(topsatenRandint, out, t_out.get(), low, high, shape.get(),
                    gcu::ToTopsatenDataType(out.scalar_type()),
                    TOPSATEN_LAYOUT_STRIDED, pin_memory.value_or(false), gen);
  return out;
}

at::Tensor RandpermGeneratorKernelGcu(
    int64_t n, ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  auto out = make_empty({n}, dtype.value_or(at::kLong), layout, device, pin_memory);
  if (!supported(out)) {
    auto cpu = at::randperm(n, cpu_generator_for(out, generator),
                            out.options().device(at::kCPU));
    return cpu_to_device(cpu, out.device());
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(topsatenRandperm, out, t_out.get(), n,
                    gcu::ToTopsatenDataType(out.scalar_type()),
                    TOPSATEN_LAYOUT_STRIDED, pin_memory.value_or(false), gen);
  return out;
}

at::Tensor& RandomInplaceToKernelGcu(
    at::Tensor& self, int64_t to, ::std::optional<at::Generator> generator) {
  if (!supported(self)) {
    auto cpu = self.cpu();
    cpu.random_(to, cpu_generator_for(self, generator));
    self.copy_(cpu);
    return self;
  }
  auto gen = make_generator(generator, self);
  gcu::TopsatenTensorWrapper t_self(self);
  EXEC_TOPSATEN_CMD(topsatenRandom, self, t_self.get(), to, gen);
  return self;
}

// random_() with no bound. Delegates to the [0, to) overload rather than
// topsaten's own no-bound form, which fills the dtype's full signed range --
// ATen's contract is [0, iinfo(dtype).max].
at::Tensor& RandomInplaceKernelGcu(
    at::Tensor& self, ::std::optional<at::Generator> generator) {
  if (!supported(self)) {
    auto cpu = self.cpu();
    cpu.random_(cpu_generator_for(self, generator));
    self.copy_(cpu);
    return self;
  }
  return RandomInplaceToKernelGcu(
      self, default_upper_bound(self.scalar_type()), generator);
}

// random_(from, to). An absent `to` means the dtype's default upper bound.
at::Tensor& RandomInplaceFromKernelGcu(
    at::Tensor& self, int64_t from, ::std::optional<int64_t> to,
    ::std::optional<at::Generator> generator) {
  auto upper = to.has_value() ? *to : default_upper_bound(self.scalar_type());
  TORCH_CHECK(from < upper, "random_ lower bound out of range");
  if (!supported(self)) {
    auto cpu = self.cpu();
    cpu.random_(from, upper, cpu_generator_for(self, generator));
    self.copy_(cpu);
    return self;
  }
  auto gen = make_generator(generator, self);
  gcu::TopsatenTensorWrapper t_self(self);
  EXEC_TOPSATEN_CMD(topsatenRandom, self, t_self.get(), from, &upper, gen);
  return self;
}

// --- uniform family ---------------------------------------------------------
//
// topsatenRngUniform fills [from, to). Everything in the rand/rand_like family
// is expressed through it so one native kernel covers the whole group.

at::Tensor& UniformInplaceKernelGcu(
    at::Tensor& self, double from, double to,
    ::std::optional<at::Generator> generator) {
  if (!supported(self)) {
    auto cpu = self.cpu();
    cpu.uniform_(from, to, cpu_generator_for(self, generator));
    self.copy_(cpu);
    return self;
  }
  auto gen = make_generator(generator, self);
  gcu::TopsatenTensorWrapper t_self(self);
  EXEC_TOPSATEN_CMD(topsatenRngUniform, self, t_self.get(), from, to, gen);
  return self;
}

at::Tensor RandGeneratorKernelGcu(
    at::IntArrayRef size, ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  return UniformInplaceKernelGcu(out, 0.0, 1.0, generator);
}

at::Tensor RandKernelGcu(
    at::IntArrayRef size, ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout, ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  return RandGeneratorKernelGcu(size, {}, dtype, layout, device, pin_memory);
}

at::Tensor& RandOutKernelGcu(at::IntArrayRef size, at::Tensor& out) {
  out.resize_(size);
  return UniformInplaceKernelGcu(out, 0.0, 1.0, {});
}

at::Tensor& RandNamesOutKernelGcu(
    at::IntArrayRef size, ::std::optional<at::DimnameList> names,
    at::Tensor& out) {
  TORCH_CHECK(!names.has_value(), "rand.names_out: named tensors are not supported on flagos");
  return RandOutKernelGcu(size, out);
}

at::Tensor RandLikeGeneratorKernelGcu(
    const at::Tensor& self, ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  auto out = at::empty_like(self, dtype, layout, device, pin_memory, memory_format);
  return UniformInplaceKernelGcu(out, 0.0, 1.0, generator);
}

at::Tensor RandLikeKernelGcu(
    const at::Tensor& self, ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout, ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  return RandLikeGeneratorKernelGcu(self, {}, dtype, layout, device, pin_memory,
                                    memory_format);
}

at::Tensor& RandLikeOutKernelGcu(
    const at::Tensor& self, ::std::optional<at::MemoryFormat> memory_format,
    at::Tensor& out) {
  return UniformInplaceKernelGcu(out, 0.0, 1.0, {});
}

// --- normal family ----------------------------------------------------------

at::Tensor& NormalInplaceKernelGcu(
    at::Tensor& self, double mean, double std,
    ::std::optional<at::Generator> generator) {
  if (!supported(self)) {
    auto cpu = self.cpu();
    cpu.normal_(mean, std, cpu_generator_for(self, generator));
    self.copy_(cpu);
    return self;
  }
  auto gen = make_generator(generator, self);
  gcu::TopsatenTensorWrapper t_self(self);
  EXEC_TOPSATEN_CMD(topsatenNormal, self, t_self.get(), mean, std, gen);
  return self;
}

at::Tensor NormalFloatFloatKernelGcu(
    double mean, double std, at::IntArrayRef size,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  return NormalInplaceKernelGcu(out, mean, std, generator);
}

// normal(Tensor mean, float std) and the two remaining overloads take a
// per-element parameter tensor. topsaten exposes matching overloads, but the
// broadcast between mean and std is ATen's job, so the shape is resolved here
// and the elementwise draw is expressed as `mean + std * standard_normal`,
// which keeps one native call per op and one seed per operation.
at::Tensor NormalTensorFloatKernelGcu(
    const at::Tensor& mean, double std,
    ::std::optional<at::Generator> generator) {
  auto out = at::empty_like(mean);
  NormalInplaceKernelGcu(out, 0.0, 1.0, generator);
  return mean + out * std;
}

at::Tensor NormalTensorTensorKernelGcu(
    const at::Tensor& mean, const at::Tensor& std,
    ::std::optional<at::Generator> generator) {
  auto shape = at::infer_size(mean.sizes(), std.sizes());
  auto out = at::empty(shape, mean.options());
  NormalInplaceKernelGcu(out, 0.0, 1.0, generator);
  return mean + out * std;
}

// --- integer factories ------------------------------------------------------

at::Tensor RandintLowKernelGcu(
    int64_t low, int64_t high, at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  return RandintLowGeneratorKernelGcu(low, high, size, {}, dtype, layout,
                                      device, pin_memory);
}

at::Tensor RandintKernelGcu(
    int64_t high, at::IntArrayRef size, ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout, ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  return RandintLowKernelGcu(0, high, size, dtype, layout, device, pin_memory);
}

at::Tensor& RandintLowOutKernelGcu(
    int64_t low, int64_t high, at::IntArrayRef size, at::Tensor& out) {
  out.resize_(size);
  return RandomInplaceFromKernelGcu(out, low, high, {});
}

at::Tensor& RandintOutKernelGcu(
    int64_t high, at::IntArrayRef size, at::Tensor& out) {
  return RandintLowOutKernelGcu(0, high, size, out);
}

at::Tensor RandintLikeLowDtypeKernelGcu(
    const at::Tensor& self, int64_t low, int64_t high,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  auto out = at::empty_like(self, dtype, layout, device, pin_memory, memory_format);
  return RandomInplaceFromKernelGcu(out, low, high, {});
}

at::Tensor RandintLikeKernelGcu(
    const at::Tensor& self, int64_t high, ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout, ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  return RandintLikeLowDtypeKernelGcu(self, 0, high, dtype, layout, device,
                                      pin_memory, memory_format);
}

at::Tensor& RandintLikeLowDtypeOutKernelGcu(
    const at::Tensor& self, int64_t low, int64_t high,
    ::std::optional<at::MemoryFormat> memory_format, at::Tensor& out) {
  return RandomInplaceFromKernelGcu(out, low, high, {});
}

at::Tensor& RandintLikeOutKernelGcu(
    const at::Tensor& self, int64_t high,
    ::std::optional<at::MemoryFormat> memory_format, at::Tensor& out) {
  return RandomInplaceFromKernelGcu(out, 0, high, {});
}

at::Tensor RandpermKernelGcu(
    int64_t n, ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout, ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  return RandpermGeneratorKernelGcu(n, {}, dtype, layout, device, pin_memory);
}

at::Tensor& RandpermOutKernelGcu(int64_t n, at::Tensor& out) {
  out.resize_({n});
  // topsatenRandperm writes a permutation of [0, n) directly into `out`.
  if (!supported(out)) {
    auto cpu = at::randperm(n, cpu_generator_for(out, {}),
                            out.options().device(at::kCPU));
    out.copy_(cpu);
    return out;
  }
  auto gen = make_generator({}, out);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(topsatenRandperm, out, t_out.get(), n,
                    gcu::ToTopsatenDataType(out.scalar_type()),
                    TOPSATEN_LAYOUT_STRIDED, false, gen);
  return out;
}

// --- randn_like without a generator ----------------------------------------

at::Tensor RandnLikeKernelGcu(
    const at::Tensor& self, ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout, ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  return RandnLikeGeneratorKernelGcu(self, {}, dtype, layout, device,
                                     pin_memory, memory_format);
}

at::Tensor& RandnLikeOutKernelGcu(
    const at::Tensor& self, ::std::optional<at::MemoryFormat> memory_format,
    at::Tensor& out) {
  return RandnLikeGeneratorOutKernelGcu(self, {}, memory_format, out);
}

at::Tensor& RandnNamesOutKernelGcu(
    at::IntArrayRef size, ::std::optional<at::DimnameList> names,
    at::Tensor& out) {
  TORCH_CHECK(!names.has_value(), "randn.names_out: named tensors are not supported on flagos");
  out.resize_(size);
  return NormalInplaceKernelGcu(out, 0.0, 1.0, {});
}

// --- dropout ----------------------------------------------------------------

::std::tuple<at::Tensor, at::Tensor> NativeDropoutKernelGcu(
    const at::Tensor& input, double p, ::std::optional<bool> train) {
  // Eval mode, or p == 0: identity with an all-true mask, no draw consumed.
  if (!train.value_or(true) || p == 0.0) {
    return {input.clone(),
            at::ones(input.sizes(), input.options().dtype(at::kBool))};
  }
  auto out = at::empty_like(input);
  auto mask = at::empty(input.sizes(), input.options().dtype(at::kBool));
  if (!supported(input)) {
    auto [cpu_out, cpu_mask] = at::native_dropout(input.cpu(), p, train);
    out.copy_(cpu_out);
    mask.copy_(cpu_mask);
    return {out, mask};
  }
  auto gen = make_generator({}, out);
  gcu::TopsatenTensorWrapper t_out(out), t_mask(mask), t_input(input);
  EXEC_TOPSATEN_CMD(topsatenNativeDropout, out, t_out.get(), t_mask.get(),
                    t_input.get(), p, true, gen);
  return {out, mask};
}

at::Tensor NativeDropoutBackwardKernelGcu(
    const at::Tensor& grad_output, const at::Tensor& mask, double scale) {
  auto out = at::empty_like(grad_output);
  if (!supported(grad_output)) {
    auto cpu = at::native_dropout_backward(grad_output.cpu(), mask.cpu(), scale);
    out.copy_(cpu);
    return out;
  }
  gcu::TopsatenTensorWrapper t_out(out), t_grad(grad_output), t_mask(mask);
  EXEC_TOPSATEN_CMD(topsatenNativeDropoutBackward, out, t_out.get(),
                    t_grad.get(), t_mask.get(), scale);
  return out;
}

// --- distributions with no topsaten kernel ---------------------------------
//
// _standard_gamma and _sample_dirichlet have no topsaten entry point, so they
// take the CPU reference with a seed reserved from the flagos generator -- the
// same treatment Ascend gives them.

at::Tensor PrivStandardGammaKernelGcu(
    const at::Tensor& self, ::std::optional<at::Generator> generator) {
  return at::_standard_gamma(self.cpu(), cpu_generator_for(self, generator))
      .to(self.device());
}

at::Tensor PrivSampleDirichletKernelGcu(
    const at::Tensor& self, ::std::optional<at::Generator> generator) {
  return at::_sample_dirichlet(self.cpu(), cpu_generator_for(self, generator))
      .to(self.device());
}

at::Tensor BinomialKernelGcu(
    const at::Tensor& count, const at::Tensor& prob,
    ::std::optional<at::Generator> generator) {
  auto shape = at::infer_size(count.sizes(), prob.sizes());
  auto out = at::empty(shape, count.options());
  if (!supported(count) || !supported(prob)) {
    return at::binomial(count.cpu(), prob.cpu(),
                        cpu_generator_for(out, generator))
        .to(count.device());
  }
  auto count_b = count.expand(shape).contiguous();
  auto prob_b = prob.expand(shape).contiguous();
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out), t_count(count_b), t_prob(prob_b);
  EXEC_TOPSATEN_CMD(topsatenBinomial, out, t_out.get(), t_count.get(),
                    t_prob.get(), gen);
  return out;
}

at::Tensor& BernoulliInplaceTensorKernelGcu(
    at::Tensor& self, const at::Tensor& p,
    ::std::optional<at::Generator> generator) {
  if (!supported(self) || !supported(p)) {
    auto cpu = self.cpu();
    cpu.bernoulli_(p.cpu(), cpu_generator_for(self, generator));
    self.copy_(cpu);
    return self;
  }
  auto p_b = p.expand(self.sizes()).contiguous().to(self.scalar_type());
  auto gen = make_generator(generator, self);
  gcu::TopsatenTensorWrapper t_self(self), t_p(p_b);
  EXEC_TOPSATEN_CMD(topsatenBernoulli_, self, t_self.get(), t_p.get(), gen);
  return self;
}

at::Tensor& BernoulliInplaceFloatKernelGcu(
    at::Tensor& self, double p, ::std::optional<at::Generator> generator) {
  if (!supported(self)) {
    auto cpu = self.cpu();
    cpu.bernoulli_(p, cpu_generator_for(self, generator));
    self.copy_(cpu);
    return self;
  }
  auto gen = make_generator(generator, self);
  gcu::TopsatenTensorWrapper t_self(self);
  EXEC_TOPSATEN_CMD(topsatenBernoulli, self, t_self.get(), p, gen);
  return self;
}

at::Tensor BernoulliKernelGcu(
    const at::Tensor& self, ::std::optional<at::Generator> generator) {
  auto out = at::empty_like(self);
  if (!supported(out)) {
    return at::bernoulli(self.cpu(), cpu_generator_for(out, generator))
        .to(self.device());
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out), t_self(self);
  EXEC_TOPSATEN_CMD(topsatenBernoulli, out, t_out.get(), t_self.get(), gen);
  return out;
}

at::Tensor ExponentialKernelGcu(
    const at::Tensor& self, double lambda, ::std::optional<at::Generator> generator) {
  auto out = at::empty_like(self);
  if (!supported(out)) {
    return at::exponential(self.cpu(), lambda, cpu_generator_for(out, generator))
        .to(self.device());
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(topsatenExponential, out, t_out.get(), lambda, gen);
  return out;
}

at::Tensor& ExponentialInplaceKernelGcu(
    at::Tensor& self, double lambda, ::std::optional<at::Generator> generator) {
  if (!supported(self)) {
    auto cpu = self.cpu();
    cpu.exponential_(lambda, cpu_generator_for(self, generator));
    self.copy_(cpu);
    return self;
  }
  auto gen = make_generator(generator, self);
  gcu::TopsatenTensorWrapper t_self(self);
  EXEC_TOPSATEN_CMD(topsatenExponential, self, t_self.get(), lambda, gen);
  return self;
}

at::Tensor MultinomialKernelGcu(
    const at::Tensor& self, int64_t num_samples, bool replacement,
    ::std::optional<at::Generator> generator) {
  auto out = at::empty({self.size(0), num_samples},
                       self.options().dtype(at::kLong));
  if (!supported(self) || !gcu::TopsatenSupportsDtype(out.scalar_type())) {
    return at::multinomial(self.cpu(), num_samples, replacement,
                           cpu_generator_for(out, generator))
        .to(self.device());
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out), t_self(self);
  EXEC_TOPSATEN_CMD(topsatenMultinomial, out, t_out.get(), t_self.get(),
                    num_samples, replacement, gen);
  return out;
}

at::Tensor PoissonKernelGcu(
    const at::Tensor& self, ::std::optional<at::Generator> generator) {
  auto out = at::empty_like(self);
  if (!supported(self)) {
    return at::poisson(self.cpu(), cpu_generator_for(out, generator))
        .to(self.device());
  }
  auto gen = make_generator(generator, out);
  gcu::TopsatenTensorWrapper t_out(out), t_self(self);
  EXEC_TOPSATEN_CMD(topsatenPoisson, out, t_out.get(), t_self.get(), gen);
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(RandFn, rand_dispatcher, Backend::kGcu, RandKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandGeneratorFn, rand_generator_dispatcher, Backend::kGcu, RandGeneratorKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandOutFn, rand_out_dispatcher, Backend::kGcu, RandOutKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandNamesOutFn, rand_names_out_dispatcher, Backend::kGcu, RandNamesOutKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandLikeFn, rand_like_dispatcher, Backend::kGcu, RandLikeKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandLikeGeneratorFn, rand_like_generator_dispatcher, Backend::kGcu, RandLikeGeneratorKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandLikeOutFn, rand_like_out_dispatcher, Backend::kGcu, RandLikeOutKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(UniformInplaceFn, uniform_inplace_dispatcher, Backend::kGcu, UniformInplaceKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(NormalInplaceFn, normal_inplace_dispatcher, Backend::kGcu, NormalInplaceKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(NormalFloatFloatFn, normal_float_float_dispatcher, Backend::kGcu, NormalFloatFloatKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(NormalTensorFloatFn, normal_tensor_float_dispatcher, Backend::kGcu, NormalTensorFloatKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(NormalTensorTensorFn, normal_tensor_tensor_dispatcher, Backend::kGcu, NormalTensorTensorKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandintFn, randint_dispatcher, Backend::kGcu, RandintKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandintLowFn, randint_low_dispatcher, Backend::kGcu, RandintLowKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandintOutFn, randint_out_dispatcher, Backend::kGcu, RandintOutKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandintLowOutFn, randint_low_out_dispatcher, Backend::kGcu, RandintLowOutKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandintLikeFn, randint_like_dispatcher, Backend::kGcu, RandintLikeKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandintLikeLowDtypeFn, randint_like_low_dtype_dispatcher, Backend::kGcu, RandintLikeLowDtypeKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandintLikeOutFn, randint_like_out_dispatcher, Backend::kGcu, RandintLikeOutKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandintLikeLowDtypeOutFn, randint_like_low_dtype_out_dispatcher, Backend::kGcu, RandintLikeLowDtypeOutKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandpermFn, randperm_dispatcher, Backend::kGcu, RandpermKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandpermOutFn, randperm_out_dispatcher, Backend::kGcu, RandpermOutKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeFn, randn_like_dispatcher, Backend::kGcu, RandnLikeKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeOutFn, randn_like_out_dispatcher, Backend::kGcu, RandnLikeOutKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandnNamesOutFn, randn_names_out_dispatcher, Backend::kGcu, RandnNamesOutKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandomInplaceFromFn, random_inplace_from_dispatcher, Backend::kGcu, RandomInplaceFromKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(NativeDropoutFn, native_dropout_dispatcher, Backend::kGcu, NativeDropoutKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(NativeDropoutBackwardFn, native_dropout_backward_dispatcher, Backend::kGcu, NativeDropoutBackwardKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(BernoulliInplaceTensorFn, bernoulli_inplace_tensor_dispatcher, Backend::kGcu, BernoulliInplaceTensorKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(BinomialFn, binomial_dispatcher, Backend::kGcu, BinomialKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(PrivStandardGammaFn, priv_standard_gamma_dispatcher, Backend::kGcu, PrivStandardGammaKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(PrivSampleDirichletFn, priv_sample_dirichlet_dispatcher, Backend::kGcu, PrivSampleDirichletKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandnFn, randn_dispatcher, Backend::kGcu, RandnKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandnGeneratorFn, randn_generator_dispatcher, Backend::kGcu, RandnGeneratorKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeGeneratorFn, randn_like_generator_dispatcher, Backend::kGcu, RandnLikeGeneratorKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeGeneratorOutFn, randn_like_generator_out_dispatcher, Backend::kGcu, RandnLikeGeneratorOutKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandintGeneratorFn, randint_generator_dispatcher, Backend::kGcu, RandintGeneratorKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandintLowGeneratorFn, randint_low_generator_dispatcher, Backend::kGcu, RandintLowGeneratorKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandpermGeneratorFn, randperm_generator_dispatcher, Backend::kGcu, RandpermGeneratorKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandomInplaceFn, random_inplace_dispatcher, Backend::kGcu, RandomInplaceKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(RandomInplaceToFn, random_inplace_to_dispatcher, Backend::kGcu, RandomInplaceToKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(BernoulliFn, bernoulli_dispatcher, Backend::kGcu, BernoulliKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(BernoulliInplaceFloatFn, bernoulli_inplace_float_dispatcher, Backend::kGcu, BernoulliInplaceFloatKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(ExponentialFn, exponential_dispatcher, Backend::kGcu, ExponentialKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(ExponentialInplaceFn, exponential_inplace_dispatcher, Backend::kGcu, ExponentialInplaceKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(MultinomialFn, multinomial_dispatcher, Backend::kGcu, MultinomialKernelGcu)
REGISTER_IMPL_TO_DISPATCHER(PoissonFn, poisson_dispatcher, Backend::kGcu, PoissonKernelGcu)

} // namespace at::native::flagos
