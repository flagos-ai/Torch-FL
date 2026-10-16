// Copyright (c) 2026, BAAI. All rights reserved.
//
// Random-number kernels for the TsingMicro (TX8) backend.
//
// The device has no random-number unit, so every draw is taken on the host and
// copied back -- the same "declined to the host" arm the GCU, Kunlunxin, Ascend
// and MUSA backends carry, and for the same reason: the seed plumbing is
// backend-independent, only the device-side draw would not be. This file
// therefore carries no vendor call at all (no tx_runtime, no txdnn), which is
// also why it needs no SDK to build.
//
// What it *does* provide is the seed contract the rest of the repo depends on --
// every operation reserves one 64-bit seed from the flagos PrivateUse1 generator
// (c10::flagos::ReserveSeed) instead of reaching for ATen's own default
// generator, so torch.flagos.manual_seed / get_rng_state / set_rng_state /
// manual_seed_all own the stream that the ops actually consume. That contract is
// what the shared tests/integration/ops/test_rng_dispatch.py suite asserts.
//
// Registration is split, unlike the other vendors: the REGISTER_IMPL_TO_DISPATCHER
// lines below fill Backend::kTsingMicro slots, but those slots are only *selected*
// when the conf names them, and tsingmicro's conf is hand-written and names no RNG
// op. The ops are therefore claimed for real by a direct m.impl() in
// csrc/aten/register.cc, next to the convolution wrappers; these slot fills exist
// so the codegen can route them in the future and so the shape matches its sibling
// rng.cc files. Without the direct claim the op reaches the generic CPU fallback,
// which redispatchs randn.generator on the CPU and trips its CPU-generator
// assertion against a flagos generator ("Expected a 'cpu' device type for
// generator but found 'flagos'").

#if defined(USE_TSINGMICRO)

#include "../../generated/ops.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Generator.h>
#include <ATen/InferSize.h>
#include <ATen/ops/_sample_dirichlet.h>
#include <ATen/ops/_standard_gamma.h>
#include <ATen/ops/bernoulli.h>
#include <ATen/ops/binomial.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/exponential.h>
#include <ATen/ops/multinomial.h>
#include <ATen/ops/native_dropout.h>
#include <ATen/ops/native_dropout_backward.h>
#include <ATen/ops/normal.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/poisson.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/randn.h>
#include <ATen/ops/randperm.h>
#include <c10/core/Device.h>

#include "runtime/generator.h"

#include <limits>

namespace at::native::flagos {
namespace {

// One 64-bit seed per stochastic operation, drawn from the flagos PrivateUse1
// generator -- the same contract as the Ascend, GCU and MUSA backends.
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

// A CPU generator seeded from the reserved flagos seed. The host draws below
// must not hand their flagos generator straight to a CPU ATen kernel -- that
// hits the same device-type check -- but they still have to consume from the
// flagos stream so a host draw stays ordered against every other backend's.
at::Generator cpu_generator_for(
    const at::Tensor& tensor,
    const ::std::optional<at::Generator>& generator) {
  return at::detail::createCPUGenerator(next_seed(tensor, generator));
}

// Same, for the factory form: there is no device tensor to key on yet, only the
// requested target device.
at::Generator cpu_generator_for_device(
    const ::std::optional<at::Device>& device,
    const ::std::optional<at::Generator>& generator) {
  auto index = device.has_value() ? device->index() : 0;
  return at::detail::createCPUGenerator(c10::flagos::ReserveSeed(
      generator, static_cast<c10::DeviceIndex>(index)));
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
int64_t default_upper_bound(at::ScalarType dtype) {
  switch (dtype) {
    case at::kBool:
      return 2;
    case at::kByte:
      return 256;
    case at::kChar:
      return 128;
    case at::kShort:
      return 32768;
    case at::kInt:
      return int64_t(std::numeric_limits<int32_t>::max()) + 1;
    case at::kLong:
      // int64_t cannot represent max + 1, so the exclusive upper bound loses
      // one value out of 2^63, matching the Ascend limitation.
      return std::numeric_limits<int64_t>::max();
    case at::kHalf:
      return int64_t{1} << 11;
    case at::kBFloat16:
      return int64_t{1} << 8;
    case at::kFloat:
      return int64_t{1} << 24;
    case at::kDouble:
      return int64_t{1} << 53;
    default:
      TORCH_CHECK(false, "unsupported random_ dtype: ", dtype);
  }
}

} // namespace

at::Tensor RandnKernelTsingMicro(
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  return randn_cpu(size, dtype, layout, device, pin_memory, {});
}

at::Tensor RandnGeneratorKernelTsingMicro(
    at::IntArrayRef size,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  return randn_cpu(size, dtype, layout, device, pin_memory, generator);
}

at::Tensor RandnLikeGeneratorKernelTsingMicro(
    const at::Tensor& self,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  auto cpu = at::empty_like(
      self.cpu(), dtype, layout, at::kCPU, pin_memory, memory_format);
  cpu.normal_(0.0, 1.0, cpu_generator_for(self, generator));
  return cpu_to_device(cpu, self.device());
}

at::Tensor& RandnLikeGeneratorOutKernelTsingMicro(
    const at::Tensor& self,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::MemoryFormat> memory_format,
    at::Tensor& out) {
  auto cpu = out.cpu();
  cpu.normal_(0.0, 1.0, cpu_generator_for(out, generator));
  out.copy_(cpu);
  return out;
}

at::Tensor RandintLowGeneratorKernelTsingMicro(
    int64_t low,
    int64_t high,
    at::IntArrayRef size,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory);

at::Tensor RandintGeneratorKernelTsingMicro(
    int64_t high,
    at::IntArrayRef size,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  return RandintLowGeneratorKernelTsingMicro(
      0, high, size, generator, dtype, layout, device, pin_memory);
}

at::Tensor RandintLowGeneratorKernelTsingMicro(
    int64_t low,
    int64_t high,
    at::IntArrayRef size,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  return randint_cpu(
      low, high, size, dtype, layout, device, pin_memory, generator);
}

at::Tensor RandpermGeneratorKernelTsingMicro(
    int64_t n,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  auto target = device.value_or(at::Device(at::kPrivateUse1, 0));
  auto cpu = at::randperm(
      n, cpu_generator_for_device(target, generator),
      at::TensorOptions()
          .dtype(dtype.value_or(at::kLong))
          .layout(layout.value_or(at::kStrided))
          .device(at::kCPU)
          .pinned_memory(pin_memory.value_or(false)));
  return cpu_to_device(cpu, target);
}

at::Tensor& RandomInplaceToKernelTsingMicro(
    at::Tensor& self, int64_t to, ::std::optional<at::Generator> generator) {
  auto cpu = self.cpu();
  cpu.random_(to, cpu_generator_for(self, generator));
  self.copy_(cpu);
  return self;
}

// random_() with no bound. Delegated to the [0, to) overload so the exclusive
// upper bound follows ATen's per-dtype contract rather than filling the dtype's
// full signed range.
at::Tensor& RandomInplaceKernelTsingMicro(
    at::Tensor& self, ::std::optional<at::Generator> generator) {
  auto cpu = self.cpu();
  cpu.random_(cpu_generator_for(self, generator));
  self.copy_(cpu);
  return self;
}

// random_(from, to). An absent `to` means the dtype's default upper bound.
at::Tensor& RandomInplaceFromKernelTsingMicro(
    at::Tensor& self,
    int64_t from,
    ::std::optional<int64_t> to,
    ::std::optional<at::Generator> generator) {
  auto upper = to.has_value() ? *to : default_upper_bound(self.scalar_type());
  TORCH_CHECK(from < upper, "random_ lower bound out of range");
  auto cpu = self.cpu();
  cpu.random_(from, upper, cpu_generator_for(self, generator));
  self.copy_(cpu);
  return self;
}

at::Tensor& UniformInplaceKernelTsingMicro(
    at::Tensor& self,
    double from,
    double to,
    ::std::optional<at::Generator> generator) {
  auto cpu = self.cpu();
  cpu.uniform_(from, to, cpu_generator_for(self, generator));
  self.copy_(cpu);
  return self;
}

at::Tensor RandGeneratorKernelTsingMicro(
    at::IntArrayRef size,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  return UniformInplaceKernelTsingMicro(out, 0.0, 1.0, generator);
}

at::Tensor RandKernelTsingMicro(
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  return RandGeneratorKernelTsingMicro(size, {}, dtype, layout, device, pin_memory);
}

at::Tensor& RandOutKernelTsingMicro(at::IntArrayRef size, at::Tensor& out) {
  out.resize_(size);
  return UniformInplaceKernelTsingMicro(out, 0.0, 1.0, {});
}

at::Tensor& RandNamesOutKernelTsingMicro(
    at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor& out) {
  TORCH_CHECK(
      !names.has_value(),
      "rand.names_out: named tensors are not supported on flagos");
  return RandOutKernelTsingMicro(size, out);
}

at::Tensor RandLikeGeneratorKernelTsingMicro(
    const at::Tensor& self,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  auto out =
      at::empty_like(self, dtype, layout, device, pin_memory, memory_format);
  return UniformInplaceKernelTsingMicro(out, 0.0, 1.0, generator);
}

at::Tensor RandLikeKernelTsingMicro(
    const at::Tensor& self,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  return RandLikeGeneratorKernelTsingMicro(
      self, {}, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor& RandLikeOutKernelTsingMicro(
    const at::Tensor& self,
    ::std::optional<at::MemoryFormat> memory_format,
    at::Tensor& out) {
  return UniformInplaceKernelTsingMicro(out, 0.0, 1.0, {});
}

at::Tensor& NormalInplaceKernelTsingMicro(
    at::Tensor& self,
    double mean,
    double std,
    ::std::optional<at::Generator> generator) {
  auto cpu = self.cpu();
  cpu.normal_(mean, std, cpu_generator_for(self, generator));
  self.copy_(cpu);
  return self;
}

at::Tensor NormalFloatFloatKernelTsingMicro(
    double mean,
    double std,
    at::IntArrayRef size,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  return NormalInplaceKernelTsingMicro(out, mean, std, generator);
}

// normal(Tensor mean, float std) and the remaining overloads take a per-element
// parameter tensor. The broadcast between mean and std is ATen's job, so the
// shape is resolved here and the elementwise draw is expressed as
// `mean + std * standard_normal`, which keeps one seed per operation.
at::Tensor NormalTensorFloatKernelTsingMicro(
    const at::Tensor& mean, double std, ::std::optional<at::Generator> generator) {
  auto out = at::empty_like(mean);
  NormalInplaceKernelTsingMicro(out, 0.0, 1.0, generator);
  return mean + out * std;
}

at::Tensor NormalTensorTensorKernelTsingMicro(
    const at::Tensor& mean,
    const at::Tensor& std,
    ::std::optional<at::Generator> generator) {
  auto shape = at::infer_size(mean.sizes(), std.sizes());
  auto out = at::empty(shape, mean.options());
  NormalInplaceKernelTsingMicro(out, 0.0, 1.0, generator);
  return mean + out * std;
}

at::Tensor RandintLowKernelTsingMicro(
    int64_t low,
    int64_t high,
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  return RandintLowGeneratorKernelTsingMicro(
      low, high, size, {}, dtype, layout, device, pin_memory);
}

at::Tensor RandintKernelTsingMicro(
    int64_t high,
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  return RandintLowKernelTsingMicro(0, high, size, dtype, layout, device, pin_memory);
}

at::Tensor& RandintLowOutKernelTsingMicro(
    int64_t low, int64_t high, at::IntArrayRef size, at::Tensor& out) {
  out.resize_(size);
  return RandomInplaceFromKernelTsingMicro(out, low, high, {});
}

at::Tensor& RandintOutKernelTsingMicro(
    int64_t high, at::IntArrayRef size, at::Tensor& out) {
  return RandintLowOutKernelTsingMicro(0, high, size, out);
}

at::Tensor RandintLikeLowDtypeKernelTsingMicro(
    const at::Tensor& self,
    int64_t low,
    int64_t high,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  auto out =
      at::empty_like(self, dtype, layout, device, pin_memory, memory_format);
  return RandomInplaceFromKernelTsingMicro(out, low, high, {});
}

at::Tensor RandintLikeKernelTsingMicro(
    const at::Tensor& self,
    int64_t high,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  return RandintLikeLowDtypeKernelTsingMicro(
      self, 0, high, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor& RandintLikeLowDtypeOutKernelTsingMicro(
    const at::Tensor& self,
    int64_t low,
    int64_t high,
    ::std::optional<at::MemoryFormat> memory_format,
    at::Tensor& out) {
  return RandomInplaceFromKernelTsingMicro(out, low, high, {});
}

at::Tensor& RandintLikeOutKernelTsingMicro(
    const at::Tensor& self,
    int64_t high,
    ::std::optional<at::MemoryFormat> memory_format,
    at::Tensor& out) {
  return RandomInplaceFromKernelTsingMicro(out, 0, high, {});
}

at::Tensor RandpermKernelTsingMicro(
    int64_t n,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  return RandpermGeneratorKernelTsingMicro(n, {}, dtype, layout, device, pin_memory);
}

at::Tensor& RandpermOutKernelTsingMicro(int64_t n, at::Tensor& out) {
  out.resize_({n});
  auto cpu = at::randperm(
      n, cpu_generator_for(out, {}), out.options().device(at::kCPU));
  out.copy_(cpu);
  return out;
}

at::Tensor RandnLikeKernelTsingMicro(
    const at::Tensor& self,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  return RandnLikeGeneratorKernelTsingMicro(
      self, {}, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor& RandnLikeOutKernelTsingMicro(
    const at::Tensor& self,
    ::std::optional<at::MemoryFormat> memory_format,
    at::Tensor& out) {
  return RandnLikeGeneratorOutKernelTsingMicro(self, {}, memory_format, out);
}

at::Tensor& RandnNamesOutKernelTsingMicro(
    at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor& out) {
  TORCH_CHECK(
      !names.has_value(),
      "randn.names_out: named tensors are not supported on flagos");
  out.resize_(size);
  return NormalInplaceKernelTsingMicro(out, 0.0, 1.0, {});
}

::std::tuple<at::Tensor, at::Tensor> NativeDropoutKernelTsingMicro(
    const at::Tensor& input, double p, ::std::optional<bool> train) {
  // Eval mode, or p == 0: identity with an all-true mask, no draw consumed.
  if (!train.value_or(true) || p == 0.0) {
    return {
        input.clone(),
        at::ones(input.sizes(), input.options().dtype(at::kBool))};
  }
  auto out = at::empty_like(input);
  auto mask = at::empty(input.sizes(), input.options().dtype(at::kBool));
  auto [cpu_out, cpu_mask] = at::native_dropout(input.cpu(), p, train);
  out.copy_(cpu_out);
  mask.copy_(cpu_mask);
  return {out, mask};
}

at::Tensor NativeDropoutBackwardKernelTsingMicro(
    const at::Tensor& grad_output, const at::Tensor& mask, double scale) {
  auto cpu = at::native_dropout_backward(grad_output.cpu(), mask.cpu(), scale);
  return cpu.to(grad_output.device());
}

// --- distributions with no vendor kernel ------------------------------------

at::Tensor PrivStandardGammaKernelTsingMicro(
    const at::Tensor& self, ::std::optional<at::Generator> generator) {
  return at::_standard_gamma(self.cpu(), cpu_generator_for(self, generator))
      .to(self.device());
}

at::Tensor PrivSampleDirichletKernelTsingMicro(
    const at::Tensor& self, ::std::optional<at::Generator> generator) {
  return at::_sample_dirichlet(self.cpu(), cpu_generator_for(self, generator))
      .to(self.device());
}

at::Tensor BinomialKernelTsingMicro(
    const at::Tensor& count,
    const at::Tensor& prob,
    ::std::optional<at::Generator> generator) {
  auto shape = at::infer_size(count.sizes(), prob.sizes());
  auto out = at::empty(shape, count.options());
  return at::binomial(
             count.cpu(), prob.cpu(), cpu_generator_for(out, generator))
      .to(count.device());
}

at::Tensor& BernoulliInplaceTensorKernelTsingMicro(
    at::Tensor& self,
    const at::Tensor& p,
    ::std::optional<at::Generator> generator) {
  auto cpu = self.cpu();
  cpu.bernoulli_(p.cpu(), cpu_generator_for(self, generator));
  self.copy_(cpu);
  return self;
}

at::Tensor& BernoulliInplaceFloatKernelTsingMicro(
    at::Tensor& self, double p, ::std::optional<at::Generator> generator) {
  auto cpu = self.cpu();
  cpu.bernoulli_(p, cpu_generator_for(self, generator));
  self.copy_(cpu);
  return self;
}

at::Tensor BernoulliKernelTsingMicro(
    const at::Tensor& self, ::std::optional<at::Generator> generator) {
  return at::bernoulli(self.cpu(), cpu_generator_for(self, generator))
      .to(self.device());
}

at::Tensor ExponentialKernelTsingMicro(
    const at::Tensor& self, double lambda, ::std::optional<at::Generator> generator) {
  return at::exponential(self.cpu(), lambda, cpu_generator_for(self, generator))
      .to(self.device());
}

at::Tensor& ExponentialInplaceKernelTsingMicro(
    at::Tensor& self, double lambda, ::std::optional<at::Generator> generator) {
  auto cpu = self.cpu();
  cpu.exponential_(lambda, cpu_generator_for(self, generator));
  self.copy_(cpu);
  return self;
}

at::Tensor MultinomialKernelTsingMicro(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    ::std::optional<at::Generator> generator) {
  auto out =
      at::empty({self.size(0), num_samples}, self.options().dtype(at::kLong));
  return at::multinomial(
             self.cpu(), num_samples, replacement,
             cpu_generator_for(out, generator))
      .to(self.device());
}

at::Tensor PoissonKernelTsingMicro(
    const at::Tensor& self, ::std::optional<at::Generator> generator) {
  return at::poisson(self.cpu(), cpu_generator_for(self, generator))
      .to(self.device());
}

REGISTER_IMPL_TO_DISPATCHER(RandFn, rand_dispatcher, Backend::kTsingMicro, RandKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandGeneratorFn, rand_generator_dispatcher, Backend::kTsingMicro, RandGeneratorKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandOutFn, rand_out_dispatcher, Backend::kTsingMicro, RandOutKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandNamesOutFn, rand_names_out_dispatcher, Backend::kTsingMicro, RandNamesOutKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandLikeFn, rand_like_dispatcher, Backend::kTsingMicro, RandLikeKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandLikeGeneratorFn, rand_like_generator_dispatcher, Backend::kTsingMicro, RandLikeGeneratorKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandLikeOutFn, rand_like_out_dispatcher, Backend::kTsingMicro, RandLikeOutKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(UniformInplaceFn, uniform_inplace_dispatcher, Backend::kTsingMicro, UniformInplaceKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(NormalInplaceFn, normal_inplace_dispatcher, Backend::kTsingMicro, NormalInplaceKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(NormalFloatFloatFn, normal_float_float_dispatcher, Backend::kTsingMicro, NormalFloatFloatKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(NormalTensorFloatFn, normal_tensor_float_dispatcher, Backend::kTsingMicro, NormalTensorFloatKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(NormalTensorTensorFn, normal_tensor_tensor_dispatcher, Backend::kTsingMicro, NormalTensorTensorKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandintFn, randint_dispatcher, Backend::kTsingMicro, RandintKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandintLowFn, randint_low_dispatcher, Backend::kTsingMicro, RandintLowKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandintOutFn, randint_out_dispatcher, Backend::kTsingMicro, RandintOutKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandintLowOutFn, randint_low_out_dispatcher, Backend::kTsingMicro, RandintLowOutKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandintLikeFn, randint_like_dispatcher, Backend::kTsingMicro, RandintLikeKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandintLikeLowDtypeFn, randint_like_low_dtype_dispatcher, Backend::kTsingMicro, RandintLikeLowDtypeKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandintLikeOutFn, randint_like_out_dispatcher, Backend::kTsingMicro, RandintLikeOutKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandintLikeLowDtypeOutFn, randint_like_low_dtype_out_dispatcher, Backend::kTsingMicro, RandintLikeLowDtypeOutKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandpermFn, randperm_dispatcher, Backend::kTsingMicro, RandpermKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandpermOutFn, randperm_out_dispatcher, Backend::kTsingMicro, RandpermOutKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeFn, randn_like_dispatcher, Backend::kTsingMicro, RandnLikeKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeOutFn, randn_like_out_dispatcher, Backend::kTsingMicro, RandnLikeOutKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandnNamesOutFn, randn_names_out_dispatcher, Backend::kTsingMicro, RandnNamesOutKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandomInplaceFromFn, random_inplace_from_dispatcher, Backend::kTsingMicro, RandomInplaceFromKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(NativeDropoutFn, native_dropout_dispatcher, Backend::kTsingMicro, NativeDropoutKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(NativeDropoutBackwardFn, native_dropout_backward_dispatcher, Backend::kTsingMicro, NativeDropoutBackwardKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(BernoulliInplaceTensorFn, bernoulli_inplace_tensor_dispatcher, Backend::kTsingMicro, BernoulliInplaceTensorKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(BinomialFn, binomial_dispatcher, Backend::kTsingMicro, BinomialKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(PrivStandardGammaFn, priv_standard_gamma_dispatcher, Backend::kTsingMicro, PrivStandardGammaKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(PrivSampleDirichletFn, priv_sample_dirichlet_dispatcher, Backend::kTsingMicro, PrivSampleDirichletKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandnFn, randn_dispatcher, Backend::kTsingMicro, RandnKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandnGeneratorFn, randn_generator_dispatcher, Backend::kTsingMicro, RandnGeneratorKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeGeneratorFn, randn_like_generator_dispatcher, Backend::kTsingMicro, RandnLikeGeneratorKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeGeneratorOutFn, randn_like_generator_out_dispatcher, Backend::kTsingMicro, RandnLikeGeneratorOutKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandintGeneratorFn, randint_generator_dispatcher, Backend::kTsingMicro, RandintGeneratorKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandintLowGeneratorFn, randint_low_generator_dispatcher, Backend::kTsingMicro, RandintLowGeneratorKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandpermGeneratorFn, randperm_generator_dispatcher, Backend::kTsingMicro, RandpermGeneratorKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandomInplaceFn, random_inplace_dispatcher, Backend::kTsingMicro, RandomInplaceKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(RandomInplaceToFn, random_inplace_to_dispatcher, Backend::kTsingMicro, RandomInplaceToKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(BernoulliFn, bernoulli_dispatcher, Backend::kTsingMicro, BernoulliKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(BernoulliInplaceFloatFn, bernoulli_inplace_float_dispatcher, Backend::kTsingMicro, BernoulliInplaceFloatKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(ExponentialFn, exponential_dispatcher, Backend::kTsingMicro, ExponentialKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(ExponentialInplaceFn, exponential_inplace_dispatcher, Backend::kTsingMicro, ExponentialInplaceKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(MultinomialFn, multinomial_dispatcher, Backend::kTsingMicro, MultinomialKernelTsingMicro)
REGISTER_IMPL_TO_DISPATCHER(PoissonFn, poisson_dispatcher, Backend::kTsingMicro, PoissonKernelTsingMicro)

} // namespace at::native::flagos

#endif // USE_TSINGMICRO
