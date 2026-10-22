// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include "runtime/generator.h"
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

namespace {

// Reserve every seed from the public flagos generator. This keeps native
// Ascend RNG and torch.flagos state APIs on one stream per device.
uint64_t next_seed(const at::Tensor& tensor,
                   const ::std::optional<at::Generator>& generator) {
  TORCH_CHECK(
      tensor.device().is_privateuseone(),
      "Expected a matching device type for generator and tensor, but found a "
      "flagos generator with ", tensor.device(), " tensor");
  return c10::flagos::ReserveSeed(
      generator, static_cast<c10::DeviceIndex>(tensor.device().index()));
}

uint64_t next_seed(const at::Tensor& tensor) {
  return next_seed(tensor, std::nullopt);
}

at::Generator cpu_generator_for(
    const at::Tensor& tensor,
    const ::std::optional<at::Generator>& generator) {
  return at::detail::createCPUGenerator(next_seed(tensor, generator));
}

at::Tensor make_empty(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype,
                      ::std::optional<at::Layout> layout,
                      ::std::optional<at::Device> device,
                      ::std::optional<bool> pin_memory) {
  auto options = at::TensorOptions()
      .dtype(dtype.value_or(at::kFloat))
      .layout(layout.value_or(at::kStrided))
      .device(device.value_or(at::Device(at::kPrivateUse1, 0)))
      .pinned_memory(pin_memory.value_or(false));
  return at::empty(size, options);
}

// aclnnInplaceNormal on `self`, with mean/std passed through a typed signature.
//
// The typed EXEC_ASCEND_CMD_SIG is mandatory here, not stylistic:
// aclnnInplaceNormalGetWorkspaceSize declares `float mean, float std`, and the
// plain macro calls through `int (*)(...)`. Varargs promotion passes each float
// as a 64-bit double, and the callee's `float` read takes only its low 32 bits --
// which for 1.0 is exactly 0. That gave std=0, i.e. torch.randn() silently
// returning all zeros (issue #66). Contrast aclnnInplaceUniform and
// aclnnInplaceRandom, which declare `double`/`int64_t`, so the variadic path is
// fine for those.
void inplace_normal_(const at::Tensor& self, double mean, double std,
                     int64_t seed) {
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  EXEC_ASCEND_CMD_SIG(aclnnInplaceNormal,
                      (const aclTensor*, float, float, int64_t, int64_t,
                       uint64_t*, aclOpExecutor**),
                      const_cast<aclTensor*>(acl_self.get()),
                      static_cast<float>(mean), static_cast<float>(std),
                      seed, static_cast<int64_t>(0));
}

void inplace_uniform_(const at::Tensor& self, double from, double to,
                      int64_t seed) {
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  EXEC_ASCEND_CMD(aclnnInplaceUniform, const_cast<aclTensor*>(acl_self.get()),
                  from, to, static_cast<uint64_t>(seed),
                  static_cast<uint64_t>(0));
}

void inplace_random_(const at::Tensor& self, int64_t from, int64_t to,
                     int64_t seed) {
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  EXEC_ASCEND_CMD(aclnnInplaceRandom, const_cast<aclTensor*>(acl_self.get()),
                  from, to, seed, static_cast<int64_t>(0));
}

// Exclusive upper bound for random_ when the caller omits `to`. Floating
// dtypes follow ATen's largest-exact-integer limits.
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
      // one value out of 2^63, matching the previous Ascend limitation.
      return std::numeric_limits<int64_t>::max();
    case at::kHalf: return int64_t{1} << 11;
    case at::kBFloat16: return int64_t{1} << 8;
    case at::kFloat: return int64_t{1} << 24;
    case at::kDouble: return int64_t{1} << 53;
    default:
      TORCH_CHECK(false, "unsupported random_ dtype: ", dtype);
  }
}

at::Tensor& random_with_range(at::Tensor& self, int64_t low, int64_t high,
                              const ::std::optional<at::Generator>& generator) {
  inplace_random_(self, low, high, next_seed(self, generator));
  return self;
}

// The default random_ overload is non-negative, matching ATen.
at::Tensor& random_default(at::Tensor& self,
                           const ::std::optional<at::Generator>& generator) {
  return random_with_range(
      self, 0, default_upper_bound(self.scalar_type()), generator);
}

at::Tensor& random_from(at::Tensor& self, int64_t from,
                        const ::std::optional<int64_t>& to,
                        const ::std::optional<at::Generator>& generator) {
  if (to.has_value()) return random_with_range(self, from, *to, generator);
  auto upper = default_upper_bound(self.scalar_type());
  TORCH_CHECK(from < upper, "random_ lower bound out of range");
  return random_with_range(self, from, upper, generator);
}


} // namespace

// randn(int[] size, *, ScalarType?, Layout?, Device?, bool? pin_memory)
//   -> Tensor of N(0, 1) samples. aclnnInplaceNormal(selfRef, mean, std, seed, offset).
at::Tensor RandnGeneratorKernelAscend(
    at::IntArrayRef size, ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(device));
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  inplace_normal_(out, 0.0, 1.0, next_seed(out, generator));
  return out;
}

at::Tensor RandnKernelAscend(at::IntArrayRef size,
                             ::std::optional<at::ScalarType> dtype,
                             ::std::optional<at::Layout> layout,
                             ::std::optional<at::Device> device,
                             ::std::optional<bool> pin_memory) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(device));
  return RandnGeneratorKernelAscend(
      size, std::nullopt, dtype, layout, device, pin_memory);
}

// rand(int[] size, ...) -> Tensor of U[0, 1) samples.
// aclnnInplaceUniform(selfRef, from, to, seed, offset).
at::Tensor RandGeneratorKernelAscend(
    at::IntArrayRef size, ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(device));
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  inplace_uniform_(out, 0.0, 1.0, next_seed(out, generator));
  return out;
}

at::Tensor RandKernelAscend(at::IntArrayRef size,
                            ::std::optional<at::ScalarType> dtype,
                            ::std::optional<at::Layout> layout,
                            ::std::optional<at::Device> device,
                            ::std::optional<bool> pin_memory) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(device));
  return RandGeneratorKernelAscend(
      size, std::nullopt, dtype, layout, device, pin_memory);
}

// randint.low(int low, int high, int[] size, ...) -> Tensor of ints in [low, high).
// aclnnInplaceRandom(selfRef, from, to, seed, offset).
at::Tensor RandintLowGeneratorKernelAscend(
    int64_t low, int64_t high, at::IntArrayRef size,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(device));
  auto out = make_empty(
      size, dtype.value_or(at::kLong), layout, device, pin_memory);
  return random_with_range(out, low, high, generator);
}

at::Tensor RandintLowKernelAscend(int64_t low, int64_t high, at::IntArrayRef size,
                                  ::std::optional<at::ScalarType> dtype,
                                  ::std::optional<at::Layout> layout,
                                  ::std::optional<at::Device> device,
                                  ::std::optional<bool> pin_memory) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(device));
  return RandintLowGeneratorKernelAscend(
      low, high, size, std::nullopt, dtype, layout, device, pin_memory);
}

at::Tensor RandintGeneratorKernelAscend(
    int64_t high, at::IntArrayRef size,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(device));
  return RandintLowGeneratorKernelAscend(
      0, high, size, generator, dtype, layout, device, pin_memory);
}

// randint(int high, int[] size, ...) -> ints in [0, high). Delegates to the
// low overload with low=0.
at::Tensor RandintKernelAscend(int64_t high, at::IntArrayRef size,
                               ::std::optional<at::ScalarType> dtype,
                               ::std::optional<at::Layout> layout,
                               ::std::optional<at::Device> device,
                               ::std::optional<bool> pin_memory) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(device));
  return RandintGeneratorKernelAscend(
      high, size, std::nullopt, dtype, layout, device, pin_memory);
}

REGISTER_IMPL_TO_DISPATCHER(RandnFn, randn_dispatcher, Backend::kAscend, RandnKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandnGeneratorFn, randn_generator_dispatcher, Backend::kAscend, RandnGeneratorKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandFn, rand_dispatcher, Backend::kAscend, RandKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandGeneratorFn, rand_generator_dispatcher, Backend::kAscend, RandGeneratorKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandintFn, randint_dispatcher, Backend::kAscend, RandintKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandintGeneratorFn, randint_generator_dispatcher, Backend::kAscend, RandintGeneratorKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandintLowFn, randint_low_dispatcher, Backend::kAscend, RandintLowKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandintLowGeneratorFn, randint_low_generator_dispatcher, Backend::kAscend, RandintLowGeneratorKernelAscend)

// =========================================================================
// Inplace RNG ops
// =========================================================================

// normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None)
at::Tensor& NormalInplaceKernelAscend(at::Tensor& self, double mean, double std,
                                      ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  inplace_normal_(self, mean, std, next_seed(self, generator));
  return self;
}

// uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None)
at::Tensor& UniformInplaceKernelAscend(at::Tensor& self, double from, double to,
                                       ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  inplace_uniform_(self, from, to, next_seed(self, generator));
  return self;
}

// random_(Tensor(a!) self, *, Generator? generator=None) -> [0, dtype.max]
at::Tensor& RandomInplaceKernelAscend(at::Tensor& self,
                                      ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  return random_default(self, generator);
}

// random_(Tensor(a!) self, int to, *, Generator? generator=None) -> [0, to)
at::Tensor& RandomInplaceToKernelAscend(at::Tensor& self, int64_t to,
                                        ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  return random_with_range(self, 0, to, generator);
}

// random_(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> [from, to)
at::Tensor& RandomInplaceFromKernelAscend(at::Tensor& self, int64_t from,
                                          ::std::optional<int64_t> to,
                                          ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  return random_from(self, from, to, generator);
}

REGISTER_IMPL_TO_DISPATCHER(NormalInplaceFn, normal_inplace_dispatcher, Backend::kAscend, NormalInplaceKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(UniformInplaceFn, uniform_inplace_dispatcher, Backend::kAscend, UniformInplaceKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandomInplaceFn, random_inplace_dispatcher, Backend::kAscend, RandomInplaceKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandomInplaceToFn, random_inplace_to_dispatcher, Backend::kAscend, RandomInplaceToKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandomInplaceFromFn, random_inplace_from_dispatcher, Backend::kAscend, RandomInplaceFromKernelAscend)

// =========================================================================
// normal.* factory overloads
// =========================================================================

// normal.float_float(float mean, float std, int[] size, *, Generator?, ...) -> Tensor
at::Tensor NormalFloatFloatKernelAscend(double mean, double std, at::IntArrayRef size,
                                        ::std::optional<at::Generator> generator,
                                        ::std::optional<at::ScalarType> dtype,
                                        ::std::optional<at::Layout> layout,
                                        ::std::optional<at::Device> device,
                                        ::std::optional<bool> pin_memory) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(device));
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_out(out);
  EXEC_ASCEND_CMD_SIG(aclnnNormalFloatFloat,
                      (float, float, int64_t, int64_t, aclTensor*,
                       uint64_t*, aclOpExecutor**),
                      static_cast<float>(mean), static_cast<float>(std),
                      next_seed(out, generator), static_cast<int64_t>(0),
                      const_cast<aclTensor*>(acl_out.get()));
  return out;
}

// normal.Tensor_float(Tensor mean, float std, *, Generator?) -> Tensor
at::Tensor NormalTensorFloatKernelAscend(const at::Tensor& mean, double std,
                                         ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(mean));
  auto out = at::empty_like(mean);
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_mean(mean), acl_out(out);
  EXEC_ASCEND_CMD_SIG(aclnnNormalTensorFloat,
                      (const aclTensor*, float, int64_t, int64_t, aclTensor*,
                       uint64_t*, aclOpExecutor**),
                      acl_mean.get(), static_cast<float>(std),
                      next_seed(mean, generator), static_cast<int64_t>(0),
                      const_cast<aclTensor*>(acl_out.get()));
  return out;
}

// normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator?) -> Tensor
at::Tensor NormalTensorTensorKernelAscend(const at::Tensor& mean, const at::Tensor& std,
                                          ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(mean));
  auto out = at::empty_like(mean);
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_mean(mean), acl_std(std), acl_out(out);
  EXEC_ASCEND_CMD(aclnnNormalTensorTensor,
                  acl_mean.get(), acl_std.get(),
                  next_seed(mean, generator), static_cast<int64_t>(0),
                  const_cast<aclTensor*>(acl_out.get()));
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(NormalFloatFloatFn, normal_float_float_dispatcher, Backend::kAscend, NormalFloatFloatKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(NormalTensorFloatFn, normal_tensor_float_dispatcher, Backend::kAscend, NormalTensorFloatKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(NormalTensorTensorFn, normal_tensor_tensor_dispatcher, Backend::kAscend, NormalTensorTensorKernelAscend)

// =========================================================================
// bernoulli
// =========================================================================

// bernoulli(Tensor self, *, Generator?) -> Tensor (self is probability)
at::Tensor BernoulliKernelAscend(const at::Tensor& self,
                                 ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  auto out = at::empty_like(self);
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self), acl_out(out);
  // aclnnBernoulliTensor: `selfAsProbabilityTensor` instead of scalar
  EXEC_ASCEND_CMD(aclnnBernoulliTensor,
                  acl_self.get(), acl_self.get(),
                  next_seed(self, generator), static_cast<int64_t>(0),
                  const_cast<aclTensor*>(acl_out.get()));
  return out;
}

// bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator?) -> Tensor
at::Tensor& BernoulliInplaceFloatKernelAscend(at::Tensor& self, double p,
                                              ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_p(at::Scalar(p), at::kFloat);
  EXEC_ASCEND_CMD_SIG(aclnnInplaceBernoulli,
                      (const aclTensor*, const aclScalar*, int64_t, int64_t,
                       uint64_t*, aclOpExecutor**),
                      const_cast<aclTensor*>(acl_self.get()), acl_p.get(),
                      next_seed(self, generator), static_cast<int64_t>(0));
  return self;
}

// bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator?) -> Tensor
at::Tensor& BernoulliInplaceTensorKernelAscend(at::Tensor& self, const at::Tensor& p,
                                               ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self), acl_p(p);
  EXEC_ASCEND_CMD(aclnnInplaceBernoulliTensor,
                  const_cast<aclTensor*>(acl_self.get()), acl_p.get(),
                  next_seed(self, generator), static_cast<int64_t>(0));
  return self;
}

REGISTER_IMPL_TO_DISPATCHER(BernoulliFn, bernoulli_dispatcher, Backend::kAscend, BernoulliKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(BernoulliInplaceFloatFn, bernoulli_inplace_float_dispatcher, Backend::kAscend, BernoulliInplaceFloatKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(BernoulliInplaceTensorFn, bernoulli_inplace_tensor_dispatcher, Backend::kAscend, BernoulliInplaceTensorKernelAscend)

// =========================================================================
// randperm
// =========================================================================

// randperm(int n, *, ScalarType?, Layout?, Device?, bool? pin_memory) -> Tensor
at::Tensor RandpermKernelAscend(int64_t n,
                                ::std::optional<at::ScalarType> dtype,
                                ::std::optional<at::Layout> layout,
                                ::std::optional<at::Device> device,
                                ::std::optional<bool> pin_memory) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(device));
  auto out = make_empty({n}, dtype.value_or(at::kLong), layout, device, pin_memory);
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_out(out);
  EXEC_ASCEND_CMD(aclnnRandperm, n, next_seed(out), static_cast<int64_t>(0),
                  const_cast<aclTensor*>(acl_out.get()));
  return out;
}

// randperm.out(int n, *, Tensor(a!) out) -> Tensor
at::Tensor& RandpermOutKernelAscend(int64_t n, at::Tensor& out) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(out));
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_out(out);
  EXEC_ASCEND_CMD(aclnnRandperm, n, next_seed(out), static_cast<int64_t>(0),
                  const_cast<aclTensor*>(acl_out.get()));
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(RandpermFn, randperm_dispatcher, Backend::kAscend, RandpermKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandpermOutFn, randperm_out_dispatcher, Backend::kAscend, RandpermOutKernelAscend)

// =========================================================================
// *_like factory ops (delegate to the base factory + empty_like)
// =========================================================================

at::Tensor RandLikeGeneratorKernelAscend(
    const at::Tensor& self, ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  auto out = at::empty_like(
      self, dtype, layout, device, pin_memory, memory_format);
  inplace_uniform_(out, 0.0, 1.0, next_seed(out, generator));
  return out;
}

at::Tensor RandLikeKernelAscend(const at::Tensor& self,
                                ::std::optional<at::ScalarType> dtype,
                                ::std::optional<at::Layout> layout,
                                ::std::optional<at::Device> device,
                                ::std::optional<bool> pin_memory,
                                ::std::optional<at::MemoryFormat> memory_format) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  return RandLikeGeneratorKernelAscend(
      self, std::nullopt, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor RandnLikeGeneratorKernelAscend(
    const at::Tensor& self, ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  auto out = at::empty_like(
      self, dtype, layout, device, pin_memory, memory_format);
  inplace_normal_(out, 0.0, 1.0, next_seed(out, generator));
  return out;
}

at::Tensor RandnLikeKernelAscend(const at::Tensor& self,
                                 ::std::optional<at::ScalarType> dtype,
                                 ::std::optional<at::Layout> layout,
                                 ::std::optional<at::Device> device,
                                 ::std::optional<bool> pin_memory,
                                 ::std::optional<at::MemoryFormat> memory_format) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  return RandnLikeGeneratorKernelAscend(
      self, std::nullopt, dtype, layout, device, pin_memory, memory_format);
}

REGISTER_IMPL_TO_DISPATCHER(RandLikeFn, rand_like_dispatcher, Backend::kAscend, RandLikeKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandLikeGeneratorFn, rand_like_generator_dispatcher, Backend::kAscend, RandLikeGeneratorKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeFn, randn_like_dispatcher, Backend::kAscend, RandnLikeKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeGeneratorFn, randn_like_generator_dispatcher, Backend::kAscend, RandnLikeGeneratorKernelAscend)

// randint_like(Tensor self, int high, *, dtype?, layout?, device?, pin_memory?, memory_format?)
at::Tensor RandintLikeKernelAscend(const at::Tensor& self, int64_t high,
                                   ::std::optional<at::ScalarType> dtype,
                                   ::std::optional<at::Layout> layout,
                                   ::std::optional<at::Device> device,
                                   ::std::optional<bool> pin_memory,
                                   ::std::optional<at::MemoryFormat> memory_format) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  auto out = at::empty_like(self, dtype.value_or(at::kLong), layout, device, pin_memory, memory_format);
  inplace_random_(out, 0, high, next_seed(out));
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(RandintLikeFn, randint_like_dispatcher, Backend::kAscend, RandintLikeKernelAscend)

// randint_like.low_dtype(Tensor self, int low, int high, *, ...) -> Tensor
at::Tensor RandintLikeLowDtypeKernelAscend(const at::Tensor& self, int64_t low, int64_t high,
                                           ::std::optional<at::ScalarType> dtype,
                                           ::std::optional<at::Layout> layout,
                                           ::std::optional<at::Device> device,
                                           ::std::optional<bool> pin_memory,
                                           ::std::optional<at::MemoryFormat> memory_format) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  auto out = at::empty_like(self, dtype.value_or(at::kLong), layout, device, pin_memory, memory_format);
  inplace_random_(out, low, high, next_seed(out));
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(RandintLikeLowDtypeFn, randint_like_low_dtype_dispatcher, Backend::kAscend, RandintLikeLowDtypeKernelAscend)

// =========================================================================
// out-variants
//
// Each fills the caller's tensor in place. `size`/`self` only determine the
// shape, which ATen has already resized `out` to, so they are unused here --
// deliberately: honouring them would mean resizing a tensor the caller owns.
// =========================================================================

at::Tensor& RandOutKernelAscend(at::IntArrayRef, at::Tensor& out) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(out));
  inplace_uniform_(out, 0.0, 1.0, next_seed(out));
  return out;
}

at::Tensor& RandNamesOutKernelAscend(at::IntArrayRef, ::std::optional<at::DimnameList>,
                                     at::Tensor& out) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(out));
  inplace_uniform_(out, 0.0, 1.0, next_seed(out));
  return out;
}

at::Tensor& RandnNamesOutKernelAscend(at::IntArrayRef, ::std::optional<at::DimnameList>,
                                      at::Tensor& out) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(out));
  inplace_normal_(out, 0.0, 1.0, next_seed(out));
  return out;
}

at::Tensor& RandLikeOutKernelAscend(const at::Tensor&, ::std::optional<at::MemoryFormat>,
                                    at::Tensor& out) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(out));
  inplace_uniform_(out, 0.0, 1.0, next_seed(out));
  return out;
}

at::Tensor& RandnLikeOutKernelAscend(const at::Tensor&, ::std::optional<at::MemoryFormat>,
                                     at::Tensor& out) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(out));
  inplace_normal_(out, 0.0, 1.0, next_seed(out));
  return out;
}

at::Tensor& RandintLowOutKernelAscend(int64_t low, int64_t high,
                                      at::IntArrayRef size, at::Tensor& out) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(out));
  out.resize_(size);
  return random_with_range(out, low, high, std::nullopt);
}

at::Tensor& RandintOutKernelAscend(int64_t high, at::IntArrayRef size,
                                   at::Tensor& out) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(out));
  return RandintLowOutKernelAscend(0, high, size, out);
}

at::Tensor& RandintLikeOutKernelAscend(const at::Tensor&, int64_t high,
                                       ::std::optional<at::MemoryFormat>, at::Tensor& out) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(out));
  inplace_random_(out, 0, high, next_seed(out));
  return out;
}

at::Tensor& RandintLikeLowDtypeOutKernelAscend(const at::Tensor&, int64_t low, int64_t high,
                                               ::std::optional<at::MemoryFormat>,
                                               at::Tensor& out) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(out));
  inplace_random_(out, low, high, next_seed(out));
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(RandOutFn, rand_out_dispatcher, Backend::kAscend, RandOutKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandNamesOutFn, rand_names_out_dispatcher, Backend::kAscend, RandNamesOutKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandnNamesOutFn, randn_names_out_dispatcher, Backend::kAscend, RandnNamesOutKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandLikeOutFn, rand_like_out_dispatcher, Backend::kAscend, RandLikeOutKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeOutFn, randn_like_out_dispatcher, Backend::kAscend, RandnLikeOutKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandintLowOutFn, randint_low_out_dispatcher, Backend::kAscend, RandintLowOutKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandintOutFn, randint_out_dispatcher, Backend::kAscend, RandintOutKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandintLikeOutFn, randint_like_out_dispatcher, Backend::kAscend, RandintLikeOutKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(RandintLikeLowDtypeOutFn, randint_like_low_dtype_out_dispatcher, Backend::kAscend, RandintLikeLowDtypeOutKernelAscend)

// =========================================================================
// Uniform-derived distributions (inverse-transform sampling)
//
// CANN has no aclnn kernel for any of these, but each is a closed-form transform
// of a U(0,1) draw, so they compose out of aclnnInplaceUniform plus arithmetic
// that already has kernels. That keeps them on-device -- the alternative, a CPU
// round-trip, costs more than the transform.
//
// Each draws on the half-open interval its transform requires:
//   exponential_  -X = -log(1-U)/lambda      needs U < 1  (log(0) at U=1)
//   log_normal_    X = exp(N(mean, std))     via normal_, no uniform needed
//   cauchy_        X = median + sigma*tan(pi*(U-0.5))   needs U in (0,1)
//   geometric_     X = ceil(log(U)/log(1-p)) needs U > 0  (log(0) at U=0)
// =========================================================================

// Smallest step that keeps a draw strictly inside (0, 1) for `dtype`. Using
// float32 eps for a float16 tensor would round back to the excluded endpoint.
double interval_eps(at::ScalarType dtype) {
  switch (dtype) {
    case at::kHalf: return 1e-3;
    case at::kBFloat16: return 1e-2;
    case at::kDouble: return 1e-15;
    default: return 1e-7;
  }
}

// exponential_(Tensor(a!) self, float lambd=1, *, Generator?) -> Tensor
at::Tensor& ExponentialInplaceKernelAscend(at::Tensor& self, double lambd,
                                           ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  TORCH_CHECK(lambd > 0, "exponential_ expects lambd > 0, got ", lambd);
  auto eps = interval_eps(self.scalar_type());
  // U in [0, 1-eps] so that log1p(-U) stays finite.
  inplace_uniform_(self, 0.0, 1.0 - eps, next_seed(self, generator));
  // -log(1-U)/lambda, via log1p for accuracy near U=0. Built from out-of-place
  // ops (the in-place spellings have no aclnn route) and written back with
  // set_data-free copy semantics via the dispatcher's own `add_`.
  auto r = at::div(at::log1p(at::neg(self)), -lambd);
  self.zero_().add_(r);
  return self;
}

// log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator?) -> Tensor
at::Tensor& LogNormalInplaceKernelAscend(at::Tensor& self, double mean, double std,
                                         ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  TORCH_CHECK(std > 0, "log_normal_ expects std > 0, got ", std);
  inplace_normal_(self, mean, std, next_seed(self, generator));
  auto r = at::exp(self);
  self.zero_().add_(r);
  return self;
}

// cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator?) -> Tensor
at::Tensor& CauchyInplaceKernelAscend(at::Tensor& self, double median, double sigma,
                                      ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  auto eps = interval_eps(self.scalar_type());
  // U in (0, 1): tan(pi*(U-0.5)) diverges at both endpoints.
  inplace_uniform_(self, eps, 1.0 - eps, next_seed(self, generator));
  constexpr double kPi = 3.14159265358979323846;
  auto r = at::add(at::mul(at::tan(at::mul(at::sub(self, 0.5), kPi)), sigma), median);
  self.zero_().add_(r);
  return self;
}

// geometric_(Tensor(a!) self, float p, *, Generator?) -> Tensor
// Number of Bernoulli(p) trials up to and including the first success, so the
// support is {1, 2, ...} -- matching ATen, not the {0, 1, ...} convention.
at::Tensor& GeometricInplaceKernelAscend(at::Tensor& self, double p,
                                         ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  TORCH_CHECK(p > 0 && p < 1, "geometric_ expects 0 < p < 1, got ", p);
  auto eps = interval_eps(self.scalar_type());
  // U in (0, 1]: log(U) is -inf at U=0.
  inplace_uniform_(self, eps, 1.0, next_seed(self, generator));
  auto r = at::clamp_min(at::ceil(at::div(at::log(self), std::log1p(-p))), 1);
  self.zero_().add_(r);
  return self;
}

REGISTER_IMPL_TO_DISPATCHER(ExponentialInplaceFn, exponential_inplace_dispatcher, Backend::kAscend, ExponentialInplaceKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(LogNormalInplaceFn, log_normal_inplace_dispatcher, Backend::kAscend, LogNormalInplaceKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(CauchyInplaceFn, cauchy_inplace_dispatcher, Backend::kAscend, CauchyInplaceKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(GeometricInplaceFn, geometric_inplace_dispatcher, Backend::kAscend, GeometricInplaceKernelAscend)

// =========================================================================
// native_dropout
// =========================================================================

// native_dropout(Tensor input, float p, bool? train) -> (Tensor output, Tensor mask)
//
// aclnn's maskOut is a bit-packed uint8 buffer, not the bool tensor ATen's schema
// promises, so it cannot be returned directly. We derive the bool mask from the
// output instead (a dropped element is exactly 0), which is what
// native_dropout_backward consumes anyway.
::std::tuple<at::Tensor, at::Tensor> NativeDropoutKernelAscend(
    const at::Tensor& input, double p, ::std::optional<bool> train) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(input));
  // Inference, or p==0: identity with an all-true mask.
  if (!train.value_or(true) || p == 0.0) {
    return {input.clone(), at::ones_like(input, input.options().dtype(at::kBool))};
  }
  // p==1 drops everything; aclnn's scale factor 1/(1-p) would divide by zero.
  if (p == 1.0) {
    return {at::zeros_like(input), at::zeros_like(input, input.options().dtype(at::kBool))};
  }
  // CANN's native dropout returns a bit-packed uint8 mask, while ATen requires
  // a bool tensor matching the input shape. Build that public contract directly
  // from one on-device uniform draw so zero-valued input elements do not get
  // mistaken for dropped elements.
  auto out = at::empty_like(input);
  inplace_uniform_(out, 0.0, 1.0, next_seed(input));
  auto mask = out.ge(p);
  out.copy_(input);
  out.mul_(mask);
  out.mul_(1.0 / (1.0 - p));
  return {out, mask};
}

REGISTER_IMPL_TO_DISPATCHER(NativeDropoutFn, native_dropout_dispatcher, Backend::kAscend, NativeDropoutKernelAscend)

// =========================================================================
// Rejection-sampled distributions: CPU round-trip
//
// CANN 9.0.0 has no aclnn kernel for any of these, and unlike the four above
// they are not closed-form transforms of a uniform draw -- each needs rejection
// sampling (Marsaglia-Tsang for gamma, PTRS/inversion for poisson and binomial),
// which means a data-dependent loop per element. There is no way to express that
// from the composed aclnn ops available here.
//
// So they run on CPU and copy back. That is a real cost -- two transfers plus
// host compute, and it serializes against the device stream -- but it is bounded
// and correct, where the alternative is "backend not registered". These ops do
// not appear in transformer training or inference paths; they show up in
// probabilistic models, where the sample is usually small relative to the model.
//
// Reproducibility is preserved: each fallback reserves a seed from the same
// per-device flagos generator as the aclnn paths, then initializes a temporary
// CPU generator for the host implementation.
//
// Replace with an on-device kernel if any of these lands in a hot path, or if
// CANN gains the corresponding aclnn API.
// =========================================================================

at::Tensor PoissonKernelAscend(const at::Tensor& self,
                               ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  auto cpu_generator = cpu_generator_for(self, generator);
  return at::poisson(self.cpu(), cpu_generator).to(self.device());
}

at::Tensor StandardGammaKernelAscend(const at::Tensor& self,
                                     ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  auto cpu_generator = cpu_generator_for(self, generator);
  return at::_standard_gamma(self.cpu(), cpu_generator).to(self.device());
}

at::Tensor SampleDirichletKernelAscend(const at::Tensor& self,
                                       ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  auto cpu_generator = cpu_generator_for(self, generator);
  return at::_sample_dirichlet(self.cpu(), cpu_generator).to(self.device());
}

at::Tensor BinomialKernelAscend(const at::Tensor& count, const at::Tensor& prob,
                                ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(count));
  auto cpu_generator = cpu_generator_for(count, generator);
  return at::binomial(count.cpu(), prob.cpu(), cpu_generator).to(count.device());
}

REGISTER_IMPL_TO_DISPATCHER(PoissonFn, poisson_dispatcher, Backend::kAscend, PoissonKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(PrivStandardGammaFn, priv_standard_gamma_dispatcher, Backend::kAscend, StandardGammaKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(PrivSampleDirichletFn, priv_sample_dirichlet_dispatcher, Backend::kAscend, SampleDirichletKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(BinomialFn, binomial_dispatcher, Backend::kAscend, BinomialKernelAscend)

} // namespace at::native::flagos
