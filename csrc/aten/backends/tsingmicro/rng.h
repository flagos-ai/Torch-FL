// Copyright (c) 2026, BAAI. All rights reserved.
//
// Declarations for the TsingMicro RNG kernels, so csrc/aten/register.cc can
// claim them on PrivateUse1. The definitions live in rng.cc; unlike the other
// vendors, tsingmicro's registration is a direct m.impl() in register.cc (its
// conf names no RNG op and the FlagGems codegen would not emit one for a
// `tsingmicro` route), so the symbols have to cross a translation-unit boundary.
//
// The signatures mirror the dispatcher types in generated/ops.h.

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Dimname.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Layout.h>
#include <c10/core/Device.h>
#include <c10/core/MemoryFormat.h>

#include <array>
#include <optional>
#include <tuple>

namespace at::native::flagos {

at::Tensor& BernoulliInplaceFloatKernelTsingMicro(
    at::Tensor& self, double p, ::std::optional<at::Generator> generator);

at::Tensor& BernoulliInplaceTensorKernelTsingMicro(
    at::Tensor& self,
    const at::Tensor& p,
    ::std::optional<at::Generator> generator);

at::Tensor BernoulliKernelTsingMicro(
    const at::Tensor& self, ::std::optional<at::Generator> generator);

at::Tensor BinomialKernelTsingMicro(
    const at::Tensor& count,
    const at::Tensor& prob,
    ::std::optional<at::Generator> generator);

at::Tensor& ExponentialInplaceKernelTsingMicro(
    at::Tensor& self, double lambda, ::std::optional<at::Generator> generator);

at::Tensor ExponentialKernelTsingMicro(
    const at::Tensor& self, double lambda, ::std::optional<at::Generator> generator);

at::Tensor MultinomialKernelTsingMicro(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    ::std::optional<at::Generator> generator);

at::Tensor NativeDropoutBackwardKernelTsingMicro(
    const at::Tensor& grad_output, const at::Tensor& mask, double scale);

::std::tuple<at::Tensor, at::Tensor> NativeDropoutKernelTsingMicro(
    const at::Tensor& input, double p, ::std::optional<bool> train);

at::Tensor NormalFloatFloatKernelTsingMicro(
    double mean,
    double std,
    at::IntArrayRef size,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory);

at::Tensor& NormalInplaceKernelTsingMicro(
    at::Tensor& self,
    double mean,
    double std,
    ::std::optional<at::Generator> generator);

at::Tensor NormalTensorFloatKernelTsingMicro(
    const at::Tensor& mean, double std, ::std::optional<at::Generator> generator);

at::Tensor NormalTensorTensorKernelTsingMicro(
    const at::Tensor& mean,
    const at::Tensor& std,
    ::std::optional<at::Generator> generator);

at::Tensor PoissonKernelTsingMicro(
    const at::Tensor& self, ::std::optional<at::Generator> generator);

at::Tensor PrivSampleDirichletKernelTsingMicro(
    const at::Tensor& self, ::std::optional<at::Generator> generator);

at::Tensor PrivStandardGammaKernelTsingMicro(
    const at::Tensor& self, ::std::optional<at::Generator> generator);

at::Tensor RandGeneratorKernelTsingMicro(
    at::IntArrayRef size,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory);

at::Tensor RandKernelTsingMicro(
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory);

at::Tensor RandLikeGeneratorKernelTsingMicro(
    const at::Tensor& self,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format);

at::Tensor RandLikeKernelTsingMicro(
    const at::Tensor& self,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format);

at::Tensor& RandLikeOutKernelTsingMicro(
    const at::Tensor& self,
    ::std::optional<at::MemoryFormat> memory_format,
    at::Tensor& out);

at::Tensor& RandNamesOutKernelTsingMicro(
    at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor& out);

at::Tensor& RandOutKernelTsingMicro(at::IntArrayRef size, at::Tensor& out);

at::Tensor RandintGeneratorKernelTsingMicro(
    int64_t high,
    at::IntArrayRef size,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory);

at::Tensor RandintKernelTsingMicro(
    int64_t high,
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory);

at::Tensor RandintLikeKernelTsingMicro(
    const at::Tensor& self,
    int64_t high,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format);

at::Tensor RandintLikeLowDtypeKernelTsingMicro(
    const at::Tensor& self,
    int64_t low,
    int64_t high,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format);

at::Tensor& RandintLikeLowDtypeOutKernelTsingMicro(
    const at::Tensor& self,
    int64_t low,
    int64_t high,
    ::std::optional<at::MemoryFormat> memory_format,
    at::Tensor& out);

at::Tensor& RandintLikeOutKernelTsingMicro(
    const at::Tensor& self,
    int64_t high,
    ::std::optional<at::MemoryFormat> memory_format,
    at::Tensor& out);

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
    ::std::optional<bool> pin_memory);

at::Tensor RandintLowKernelTsingMicro(
    int64_t low,
    int64_t high,
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory);

at::Tensor& RandintLowOutKernelTsingMicro(
    int64_t low, int64_t high, at::IntArrayRef size, at::Tensor& out);

at::Tensor& RandintOutKernelTsingMicro(
    int64_t high, at::IntArrayRef size, at::Tensor& out);

at::Tensor RandnGeneratorKernelTsingMicro(
    at::IntArrayRef size,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory);

at::Tensor RandnKernelTsingMicro(
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory);

at::Tensor RandnLikeGeneratorKernelTsingMicro(
    const at::Tensor& self,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format);

at::Tensor& RandnLikeGeneratorOutKernelTsingMicro(
    const at::Tensor& self,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::MemoryFormat> memory_format,
    at::Tensor& out);

at::Tensor RandnLikeKernelTsingMicro(
    const at::Tensor& self,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format);

at::Tensor& RandnLikeOutKernelTsingMicro(
    const at::Tensor& self,
    ::std::optional<at::MemoryFormat> memory_format,
    at::Tensor& out);

at::Tensor& RandnNamesOutKernelTsingMicro(
    at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor& out);

at::Tensor& RandomInplaceFromKernelTsingMicro(
    at::Tensor& self,
    int64_t from,
    ::std::optional<int64_t> to,
    ::std::optional<at::Generator> generator);

at::Tensor& RandomInplaceKernelTsingMicro(
    at::Tensor& self, ::std::optional<at::Generator> generator);

at::Tensor& RandomInplaceToKernelTsingMicro(
    at::Tensor& self, int64_t to, ::std::optional<at::Generator> generator);

at::Tensor RandpermGeneratorKernelTsingMicro(
    int64_t n,
    ::std::optional<at::Generator> generator,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory);

at::Tensor RandpermKernelTsingMicro(
    int64_t n,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory);

at::Tensor& RandpermOutKernelTsingMicro(int64_t n, at::Tensor& out);

at::Tensor& UniformInplaceKernelTsingMicro(
    at::Tensor& self,
    double from,
    double to,
    ::std::optional<at::Generator> generator);

} // namespace at::native::flagos
