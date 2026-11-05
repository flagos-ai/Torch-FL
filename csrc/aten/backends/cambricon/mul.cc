// Copyright (c) 2026, BAAI. All rights reserved.
//
// `mul.Tensor` on Cambricon, through CNNL.
//
// This is the first vendor-native kernel on this platform, and it exists for a
// measured reason: FlagGems' cambricon `mul` cannot serve it. Its device guard
// compares `tensor.device.type` against the vendor's expected device name
// ("mlu"), and torch_fl registers PrivateUse1 as "flagos", so the comparison is
// always true and the op redispatches into itself until the stack is exhausted
// (flagos-ai/FlagGems#8448). With no FlagGems route and no vendor kernel, every
// `mul` took `cpu_fallback` -- 330 of 333 fallbacks on a single Qwen3 training
// step, the largest source of host round-trips in the model.
//
// cnnlOpTensor is the elementwise binary entry point. It takes five arguments
// worth knowing about: a descriptor per operand, an alpha per operand and a
// beta for the output, computing `c = beta*c + alpha1*a + alpha2*b` semantics
// across the op named in the operation descriptor. For a plain multiply the
// shape used here is alpha1=1, alpha2=1, beta=0.

#include "aten/backends/cambricon/cambricon_common.h"
#include "aten/backends/cambricon/attention.h"

#ifdef USE_CAMBRICON

#include "aten/generated/ops.h"

#include <ATen/ops/empty.h>

namespace at::native::flagos::cambricon {

namespace {

// dtypes this kernel serves. fp32 only, because that is what was measured:
// MLU590, 1-D through 4-D plus a 2M-element 2-D pair, every element exact.
//
// Half and bfloat16 go through the same call without erroring but return wrong
// values (fp16 off by up to 5.4, bfloat16 by 2.3 on the same inputs), and the
// cause is not yet identified -- it is not the alpha/beta representation, since
// giving them their own dtype-sized buffers did not change the result. They stay
// on the CPU round trip below until there is a measurement that says otherwise;
// listing a dtype here is a claim, and this comment has to be able to back it.
bool SupportedDtype(at::ScalarType dtype) {
  return dtype == at::kFloat;
}

// The slow path, written out rather than re-dispatched: this function is reached
// through m.impl, so asking the dispatcher for `mul.Tensor` again from here would
// land on itself.
at::Tensor MultiplyOnHost(const at::Tensor& self, const at::Tensor& other) {
  return (self.to(at::kCPU) * other.to(at::kCPU)).to(self.device());
}

} // namespace

at::Tensor MulTensor(const at::Tensor& self, const at::Tensor& other) {
  // The kernel serves exactly the shapes cnnlOpTensor was measured on: two
  // operands with identical dims. Broadcasting and the scalar overloads are
  // aten's TensorIterator expanding one side first, and this kernel does not
  // reproduce that -- so rather than raise, which would turn a legal `a * 2.5`
  // into a failure, those calls take a CPU round trip instead.
  //
  // The round trip is written out rather than re-dispatched: this function is
  // reached through m.impl, and asking for `mul.Tensor` again from here would
  // land on itself. `numel() == other.numel()` would be the wrong test --
  // (4,1) and (4,) have equal numel and different shapes.
  //
  // An unmeasured dtype takes the same route. Not a `TORCH_CHECK`: a precision
  // this kernel cannot vouch for is a reason to go slow, not to fail an op that
  // worked before the kernel existed.
  const bool served = SupportedDtype(self.scalar_type());
  if (!served || self.sizes() != other.sizes()) {
    return MultiplyOnHost(self, other);
  }
  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "flagos cambricon: mul.Tensor requires matching dtypes, got ",
      self.scalar_type(),
      " and ",
      other.scalar_type());
  // `cnnlSetTensorDescriptor` describes a dense ARRAY layout, so a strided
  // operand would be read as if it were contiguous and produce wrong numbers
  // rather than an error. Strided inputs take the CPU round trip until a
  // measured strided path exists (cnnlSetTensorDescriptorEx takes explicit
  // strides, but nothing has compared it against materialising a copy).
  //
  // This has to be a fallback and not a check: autograd hands `mul` grads that
  // are expanded views (a broadcast operand's gradient is a `.expand()` of the
  // reduced one), so raising here would fail every backward through a multiply.
  if (!self.is_contiguous() || !other.is_contiguous()) {
    return MultiplyOnHost(self, other);
  }

  CambriconDeviceGuard guard(self);

  at::Tensor result = at::empty(self.sizes(), self.options());
  if (self.numel() == 0) {
    return result;
  }

  TensorDesc a_desc;
  TensorDesc b_desc;
  TensorDesc c_desc;
  a_desc.Set(self);
  b_desc.Set(other);
  c_desc.Set(result);

  OpTensorDesc op_desc(CNNL_OP_TENSOR_MUL, self.scalar_type());

  // The scalars are typed by the *compute* dtype, which cnnl takes from the
  // operation descriptor -- and this kernel names the same dtype there, so for
  // half and bfloat16 the values must be in that dtype's own representation.
  // Passing floats and reading the low two bytes (which works by accident for
  // fp16 when both are 1.0/0.0) produced NaN for fp16 and a 2.3 error for
  // bfloat16, so each precision gets its own buffers.
  float alpha_f32 = 1.0f;
  float beta_f32 = 0.0f;
  at::Half alpha_f16 = static_cast<at::Half>(1.0f);
  at::Half beta_f16 = static_cast<at::Half>(0.0f);
  at::BFloat16 alpha_bf16 = static_cast<at::BFloat16>(1.0f);
  at::BFloat16 beta_bf16 = static_cast<at::BFloat16>(0.0f);

  const void* alpha = &alpha_f32;
  const void* beta = &beta_f32;
  if (self.scalar_type() == at::kHalf) {
    alpha = &alpha_f16;
    beta = &beta_f16;
  } else if (self.scalar_type() == at::kBFloat16) {
    alpha = &alpha_bf16;
    beta = &beta_bf16;
  }

  CheckCnnl(
      cnnlOpTensor(
          HandleForDevice(static_cast<int>(self.device().index())),
          op_desc.get(),
          alpha,
          a_desc.get(),
          self.const_data_ptr(),
          alpha,
          b_desc.get(),
          other.const_data_ptr(),
          /*workspace=*/nullptr,
          /*workspace_size=*/0,
          beta,
          c_desc.get(),
          result.mutable_data_ptr()),
      "cnnlOpTensor");

  return result;
}

} // namespace at::native::flagos::cambricon

// The dispatcher binding. `route()` in gen_vendor_confs.py picks `cambricon`
// for an op by name, and Dispatcher::GetFn answers that name only for backends
// registered here -- without this line the conf would say `cambricon` and the
// call would still land on whatever fallback the registry resolves to.
namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    MulTensorFn, mul_tensor_dispatcher, Backend::kCambricon, cambricon::MulTensor);

} // namespace at::native::flagos

#endif // USE_CAMBRICON
