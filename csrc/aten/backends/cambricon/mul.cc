// Copyright (c) 2026, BAAI. All rights reserved.
//
// Device-only mul.Tensor through CNNL. FlagGems' MLU device guard redispatches
// flagos operands recursively (flagos-ai/FlagGems#8448). CNNL half computation
// was inaccurate in the SDK probe, so half/bfloat16 accumulate in float32.
#include "aten/backends/cambricon/cambricon_common.h"
#include "aten/backends/cambricon/attention.h"

#ifdef USE_CAMBRICON
#include "aten/generated/ops.h"
#include <ATen/ops/result_type.h>

namespace at::native::flagos::cambricon {
namespace {
cnnlCastDataType_t ToFloatCast(at::ScalarType type) {
  switch (type) {
    case at::kHalf: return CNNL_CAST_HALF_TO_FLOAT;
    case at::kBFloat16: return CNNL_CAST_BFLOAT16_TO_FLOAT;
    default: TORCH_CHECK(false, "Unsupported mul compute cast: ", type);
  }
}
cnnlCastDataType_t FromFloatCast(at::ScalarType type) {
  switch (type) {
    case at::kHalf: return CNNL_CAST_FLOAT_TO_HALF;
    case at::kBFloat16: return CNNL_CAST_FLOAT_TO_BFLOAT16;
    default: TORCH_CHECK(false, "Unsupported mul output cast: ", type);
  }
}
at::Tensor Widen(const at::Tensor& input, cnnlHandle_t handle) {
  if (input.scalar_type() == at::kFloat) return input;
  auto output = at::empty(input.sizes(), input.options().dtype(at::kFloat));
  TensorDesc from, to;
  from.Set(input);
  to.Set(output);
  CheckCnnl(cnnlCastDataType(handle, from.get(), input.const_data_ptr(),
                           ToFloatCast(input.scalar_type()), to.get(),
                           output.mutable_data_ptr()), "cnnlCastDataType (mul input)");
  return output;
}
} // namespace

at::Tensor MulTensor(const at::Tensor& self, const at::Tensor& other) {
  // Determine promotion BEFORE broadcasting: aten gives zero-dimensional
  // scalars lower precedence than dimensioned tensors, even for CPU doubles.
  const auto dtype = at::result_type(self, other);
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16 ||
                  dtype == at::kInt,
              "flagos cambricon: mul.Tensor unsupported result dtype ", dtype);
  const auto device = self.is_cpu() ? other.device() : self.device();
  TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1,
              "flagos cambricon: mul requires a device operand");
  for (const auto* tensor : {&self, &other}) {
    TORCH_CHECK(tensor->device() == device ||
                    (tensor->is_cpu() && tensor->dim() == 0),
                "flagos cambricon: mul only accepts CPU tensors as scalars");
  }
  CambriconDeviceGuard guard(self.is_cpu() ? other : self);
  // Upload an existing CPU scalar, never download device data. Passing its
  // host pointer to CNNL causes an asynchronous device memory fault.
  auto a = self.to(device, dtype);
  auto b = other.to(device, dtype);
  const auto shape = at::infer_size(self.sizes(), other.sizes());
  auto result = at::empty(shape, a.options());
  if (result.numel() == 0) return result;
  // CNNL ARRAY descriptors cannot express rank zero. Treat scalar*scalar as
  // one element internally and restore the aten scalar shape on return.
  std::vector<int64_t> dense_shape(shape.begin(), shape.end());
  if (dense_shape.empty()) dense_shape.push_back(1);
  a = a.expand(dense_shape).contiguous();
  b = b.expand(dense_shape).contiguous();
  const auto handle = HandleForDevice(static_cast<int>(device.index()));
  const bool integral = dtype == at::kInt;
  auto af = integral ? a : Widen(a, handle);
  auto bf = integral ? b : Widen(b, handle);
  const auto compute = integral ? dtype : at::kFloat;
  auto wide = at::empty(dense_shape, a.options().dtype(compute));
  TensorDesc ad, bd, cd;
  ad.Set(af); bd.Set(bf); cd.Set(wide);
  OpTensorDesc op(CNNL_OP_TENSOR_MUL, compute);
  const float one = 1.0f, zero = 0.0f;
  const int32_t int_one = 1, int_zero = 0;
  const void* alpha = integral ? static_cast<const void*>(&int_one) : &one;
  const void* beta = integral ? static_cast<const void*>(&int_zero) : &zero;
  CheckCnnl(cnnlOpTensor(handle, op.get(), alpha, ad.get(), af.const_data_ptr(),
                       alpha, bd.get(), bf.const_data_ptr(), nullptr, 0,
                       beta, cd.get(), wide.mutable_data_ptr()), "cnnlOpTensor");
  if (dtype == compute) return wide.reshape(shape);
  auto dense_result = result.reshape(dense_shape);
  TensorDesc out;
  out.Set(dense_result);
  CheckCnnl(cnnlCastDataType(handle, cd.get(), wide.const_data_ptr(),
                           FromFloatCast(dtype), out.get(),
                           dense_result.mutable_data_ptr()), "cnnlCastDataType (mul output)");
  return result;
}
// The out= spelling. It is the same arithmetic into a caller-provided buffer, so
// it forwards to the functional kernel and copies the result in rather than
// repeating the descriptor work; `copy_` is a runtime copy on this backend, not
// a host round trip. Served here because a Qwen3 training step reaches
// `mul.out` directly and it was the largest remaining cpu_fallback there.
//
// An out= buffer that does not yet carry the result shape is resized here. aten
// resizes a wrong-sized buffer before dispatching, so the shapes that reach this
// kernel are the right one and the empty one a caller passes to be filled;
// copying into a non-empty mismatch instead reaches the device with a
// destination the copy cannot map, and that faults rather than raising.
at::Tensor& MulOut(const at::Tensor& self, const at::Tensor& other,
                   at::Tensor& out) {
  auto result = MulTensor(self, other);
  if (out.sizes() != result.sizes()) {
    // The empty buffer aten's out= contract allows, and a wrong-sized one that
    // reached the kernel before the dispatcher resized it.
    out.resize_(result.sizes());
  }
  out.copy_(result);
  return out;
}
} // namespace at::native::flagos::cambricon

namespace at::native::flagos {
REGISTER_IMPL_TO_DISPATCHER(
    MulTensorFn, mul_tensor_dispatcher, Backend::kCambricon, cambricon::MulTensor);
REGISTER_IMPL_TO_DISPATCHER(
    MulOutFn, mul_out_dispatcher, Backend::kCambricon, cambricon::MulOut);
} // namespace at::native::flagos
#endif // USE_CAMBRICON
