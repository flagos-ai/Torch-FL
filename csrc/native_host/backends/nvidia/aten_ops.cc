// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "precision_runtime.h"
#include "backends/nvidia/aten_ops.h"
#include "backends/nvidia/cuda_inputs.h"
#include "backends/nvidia/tensor_ops.h"

#include <ATen/MemoryOverlap.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/Pow.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/_assert_async_native.h>
#include <ATen/ops/_index_put_impl.h>
#include <ATen/ops/_linalg_check_errors_native.h>
#include <ATen/ops/_linalg_solve_ex.h>
#include <ATen/ops/_upsample_nearest_exact2d_native.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/cudnn_convolution_native.h>
#include <ATen/ops/div_native.h>
#include <ATen/ops/index_copy.h>
#include <ATen/ops/index_native.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/linalg_vector_norm_native.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/native_layer_norm_native.h>
#include <ATen/ops/prod_native.h>
#include <ATen/ops/scatter_add.h>
#include <c10/util/irange.h>
#include <torch/library.h>

namespace at::native {
namespace {

void FlagosIndex(at::TensorIteratorBase& iterator, at::IntArrayRef sizes,
                 at::IntArrayRef strides) {
  flagos::vendor::VendorScope scope(iterator.tensor(0));
  flagos::vendor::DescribeCudaMemory(iterator);
  index_stub(at::kCUDA, iterator, sizes, strides);
}

void FlagosNorm(at::TensorIterator& iterator, const at::Scalar& order) {
  flagos::vendor::DescribeCudaMemory(iterator);
  norm_stub(at::kCUDA, iterator, order);
}

}  // namespace

REGISTER_PRIVATEUSE1_DISPATCH(index_stub, &FlagosIndex);
REGISTER_PRIVATEUSE1_DISPATCH(norm_stub, &FlagosNorm);
}  // namespace at::native

namespace at::native::flagos::vendor {

at::Tensor NvidiaConvolution(const at::Tensor& input, const at::Tensor& weight,
                             const std::optional<at::Tensor>& bias,
                             at::IntArrayRef stride, at::IntArrayRef padding,
                             at::IntArrayRef dilation, int64_t groups) {
  VendorScope scope(input);
  auto& context = at::globalContext();
  auto memory_format = at::native::cudnn_conv_suggest_memory_format(input, weight);
  auto result = at::empty(
      at::native::conv_output_size(input.sizes(), weight.sizes(), padding, stride, dilation),
      input.options(), memory_format);
  at::native::cudnn_convolution_out(
      input, weight, padding, stride, dilation, groups, context.benchmarkCuDNN(),
      context.deterministicCuDNN() || context.deterministicAlgorithms(),
      context.allowTF32CuDNN(), result);
  if (bias && bias->defined()) {
    auto broadcast = std::vector<int64_t>(result.dim(), 1);
    broadcast[1] = bias->numel();
    result.add_(bias->reshape(broadcast));
  }
  return result;
}

at::Tensor NvidiaMul(const at::Tensor& self, const at::Tensor& other) {
  return Mul(self, other);
}

at::Tensor NvidiaDiv(const at::Tensor& self, const at::Tensor& divisor) {
  VendorScope scope(self.is_privateuseone() ? self : divisor);
  StructuredOutput<at::native::structured_div_out> kernel;
  kernel.meta(self, divisor);
  DescribeCudaMemory(kernel);
  at::native::div_true_stub(at::kCUDA, kernel);
  return kernel.output;
}

at::Tensor NvidiaDiv(const at::Tensor& self, const at::Scalar& divisor) {
  return NvidiaDiv(self, at::native::wrapped_scalar_tensor(divisor));
}

at::Tensor& NvidiaDivInplace(at::Tensor& self, const at::Tensor& divisor) {
  VendorScope scope(self);
  auto iterator = at::TensorIterator::binary_float_op(self, self, divisor);
  DescribeCudaMemory(iterator);
  at::native::div_true_stub(at::kCUDA, iterator);
  return self;
}

at::Tensor& NvidiaDivInplace(at::Tensor& self, const at::Scalar& divisor) {
  return NvidiaDivInplace(self, at::native::wrapped_scalar_tensor(divisor));
}

at::Tensor& NvidiaMmOut(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  TORCH_CHECK(out.device() == self.device() && out.scalar_type() == self.scalar_type(),
              "mm: output must have the same dtype and device as the input");
  at::assert_no_internal_overlap(out);
  return WithCudaInputs({self, mat2}, [&](at::TensorList inputs) -> at::Tensor& {
    if (self.dim() == 2 && mat2.dim() == 2 && out.dim() == 2 &&
        out.size(0) == self.size(0) && out.size(1) == mat2.size(1)) {
      auto output = FlagosToCudaView(out);
      at::mm_out(output, inputs[0], inputs[1]);
      return out;
    }
    const auto result = at::mm(inputs[0], inputs[1]);
    at::native::resize_output(out, result.sizes());
    return CopyCudaResult(out, result);
  });
}

at::Tensor NvidiaVectorNorm(const at::Tensor& input, const at::Scalar& order,
                            at::OptionalIntArrayRef dim, bool keepdim,
                            std::optional<at::ScalarType> dtype) {
  VendorScope scope(input);
  return StructuredCall<at::native::structured_linalg_vector_norm_out>(
      input, order, dim, keepdim, dtype);
}

at::Tensor NvidiaScatterAdd(const at::Tensor& self, int64_t dim,
                            const at::Tensor& index, const at::Tensor& source) {
  return WithCudaInputs({self, index, source}, [&](at::TensorList inputs) {
    const auto cuda = at::scatter_add(inputs[0], dim, inputs[1], inputs[2]);
    auto result = at::empty_strided(cuda.sizes(), cuda.strides(), self.options());
    CopyCudaResult(result, cuda);
    return result;
  });
}

at::Tensor& NvidiaScatterAddInplace(at::Tensor& self, int64_t dim,
                                   const at::Tensor& index, const at::Tensor& source) {
  return WithCudaInputs({self, index, source}, [&](at::TensorList inputs) -> at::Tensor& {
    const auto result = at::scatter_add(inputs[0], dim, inputs[1], inputs[2]);
    return CopyCudaResult(self, result);
  });
}

at::Tensor& NvidiaIndexPut(at::Tensor& self,
                          const c10::List<std::optional<at::Tensor>>& indices,
                          const at::Tensor& values, bool accumulate, bool unsafe) {
  at::assert_no_overlap(self, values);
  c10::SmallVector<at::Tensor, 6> operands{self, values};
  for (const auto i : c10::irange(indices.size())) {
    const auto index = indices.get(i);
    if (index && index->defined()) {
      at::assert_no_overlap(self, *index);
      operands.push_back(*index);
    } else {
      operands.emplace_back();
    }
  }
  return WithCudaInputs(operands, [&](at::TensorList inputs) -> at::Tensor& {
    auto output = inputs[0];
    c10::List<std::optional<at::Tensor>> cuda_indices;
    for (const auto& index : inputs.slice(2)) {
      cuda_indices.push_back(index.defined() ? std::optional(index) : std::nullopt);
    }
    at::_index_put_impl_(output, cuda_indices, inputs[1], accumulate, unsafe);
    return self;
  });
}

namespace {

at::Tensor& MulOut(const at::Tensor& self, const at::Tensor& other,
                   at::Tensor& out) {
  VendorScope scope(out);
  auto iterator = at::TensorIterator::binary_op(out, self, other);
  DescribeCudaMemory(iterator);
  at::native::mul_stub(at::kCUDA, iterator);
  return out;
}

at::Tensor& PowTensorOut(const at::Tensor& self, const at::Tensor& exponent,
                         at::Tensor& out) {
  VendorScope scope(out);
  auto iterator = at::TensorIterator::binary_op(out, self, exponent);
  DescribeCudaMemory(iterator);
  at::native::pow_tensor_tensor_stub(at::kCUDA, iterator);
  return out;
}

bool Equal(const at::Tensor& self, const at::Tensor& other) {
  if (!at::namedinference::are_names_equal(
          self.unsafeGetTensorImpl(), other.unsafeGetTensorImpl())) {
    return false;
  }
  TORCH_CHECK(self.device() == other.device(),
              "Cannot compare tensors on different devices: ", self.device(),
              " and ", other.device());
  if (!self.sizes().equals(other.sizes())) return false;
  if (self.numel() == 0) return true;
  if (self.is_alias_of(other) && self.storage_offset() == other.storage_offset() &&
      self.dtype() == other.dtype() && self.layout() == other.layout() &&
      self.strides().equals(other.strides()) && self.is_neg() == other.is_neg() &&
      self.is_conj() == other.is_conj()) {
    return true;
  }
  at::NoNamesGuard guard;
  return at::eq(self, other).all().item<bool>();
}

at::Tensor& IndexOut(const at::Tensor& input,
                     const c10::List<std::optional<at::Tensor>>& indices,
                     at::Tensor& out) {
  TORCH_CHECK(out.device() == input.device(), "index: output must be on the input device");
  TORCH_CHECK(out.scalar_type() == input.scalar_type(), "index: output dtype must match the input");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, input);
  VendorScope scope(input);
  StructuredOutput<at::native::structured_index_out> kernel;
  auto metadata = kernel.meta(input, at::IOptTensorListRef(indices));
  kernel.impl(input, std::move(metadata.sizes), std::move(metadata.strides), kernel.output);
  at::native::resize_output(out, kernel.output.sizes());
  return out.copy_(kernel.output);
}

at::Tensor IndexSelect(const at::Tensor& input, int64_t dim, const at::Tensor& index) {
  return WithCudaInputs({input, index}, [&](at::TensorList inputs) {
    const auto native = at::index_select(inputs[0], dim, inputs[1]);
    auto result = at::empty_strided(native.sizes(), native.strides(), input.options());
    CopyCudaResult(result, native);
    return result;
  });
}

at::Tensor& IndexCopyOut(const at::Tensor& input, int64_t dim,
                         const at::Tensor& index, const at::Tensor& source,
                         at::Tensor& out) {
  TORCH_CHECK(out.device() == input.device() && out.scalar_type() == input.scalar_type(),
              "index_copy: output must match the input dtype and device");
  at::assert_no_internal_overlap(out);
  return WithCudaInputs({input, index, source}, [&](at::TensorList inputs) -> at::Tensor& {
    const auto result = at::index_copy(inputs[0], dim, inputs[1], inputs[2]);
    at::native::resize_output(out, result.sizes());
    return CopyCudaResult(out, result);
  });
}

at::Tensor& AddmmOut(const at::Tensor& bias, const at::Tensor& left,
                     const at::Tensor& right, const at::Scalar& beta,
                     const at::Scalar& alpha, at::Tensor& out) {
  return WithCudaInputs({bias, left, right}, [&](at::TensorList inputs) -> at::Tensor& {
    TORCH_CHECK(out.device() == left.device() && out.scalar_type() == left.scalar_type(),
                "addmm: output must have the same dtype and device as the input");
    at::assert_no_internal_overlap(out);
    const auto result = at::addmm(inputs[0], inputs[1], inputs[2], beta, alpha);
    at::native::resize_output(out, result.sizes());
    return CopyCudaResult(out, result);
  });
}

using SolveOutputs = std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>;

SolveOutputs SolveExOut(const at::Tensor& a, const at::Tensor& b, bool left,
                        bool check_errors, at::Tensor& result, at::Tensor& lu,
                        at::Tensor& pivots, at::Tensor& info) {
  return WithCudaInputs({a, b}, [&](at::TensorList inputs) {
    const auto [x, factors, permutation, status] =
        at::_linalg_solve_ex(inputs[0], inputs[1], left, check_errors);
    for (const auto& [output, value] : {std::pair{&result, &x}, std::pair{&lu, &factors},
                                      std::pair{&pivots, &permutation}, std::pair{&info, &status}}) {
      TORCH_CHECK(output->device() == a.device() && output->scalar_type() == value->scalar_type(),
                  "linalg.solve_ex: incompatible output dtype or device");
      at::assert_no_internal_overlap(*output);
      at::native::resize_output(*output, value->sizes());
      CopyCudaResult(*output, *value);
    }
    return std::tie(result, lu, pivots, info);
  });
}

at::Tensor& UpsampleExactOut(const at::Tensor& input, at::IntArrayRef size,
                             std::optional<double> height_scale,
                             std::optional<double> width_scale, at::Tensor& out) {
  TORCH_CHECK(out.device() == input.device() && out.scalar_type() == input.scalar_type(),
              "nearest-exact: output must match the input dtype and device");
  VendorScope scope(input);
  StructuredOutput<at::native::structured__upsample_nearest_exact2d_out_cuda> kernel;
  kernel.meta(input, size, height_scale, width_scale);
  if (kernel.output.numel()) {
    kernel.impl(input, size, height_scale, width_scale, kernel.output);
  }
  at::native::resize_output(out, kernel.output.sizes());
  return out.copy_(kernel.output);
}

at::Tensor& ProdIntOut(const at::Tensor& self, int64_t dim, bool keepdim,
                      std::optional<at::ScalarType> dtype, at::Tensor& out) {
  TORCH_CHECK(out.device() == self.device(), "prod(): output device must match the input");
  TORCH_CHECK(!dtype || *dtype == out.scalar_type(), "prod(): dtype must match the output tensor");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, self);
  VendorScope scope(self);
  StructuredOutput<at::native::structured_prod_out> kernel;
  kernel.meta(self, dim, keepdim, out.scalar_type());
  auto& result = kernel.output;
  if (self.numel() == 0) {
    result.fill_(1);
  } else {
    auto iterator = at::native::make_reduction(
        "prod", result, self, dim, keepdim, out.scalar_type());
    DescribeCudaMemory(iterator);
    at::native::prod_stub(at::kCUDA, iterator);
  }
  at::native::resize_output(out, result.sizes());
  return out.copy_(result);
}

}  // namespace

#include "aten_registration.inc"

}  // namespace at::native::flagos::vendor
