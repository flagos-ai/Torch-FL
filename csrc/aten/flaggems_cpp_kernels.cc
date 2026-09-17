// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// FlagGems C++ dispatch kernels (kFlagGemsCpp).
//
// Thin wrappers that box PrivateUse1 (flagos) tensors to CUDA device-type
// metadata (zero-copy, via DeviceBoxingGuard), call the corresponding
// flag_gems C++ function — which JIT-compiles and launches the Triton kernel
// directly without touching Python or the GIL — then unbox the result back
// to flagos.
//
// This file is compiled only when FLAGGEMS_CPP=ON (csrc/CMakeLists.txt
// defines FLAGOS_FLAGGEMS_CPP). kFlagGemsCpp is registered here; kFlagGems is
// registered by generated/flaggems_python_kernels.cc. Whether an op actually
// dispatches here is stated by the `flaggems_cpp` keys in the active
// backends_<platform>.conf; there is no runtime switch. A wheel built without
// liboperators.so simply has no implementation registered for this backend.

#ifdef FLAGOS_FLAGGEMS_CPP

#include "device_boxing.h"
#include "generated/ops.h"
#include "flag_gems/operators.h"

namespace at::native::flagos {

// ---- mm --------------------------------------------------------
// MmFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&)
at::Tensor MmKernelCpp(const at::Tensor& mat1, const at::Tensor& mat2) {
  DeviceBoxingGuard guard(mat1, mat2);
  auto result = flag_gems::mm_tensor(mat1, mat2);
  UnboxToFlagos(result);
  return result;
}
REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kFlagGemsCpp,
                            MmKernelCpp);

// ---- bmm -------------------------------------------------------
// BmmFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&)
at::Tensor BmmKernelCpp(const at::Tensor& self, const at::Tensor& mat2) {
  DeviceBoxingGuard guard(self, mat2);
  auto result = flag_gems::bmm(self, mat2);
  UnboxToFlagos(result);
  return result;
}
REGISTER_IMPL_TO_DISPATCHER(BmmFn, bmm_dispatcher, Backend::kFlagGemsCpp,
                            BmmKernelCpp);

// ---- bmm.out ---------------------------------------------------
// BmmOutFn = at::Tensor& (*)(const at::Tensor&, const at::Tensor&,
//                             at::Tensor&)
at::Tensor& BmmOutKernelCpp(
    const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  DeviceBoxingGuard guard(self, mat2, out);
  auto result = flag_gems::bmm(self, mat2);
  out.copy_(result);
  return out;
}
REGISTER_IMPL_TO_DISPATCHER(BmmOutFn, bmm_out_dispatcher, Backend::kFlagGemsCpp,
                            BmmOutKernelCpp);

// ---- addmm -----------------------------------------------------
// AddmmFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&,
//                           const at::Tensor&, const at::Scalar&,
//                           const at::Scalar&)
at::Tensor AddmmKernelCpp(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  DeviceBoxingGuard guard(self, mat1, mat2);
  auto result = flag_gems::addmm(self, mat1, mat2, beta, alpha);
  UnboxToFlagos(result);
  return result;
}
REGISTER_IMPL_TO_DISPATCHER(AddmmFn, addmm_dispatcher, Backend::kFlagGemsCpp,
                            AddmmKernelCpp);

// ---- addmm.out -------------------------------------------------
// AddmmOutFn = at::Tensor& (*)(const at::Tensor&, const at::Tensor&,
//                               const at::Tensor&, const at::Scalar&,
//                               const at::Scalar&, at::Tensor&)
at::Tensor& AddmmOutKernelCpp(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  DeviceBoxingGuard guard(self, mat1, mat2, out);
  flag_gems::addmm_out(self, mat1, mat2, beta, alpha, out);
  return out;
}
REGISTER_IMPL_TO_DISPATCHER(AddmmOutFn, addmm_out_dispatcher,
                            Backend::kFlagGemsCpp, AddmmOutKernelCpp);

// ---- embedding -------------------------------------------------
// EmbeddingFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&,
//                               int64_t, bool, bool)
at::Tensor EmbeddingKernelCpp(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  DeviceBoxingGuard guard(weight, indices);
  auto result =
      flag_gems::embedding(weight, indices, padding_idx, scale_grad_by_freq,
                           sparse);
  UnboxToFlagos(result);
  return result;
}
REGISTER_IMPL_TO_DISPATCHER(EmbeddingFn, embedding_dispatcher,
                            Backend::kFlagGemsCpp, EmbeddingKernelCpp);

// ---- _softmax --------------------------------------------------
// PrivSoftmaxFn = at::Tensor (*)(const at::Tensor&, int64_t, bool)
at::Tensor PrivSoftmaxKernelCpp(
    const at::Tensor& self, int64_t dim, bool half_to_float) {
  DeviceBoxingGuard guard(self);
  auto result = flag_gems::softmax(self, dim, half_to_float);
  UnboxToFlagos(result);
  return result;
}
REGISTER_IMPL_TO_DISPATCHER(PrivSoftmaxFn, priv_softmax_dispatcher,
                            Backend::kFlagGemsCpp, PrivSoftmaxKernelCpp);

// ---- _softmax_backward_data ------------------------------------
// PrivSoftmaxBackwardDataFn = at::Tensor (*)(const at::Tensor&,
//     const at::Tensor&, int64_t, at::ScalarType)
at::Tensor PrivSoftmaxBackwardDataKernelCpp(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype) {
  DeviceBoxingGuard guard(grad_output, output);
  auto result =
      flag_gems::softmax_backward(grad_output, output, dim, input_dtype);
  UnboxToFlagos(result);
  return result;
}
REGISTER_IMPL_TO_DISPATCHER(PrivSoftmaxBackwardDataFn,
                            priv_softmax_backward_data_dispatcher,
                            Backend::kFlagGemsCpp,
                            PrivSoftmaxBackwardDataKernelCpp);

// ---- sum -------------------------------------------------------
// SumFn = at::Tensor (*)(const at::Tensor&, std::optional<at::ScalarType>)
at::Tensor SumKernelCpp(
    const at::Tensor& self, ::std::optional<at::ScalarType> dtype) {
  DeviceBoxingGuard guard(self);
  auto result = flag_gems::sum(self, dtype);
  UnboxToFlagos(result);
  return result;
}
REGISTER_IMPL_TO_DISPATCHER(SumFn, sum_dispatcher, Backend::kFlagGemsCpp,
                            SumKernelCpp);

// ---- sum.dim_IntList -------------------------------------------
// SumDimIntlistFn = at::Tensor (*)(const at::Tensor&,
//     at::OptionalIntArrayRef, bool, std::optional<at::ScalarType>)
at::Tensor SumDimIntlistKernelCpp(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    ::std::optional<at::ScalarType> dtype) {
  DeviceBoxingGuard guard(self);
  auto result = flag_gems::sum_dim(self, dim, keepdim, dtype);
  UnboxToFlagos(result);
  return result;
}
REGISTER_IMPL_TO_DISPATCHER(SumDimIntlistFn, sum_dim_intlist_dispatcher,
                            Backend::kFlagGemsCpp, SumDimIntlistKernelCpp);

// ---- max -------------------------------------------------------
// MaxFn = at::Tensor (*)(const at::Tensor&)
at::Tensor MaxKernelCpp(const at::Tensor& self) {
  DeviceBoxingGuard guard(self);
  auto result = flag_gems::max(self);
  UnboxToFlagos(result);
  return result;
}
REGISTER_IMPL_TO_DISPATCHER(MaxFn, max_dispatcher, Backend::kFlagGemsCpp,
                            MaxKernelCpp);

// ---- max.dim ---------------------------------------------------
// MaxDimFn = std::tuple<at::Tensor, at::Tensor> (*)(const at::Tensor&,
//     int64_t, bool)
::std::tuple<at::Tensor, at::Tensor> MaxDimKernelCpp(
    const at::Tensor& self, int64_t dim, bool keepdim) {
  DeviceBoxingGuard guard(self);
  auto [values, indices] = flag_gems::max_dim(self, dim, keepdim);
  UnboxToFlagos(values);
  UnboxToFlagos(indices);
  return {values, indices};
}
REGISTER_IMPL_TO_DISPATCHER(MaxDimFn, max_dim_dispatcher, Backend::kFlagGemsCpp,
                            MaxDimKernelCpp);

// ---- argmax ----------------------------------------------------
// ArgmaxFn = at::Tensor (*)(const at::Tensor&, std::optional<int64_t>, bool)
at::Tensor ArgmaxKernelCpp(
    const at::Tensor& self,
    ::std::optional<int64_t> dim,
    bool keepdim) {
  DeviceBoxingGuard guard(self);
  auto result = flag_gems::argmax(self, dim, keepdim);
  UnboxToFlagos(result);
  return result;
}
REGISTER_IMPL_TO_DISPATCHER(ArgmaxFn, argmax_dispatcher, Backend::kFlagGemsCpp,
                            ArgmaxKernelCpp);

// ---- nonzero ---------------------------------------------------
// NonzeroFn = at::Tensor (*)(const at::Tensor&)
at::Tensor NonzeroKernelCpp(const at::Tensor& self) {
  DeviceBoxingGuard guard(self);
  auto result = flag_gems::nonzero(self);
  UnboxToFlagos(result);
  return result;
}
REGISTER_IMPL_TO_DISPATCHER(NonzeroFn, nonzero_dispatcher, Backend::kFlagGemsCpp,
                            NonzeroKernelCpp);

// ---- sort ------------------------------------------------------
// SortFn = std::tuple<at::Tensor, at::Tensor> (*)(const at::Tensor&,
//     int64_t, bool)
::std::tuple<at::Tensor, at::Tensor> SortKernelCpp(
    const at::Tensor& self, int64_t dim, bool descending) {
  DeviceBoxingGuard guard(self);
  auto [values, indices] = flag_gems::sort(self, dim, descending);
  UnboxToFlagos(values);
  UnboxToFlagos(indices);
  return {values, indices};
}
REGISTER_IMPL_TO_DISPATCHER(SortFn, sort_dispatcher, Backend::kFlagGemsCpp,
                            SortKernelCpp);

// ---- sort.stable -----------------------------------------------
// SortStableFn = std::tuple<at::Tensor, at::Tensor> (*)(const at::Tensor&,
//     std::optional<bool>, int64_t, bool)
::std::tuple<at::Tensor, at::Tensor> SortStableKernelCpp(
    const at::Tensor& self,
    ::std::optional<bool> stable,
    int64_t dim,
    bool descending) {
  DeviceBoxingGuard guard(self);
  auto [values, indices] =
      flag_gems::sort_stable(self, stable, dim, descending);
  UnboxToFlagos(values);
  UnboxToFlagos(indices);
  return {values, indices};
}
REGISTER_IMPL_TO_DISPATCHER(SortStableFn, sort_stable_dispatcher,
                            Backend::kFlagGemsCpp, SortStableKernelCpp);

// ---- topk ------------------------------------------------------
// TopkFn = std::tuple<at::Tensor, at::Tensor> (*)(const at::Tensor&,
//     int64_t, int64_t, bool, bool)
::std::tuple<at::Tensor, at::Tensor> TopkKernelCpp(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  DeviceBoxingGuard guard(self);
  auto [values, indices] = flag_gems::topk(self, k, dim, largest, sorted);
  UnboxToFlagos(values);
  UnboxToFlagos(indices);
  return {values, indices};
}
REGISTER_IMPL_TO_DISPATCHER(TopkFn, topk_dispatcher, Backend::kFlagGemsCpp,
                            TopkKernelCpp);

// ---- zeros -----------------------------------------------------
// ZerosFn = at::Tensor (*)(at::IntArrayRef,
//     std::optional<at::ScalarType>, std::optional<at::Layout>,
//     std::optional<at::Device>, std::optional<bool>)
// Factory op: no input tensor to box. Convert flagos:N device to cuda:N,
// then unbox the created tensor back to flagos.
at::Tensor ZerosKernelCpp(
    at::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory) {
  ::std::optional<at::Device> cuda_device = device;
  if (device.has_value() &&
      device->type() == c10::DeviceType::PrivateUse1) {
    cuda_device = at::Device(c10::DeviceType::CUDA, device->index());
  }
  auto result =
      flag_gems::zeros(size, dtype, layout, cuda_device, pin_memory);
  UnboxToFlagos(result);
  return result;
}
REGISTER_IMPL_TO_DISPATCHER(ZerosFn, zeros_dispatcher, Backend::kFlagGemsCpp,
                            ZerosKernelCpp);

}  // namespace at::native::flagos

#endif  // FLAGOS_FLAGGEMS_CPP
