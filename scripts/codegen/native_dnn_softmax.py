# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate the shared FP32 DNN softmax host wrappers."""

DNN_SOFTMAX = """\
bool CanUseDnnSoftmax(const at::Tensor& self, int64_t dim) {
  return self.scalar_type() == at::kFloat && self.is_contiguous() && !self.is_neg() &&
      self.numel() && self.dim() > 0 &&
      (dim == -1 || dim == self.dim() - 1) && self.size(-1) <= kMaxGrid &&
      self.numel() / self.size(-1) <= kMaxGrid;
}

struct SoftmaxDescriptor {
  cudnnTensorDescriptor_t value = nullptr;

  explicit SoftmaxDescriptor(const at::Tensor& tensor) {
    auto status = cudnnCreateTensorDescriptor(&value);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS,
        "CoreX softmax descriptor: ", cudnnGetErrorString(status));
    const int columns = tensor.size(-1);
    const int rows = tensor.numel() / columns;
    status = cudnnSetTensor4dDescriptor(value, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        rows, columns, 1, 1);
    if (status != CUDNN_STATUS_SUCCESS) {
      cudnnDestroyTensorDescriptor(value);
      TORCH_CHECK(false, "CoreX softmax layout: ", cudnnGetErrorString(status));
    }
  }

  ~SoftmaxDescriptor() { cudnnDestroyTensorDescriptor(value); }
  SoftmaxDescriptor(const SoftmaxDescriptor&) = delete;
  SoftmaxDescriptor& operator=(const SoftmaxDescriptor&) = delete;
};

template <bool logarithmic>
at::Tensor SoftmaxFp32(const at::Tensor& self, int64_t dim, bool half_to_float) {
  if (!half_to_float && CanUseDnnSoftmax(self, dim)) {
    IluvatarDeviceGuard guard(self);
    const auto handle = GetCudnnHandle(IluvatarDeviceIndex(self));
    auto out = at::empty(self.sizes(), self.options());
    const SoftmaxDescriptor descriptor(self);
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    constexpr auto algorithm = logarithmic ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE;
    const auto status = cudnnSoftmaxForward(handle, algorithm,
        CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, descriptor.value, self.const_data_ptr(),
        &beta, descriptor.value, out.mutable_data_ptr());
    if (status == CUDNN_STATUS_SUCCESS) return out;
    TORCH_CHECK(status == CUDNN_STATUS_NOT_SUPPORTED,
        "CoreX softmax: ", cudnnGetErrorString(status));
  }
  auto& dispatcher = logarithmic ? priv_log_softmax_dispatcher : priv_softmax_dispatcher;
  return dispatcher.DispatchBackend(Backend::kFlagGems, self, dim, half_to_float);
}

template <bool logarithmic>
at::Tensor SoftmaxBackwardFp32(const at::Tensor& grad, const at::Tensor& output,
                               int64_t dim, at::ScalarType input_dtype) {
  if (input_dtype == at::kFloat && CanUseDnnSoftmax(output, dim) &&
      grad.device() == output.device() && grad.scalar_type() == at::kFloat &&
      grad.sizes() == output.sizes() && grad.is_contiguous() && !grad.is_neg()) {
    IluvatarDeviceGuard guard(output);
    const auto handle = GetCudnnHandle(IluvatarDeviceIndex(output));
    auto result = at::empty(output.sizes(), output.options());
    const SoftmaxDescriptor descriptor(output);
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    constexpr auto algorithm = logarithmic ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE;
    const auto status = cudnnSoftmaxBackward(handle, algorithm, CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, descriptor.value, output.const_data_ptr(), descriptor.value, grad.const_data_ptr(),
        &beta, descriptor.value, result.mutable_data_ptr());
    if (status == CUDNN_STATUS_SUCCESS) return result;
    TORCH_CHECK(status == CUDNN_STATUS_NOT_SUPPORTED,
        "CoreX softmax backward: ", cudnnGetErrorString(status));
  }
  auto& dispatcher = logarithmic ? priv_log_softmax_backward_data_dispatcher
                                : priv_softmax_backward_data_dispatcher;
  return dispatcher.DispatchBackend(Backend::kFlagGems, grad, output, dim, input_dtype);
}

"""
