// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#if defined(FLAGOS_NATIVE_THEAD_VENDOR)
#include "../../generated/ops.h"
#include "../../../native_host/backends/thead/eager_precision.h"

#include <ATen/ops/empty.h>
#include <c10/core/DeviceGuard.h>
#include <flagos.h>

#include <limits>

namespace at::native::flagos {
namespace {

bool Dense(const at::Tensor& input, at::ScalarType dtype) {
  return input.device().is_privateuseone() && input.scalar_type() == dtype &&
      input.is_contiguous() && !input.is_neg() && input.numel() > 0 &&
      input.numel() <= std::numeric_limits<int>::max() / 2;
}

bool LastDim(const at::Tensor& input, int64_t dim) {
  return input.dim() > 0 && (dim == -1 || dim == input.dim() - 1);
}

bool Aligned16(const at::Tensor& input) {
  return reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 16 == 0;
}

bool SameDense(const at::Tensor& input, const at::Tensor& other) {
  return Dense(other, input.scalar_type()) && input.device() == other.device() &&
      input.sizes().equals(other.sizes());
}

template <typename Launch>
at::Tensor LaunchOnDevice(const at::Tensor& input, at::IntArrayRef shape, Launch&& launch) {
  const c10::OptionalDeviceGuard guard(input.device());
  auto out = at::empty(shape, input.options());
  const auto stream = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(input.device().index()));
  const auto status = launch(out, stream);
  TORCH_CHECK(status == cudaSuccess, "T-Head eager precision kernel: ", cudaGetErrorString(status));
  return out;
}

at::Tensor TryReduce(const at::Tensor& input, at::OptionalIntArrayRef dims, bool keepdim,
                     std::optional<at::ScalarType> dtype, bool mean) {
  if (!Dense(input, at::kFloat) || (dtype && *dtype != at::kFloat) || !dims ||
      dims->size() != 1 || !LastDim(input, dims->front()) || !Aligned16(input)) {
    return {};
  }
  const int64_t width = input.size(-1);
  if (width != 128 && width != 1024) return {};
  auto shape = input.sizes().vec();
  if (keepdim) shape.back() = 1;
  else shape.pop_back();
  return LaunchOnDevice(input, shape, [&](at::Tensor& out, cudaStream_t stream) {
    return thead_eager::ReduceLastDim(input.const_data_ptr<float>(), out.data_ptr<float>(),
        input.numel() / width, width, mean, stream);
  });
}

at::Tensor TryLeadingBFloat16Sum(const at::Tensor& input, at::OptionalIntArrayRef dims,
                                 bool keepdim, std::optional<at::ScalarType> dtype) {
  constexpr int64_t width = 128;
  constexpr int64_t tokens = 128;
  constexpr int64_t axes[] = {0, 1, 2};
  if (!input.device().is_privateuseone() || input.scalar_type() != at::kBFloat16 ||
      (dtype && *dtype != at::kBFloat16) || !dims || !dims->equals(axes) ||
      input.dim() != 4 || input.is_neg() || input.size(0) != 1 ||
      input.size(1) != tokens || (input.size(2) != 8 && input.size(2) != 16) ||
      input.size(3) != width || input.stride(1) != width ||
      input.stride(2) != tokens * width || input.stride(3) != 1 || !Aligned16(input)) {
    return {};
  }
  // Q/K norm weight gradients are stored with heads before tokens in memory.
  const int64_t shape[] = {1, 1, 1, width};
  return LaunchOnDevice(input, at::IntArrayRef(shape).slice(keepdim ? 0 : 3),
      [&](at::Tensor& out, cudaStream_t stream) {
        return thead_eager::SumLeadingBFloat16(
            static_cast<const uint16_t*>(input.const_data_ptr()),
            static_cast<uint16_t*>(out.mutable_data_ptr()), input.numel() / width, stream);
      });
}

at::Tensor MeanDim(const at::Tensor& input, at::OptionalIntArrayRef dims, bool keepdim,
                   std::optional<at::ScalarType> dtype) {
  if (auto exact = TryReduce(input, dims, keepdim, dtype, true); exact.defined()) return exact;
  return mean_dim_dispatcher.DispatchBackend(Backend::kFlagGems, input, dims, keepdim, dtype);
}

at::Tensor SumDim(const at::Tensor& input, at::OptionalIntArrayRef dims, bool keepdim,
                  std::optional<at::ScalarType> dtype) {
  if (auto exact = TryReduce(input, dims, keepdim, dtype, false); exact.defined()) return exact;
  if (auto exact = TryLeadingBFloat16Sum(input, dims, keepdim, dtype); exact.defined()) return exact;
  return sum_dim_intlist_dispatcher.DispatchBackend(Backend::kFlagGems, input, dims, keepdim, dtype);
}

at::Tensor Rsqrt(const at::Tensor& input) {
  if (!Dense(input, at::kFloat)) return rsqrt_dispatcher.DispatchBackend(Backend::kFlagGems, input);
  return LaunchOnDevice(input, input.sizes(), [&](at::Tensor& out, cudaStream_t stream) {
    return thead_eager::Rsqrt(
        input.const_data_ptr<float>(), out.data_ptr<float>(), input.numel(), stream);
  });
}

at::Tensor Softmax(const at::Tensor& input, int64_t dim, bool half_to_float) {
  if (!Dense(input, at::kFloat) || !LastDim(input, dim) ||
      input.size(-1) != 128 || half_to_float) {
    return priv_softmax_dispatcher.DispatchBackend(Backend::kFlagGems, input, dim, half_to_float);
  }
  return LaunchOnDevice(input, input.sizes(), [&](at::Tensor& out, cudaStream_t stream) {
    return thead_eager::Softmax(
        input.const_data_ptr<float>(), out.data_ptr<float>(), input.numel() / 128, stream);
  });
}

at::Tensor SoftmaxBackward(const at::Tensor& grad, const at::Tensor& output, int64_t dim,
                           at::ScalarType dtype) {
  if (dtype != at::kFloat || !Dense(output, dtype) || !SameDense(output, grad) ||
      !LastDim(output, dim) || output.size(-1) != 128) {
    return priv_softmax_backward_data_dispatcher.DispatchBackend(
        Backend::kFlagGems, grad, output, dim, dtype);
  }
  return LaunchOnDevice(output, output.sizes(), [&](at::Tensor& out, cudaStream_t stream) {
    return thead_eager::SoftmaxBackward(grad.const_data_ptr<float>(), output.const_data_ptr<float>(),
        out.data_ptr<float>(), output.numel() / 128, stream);
  });
}

bool WideFloatLast(const at::Tensor& input, int64_t dim) {
  return Dense(input, at::kFloat) && input.dim() == 2 && LastDim(input, dim) &&
      input.size(-1) >= 65536 && input.size(-1) % 4 == 0 && Aligned16(input);
}

at::Tensor LogSoftmax(const at::Tensor& input, int64_t dim, bool half_to_float) {
  if (half_to_float || !WideFloatLast(input, dim)) {
    return priv_log_softmax_dispatcher.DispatchBackend(Backend::kFlagGems, input, dim, half_to_float);
  }
  return LaunchOnDevice(input, input.sizes(), [&](at::Tensor& out, cudaStream_t stream) {
    return thead_eager::WideLogSoftmax(
        input.const_data_ptr<float>(), out.data_ptr<float>(), input.size(0), input.size(-1), stream);
  });
}

at::Tensor LogSoftmaxBackward(const at::Tensor& grad, const at::Tensor& log_prob,
                              int64_t dim, at::ScalarType dtype) {
  if (dtype != at::kFloat || !WideFloatLast(log_prob, dim) ||
      !SameDense(log_prob, grad) || !Aligned16(grad)) {
    return priv_log_softmax_backward_data_dispatcher.DispatchBackend(
        Backend::kFlagGems, grad, log_prob, dim, dtype);
  }
  return LaunchOnDevice(log_prob, log_prob.sizes(), [&](at::Tensor& out, cudaStream_t stream) {
    return thead_eager::WideLogSoftmaxBackward(
        grad.const_data_ptr<float>(), log_prob.const_data_ptr<float>(), out.data_ptr<float>(),
        log_prob.size(0), log_prob.size(-1), stream);
  });
}

at::Tensor Silu(const at::Tensor& input) {
  if (!Dense(input, at::kBFloat16)) {
    return silu_dispatcher.DispatchBackend(Backend::kFlagGems, input);
  }
  return LaunchOnDevice(input, input.sizes(), [&](at::Tensor& out, cudaStream_t stream) {
    return thead_eager::Silu(static_cast<const uint16_t*>(input.const_data_ptr()),
        static_cast<uint16_t*>(out.mutable_data_ptr()), input.numel(), stream);
  });
}

at::Tensor SiluBackward(const at::Tensor& grad, const at::Tensor& input) {
  if (!Dense(input, at::kBFloat16) || !SameDense(input, grad)) {
    return silu_backward_dispatcher.DispatchBackend(Backend::kFlagGems, grad, input);
  }
  return LaunchOnDevice(input, input.sizes(), [&](at::Tensor& out, cudaStream_t stream) {
    return thead_eager::SiluBackward(
        static_cast<const uint16_t*>(grad.const_data_ptr()),
        static_cast<const uint16_t*>(input.const_data_ptr()),
        static_cast<uint16_t*>(out.mutable_data_ptr()), input.numel(), stream);
  });
}

REGISTER_IMPL_TO_DISPATCHER(MeanDimFn, mean_dim_dispatcher, Backend::kThead, MeanDim)
REGISTER_IMPL_TO_DISPATCHER(SumDimIntlistFn, sum_dim_intlist_dispatcher, Backend::kThead, SumDim)
REGISTER_IMPL_TO_DISPATCHER(RsqrtFn, rsqrt_dispatcher, Backend::kThead, Rsqrt)
REGISTER_IMPL_TO_DISPATCHER(PrivSoftmaxFn, priv_softmax_dispatcher, Backend::kThead, Softmax)
REGISTER_IMPL_TO_DISPATCHER(
    PrivSoftmaxBackwardDataFn, priv_softmax_backward_data_dispatcher, Backend::kThead, SoftmaxBackward)
REGISTER_IMPL_TO_DISPATCHER(PrivLogSoftmaxFn, priv_log_softmax_dispatcher, Backend::kThead, LogSoftmax)
REGISTER_IMPL_TO_DISPATCHER(PrivLogSoftmaxBackwardDataFn, priv_log_softmax_backward_data_dispatcher,
    Backend::kThead, LogSoftmaxBackward)
REGISTER_IMPL_TO_DISPATCHER(SiluFn, silu_dispatcher, Backend::kThead, Silu)
REGISTER_IMPL_TO_DISPATCHER(SiluBackwardFn, silu_backward_dispatcher, Backend::kThead, SiluBackward)

}  // namespace
}  // namespace at::native::flagos
#endif
