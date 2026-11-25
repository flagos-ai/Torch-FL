// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Composite PrivateUse1 implementations for PyTorch's SyncBatchNorm primitives:
//   - aten::batch_norm_stats / .out
//   - aten::batch_norm_gather_stats / .out
//   - aten::batch_norm_gather_stats_with_counts / .out
//   - aten::batch_norm_elemt / .out
//   - aten::batch_norm_backward_reduce / .out
//   - aten::batch_norm_backward_elemt / .out
//
// Upstream PyTorch registers these six operators only on the CUDA dispatch key
// and provides no CPU kernels for them, so PrivateUse1's boxed cpu_fallback
// fails with NotImplementedError on non-boxing builds. Expressing them through
// standard ATen reductions and elementwise ops executes directly on FlagOS
// devices across every backend, preserves FP32/FP64 accumulation, and avoids
// the FlagGems #9378 sample-count / Bessel-correction defects.

#include <ATen/AccumulateType.h>
#include <ATen/Functions.h>
#include <c10/util/SmallVector.h>
#include <torch/library.h>

#include <optional>
#include <tuple>

namespace at::flagos {
namespace {

c10::ScalarType SyncBnAccType(c10::ScalarType dtype) {
  return at::toAccumulateType(dtype, /*is_cuda=*/true);
}

c10::SmallVector<int64_t, 4> SyncBnReduceDims(int64_t ndim) {
  c10::SmallVector<int64_t, 4> dims;
  if (ndim > 1) {
    dims.reserve(ndim - 1);
  }
  for (int64_t d = 0; d < ndim; ++d) {
    if (d != 1) {
      dims.push_back(d);
    }
  }
  return dims;
}

c10::SmallVector<int64_t, 5> SyncBnChannelShape(int64_t ndim, int64_t channels) {
  c10::SmallVector<int64_t, 5> shape(ndim, 1);
  shape[1] = channels;
  return shape;
}

at::Tensor ToAcc(const at::Tensor& t, c10::ScalarType acc_dtype) {
  return t.scalar_type() == acc_dtype ? t : t.to(acc_dtype);
}

// Use `add(a, neg(b))`, `mul(a, reciprocal(b))`, and device scalar tensors via
// `full({}, v, t.options())` so backends that leave `sub.Tensor`, `div.Tensor`,
// `add.Scalar`, or `mul.Scalar` at `none` keep all SyncBatchNorm math on the
// accelerated device path (including under `strict_host_routes(true)`).
at::Tensor SubOnDevice(const at::Tensor& a, const at::Tensor& b) {
  return at::add(a, at::neg(b));
}

at::Tensor DivOnDevice(const at::Tensor& a, const at::Tensor& b) {
  return at::mul(a, at::reciprocal(b));
}

at::Tensor AddScalarOnDevice(const at::Tensor& t, double value) {
  return at::add(t, at::full({}, value, t.options()));
}

at::Tensor MulScalarOnDevice(const at::Tensor& t, double value) {
  return at::mul(t, at::full({}, value, t.options()));
}

::std::tuple<at::Tensor, at::Tensor> SyncBatchNormStats(
    const at::Tensor& input, double eps) {
  TORCH_CHECK(
      input.dim() >= 2,
      "batch_norm_stats expects input with dim >= 2, got ",
      input.dim());
  TORCH_CHECK(
      input.numel() > 0, "batch_norm_stats expects non-empty input tensor");

  const auto acc_dtype = SyncBnAccType(input.scalar_type());
  const int64_t channels = input.size(1);
  const int64_t n = input.numel() / channels;
  const auto reduce_dims = SyncBnReduceDims(input.dim());
  const auto view_shape = SyncBnChannelShape(input.dim(), channels);

  at::Tensor x = ToAcc(input, acc_dtype);
  at::Tensor mean = at::mean(x, reduce_dims, /*keepdim=*/false);
  at::Tensor centered = SubOnDevice(x, mean.view(view_shape));
  at::Tensor var_sum = at::sum(at::mul(centered, centered), reduce_dims, /*keepdim=*/false);
  at::Tensor invstd = at::rsqrt(
      AddScalarOnDevice(MulScalarOnDevice(var_sum, 1.0 / static_cast<double>(n)), eps));
  return ::std::make_tuple(mean, invstd);
}

::std::tuple<at::Tensor&, at::Tensor&> SyncBatchNormStatsOut(
    const at::Tensor& input,
    double eps,
    at::Tensor& out0,
    at::Tensor& out1) {
  auto [mean, invstd] = SyncBatchNormStats(input, eps);
  out0.resize_as_(mean).copy_(mean);
  out1.resize_as_(invstd).copy_(invstd);
  return ::std::tie(out0, out1);
}

::std::tuple<at::Tensor, at::Tensor> SyncBatchNormGatherStatsWithCounts(
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const ::std::optional<at::Tensor>& running_mean,
    const ::std::optional<at::Tensor>& running_var,
    double momentum,
    double eps,
    const at::Tensor& counts) {
  TORCH_CHECK(
      mean.dim() == 2 && invstd.dim() == 2,
      "batch_norm_gather_stats_with_counts expects 2-D mean and invstd");
  TORCH_CHECK(
      mean.sizes() == invstd.sizes(),
      "batch_norm_gather_stats_with_counts expects matching mean/invstd sizes");

  const auto acc_dtype = SyncBnAccType(input.scalar_type());
  const int64_t channels = mean.size(1);
  const auto acc_opts = mean.options().dtype(acc_dtype);

  if (mean.size(0) == 0) {
    return ::std::make_tuple(
        at::zeros({channels}, acc_opts), at::zeros({channels}, acc_opts));
  }

  at::Tensor mean_acc = ToAcc(mean, acc_dtype);
  at::Tensor invstd_acc = ToAcc(invstd, acc_dtype);
  at::Tensor counts_acc = counts.to(acc_opts).reshape({-1, 1});
  TORCH_CHECK(
      counts_acc.size(0) == mean_acc.size(0),
      "batch_norm_gather_stats_with_counts expects counts length (",
      counts_acc.size(0),
      ") to match mean.size(0) (",
      mean_acc.size(0),
      ")");

  at::Tensor total_count = at::sum(counts_acc);
  at::Tensor inv_total = at::reciprocal(total_count);
  at::Tensor global_mean =
      at::mul(at::sum(at::mul(mean_acc, counts_acc), /*dim=*/0), inv_total);

  // Each rank's invstd was computed as rsqrt(var_r + eps), so var_r = 1/invstd_r^2 - eps.
  at::Tensor per_rank_var =
      AddScalarOnDevice(at::reciprocal(at::mul(invstd_acc, invstd_acc)), -eps);
  at::Tensor mean_diff = SubOnDevice(mean_acc, global_mean.unsqueeze(0));
  at::Tensor sum_sq = at::sum(
      at::mul(
          counts_acc,
          at::add(per_rank_var, at::mul(mean_diff, mean_diff))),
      /*dim=*/0);

  at::Tensor biased_var = at::mul(sum_sq, inv_total);
  at::Tensor global_invstd = at::rsqrt(AddScalarOnDevice(biased_var, eps));

  if (running_mean.has_value() && running_mean->defined()) {
    at::Tensor rm_acc = ToAcc(*running_mean, acc_dtype);
    at::Tensor updated_rm = at::add(
        MulScalarOnDevice(rm_acc, 1.0 - momentum),
        MulScalarOnDevice(global_mean, momentum));
    running_mean->copy_(updated_rm.to(running_mean->scalar_type()));
  }
  if (running_var.has_value() && running_var->defined()) {
    // Bessel's correction across the global sample count: sum_sq / max(N - 1, 1).
    at::Tensor bessel_denom = at::clamp_min(AddScalarOnDevice(total_count, -1.0), 1.0);
    at::Tensor unbiased_var = DivOnDevice(sum_sq, bessel_denom);
    at::Tensor rv_acc = ToAcc(*running_var, acc_dtype);
    at::Tensor updated_rv = at::add(
        MulScalarOnDevice(rv_acc, 1.0 - momentum),
        MulScalarOnDevice(unbiased_var, momentum));
    running_var->copy_(updated_rv.to(running_var->scalar_type()));
  }

  return ::std::make_tuple(global_mean, global_invstd);
}

::std::tuple<at::Tensor&, at::Tensor&> SyncBatchNormGatherStatsWithCountsOut(
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const ::std::optional<at::Tensor>& running_mean,
    const ::std::optional<at::Tensor>& running_var,
    double momentum,
    double eps,
    const at::Tensor& counts,
    at::Tensor& out0,
    at::Tensor& out1) {
  auto [g_mean, g_invstd] = SyncBatchNormGatherStatsWithCounts(
      input, mean, invstd, running_mean, running_var, momentum, eps, counts);
  out0.resize_as_(g_mean).copy_(g_mean);
  out1.resize_as_(g_invstd).copy_(g_invstd);
  return ::std::tie(out0, out1);
}

::std::tuple<at::Tensor, at::Tensor> SyncBatchNormGatherStats(
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const ::std::optional<at::Tensor>& running_mean,
    const ::std::optional<at::Tensor>& running_var,
    double momentum,
    double eps,
    int64_t count) {
  const auto acc_dtype = SyncBnAccType(input.scalar_type());
  at::Tensor counts =
      at::full({mean.size(0)}, count, mean.options().dtype(acc_dtype));
  return SyncBatchNormGatherStatsWithCounts(
      input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}

::std::tuple<at::Tensor&, at::Tensor&> SyncBatchNormGatherStatsOut(
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const ::std::optional<at::Tensor>& running_mean,
    const ::std::optional<at::Tensor>& running_var,
    double momentum,
    double eps,
    int64_t count,
    at::Tensor& out0,
    at::Tensor& out1) {
  auto [g_mean, g_invstd] = SyncBatchNormGatherStats(
      input, mean, invstd, running_mean, running_var, momentum, eps, count);
  out0.resize_as_(g_mean).copy_(g_mean);
  out1.resize_as_(g_invstd).copy_(g_invstd);
  return ::std::tie(out0, out1);
}

at::Tensor SyncBatchNormElemt(
    const at::Tensor& input,
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double /*eps*/) {
  TORCH_CHECK(
      input.dim() >= 2,
      "batch_norm_elemt expects input with dim >= 2, got ",
      input.dim());
  if (input.numel() == 0) {
    return at::empty_like(input);
  }

  const auto acc_dtype = SyncBnAccType(input.scalar_type());
  const auto view_shape = SyncBnChannelShape(input.dim(), input.size(1));

  at::Tensor x = ToAcc(input, acc_dtype);
  at::Tensor m = ToAcc(mean, acc_dtype).view(view_shape);
  at::Tensor v = ToAcc(invstd, acc_dtype).view(view_shape);
  at::Tensor y = at::mul(SubOnDevice(x, m), v);

  if (weight.has_value() && weight->defined()) {
    y = at::mul(y, ToAcc(*weight, acc_dtype).view(view_shape));
  }
  if (bias.has_value() && bias->defined()) {
    y = at::add(y, ToAcc(*bias, acc_dtype).view(view_shape));
  }
  return y.scalar_type() == input.scalar_type() ? y : y.to(input.scalar_type());
}

at::Tensor& SyncBatchNormElemtOut(
    const at::Tensor& input,
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double eps,
    at::Tensor& out) {
  at::Tensor res = SyncBatchNormElemt(input, weight, bias, mean, invstd, eps);
  out.resize_as_(res).copy_(res);
  return out;
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
SyncBatchNormBackwardReduce(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const ::std::optional<at::Tensor>& weight,
    bool input_g,
    bool weight_g,
    bool bias_g) {
  TORCH_CHECK(
      input.dim() >= 2,
      "batch_norm_backward_reduce expects input with dim >= 2, got ",
      input.dim());

  const auto acc_dtype = SyncBnAccType(input.scalar_type());
  const int64_t channels = input.size(1);
  const auto reduce_dims = SyncBnReduceDims(input.dim());
  const auto view_shape = SyncBnChannelShape(input.dim(), channels);
  const bool has_weight = weight.has_value() && weight->defined();

  at::Tensor dy = ToAcc(grad_out, acc_dtype);
  at::Tensor sum_dy_val;
  if (input_g || bias_g) {
    sum_dy_val = at::sum(dy, reduce_dims, /*keepdim=*/false);
  }

  at::Tensor sum_dy_xmu_val;
  if (input_g || (weight_g && has_weight)) {
    at::Tensor x = ToAcc(input, acc_dtype);
    at::Tensor m = ToAcc(mean, acc_dtype).view(view_shape);
    sum_dy_xmu_val =
        at::sum(at::mul(dy, SubOnDevice(x, m)), reduce_dims, /*keepdim=*/false);
  }

  const auto param_dtype =
      has_weight ? weight->scalar_type() : grad_out.scalar_type();
  at::Tensor grad_weight;
  if (weight_g && has_weight) {
    at::Tensor v = ToAcc(invstd, acc_dtype);
    grad_weight = at::mul(sum_dy_xmu_val, v).to(param_dtype);
  }
  at::Tensor grad_bias;
  if (bias_g) {
    grad_bias = sum_dy_val.to(param_dtype);
  }

  return ::std::make_tuple(
      input_g ? sum_dy_val : at::Tensor(),
      input_g ? sum_dy_xmu_val : at::Tensor(),
      grad_weight,
      grad_bias);
}

::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
SyncBatchNormBackwardReduceOut(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const ::std::optional<at::Tensor>& weight,
    bool input_g,
    bool weight_g,
    bool bias_g,
    at::Tensor& out0,
    at::Tensor& out1,
    at::Tensor& out2,
    at::Tensor& out3) {
  auto [s_dy, s_dy_xmu, gw, gb] = SyncBatchNormBackwardReduce(
      grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
  if (s_dy.defined()) {
    out0.resize_as_(s_dy).copy_(s_dy);
  }
  if (s_dy_xmu.defined()) {
    out1.resize_as_(s_dy_xmu).copy_(s_dy_xmu);
  }
  if (gw.defined()) {
    out2.resize_as_(gw).copy_(gw);
  }
  if (gb.defined()) {
    out3.resize_as_(gb).copy_(gb);
  }
  return ::std::tie(out0, out1, out2, out3);
}

at::Tensor SyncBatchNormBackwardElemt(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const ::std::optional<at::Tensor>& weight,
    const at::Tensor& sum_dy,
    const at::Tensor& sum_dy_xmu,
    const at::Tensor& count) {
  TORCH_CHECK(
      input.dim() >= 2,
      "batch_norm_backward_elemt expects input with dim >= 2, got ",
      input.dim());
  if (input.numel() == 0) {
    return at::empty_like(input);
  }

  const auto acc_dtype = SyncBnAccType(input.scalar_type());
  const auto view_shape = SyncBnChannelShape(input.dim(), input.size(1));

  at::Tensor dy = ToAcc(grad_out, acc_dtype);
  at::Tensor x = ToAcc(input, acc_dtype);
  at::Tensor m = ToAcc(mean, acc_dtype).view(view_shape);
  at::Tensor v = ToAcc(invstd, acc_dtype).view(view_shape);
  at::Tensor s_dy = ToAcc(sum_dy, acc_dtype).view(view_shape);
  at::Tensor s_dy_xmu = ToAcc(sum_dy_xmu, acc_dtype).view(view_shape);

  // Sum `count` across all participating ranks (counts is a 1-D tensor of
  // per-rank element counts saved by SyncBatchNorm.forward).
  at::Tensor total_count = at::sum(count.to(x.options().dtype(acc_dtype)));
  at::Tensor inv_total = at::reciprocal(total_count);
  at::Tensor mean_dy = at::mul(s_dy, inv_total);
  at::Tensor mean_dy_xmu = at::mul(s_dy_xmu, inv_total);

  at::Tensor proj =
      at::mul(SubOnDevice(x, m), at::mul(at::mul(v, v), mean_dy_xmu));
  at::Tensor dx = at::mul(SubOnDevice(SubOnDevice(dy, mean_dy), proj), v);
  if (weight.has_value() && weight->defined()) {
    dx = at::mul(dx, ToAcc(*weight, acc_dtype).view(view_shape));
  }
  return dx.scalar_type() == input.scalar_type() ? dx
                                                 : dx.to(input.scalar_type());
}

at::Tensor& SyncBatchNormBackwardElemtOut(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const ::std::optional<at::Tensor>& weight,
    const at::Tensor& sum_dy,
    const at::Tensor& sum_dy_xmu,
    const at::Tensor& count,
    at::Tensor& out) {
  at::Tensor res = SyncBatchNormBackwardElemt(
      grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
  out.resize_as_(res).copy_(res);
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("batch_norm_stats", SyncBatchNormStats);
  m.impl("batch_norm_stats.out", SyncBatchNormStatsOut);
  m.impl("batch_norm_gather_stats", SyncBatchNormGatherStats);
  m.impl("batch_norm_gather_stats.out", SyncBatchNormGatherStatsOut);
  m.impl(
      "batch_norm_gather_stats_with_counts",
      SyncBatchNormGatherStatsWithCounts);
  m.impl(
      "batch_norm_gather_stats_with_counts.out",
      SyncBatchNormGatherStatsWithCountsOut);
  m.impl("batch_norm_elemt", SyncBatchNormElemt);
  m.impl("batch_norm_elemt.out", SyncBatchNormElemtOut);
  m.impl("batch_norm_backward_reduce", SyncBatchNormBackwardReduce);
  m.impl("batch_norm_backward_reduce.out", SyncBatchNormBackwardReduceOut);
  m.impl("batch_norm_backward_elemt", SyncBatchNormBackwardElemt);
  m.impl("batch_norm_backward_elemt.out", SyncBatchNormBackwardElemtOut);
}

} // namespace at::flagos
