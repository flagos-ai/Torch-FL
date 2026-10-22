// Copyright (c) 2026, BAAI. All rights reserved.
//
// Handwritten mudnn kernels for convolution_overrideable and
// convolution_backward_overrideable.
//
// These two cannot be left to the cpu_fallback like the rest of the uncovered
// ops: ATen's default for `*_overrideable` is a hard
// `TORCH_CHECK(false, "convolution_overrideable not implemented")`, so a
// PrivateUse1 tensor reaching it raises instead of being boxed to CPU. The
// at::musa::* generation used to claim both; with that gone they need real
// kernels, and mudnn's Convolution class provides them.
//
// What mudnn's Convolution supports, measured on MTT S5000 (mudnn v3300):
//   * 2 spatial dims only. `SetNdInfo(1, ...)` is accepted but Run then reports
//     "Unsupported Convolution config", and 3 dims reports "Unexpected tensor
//     format NCHW". conv1d is therefore run as 2D with a unit H dim, which
//     matches CPU exactly; 3D takes the CPU fallback.
//   * NCHW-contiguous operands only ("ConvolutionRun only support contiguous
//     tensor" for channels-last strides), unlike the elementwise ops where a
//     mudnn Tensor's strides are honoured.
//   * float / half / bfloat16. DOUBLE is rejected outright ("Invalid
//     Convolution data type DOUBLE").
//   * groups (incl. depthwise), stride and dilation all match CPU exactly.
//   * The algorithm has to be chosen by trial. GetRecommendForwardAlgorithm can
//     name one that then fails the Run (it picked DIRECT for a grouped conv2d,
//     which reports "Unsupported Convolution configs for algorithm DIRECT"), so
//     the recommendation is only the first candidate.
//
// Bias is *not* fused. `RunFusion` accepts a {1, C, 1, 1} bias only for the
// plain 2D case -- it reports "Bias shape doesn't match" once H == 1 or
// groups > 1 -- so bias is a separate broadcast Binary::ADD, which mudnn reads
// through 0-strides with no materialization.

#ifdef USE_MUSA

#include "mudnn_common.h"

#include "../../common.h"
#include "../../generated/ops.h"

#include <ATen/ATen.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/convolution_backward.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

#include <algorithm>
#include <array>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace at::native::flagos {

namespace {

namespace musa_ops = at::native::flagos::musa_ops;

// float/half/bfloat16 are the dtypes mudnn's Convolution accepts.
bool ConvSupportsDtype(at::ScalarType type) {
  return type == at::kFloat || type == at::kHalf || type == at::kBFloat16;
}

// True when mudnn can run this configuration at all. Everything else (3D,
// transposed, double/integral, mismatched dtypes) goes to CPU.
bool ConvSupported(
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transposed) {
  if (transposed) {
    return false;
  }
  const int64_t spatial = input.dim() - 2;
  if (spatial != 1 && spatial != 2) {
    return false;
  }
  if (weight.dim() != input.dim()) {
    return false;
  }
  return ConvSupportsDtype(input.scalar_type()) &&
      input.scalar_type() == weight.scalar_type();
}

// Repeats a size-1 IntArrayRef out to `spatial` entries, the way aten's
// convolution front-end already does for its own callers.
std::vector<int64_t> ExpandParam(at::IntArrayRef param, int64_t spatial) {
  std::vector<int64_t> out(static_cast<size_t>(spatial), 1);
  for (int64_t i = 0; i < spatial; ++i) {
    out[static_cast<size_t>(i)] =
        param.size() == 1 ? param[0] : param[static_cast<size_t>(i)];
  }
  return out;
}

// A 1-spatial-dim tensor viewed as 2D with a leading unit H: (N, C, L) ->
// (N, C, 1, L). Returns `t` untouched when it is already 4D.
at::Tensor To4d(const at::Tensor& t) {
  return t.dim() == 3 ? t.unsqueeze(2) : t;
}

// Convolution params for the 4D view: a 1D conv becomes a 2D conv whose H
// extent is 1, so H gets pad 0, stride 1, dilation 1.
struct Conv2dParams {
  int pad[2];
  int stride[2];
  int dilation[2];
};

Conv2dParams To2dParams(
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t spatial) {
  const auto s = ExpandParam(stride, spatial);
  const auto p = ExpandParam(padding, spatial);
  const auto d = ExpandParam(dilation, spatial);
  Conv2dParams out{};
  const int lead = spatial == 1 ? 1 : 0;  // 1D: index 0 is the synthetic H
  for (int i = 0; i < 2; ++i) {
    if (i < lead) {
      out.pad[i] = 0;
      out.stride[i] = 1;
      out.dilation[i] = 1;
    } else {
      const size_t j = static_cast<size_t>(i - lead);
      out.pad[i] = static_cast<int>(p[j]);
      out.stride[i] = static_cast<int>(s[j]);
      out.dilation[i] = static_cast<int>(d[j]);
    }
  }
  return out;
}

// Standard aten output extent per spatial dim.
int64_t OutExtent(
    int64_t in, int64_t kernel, int64_t pad, int64_t stride, int64_t dilation) {
  return (in + 2 * pad - (dilation * (kernel - 1) + 1)) / stride + 1;
}

// mudnn::Convolution is non-copyable (ImplBase deletes the copy ctor), so it is
// configured in place rather than returned by value.
void ConfigureConvOp(
    musa_ops::mudnn::Convolution& op,
    const Conv2dParams& params,
    int64_t groups) {
  op.SetNdInfo(2, params.pad, params.stride, params.dilation);
  op.SetGroups(static_cast<int>(groups));
}

// Identifies one convolution configuration, for the algorithm cache below.
// Two calls with the same key accept the same algorithms.
struct ConvKey {
  std::vector<int64_t> in;
  std::vector<int64_t> weight;
  int pad[2];
  int stride[2];
  int dilation[2];
  int64_t groups;
  int dtype;
  int phase;  // 0 fwd, 1 bwd-data, 2 bwd-filter

  bool operator==(const ConvKey& o) const {
    return in == o.in && weight == o.weight && groups == o.groups &&
        dtype == o.dtype && phase == o.phase &&
        std::equal(pad, pad + 2, o.pad) &&
        std::equal(stride, stride + 2, o.stride) &&
        std::equal(dilation, dilation + 2, o.dilation);
  }
};

struct ConvKeyHash {
  size_t operator()(const ConvKey& k) const {
    size_t h = std::hash<int64_t>{}(k.groups) ^ (std::hash<int>{}(k.dtype) << 1) ^
        (std::hash<int>{}(k.phase) << 2);
    const auto mix = [&h](int64_t v) {
      h = h * 1000003u + static_cast<size_t>(v);
    };
    for (int64_t v : k.in) mix(v);
    for (int64_t v : k.weight) mix(v);
    for (int i = 0; i < 2; ++i) {
      mix(k.pad[i]);
      mix(k.stride[i]);
      mix(k.dilation[i]);
    }
    return h;
  }
};

ConvKey MakeConvKey(
    const at::Tensor& in,
    const at::Tensor& weight,
    const Conv2dParams& params,
    int64_t groups,
    int phase) {
  ConvKey key{};
  key.in = in.sizes().vec();
  key.weight = weight.sizes().vec();
  for (int i = 0; i < 2; ++i) {
    key.pad[i] = params.pad[i];
    key.stride[i] = params.stride[i];
    key.dilation[i] = params.dilation[i];
  }
  key.groups = groups;
  key.dtype = static_cast<int>(in.scalar_type());
  key.phase = phase;
  return key;
}

// Runs `run(algo)` over the candidate algorithms until one succeeds, and
// remembers which one worked for this configuration.
//
// The probing exists because a mudnn algorithm can be accepted at configure
// time and then rejected by the Run: GetRecommendForwardAlgorithm named DIRECT
// for a grouped conv2d, which fails with "Unsupported Convolution configs for
// algorithm DIRECT". mudnn logs each such rejection to stderr, so without the
// cache a training loop would reprint it on every step; with it, a given shape
// pays the failed attempt once.
template <typename Algo, typename RunFn>
musa_ops::mudnn::Status RunWithAlgoFallback(
    const ConvKey& key, const std::vector<Algo>& candidates, RunFn&& run) {
  static std::mutex mutex;
  static std::unordered_map<ConvKey, int, ConvKeyHash> cache;

  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
      const auto status = run(static_cast<Algo>(it->second));
      if (status == musa_ops::mudnn::Status::SUCCESS) {
        return status;
      }
      cache.erase(it);  // stale (e.g. a workspace failure); fall through.
    }
  }

  auto status = musa_ops::mudnn::Status::NOT_SUPPORTED;
  for (const Algo algo : candidates) {
    status = run(algo);
    if (status == musa_ops::mudnn::Status::SUCCESS) {
      std::lock_guard<std::mutex> lock(mutex);
      cache[key] = static_cast<int>(algo);
      return status;
    }
  }
  return status;
}

// The recommended algorithm first, then every other one, each listed once.
// Built by these helpers rather than inline, because a braced list inside an
// EXEC_MUDNN_CMD argument would be split on its commas by the preprocessor.
std::vector<musa_ops::mudnn::Convolution::Algorithm> ForwardAlgos(
    musa_ops::mudnn::Convolution::Algorithm recommended) {
  using A = musa_ops::mudnn::Convolution::Algorithm;
  std::vector<A> out{recommended};
  for (const A a : {A::IMPLICIT_GEMM, A::GEMM, A::DIRECT, A::WINOGRAD_NONFUSED}) {
    if (a != recommended) {
      out.push_back(a);
    }
  }
  return out;
}

std::vector<musa_ops::mudnn::Convolution::AlgorithmBwdData> BwdDataAlgos() {
  using A = musa_ops::mudnn::Convolution::AlgorithmBwdData;
  return {A::IMPLICIT_GEMM, A::GEMM, A::DIRECT, A::WINOGRAD_NONFUSED};
}

std::vector<musa_ops::mudnn::Convolution::AlgorithmBwdFilter> BwdFilterAlgos() {
  using A = musa_ops::mudnn::Convolution::AlgorithmBwdFilter;
  return {A::IMPLICIT_GEMM, A::GEMM, A::DIRECT, A::WINOGRAD_NONFUSED};
}

// out = out + bias, with bias broadcast over N and the spatial dims. mudnn
// reads the 0-strides of the expanded view directly, so this is one extra
// elementwise pass and no copy.
at::Tensor AddBias(const at::Tensor& out, const at::Tensor& bias) {
  std::vector<int64_t> view_shape(static_cast<size_t>(out.dim()), 1);
  view_shape[1] = bias.numel();
  const at::Tensor bias_b =
      bias.to(out.scalar_type()).view(view_shape).expand(out.sizes());

  auto result = at::empty(out.sizes(), out.options());
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::MudnnTensorWrapper t_bias(bias_b);
  musa_ops::MudnnTensorWrapper t_result(result);
  musa_ops::mudnn::Binary op;
  op.SetMode(musa_ops::mudnn::Binary::Mode::ADD);
  EXEC_MUDNN_CMD(
      "convolution bias",
      out,
      op.Run(_mudnn_h, t_result.get(), t_out.get(), t_bias.get()));
  return result;
}

} // namespace

at::Tensor ConvolutionOverrideableKernelMusa(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups) {
  if (!ConvSupported(input, weight, transposed)) {
    const auto bias_cpu = bias.has_value() && bias->defined()
        ? std::optional<at::Tensor>(bias->cpu())
        : std::nullopt;
    return at::convolution(
               input.cpu(),
               weight.cpu(),
               bias_cpu,
               stride,
               padding,
               dilation,
               transposed,
               output_padding,
               groups)
        .to(input.device());
  }

  const int64_t spatial = input.dim() - 2;
  const auto params = To2dParams(stride, padding, dilation, spatial);

  // mudnn needs NCHW-contiguous operands, so a channels-last or sliced input is
  // materialized here (MudnnCopy does that on device via clone/contiguous).
  const at::Tensor in4 = To4d(input).contiguous();
  const at::Tensor w4 = To4d(weight).contiguous();

  std::vector<int64_t> out_sizes{in4.size(0), w4.size(0)};
  for (int i = 0; i < 2; ++i) {
    out_sizes.push_back(OutExtent(
        in4.size(2 + i), w4.size(2 + i), params.pad[i], params.stride[i],
        params.dilation[i]));
  }
  auto out4 = at::empty(out_sizes, in4.options());

  musa_ops::MudnnTensorWrapper t_in(in4);
  musa_ops::MudnnTensorWrapper t_w(w4);
  musa_ops::MudnnTensorWrapper t_out(out4);
  musa_ops::mudnn::Convolution op;
  ConfigureConvOp(op, params, groups);
  auto recommended = musa_ops::mudnn::Convolution::Algorithm::IMPLICIT_GEMM;
  {
    musa_ops::MusaDeviceGuard guard(in4);
    op.GetRecommendForwardAlgorithm(
        musa_ops::GetMudnnHandle(), recommended, t_out.get(), t_in.get(),
        t_w.get());
  }
  const auto algos = ForwardAlgos(recommended);
  const auto key = MakeConvKey(in4, w4, params, groups, /*phase=*/0);
  EXEC_MUDNN_CMD(
      "convolution_overrideable",
      in4,
      RunWithAlgoFallback(
          key,
          algos,
          [&](musa_ops::mudnn::Convolution::Algorithm algo) {
            return op.Run(
                _mudnn_h,
                t_out.get(),
                t_in.get(),
                t_w.get(),
                algo,
                musa_ops::MudnnWorkspaceFor(in4));
          }));

  if (bias.has_value() && bias->defined() && bias->numel() > 0) {
    out4 = AddBias(out4, *bias);
  }
  // Drop the synthetic H dim again for the 1D case.
  return spatial == 1 ? out4.squeeze(2) : out4;
}

REGISTER_IMPL_TO_DISPATCHER(
    ConvolutionOverrideableFn,
    convolution_overrideable_dispatcher,
    Backend::kMusa,
    ConvolutionOverrideableKernelMusa)

std::tuple<at::Tensor, at::Tensor, at::Tensor>
ConvolutionBackwardOverrideableKernelMusa(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  if (!ConvSupported(input, weight, transposed) ||
      !ConvSupportsDtype(grad_output.scalar_type())) {
    auto res = at::convolution_backward(
        grad_output.cpu(),
        input.cpu(),
        weight.cpu(),
        std::nullopt,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask);
    const auto to_dev = [&](const at::Tensor& t) {
      return t.defined() ? t.to(input.device()) : t;
    };
    return std::make_tuple(
        to_dev(std::get<0>(res)),
        to_dev(std::get<1>(res)),
        to_dev(std::get<2>(res)));
  }

  const int64_t spatial = input.dim() - 2;
  const auto params = To2dParams(stride, padding, dilation, spatial);

  const at::Tensor in4 = To4d(input).contiguous();
  const at::Tensor w4 = To4d(weight).contiguous();
  const at::Tensor gy4 = To4d(grad_output).contiguous().to(in4.scalar_type());

  musa_ops::mudnn::Convolution op;
  ConfigureConvOp(op, params, groups);
  at::Tensor grad_input;
  at::Tensor grad_weight;
  at::Tensor grad_bias;

  if (output_mask[0]) {
    auto gx4 = at::empty(in4.sizes(), in4.options());
    musa_ops::MudnnTensorWrapper t_gx(gx4);
    musa_ops::MudnnTensorWrapper t_gy(gy4);
    musa_ops::MudnnTensorWrapper t_w(w4);
    const auto algos = BwdDataAlgos();
    const auto key = MakeConvKey(in4, w4, params, groups, /*phase=*/1);
    EXEC_MUDNN_CMD(
        "convolution_backward_overrideable (data)",
        in4,
        RunWithAlgoFallback(
            key,
            algos,
            [&](musa_ops::mudnn::Convolution::AlgorithmBwdData algo) {
              return op.RunBwdData(
                  _mudnn_h,
                  t_gx.get(),
                  t_gy.get(),
                  t_w.get(),
                  algo,
                  musa_ops::MudnnWorkspaceFor(in4));
            }));
    grad_input = spatial == 1 ? gx4.squeeze(2) : gx4;
  }

  if (output_mask[1]) {
    auto gw4 = at::empty(w4.sizes(), w4.options());
    musa_ops::MudnnTensorWrapper t_gw(gw4);
    musa_ops::MudnnTensorWrapper t_in(in4);
    musa_ops::MudnnTensorWrapper t_gy(gy4);
    // RunBwdFilter's `data`/`filter` slots take the forward input and the
    // incoming gradient respectively (verified against a CPU reference).
    const auto algos = BwdFilterAlgos();
    const auto key = MakeConvKey(in4, w4, params, groups, /*phase=*/2);
    EXEC_MUDNN_CMD(
        "convolution_backward_overrideable (filter)",
        in4,
        RunWithAlgoFallback(
            key,
            algos,
            [&](musa_ops::mudnn::Convolution::AlgorithmBwdFilter algo) {
              return op.RunBwdFilter(
                  _mudnn_h,
                  t_gw.get(),
                  t_in.get(),
                  t_gy.get(),
                  algo,
                  musa_ops::MudnnWorkspaceFor(in4));
            }));
    grad_weight = spatial == 1 ? gw4.squeeze(2) : gw4;
  }

  if (output_mask[2]) {
    // Sum over N and the spatial dims. This lands on the generated mudnn
    // Reduce kernel, so it stays on device.
    std::vector<int64_t> dims;
    for (int64_t d = 0; d < grad_output.dim(); ++d) {
      if (d != 1) {
        dims.push_back(d);
      }
    }
    grad_bias = grad_output.sum(dims);
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

REGISTER_IMPL_TO_DISPATCHER(
    ConvolutionBackwardOverrideableFn,
    convolution_backward_overrideable_dispatcher,
    Backend::kMusa,
    ConvolutionBackwardOverrideableKernelMusa)

} // namespace at::native::flagos

#endif // USE_MUSA
