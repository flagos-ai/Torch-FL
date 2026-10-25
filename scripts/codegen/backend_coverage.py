#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measured backend coverage sets, consumed by the conf generators.

Kept as data rather than as backends_*.conf files so that torch_fl/configs/
holds exactly one conf per platform and nothing else.

Two kinds of content live here, and they are maintained differently:

  - ``TILEOPS_OPS`` is GENERATED. scripts/codegen/codegen_tileops.py patches
    this block in place on every run; edit the TileOPs manifest, not this file.
  - ``FLAGGEMS_PYTHON_OPS`` and ``FLAGGEMS_CPP_OPS`` are hand-maintained
    measurements, taken from the installed flag_gems package on CUDA. Update
    them by hand when the FlagGems cohort changes, then regenerate the confs.

gen_vendor_confs.py consumes these as the *ceiling* on what a conf may claim;
each platform then intersects them with its own PrivateUse1 registration set.

FLAGGEMS_PYTHON_OPS is **derived from the generated kernels, not from the
installed package**, and that distinction is the point of this note.

It used to be a hand-maintained list of 640 names. ``mul.Tensor`` was never in it,
although FlagGems has served it for the whole time the list existed, so every
``mul`` fell to ``cpu_fallback`` -- 1522 host round trips per forward on the
OCR-RM value head, the largest single source of copies in that model.

The obvious fix, copying ``flag_gems._FULL_CONFIG`` (1097 entries in 5.4.0), is
wrong: 410 of those cannot produce a kernel here. ``discover_flaggems_ops`` has
static gates (arity, argument types, ``out=`` contract) and a name the package
registers is not a name codegen can wrap. Writing the package list here made
``test_conf_ops_have_flagos_python_kernels`` fail on exactly those 410, which is
the test working correctly.

The ceiling is therefore the set of ops that actually have a
``Backend::kFlagGems`` kernel, recovered from the generated sources::

    # op -> wrapper from register.inc, wrapper body -> dispatcher,
    # dispatcher -> REGISTER_IMPL_TO_DISPATCHER(..., Backend::kFlagGems, ...)
    # in flaggems_python_kernels.cc

A ceiling wider than the generated kernels is a claim the build cannot honour;
this one is a statement of what the build can serve. Re-derive it whenever
``flag_gems`` or the codegen gates change -- a name that gains a kernel and is
missing here is invisible to the router, which is how ``mul.Tensor`` was lost.

The C++ runtime's ops (``FLAGGEMS_CPP_OPS``, the ``mm``/``depthwise_conv`` family
served by liboperators.so) are unioned in on top, because this set is also the
ceiling the C++ set must be a subset of: an op reachable only through
liboperators.so still needs to pass the coverage gate, and deriving from the
Python kernels alone drops ``addmm``, ``bmm`` and ``mm`` -- which is exactly what
``test_flaggems_cpp_set_is_a_subset_of_the_python_set`` caught.

The hold sets (``FLAGGEMS_HOLDS``) are subtracted: an op that keeps a kernel for
``FLAGOS_OP_<op>=flag_gems`` overrides but must not be a default route cannot sit
in the ceiling the confs are built from, or every platform routes it there --
``test_override_only_ops_are_not_default_routes`` is the check, and it is what
caught the first draft of this change.

What that leaves is 640 ops, the same count as the hand-maintained list it
replaces, with two names swapped: ``mul.Tensor`` gains a route and
``alias_copy.out`` loses one. ``mul.Tensor`` is the point of the exercise -- it had
never been in the list despite having a kernel, and at 1522 calls per forward it
was the largest single source of host round trips in the OCR-RM value head.
``alias_copy.out`` has no ``kFlagGems`` dispatcher, so it cannot be routed.
"""

# Ops held OFF the FlagGems route on every platform.
#
# Mirrors ``codegen_ops.FLAGGEMS_RUNTIME_BROKEN`` (plus its now-empty
# ``FLAGGEMS_RECURSIVE_FALLBACK``). Duplicated here as data rather than imported,
# because the conf generators run on hosts where ``codegen_ops`` cannot be
# imported -- it needs torchgen and the aten schema, and it exits from import
# when torchgen is absent. An import that silently degrades to an empty set
# makes the generated confs depend on the host: measured on CI, an empty hold list
# turned backends_metax.conf into 593/1434 instead of 586/1441, so ``--check``
# reported a file the repository has as stale.
#
# codegen_ops.py remains the source of truth for *why* each op is held; this copy
# is the routing consequence. Keep the two in step -- a test asserts it.
FLAGGEMS_HOLDS = frozenset({
    "_cdist_backward", "_embedding_bag_per_sample_weights_backward", "_fake_quantize_learnable_per_tensor_affine_backward", "_log_softmax.out",
    "_log_softmax_backward_data.out", "_native_batch_norm_legit_functional", "_softmax.out", "_softmax_backward_data.out",
    "_upsample_bilinear2d_aa", "_upsample_nearest_exact2d_backward", "adaptive_max_pool3d_backward", "addcdiv.out",
    "addcmul.out", "as_strided_copy.out", "baddbmm.out", "binary_cross_entropy",
    "binary_cross_entropy.out", "binary_cross_entropy_with_logits", "cudnn_batch_norm_backward", "gcd",
    "gcd.out", "hardtanh_backward", "igammac", "igammac.out",
    "igammac_", "im2col", "index_select", "leaky_relu_",
    "linalg_ldl_solve", "logit.out", "max_unpool3d", "median.dim_values",
    "median.out", "mul_.Tensor", "nanmedian.dim_values", "nanmedian.out",
    "native_layer_norm_backward", "smooth_l1_loss", "smooth_l1_loss.out", "smooth_l1_loss_backward",
    "special_bessel_j1", "special_i1.out", "special_modified_bessel_i0", "special_modified_bessel_i0.out",
    "unsqueeze", "unsqueeze_", "upsample_bilinear2d", "upsample_trilinear3d",
})  # fmt: skip

# The FlagGems ceiling as it stood before the 2026-10-22 CUDA re-alignment.
#
# ``FLAGGEMS_PYTHON_OPS`` is what the cuda conf is built from. Every other vendor
# keeps this frozen copy, because widening their ceiling is a route change on
# hardware nobody measured this round: it moves ops onto FlagGems' Python path on
# ascend, gcu, musa, ppu, metax and dcu, each of which needs its own accuracy and
# performance evidence before the wider set can be trusted there. The 49 ops cuda
# gains this way (``mul.Tensor`` among them) are the ones those platforms still
# route to their own kernels.
#
# This is a floor, not a target. A vendor stops reading it the moment someone
# runs the differential survey on that platform and lands the result -- the same
# way ``BOXING_TRITON_GAPS`` records per-platform measurements. Until then the
# frozen set is what keeps this change from silently re-routing six platforms it
# never measured.
FLAGGEMS_PYTHON_OPS_FROZEN = frozenset({
    "_adaptive_avg_pool2d", "_adaptive_avg_pool2d_backward", "_adaptive_avg_pool3d_backward", "_batch_norm_no_update",
    "_batch_norm_with_update_functional", "_cdist_forward", "_cholesky_solve_helper", "_compute_linear_combination",
    "_compute_linear_combination.out", "_conj", "_conj_copy.out", "_conv_depthwise2d",
    "_convert_weight_to_int4pack", "_dirichlet_grad", "_efficient_attention_backward", "_embedding_bag_dense_backward",
    "_euclidean_dist", "_fake_quantize_learnable_per_channel_affine_backward", "_fake_quantize_learnable_per_tensor_affine", "_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
    "_flash_attention_backward", "_fused_moving_avg_obs_fq_helper", "_fused_rms_norm", "_fused_rms_norm_backward",
    "_is_all_true", "_linalg_eigvals", "_log_softmax", "_log_softmax_backward_data",
    "_masked_scale", "_native_batch_norm_legit", "_native_batch_norm_legit.no_stats", "_native_batch_norm_legit.no_stats_out",
    "_native_batch_norm_legit.out", "_native_batch_norm_legit_no_training", "_pdist_backward", "_pdist_forward",
    "_prelu_kernel", "_scaled_dot_product_cudnn_attention_backward", "_scaled_dot_product_efficient_attention", "_scaled_dot_product_efficient_attention_backward",
    "_scaled_dot_product_flash_attention_backward", "_softmax", "_softmax_backward_data", "_sparse_semi_structured_addmm",
    "_sparse_semi_structured_mm", "_thnn_fused_lstm_cell", "_thnn_fused_lstm_cell_backward_impl", "_unique2",
    "_unsafe_view", "_upsample_bicubic2d_aa_backward", "_upsample_bilinear2d_aa_backward", "_upsample_nearest_exact1d_backward",
    "_upsample_nearest_exact1d_backward.grad_input", "_weight_int8pack_mm", "_weight_norm_interface", "_weight_norm_interface_backward",
    "abs", "abs_", "acos", "acos_",
    "acosh", "acosh_", "adaptive_avg_pool3d_backward.grad_input", "adaptive_max_pool2d",
    "adaptive_max_pool2d_backward", "adaptive_max_pool3d", "add.Tensor", "add_.Tensor",
    "addbmm", "addbmm_", "addcdiv", "addcdiv_",
    "addcmul", "addcmul_", "addmm", "addmm.dtype",
    "addmm.dtype_out", "addmm.out", "addmm_", "addmv",
    "addmv.out", "addmv_", "addr", "addr_",
    "affine_grid_generator", "alias", "alias_copy.out", "all",
    "all.dim", "all.dims", "amax", "amin",
    "aminmax", "angle", "any", "any.dim",
    "any.dims", "arange", "arange.start", "arange.start_step",
    "argmax", "argmin", "asin", "asin_",
    "asinh", "asinh.out", "asinh_", "atan",
    "atan2", "atan2.out", "atan2_", "atan_",
    "atanh", "atanh_", "avg_pool2d", "avg_pool2d_backward",
    "avg_pool3d", "avg_pool3d_backward", "baddbmm", "baddbmm_",
    "bernoulli_.float", "binary_cross_entropy_backward", "bincount", "bitwise_and.Scalar",
    "bitwise_and.Scalar_Tensor", "bitwise_and.Tensor", "bitwise_and_.Scalar", "bitwise_and_.Tensor",
    "bitwise_not", "bitwise_not_", "bitwise_or.Scalar", "bitwise_or.Scalar_Tensor",
    "bitwise_or.Tensor", "bitwise_or_.Scalar", "bitwise_or_.Tensor", "bitwise_right_shift.Tensor",
    "bitwise_right_shift_.Tensor", "bitwise_xor.Scalar", "bitwise_xor.Scalar_Tensor", "bitwise_xor.Tensor",
    "bitwise_xor_.Scalar", "bitwise_xor_.Tensor", "blackman_window", "blackman_window.periodic",
    "bmm", "bmm.out", "bucketize.Tensor", "cat",
    "ceil", "ceil.out", "ceil_", "celu",
    "celu_", "channel_shuffle", "cholesky_inverse", "cholesky_solve",
    "cholesky_solve.out", "clamp", "clamp.Tensor", "clamp_",
    "clamp_.Tensor", "clamp_max", "clamp_max_", "clamp_min",
    "clamp_min_", "col2im", "conj_physical_", "constant_pad_nd",
    "convolution_overrideable", "convolution_overrideable.out", "copysign.out", "copysign_.Tensor",
    "cos", "cos_", "cosh", "cosh.out",
    "cosh_", "count_nonzero", "cudnn_convolution", "cudnn_convolution_transpose",
    "cummax", "cummin", "cumprod", "cumprod_",
    "cumsum", "cumsum.out", "cumsum_", "deg2rad",
    "deg2rad.out", "deg2rad_", "dequantize.self", "diagonal_backward",
    "digamma", "digamma_", "dist", "div.Scalar",
    "div.Scalar_mode", "div.Tensor", "div.Tensor_mode", "div.out",
    "div_.Scalar", "div_.Scalar_mode", "div_.Tensor", "div_.Tensor_mode",
    "dot", "elu", "elu_", "elu_backward",
    "embedding", "embedding_renorm_", "empty_permuted", "eq.Scalar",
    "eq.Tensor", "eq_.Scalar", "eq_.Tensor", "erf",
    "erf_", "erfc", "erfc_", "erfinv",
    "erfinv_", "exp", "exp.out", "exp2",
    "exp2_", "exp_", "expm1", "expm1.out",
    "expm1_", "exponential_", "eye", "eye.m",
    "fake_quantize_per_channel_affine_cachemask", "fake_quantize_per_channel_affine_cachemask.out", "fill.Scalar", "fill.Scalar_out",
    "fill.Tensor", "fill.Tensor_out", "fill_.Scalar", "fill_.Tensor",
    "flip", "floor", "floor.out", "floor_",
    "floor_divide", "floor_divide.Scalar", "floor_divide_.Scalar", "floor_divide_.Tensor",
    "fmax", "fmax.out", "fmin", "fmin.out",
    "fmod.Scalar", "fmod.Tensor", "fmod_.Scalar", "fmod_.Tensor",
    "frac", "frac_", "fractional_max_pool2d", "fractional_max_pool2d_backward",
    "full", "full_like", "gcd_", "ge.Scalar",
    "ge.Tensor", "gelu", "gelu_", "gelu_backward",
    "glu", "glu_backward", "grid_sampler_3d", "grid_sampler_3d_backward",
    "gt.Scalar", "gt.Tensor", "gt_.Scalar", "gt_.Tensor",
    "hardshrink", "hardshrink.out", "hardsigmoid", "hardsigmoid.out",
    "hardsigmoid_backward", "hardswish", "hardswish.out", "hardswish_",
    "hardswish_backward", "hardtanh", "hardtanh.out", "hardtanh_",
    "hash_tensor", "heaviside", "heaviside_", "histc",
    "huber_loss", "huber_loss.out", "hypot", "hypot_",
    "i0", "i0.out", "igamma", "igamma_",
    "index_add", "index_add_", "index_copy", "index_copy_",
    "index_fill.int_Scalar", "index_fill.int_Tensor", "index_fill_.int_Scalar", "index_fill_.int_Tensor",
    "index_reduce_", "isin.Scalar_Tensor", "isin.Tensor_Scalar", "isin.Tensor_Tensor",
    "isinf", "isnan", "isneginf", "isneginf.out",
    "isposinf", "kthvalue", "lcm", "lcm_",
    "le.Scalar", "le.Tensor", "le_.Scalar", "le_.Tensor",
    "leaky_relu", "leaky_relu.out", "leaky_relu_backward", "lerp.Scalar",
    "lerp.Tensor", "lerp_.Scalar", "lerp_.Tensor", "lgamma",
    "lgamma_", "lift", "lift.out", "linalg_cross",
    "linalg_cross.out", "linalg_eig", "linalg_householder_product", "linalg_ldl_factor_ex",
    "linalg_lstsq", "linalg_lu", "linalg_lu.out", "linalg_lu_factor_ex",
    "linalg_lu_factor_ex.out", "linalg_matrix_exp", "linalg_matrix_exp.out", "linalg_qr",
    "linalg_qr.out", "linalg_solve_triangular", "linalg_solve_triangular.out", "linalg_vector_norm",
    "linspace", "log", "log10", "log10.out",
    "log10_", "log1p", "log1p.out", "log1p_",
    "log2", "log2_", "log_sigmoid_backward", "log_sigmoid_backward.grad_input",
    "log_sigmoid_forward", "logaddexp", "logaddexp.out", "logaddexp2",
    "logaddexp2.out", "logcumsumexp", "logcumsumexp.out", "logical_and",
    "logical_and_", "logical_not", "logical_not_", "logical_or",
    "logical_or_", "logical_xor", "logical_xor_", "logit",
    "logit_", "logit_backward", "logspace", "logsumexp",
    "lt.Scalar", "lt.Tensor", "lt_.Scalar", "lt_.Tensor",
    "lu_unpack", "lu_unpack.out", "masked_fill.Scalar", "masked_fill.Tensor",
    "masked_fill_.Scalar", "masked_fill_.Tensor", "masked_scatter", "masked_scatter_",
    "masked_scatter_backward", "masked_select", "max", "max.dim",
    "max_pool2d_with_indices", "max_pool2d_with_indices_backward", "max_pool3d_with_indices", "max_pool3d_with_indices_backward",
    "max_unpool2d", "mean", "mean.dim", "median",
    "median.dim", "min", "min.dim", "miopen_batch_norm",
    "miopen_batch_norm_backward", "mish", "mish_", "mm",
    "mm.out", "mode", "mse_loss", "mse_loss_backward",
    "multinomial", "mv", "nan_to_num", "nan_to_num_",
    "nanmedian", "nanmedian.dim", "nansum", "nansum.out",
    "native_batch_norm", "native_batch_norm_backward", "native_dropout", "native_dropout_backward",
    "native_group_norm", "native_group_norm_backward", "native_layer_norm", "ne.Scalar",
    "ne.Tensor", "ne_.Scalar", "ne_.Tensor", "neg",
    "neg_", "new_ones", "nextafter", "nextafter.out",
    "nextafter_", "nll_loss2d_backward", "nll_loss2d_forward", "nll_loss_backward",
    "nll_loss_forward", "nonzero", "nonzero_static", "norm.Scalar",
    "norm.ScalarOpt_dim", "ones", "ones_like", "ormqr",
    "pixel_unshuffle.out", "polar", "polygamma", "polygamma.out",
    "polygamma_", "pow.Scalar", "pow.Tensor_Scalar", "pow.Tensor_Tensor",
    "pow_.Scalar", "pow_.Tensor", "prod", "prod.dim_int",
    "rad2deg", "rad2deg_", "rand", "rand_like",
    "randint", "randint_like", "randn", "randn_like",
    "randperm", "range", "reciprocal", "reciprocal_",
    "reflection_pad1d", "reflection_pad1d.out", "reflection_pad1d_backward", "reflection_pad2d",
    "reflection_pad2d.out", "reflection_pad2d_backward", "reflection_pad3d", "reflection_pad3d.out",
    "reflection_pad3d_backward", "relu", "relu_", "remainder.Scalar",
    "remainder.Scalar_Tensor", "remainder.Tensor", "remainder_.Scalar", "remainder_.Tensor",
    "renorm", "renorm_", "repeat_interleave.Tensor", "replication_pad1d",
    "replication_pad1d.out", "replication_pad2d", "replication_pad2d.out", "replication_pad2d_backward",
    "replication_pad2d_backward.grad_input", "replication_pad3d", "replication_pad3d_backward", "roll",
    "rot90", "round", "round.out", "round_",
    "rrelu_with_noise_backward", "rsqrt", "rsqrt_", "rsub.Scalar",
    "rsub.Tensor", "scalar_tensor", "scatter.reduce", "scatter.src",
    "scatter_.reduce", "scatter_.src", "scatter_add", "scatter_add_",
    "scatter_reduce.two", "scatter_reduce.two_out", "scatter_reduce_.two", "sgn",
    "sgn.out", "sgn_", "sigmoid", "sigmoid_",
    "sigmoid_backward", "sign", "sign.out", "signbit",
    "signbit.out", "silu", "silu_", "silu_backward",
    "sin", "sin_", "sinc", "sinc_",
    "sinh", "sinh_", "slice.Tensor", "slice_backward",
    "soft_margin_loss", "soft_margin_loss_backward", "softplus", "softplus_backward",
    "softshrink", "softshrink.out", "sort", "sort.stable",
    "special_airy_ai", "special_airy_ai.out", "special_bessel_j0", "special_bessel_y0",
    "special_bessel_y1", "special_chebyshev_polynomial_u", "special_chebyshev_polynomial_u.n_scalar", "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w", "special_chebyshev_polynomial_w.out", "special_erfcx", "special_hermite_polynomial_h",
    "special_hermite_polynomial_h.n_scalar", "special_i0e", "special_i0e.out", "special_i1",
    "special_i1e", "special_i1e.out", "special_legendre_polynomial_p", "special_log_ndtr",
    "special_modified_bessel_k0", "special_modified_bessel_k0.out", "special_modified_bessel_k1", "special_modified_bessel_k1.out",
    "special_ndtri", "special_scaled_modified_bessel_k1", "special_scaled_modified_bessel_k1.out", "special_shifted_chebyshev_polynomial_t",
    "special_shifted_chebyshev_polynomial_u", "special_shifted_chebyshev_polynomial_v", "special_shifted_chebyshev_polynomial_w", "special_xlog1py",
    "sqrt", "sqrt_", "std.correction", "sub.Tensor",
    "sub_.Tensor", "sum", "sum.IntList_out", "sum.dim_IntList",
    "sum.out", "take", "take.out", "tan",
    "tan_", "tanh", "tanh_", "tanh_backward",
    "threshold", "threshold_", "threshold_backward", "topk",
    "trace", "transpose.int", "tril", "tril.out",
    "tril_", "triu", "triu_", "trunc",
    "trunc_", "unfold_backward", "uniform_", "unique_consecutive",
    "unique_dim", "upsample_bicubic2d", "upsample_linear1d_backward", "var.correction",
    "var_mean.correction", "vdot", "view_as_complex", "where.self",
    "where.self_out", "xlogy.OutScalar_Other", "xlogy.OutScalar_Self", "xlogy.OutTensor",
    "xlogy.Scalar_Other", "xlogy.Scalar_Self", "xlogy.Tensor", "xlogy_.Scalar_Other",
    "xlogy_.Tensor", "zero_", "zeros", "zeros_like",
})  # fmt: skip

# Ops flag_gems exposes on its Python/Triton path (Backend::kFlagGems).
# Measured on CUDA, so this is a ceiling and not a per-platform routing set.
FLAGGEMS_PYTHON_OPS = frozenset({
    "_adaptive_avg_pool2d", "_adaptive_avg_pool2d_backward", "_adaptive_avg_pool3d_backward", "_batch_norm_no_update",
    "_batch_norm_with_update_functional", "_cdist_forward", "_cholesky_solve_helper", "_compute_linear_combination",
    "_compute_linear_combination.out", "_conj", "_conj_copy.out", "_conv_depthwise2d",
    "_convert_weight_to_int4pack", "_dirichlet_grad", "_efficient_attention_backward", "_embedding_bag_dense_backward",
    "_euclidean_dist", "_fake_quantize_learnable_per_channel_affine_backward", "_fake_quantize_learnable_per_tensor_affine", "_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
    "_flash_attention_backward", "_fused_moving_avg_obs_fq_helper", "_fused_rms_norm", "_fused_rms_norm_backward",
    "_is_all_true", "_linalg_eigvals", "_log_softmax", "_log_softmax_backward_data",
    "_masked_scale", "_native_batch_norm_legit", "_native_batch_norm_legit.no_stats", "_native_batch_norm_legit.no_stats_out",
    "_native_batch_norm_legit.out", "_native_batch_norm_legit_no_training", "_pdist_backward", "_pdist_forward",
    "_prelu_kernel", "_scaled_dot_product_cudnn_attention_backward", "_scaled_dot_product_efficient_attention", "_scaled_dot_product_efficient_attention_backward",
    "_scaled_dot_product_flash_attention_backward", "_softmax", "_softmax_backward_data", "_sparse_semi_structured_addmm",
    "_sparse_semi_structured_mm", "_thnn_fused_lstm_cell", "_thnn_fused_lstm_cell_backward_impl", "_unique2",
    "_unsafe_view", "_upsample_bicubic2d_aa_backward", "_upsample_bilinear2d_aa_backward", "_upsample_nearest_exact1d_backward",
    "_upsample_nearest_exact1d_backward.grad_input", "_weight_int8pack_mm", "_weight_norm_interface", "_weight_norm_interface_backward",
    "abs", "abs_", "acos", "acos_",
    "acosh", "acosh_", "adaptive_avg_pool3d_backward.grad_input", "adaptive_max_pool2d",
    "adaptive_max_pool2d_backward", "adaptive_max_pool3d", "add.Tensor", "add_.Tensor",
    "addbmm", "addbmm_", "addcdiv", "addcdiv_",
    "addcmul", "addcmul_", "addmm", "addmm.dtype",
    "addmm.dtype_out", "addmm.out", "addmm_", "addmv",
    "addmv.out", "addmv_", "addr", "addr_",
    "affine_grid_generator", "alias", "all", "all.dim",
    "all.dims", "amax", "amin", "aminmax",
    "angle", "any", "any.dim", "any.dims",
    "arange", "arange.start", "arange.start_step", "argmax",
    "argmin", "asin", "asin_", "asinh",
    "asinh.out", "asinh_", "atan", "atan2",
    "atan2.out", "atan2_", "atan_", "atanh",
    "atanh_", "avg_pool2d", "avg_pool2d_backward", "avg_pool3d",
    "avg_pool3d_backward", "baddbmm", "baddbmm_", "bernoulli_.float",
    "binary_cross_entropy_backward", "bincount", "bitwise_and.Scalar", "bitwise_and.Scalar_Tensor",
    "bitwise_and.Tensor", "bitwise_and_.Scalar", "bitwise_and_.Tensor", "bitwise_not",
    "bitwise_not_", "bitwise_or.Scalar", "bitwise_or.Scalar_Tensor", "bitwise_or.Tensor",
    "bitwise_or_.Scalar", "bitwise_or_.Tensor", "bitwise_right_shift.Tensor", "bitwise_right_shift_.Tensor",
    "bitwise_xor.Scalar", "bitwise_xor.Scalar_Tensor", "bitwise_xor.Tensor", "bitwise_xor_.Scalar",
    "bitwise_xor_.Tensor", "blackman_window", "blackman_window.periodic", "bmm",
    "bmm.out", "bucketize.Tensor", "cat", "ceil",
    "ceil.out", "ceil_", "celu", "celu_",
    "channel_shuffle", "cholesky_inverse", "cholesky_solve", "cholesky_solve.out",
    "clamp", "clamp.Tensor", "clamp_", "clamp_.Tensor",
    "clamp_max", "clamp_max_", "clamp_min", "clamp_min_",
    "col2im", "conj_physical_", "constant_pad_nd", "convolution_overrideable",
    "convolution_overrideable.out", "copysign.out", "copysign_.Tensor", "cos",
    "cos_", "cosh", "cosh.out", "cosh_",
    "count_nonzero", "cudnn_convolution", "cudnn_convolution_transpose", "cummax",
    "cummin", "cumprod", "cumprod_", "cumsum",
    "cumsum.out", "cumsum_", "deg2rad", "deg2rad.out",
    "deg2rad_", "dequantize.self", "diagonal_backward", "digamma",
    "digamma_", "dist", "div.Scalar", "div.Scalar_mode",
    "div.Tensor", "div.Tensor_mode", "div.out", "div_.Scalar",
    "div_.Scalar_mode", "div_.Tensor", "div_.Tensor_mode", "dot",
    "elu", "elu_", "elu_backward", "embedding",
    "embedding_renorm_", "empty_permuted", "eq.Scalar", "eq.Tensor",
    "eq_.Scalar", "eq_.Tensor", "erf", "erf_",
    "erfc", "erfc_", "erfinv", "erfinv_",
    "exp", "exp.out", "exp2", "exp2_",
    "exp_", "expm1", "expm1.out", "expm1_",
    "exponential_", "eye", "eye.m", "fake_quantize_per_channel_affine_cachemask",
    "fake_quantize_per_channel_affine_cachemask.out", "fill.Scalar", "fill.Scalar_out", "fill.Tensor",
    "fill.Tensor_out", "fill_.Scalar", "fill_.Tensor", "flip",
    "floor", "floor.out", "floor_", "floor_divide",
    "floor_divide.Scalar", "floor_divide_.Scalar", "floor_divide_.Tensor", "fmax",
    "fmax.out", "fmin", "fmin.out", "fmod.Scalar",
    "fmod.Tensor", "fmod_.Scalar", "fmod_.Tensor", "frac",
    "frac_", "fractional_max_pool2d", "fractional_max_pool2d_backward", "full",
    "full_like", "gcd_", "ge.Scalar", "ge.Tensor",
    "gelu", "gelu_", "gelu_backward", "glu",
    "glu_backward", "grid_sampler_3d", "grid_sampler_3d_backward", "gt.Scalar",
    "gt.Tensor", "gt_.Scalar", "gt_.Tensor", "hardshrink",
    "hardshrink.out", "hardsigmoid", "hardsigmoid.out", "hardsigmoid_backward",
    "hardswish", "hardswish.out", "hardswish_", "hardswish_backward",
    "hardtanh", "hardtanh.out", "hardtanh_", "hash_tensor",
    "heaviside", "heaviside_", "histc", "huber_loss",
    "huber_loss.out", "hypot", "hypot_", "i0",
    "i0.out", "igamma", "igamma_", "index_add",
    "index_add_", "index_copy", "index_copy_", "index_fill.int_Scalar",
    "index_fill.int_Tensor", "index_fill_.int_Scalar", "index_fill_.int_Tensor", "index_reduce_",
    "isin.Scalar_Tensor", "isin.Tensor_Scalar", "isin.Tensor_Tensor", "isinf",
    "isnan", "isneginf", "isneginf.out", "isposinf",
    "kthvalue", "lcm", "lcm_", "le.Scalar",
    "le.Tensor", "le_.Scalar", "le_.Tensor", "leaky_relu",
    "leaky_relu.out", "leaky_relu_backward", "lerp.Scalar", "lerp.Tensor",
    "lerp_.Scalar", "lerp_.Tensor", "lgamma", "lgamma_",
    "lift", "lift.out", "linalg_cross", "linalg_cross.out",
    "linalg_eig", "linalg_householder_product", "linalg_ldl_factor_ex", "linalg_lstsq",
    "linalg_lu", "linalg_lu.out", "linalg_lu_factor_ex", "linalg_lu_factor_ex.out",
    "linalg_matrix_exp", "linalg_matrix_exp.out", "linalg_qr", "linalg_qr.out",
    "linalg_solve_triangular", "linalg_solve_triangular.out", "linalg_vector_norm", "linspace",
    "log", "log10", "log10.out", "log10_",
    "log1p", "log1p.out", "log1p_", "log2",
    "log2_", "log_sigmoid_backward", "log_sigmoid_backward.grad_input", "log_sigmoid_forward",
    "logaddexp", "logaddexp.out", "logaddexp2", "logaddexp2.out",
    "logcumsumexp", "logcumsumexp.out", "logical_and", "logical_and_",
    "logical_not", "logical_not_", "logical_or", "logical_or_",
    "logical_xor", "logical_xor_", "logit", "logit_",
    "logit_backward", "logspace", "logsumexp", "lt.Scalar",
    "lt.Tensor", "lt_.Scalar", "lt_.Tensor", "lu_unpack",
    "lu_unpack.out", "masked_fill.Scalar", "masked_fill.Tensor", "masked_fill_.Scalar",
    "masked_fill_.Tensor", "masked_scatter", "masked_scatter_", "masked_scatter_backward",
    "masked_select", "max", "max.dim", "max_pool2d_with_indices",
    "max_pool2d_with_indices_backward", "max_pool3d_with_indices", "max_pool3d_with_indices_backward", "max_unpool2d",
    "mean", "mean.dim", "median", "median.dim",
    "min", "min.dim", "miopen_batch_norm", "miopen_batch_norm_backward",
    "mish", "mish_", "mm", "mm.out",
    "mode", "mse_loss", "mse_loss_backward", "mul.Tensor",
    "multinomial", "mv", "nan_to_num", "nan_to_num_",
    "nanmedian", "nanmedian.dim", "nansum", "nansum.out",
    "native_batch_norm", "native_batch_norm_backward", "native_dropout", "native_dropout_backward",
    "native_group_norm", "native_group_norm_backward", "native_layer_norm", "ne.Scalar",
    "ne.Tensor", "ne_.Scalar", "ne_.Tensor", "neg",
    "neg_", "new_ones", "nextafter", "nextafter.out",
    "nextafter_", "nll_loss2d_backward", "nll_loss2d_forward", "nll_loss_backward",
    "nll_loss_forward", "nonzero", "nonzero_static", "norm.Scalar",
    "norm.ScalarOpt_dim", "ones", "ones_like", "ormqr",
    "pixel_unshuffle.out", "polar", "polygamma", "polygamma.out",
    "polygamma_", "pow.Scalar", "pow.Tensor_Scalar", "pow.Tensor_Tensor",
    "pow_.Scalar", "pow_.Tensor", "prod", "prod.dim_int",
    "rad2deg", "rad2deg_", "rand", "rand_like",
    "randint", "randint_like", "randn", "randn_like",
    "randperm", "range", "reciprocal", "reciprocal_",
    "reflection_pad1d", "reflection_pad1d.out", "reflection_pad1d_backward", "reflection_pad2d",
    "reflection_pad2d.out", "reflection_pad2d_backward", "reflection_pad3d", "reflection_pad3d.out",
    "reflection_pad3d_backward", "relu", "relu_", "remainder.Scalar",
    "remainder.Scalar_Tensor", "remainder.Tensor", "remainder_.Scalar", "remainder_.Tensor",
    "renorm", "renorm_", "repeat_interleave.Tensor", "replication_pad1d",
    "replication_pad1d.out", "replication_pad2d", "replication_pad2d.out", "replication_pad2d_backward",
    "replication_pad2d_backward.grad_input", "replication_pad3d", "replication_pad3d_backward", "roll",
    "rot90", "round", "round.out", "round_",
    "rrelu_with_noise_backward", "rsqrt", "rsqrt_", "rsub.Scalar",
    "rsub.Tensor", "scalar_tensor", "scatter.reduce", "scatter.src",
    "scatter_.reduce", "scatter_.src", "scatter_add", "scatter_add_",
    "scatter_reduce.two", "scatter_reduce.two_out", "scatter_reduce_.two", "sgn",
    "sgn.out", "sgn_", "sigmoid", "sigmoid_",
    "sigmoid_backward", "sign", "sign.out", "signbit",
    "signbit.out", "silu", "silu_", "silu_backward",
    "sin", "sin_", "sinc", "sinc_",
    "sinh", "sinh_", "slice.Tensor", "slice_backward",
    "soft_margin_loss", "soft_margin_loss_backward", "softplus", "softplus_backward",
    "softshrink", "softshrink.out", "sort", "sort.stable",
    "special_airy_ai", "special_airy_ai.out", "special_bessel_j0", "special_bessel_y0",
    "special_bessel_y1", "special_chebyshev_polynomial_u", "special_chebyshev_polynomial_u.n_scalar", "special_chebyshev_polynomial_v",
    "special_chebyshev_polynomial_w", "special_chebyshev_polynomial_w.out", "special_erfcx", "special_hermite_polynomial_h",
    "special_hermite_polynomial_h.n_scalar", "special_i0e", "special_i0e.out", "special_i1",
    "special_i1e", "special_i1e.out", "special_legendre_polynomial_p", "special_log_ndtr",
    "special_modified_bessel_k0", "special_modified_bessel_k0.out", "special_modified_bessel_k1", "special_modified_bessel_k1.out",
    "special_ndtri", "special_scaled_modified_bessel_k1", "special_scaled_modified_bessel_k1.out", "special_shifted_chebyshev_polynomial_t",
    "special_shifted_chebyshev_polynomial_u", "special_shifted_chebyshev_polynomial_v", "special_shifted_chebyshev_polynomial_w", "special_xlog1py",
    "sqrt", "sqrt_", "std.correction", "sub.Tensor",
    "sub_.Tensor", "sum", "sum.IntList_out", "sum.dim_IntList",
    "sum.out", "take", "take.out", "tan",
    "tan_", "tanh", "tanh_", "tanh_backward",
    "threshold", "threshold_", "threshold_backward", "topk",
    "trace", "transpose.int", "tril", "tril.out",
    "tril_", "triu", "triu_", "trunc",
    "trunc_", "unfold_backward", "uniform_", "unique_consecutive",
    "unique_dim", "upsample_bicubic2d", "upsample_linear1d_backward", "var.correction",
    "var_mean.correction", "vdot", "view_as_complex", "where.self",
    "where.self_out", "xlogy.OutScalar_Other", "xlogy.OutScalar_Self", "xlogy.OutTensor",
    "xlogy.Scalar_Other", "xlogy.Scalar_Self", "xlogy.Tensor", "xlogy_.Scalar_Other",
    "xlogy_.Tensor", "zero_", "zeros", "zeros_like",
})  # fmt: skip

# Ops flag_gems' C++ runtime (liboperators.so) implements (Backend::kFlagGemsCpp).
# A strict subset of FLAGGEMS_PYTHON_OPS: the C++ path only changes the entry
# point, so withholding it never costs operator coverage.
FLAGGEMS_CPP_OPS = frozenset({
    "_softmax", "_softmax_backward_data", "addmm", "addmm.out", "argmax",
    "bmm", "bmm.out", "embedding", "max", "max.dim", "mm", "nonzero", "sort",
    "sort.stable", "sum", "sum.dim_IntList", "topk", "zeros",
})  # fmt: skip

# Ops with a TileOps shim in torch_fl/tileops/generated/shims (Backend::kTileOps),
# reached through CallPythonOp_Generic.
#
# @generated by scripts/codegen/codegen_tileops.py -- DO NOT EDIT.
# The generator rewrites this block in place, so a hand edit is lost on the next
# run. Change the TileOPs manifest instead, then regenerate.
TILEOPS_OPS = frozenset({
    "_log_softmax", "_softmax", "abs", "add.Tensor", "all.dim", "amax", "amin",
    "any.dim", "bitwise_and.Tensor", "bitwise_not", "bitwise_or.Tensor",
    "bitwise_xor.Tensor", "ceil", "cos", "count_nonzero.dim_IntList",
    "eq.Tensor", "erf", "exp", "expm1", "floor", "floor_divide", "ge.Tensor",
    "gt.Tensor", "hardsigmoid", "hardswish", "isinf", "isnan", "le.Tensor",
    "log", "log1p", "logical_and", "logical_not", "logical_or", "logsumexp",
    "lt.Tensor", "maximum", "mean.dim", "minimum", "mish", "mul.Tensor",
    "ne.Tensor", "neg", "pow.Tensor_Tensor", "prod.dim_int", "reciprocal",
    "relu", "remainder.Tensor", "rsqrt", "sigmoid", "sign", "silu", "sin",
    "sqrt", "std.correction", "sub.Tensor", "sum.dim_IntList", "tanh", "trunc",
    "var.correction", "var_mean.correction",
})  # fmt: skip
