# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Measured platform rules for native-host wheel construction.

Precision exceptions and published source selections share one validated
dependency cohort. Builders embed their resolved rules in wheel manifests;
runtime activation never imports this module or reads the source checkout.
"""

from copy import deepcopy


NATIVE_BMM_ADAPTATIONS = {
    "metax": {
        "flag_gems": "5.4.0",
        "vendor_torch": "2.10.0+metax3.8.1.0",
        "operators": ("bmm", "bmm.out"),
        "dtype": "Float",
        "contract": "Matching 3-D batches with contracted extent one.",
        "implementation": "Original published pointwise kernels; TF32 nearest-even operands when the caller allows TF32.",
        "reason": "Published bmm fixes allow_tf32=False; actual Qwen RoPE products differ from the manufacturer's TF32 results by up to 0.010868073. GPU nearest-even operand rounding reproduces all captured products bitwise.",
        "retirement": "Remove after the published bmm supports K=1 and honors the same TF32 context with paired model precision/performance evidence.",
    }
}


NATIVE_COPY_ADAPTATIONS = {
    "metax": {
        "flag_gems": "5.4.0",
        "flagtree": "0.7.0.post2+metax3.6",
        "source_dtype": "Bool",
        "destination_dtype": "BFloat16",
        "intermediate_dtype": "Float",
        "implementation": "Two original published GPU copy kernels; Boolean values are exact through FP32.",
        "reason": "The original direct copy in the eager causal-mask multiply fails MACA lowering: llvm.uitofp produces i16 instead of a floating-point result.",
        "retirement": "Remove after the published compiler passes the direct Bool-to-BF16 GPU copy and paired full-model checks.",
    }
}


NATIVE_SDK_PRECISION = {
    "metax": {
        "schema_version": 1,
        "vendor": "metax",
        "flag_gems": "5.4.0",
        "flagtree": "0.7.0.post2+metax3.6",
        "measured_sdk_version": "3.8.1.3",
        "vendor_torch": "2.10.0+metax3.8.1.0",
        "issue": "https://github.com/flagos-ai/Torch-FL/issues/832",
        "evidence": "All 27 actual first-layer BF16 matrix products from prefill and two teacher-forced decode steps match the installed SDK BLAS bitwise; original published FlagGems products differ at identical inputs.",
        "retirement": "Remove each exception after a published FlagGems wheel passes the paired full-model logits gate and repeated eager performance checks.",
        "operators": [
            {"op": "mm", "rank": 2, "dtypes": ["BFloat16"]},
            {"op": "bmm", "rank": 3, "dtypes": ["BFloat16"]},
        ],
    },
    "thead": {
        "schema_version": 1,
        "vendor": "thead",
        "flag_gems": "5.4.0",
        "flagtree": "0.7.0+ppu3.6",
        "measured_sdk_version": "2.1.0-a5f865",
        "vendor_torch": "2.10.0",
        "evidence": "Direct SDK BF16 output with FP32 accumulation matches stock Torch bitwise on captured eager Qwen3 matrix products and forward/backward batched products, including dense and transposed weights.",
        "retirement": "Remove each exception once a published FlagGems wheel matches paired eager training and inference precision.",
        "operators": [
            {"op": "mm", "rank": 2, "dtypes": ["BFloat16"]},
            {"op": "bmm", "rank": 3, "dtypes": ["BFloat16"]},
        ],
    },
}


NATIVE_REDUCTION_ADAPTATIONS = {
    "metax": {
        "flag_gems": "5.4.0",
        "flagtree": "0.7.0.post2+metax3.6",
        "vendor_torch": "2.10.0+metax3.8.1.0",
        "operator": "mean.dim",
        "dtype": "Float",
        "extents": [128, 1024],
        "contract": "Contiguous FP32 input, one last-axis reduction, FP32 output, nonempty rows.",
        "max_threads": 512,
        "warp_size": 64,
        "split_warp_extent": 256,
        "split_warp_width": 16,
        "vectorize_above": 128,
        "vector_width": 4,
        "implementation": "GPU layout permutation and launch metadata for the original published mean_dim_kernel_inner; one element per lane preserves balanced reduction stages.",
        "evidence": "The original reduction order differs at actual Qwen normalization inputs. The normalized original GPU kernel matches all 48 paired random shape/seed cases bitwise, including one-row decode and prefill geometry.",
        "retirement": "Remove after a published FlagGems mean passes paired full-model precision and repeated eager performance checks with the manufacturer reduction order.",
    }
}


_FOREACH_OPTIMIZER_OPS = (
    "_foreach_add_.Scalar",
    "_foreach_addcdiv_.ScalarList",
    "_foreach_addcmul_.Scalar",
    "_foreach_div_.ScalarList",
    "_foreach_lerp_.Scalar",
    "_foreach_mul_.Scalar",
    "_foreach_sqrt",
)

_PUBLISHED_BF16_LERP = {
    "id": "published_bf16_lerp",
    "op": "aten::lerp_.Scalar",
    "module": "flag_gems.runtime.backend._kunlunxin.ops.lerp",
    "function": "lerp_scalar_",
    "sha256": "5ef46de3e8437db0a0111cb63eee6bf2ba2e4d6195aaa39aecafe19658ddc47a",
    "dtypes": ["BFloat16"],
    "same_tensor_metadata": True,
    "min_tensors": 2,
    "shared_pointwise": True,
    "reason": "The bundled generic BF16 lerp rounds the end-minus-input term before FP32 multiplication. The published XPU scalar math converts both operands to FP32 before subtraction.",
}


_COHORTS = {
    "kunlunxin": {
        "schema_version": 1,
        "flag_gems": "5.4.0",
        "operators": [],
        "sources": [
            {
                **_PUBLISHED_BF16_LERP,
                "id": "kunlunxin_bf16_lerp_round_even",
                "adaptation": "kunlunxin_bf16_lerp_round_even",
                "shared_pointwise": False,
                "reason": "The published XPU conversion changes BF16 first-moment values at the rounding boundary. Explicit FP32 fused lerp followed by one BF16 round reproduces the vendor without intermediate tensor conversions.",
            },
        ],
    },
    "tsingmicro": {
        "schema_version": 1,
        "flag_gems": "5.4.0",
        "operators": [],
        "sources": [
            {
                "id": "published_tsingmicro_bool_fill",
                "op": "aten::fill_.Scalar",
                "module": "flag_gems.runtime.backend._tsingmicro.ops.fill",
                "function": "fill_scalar_",
                "sha256": "b15dd66bff906ac8157ae0fb849faa1e6447e369593567069f3796945715cee6",
                "dtypes": ["Bool"],
                "reason": "The typed Bool host resolves the complex-dtype check at build time and uses the published fill kernel with fixed launch metadata.",
            },
            {
                "id": "tsingmicro_embedding_bf16_input_order",
                "op": "aten::embedding_dense_backward",
                "module": "flag_gems.ops.embedding_dense_backward",
                "function": "embedding_dense_backward",
                "sha256": "7f28f60d77ef087943f43da45ccf8ad77f411d4b1ff7cbc61f2f91c578da98ef",
                "dtypes": ["BFloat16"],
                "adaptation": "tsingmicro_embedding_bf16_input_order",
                "reason": "The measured TX compiler rejects the published masked FP32 scatter, and stock TX BF16 gradients round after each input-order update to repeated rows.",
            },
            {
                "id": "tsingmicro_argmax_first_nan",
                "op": "aten::argmax",
                "module": "flag_gems.runtime.backend._tsingmicro.ops.argmax",
                "function": "argmax",
                "sha256": "4a9697a4bed8d70d4a754d8ea705adf6e4bdc8be6654dd7205aa47e06ae835a2",
                "dtypes": ["Float"],
                "adaptation": "tsingmicro_argmax_first_nan",
                "reason": "The published cross-tile FP32 reduction ignores NaN. Stock greedy generation chooses the first NaN while preserving the first index for other ties.",
            },
            {
                "id": "published_tsingmicro_scalar_divide_out",
                "op": "aten::div.out",
                "module": "flag_gems.ops.div",
                "function": "true_divide_out",
                "sha256": "e8e80ec353f76619fb10881d2dfa5c0b77a131adc64c0207da54ffe2cd17f6d8",
                "dtypes": ["BFloat16"],
                "same_tensor_metadata": True,
                "allow_wrapped_numbers": True,
                "min_tensors": 2,
                "reason": "The published scalar out path retains the FP32 divisor for dense BF16 AdamW updates.",
            },
            {
                "id": "tsingmicro_addcdiv_bf16_scale_first",
                "op": "aten::addcdiv.out",
                "module": "flag_gems.ops.addcdiv",
                "function": "addcdiv_out",
                "sha256": "05193b183a137856c9998767f3b1b2d31928c8bbd7164001adf4adf0ed8dea26",
                "dtypes": ["BFloat16"],
                "adaptation": "tsingmicro_addcdiv_bf16_scale_first",
                "reason": "Round the eagerly scaled FP32 quotient before adding its BF16 input, as in the stock TsingMicro eager optimizer.",
            },
            {
                "id": "tsingmicro_float_scalar_divide_masked",
                "op": "aten::div.Scalar",
                "module": "flag_gems.ops.div",
                "function": "true_divide",
                "sha256": "e8e80ec353f76619fb10881d2dfa5c0b77a131adc64c0207da54ffe2cd17f6d8",
                "dtypes": ["Float"],
                "adaptation": "tsingmicro_float_scalar_divide_masked",
                "reason": "Masked loads preserve expanded FP32 gradients when the TsingMicro block-pointer path misreads a zero stride.",
            },
            {
                "id": "tsingmicro_sin_reduced_argument",
                "op": "aten::sin",
                "module": "flag_gems.ops.sin",
                "function": "sin",
                "sha256": "2a883b75017c31ed5f2a3f1b74a7de8db9d64d4caa96e03c3cebdffdca9f1cc6",
                "dtypes": ["Float"],
                "adaptation": "tsingmicro_sin_reduced_argument",
                "reason": "Accurate phase reduction near multiples of pi preserves the stock BF16 rotary values.",
            },
            {
                "id": "tsingmicro_silu_stable_positive",
                "op": "aten::silu",
                "module": "flag_gems.runtime.backend._tsingmicro.ops.silu",
                "function": "silu",
                "sha256": "8450410ff97537e4a9c2dd931df242e24d01e4c69f5da41f8b81181c23651f38",
                "dtypes": ["BFloat16"],
                "adaptation": "tsingmicro_silu_stable_positive",
                "reason": "The stable positive SiLU expression preserves the stock rounding at BF16 midpoints and uses the shared dense pointwise host.",
            },
            {
                "id": "tsingmicro_silu_backward_bf16_stages",
                "op": "aten::silu_backward",
                "module": "flag_gems.runtime.backend._tsingmicro.ops.silu",
                "function": "silu_backward",
                "sha256": "8450410ff97537e4a9c2dd931df242e24d01e4c69f5da41f8b81181c23651f38",
                "dtypes": ["BFloat16"],
                "adaptation": "tsingmicro_silu_backward_bf16_stages",
                "reason": "The functional vendor SiLU backward rounds each derivative operation to BF16 before multiplying the incoming gradient.",
            },
            {
                "id": "tsingmicro_sum_ordered",
                "op": "aten::sum.dim_IntList",
                "module": "flag_gems.ops.sum",
                "function": "sum_dim",
                "sha256": "5259b9f60445ef4a99e85387bc1e4c7a2264681170c695c6fc9b94378fa6348b",
                "dtypes": ["Float", "BFloat16"],
                "adaptation": "tsingmicro_sum_ordered",
                "reason": "Use the stock FP32 accumulation order for the measured final-axis Float and leading-axis BF16 sums.",
            },
            {
                "id": "tsingmicro_nll_mean_forward_integer_offsets",
                "op": "aten::nll_loss_forward",
                "module": "flag_gems.ops.nllloss",
                "function": "nll_loss_forward",
                "sha256": "0c59ab8a7a0bdcdb4352966118a240acba420d545ba3f5fb3810e7f28ecb69f9",
                "dtypes": ["Float"],
                "adaptation": "tsingmicro_nll_mean_forward_integer_offsets",
                "reason": "Scalar integer gathers avoid rounding binary32 offsets above 2**24 for contiguous Qwen3 logits.",
            },
            {
                "id": "tsingmicro_nll_mean_backward_integer_offsets",
                "op": "aten::nll_loss_backward",
                "module": "flag_gems.ops.nllloss",
                "function": "nll_loss_backward",
                "sha256": "0c59ab8a7a0bdcdb4352966118a240acba420d545ba3f5fb3810e7f28ecb69f9",
                "dtypes": ["Float"],
                "adaptation": "tsingmicro_nll_mean_backward_integer_offsets",
                "reason": "Scalar integer scatters preserve the exact class coordinate and leave ignored targets at zero.",
            },
            {
                "id": "tsingmicro_long_add_scalar_lanes",
                "op": "aten::add.Tensor",
                "module": "flag_gems.runtime.backend._tsingmicro.ops.add",
                "function": "add",
                "sha256": "e61a60e0c4edd4c8964d9d0dc05ab38e5a703bb99e479208f12dbf7e3575f6de",
                "dtypes": ["Long"],
                "adaptation": "tsingmicro_long_add_scalar_lanes",
                "reason": "TXDNN can read stale lanes of Long metadata tensors. The scalar FlagGems lowering preserves full-width integer data and does not share the TXDNN arithmetic buffers.",
            },
            {
                "id": "tsingmicro_long_sub_scalar_lanes",
                "op": "aten::sub.Tensor",
                "module": "flag_gems.runtime.backend._tsingmicro.ops.sub",
                "function": "sub",
                "sha256": "1407e41c265286f03a07a7784ebd5cdef298eacb5816506816ae41995baca07d",
                "dtypes": ["Long"],
                "adaptation": "tsingmicro_long_sub_scalar_lanes",
                "reason": "TXDNN can read stale lanes of Long metadata tensors. The scalar FlagGems lowering preserves full-width integer data and does not share the TXDNN arithmetic buffers.",
            },
        ],
    },
    "mthreads": {
        "schema_version": 1,
        "flag_gems": "5.4.0",
        "operators": [],
        "sources": [_PUBLISHED_BF16_LERP],
    },
    "thead": {
        "schema_version": 1,
        "flag_gems": "5.4.0",
        "vendor_torch": "2.10.0",
        "torch_git_version": "e42faac0fd33a8652c8325146b75675454b3daa3",
        "cpu_torch_git_version": "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0",
        "cuda": "13.0",
        "libraries": {
            "libtorch_cuda.so": "77a5f83115c723f00f89a79cd92e8aa8b1d3fa6a8de3cb5249d59e40895cc141",
            "libc10_cuda.so": "ff95bb742287ebee5047df464a5633e3ea9c236a474485d80d20d83193006b30",
        },
        "host_headers": {
            "c10/cuda/impl/cuda_cmake_macros.h": "6f818f1c621e74271494499dc885094ae7aac7a7316209423c00861b8b410f44",
        },
        "retirement": "Remove each exception after a published FlagGems wheel passes the paired eager training and inference gates.",
        "operators": [
            {
                "op": "mul.Tensor",
                "family": "binary_pointwise",
                "stub": "mul_stub",
                "dtypes": ["BFloat16", "Float"],
                "guarded": True,
                "reason": "The ordinary functional multiplication reaches the out adapter, which launches a separate copy of the computed result. The published vendor kernel writes directly into the ATen output layout.",
            },
            {
                "op": "cat",
                "family": "concatenate",
                "dtypes": ["BFloat16", "Float"],
                "guarded": True,
                "reason": "The published vendor concatenation handles the strided rotary inputs and the eager cache append layouts with bitwise agreement.",
            },
            *(
                {
                    "op": name,
                    "family": "foreach",
                    "dtypes": ["BFloat16"],
                    "guarded": True,
                    "reason": "The measured AdamW batch uses the published vendor list operation with the same sampled BF16 gradients and updates.",
                }
                for name in _FOREACH_OPTIMIZER_OPS
            ),
        ],
        "sources": [_PUBLISHED_BF16_LERP],
    },
    "ascend": {
        "schema_version": 1,
        "flag_gems": "5.4.0",
        "vendor_torch": "2.10.0+cpu",
        "torch_git_version": "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0",
        "issue": "https://github.com/flagos-ai/Torch-FL/issues/833",
        "retirement": "Remove the exception after a published FlagGems wheel passes the paired vendor BF16 logit gate and repeated eager performance checks.",
        "operators": [
            {
                "op": "_softmax",
                "family": "aclnn",
                "dtypes": ["Float"],
                "reason": "Published FP32 softmax fails the fixed BF16 vector-logit limits against torch-npu 2.10. Other FlagGems operators retain their routes.",
            },
        ],
        "sources": [
            {
                "id": "published_concat",
                "op": "aten::cat",
                "module": "flag_gems.runtime.backend._tsingmicro.ops.cat",
                "function": "cat",
                "sha256": "a503aae481a31e6f428180722b7e11693fb8fac03ec516013ea88fd724f03534",
                "dtypes": ["Float", "BFloat16"],
                "same_tensor_metadata": True,
                "min_tensors": 2,
                "unit_stride_axis": True,
                "reason": "The published pointwise concat implementation preserves homogeneous input values and avoids the Ascend strided-copy device bottleneck.",
            },
        ],
    },
    "metax": {
        "schema_version": 1,
        "flag_gems": "5.4.0",
        "vendor_torch": "2.10.0+metax3.8.1.0",
        "torch_git_version": "f0eb8bf5e72d15327617097aa264c899b3581986",
        "cpu_torch_git_version": "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0",
        "issue": "https://github.com/flagos-ai/Torch-FL/issues/832",
        "retirement": "Remove each exception after a published FlagGems wheel passes the paired full-model logits gate and repeated eager performance checks.",
        "host_headers": {
            "ATen/native/cuda/maca_kernels/Context.h": "97e838330b718214d3c6778e0b54feef2ff55d9c5d60d4471fa168636a2ec237",
            "ATen/native/cuda/maca_kernels/Context_.h": "9c09c849ade51a96fd3f6d94eef7394e0ce8918ac6b7187cd52b8e63430974bf",
        },
        "operators": [
            {
                "op": "silu",
                "family": "pointwise_stub",
                "stub": "silu_stub",
                "dtypes": ["BFloat16"],
                "reason": "The first full-model divergence is BF16 SiLU(5.9375): published FlagGems returns 5.90625, while the manufacturer returns 5.9375. The original vendor stub matches all 65,536 BF16 bit patterns with the official CPU core retained.",
            },
            {
                "op": "_softmax",
                "family": "structured",
                "dtypes": ["Float"],
                "half_to_float": False,
                "reason": "Published softmax differs at 335 of 476 actual model inputs, crossing BF16 rounding boundaries in eight cases and first changing decode-5 attention weights. The original structured vendor implementation matches all 476 FP32 and BF16 comparisons bitwise with the official CPU core retained.",
            },
            {
                "op": "mean.dim",
                "family": "reduction",
                "stub": "mean_stub",
                "dtypes": ["Float"],
                "output_dtype": "Float",
                "reason": "The published mean has a different reduction order. Its precise original-kernel layout normalization adds 226 profiled mean calls and 25.024 ms across two model forwards; the current full traversal reaches only 50.3% of manufacturer throughput. Reuse the original vendor reduction to retain the measured order without the layout-copy composition.",
            },
        ],
        "sources": [
            {
                "id": "published_mul",
                "op": "aten::mul.Tensor",
                "module": "flag_gems.runtime.backend._sunrise.ops.mul",
                "function": "mul",
                "allow_wrapped_numbers": True,
                "sha256": "35a2059ab4f4f80f0e5bd7d09c190e3a5008811fa9cff4a5f6621e4851d0190b",
                "dtypes": ["Float", "BFloat16"],
                "reason": "The current native profile spends 18.0 ms of host self time in 794 multiply calls across two forwards. Validate the unchanged published pointwise source already admitted in the NVIDIA cohort against this measured MetaX precision/performance baseline.",
            }
        ],
    },
    "nvidia": {
        "schema_version": 1,
        "flag_gems": "5.4.0",
        "vendor_torch": "2.10.0+cu130",
        "torch_git_version": "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0",
        "issue": "https://github.com/flagos-ai/Torch-FL/issues/840",
        "retirement": "Remove each exception after a published FlagGems wheel passes its paired vendor precision and repeated eager performance checks.",
        "operators": [
            {
                "op": "_log_softmax",
                "family": "structured",
                "dtypes": ["Float"],
                "reason": "At identical FP32 loss inputs the published reduction changes log probabilities by up to 3.82e-6. The vendor order is required to reproduce the paired training trajectory.",
            },
            {
                "op": "_log_softmax_backward_data",
                "family": "structured",
                "dtypes": ["Float"],
                "reason": "Use the same vendor gradient as the FP32 out variant and eliminate the eager functional-to-out redispatch.",
            },
            {
                "op": "_log_softmax_backward_data.out",
                "family": "softmax_backward_out",
                "dtypes": ["Float"],
                "reason": "The published FP32 cross-entropy gradient differs from the vendor reduction and its first call enqueues from Python. The structured CUDA entry preserves the matching forward's gradient.",
            },
            {
                "op": "_softmax",
                "family": "structured",
                "dtypes": ["Float"],
                "reason": "FP32 softmax differences cross a BF16 rounding boundary and propagate into failing logits.",
            },
            {
                "op": "_softmax_backward_data",
                "family": "structured",
                "dtypes": ["Float"],
                "reason": "Use the same vendor gradient as the FP32 out variant and eliminate the eager functional-to-out redispatch.",
            },
            {
                "op": "_softmax_backward_data.out",
                "family": "softmax_backward_out",
                "dtypes": ["Float"],
                "reason": "The structured CUDA backward keeps the vendor order for FP32 attention probabilities and avoids a Python enqueue on the first training step.",
            },
            {
                "op": "bmm",
                "family": "blas",
                "rank": 3,
                "dtypes": ["BFloat16"],
                "reason": "An isolated decode attention product differs before error propagation.",
            },
            {
                "op": "cat",
                "family": "concatenate",
                "dtypes": ["BFloat16", "Float", "Long"],
                "reason": "The same structured CUDA cat cuts eager dispatch overhead in native training while preserving FP32 rotary operands, BF16 values, and integer mask inputs.",
            },
            {
                "op": "mean.dim",
                "family": "reduction",
                "stub": "mean_stub",
                "empty": "nan",
                "dtypes": ["Float"],
                "reason": "Reduction order changes BF16 rounding in normalization.",
            },
            {
                "op": "mm",
                "family": "blas",
                "rank": 2,
                "dtypes": ["BFloat16"],
                "reason": "The original matrix reduction differs from the vendor BLAS result at identical inputs.",
            },
            {
                "op": "mul.Tensor",
                "family": "binary_pointwise",
                "stub": "mul_stub",
                "dtypes": ["BFloat16", "Float"],
                "reason": "The CUDA tensor iterator and published multiply preserve eager promotion, broadcasting and CPU scalar operands. The paired A100 training check rose from 0.972x to 1.009x with the direct multiply, concatenation and softmax gradient paths.",
            },
            {
                "op": "pow.Tensor_Scalar",
                "family": "structured",
                "dtypes": ["Float"],
                "reason": "The general power intrinsic differs from the vendor scalar-square path in normalization.",
            },
            {
                "op": "rsqrt",
                "family": "structured",
                "dtypes": ["Float"],
                "reason": "Reciprocal of square root differs from the vendor reciprocal-square-root instruction.",
            },
            {
                "op": "silu_backward",
                "family": "structured",
                "dtypes": ["BFloat16"],
                "reason": "At identical captured operands the published kernel changes 19 BF16 elements in the last block's first-step SiLU gradient. The vendor entry, together with the measured reductions, makes all sampled gradients and updates bitwise equal at steps 1 and 8.",
            },
            {
                "op": "sum.dim_IntList",
                "family": "reduction",
                "stub": "sum_stub",
                "empty": "zero",
                "dtypes": ["BFloat16", "Float"],
                "reason": "The original vendor accumulation reduces layer 0 Q-normalization's first-step gradient relative L2 error from 14.38% to 6.68% on the paired BF16 workload. All other sampled first-step gradients are within 10%.",
            },
            *(
                {
                    "op": name,
                    "family": "foreach",
                    "dtypes": ["BFloat16"],
                    "reason": "Published FlagGems lacks this batched AdamW kernel. The original CUDA list operations raise full-model training throughput from 0.653x to 0.875x on the same A100 with the default optimizer.",
                }
                for name in _FOREACH_OPTIMIZER_OPS
            ),
        ],
        "sources": [
            {
                "id": "published_mul",
                "op": "aten::mul.Tensor",
                "module": "flag_gems.runtime.backend._sunrise.ops.mul",
                "function": "mul",
                "sha256": "35a2059ab4f4f80f0e5bd7d09c190e3a5008811fa9cff4a5f6621e4851d0190b",
                "dtypes": ["Float", "BFloat16"],
                "reason": "The published pointwise implementation preserves vendor logits while reducing eager host overhead in the paired BF16 workload.",
            },
            _PUBLISHED_BF16_LERP,
        ],
    },
}


# HCU uses the published DTK implementations directly on FlagOS storage. These
# entries retain the vendor reduction/BLAS order at the measured input dtypes.
_COHORTS["hygon"] = {
    "schema_version": 1,
    "flag_gems": "5.4.0",
    "vendor_torch": "2.10.0+das.opt1.dtk2604.2606151054.g98b1a8",
    "vendor_source_version": "2.10.0",
    "torch_git_version": "98b1a89bbf5bcf9572ae9e23279e6ecee92857ed",
    "cpu_torch_git_version": "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0",
    "hip": "6.3.26113",
    "retirement": "Remove each exception after a published FlagGems wheel passes the paired HCU precision and repeated eager performance checks for the affected model path.",
    "operators": [
        {
            "op": "bmm",
            "family": "structured",
            "dtypes": ["BFloat16"],
            "reason": "The 481-token attention reduction first breaks vendor logits at decode step 458; replacing only BF16 bmm restores all 512 decode steps exactly.",
        },
        {
            "op": "mm",
            "family": "structured",
            "dtypes": ["BFloat16"],
            "reason": "Identical first-layer normalization inputs produce different BF16 projection outputs with the published FlagGems reduction.",
        },
        {
            "op": "cat",
            "family": "concatenate",
            "dtypes": ["BFloat16", "Float"],
            "reason": "The published HCU cat implements the same eager operator and matches the original values. Paired BF16 Qwen decode measured 0.914x with this kernel, 0.874x with FlagGems strided copies, and 0.842x with the published generic FlagGems cat.",
        },
        {
            "op": "_softmax",
            "family": "structured",
            "dtypes": ["Float"],
            "reason": "FP32 attention rounding propagates across BF16 conversion boundaries.",
        },
        {
            "op": "_log_softmax",
            "family": "structured",
            "dtypes": ["Float"],
            "reason": "Keep loss probabilities and gradients on the original HCU reduction order for BF16 Qwen training.",
        },
        {
            "op": "_log_softmax_backward_data",
            "family": "structured",
            "dtypes": ["Float"],
            "reason": "The paired training run uses the original loss gradient and removes the first-step Python enqueue.",
        },
        {
            "op": "_softmax_backward_data",
            "family": "structured",
            "dtypes": ["Float"],
            "reason": "The paired training run uses the original attention gradient and removes the first-step Python enqueue.",
        },
        {
            "op": "mean.dim",
            "family": "reduction",
            "stub": "mean_stub",
            "empty": "nan",
            "dtypes": ["Float"],
            "reason": "Normalization requires the published HCU reduction order and empty-reduction identity.",
        },
        {
            "op": "rsqrt",
            "family": "structured",
            "dtypes": ["Float"],
            "reason": "Normalization requires the original HCU reciprocal-square-root arithmetic.",
        },
        {
            "op": "pow.Tensor_Scalar",
            "family": "structured",
            "dtypes": ["Float"],
            "reason": "The first differing operation on identical captured operands is FP32 power in normalization backward. The original HCU power gives passing paired gradient and update checks.",
        },
        {
            "op": "silu_backward",
            "family": "structured",
            "dtypes": ["BFloat16"],
            "reason": "The measured training trajectory uses the original HCU BF16 activation gradient together with the scoped reductions.",
        },
        {
            "op": "sum.dim_IntList",
            "family": "reduction",
            "stub": "sum_stub",
            "empty": "zero",
            "dtypes": ["BFloat16", "Float"],
            "reason": "Use the original reduction order and remove 5469 first-step Python sum enqueues on the paired Qwen training workload.",
        },
        *(
            {
                "op": name,
                "family": "foreach",
                "dtypes": ["BFloat16"],
                "reason": "Published FlagGems lacks the batched AdamW kernel. The pinned HCU Torch list operations raise full-model training throughput from 0.644x to 0.816x on BW1101 with the default optimizer.",
            }
            for name in _FOREACH_OPTIMIZER_OPS
        ),
    ],
    "sources": [
        {
            "id": "published_mul",
            "op": "aten::mul.Tensor",
            "module": "flag_gems.runtime.backend._sunrise.ops.mul",
            "function": "mul",
            "allow_wrapped_numbers": True,
            "sha256": "35a2059ab4f4f80f0e5bd7d09c190e3a5008811fa9cff4a5f6621e4851d0190b",
            "dtypes": ["Float", "BFloat16"],
            "reason": "The published pointwise source admits typed host lowering instead of HCU broadcast bookkeeping; paired FP32/BF16 products are identical.",
        }
    ],
}


# CoreX uses the same published multiplication source as the HCU typed host.
# No vendor Torch library is linked, and the generic FlagGems route still owns
# inputs outside the verified FP32 and BF16 signatures.
_COHORTS["iluvatar"] = {
    "schema_version": 1,
    "flag_gems": "5.4.0",
    "operators": [],
    "sources": [
        {
            **_COHORTS["hygon"]["sources"][0],
            "reason": "CoreX's tuned multiply spends 507.7 ms across 25,408 calls in the paired eager decode profile. Use the shared published pointwise host for FP32 and BF16 products.",
        }
    ],
}


def _cohort(vendor):
    if vendor not in _COHORTS:
        raise ValueError(f"unmeasured native-host vendor: {vendor}")
    return deepcopy(_COHORTS[vendor])


def precision_policy(vendor):
    policy = _cohort(vendor)
    del policy["sources"]
    return {"vendor": vendor, **policy}


def source_policy(vendor):
    policy = _cohort(vendor)
    return {
        "schema_version": policy["schema_version"],
        "vendor": vendor,
        "flag_gems": policy["flag_gems"],
        "requires_native_vendor": vendor in {"nvidia", "hygon"},
        "entries": policy["sources"],
    }


def configured_dtype_routes(vendor, *, policy=None):
    """Render measured input-dtype clauses for the vendor conf generator."""
    return {
        row["op"]: ",".join(row["dtypes"]) + ":" + vendor
        for row in (precision_policy(vendor) if policy is None else policy)["operators"]
        if not row.get("guarded", False)
    }


def render_config_routes(text, vendor, *, policy=None):
    rules = configured_dtype_routes(vendor, policy=policy)
    lines, seen = [], set()
    header = "# Dtype clauses override the base route only when the input matches and the kernel is installed."
    add_header = header not in text
    for line in text.splitlines():
        body, comment, annotation = line.partition("#")
        name, assignment, value = body.partition("=")
        if not assignment:
            lines.append(line)
            continue
        if add_header:
            lines.extend(
                [
                    header,
                    f"# Example: mm = flaggems; BFloat16:{vendor}. Unmatched or unavailable clauses use the base.",
                ]
            )
            add_header = False
        op = name.strip()
        base = value.split(";", 1)[0].strip()
        if op in rules:
            base += "; " + rules[op]
            seen.add(op)
        suffix = "  #" + annotation if comment else ""
        lines.append(f"{op} = {base}{suffix}")
    if seen != rules.keys():
        raise ValueError(
            f"native precision operators missing from conf: {sorted(rules.keys() - seen)}"
        )
    return "\n".join(lines) + "\n"


def route_backends(value):
    """Enumerate the base and conditional backend names for conf validation."""
    parts = value.split("#", 1)[0].split(";")
    result = [parts[0].strip()]
    for clause in parts[1:]:
        dtypes, colon, backend = clause.partition(":")
        if not colon or not dtypes.strip() or not backend.strip():
            raise ValueError(f"invalid dtype route: {clause}")
        result.append(backend.strip())
    return result
