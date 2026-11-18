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
    }
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


_COHORTS = {
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
        "retirement": "Remove an exception after a published FlagGems wheel passes the paired vendor logit gate and repeated eager performance checks.",
        "operators": [
            {
                "op": "_softmax",
                "family": "structured",
                "dtypes": ["Float"],
                "reason": "FP32 softmax differences cross a BF16 rounding boundary and propagate into failing logits.",
            },
            {
                "op": "bmm",
                "family": "blas",
                "rank": 3,
                "dtypes": ["BFloat16"],
                "reason": "An isolated decode attention product differs before error propagation.",
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
        ],
    },
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
        "requires_native_vendor": vendor in {"nvidia", "metax"},
        "entries": policy["sources"],
    }


def configured_dtype_routes(vendor, *, policy=None):
    """Render measured input-dtype clauses for the vendor conf generator."""
    return {
        row["op"]: ",".join(row["dtypes"]) + ":" + vendor
        for row in (precision_policy(vendor) if policy is None else policy)["operators"]
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
                    "# Example: mm = flaggems; BFloat16:nvidia. Unmatched or unavailable clauses use the base.",
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
