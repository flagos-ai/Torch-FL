# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Measured platform rules for native-host wheel construction.

Precision exceptions and published source selections share one validated
dependency cohort. Builders embed their resolved rules in wheel manifests;
runtime activation never imports this module or reads the source checkout.
"""

from copy import deepcopy


_COHORTS = {
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
        "requires_native_vendor": True,
        "entries": policy["sources"],
    }
