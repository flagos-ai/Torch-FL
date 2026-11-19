# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Measured platform rules for native-host wheel construction.

Precision exceptions and published source selections share one validated
dependency cohort. Builders embed their resolved rules in wheel manifests;
runtime activation never imports this module or reads the source checkout.
"""

from copy import deepcopy


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
    "retirement": "Remove each exception after a published FlagGems wheel passes the paired HCU logit gate and repeated eager performance checks.",
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
            "op": "_softmax",
            "family": "structured",
            "dtypes": ["Float"],
            "reason": "FP32 attention rounding propagates across BF16 conversion boundaries.",
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


def configured_dtype_routes(vendor):
    """Render measured input-dtype clauses for the vendor conf generator."""
    return {
        row["op"]: ",".join(row["dtypes"]) + ":" + vendor
        for row in precision_policy(vendor)["operators"]
    }


def render_config_routes(text, vendor):
    rules = configured_dtype_routes(vendor)
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
