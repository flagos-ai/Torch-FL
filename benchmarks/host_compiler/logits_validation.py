# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Versioned logit acceptance, independent of timing and model execution."""

import math
import hashlib
import os

import torch


BF16_VECTOR_LIMITS = (
    ("phase_relative_l2", 0.02),
    ("token_relative_l2", 0.10),
    ("token_p99_scaled_abs", 0.25),
    ("token_max_scaled_abs", 0.50),
)
BF16_SCALE_FLOOR = 1e-6


class LogitsValidationFailed(RuntimeError):
    """A completed benchmark wrote its report but failed accuracy acceptance."""


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def math_settings():
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "NVIDIA_TF32_OVERRIDE",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
    }


def validate_tolerances(atol, rtol):
    if not all(math.isfinite(value) and value >= 0 for value in (atol, rtol)):
        raise ValueError("logit tolerances must be finite and nonnegative")


def tensor_accuracy(actual, expected, *, atol=1e-3, rtol=1e-3):
    """Require every finite logit to satisfy abs(a-b) <= atol + rtol*abs(b)."""
    validate_tolerances(atol, rtol)
    if actual.shape != expected.shape or actual.numel() == 0:
        raise ValueError("logit shapes must match exactly and be nonempty")
    if not actual.is_floating_point() or not expected.is_floating_point():
        raise ValueError("logits must have a real floating-point dtype")
    if actual.dtype != expected.dtype:
        raise ValueError("logit dtype must match the vendor reference")
    # Higher precision in the checker avoids BF16 arithmetic masking a failure.
    # These detached CPU tensors are evaluated outside all timed model forwards.
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    finite = torch.isfinite(a) & torch.isfinite(b)
    close = torch.isclose(a, b, atol=atol, rtol=rtol, equal_nan=False) & finite
    nonfinite = int((~finite).sum())
    violations = int((~close).sum())
    difference = (a - b).abs()
    limit = atol + rtol * b.abs()
    scaled = torch.where(limit > 0, difference / limit, 0.0)
    scaled = torch.where((limit == 0) & (difference != 0), math.inf, scaled)
    maximum = float(scaled.max())
    return {
        "atol": atol,
        "rtol": rtol,
        "passed": violations == 0,
        "elements": a.numel(),
        "violations": violations,
        "nonfinite": nonfinite,
        "max_abs": None if nonfinite else float(difference.max()),
        "max_scaled_error": maximum if math.isfinite(maximum) else None,
    }


def bf16_vector_policy():
    """Return the fixed raw-logit policy; no per-operator tolerance overrides."""
    return {
        "name": "bf16-v1",
        "limits": dict(BF16_VECTOR_LIMITS),
        "scale": "max(reference token RMS, 1e-6)",
        "scale_floor": BF16_SCALE_FLOOR,
        "vocabulary_axis": -1,
        "center_logits": False,
        "checker_dtype": "float64",
        "scope": "all prefill positions and every teacher-forced decode step",
    }


def bf16_vector_accuracy(actual, expected):
    """Check every raw BF16 logit vector, including its tail and worst element."""
    if actual.shape != expected.shape or actual.numel() == 0:
        raise ValueError("logit shapes must match exactly and be nonempty")
    if actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
        raise ValueError("bf16-v1 requires the vendor's bfloat16 logit dtype")
    if actual.ndim < 2:
        raise ValueError("logits require token and vocabulary dimensions")
    vocabulary = actual.shape[-1]
    a = actual.detach().cpu().double().reshape(-1, vocabulary)
    b = expected.detach().cpu().double().reshape(-1, vocabulary)
    limits = dict(BF16_VECTOR_LIMITS)
    finite = torch.isfinite(a) & torch.isfinite(b)
    valid_rows = finite.all(-1)
    positions = a.shape[0]
    rows = []
    squared_error = squared_reference = 0.0
    for index in range(positions):
        if not bool(valid_rows[index]):
            rows.append(
                {
                    "position": index,
                    "passed": False,
                    "failed_checks": ["nonfinite"],
                    "relative_l2": None,
                    "p99_scaled_abs": None,
                    "max_scaled_abs": None,
                    "max_abs": None,
                }
            )
            continue
        error = (a[index] - b[index]).abs()
        reference_square = float(b[index].square().sum())
        error_square = float(error.square().sum())
        scale = max(math.sqrt(reference_square / vocabulary), BF16_SCALE_FLOOR)
        metrics = {
            "relative_l2": math.sqrt(error_square / vocabulary) / scale,
            "p99_scaled_abs": float(torch.quantile(error, 0.99)) / scale,
            "max_scaled_abs": float(error.max()) / scale,
            "max_abs": float(error.max()),
        }
        failed = [
            name
            for name in ("relative_l2", "p99_scaled_abs", "max_scaled_abs")
            if metrics[name] > limits["token_" + name]
        ]
        rows.append(
            {
                "position": index,
                "passed": not failed,
                "failed_checks": failed,
                **metrics,
            }
        )
        squared_error += error_square
        squared_reference += reference_square
    failed = [row["position"] for row in rows if not row["passed"]]
    nonfinite = int((~finite).sum())
    return {
        "policy": "bf16-v1",
        "passed": not failed,
        "positions": positions,
        "passed_positions": positions - len(failed),
        "failed_positions": failed,
        "elements": a.numel(),
        "nonfinite": nonfinite,
        "squared_error": None if nonfinite else squared_error,
        "squared_reference": None if nonfinite else squared_reference,
        "rows": rows,
    }


def summarize_bf16_accuracy(comparisons, expected_count):
    """Require per-token limits and separate prefill/decode aggregate limits."""
    if len(comparisons) != expected_count or expected_count < 2:
        raise ValueError("logit comparison count must include prefill and decode")
    if any(row.get("policy") != "bf16-v1" for row in comparisons):
        raise ValueError("mixed or missing logit acceptance policies")
    limits = dict(BF16_VECTOR_LIMITS)
    phases = {}
    for name, groups in (
        ("prefill", comparisons[:1]),
        ("decode", comparisons[1:]),
    ):
        finite = all(row["nonfinite"] == 0 for row in groups)
        relative = None
        if finite:
            elements = sum(row["elements"] for row in groups)
            denominator = max(
                sum(row["squared_reference"] for row in groups),
                elements * BF16_SCALE_FLOOR**2,
            )
            relative = math.sqrt(
                sum(row["squared_error"] for row in groups) / denominator
            )
        phases[name] = {
            "passed": finite
            and all(row["passed"] for row in groups)
            and relative <= limits["phase_relative_l2"],
            "relative_l2": relative,
            "positions": sum(row["positions"] for row in groups),
            "passed_positions": sum(row["passed_positions"] for row in groups),
        }
    return {
        "policy": bf16_vector_policy(),
        "comparisons": len(comparisons),
        "passed_comparisons": sum(row["passed"] for row in comparisons),
        "failed_steps": [i for i, row in enumerate(comparisons) if not row["passed"]],
        "failed_phases": [name for name, row in phases.items() if not row["passed"]],
        "positions": sum(row["positions"] for row in comparisons),
        "passed_positions": sum(row["passed_positions"] for row in comparisons),
        "phases": phases,
        "passed": all(row["passed"] for row in phases.values()),
    }


def validate_reference(reference, steps, *, contract=None, prompt_ids=None):
    """Reject incomplete references and, for acceptance, incompatible workloads."""
    if len(reference.get("inputs", ())) < steps:
        raise ValueError("reference has fewer teacher-forced inputs than requested")
    if len(reference.get("logits", ())) < steps + 1:
        raise ValueError("reference must cover prefill and every requested decode step")
    if contract is not None:
        if "prefill_logits" not in reference:
            raise ValueError("reference must include every prefill position's logits")
        producer = reference.get("producer", {})
        if producer.get("backend") != "vendor":
            raise ValueError("acceptance requires a native vendor Torch reference")
        if reference.get("contract") != contract:
            raise ValueError(
                "reference contract differs or is missing; regenerate the vendor "
                "reference with the same checkpoint, dtype and math settings"
            )
        if producer.get("math_settings_after") != contract["math_settings"]:
            raise ValueError("reference math settings changed during execution")
        stored = reference.get("prompt_ids")
        if stored is None or not torch.equal(stored, prompt_ids.cpu()):
            raise ValueError("reference prompt token IDs differ or are missing")


def summarize_accuracy(comparisons, expected_count, *, atol, rtol):
    if len(comparisons) != expected_count or expected_count < 1:
        raise ValueError(
            "logit comparison count does not cover the requested traversal"
        )
    failed = [index for index, row in enumerate(comparisons) if not row["passed"]]
    return {
        "atol": atol,
        "rtol": rtol,
        "comparisons": len(comparisons),
        "passed_comparisons": len(comparisons) - len(failed),
        "failed_steps": failed,
        "violations": sum(row["violations"] for row in comparisons),
        "passed": not failed,
    }
