# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Elementwise logit acceptance, independent of timing and model execution."""

import math
import hashlib
import os

import torch


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
