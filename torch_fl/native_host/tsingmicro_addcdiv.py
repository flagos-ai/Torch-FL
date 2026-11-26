# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the stock FP32 operation order for the BF16 update."""


def render(published, _root):
    """Reuse the published pointwise wrapper with an accurately rounded quotient."""
    text = published.decode("utf-8")
    imported = "import triton\n"
    original = "    return x + value * (t1 / t2)\n"
    updated = (
        "    numerator = value * t1.to(tl.float32)\n"
        "    denominator = t2.to(tl.float32)\n"
        "    quotient = tl.div_rn(numerator, denominator)\n"
        "    reciprocal = tl.div_rn(1.0, denominator)\n"
        "    residual = tl.fma(-quotient, denominator, numerator)\n"
        "    corrected = tl.fma(residual, reciprocal, quotient)\n"
        "    bounded = (tl.abs(quotient) < float('inf'))\n"
        "    bounded &= (tl.abs(reciprocal) < float('inf'))\n"
        "    bounded &= (tl.abs(denominator) < float('inf'))\n"
        "    quotient = tl.where(bounded & (residual != 0.0), corrected, quotient)\n"
        "    return x.to(tl.float32) + quotient\n"
    )
    host = "def addcdiv_out(inp, tensor1, tensor2, *, value=1.0, out):\n"
    if any(text.count(marker) != 1 for marker in (imported, original, host)):
        raise ValueError("unexpected published FlagGems addcdiv expression")
    return (
        text.replace(imported, imported + "import triton.language as tl\n", 1)
        .replace(original, updated, 1)
        .replace(host, "def addcdiv_out(inp, tensor1, tensor2, *, value, out):\n", 1)
    )
