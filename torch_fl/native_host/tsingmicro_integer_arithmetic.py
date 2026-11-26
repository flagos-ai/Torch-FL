# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve Long arithmetic while lowering the pinned TsingMicro scalar math."""

import ast


_HOST = """
import triton
import triton.language as tl
import torch

from flag_gems.utils import libentry

@triton.jit
def published_scalar_math(x, y, alpha):
    return {expression}

@libentry()
@triton.jit
def {operation}_scalar_lanes(a, b, out, n, WRAPPED: tl.constexpr):
    lane = tl.program_id(0)
    if lane < n:
        x = tl.load(a + lane)
        if WRAPPED:
            y = b
        else:
            y = tl.load(b + lane)
        value = published_scalar_math(x, y, 1)
        tl.store(out + lane, value)

def {operation}(A, B, *, alpha=1):
    out = torch.empty(A.shape, dtype=A.dtype, device=A.device)
    count = A.numel()
    if count != 0:
        if isinstance(B, torch.Tensor):
            {operation}_scalar_lanes[(count,)](A, B, out, count, False)
        else:
            {operation}_scalar_lanes[(count,)](A, B, out, count, True)
    return out
"""


def render(published, _root):
    """Keep the published expression and replace vector loads with scalar ones."""
    variants = {"add_func": "x + y * alpha", "sub_func": "x - y * alpha"}
    (function,) = (
        node
        for node in ast.parse(published).body
        if isinstance(node, ast.FunctionDef) and node.name in variants
    )
    (result,) = function.body
    if (
        not isinstance(result, ast.Return)
        or ast.unparse(result.value) != variants[function.name]
    ):
        raise ValueError("published TsingMicro integer arithmetic changed")

    # The TX vector lowering converts Long operands through FP32. The scalar
    # form keeps the published integer expression in the device compiler.
    return _HOST.format(
        operation=function.name.removesuffix("_func"),
        expression=ast.unparse(result.value),
    ).lstrip()
