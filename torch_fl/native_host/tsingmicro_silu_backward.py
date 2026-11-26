# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the BF16 intermediate rounding in TsingMicro's eager SiLU backward."""


def render(published, _root):
    """Retain the published pointwise launch and change its scalar arithmetic."""
    text = published.decode("utf-8")
    original = """    dy_fp32 = dy.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + tl.exp(-x_fp32))
    dx = dy_fp32 * sigma * (1.0 + x_fp32 * (1.0 - sigma))
    return dx"""
    rounded = """    dy_fp32 = dy.to(tl.float32)
    x_fp32 = x.to(tl.float32)
    sigma = div_rn(1.0, 1.0 + tl.exp(-x_fp32)).to(tl.bfloat16).to(tl.float32)
    complement = (1.0 - sigma).to(tl.bfloat16).to(tl.float32)
    term = (x_fp32 * complement).to(tl.bfloat16).to(tl.float32)
    correction = (1.0 + term).to(tl.bfloat16).to(tl.float32)
    derivative = (sigma * correction).to(tl.bfloat16).to(tl.float32)
    return dy_fp32 * derivative"""
    host = """def silu_backward(grad_output, self):
    logger.debug("GEMS_TSINGMICRO SILU BACKWARD")
    grad_input = silu_backward_kernel(self, grad_output)
    return grad_input"""
    if text.count(original) != 1 or text.count(host) != 1:
        raise ValueError("unexpected published FlagGems SiLU backward")
    return text.replace(original, rounded, 1).replace(
        host,
        "def silu_backward(grad_output, self):\n"
        "    return silu_backward_kernel(self, grad_output)",
        1,
    )
