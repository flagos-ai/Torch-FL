# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage the published BF16 SiLU with accurate positive rounding on TsingMicro."""


def render(published, _root):
    """Retain the pointwise kernel interface and lower the dense eager host."""
    text = published.decode("utf-8")
    signature = "def silu_forward_sigmoid(x):"
    start = text.index("    # 0.0 - x_fp32", text.index(signature))
    stop = text.index("\n\n\n@pointwise_dynamic", start)
    if text[start:stop].count("return x_fp32 * sigma") != 1:
        raise ValueError("unexpected published FlagGems SiLU body")
    # Subtracting the small correction preserves the BF16 midpoint at x=5.9375.
    accurate = """    z = tl.exp(0.0 - x_fp32)
    normal = x_fp32 * (1.0 / (1.0 + z))
    correction = z * div_rn(x_fp32, 1.0 + z)
    accurate = x_fp32 - correction
    bounded = (x_fp32 > 0.0) & (x_fp32 < 16.0)
    return tl.where(bounded, accurate, normal)"""
    text = text[:start] + accurate + text[stop:]
    eager = """def silu(self):
    logger.debug("GEMS_TSINGMICRO SILU FORWARD")
    collapsed = _collapse_contiguous_dims(self)
    output = silu_forward_sigmoid(collapsed)
    return output.view(self.shape)"""
    if text.count(eager) != 1:
        raise ValueError("unexpected published FlagGems SiLU host")
    return text.replace(eager, "def silu(self):\n    return silu_forward_sigmoid(self)")
