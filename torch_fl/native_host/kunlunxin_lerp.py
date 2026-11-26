# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate the published XPU BF16 scalar lerp with one precise final round."""

import ast


def render(payload, _root):
    source = payload.decode("utf-8")
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "lerp_scalar_kernel_head"
    )
    body = ast.get_source_segment(source, function)
    original = "    return (input32 + weight32 * (end32 - input32)).to(input.dtype)"
    if body.count(original) != 1:
        raise RuntimeError("published XPU lerp changed its final BF16 conversion")
    # Round the high 16 bits to even and quiet NaNs before the XPU BF16 store.
    # The reconstructed FP32 value is already representable in BF16.
    precise = """    result = tl.fma(weight32, end32 - input32, input32)
    bits = result.to(tl.uint32, bitcast=True)
    high = bits >> 16
    rounded = (bits + 32767 + (high & 1)) >> 16
    nan = (bits & 2147483647) > 2139095040
    rounded = tl.where(nan, high | 64, rounded)
    return (rounded << 16).to(tl.float32, bitcast=True)"""
    return (
        "import triton\nfrom triton import language as tl\n\n"
        '@triton.jit(do_not_specialize=["weight"])\n'
        + body.replace(original, precise)
        + "\n\n\n"
        "from _kunlunxin.utils.pointwise_dynamic import pointwise_dynamic\n\n"
        "lerp_scalar_kernel = pointwise_dynamic(\n"
        "    is_tensor=[True, True, False],\n"
        "    dtypes=[None, None, float],\n"
        '    promotion_methods=[(0, 1, "DEFAULT")],\n'
        ")(lerp_scalar_kernel_head)\n\n\n"
        "def lerp_scalar_(input, end, weight):\n"
        "    return lerp_scalar_kernel(input, end, weight, out0=input)\n"
    )
