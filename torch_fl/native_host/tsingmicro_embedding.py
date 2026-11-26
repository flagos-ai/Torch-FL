# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build the scoped TX embedding variant from the published FlagGems host."""

import ast
from hashlib import sha256


_KERNEL = """
@triton.jit(do_not_specialize=["num_weights", "padding_idx"])
def embedding_bf16_input_order(
    grad_output_ptr,
    indices_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
    TOKENS: tl.constexpr,
):
    token = tl.program_id(0)
    tile = tl.program_id(1)
    index = tl.load(indices_ptr + token)
    if (index != padding_idx) & (index >= 0) & (index < num_weights):
        positions = tl.arange(0, TOKENS)
        indices = tl.load(indices_ptr + positions)
        first = tl.min(tl.where(indices == index, positions, TOKENS), axis=0)
        if token == first:
            columns = tile * BLOCK_D + tl.arange(0, BLOCK_D)
            accumulated = tl.full((BLOCK_D,), 0, tl.float32)
            for source in range(token, TOKENS):
                if tl.load(indices_ptr + source) == index:
                    contribution = tl.load(
                        grad_output_ptr + source * EMBED_DIM + columns
                    ).to(tl.float32)
                    accumulated = (accumulated + contribution).to(tl.bfloat16).to(tl.float32)
            tl.store(grad_weight_ptr + index * EMBED_DIM + columns, accumulated.to(tl.bfloat16))
"""


_ZERO_PROVIDER = "runtime/backend/_tsingmicro/ops/zeros.py"
_ZERO_SOURCE_SHA256 = "37db4658618b65b598fc9ae43a2366c61fb83ace411a0c6cb3a25c67666f6caf"


def render(published, root):
    """Preserve the published layout and launch recipe for the typed variant."""
    zero_source = root / _ZERO_PROVIDER
    if sha256(zero_source.read_bytes()).hexdigest() != _ZERO_SOURCE_SHA256:
        raise ValueError("published TsingMicro zeros source changed")
    original = ast.parse(published)
    (host,) = (
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name == "embedding_dense_backward"
    )
    (allocation,) = (
        node
        for node in host.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "grad_weight_fp32"
    )
    (dtype,) = (
        keyword for keyword in allocation.value.keywords if keyword.arg == "dtype"
    )
    dtype.value = ast.parse("grad_output.dtype", mode="eval").body
    allocation.value.func.attr = "empty"
    initialization = ast.parse("""
zero_elements = grad_weight_fp32.numel()
zero_grid = (min(triton.cdiv(zero_elements, 1024), _ZERO_CORES),)
_zero_kernel[zero_grid](
    grad_weight_fp32, zero_elements, BLOCK_SIZE=1024, num_warps=1, num_ctas=1, num_stages=1
)
""").body
    # The pinned first configuration writes exactly the same zeros without an
    # autotuner launch. Keep the published persistent grid and allocation shape.
    offset = host.body.index(allocation) + 1
    host.body[offset:offset] = initialization
    (frequency,) = (
        node
        for node in host.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "scale_grad_by_freq"
    )
    (launch,) = frequency.orelse
    assert launch.value.func.value.id == "_embedding_dense_backward_kernel"
    launch.value.func.value.id = "embedding_bf16_input_order"
    launch.value.keywords.append(
        ast.keyword(arg="TOKENS", value=ast.Name("N", ast.Load()))
    )
    offset = host.body.index(frequency)
    host.body[offset : offset + 1] = [launch]
    # The typed entry checks device identity before calling this variant.
    (device_check,) = (
        node
        for node in host.body
        if isinstance(node, ast.If)
        and any(
            isinstance(part, ast.Name) and part.id == "flag_gems"
            for part in ast.walk(node.test)
        )
    )
    host.body.remove(device_check)
    for node in ast.walk(host):
        if isinstance(node, ast.Name) and node.id == "grad_weight_fp32":
            node.id = "grad_weight"
    preamble = [node for node in original.body if not isinstance(node, ast.FunctionDef)]
    zero_provider = ast.parse("""
from flag_gems.runtime.backend._tsingmicro.ops.zeros import TOTAL_CORE_NUM, zeros_kernel
_ZERO_CORES = TOTAL_CORE_NUM
_zero_kernel = zeros_kernel.fn.fn
""").body
    generated = ast.Module(
        body=[*preamble, *ast.parse(_KERNEL).body, *zero_provider, host],
        type_ignores=[],
    )
    return (
        "# Generated from the published FlagGems embedding host.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
        + ast.unparse(ast.fix_missing_locations(generated))
        + "\n"
    )
