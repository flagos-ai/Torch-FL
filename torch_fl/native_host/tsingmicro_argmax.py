# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build the TsingMicro FP32 argmax variant from the published host and kernel."""

import ast
from hashlib import sha256


def render(published, _root):
    """Preserve the published indexed reduction and order NaN before numbers."""
    from .backends.tsingmicro import Backend

    tree = ast.parse(published)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    kernel = functions["argmax_kernel"]
    fixed = Backend.eager_kernel_configs[kernel.name, sha256(published).hexdigest()]
    # The published four-row tile fits the integer reduction's shared memory.
    # A single logical row retains one program and the same vocabulary tiles.
    config = dict(fixed, BLOCK_M=4)
    (tuner,) = (
        node
        for node in kernel.decorator_list
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "libtuner"
    )
    kernel.decorator_list.remove(tuner)

    assignments = {
        ast.unparse(node.targets[0]): node
        for node in kernel.body
        if isinstance(node, ast.Assign)
    }
    kernel.body.remove(assignments["acc_type"])
    assignments["max_values"].value = ast.parse(
        "tl.full([BLOCK_M], dtype=tl.int32, value=-0x7f800000)", mode="eval"
    ).body
    (loop,) = (node for node in kernel.body if isinstance(node, ast.For))
    (reduction,) = (
        node
        for node in loop.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "tl.max"
    )
    # Signed magnitudes preserve numeric ordering and unify positive and
    # negative zeros. Give all NaN payloads the same highest key.
    order = ast.parse("""
bits = inp_vals.to(tl.int32, bitcast=True)
magnitude = bits & 0x7fffffff
ordered = tl.where(bits < 0, -magnitude, magnitude)
ordered = tl.where(magnitude > 0x7f800000, 0x7fffffff, ordered)
""").body
    offset = loop.body.index(reduction)
    loop.body[offset:offset] = order
    reduction.value.args[0] = ast.Name("ordered", ast.Load())

    host = functions["argmax"]
    (dimension,) = (
        node
        for node in host.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "dim is None"
    )
    host.body = ast.parse("assert dim is not None").body + dimension.orelse
    (launch,) = (
        node
        for node in ast.walk(host)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Subscript)
        and ast.unparse(node.func.value) == kernel.name
    )
    launch.keywords.extend(
        ast.keyword(arg=name, value=ast.Constant(value))
        for name, value in config.items()
    )
    preamble = ast.parse("""
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min
from flag_gems.utils.shape_utils import can_use_int32_index
""").body
    result = ast.Module(body=[*preamble, kernel, host], type_ignores=[])
    return (
        "# Generated from the published FlagGems argmax host.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
        + ast.unparse(ast.fix_missing_locations(result))
        + "\n"
    )
