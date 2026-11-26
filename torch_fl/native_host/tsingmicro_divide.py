# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage the shared FlagGems scalar divide with masked loads on TsingMicro."""

import ast


_POINTWISE = """
from copy import copy
from flag_gems.ops.div import true_div_func_tensor_scalar as _published_divide
from flag_gems.utils.pointwise_dynamic import PointwiseDynamicFunction

_masked_config = copy(_published_divide.config)
_masked_config.prefer_block_pointer = False
true_div_func_tensor_scalar = PointwiseDynamicFunction(
    _published_divide.fx, _published_divide._scalar_fn, config=_masked_config
)
"""


def render(published, _root):
    """Select the scalar branch and keep its JIT arithmetic unchanged."""
    (host,) = (
        node
        for node in ast.parse(published).body
        if isinstance(node, ast.FunctionDef) and node.name == "true_divide"
    )
    (scalar_return,) = (
        node
        for node in ast.walk(host)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "true_div_func_tensor_scalar"
    )
    host.body = [scalar_return]
    return _POINTWISE + "\n" + ast.unparse(host) + "\n"
