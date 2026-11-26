# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve integer addresses for the pinned TsingMicro NLL host."""

import ast


_FORWARD_GATHER = """
pid_n = tl.program_id(0)
total_out = 0.0
total_wgt = 0.0
for lane in range(BLOCK_N):
    row = pid_n * BLOCK_N + lane
    if row < N:
        target = tl.load(tgt_ptr + row)
        if (target != ignore_index) & (target >= 0) & (target < C):
            value = tl.load(inp_ptr + row * C + target.to(tl.int32)).to(tl.float32)
            total_out -= value
            total_wgt += 1.0
"""

_BACKWARD_SCATTER = """
pid_n = tl.program_id(0)
out_grad = tl.load(out_grad_ptr).to(tl.float32)
total_w = tl.load(total_weight).to(tl.float32)
inp_grad = -out_grad / total_w
for lane in range(BLOCK_N):
    row = pid_n * BLOCK_N + lane
    if row < N:
        target = tl.load(tgt_ptr + row)
        if (target != ignore_index) & (target >= 0) & (target < C):
            tl.store(inp_grad_ptr + row * C + target.to(tl.int32), inp_grad)
"""


def render(published, _root):
    """Adapt the mean reduction while retaining the published host entrypoints."""
    tree = ast.parse(published)
    kernels = {"nll_loss_forward_kernel", "nll_loss_backward_kernel"}
    adapted = set()
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name not in kernels:
            continue
        # TX converts vector offset arithmetic through FP32, which can round
        # class indices at flat offsets above 2**24. Scalar offsets stay integer.
        if function.name == "nll_loss_forward_kernel":
            replacement = ast.parse(_FORWARD_GATHER).body
            (reduction_none,) = (
                node
                for node in function.body
                if isinstance(node, ast.If)
                and ast.unparse(node.test) == "reduction == 0"
            )
            (reduction_mean,) = reduction_none.orelse
            assert ast.unparse(reduction_mean.test) == "reduction == 1"
            assert ast.unparse(reduction_mean.body[0]) == "total_out = tl.sum(out)"
            assert ast.unparse(reduction_mean.body[1]) == "total_wgt = tl.sum(wgt_tgt)"
            replacement += reduction_mean.body[2:]
        else:
            replacement = ast.parse(_BACKWARD_SCATTER).body
        function.body = [
            ast.If(
                test=ast.parse("reduction == 1 and wgt_ptr is None", mode="eval").body,
                body=replacement,
                orelse=function.body,
            )
        ]
        adapted.add(function.name)
    if adapted != kernels:
        raise ValueError("published TsingMicro NLL kernels changed")
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
