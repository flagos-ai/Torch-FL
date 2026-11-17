# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preserve runtime scalars in PyTorch 2.10's Triton mutation analysis.

Its analysis converts every non-tensor argument into a constexpr. This breaks
perfectly valid kernels that call `.to()` on a runtime scalar (for example the
FlagGems pointwise factory). Its fallback then drops TensorBox writes in the
Inductor pass, allowing the entire launch to be eliminated. Keep scalar ABI
slots and their types instead. This patch exists only in the isolated worker.
"""

import ast
import inspect

import torch
from torch._higher_order_ops import triton_kernel_wrap as wrap
from torch._inductor.ir import TensorBox


class PreserveScalarArguments(ast.NodeTransformer):
    def __init__(self):
        self.replaced = set()

    def visit_Assign(self, node):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return self.generic_visit(node)
        name = node.targets[0].id
        replacements = {
            "constants": """
constants = {p.name: ordered_args[p.name] for p in kernel.params if p.is_constexpr}
""",
            "ordered_tensor_names": """
ordered_tensor_names = list(itertools.chain.from_iterable(
    get_tensor_names(name, arg) if is_tensor_like_arg(arg)
    else [name + ' SCALAR PLACEHOLDER']
    for i, (name, arg) in enumerate(ordered_args.items())
    if not kernel.params[i].is_constexpr
))
""",
        }
        if name in replacements:
            self.replaced.add(name)
            return ast.parse(replacements[name]).body[0]
        return self.generic_visit(node)


def install():
    tree = ast.parse(inspect.getsource(wrap.generate_ttir))
    transform = PreserveScalarArguments()
    tree = ast.fix_missing_locations(transform.visit(tree))
    if transform.replaced != {"constants", "ordered_tensor_names"}:
        raise RuntimeError("unsupported PyTorch mutation-analysis source")
    namespace = dict(wrap.__dict__)
    exec(compile(tree, "<flagos scalar-aware mutation analysis>", "exec"), namespace)
    wrap.generate_ttir = namespace["generate_ttir"]
    identify = wrap.identify_mutated_tensors

    def tensor_writes(kernel, kwargs, metadata):
        # Pointer arithmetic depends on scalar offsets/strides. The upstream
        # dependency walk marks those scalar parameters too; they are ABI slots,
        # never writable buffers or inputs to functionalization's clone pass.
        writes = [
            name
            for name in identify(kernel, kwargs, metadata)
            if name in kwargs and isinstance(kwargs[name], (torch.Tensor, TensorBox))
        ]
        if not writes:
            raise RuntimeError("cannot prove any Triton output writes; refusing DCE")
        return writes

    wrap.identify_mutated_tensors = tensor_writes
