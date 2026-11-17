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

"""Expose stable Triton TMA descriptors to make_fx, using existing Inductor IR."""

import re

import torch
from torch._higher_order_ops import triton_kernel_wrap as wrap
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.virtualized import V
from torch.utils import _triton
from triton.tools.tensor_descriptor import TensorDescriptor

compiled = {}


def install():
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    save_kernel = CachingAutotuner.save_gpu_kernel

    def save_metadata(self, stream, launcher):
        result = save_kernel(self, stream, launcher)
        metadata = getattr(launcher.bin.metadata, "tensordesc_meta", None)
        if metadata:
            compiled[self.inductor_meta["kernel_name"]] = {
                "descriptors": metadata,
                "signature": self.triton_meta["signature"],
            }
        return result

    CachingAutotuner.save_gpu_kernel = save_metadata
    # CPU Torch cannot infer this from torch.version.cuda. The worker targets
    # the CUDA driver through PrivateUse1 and validates the backend separately.
    _triton.has_triton_tensor_descriptor_host_tma = lambda: True

    def call_hop(self, variable, grids, combined_args, tx):
        if tx is not None:
            raise RuntimeError("descriptor adapter is for make_fx only")
        metadata = {}
        combined_args = dict(combined_args)
        for name, value in combined_args.items():
            if isinstance(value, TensorDescriptor):
                if getattr(value, "padding", "zero") != "zero":
                    raise ValueError(
                        "TMA padding mode requires explicit native lowering"
                    )
                combined_args[name] = value.base.as_strided(value.shape, value.strides)
                metadata[name] = wrap.create_tma_stable_metadata(value.block_shape)
            elif isinstance(value, torch.SymBool):
                combined_args[name] = bool(value)
        graphable, constants = self.store_non_graphable_args(combined_args)
        return wrap.triton_kernel_wrapper_mutation(
            kernel_idx=variable.kernel_idx,
            constant_args_idx=constants,
            grid=grids,
            tma_descriptor_metadata=metadata,
            kwargs=graphable,
        )

    wrap.TracingTritonHOPifier.call_HOP = call_hop
    original = PythonWrapperCodegen._generate_tma_descriptor_call_stable

    def descriptor_example(self, desc, apply_size_hints=False):
        if not apply_size_hints:
            return original(self, desc, apply_size_hints)
        block = tuple(
            V.graph.sizevars.atomically_apply_size_hint(d) for d in desc.block_shape
        )
        # generate_example_arg_value has already allocated a Python tensor with
        # the descriptor view's shape and strides under this buffer name. A C++
        # reinterpret-view expression is not valid syntax in the Python tuner.
        return f"triton.tools.tensor_descriptor.TensorDescriptor.from_tensor({desc.tensor.get_name()}, {block})"

    PythonWrapperCodegen._generate_tma_descriptor_call_stable = descriptor_example


def native_abi(source):
    """Adapt PyTorch 2.10's old stable-TMA launcher to FlagTree's actual ABI."""
    if not compiled:
        return source
    source, removed = re.subn(r"struct StableTMADescriptor\s*\{[^}]*\};", "", source)
    if removed != 1:
        raise ValueError("unknown stable-TMA structure declaration")
    source = "#include <torch/csrc/inductor/cpp_wrapper/host_tma.h>\n" + source
    source = source.replace("StableTMADescriptor", "FlagosTMADescriptor")
    initialization = re.compile(
        r"initTMADescriptor\(&(?P<name>\w+)\.m, (?P<ptr>.+), \d+L?, \d+, (?P=name)\.block_shape, (?P=name)\.global_shape, (?P=name)\.strides\);"
    )
    source, count = initialization.subn(
        lambda m: f"{m['name']}.base_ptr = {m['ptr']};", source
    )
    if not count:
        raise ValueError("unknown stable-TMA initialization code")
    pattern = re.compile(
        r"(?P<args>\s*void\* kernel_args_\[\] = \{[^\n]+\};)\n(?P<call>\s*launchKernel\((?P<kernel>\w+),[^\n]+\);)"
    )
    updated = set()

    def launch(match):
        kernel = match["kernel"]
        if kernel not in compiled:
            return match[0]
        metadata = compiled[kernel]["descriptors"]
        names = re.findall(r"&(\w+)\.m", match["args"])
        if len(names) != len(metadata):
            raise ValueError("TMA descriptor metadata/ABI arity mismatch")
        lines, arguments = [], match["args"]
        for name, meta in zip(names, metadata):
            if meta is None or meta["fp4_padded"]:
                raise ValueError("unsupported non-TMA or padded descriptor ABI")
            block = meta["block_size"]
            rank = len(block)
            if not 1 <= rank <= 5:
                raise ValueError("unsupported descriptor rank")
            element_type = {8: 10, 9: 8, 10: 9}.get(
                meta["elem_type"], meta["elem_type"]
            )
            lines.append(
                f"    const uint32_t {name}_physical[] = {{{', '.join(map(str, block))}}};"
            )
            lines.append(
                f"    flagos_init_tma({name}, {rank}, {meta['elem_size']}, {element_type}, {meta['swizzle']}, {name}_physical);"
            )
            values = ", ".join(
                f"static_cast<int32_t>({name}.global_shape[{i}])" for i in range(rank)
            )
            lines.append(f"    int32_t {name}_shape[] = {{{values}}};")
            for i in range(rank):
                arguments = arguments.replace(
                    f"&{name}.block_shape[{i}]", f"&{name}_shape[{i}]"
                )
        updated.add(kernel)
        return "\n".join(lines) + arguments + "\n" + match["call"]

    source = pattern.sub(launch, source)
    if updated != set(compiled):
        raise ValueError("not every TMA kernel received its compiled descriptor ABI")
    return source
