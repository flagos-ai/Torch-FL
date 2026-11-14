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

"""Isolated capture/codegen worker; never imported by the native hot path."""

import ctypes
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import re
from pathlib import Path
import sys
import traceback
import types

import torch
from torch._inductor import config
from torch._inductor.codegen.cpp_utils import DEVICE_TO_ATEN, DEVICE_TO_INT
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten
from torch.utils.cpp_extension import load

import torch_fl
from torch_fl.compile.inductor_codegen import (
    FlagOSDeviceOpOverrides,
    register_flagos_codegen,
)
from torch_fl.host_compiler import buffers, descriptors, mutations, source  # noqa: F401
from torch_fl.host_compiler.guards import native_guard
from torch_fl.host_compiler.lowering import DECOMPOSITIONS

_lib = None


def raw_stream(index):
    return _lib.GetCurrentStreamForDevice(index) or 0


def setup():
    global _lib
    descriptors.install()
    mutations.install()
    ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    _lib = ctypes.CDLL(
        str(Path(torch_fl.__file__).parent / "lib/libflagos.so"),
        mode=ctypes.RTLD_GLOBAL,
    )
    _lib.GetCurrentStreamForDevice.argtypes = [ctypes.c_int]
    _lib.GetCurrentStreamForDevice.restype = ctypes.c_void_p
    # Generated Python code executes only in this worker; native entries read
    # the current stream through the plugin's C ABI on every invocation.
    FlagOSDeviceOpOverrides.import_get_raw_stream_as = lambda self, name: (
        f"from __main__ import raw_stream as {name}"
    )
    register_flagos_codegen()
    DEVICE_TO_ATEN["flagos"] = "at::kPrivateUse1"
    DEVICE_TO_INT["flagos"] = 16
    config.compile_threads = 1
    old_sqrt = math.sqrt
    math.sqrt = lambda v: (
        torch.sym_sqrt(v) if isinstance(v, torch.SymInt) else old_sqrt(v)
    )


def decode(value, tensors):
    if not isinstance(value, dict):
        return value
    if value.get("tensor"):
        if value["conj"] or value["neg"]:
            raise ValueError("unresolved tensor math bits")
        shape, stride, offset = value["shape"], value["stride"], value["offset"]
        size = offset + 1 + sum((n - 1) * s for n, s in zip(shape, stride))
        dtype = getattr(torch, value["dtype"])
        cpu = torch.zeros(max(size, 1), dtype=dtype)
        if cpu.is_floating_point():
            cpu.normal_(0, 0.25)
        base = cpu.to(value["device"])
        tensor = base.as_strided(shape, stride, offset)
        tensors.append(tensor)
        return ("tensor", len(tensors) - 1)
    if "sequence" in value:
        return (value["sequence"], [decode(v, tensors) for v in value["items"]])
    if "dtype" in value:
        return getattr(torch, value["dtype"])
    raise ValueError(value)


def materialize(value, tensors):
    if isinstance(value, tuple):
        tag, payload = value
        if tag == "tensor":
            return tensors[payload]
        values = [materialize(v, tensors) for v in payload]
        return tuple(values) if tag == "tuple" else values
    return value


def complete_defaults(kernel, args, kwargs):
    supplied = set(kernel.arg_names[: len(args)]) | set(kwargs)
    for name, parameter in kernel.signature.parameters.items():
        if name not in supplied and parameter.default is not inspect.Parameter.empty:
            kwargs[name] = parameter.default

    def constant(value):
        if isinstance(value, torch.SymInt):
            return int(value)
        if isinstance(value, torch.SymBool):
            return bool(value)
        if isinstance(value, torch.SymFloat):
            return float(value)
        if isinstance(value, tuple):
            return tuple(constant(v) for v in value)
        if isinstance(value, list):
            return [constant(v) for v in value]
        return value

    args = list(args)
    for i, parameter in enumerate(kernel.params):
        if parameter.is_constexpr:
            if i < len(args):
                args[i] = constant(args[i])
            elif parameter.name in kwargs:
                kwargs[parameter.name] = constant(kwargs[parameter.name])
    return tuple(args), kwargs


def compile_request(request):
    setup()
    spec = json.loads(request.read_text())
    tensors = []
    args = decode(spec["args"], tensors)
    kwargs = {k: decode(v, tensors) for k, v in spec["kwargs"].items()}
    func = getattr(importlib.import_module(spec["module"]), spec["name"])

    def call(*inputs):
        result = func(
            *materialize(args, inputs),
            **{k: materialize(v, inputs) for k, v in kwargs.items()},
        )
        return tuple(tree_flatten(result)[0])

    entry = importlib.import_module("flag_gems.utils.libentry").LibEntry
    original = entry.run
    launches = []
    # Autotune only fresh worker-owned buffers. A selected configuration is a
    # kernel specialization, not a replay of a user's allocation or stream.
    prototype = call(*tensors)
    fresh_views = [
        any(torch._C._is_alias_of(value, t) for t in tensors)
        and all(value._cdata != t._cdata for t in tensors)
        for value in prototype
    ]

    def capture_run(self, *args, **kwargs):
        import triton

        grid = kwargs.pop("grid")
        kwargs.pop("warmup", None)
        args = tuple(buffers.tensor_argument(v) for v in args)
        kwargs = {k: buffers.tensor_argument(v) for k, v in kwargs.items()}
        fn = self.fn
        while not isinstance(fn, triton.runtime.JITFunction):
            if isinstance(fn, triton.runtime.Heuristics):
                values = {**dict(zip(fn.arg_names, args)), **kwargs}
                for key, heuristic in fn.values.items():
                    kwargs[key] = heuristic(values)
                    if isinstance(kwargs[key], torch.SymBool):
                        kwargs[key] = bool(kwargs[key])
                    values[key] = kwargs[key]
                fn = fn.fn
            elif isinstance(fn, triton.runtime.Autotuner):
                chosen = fn.best_config
                kwargs.update(chosen.kwargs)
                kwargs.update(num_warps=chosen.num_warps, num_stages=chosen.num_stages)
                values = {**dict(zip(fn.arg_names, args)), **kwargs}
                if chosen.pre_hook:
                    chosen.pre_hook(values)
                if getattr(fn, "user_defined_pre_hook", False) or getattr(
                    fn, "user_defined_post_hook", False
                ):
                    raise ValueError(
                        "autotuner stateful hooks require explicit lowering"
                    )
                fn = fn.fn
            else:
                raise TypeError(f"unsupported Triton host wrapper: {type(fn)}")
        args, kwargs = complete_defaults(fn, args, kwargs)
        launches.append(self.jit_function.__name__)

        def emit(*values, **options):
            launch_grid = options.pop("grid", grid)
            options.pop("warmup", None)
            return torch.library.wrap_triton(fn)[launch_grid](*values, **options)

        runner = fn.__dict__.get("run")
        if isinstance(runner, types.FunctionType):
            replaced = 0

            def adapt(function):
                nonlocal replaced
                if (
                    isinstance(function, types.MethodType)
                    and function.__self__ is fn
                    and function.__func__ is original_jit
                ):
                    replaced += 1
                    return emit
                if (
                    not isinstance(function, types.FunctionType)
                    or not function.__closure__
                ):
                    return function
                cells = []
                for cell in function.__closure__:
                    value = adapt(cell.cell_contents)
                    cells.append((lambda value: lambda: value)(value).__closure__[0])
                return types.FunctionType(
                    function.__code__,
                    function.__globals__,
                    function.__name__,
                    function.__defaults__,
                    tuple(cells),
                )

            adapted = adapt(runner)
            if not replaced:
                raise ValueError("cannot lower the custom Triton run hook")
            return adapted(*args, grid=grid, warmup=False, **kwargs)
        return emit(*args, **kwargs)

    entry.run = capture_run
    import triton

    original_jit = triton.runtime.JITFunction.run

    def capture_jit(self, *args, **kwargs):
        grid = kwargs.pop("grid")
        kwargs.pop("warmup", None)
        args, kwargs = complete_defaults(self, args, kwargs)
        launches.append(self.__name__)
        return torch.library.wrap_triton(self)[grid](*args, **kwargs)

    triton.runtime.JITFunction.run = capture_jit
    try:
        graph = make_fx(
            call, decomposition_table=DECOMPOSITIONS, tracing_mode="symbolic"
        )(*tensors)
    finally:
        entry.run = original
        triton.runtime.JITFunction.run = original_jit
    (request.parent / "graph.py").write_text(graph.code)
    allowed = (
        "aten.empty",
        "aten.new_empty",
        "aten.view",
        "aten.as_strided",
        "aten.permute",
        "aten.transpose",
        "aten.expand",
        "aten.slice",
        "aten.select",
        "aten.squeeze",
        "aten.unsqueeze",
        "aten.detach",
        "aten.alias",
        "aten.sym_",
        "aten._unsafe_view",
        "triton_kernel_wrapper_",
    )
    import operator

    for node in graph.graph.nodes:
        if node.op != "call_function" or node.target == operator.getitem:
            continue
        if getattr(node.target, "__module__", "") == "_operator":
            if isinstance(node.meta.get("val"), torch.Tensor):
                raise ValueError(f"numerical residual: {node.target}")
            continue
        if not str(node.target).startswith(allowed):
            raise ValueError(
                f"unsupported residual (no replacement kernels): {node.target}"
            )
    from torch._inductor.codecache import CppWrapperCodeCache

    generated = []

    def capture_binding(*args, **kwargs):
        generated.append(kwargs["main_code"])

        def unused_python_entry(*args):
            raise RuntimeError("host compiler must not execute the Python wrapper")

        return unused_python_entry

    original_binding = CppWrapperCodeCache.load_pybinding
    CppWrapperCodeCache.load_pybinding = staticmethod(capture_binding)
    try:
        has_descriptors = any(
            n.kwargs.get("tma_descriptor_metadata") for n in graph.graph.nodes
        )
        patches = {
            "cpp_wrapper": True,
            "triton.cudagraphs": False,
            "triton.store_cubin": True,
            "triton.autotune_at_compile_time": True,
        }
        if has_descriptors:
            fake_inputs = [
                n.meta["val"] for n in graph.graph.nodes if n.op == "placeholder"
            ]
            with config.patch(patches):
                compile_fx_inner(
                    graph, fake_inputs, cpp_wrapper=True, is_inference=True
                )
        else:
            compile_fx(graph, tensors, config_patches=patches)
    finally:
        CppWrapperCodeCache.load_pybinding = original_binding
    if not generated and not launches:
        placeholders = [n for n in graph.graph.nodes if n.op == "placeholder"]
        output = next(n for n in graph.graph.nodes if n.op == "output").args[0]
        if all(value in placeholders for value in output):
            lines = [
                "#include <torch/csrc/inductor/cpp_wrapper/flagos.h>",
                "#include <torch/csrc/inductor/aoti_torch/utils.h>",
                "void inductor_entry_impl(AtenTensorHandle* in, AtenTensorHandle* out) {",
                f"  auto inputs = steal_from_raw_handles_to_raii_handles(in, {len(placeholders)});",
            ]
            for i, value in enumerate(output):
                index = placeholders.index(value)
                lines.append(
                    f"  out[{i}] = torch::aot_inductor::new_tensor_handle(at::Tensor(*torch::aot_inductor::tensor_handle_to_tensor_pointer(inputs[{index}])));"
                )
            lines.append("}")
            generated.append("\n".join(lines))
    if len(generated) != 1:
        raise ValueError(f"expected one generated wrapper, got {len(generated)}")
    cpp = generated[0]
    (request.parent / "descriptor_metadata.json").write_text(
        json.dumps(descriptors.compiled, indent=2)
    )
    cpp = descriptors.native_abi(cpp)
    kernel_calls = re.findall(r"launchKernel\((\w+),", cpp)
    for name in kernel_calls:
        if not any(
            name == original or name.startswith(original + "_") for original in launches
        ):
            raise ValueError(f"generated replacement kernel is forbidden: {name}")
    if launches and not kernel_calls:
        raise ValueError("all original FlagGems launches were eliminated")
    gil_entries = cpp.count("py::gil_scoped_release_simple release;")
    if gil_entries not in (0, 1) or (launches and gil_entries != 1):
        raise ValueError("unknown generated GIL ABI")
    cpp = cpp.replace("py::gil_scoped_release_simple release;", "// Native entry.")
    cpp += "\n" + native_guard(graph)
    cpp += '\n#include <ATen/ops/alias_native.h>\n#include <torch/csrc/inductor/aoti_torch/utils.h>\nextern "C" void flagos_host_entry(AtenTensorHandle* in, AtenTensorHandle* out) {\n  inductor_entry_impl(in, out);\n'
    for i, fresh in enumerate(fresh_views):
        if fresh:
            cpp += f"  auto* value_{i} = torch::aot_inductor::tensor_handle_to_tensor_pointer(out[{i}]);\n  *value_{i} = at::native::alias(*value_{i});\n"
    cpp += "}\n"
    target = request.parent / "host.cpp"
    target.write_text(cpp)
    headers = b"".join(
        p.read_bytes() for p in sorted((Path(__file__).parent / "include").rglob("*.h"))
    )
    digest = hashlib.sha256(
        cpp.encode() + headers + torch.__version__.encode()
    ).hexdigest()
    immutable_source = request.parent / ("host-" + digest[:16] + ".cpp")
    immutable_source.write_text(cpp)
    library = load(
        name="flagos_host_" + request.parent.name + "_" + digest[:16],
        sources=[str(immutable_source)],
        extra_include_paths=[
            str(Path(__file__).parent / "include"),
            "/usr/local/cuda/include",
        ],
        extra_cflags=["-O2"],
        is_python_module=False,
    )
    torch.ops.flagos.host_program_register(
        spec["key"], library, spec["outputs"], Path(library).stat().st_size
    )
    before = [t.cpu() for t in tensors]
    expected = call(*tensors)
    expected_cpu = [t.cpu() for t in expected]
    after = [t.cpu() for t in tensors]
    aliases = [
        [torch._C._is_alias_of(result, t) for t in tensors] for result in expected
    ]
    identities = [[result._cdata == t._cdata for t in tensors] for result in expected]
    for tensor, old, new in zip(tensors, before, after):
        if not torch.equal(old, new):
            tensor.copy_(old)
    actual = torch.ops.flagos.host_program_run(spec["key"], tensors)
    for result, reference, alias, identity in zip(
        actual, expected_cpu, aliases, identities
    ):
        torch.testing.assert_close(result.cpu(), reference, equal_nan=True)
        if alias != [torch._C._is_alias_of(result, t) for t in tensors]:
            raise ValueError("generated output changed the input alias contract")
        if identity != [result._cdata == t._cdata for t in tensors]:
            raise ValueError("generated output changed tensor identity")
    for tensor, reference in zip(tensors, after):
        torch.testing.assert_close(tensor.cpu(), reference, equal_nan=True)
    artifact = {
        "key": spec["key"],
        "outputs": spec["outputs"],
        "library": library,
        "kernels": launches,
        "guards": [str(g.expr) for g in graph.shape_env.guards],
        "source": str(immutable_source),
        "source_sha256": digest,
        "library_sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        "cohort": {
            "torch": torch.__version__,
            "flag_gems": importlib.metadata.version("flag_gems"),
            "flagtree": importlib.metadata.version("flagtree"),
        },
        "native_kernel_calls": kernel_calls,
    }
    (request.parent / "artifact.json").write_text(json.dumps(artifact, indent=2))
    (request.parent / "failure.txt").unlink(missing_ok=True)


if __name__ == "__main__":
    request = Path(sys.argv[1]).resolve()
    try:
        with torch.no_grad():
            compile_request(request)
    except Exception:
        (request.parent / "failure.txt").write_text(traceback.format_exc())
        raise
