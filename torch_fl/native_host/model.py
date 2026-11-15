# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Execute model graphs through ATen and the precompiled FlagGems hosts."""

import functools
import hashlib


def _devices(value):
    import torch

    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, tuple):
        return tuple(_devices(item) for item in value)
    if isinstance(value, list):
        return [_devices(item) for item in value]
    if isinstance(value, dict):
        return {key: _devices(item) for key, item in value.items()}
    return value


def _check_graph(graph):
    for node in graph.nodes():
        if "Fusion" in node.kind() or "TensorExpr" in node.kind():
            raise RuntimeError("FlagGems model hosts do not permit numerical fusion")
        for block in node.blocks():
            _check_graph(block)


class ModelHostBackend:
    """A model host compiler that generates no numerical kernels or native SOs."""

    def __init__(self):
        self.graphs = []

    def __call__(self, graph_module, example_inputs):
        import torch

        for node in list(graph_module.graph.nodes):
            if (
                node.op == "call_function"
                and node.target is torch._C._log_api_usage_once
            ):
                if node.users or any(type(value) is not str for value in node.args):
                    raise RuntimeError("unsupported model API-usage record")
                node.target(*node.args, **node.kwargs)
                graph_module.graph.erase_node(node)
        for node in graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is torch.tensor
                and len(node.args) == 1
                and isinstance(node.args[0], (list, tuple))
                and not node.args[0]
            ):
                # A zero-element factory carries no numerical values. Spell its
                # shape explicitly to avoid TorchScript's List[Tensor] inference.
                node.target = torch.empty
                node.args = ((0,),)
            node.args = _devices(node.args)
            node.kwargs = _devices(node.kwargs)
        graph_module = torch.fx.GraphModule(graph_module, graph_module.graph)
        module = torch.jit.script(graph_module)
        _check_graph(module.graph)
        record = {
            "nodes": len(list(graph_module.graph.nodes)),
            "source_sha256": hashlib.sha256(graph_module.code.encode()).hexdigest(),
        }
        self.graphs.append(record)
        audited = False

        def execute(*args):
            nonlocal audited
            if torch.is_grad_enabled():
                raise RuntimeError("FlagGems model hosts require torch.no_grad()")
            with torch.jit.optimized_execution(True):
                result = module(*args)
            if not audited:
                graph = torch.jit.last_executed_optimized_graph()
                _check_graph(graph)
                record["execution_sha256"] = hashlib.sha256(
                    str(graph).encode()
                ).hexdigest()
                audited = True
            return result

        return execute


def compile_model(model, *, dynamic=True, fullgraph=True):
    """Compile inference host control flow while retaining FlagGems numerical ops.

    For an nn.Module, replace its forward method and return the same module so
    higher-level methods, including generation loops, use the compiled forward.
    Calls require torch.no_grad(). Numerical JIT fusers are disabled for this
    explicit FlagGems execution mode; graph guards retain dynamic-shape checks.
    """
    import torch
    from torch_fl import native_host

    if native_host._hosts is None or not hasattr(native_host._hosts, "typed"):
        raise RuntimeError("model hosts require a wheel with precompiled typed hosts")
    for name in (
        "_jit_set_texpr_fuser_enabled",
        "_jit_override_can_fuse_on_cpu",
        "_jit_override_can_fuse_on_gpu",
    ):
        getattr(torch._C, name)(False)
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(False)
    backend = ModelHostBackend()
    function = model.forward if isinstance(model, torch.nn.Module) else model
    compiled = torch.compile(
        function, backend=backend, dynamic=dynamic, fullgraph=fullgraph
    )

    @functools.wraps(function)
    def forward(*args, **kwargs):
        if torch.is_grad_enabled():
            raise RuntimeError("FlagGems model hosts require torch.no_grad()")
        with torch._dynamo.config.patch(specialize_float=True):
            return compiled(*args, **kwargs)

    forward._flagos_host_backend = backend
    if isinstance(model, torch.nn.Module):
        model.forward = forward
        return model
    return forward
