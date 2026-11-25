# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Use native CUDA numerics for Cosmos inference on NVIDIA."""

from contextlib import contextmanager

import torch
from torch._ops import OpOverload
from torch.utils._python_dispatch import (
    TorchDispatchMode,
    _get_current_dispatch_mode,
)
from torch.utils._pytree import tree_leaves, tree_map

from torch_fl import _C


def _to_flagos(value):
    if isinstance(value, torch.Tensor) and value.device.type == "cuda":
        return _C._cuda_to_flagos_view(value)
    return value


class _CudaNumerics(TorchDispatchMode):
    supports_higher_order_operators = True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if not isinstance(func, OpOverload):
            return func(*args, **kwargs)
        schema = func._schema
        if (
            not schema.name.startswith("aten::")
            or schema.is_mutable
            or any(result.alias_info is not None for result in schema.returns)
            or torch.Tag.nondeterministic_seeded in func.tags
        ):
            return func(*args, **kwargs)

        operands = (args, kwargs)
        tensors = [
            value for value in tree_leaves(operands) if isinstance(value, torch.Tensor)
        ]
        flagos = [value for value in tensors if value.device.type == "flagos"]
        if not flagos or any(
            value.device.type not in ("cpu", "flagos") for value in tensors
        ):
            return func(*args, **kwargs)
        if torch.is_grad_enabled() and any(value.requires_grad for value in tensors):
            return func(*args, **kwargs)
        if any(
            type(value) not in (torch.Tensor, torch.nn.Parameter)
            or value.layout != torch.strided
            or value.is_conj()
            or value.is_neg()
            for value in flagos
        ):
            return func(*args, **kwargs)
        devices = {value.device.index for value in flagos}
        if len(devices) != 1:
            return func(*args, **kwargs)

        index = devices.pop()
        views = {}

        def to_cuda(value):
            if isinstance(value, torch.Tensor) and value.device.type == "flagos":
                key = id(value)
                if key not in views:
                    views[key] = _C._flagos_to_cuda_view(value)
                return views[key]
            if isinstance(value, torch.device) and value.type == "flagos":
                return torch.device("cuda", value.index)
            return value

        # Wait for work from either allocator's streams before sharing storage.
        torch.flagos.synchronize(index)
        with torch.cuda.device(index):
            cuda_args, cuda_kwargs = tree_map(to_cuda, operands)
            result = func(*cuda_args, **cuda_kwargs)
            torch.cuda.synchronize(index)
        destination = kwargs.get("device")
        if isinstance(destination, torch.device) and destination.type == "cuda":
            return result
        return tree_map(_to_flagos, result)


@contextmanager
def suspend_cuda_numerics():
    """Let the original attention operator lower its native CUDA kernels."""
    active = _get_current_dispatch_mode()
    if not isinstance(active, _CudaNumerics):
        yield
        return

    # Exiting the scope also restores Dynamo's dispatch-mode state before
    # the original torch.compile wrapper is called.
    active.__exit__(None, None, None)
    try:
        yield
    finally:
        active.__enter__()


_mode = None


def install_cuda_numerics():
    """Activate native numerics for this dedicated inference thread."""
    global _mode
    if _mode is None:
        _mode = _CudaNumerics()
        _mode.__enter__()
