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

"""aten entry points for the ops that take weights.

The elementwise table in `eager.py` registers ops one aten name at a time,
because each one is a single `.hbm` with no parameters. convolution, batch norm
and pooling cannot work that way: their weights are baked into the artifact, so
the artifact belongs to a module instance rather than to an op name. These
impls rebuild an `nn.Module` from the aten arguments, hand it to
`EagerDispatcher.run_module`, and decline the call when that returns None.

Declining is deliberate: each wrapper raises NotImplementedError, which sends
the operator back through the backend's own `cpu_fallback` registration, so a
module that cannot be compiled behaves exactly as it did before this file
existed. Nothing here reimplements a kernel.
"""

from __future__ import annotations

import logging
import threading

import torch
import torch.nn as nn

log = logging.getLogger("torch_fl.bpu")

# Rebuilding a module per call would be pointless: the compiled artifact is
# keyed on the weights, so reusing one Conv2d per weight tensor is what lets
# repeated calls reach the cache instead of re-deriving it.
_holders: dict[tuple, int] = {}
_holder_lock = threading.Lock()


def _pair(v) -> tuple[int, int]:
    if isinstance(v, (tuple, list)):
        return (int(v[0]), int(v[-1]))
    return (int(v), int(v))


def _module_cache_key(kind, weight, bias, *params):
    """Identify a module by its parameter storage and configuration.

    Hashing the weights themselves is the obvious key and the wrong one: an
    nn.Conv2d holds megabytes, and re-hashing them per call cost 75 ms of a
    163 ms ResNet-18 forward. A module's nn.Parameter objects are stable for
    the life of the model -- aten passes the same tensors every time -- so
    their identity plus the view layout identifies the weight set just as
    well. The version counter catches a module whose weights were edited in
    place, which is the one case identity alone would miss.
    """
    parts = [
        kind,
        str(weight.data_ptr()),
        str(weight._version),
        repr(tuple(weight.shape)),
    ]
    parts.append(str(bias.data_ptr()) if bias is not None else "-")
    if bias is not None:
        parts.append(str(bias._version))
    parts.append(repr(params))
    return "|".join(parts)


class _FunctionalOp(nn.Module):
    """A parameterised op expressed through the functional API.

    aten's counterpart of this operator is a function, so its BPU form should
    call the functional API too rather than build an nn.Module -- initialise,
    copy weights in, and throw the scaffolding away on every call. This holds
    only the tensors the function needs; symbolic_trace sees one conv2d call.
    """

    def __init__(self, weight, stride: int, groups: int):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.stride = stride
        self.groups = groups

    def forward(self, x):
        return torch.nn.functional.conv2d(
            x, self.weight, None, self.stride, 0, 1, self.groups
        )


def _avg_pool_as_conv(channels: int, kernel: int, stride: int) -> nn.Module:
    """Average pooling as the depthwise conv that computes the same mean.

    Cached by shape: every avg_pool with the same channel count, window and
    stride is the same operator, and its only weight is a constant. Building
    the weight once also keeps the artifact cache warm, since the compiled
    kernel is keyed on the module instance.
    """
    key = f"avgpool|{channels}|{kernel}|{stride}"
    with _holder_lock:
        cached = _holders.get(key)
    if cached is not None:
        return cached
    weight = torch.full(
        (channels, 1, kernel, kernel), 1.0 / (kernel * kernel), dtype=torch.float32
    )
    module = _FunctionalOp(weight, stride, channels)
    with _holder_lock:
        _holders[key] = module
    return module


def _cached_module(key, build):
    with _holder_lock:
        module = _holders.get(key)
    if module is not None:
        return module
    module = build()
    if module is not None:
        with _holder_lock:
            _holders[key] = module
    return module


def _conv_module(inp, weight, bias, stride, padding, dilation, groups):
    """An nn.Conv2d reproducing the aten call, or None for shapes we do not serve."""
    if inp.dim() != 4 or weight.dim() != 4:
        return None
    key = _module_cache_key(
        "conv",
        weight,
        bias,
        _pair(stride),
        _pair(padding),
        _pair(dilation),
        int(groups),
    )
    return _cached_module(
        key, lambda: _build_conv(weight, bias, stride, padding, dilation, groups)
    )


def _build_conv(weight, bias, stride, padding, dilation, groups):
    module = nn.Conv2d(
        int(weight.shape[1]) * int(groups),
        int(weight.shape[0]),
        _pair(weight.shape[2:]),
        stride=_pair(stride),
        padding=_pair(padding),
        dilation=_pair(dilation),
        groups=int(groups),
        bias=bias is not None,
    ).eval()
    with torch.no_grad():
        module.weight.copy_(weight.detach().cpu())
        if bias is not None:
            module.bias.copy_(bias.detach().cpu())
    return module


def build_impls(lib: torch.library.Library, dispatcher) -> None:
    """Install the impls; `dispatcher` is the active EagerDispatcher."""

    def conv_override(
        inp, weight, bias, stride, padding, dilation, transposed, output_padding, groups
    ):
        if transposed:
            raise NotImplementedError
        module = _conv_module(inp, weight, bias, stride, padding, dilation, groups)
        if module is None:
            raise NotImplementedError
        out = dispatcher.run_module(module, inp)
        if out is None:
            raise NotImplementedError
        return out

    def batch_norm_override(
        inp,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        cudnn_enabled,
    ):
        # F.batch_norm calls aten::batch_norm, and that is the name the
        # PrivateUse1 dispatch actually lands on; native_batch_norm and
        # _native_batch_norm_legit are reached only below it.
        if training or inp.dim() != 4:
            raise NotImplementedError
        module = nn.BatchNorm2d(int(inp.shape[1]), eps=float(eps)).eval()
        with torch.no_grad():
            if weight is not None:
                module.weight.copy_(weight.detach().cpu())
            if bias is not None:
                module.bias.copy_(
                    bias.detach().cpu() if hasattr(bias, "detach") else bias
                )
            if running_mean is not None:
                module.running_mean.copy_(running_mean.detach().cpu())
            if running_var is not None:
                module.running_var.copy_(running_var.detach().cpu())
        out = dispatcher.run_module(module, inp)
        if out is None:
            raise NotImplementedError
        return out

    def max_pool_override(inp, kernel_size, stride, padding=0):
        # Polypool with explicit padding would need a pad attribute the pooling
        # artifact does not carry, so those take the CPU path.
        if inp.dim() != 4 or _pair(padding) != (0, 0):
            raise NotImplementedError
        module = nn.MaxPool2d(
            _pair(kernel_size),
            stride=_pair(stride) if stride else _pair(kernel_size),
            padding=0,
        )
        out = dispatcher.run_module(module, inp)
        if out is None:
            raise NotImplementedError
        return out

    def avg_pool_override(inp, kernel_size, stride):
        # Served as a depthwise conv, not as nn.AvgPool2d.
        #
        # Both lower to the same operation, but only one of them survives the
        # quantizer. Compiled through hbir's average pool the artifact is wrong
        # by a relative L2 of 0.63 on inputs in [0, 1] -- not a scale that is
        # merely coarse, since 0.529 quantises to 0.55 on a 0.05 grid and the
        # kernel returns 0.85. With FLAGOS_BPU_QUANTIZE off the same graph is
        # bit-exact, so the fault is in how Q/DQ is applied to that op, and
        # there is no knob on this side to fix it.
        #
        # A conv whose weights are all 1/k^2 computes the same average, and its
        # quantisation is the one the toolchain actually tunes: measured 0.0133
        # at k=2 on 16x16, 0.0138 on 7x7 where the pool drops the tail, and
        # 0.0094 at k=3 on 32x32 -- conv-grade error instead of 46x worse.
        if inp.dim() != 4:
            raise NotImplementedError
        k = _pair(kernel_size)
        s = _pair(stride) if stride else k
        if k[0] != k[1] or s[0] != s[1]:
            raise NotImplementedError
        op = _avg_pool_as_conv(int(inp.shape[1]), k[0], s[0])
        out = dispatcher.run_module(op, inp)
        if out is None:
            raise NotImplementedError
        return out

    installed = []
    for name, fn in (
        ("convolution_overrideable", conv_override),
        ("batch_norm", batch_norm_override),
        ("max_pool2d", max_pool_override),
        ("avg_pool2d", avg_pool_override),
    ):
        try:
            lib.impl(name, fn, "PrivateUse1")
        except RuntimeError as exc:  # already claimed elsewhere
            log.debug("bpu eager: not claiming aten::%s (%s)", name, exc)
        else:
            installed.append(name)
    log.info("bpu eager: parameterised impls for %s", ", ".join(installed))
