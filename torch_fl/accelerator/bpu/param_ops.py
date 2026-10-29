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

"""Compiled kernels for the ops that are not elementwise.

An elementwise op is one `.hbm` per (op, bucket) and nothing else: it has no
parameters, so the artifact depends only on the shape. A conv, a batch norm or
a linear carries weights, and the artifact bakes them in, so each module
instance needs its own artifact and the cache key has to include the weight
values.

Rather than describe those ops to hbdk4 by hand, each one is traced to an FX
graph and handed to `compile_partition`, the same export -> quantize -> compile
path `torch.backends` uses for whole models. That is the difference between
working and not: `hbir.max_pool` insists on an attribute spelling the Python
API does not accept, but `nn.MaxPool2d` reaches the same op through the ONNX
frontend without complaint.

The cost is compile time on the first call for each instance -- tens of seconds
per op on this board, since hbdk4 runs under box64. `compile_partition` caches
by graph and weight hash, so the second process to want the same conv pays
nothing.
"""

from __future__ import annotations

import hashlib
import logging
import threading

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn

from .compiler import CompileError, compile_partition, find_hbdk

log = logging.getLogger("torch_fl.bpu")


class _SingleOutput(nn.Module):
    """Wrap one module so its graph returns a tuple, which the exporter wants."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return (self.inner(x),)


class ParamKernel:
    """One parameterised op compiled for one concrete input shape.

    `module` is called with the caller's tensor already on CPU; the kernel owns
    the device round-trip and returns an array shaped like the module's output.
    """

    def __init__(self, module: nn.Module, in_shape: tuple[int, ...]):
        self.module = module.eval()
        self.in_shape = tuple(int(d) for d in in_shape)
        self._rt = None
        self._model = None
        self._in_name = None
        self._out_name = None
        self._lock = threading.Lock()
        self._failed = False

    def _weights_digest(self) -> str:
        digest = hashlib.sha256()
        for p in self.module.parameters():
            digest.update(np.ascontiguousarray(p.detach().numpy()).tobytes())
        digest.update(repr(self.in_shape).encode())
        return digest.hexdigest()[:16]

    def build(self) -> bool:
        """Compile once; returns False if this op cannot be compiled."""
        with self._lock:
            if self._rt is not None:
                return True
            if self._failed:
                return False
            try:
                gm = fx.symbolic_trace(_SingleOutput(self.module))
                path, in_names, out_names = compile_partition(
                    gm, [torch.randn(*self.in_shape)]
                )
                from hbm_runtime import HB_HBMRuntime

                self._rt = HB_HBMRuntime(str(path))
                self._model = self._rt.model_names[0]
                self._in_name = in_names[0]
                self._out_name = out_names[0]
                log.info(
                    "bpu eager: compiled %s %s (%s)",
                    type(self.module).__name__,
                    self.in_shape,
                    self._weights_digest(),
                )
                return True
            except (CompileError, NotImplementedError, RuntimeError, OSError) as exc:
                log.debug(
                    "bpu eager: %s %s not compilable: %s",
                    type(self.module).__name__,
                    self.in_shape,
                    exc,
                )
                self._failed = True
                return False

    def run(self, x: np.ndarray) -> np.ndarray:
        feed = {self._model: {self._in_name: np.ascontiguousarray(x, dtype=np.float32)}}
        out = self._rt.run(feed)[self._model]
        return out[self._out_name]


class ParamKernelCache:
    """Compiled parameterised ops, keyed by module identity and input shape."""

    def __init__(self) -> None:
        self._kernels: dict[tuple[int, str, tuple], ParamKernel] = {}
        self._lock = threading.Lock()
        self.stats = {"hit": 0, "miss_compile": 0, "unavailable": 0}

    def get(self, module: nn.Module, in_shape: tuple[int, ...]) -> ParamKernel | None:
        # Keyed on the module's identity alone. param_impls already guarantees
        # one module per distinct weight set (it rebuilds and caches by weight
        # content), so hashing the weights again here was pure cost -- and it
        # was the single largest item in a forward pass: 92 ms of 163 ms for
        # ResNet-18, because every conv re-hashed megabytes of weights on every
        # call, twice (once here, once building the module).
        key = (id(module), tuple(in_shape))
        with self._lock:
            kernel = self._kernels.get(key)
            if kernel is not None:
                self.stats["hit"] = self.stats.get("hit", 0) + 1
                return kernel

        kernel = ParamKernel(module, in_shape)
        # hbdk4 is an x86-only toolchain: without an emulator or the stubs there
        # is nothing to compile with, and every op has to take its CPU path.
        if find_hbdk() is None:
            with self._lock:
                self.stats["unavailable"] += 1
            return None
        if not kernel.build():
            with self._lock:
                self.stats["unavailable"] += 1
            return None
        with self._lock:
            self._kernels[key] = kernel
            self.stats["miss_compile"] += 1
        return kernel

    @staticmethod
    def _digest(module: nn.Module) -> str:
        digest = hashlib.sha256()
        for p in module.parameters():
            digest.update(p.detach().numpy().tobytes())
        return digest.hexdigest()[:16]

    def summary(self) -> str:
        total = sum(self.stats.values())
        if not total:
            return "param kernels: none"
        return (
            f"param kernels: {self.stats['miss_compile']} compiled, "
            f"{self.stats['unavailable']} not compilable, {total} lookups"
        )
