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

"""Eager operator dispatch over a prebuilt per-operator .hbm library.

The BPU has no per-operator entry point of its own: `hbDNNInferV2` takes a
compiled artifact. This module supplies the missing layer by treating a
directory of single-operator artifacts as an operator library -- built once on
a host with hbdk4 (`scripts/vendor/build_bpu_oplib.py`), loaded here with the
board's native runtime and nothing compiled on the device.

Shapes are bucketed. The alternative, a truly dynamic artifact, does not work
on HBRT 4.7.5: the loader declares a dynamic axis with stride 0, while the
task validator requires at least 64 and separately rejects any buffer that is
not exactly the stride extent, so no caller can satisfy both. See
`docs/vendors/bpu/dynamic-shape-probe.md`. A request that misses every bucket
falls back to the reference implementation instead of failing.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

log = logging.getLogger("torch_fl.bpu")

# aten target -> operator name in the library, for the one-input ops.
# Targets are spelled the way a dispatcher sees them; `Library.impl` takes the
# same names minus the "aten." prefix, so both forms are accepted below.
_UNARY = {
    "aten.relu": "relu",
    "aten.relu.default": "relu",
    "aten.neg": "neg",
    "aten.neg.default": "neg",
    "aten.abs": "abs",
    "aten.abs.default": "abs",
    "aten.exp": "exp",
    "aten.exp.default": "exp",
    "aten.tanh": "tanh",
    "aten.tanh.default": "tanh",
    "aten.sigmoid": "sigmoid",
    "aten.sigmoid.default": "sigmoid",
    "aten.sqrt": "sqrt",
    "aten.sqrt.default": "sqrt",
    "aten.rsqrt": "rsqrt",
    "aten.rsqrt.default": "rsqrt",
    "aten.log": "log",
    "aten.log.default": "log",
}

# aten target -> operator name, for the broadcast-free two-input ops.
_BINARY = {
    "aten.add.Tensor": "add",
    "aten.sub.Tensor": "sub",
    "aten.mul.Tensor": "mul",
    "aten.maximum": "max",
    "aten.maximum.default": "max",
    "aten.minimum": "min",
    "aten.minimum.default": "min",
    "aten.div.Tensor": "div",
}

# The reference path, used whenever no bucket matches. It is deliberately the
# plain NumPy implementation: it is correct for any shape and dtype, and it is
# what the artifacts above were verified against.
_REFERENCE = {
    "relu": lambda a, b: np.maximum(a, 0),
    "neg": lambda a, b: -a,
    "abs": lambda a, b: np.abs(a),
    "exp": lambda a, b: np.exp(a),
    "tanh": lambda a, b: np.tanh(a),
    "sigmoid": lambda a, b: 1.0 / (1.0 + np.exp(-a)),
    "sqrt": lambda a, b: np.sqrt(a),
    "rsqrt": lambda a, b: 1.0 / np.sqrt(a),
    "log": lambda a, b: np.log(a),
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "max": lambda a, b: np.maximum(a, b),
    "min": lambda a, b: np.minimum(a, b),
    "div": lambda a, b: a / b,
}


class KernelLibrary:
    """The artifacts in one oplib directory, indexed for lookup by shape."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} not found")
        manifest = json.loads(manifest_path.read_text())

        self.march = manifest.get("march")
        self.dtype = manifest.get("dtype", "f32")
        self.numpy_dtype = np.dtype(manifest.get("numpy_dtype", "float32"))
        self._by_op: dict[tuple[str, int], dict[int, dict]] = {}
        for k in manifest.get("kernels", []):
            dims = tuple(int(d) for d in k["shape"])
            if len(dims) != 2 or dims[0] != 1:
                # Only the [1, N] family is generated today; anything else in
                # the manifest is ignored rather than guessed at.
                continue
            key = (k["op"], k["nargs"])
            self._by_op.setdefault(key, {})[dims[1]] = k
        self._models: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.Lock()
        self._max_cached = 32

    @property
    def ops(self) -> list[str]:
        return sorted({op for op, _ in self._by_op})

    def buckets(self, op: str, nargs: int) -> list[int]:
        return sorted(self._by_op.get((op, nargs), {}))

    def lookup(self, op: str, nargs: int, shape: tuple[int, ...]) -> dict | None:
        """Exact bucket for `shape`, or None.

        Buckets are exact on purpose. A larger bucket cannot be substituted
        because the graph's shape is compiled into the artifact: feeding a
        64-wide tensor to a 512-wide artifact produces 512 values.
        """
        if len(shape) != 2 or shape[0] != 1:
            return None
        return self._by_op.get((op, nargs), {}).get(int(shape[1]))

    def _runtime(self, entry: dict):
        path = str(self.root / entry["file"])
        with self._lock:
            rt = self._models.get(path)
            if rt is None:
                from hbm_runtime import HB_HBMRuntime

                rt = HB_HBMRuntime(path)
                self._models[path] = rt
                if len(self._models) > self._max_cached:
                    self._models.popitem(last=False)
            else:
                self._models.move_to_end(path)
        return rt

    def run(self, entry: dict, arrays: list[np.ndarray]) -> np.ndarray:
        rt = self._runtime(entry)
        model = rt.model_names[0]
        names = rt.input_names[model]
        feed = {model: {n: a for n, a in zip(names, arrays)}}
        out = rt.run(feed)[model]
        return out[rt.output_names[model][0]]


class EagerDispatcher:
    """Routes aten calls to the operator library, falling back to NumPy.

    The counter is the point of the class as much as the dispatch is: an eager
    path that silently degrades to the reference implementation is
    indistinguishable from one that works, so every call is tallied as a hit or
    a miss.
    """

    def __init__(self, root: str | Path | None, enabled: bool = True):
        self.enabled = enabled and root is not None
        self.library = KernelLibrary(root) if self.enabled else None
        self.stats = {"hit": 0, "miss_shape": 0, "miss_op": 0, "disabled": 0}
        self._lock = threading.Lock()

    def _tally(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def supported(self, target: str) -> str | None:
        return _UNARY.get(target) or _BINARY.get(target)

    def dispatch(self, target: str, arrays: list[np.ndarray]) -> np.ndarray:
        """Run `target` on the BPU when a bucket exists, else on the reference."""
        op = self.supported(target)
        if not self.enabled or self.library is None:
            self._tally("disabled")
            return self._fallback(op, arrays)
        if op is None:
            self._tally("miss_op")
            return self._fallback(None, arrays)

        shape = tuple(arrays[0].shape)
        nargs = len(arrays)
        entry = self.library.lookup(op, nargs, shape)
        if entry is None:
            self._tally("miss_shape")
            return self._fallback(op, arrays)

        try:
            out = self.library.run(
                entry,
                [
                    np.ascontiguousarray(a, dtype=self.library.numpy_dtype)
                    for a in arrays
                ],
            )
        except Exception as exc:  # noqa: BLE001 - a broken artifact must not crash the graph
            log.warning("bpu oplib %s failed, falling back: %s", entry["name"], exc)
            self._tally("miss_op")
            return self._fallback(op, arrays)
        self._tally("hit")
        return out.astype(arrays[0].dtype, copy=False)

    @staticmethod
    def _fallback(op: str | None, arrays: list[np.ndarray]) -> np.ndarray:
        if op is None or op not in _REFERENCE:
            raise NotImplementedError(f"no reference for {op}")
        fn = _REFERENCE[op]
        return fn(*arrays) if len(arrays) == 2 else fn(arrays[0], None)

    def summary(self) -> str:
        total = sum(self.stats.values())
        if not total:
            return "bpu eager: no calls"
        hit = self.stats["hit"]
        return (
            f"bpu eager: {hit}/{total} on device "
            f"(miss_shape={self.stats['miss_shape']} "
            f"miss_op={self.stats['miss_op']} "
            f"disabled={self.stats['disabled']})"
        )
