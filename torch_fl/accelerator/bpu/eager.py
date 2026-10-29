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
import math
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

log = logging.getLogger("torch_fl.bpu")

# The one-input ops, by short name. Every entry has an artifact family in the
# operator library and a reference implementation below.
_UNARY_OPS = (
    "relu",
    "neg",
    "abs",
    "exp",
    "tanh",
    "sigmoid",
    "sqrt",
    "rsqrt",
    "log",
    "ceil",
    "floor",
    "round",
    "sign",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
    "erf",
    "reciprocal",
    "mish",
    "swish",
    "gelu",
    "softplus",
    "elu",
)

# The broadcast-free two-input ops. Broadcasting is not part of the library:
# a graph is compiled at one shape, so both operands must already match it.
_BINARY_OPS = ("add", "sub", "mul", "max", "min", "div", "pow", "mod")

# aten target -> operator name. Targets are spelled the way a dispatcher sees
# them; `Library.impl` takes the same names minus the "aten." prefix. Both the
# bare name and the ".default" overload appear because callers reach for
# either, and a few ops have their own target names (torch spells negation
# "negative", and the two-input forms of pow/remainder carry an overload).
_UNARY = {f"aten.{n}": n for n in _UNARY_OPS}
_UNARY.update({f"aten.{n}.default": n for n in _UNARY_OPS})
_UNARY.update(
    {
        "aten.negative": "neg",
        "aten.negative.default": "neg",
        "aten.erf.default": "erf",
    }
)

_BINARY = {
    "aten.add.Tensor": "add",
    "aten.sub.Tensor": "sub",
    "aten.mul.Tensor": "mul",
    "aten.div.Tensor": "div",
    "aten.maximum": "max",
    "aten.maximum.default": "max",
    "aten.minimum": "min",
    "aten.minimum.default": "min",
    "aten.pow.Tensor_Tensor": "pow",
    "aten.remainder.Tensor": "mod",
    "aten.fmod.Tensor": "mod",
}


# The reference path, used whenever no bucket matches. It is deliberately the
# plain NumPy implementation: correct for any shape and dtype, and what the
# artifacts above were verified against.
def _erf(x):
    # np has no erf; math.erf vectorized is accurate and only runs on a miss.
    return np.vectorize(math.erf, otypes=[x.dtype])(x)


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
    "ceil": lambda a, b: np.ceil(a),
    "floor": lambda a, b: np.floor(a),
    "round": lambda a, b: np.round(a),
    "sign": lambda a, b: np.sign(a),
    "sin": lambda a, b: np.sin(a),
    "cos": lambda a, b: np.cos(a),
    "tan": lambda a, b: np.tan(a),
    "sinh": lambda a, b: np.sinh(a),
    "cosh": lambda a, b: np.cosh(a),
    "asin": lambda a, b: np.arcsin(a),
    "acos": lambda a, b: np.arccos(a),
    "atan": lambda a, b: np.arctan(a),
    "asinh": lambda a, b: np.arcsinh(a),
    "acosh": lambda a, b: np.arccosh(a),
    "atanh": lambda a, b: np.arctanh(a),
    "erf": lambda a, b: _erf(a),
    "reciprocal": lambda a, b: 1.0 / a,
    "mish": lambda a, b: a * np.tanh(np.log1p(np.exp(a))),
    "swish": lambda a, b: a / (1.0 + np.exp(-a)),
    "gelu": lambda a, b: 0.5 * a * (1.0 + _erf(a / np.sqrt(2.0))),
    "softplus": lambda a, b: np.log1p(np.exp(a)),
    "elu": lambda a, b: np.where(a > 0, a, np.expm1(a)),
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "max": lambda a, b: np.maximum(a, b),
    "min": lambda a, b: np.minimum(a, b),
    "div": lambda a, b: a / b,
    "pow": lambda a, b: np.power(a, b),
    "mod": lambda a, b: np.fmod(a, b),
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
        self._direct_cache: dict[str, object] = {}

    @property
    def ops(self) -> list[str]:
        return sorted({op for op, _ in self._by_op})

    def buckets(self, op: str, nargs: int) -> list[int]:
        return sorted(self._by_op.get((op, nargs), {}))

    def lookup(self, op: str, nargs: int, shape: tuple[int, ...]) -> dict | None:
        """Smallest bucket that can hold `shape`, or None.

        Buckets are inclusive-upward: a request is padded into the next bucket
        at or above its width and the result is truncated back. Padding is safe
        because every op in the library is elementwise, so the extra lanes
        cannot affect the ones before them -- verified on the board for all 15
        ops, with the padded tail coming back as the op's own result for zeros.

        Exact matching was the earlier rule and it was needlessly strict: the
        BPU's cost is flat in width (0.141 ms at 64 lanes, 0.152 ms at 512), so
        a lightly over-wide bucket costs one graph's latency, while exact
        matching sent every off-bucket shape to the host.
        """
        if len(shape) != 2 or shape[0] != 1:
            return None
        width = int(shape[1])
        buckets = self._by_op.get((op, nargs), {})
        for size in sorted(buckets):
            if size >= width:
                return buckets[size]
        return None

    @staticmethod
    def bucket_width(entry: dict) -> int:
        return int(entry["shape"][1])

    def _direct(self, entry: dict):
        """A cached-buffer executor for this artifact, or None if unavailable.

        `HB_HBMRuntime.run()` allocates and frees both device buffers on every
        call. On the board that malloc/free pair costs 0.025 ms against a
        whole-op budget of ~0.14 ms, and the C-level equivalent with the
        buffers held open runs the same work in 0.045 ms. This keeps them open.
        """
        key = str(self.root / entry["file"])
        with self._lock:
            cached = self._direct_cache.get(key)
        if cached is not None:
            return cached

        try:
            from .submit import DirectKernel

            kernel = DirectKernel(
                str(self.root / entry["file"]),
                entry["name"],
                self.bucket_width(entry),
                self.numpy_dtype,
            )
        except Exception as exc:  # noqa: BLE001 - fall back to hbm_runtime
            log.debug(
                "bpu oplib direct path unavailable for %s: %s", entry["name"], exc
            )
            return None

        with self._lock:
            self._direct_cache[key] = kernel
        return kernel

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
        """Execute `entry`, padding each array up to the bucket and cropping back.

        Returns an array shaped like the request, not like the bucket.
        """
        width = self.bucket_width(entry)
        extent = int(np.prod(arrays[0].shape))

        direct = self._direct(entry)
        if direct is not None:
            try:
                return direct.run(arrays)[:extent]
            except Exception as exc:  # noqa: BLE001 - a broken fast path must not fail the op
                log.warning("bpu oplib direct run failed on %s: %s", entry["name"], exc)
                with self._lock:
                    self._direct_cache.pop(str(self.root / entry["file"]), None)

        rt = self._runtime(entry)
        model = rt.model_names[0]
        names = rt.input_names[model]

        feed = {model: {}}
        for name, a in zip(names, arrays):
            flat = np.ascontiguousarray(a, dtype=self.numpy_dtype).reshape(-1)
            if flat.size == width:
                feed[model][name] = flat.reshape(1, width)
            elif flat.size < width:
                # Zero padding. Every library op is elementwise, so the lanes
                # past the request cannot influence the ones inside it.
                padded = np.zeros(width, self.numpy_dtype)
                padded[: flat.size] = flat
                feed[model][name] = padded.reshape(1, width)
            else:
                raise ValueError(
                    f"{entry['name']}: request of {flat.size} does not fit "
                    f"bucket {width}; lookup() should have ruled this out"
                )

        out = rt.run(feed)[model]
        result = out[rt.output_names[model][0]].reshape(-1)
        return result[: int(np.prod(arrays[0].shape))]


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
        # run() returns the request's flat extent; restore the caller's shape.
        return out.astype(arrays[0].dtype, copy=False).reshape(arrays[0].shape)

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


# The dispatcher `register_eager()` installs, so callers can read its counters.
_ACTIVE: EagerDispatcher | None = None


def oplib_root() -> str | None:
    """Directory named by FLAGOS_BPU_OPLIB, or None.

    Unset means eager stays off: the operator library is an opt-in artifact a
    deployment has to ship, and a missing directory must not turn into a
    confusing import-time failure.
    """
    root = os.environ.get("FLAGOS_BPU_OPLIB", "").strip()
    return root or None


def active() -> EagerDispatcher | None:
    """The dispatcher installed into PrivateUse1, if any."""
    return _ACTIVE


def register_eager(root: str | Path | None = None) -> EagerDispatcher | None:
    """Route the library's aten ops on PrivateUse1 through the oplib.

    Called by `torch_fl` on a BPU build when FLAGOS_BPU_OPLIB is set. Without
    it every op keeps reaching `cpu_fallback`, which is the previous behaviour,
    so a board that ships no operator library is unaffected.

    The registration is deliberately narrow: only the ops this module knows how
    to serve *and* has a reference for are claimed. Claiming an op and then
    raising would turn a working CPU fallback into an error, and claiming one
    with no reference would do the same on the first shape miss.
    """
    global _ACTIVE

    resolved = root if root is not None else oplib_root()
    if resolved is None:
        return None
    if _ACTIVE is not None:
        return _ACTIVE

    import torch

    dispatcher = EagerDispatcher(resolved)
    library = torch.library.Library("aten", "IMPL")

    def _make(target: str, nargs: int):
        def _impl(*args, **kwargs):
            arrays = [
                a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
                for a in args[:nargs]
            ]
            out = dispatcher.dispatch(target, arrays)
            tensors = [a for a in args[:nargs] if isinstance(a, torch.Tensor)]
            device = tensors[0].device if tensors else torch.device("cpu")
            return torch.from_numpy(np.ascontiguousarray(out)).to(device)

        return _impl

    for target in _UNARY:
        library.impl(target[len("aten.") :], _make(target, 1), "PrivateUse1")
    for target in _BINARY:
        library.impl(target[len("aten.") :], _make(target, 2), "PrivateUse1")

    # Keep the Library alive: it unregisters on garbage collection.
    dispatcher._torch_library = library  # noqa: SLF001
    _ACTIVE = dispatcher
    log.info("bpu eager: %d ops from %s", len(_UNARY) + len(_BINARY), resolved)
    return dispatcher
