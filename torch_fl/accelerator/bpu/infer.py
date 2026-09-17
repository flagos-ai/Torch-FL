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

"""Direct hbDNNInferV2 binding with device-resident IO buffers.

`runtime.BPURuntime` drives `hbm_runtime.run()`, which takes a dict of numpy
arrays and copies every one of them into device memory on each call. For a CNN
that is a 588 KB input and it does not matter. For an LLM decode step the KV
cache is an *input*, 336 MiB of it for Qwen3-0.6B at 4096 context, and copying
it per token costs more than the inference: measured **68.4 ms/token** against
the vendor runtime's 11.42 ms.

The vendor's own `libxlm.so` does not use that path either -- its undefined
symbols are `hbDNNInferV2`/`V3` plus `hbUCPMallocCached`. This module takes the
same approach: allocate every input and output once in UCP memory, let the
caller write into the buffers in place, and pass pointers to the device.

Two findings from bringing this up against the vendor's own Qwen3-0.6B artifact
are load-bearing, because neither is documented and both cost 3-6x if missed:

**`hbUCPReleaseTask` must not be called right after `hbUCPWaitTaskDone`.**
Releasing inline costs **28 ms**; deferring it by a single step costs 1.65 ms
and the step drops from 43.2 ms to 11.3 ms. A task handle cannot be resubmitted
(`Task is already submitted or release, status 4`), so a task is built per step
and released one step late -- see `_ReleaseRing`. Deferring without bound is not
an option: the handle pool runs dry and `hbDNNInferV2` starts returning null
handles, which shows up as an inference that silently does not run.

**`HB_DNN_USER_DEFINED_L2M_SIZES` must be set before the first inference.**
Unset, an LLM model fails with `L2 memory not enough ... user-assigned l2
memspace size: [0, 0, 0, 0]` and `hbUCPWaitTaskDone` returns -200003. The
vendor's `run_llm.sh` exports `6:6:6:6`; on this board 8:8:8:8 also fails, so
6:6:6:6 is what `ensure_l2_config()` sets.

A third is a correctness trap rather than a performance one: the KV cache is a
**sliding window**, not a buffer you append into. Getting it wrong produces
fluent, wrong text and no error at all. `KVWindow` documents the layout and how
it was recovered -- by tracing the vendor's `hbDNNInferV2` calls.

Measured on the vendor Qwen3-0.6B artifact, decode with a resident cache and
the sliding window: **12.23 ms/token (81.8 tok/s)** against the vendor `llm`
demo's 11.80 ms/token (84.8 tok/s), same .hbm, same four cores. Prefill on a
full 512-token chunk runs at 6881 tok/s against the demo's 5626-7014.
"""

from __future__ import annotations

import ctypes
import logging

import numpy as np

log = logging.getLogger("torch_fl.bpu")

MAX_DIMS = 10

# hbDNNTensorType, in declaration order (hb_dnn.h).
_TENSOR_TYPES = (
    "S4",
    "U4",
    "S8",
    "U8",
    "F16",
    "S16",
    "U16",
    "F32",
    "S32",
    "U32",
    "F64",
    "S64",
    "U64",
)
_NP_OF_TYPE = {
    "S8": np.int8,
    "U8": np.uint8,
    "F16": np.float16,
    "S16": np.int16,
    "U16": np.uint16,
    "F32": np.float32,
    "S32": np.int32,
    "U32": np.uint32,
    "F64": np.float64,
    "S64": np.int64,
    "U64": np.uint64,
}

_CORE_BIT = (1 << 0, 1 << 1, 1 << 2, 1 << 3)

HB_SYS_MEM_CACHE_INVALIDATE = 1
HB_SYS_MEM_CACHE_CLEAN = 2

# What the vendor's run_llm.sh exports. 0:0:0:0 (the default) and 8:8:8:8 both
# fail on this board with a -200003 from hbUCPWaitTaskDone.
L2M_ENV = "HB_DNN_USER_DEFINED_L2M_SIZES"
L2M_DEFAULT = "6:6:6:6"


def ensure_l2_config(value: str = L2M_DEFAULT) -> None:
    """Set the L2 memspace sizes an LLM artifact needs, if unset.

    Must happen before the runtime library reads it, which is at first
    inference. Setting it after that has no effect, so callers construct
    `Package` only after this has run.
    """
    from torch_fl import _env

    if _env.set_foreign(L2M_ENV, value):
        log.debug("%s not set; using %s", L2M_ENV, value)


class hbUCPSysMem(ctypes.Structure):
    _fields_ = [
        ("phyAddr", ctypes.c_uint64),
        ("virAddr", ctypes.c_void_p),
        ("memSize", ctypes.c_uint64),
    ]


class hbDNNTensorShape(ctypes.Structure):
    _fields_ = [
        ("dimensionSize", ctypes.c_int32 * MAX_DIMS),
        ("numDimensions", ctypes.c_int32),
    ]


class hbDNNQuantiScale(ctypes.Structure):
    _fields_ = [
        ("scaleLen", ctypes.c_int32),
        ("scaleData", ctypes.POINTER(ctypes.c_float)),
        ("zeroPointLen", ctypes.c_int32),
        ("zeroPointData", ctypes.POINTER(ctypes.c_int32)),
    ]


class hbDNNTensorProperties(ctypes.Structure):
    _fields_ = [
        ("validShape", hbDNNTensorShape),
        ("tensorType", ctypes.c_int32),
        ("scale", hbDNNQuantiScale),
        ("quantiType", ctypes.c_int32),
        ("quantizeAxis", ctypes.c_int32),
        ("alignedByteSize", ctypes.c_int64),
        ("stride", ctypes.c_int64 * MAX_DIMS),
    ]


class hbDNNTensor(ctypes.Structure):
    _fields_ = [
        ("sysMem", hbUCPSysMem),
        ("properties", hbDNNTensorProperties),
    ]


class hbUCPSchedParam(ctypes.Structure):
    _fields_ = [
        ("priority", ctypes.c_int32),
        ("customId", ctypes.c_int64),
        ("backend", ctypes.c_uint64),
        ("deviceId", ctypes.c_uint32),
    ]


class InferError(RuntimeError):
    """A libdnn/libhbucp call returned a non-zero status."""


_LIB = None


def _library() -> ctypes.CDLL:
    """The first loadable library exporting hbDNNInferV2, cached."""
    global _LIB
    if _LIB is not None:
        return _LIB
    tried = []
    for name in ("libhbucp.so", "libdnn.so", "libhbdnn.so", "libbpu.so"):
        try:
            lib = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:  # not installed, or a dependency is missing
            tried.append(f"{name}: {e}")
            continue
        if hasattr(lib, "hbDNNInferV2"):
            _LIB = lib
            return lib
        tried.append(f"{name}: no hbDNNInferV2")
    raise InferError("no library exports hbDNNInferV2:\n  " + "\n  ".join(tried))


_c_i32 = ctypes.c_int32
_c_u64 = ctypes.c_uint64
_c_vp = ctypes.c_void_p
_c_cp = ctypes.c_char_p
_P = ctypes.POINTER

_FUNCS: dict[str, object] = {}


def _fn(name: str, restype, *argtypes):
    """Look up and memoize a C entry point with its signature applied."""
    cached = _FUNCS.get(name)
    if cached is not None:
        return cached
    f = getattr(_library(), name)
    f.restype = restype
    f.argtypes = list(argtypes)
    _FUNCS[name] = f
    return f


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise InferError(f"{what} failed: rc={rc}")


def _shape_of(props: hbDNNTensorProperties) -> tuple[int, ...]:
    return tuple(
        props.validShape.dimensionSize[i] for i in range(props.validShape.numDimensions)
    )


def _dtype_of(props: hbDNNTensorProperties):
    idx = props.tensorType
    name = _TENSOR_TYPES[idx] if 0 <= idx < len(_TENSOR_TYPES) else str(idx)
    dt = _NP_OF_TYPE.get(name)
    if dt is None:
        raise InferError(f"tensor type {name} has no numpy equivalent")
    return dt


class DeviceBuffer:
    """One UCP allocation, exposed as a numpy array over its host mapping.

    `hbUCPMallocCached` memory is host-writable, so the array *is* the device
    buffer. Writing to it and calling `clean()` is what a host-to-device copy
    would otherwise do, minus the copy.
    """

    __slots__ = ("mem", "array", "nbytes")

    def __init__(self, nbytes: int, dtype, shape, device: int = 0):
        malloc = _fn("hbUCPMallocCached", _c_i32, _P(hbUCPSysMem), _c_u64, _c_i32)
        self.mem = hbUCPSysMem()
        _check(
            malloc(ctypes.byref(self.mem), nbytes, device),
            f"hbUCPMallocCached({nbytes} bytes)",
        )
        self.nbytes = nbytes
        count = int(np.prod(shape)) if shape else 1
        buf = (ctypes.c_uint8 * nbytes).from_address(self.mem.virAddr)
        self.array = np.frombuffer(buf, dtype=dtype, count=count).reshape(shape)

    def clean(self) -> None:
        """Publish host writes so the device sees them."""
        flush = _fn("hbUCPMemFlush", _c_i32, _P(hbUCPSysMem), _c_i32)
        flush(ctypes.byref(self.mem), HB_SYS_MEM_CACHE_CLEAN)

    def invalidate(self) -> None:
        """Drop stale cache lines so host reads see what the device wrote."""
        flush = _fn("hbUCPMemFlush", _c_i32, _P(hbUCPSysMem), _c_i32)
        flush(ctypes.byref(self.mem), HB_SYS_MEM_CACHE_INVALIDATE)

    def free(self) -> None:
        if self.mem.virAddr:
            _fn("hbUCPFree", _c_i32, _P(hbUCPSysMem))(ctypes.byref(self.mem))
            self.mem.virAddr = None


class _ReleaseRing:
    """Holds task handles for `depth` steps before releasing them.

    `hbUCPReleaseTask` immediately after `hbUCPWaitTaskDone` blocks for ~28 ms
    on this board -- more than twice the inference itself. Released one step
    later it costs 1.65 ms, and the decode step goes from 43.2 ms to 11.3 ms.

    The ring must stay shallow. Handles are a finite pool, and once it is empty
    `hbDNNInferV2` hands back a null handle instead of failing loudly, so the
    step appears to run at triple speed while computing nothing. Depth 1 already
    recovers the whole 28 ms, so there is no reason to go deeper.
    """

    __slots__ = ("_ring", "_depth", "_release")

    def __init__(self, depth: int = 1):
        self._ring: list = []
        self._depth = max(1, depth)
        self._release = _fn("hbUCPReleaseTask", _c_i32, _c_vp)

    def push(self, task) -> None:
        self._ring.append(task)
        while len(self._ring) > self._depth:
            self._release(self._ring.pop(0))

    def drain(self) -> None:
        while self._ring:
            self._release(self._ring.pop(0))


class Model:
    """One model inside an .hbm, with its IO buffers resident on the device.

    Buffers are allocated once. The caller writes inputs into `inputs[name]`
    and reads `outputs[name]`; `infer()` moves no bulk data, which is the whole
    point for an LLM whose KV cache is an input.
    """

    def __init__(self, handle, name: str, cores=(0, 1, 2, 3), release_depth: int = 1):
        self.handle = handle
        self.name = name
        self.backend = 0
        for c in cores:
            if not 0 <= c < len(_CORE_BIT):
                raise ValueError(f"BPU core {c} out of range")
            self.backend |= _CORE_BIT[c]

        n_in, n_out = _c_i32(), _c_i32()
        _check(
            _fn("hbDNNGetInputCount", _c_i32, _P(_c_i32), _c_vp)(
                ctypes.byref(n_in), handle
            ),
            "hbDNNGetInputCount",
        )
        _check(
            _fn("hbDNNGetOutputCount", _c_i32, _P(_c_i32), _c_vp)(
                ctypes.byref(n_out), handle
            ),
            "hbDNNGetOutputCount",
        )

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.inputs: dict[str, np.ndarray] = {}
        self.outputs: dict[str, np.ndarray] = {}
        self._in = (hbDNNTensor * n_in.value)()
        self._out = (hbDNNTensor * n_out.value)()
        self._in_bufs: list[DeviceBuffer] = []
        self._out_bufs: list[DeviceBuffer] = []
        self._index = {}
        # Indices whose buffer has been handed over to a KVWindow; their own
        # allocation is gone, so flush/invalidate must skip them.
        self._skip_in: set[int] = set()
        self._skip_out: set[int] = set()

        self._bind(
            n_in.value,
            self._in,
            self.input_names,
            self.inputs,
            self._in_bufs,
            "hbDNNGetInputName",
            "hbDNNGetInputTensorProperties",
        )
        self._bind(
            n_out.value,
            self._out,
            self.output_names,
            self.outputs,
            self._out_bufs,
            "hbDNNGetOutputName",
            "hbDNNGetOutputTensorProperties",
        )
        self._index = {n: i for i, n in enumerate(self.input_names)}
        self._oindex = {n: i for i, n in enumerate(self.output_names)}
        self._ring = _ReleaseRing(release_depth)
        self._sched = hbUCPSchedParam(
            priority=0, customId=0, backend=self.backend, deviceId=0
        )

    def rebind(self, name: str, mem: hbUCPSysMem, output: bool = False) -> None:
        """Point one tensor at `mem` instead of its own allocation.

        Used by `KVWindow` to slide the cache: the descriptor moves, the data
        does not. The buffer this displaces is dropped from the flush lists,
        since its memory is no longer the one the device reads.
        """
        if output:
            i = self._oindex[name]
            self._out[i].sysMem = mem
            self.outputs.pop(name, None)
            self._skip_out.add(i)
        else:
            i = self._index[name]
            self._in[i].sysMem = mem
            self.inputs.pop(name, None)
            self._skip_in.add(i)

    def _bind(self, count, tensors, names, arrays, bufs, name_fn, props_fn) -> None:
        get_name = _fn(name_fn, _c_i32, _P(_c_cp), _c_vp, _c_i32)
        get_props = _fn(props_fn, _c_i32, _P(hbDNNTensorProperties), _c_vp, _c_i32)
        for i in range(count):
            nm = _c_cp()
            _check(get_name(ctypes.byref(nm), self.handle, i), name_fn)
            props = hbDNNTensorProperties()
            _check(get_props(ctypes.byref(props), self.handle, i), props_fn)
            key = nm.value.decode()
            dtype = _dtype_of(props)
            shape = _shape_of(props)
            # alignedByteSize is what the device requires; the visible array
            # covers only the valid shape, which can be smaller.
            nbytes = max(
                int(props.alignedByteSize),
                int(np.prod(shape)) * np.dtype(dtype).itemsize,
            )
            buf = DeviceBuffer(nbytes, dtype, shape)
            tensors[i].sysMem = buf.mem
            tensors[i].properties = props
            names.append(key)
            arrays[key] = buf.array
            bufs.append(buf)

    def flush_inputs(self, names=None) -> None:
        """Publish host writes for `names`, or for every input if None.

        Passing the handful of small inputs that actually change per step keeps
        this at ~0.01 ms; flushing all of them would walk the whole KV cache.
        """
        if names is None:
            for i, b in enumerate(self._in_bufs):
                if i not in self._skip_in:
                    b.clean()
            return
        for n in names:
            self._in_bufs[self._index[n]].clean()

    def invalidate_outputs(self) -> None:
        for i, b in enumerate(self._out_bufs):
            if i not in self._skip_out:
                b.invalidate()

    def infer(self, timeout_ms: int = 20000) -> None:
        """Run one inference over the resident buffers.

        Raises rather than returning quietly when the handle pool is exhausted:
        a null handle here means the step would compute nothing while appearing
        to succeed.
        """
        infer_v2 = _fn(
            "hbDNNInferV2", _c_i32, _P(_c_vp), _P(hbDNNTensor), _P(hbDNNTensor), _c_vp
        )
        submit = _fn("hbUCPSubmitTask", _c_i32, _c_vp, _P(hbUCPSchedParam))
        wait = _fn("hbUCPWaitTaskDone", _c_i32, _c_vp, _c_i32)

        task = _c_vp()
        _check(
            infer_v2(ctypes.byref(task), self._out, self._in, self.handle),
            "hbDNNInferV2",
        )
        if not task.value:
            raise InferError(
                "hbDNNInferV2 returned a null task handle -- the pool is "
                "exhausted; release depth is too deep"
            )
        rc = submit(task, ctypes.byref(self._sched))
        if rc != 0:
            self._ring.push(task)
            raise InferError(f"hbUCPSubmitTask failed: rc={rc}")
        rc = wait(task, timeout_ms)
        self._ring.push(task)
        _check(rc, "hbUCPWaitTaskDone")

    def free(self) -> None:
        self._ring.drain()
        for b in (*self._in_bufs, *self._out_bufs):
            b.free()
        self._in_bufs.clear()
        self._out_bufs.clear()


class KVWindow:
    """The KV cache as a sliding view over one allocation per layer.

    An LLM graph takes `layer_i_cache_{key,value}` of `window` slots and emits
    `layer_i_new_{key,value}` for the tokens it just consumed. The obvious
    reading -- append the new K/V into the cache at the current position -- is
    wrong, and produces fluent-looking garbage rather than an error.

    What the vendor runtime actually does, visible by tracing `hbDNNInferV2`
    through `LD_PRELOAD`, is move the pointer: `sysMem.virAddr` for the cache
    advances by exactly one slot per decode step while the allocation stays
    put, and the `new_key` output is bound one full window ahead, at
    `cache_base + window * slot_bytes`. So a token's K/V is written once, by
    the device, into the slot the next step will read as part of its window.
    Nothing is ever copied.

    The consequence for the mask is the part that cannot be guessed. Inside the
    graph the attended keys are `concat(window[span:], new_keys)` for a step of
    `span` tokens, so mask column `c` refers to window slot `c + span`, and the
    last `span` columns are the tokens of the current step. With `pos` tokens
    of history and row `r` of the current step:

        open columns = [window - span - pos, window - span + r + 1)

    which reproduces the traced prefill rows (`[3584, 3584+r+1)` at pos 0,
    span 512) and decode (`[4071, 4096)` at pos 24, span 1) exactly. The mask
    is additive: 0 attends, -65504 blocks. Uniform values are a no-op, which is
    why an all-ones and an all-zeros mask return the same logits.

    Capacity is `window + max_tokens` slots. Sliding past that runs off the end
    of the allocation, so `advance()` refuses rather than corrupting memory.
    """

    __slots__ = ("layers", "window", "capacity", "pos", "_bufs", "_models")

    def __init__(self, models, layers: int, window: int, max_tokens: int):
        self.layers = layers
        self.window = window
        self.capacity = window + max_tokens
        self.pos = 0
        self._models = list(models)
        self._bufs: list[tuple[DeviceBuffer, int, DeviceBuffer, int]] = []

        probe = self._models[0]
        for i in range(layers):
            row = []
            for kind in ("key", "value"):
                a = probe.inputs[f"layer_{i}_cache_{kind}"]
                slot = int(np.prod(a.shape[2:])) * a.dtype.itemsize
                row += [
                    DeviceBuffer(
                        self.capacity * slot, np.uint8, (self.capacity * slot,)
                    ),
                    slot,
                ]
            self._bufs.append(tuple(row))

        # The per-model cache/new allocations are now dead weight -- release
        # them before rebinding, or the window doubles peak memory.
        for m in self._models:
            for i in range(layers):
                for kind in ("key", "value"):
                    m._in_bufs[m._index[f"layer_{i}_cache_{kind}"]].free()
                    m._out_bufs[m._oindex[f"layer_{i}_new_{kind}"]].free()
        self.reset()

    def reset(self) -> None:
        """Zero the whole allocation and rewind to position 0."""
        self.pos = 0
        for kb, _, vb, _ in self._bufs:
            kb.array[:] = 0
            vb.array[:] = 0
            kb.clean()
            vb.clean()
        self.bind(self._models[0], 1)

    def bind(self, model: Model, span: int) -> None:
        """Point `model` at the window for a step consuming `span` tokens."""
        if self.pos + self.window + span > self.capacity:
            raise InferError(
                f"KV window too small: a {span}-token step at position "
                f"{self.pos} writes past a capacity of {self.capacity} slots; "
                f"build the window with max_tokens >= {self.pos + span}"
            )
        for i, (kb, kslot, vb, vslot) in enumerate(self._bufs):
            for buf, slot, kind in ((kb, kslot, "key"), (vb, vslot, "value")):
                off = self.pos * slot
                model.rebind(
                    f"layer_{i}_cache_{kind}",
                    hbUCPSysMem(
                        phyAddr=buf.mem.phyAddr + off,
                        virAddr=buf.mem.virAddr + off,
                        memSize=self.window * slot,
                    ),
                )
                out = off + self.window * slot
                model.rebind(
                    f"layer_{i}_new_{kind}",
                    hbUCPSysMem(
                        phyAddr=buf.mem.phyAddr + out,
                        virAddr=buf.mem.virAddr + out,
                        memSize=span * slot,
                    ),
                    output=True,
                )

    def advance(self, span: int) -> None:
        if self.pos + span + self.window > self.capacity:
            raise InferError(
                f"KV window exhausted: {self.pos + span} tokens past a "
                f"capacity of {self.capacity - self.window}"
            )
        self.pos += span

    def mask_range(self, span: int, row: int = 0) -> tuple[int, int]:
        """Half-open column range to leave open for `row` of the current step."""
        return max(0, self.window - span - self.pos), self.window - span + row + 1

    def free(self) -> None:
        for kb, _, vb, _ in self._bufs:
            kb.free()
            vb.free()
        self._bufs.clear()


class Package:
    """An .hbm file, which may hold several named models.

    An LLM artifact holds two: `prefill` and `decode`, sharing one weight set.
    """

    def __init__(self, path: str):
        ensure_l2_config()
        init = _fn("hbDNNInitializeFromFiles", _c_i32, _P(_c_vp), _P(_c_cp), _c_i32)
        names_fn = _fn(
            "hbDNNGetModelNameList", _c_i32, _P(_P(_c_cp)), _P(_c_i32), _c_vp
        )

        self._packed = _c_vp()
        files = (_c_cp * 1)(str(path).encode())
        _check(
            init(ctypes.byref(self._packed), files, 1),
            f"hbDNNInitializeFromFiles({path})",
        )
        names = _P(_c_cp)()
        count = _c_i32()
        _check(
            names_fn(ctypes.byref(names), ctypes.byref(count), self._packed),
            "hbDNNGetModelNameList",
        )
        self.path = str(path)
        self.model_names = [names[i].decode() for i in range(count.value)]

    def model(self, name: str, cores=(0, 1, 2, 3), release_depth: int = 1) -> Model:
        if name not in self.model_names:
            raise KeyError(
                f"{self.path} has no model {name!r}; available: {self.model_names}"
            )
        get_handle = _fn("hbDNNGetModelHandle", _c_i32, _P(_c_vp), _c_vp, _c_cp)
        h = _c_vp()
        _check(
            get_handle(ctypes.byref(h), self._packed, name.encode()),
            f"hbDNNGetModelHandle({name})",
        )
        return Model(h, name, cores, release_depth)

    def release(self) -> None:
        if self._packed:
            _fn("hbDNNRelease", _c_i32, _c_vp)(self._packed)
            self._packed = None
