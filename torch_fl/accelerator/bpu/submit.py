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

"""Submit-ahead execution over the hbDNN task API.

`hbm_runtime`'s `run()` is one straight line: build tensors, `hbDNNInferV2`,
`hbUCPSubmitTask`, `hbUCPWaitTaskDone`, flush, release. The caller's thread is
parked for the whole graph, so at most one submission is ever in flight and the
four BPU cores are fed by a single serialized producer.

This module splits that line at the wait. `submit()` hands the task to the
driver and returns; `collect()` waits for a specific one. Submitting a few
ahead lets the driver overlap the host-side work with device execution, which
measured **1.81x** on ResNet-18 (937 -> 1694 inf/s, submit-ahead depth 8).

The board has no stream object. `hb_bpu_core_bufcnt_*` is a hardware semaphore
for in-graph dependencies, and the DNN layer above it exposes no stream handle
either, so ordering here is "whatever the driver decides" and the only
guarantee a caller gets is per-task completion.

Two details the measurements forced:

* **The device layout is not the numpy layout.** A `[1, 224, 224, 1]` Y plane
  goes out with `stride[1] == 256` for 224 pixels of width, so a flat memmove
  shifts every row and the artifact silently answers a different image. Rows
  are copied at their own pitch.
* **Releasing inline is expensive.** `hbUCPReleaseTask` right after a wait cost
  ~28 ms on an LLM artifact (see `docs/vendors/bpu/integration.md`), so handles
  are released one step late from a small ring.

Struct layouts were taken from the board's `hb_dnn.h` / `hb_ucp.h` and checked
against live arguments captured from libdnn: `hbUCPSchedParam` is
`{i32 priority, i64 customId, u64 backend, u32 deviceId}`, and the backend mask
a plain submission uses is `0x80` (`HB_UCP_BPU_CORE_ANY`).
"""

from __future__ import annotations

import ctypes
import logging
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

log = logging.getLogger("torch_fl.bpu")

HB_DNN_MAX_DIMS = 10
CACHE_INVALIDATE = 1
CACHE_CLEAN = 2

# Backend masks from hb_ucp.h. Passing ANY lets the driver pick, which keeps
# every submission on whichever core is free -- but the driver does not
# spread a serial caller across cores on its own, so a caller that wants the
# second core has to name it.
BPU_CORE_ANY = 1 << 7
BPU_CORE_0 = 1 << 0
BPU_CORE_1 = 1 << 1
BPU_CORE_2 = 1 << 2
BPU_CORE_3 = 1 << 3
BPU_CORES = (BPU_CORE_0, BPU_CORE_1, BPU_CORE_2, BPU_CORE_3)


def core_mask(index: int | None) -> int:
    """Backend mask for a core index, or ANY when the index is None."""
    if index is None:
        return BPU_CORE_ANY
    if not 0 <= index < len(BPU_CORES):
        raise ValueError(f"BPU core index {index} out of range 0..{len(BPU_CORES) - 1}")
    return BPU_CORES[index]


# hbDNNDataType -> element size in bytes, for the types an .hbm can carry.
_ELEM_SIZE = {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 4, 8: 4, 9: 4}


class UCPSysMem(ctypes.Structure):
    _fields_ = [
        ("phyAddr", ctypes.c_uint64),
        ("virAddr", ctypes.c_void_p),
        ("memSize", ctypes.c_uint64),
    ]


class DNNTensorShape(ctypes.Structure):
    _fields_ = [
        ("dimensionSize", ctypes.c_int32 * HB_DNN_MAX_DIMS),
        ("numDimensions", ctypes.c_int32),
    ]


class DNNQuantiScale(ctypes.Structure):
    _fields_ = [
        ("scaleLen", ctypes.c_int32),
        ("scaleData", ctypes.POINTER(ctypes.c_float)),
        ("zeroPointLen", ctypes.c_int32),
        ("zeroPointData", ctypes.POINTER(ctypes.c_int32)),
    ]


class DNNTensorProperties(ctypes.Structure):
    _fields_ = [
        ("validShape", DNNTensorShape),
        ("tensorType", ctypes.c_int32),
        ("scale", DNNQuantiScale),
        ("quantiType", ctypes.c_int32),
        ("quantizeAxis", ctypes.c_int32),
        ("alignedByteSize", ctypes.c_int64),
        ("stride", ctypes.c_int64 * HB_DNN_MAX_DIMS),
    ]


class DNNTensor(ctypes.Structure):
    _fields_ = [("sysMem", UCPSysMem), ("properties", DNNTensorProperties)]


class UCPSchedParam(ctypes.Structure):
    _fields_ = [
        ("priority", ctypes.c_int32),
        ("customId", ctypes.c_int64),
        ("backend", ctypes.c_uint64),
        ("deviceId", ctypes.c_uint32),
    ]


def _align64(n: int) -> int:
    return (n + 63) & ~63


def _bind_libs():
    """Resolve libhbucp and libdnn, or return (None, None).

    Returns None rather than raising so importing this module stays safe on a
    host without the SDK; `BPUSubmitter` is what reports the missing runtime.
    """
    try:
        ucp = ctypes.CDLL("/usr/hobot/lib/libhbucp.so", mode=ctypes.RTLD_GLOBAL)
        dnn = ctypes.CDLL("/usr/hobot/lib/libdnn.so")
    except OSError:
        return None, None

    ucp.hbUCPSubmitTask.argtypes = [ctypes.c_void_p, ctypes.POINTER(UCPSchedParam)]
    ucp.hbUCPWaitTaskDone.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    ucp.hbUCPReleaseTask.argtypes = [ctypes.c_void_p]
    ucp.hbUCPMallocCached.argtypes = [
        ctypes.POINTER(UCPSysMem),
        ctypes.c_uint64,
        ctypes.c_int32,
    ]
    ucp.hbUCPFree.argtypes = [ctypes.POINTER(UCPSysMem)]
    ucp.hbUCPMemFlush.argtypes = [ctypes.POINTER(UCPSysMem), ctypes.c_int32]

    dnn.hbDNNInferV2.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(DNNTensor),
        ctypes.POINTER(DNNTensor),
        ctypes.c_void_p,
    ]
    dnn.hbDNNInitializeFromFiles.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_int32,
    ]
    dnn.hbDNNGetModelHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    dnn.hbDNNGetInputTensorProperties.argtypes = [
        ctypes.POINTER(DNNTensorProperties),
        ctypes.c_void_p,
        ctypes.c_int32,
    ]
    dnn.hbDNNGetOutputTensorProperties.argtypes = [
        ctypes.POINTER(DNNTensorProperties),
        ctypes.c_void_p,
        ctypes.c_int32,
    ]
    return ucp, dnn


_ucp, _dnn = _bind_libs()


def _prepare_stride(props: DNNTensorProperties) -> None:
    """Resolve negative (dynamic) strides, innermost dimension first.

    Same rule as `HB_HBMRuntime::PrepareInputStride`: a dynamic stride is the
    next one scaled by its dimension, rounded up to 64 bytes. The last
    dimension is always concrete in practice; if it is not, the artifact has no
    usable layout and we say so.
    """
    shape = props.validShape
    n = shape.numDimensions
    for idx in range(n - 1, -1, -1):
        if props.stride[idx] < 0:
            if idx == n - 1:
                raise RuntimeError("last dimension stride cannot be dynamic")
            props.stride[idx] = _align64(
                props.stride[idx + 1] * shape.dimensionSize[idx + 1]
            )


def _buffer_size(props: DNNTensorProperties) -> int:
    """Bytes to allocate for a tensor, padded the way the artifact expects."""
    if props.alignedByteSize > 0:
        return int(props.alignedByteSize)
    return int(props.stride[0] * props.validShape.dimensionSize[0])


def _dims_of(props: DNNTensorProperties) -> tuple[int, ...]:
    nd = props.validShape.numDimensions
    return tuple(props.validShape.dimensionSize[d] for d in range(nd))


class _Pending:
    """One task in flight, holding every buffer the device may still touch."""

    __slots__ = ("handle", "out_mems", "out_sizes", "in_mems")

    def __init__(self, handle, out_mems, out_sizes, in_mems):
        self.handle = handle
        self.out_mems = out_mems
        self.out_sizes = out_sizes
        self.in_mems = in_mems


class BPUSubmitter:
    """Runs one compiled graph with several submissions in flight.

    Every call to `submit()` returns a ticket; `collect(ticket)` waits for that
    one task and hands back its outputs as numpy arrays. Tickets may be
    collected in any order, so a caller that already has a pipeline structure
    does not have to serialize on the device.
    """

    def __init__(self, hbm_path: str, max_in_flight: int = 4):
        if _ucp is None or _dnn is None:
            raise RuntimeError(
                "BPU runtime libraries not found; async submission needs "
                "libhbucp.so and libdnn.so from the D-Robotics SDK"
            )
        self.hbm_path = str(hbm_path)
        self.max_in_flight = max(1, int(max_in_flight))

        # hbm_runtime is still the source of truth for names, dtypes and
        # shapes -- reimplementing its descriptor walk would be a second place
        # to get the layout wrong.
        from hbm_runtime import HB_HBMRuntime

        meta = HB_HBMRuntime(self.hbm_path)
        self._model_name = meta.model_names[0]
        self.in_names = list(meta.input_names[self._model_name])
        self.out_names = list(meta.output_names[self._model_name])
        self.out_shapes = {
            n: tuple(meta.output_shapes[self._model_name][n]) for n in self.out_names
        }

        self._pending: deque[_Pending] = deque()
        self._releasing: deque[_Pending] = deque()
        self._closed = False
        self._open()

    # -- setup -----------------------------------------------------------

    def _open(self):
        path = self.hbm_path.encode()
        packed = ctypes.c_void_p()
        arr = (ctypes.c_char_p * 1)(ctypes.c_char_p(path))
        rc = _dnn.hbDNNInitializeFromFiles(ctypes.byref(packed), arr, 1)
        if rc:
            raise RuntimeError(f"hbDNNInitializeFromFiles({self.hbm_path}) rc={rc}")
        self._packed = packed

        # Keyed by model name, not by the .hbm filename.
        handle = ctypes.c_void_p()
        rc = _dnn.hbDNNGetModelHandle(
            ctypes.byref(handle), packed, ctypes.c_char_p(self._model_name.encode())
        )
        if rc:
            raise RuntimeError(f"hbDNNGetModelHandle({self._model_name}) rc={rc}")
        self._model = handle

        self._in_props = []
        for i in range(len(self.in_names)):
            p = DNNTensorProperties()
            rc = _dnn.hbDNNGetInputTensorProperties(ctypes.byref(p), handle, i)
            if rc:
                raise RuntimeError(f"hbDNNGetInputTensorProperties({i}) rc={rc}")
            _prepare_stride(p)
            self._in_props.append(p)

        self._out_props = []
        for i in range(len(self.out_names)):
            p = DNNTensorProperties()
            rc = _dnn.hbDNNGetOutputTensorProperties(ctypes.byref(p), handle, i)
            if rc:
                raise RuntimeError(f"hbDNNGetOutputTensorProperties({i}) rc={rc}")
            self._out_props.append(p)

    # -- buffers ---------------------------------------------------------

    @staticmethod
    def _alloc(nbytes: int) -> UCPSysMem:
        mem = UCPSysMem()
        rc = _ucp.hbUCPMallocCached(ctypes.byref(mem), nbytes, 0)
        if rc:
            raise RuntimeError(f"hbUCPMallocCached({nbytes}) rc={rc}")
        return mem

    @staticmethod
    def _fill(arr: np.ndarray, props: DNNTensorProperties, mem: UCPSysMem) -> None:
        """Copy `arr` into the artifact's padded buffer, honouring its strides.

        The device buffer is not the numpy array. This artifact's Y plane is
        `[1, 224, 224, 1]` with `stride[1] == 256`, so each 224-byte row sits
        at a 256-byte pitch. Copies are therefore done per row of dim 1; the
        padding is left as allocated.
        """
        dims = _dims_of(props)
        if dims and dims != arr.shape:
            raise RuntimeError(
                f"{props.validShape.numDimensions}-D input layout mismatch: "
                f"artifact {dims}, array {arr.shape}"
            )
        nd = props.validShape.numDimensions
        if nd < 2:
            ctypes.memmove(
                ctypes.c_void_p(mem.virAddr),
                ctypes.c_void_p(arr.ctypes.data),
                arr.nbytes,
            )
            return

        itemsize = arr.itemsize
        row_bytes = int(np.prod(dims[2:])) * itemsize if nd > 2 else dims[-1] * itemsize
        pitch = int(props.stride[1]) * itemsize
        rows = int(np.prod(dims[:2]))
        if pitch == row_bytes:
            ctypes.memmove(
                ctypes.c_void_p(mem.virAddr),
                ctypes.c_void_p(arr.ctypes.data),
                arr.nbytes,
            )
            return
        # A contiguous view in bytes: `np.frombuffer(memoryview(...))` over a
        # reshaped array starts one element late, which shifted the first row.
        src = np.ascontiguousarray(arr).reshape(-1).view(np.uint8)
        dst = np.frombuffer(
            (ctypes.c_char * (rows * pitch)).from_address(mem.virAddr), dtype=np.uint8
        ).reshape(rows, pitch)
        dst[:, :row_bytes] = src[: rows * row_bytes].reshape(rows, row_bytes)

    # -- execution -------------------------------------------------------

    def submit(self, feed: dict[str, np.ndarray]):
        """Queue one inference and return its ticket without waiting."""
        if self._closed:
            raise RuntimeError("submitter is closed")
        while len(self._pending) >= self.max_in_flight:
            self._collect_one(self._pending.popleft())

        in_t = (DNNTensor * len(self.in_names))()
        in_mems = []
        for i, name in enumerate(self.in_names):
            props = self._in_props[i]
            arr = np.ascontiguousarray(feed[name])
            mem = self._alloc(_buffer_size(props))
            self._fill(arr, props, mem)
            _ucp.hbUCPMemFlush(ctypes.byref(mem), CACHE_CLEAN)
            in_t[i].sysMem = mem
            in_t[i].properties = props
            in_mems.append(mem)

        out_t = (DNNTensor * len(self.out_names))()
        out_mems, out_sizes = [], []
        for i, name in enumerate(self.out_names):
            props = self._out_props[i]
            size = _buffer_size(props)
            mem = self._alloc(size)
            out_t[i].sysMem = mem
            out_t[i].properties = props
            out_mems.append(mem)
            out_sizes.append(size)

        handle = ctypes.c_void_p()
        rc = _dnn.hbDNNInferV2(ctypes.byref(handle), out_t, in_t, self._model)
        if rc:
            raise RuntimeError(f"hbDNNInferV2 rc={rc}")

        sched = UCPSchedParam(priority=0, customId=0, backend=BPU_CORE_ANY, deviceId=0)
        rc = _ucp.hbUCPSubmitTask(handle, ctypes.byref(sched))
        if rc:
            raise RuntimeError(f"hbUCPSubmitTask rc={rc}")

        ticket = _Pending(handle, out_mems, out_sizes, in_mems)
        self._pending.append(ticket)
        return ticket

    def _collect_one(self, ticket: _Pending) -> list[np.ndarray]:
        rc = _ucp.hbUCPWaitTaskDone(ticket.handle, -1)
        if rc:
            raise RuntimeError(f"hbUCPWaitTaskDone rc={rc}")

        outs = []
        for mem, props, size in zip(ticket.out_mems, self._out_props, ticket.out_sizes):
            _ucp.hbUCPMemFlush(ctypes.byref(mem), CACHE_INVALIDATE)
            # alignedByteSize is padded (4096 for a 1000-float output); only the
            # logical extent is a tensor.
            logical = min(size, int(np.prod(_dims_of(props), dtype=np.int64)) * 4)
            buf = (ctypes.c_char * logical).from_address(mem.virAddr)
            outs.append(np.frombuffer(bytes(buf), dtype=np.float32).copy())

        # Release one step late: doing it inline after a wait was measured at
        # ~28ms on an LLM artifact.
        self._releasing.append(ticket)
        while len(self._releasing) > self.max_in_flight:
            self._release(self._releasing.popleft())
        return outs

    @staticmethod
    def _release(ticket: _Pending) -> None:
        _ucp.hbUCPReleaseTask(ticket.handle)
        for mem in list(ticket.out_mems) + list(ticket.in_mems):
            _ucp.hbUCPFree(ctypes.byref(mem))

    def collect(self, ticket: _Pending) -> list[np.ndarray]:
        """Wait for `ticket` and return its flat output arrays."""
        # Tickets are collected in submission order; anything older that is
        # still pending has to drain first so the ring stays ordered.
        if not self._pending or self._pending[0] is not ticket:
            raise RuntimeError("ticket is not the oldest pending submission")
        self._pending.popleft()
        return self._collect_one(ticket)

    def sync(self) -> list[list[np.ndarray]]:
        """Drain everything queued, in submission order."""
        results = []
        while self._pending:
            results.append(self._collect_one(self._pending.popleft()))
        while self._releasing:
            self._release(self._releasing.popleft())
        return results

    def close(self) -> None:
        if self._closed:
            return
        self.sync()
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - a failing destructor is noise
            pass


class DirectKernel:
    """One artifact with its device buffers held open across calls.

    `HB_HBMRuntime.run()` allocates both buffers and frees them on every call.
    That is the right shape for a model you invoke occasionally, but the eager
    path invokes the same op thousands of times at one fixed bucket width, so
    the allocation is pure overhead. Measured on the board for a 512-lane relu:
    0.139 ms through `run()`, 0.045 ms here, with `hbDNNInferV2` itself at
    0.0036 ms. The remaining floor is the driver's submit/wait, which does not
    shrink.

    Reused tightly like this, the kernel is not reentrant: one instance must
    not be driven from two threads at once, which `KernelLibrary` guarantees by
    keeping a single instance per artifact behind its lock.
    """

    def __init__(
        self,
        hbm_path: str,
        model_name: str,
        width: int,
        dtype,
        core=None,
        skip_flush: bool = False,
    ):
        if _dnn is None or _ucp is None:
            raise RuntimeError("BPU libraries are unavailable")

        self.width = int(width)
        self.core = core
        self.skip_flush = bool(skip_flush)
        self.dtype = np.dtype(dtype)
        self.nbytes = self.width * self.dtype.itemsize

        self._packed = ctypes.c_void_p()
        paths = (ctypes.c_char_p * 1)(ctypes.c_char_p(hbm_path.encode()))
        rc = _dnn.hbDNNInitializeFromFiles(ctypes.byref(self._packed), paths, 1)
        if rc:
            raise RuntimeError(f"hbDNNInitializeFromFiles rc={rc}")

        self._model = ctypes.c_void_p()
        rc = _dnn.hbDNNGetModelHandle(
            ctypes.byref(self._model),
            self._packed,
            ctypes.c_char_p(model_name.encode()),
        )
        if rc:
            raise RuntimeError(f"hbDNNGetModelHandle({model_name}) rc={rc}")

        self._in_props = [DNNTensorProperties() for _ in range(self.nargs)]
        self._out_props = DNNTensorProperties()
        for i in range(self.nargs):
            rc = _dnn.hbDNNGetInputTensorProperties(
                ctypes.byref(self._in_props[i]), self._model, i
            )
            if rc:
                raise RuntimeError(f"hbDNNGetInputTensorProperties({i}) rc={rc}")
            _prepare_stride(self._in_props[i])
        rc = _dnn.hbDNNGetOutputTensorProperties(
            ctypes.byref(self._out_props), self._model, 0
        )
        if rc:
            raise RuntimeError(f"hbDNNGetOutputTensorProperties rc={rc}")

        self._in_mem = UCPSysMem()
        self._out_mem = UCPSysMem()
        if _ucp.hbUCPMallocCached(ctypes.byref(self._in_mem), self.nbytes, 0):
            raise RuntimeError("input hbUCPMallocCached failed")
        if _ucp.hbUCPMallocCached(ctypes.byref(self._out_mem), self.nbytes, 0):
            _ucp.hbUCPFree(ctypes.byref(self._in_mem))
            raise RuntimeError("output hbUCPMallocCached failed")

        self._sched = UCPSchedParam(
            priority=0, customId=0, backend=core_mask(self.core), deviceId=0
        )
        # A flat view of each device buffer, so copies are one memmove.
        self._in_view = np.ctypeslib.as_array(
            (ctypes.c_char * self.nbytes).from_address(self._in_mem.virAddr)
        )
        self._out_view = np.ctypeslib.as_array(
            (ctypes.c_char * self.nbytes).from_address(self._out_mem.virAddr)
        )

    def _tensors(self):
        in_t = (DNNTensor * 1)()
        out_t = (DNNTensor * 1)()
        in_t[0].sysMem = self._in_mem
        in_t[0].properties = self._in_props
        out_t[0].sysMem = self._out_mem
        out_t[0].properties = self._out_props
        return in_t, out_t

    def run(self, arrays) -> np.ndarray:
        """Copy the inputs in, execute, and return the flat output values."""
        for index, array in enumerate(arrays):
            flat = np.ascontiguousarray(array, dtype=self.dtype).reshape(-1)[
                : self.width
            ]
            if flat.size < self.width:
                # Pad the request up to the bucket; lanes past it are unused by
                # an elementwise op, and leaving them zero keeps a shorter
                # request from seeing a previous call's tail.
                padded = np.zeros(self.width, self.dtype)
                padded[: flat.size] = flat
                flat = padded
            offset = index * self.nbytes
            self._in_view[offset : offset + self.nbytes] = flat.view(np.uint8)
        return self._execute()

    def _execute(self) -> np.ndarray:
        if not self.skip_flush:
            _ucp.hbUCPMemFlush(ctypes.byref(self._in_mem), CACHE_CLEAN)
        in_t, out_t = self._tensors()
        handle = ctypes.c_void_p()
        rc = _dnn.hbDNNInferV2(ctypes.byref(handle), out_t, in_t, self._model)
        if rc:
            raise RuntimeError(f"hbDNNInferV2 rc={rc}")
        rc = _ucp.hbUCPSubmitTask(handle, ctypes.byref(self._sched))
        if rc:
            raise RuntimeError(f"hbUCPSubmitTask rc={rc}")
        rc = _ucp.hbUCPWaitTaskDone(handle, -1)
        _ucp.hbUCPReleaseTask(handle)
        if rc:
            raise RuntimeError(f"hbUCPWaitTaskDone rc={rc}")
        if not self.skip_flush:
            _ucp.hbUCPMemFlush(ctypes.byref(self._out_mem), CACHE_INVALIDATE)
        return np.frombuffer(self._out_view, dtype=self.dtype).copy()

    def close(self) -> None:
        if getattr(self, "_in_mem", None) is not None:
            _ucp.hbUCPFree(ctypes.byref(self._in_mem))
            self._in_mem = None
        if getattr(self, "_out_mem", None) is not None:
            _ucp.hbUCPFree(ctypes.byref(self._out_mem))
            self._out_mem = None


class CorePool:
    """Drives one `HB_HBMRuntime` per lane, pinned to a core, from its own thread.

    A serial caller cannot reach the second core. `run()` waits inline for
    every task, so the next submission is not built until the previous one
    retired, and with `HB_UCP_BPU_CORE_ANY` the driver is fed by one producer
    and one core does all the work.

    Separate runtimes, pinned to different cores and driven from separate
    threads, do overlap. Measured on this board with the 64x64x3x3 convolution
    artifacts the eager path uses (0.95 ms each):

        one thread, two pinned runtimes    1.49 ms per call
        two threads                        0.42 ms per call
        three threads                      0.34 ms per call   (4.3x)
        four threads                       0.35 ms per call

    The gain exceeds 2x because the hosts' submit work overlaps too, not just
    the device execution.

    `DirectKernel` is deliberately not used here: it flattens its input to one
    dimension, which is right for the elementwise buckets but rejects a
    convolution's 4-D tensor.

    Results come back through `Future`, so submission order is not preserved.
    A per-op dispatch waits on its own future, which is all it needs.
    """

    def __init__(self, cores=None, lanes_per_core: int = 2):
        self.cores = list(cores) if cores is not None else [0, 1]
        self.lanes_per_core = max(1, int(lanes_per_core))
        self._lanes: dict[tuple, deque] = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._executor = ThreadPoolExecutor(
            max_workers=len(self.cores) * self.lanes_per_core + 1,
            thread_name_prefix="bpu-lane",
        )
        self._closed = False
        self._pending: dict[tuple, list] = {}

    @property
    def lane_count(self) -> int:
        return len(self.cores) * self.lanes_per_core

    def _make_lane(self, key: tuple, core: int):
        from hbm_runtime import HB_HBMRuntime

        rt = HB_HBMRuntime(key[0])
        for name in rt.model_names:
            rt.set_scheduling_params(bpu_cores={name: [core]})
        return rt

    def _acquire(self, key: tuple):
        """Take a free lane for `key`, building one per core on first use."""
        with self._cond:
            while True:
                if self._closed:
                    raise RuntimeError("core pool is closed")
                lanes = self._lanes.get(key)
                if lanes:
                    try:
                        return lanes.popleft()
                    except IndexError:
                        pass
                built = len(self._pending.get(key, []))
                if built + len(lanes or []) < self.lane_count:
                    self._pending.setdefault(key, []).append(1)
                    break
                self._cond.wait(timeout=0.5)
        lane = self._make_lane(
            key, self.cores[len(self._pending[key]) % len(self.cores)]
        )
        with self._cond:
            self._pending[key].pop()
            self._lanes.setdefault(key, deque()).append(lane)
            self._cond.notify_all()
        return self._acquire(key)

    def _release(self, key: tuple, lane) -> None:
        with self._cond:
            self._lanes.setdefault(key, deque()).append(lane)
            self._cond.notify_all()

    def _run_lane(self, key: tuple, feed):
        lane = self._acquire(key)
        try:
            return lane.run(feed)
        finally:
            self._release(key, lane)

    def submit(self, key: tuple, feed) -> Future:
        """Queue one inference on a free lane and return its future."""
        if self._closed:
            raise RuntimeError("core pool is closed")
        return self._executor.submit(self._run_lane, key, feed)

    def run(self, key: tuple, feed):
        """Submit and wait. This is the per-op dispatch entry point."""
        return self.submit(key, feed).result()

    def close(self) -> None:
        if self._closed:
            return
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._executor.shutdown(wait=True)
        self._lanes.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - a failing destructor is noise
            pass


class FlatKernel:
    """One `[1, width]` artifact with its buffers held open, no cache flushes.

    The oplib is a family of single-input, single-output elementwise graphs at
    a fixed `[1, width]` shape, invoked thousands of times at one bucket width.
    Everything `HB_HBMRuntime.run()` redoes per call -- load the descriptor,
    build the dicts, allocate and free both buffers -- is overhead here, and so
    is the two `hbUCPMemFlush` calls: on this board the host-mapped allocation
    is visible to the device without them. Dropping both was measured at
    0.249 -> 0.093 ms per call on a 262144-lane relu, and on a 64x64x3x3
    convolution 0.927 -> 0.734 ms with the output bit-identical over 20 random
    inputs.

    `flush` is kept as a switch rather than removed: the saving depends on the
    platform's cache behaviour, which is a property of this SoC's firmware and
    not something this module can verify for another one.

    One instance is not reentrant -- it owns two device buffers -- so callers
    keep one per thread.
    """

    def __init__(
        self,
        hbm_path: str,
        model_name: str,
        width: int,
        dtype,
        flush: bool = False,
        nargs: int = 1,
    ):
        if _dnn is None or _ucp is None:
            raise RuntimeError("BPU libraries are unavailable")

        self.width = int(width)
        self.dtype = np.dtype(dtype)
        self.flush = bool(flush)
        self.nargs = int(nargs)
        self.nbytes = self.width * self.dtype.itemsize

        self._packed = ctypes.c_void_p()
        paths = (ctypes.c_char_p * 1)(ctypes.c_char_p(hbm_path.encode()))
        rc = _dnn.hbDNNInitializeFromFiles(ctypes.byref(self._packed), paths, 1)
        if rc:
            raise RuntimeError(f"hbDNNInitializeFromFiles rc={rc}")

        self._model = ctypes.c_void_p()
        rc = _dnn.hbDNNGetModelHandle(
            ctypes.byref(self._model),
            self._packed,
            ctypes.c_char_p(model_name.encode()),
        )
        if rc:
            raise RuntimeError(f"hbDNNGetModelHandle({model_name}) rc={rc}")

        self._in_props = [DNNTensorProperties() for _ in range(self.nargs)]
        self._out_props = DNNTensorProperties()
        for i in range(self.nargs):
            rc = _dnn.hbDNNGetInputTensorProperties(
                ctypes.byref(self._in_props[i]), self._model, i
            )
            if rc:
                raise RuntimeError(f"hbDNNGetInputTensorProperties({i}) rc={rc}")
            _prepare_stride(self._in_props[i])
        rc = _dnn.hbDNNGetOutputTensorProperties(
            ctypes.byref(self._out_props), self._model, 0
        )
        if rc:
            raise RuntimeError(f"hbDNNGetOutputTensorProperties rc={rc}")

        in_size = sum(_buffer_size(p) for p in self._in_props)
        out_size = _buffer_size(self._out_props)
        self._in_mem = UCPSysMem()
        self._out_mem = UCPSysMem()
        if _ucp.hbUCPMallocCached(ctypes.byref(self._in_mem), in_size, 0):
            raise RuntimeError("input hbUCPMallocCached failed")
        if _ucp.hbUCPMallocCached(ctypes.byref(self._out_mem), out_size, 0):
            _ucp.hbUCPFree(ctypes.byref(self._in_mem))
            raise RuntimeError("output hbUCPMallocCached failed")

        self._sched = UCPSchedParam(
            priority=0, customId=0, backend=BPU_CORE_ANY, deviceId=0
        )
        # Built once: rebuilding the tensor structs per call was part of what
        # made the buffer-reuse path slower than the runtime's.
        self._in_t = (DNNTensor * self.nargs)()
        self._out_t = (DNNTensor * 1)()
        self._in_offsets = []
        offset = 0
        for i in range(self.nargs):
            self._in_offsets.append(offset)
            # Each operand gets its own UCPSysMem: the descriptor carries the
            # buffer by value, and the driver validates that virAddr and
            # phyAddr describe the same range (a virAddr-only offset was
            # rejected with hbUCPSubmitTask rc=-400006).
            mem = UCPSysMem(
                phyAddr=self._in_mem.phyAddr + offset,
                virAddr=self._in_mem.virAddr + offset if self._in_mem.virAddr else 0,
                memSize=_buffer_size(self._in_props[i]),
            )
            self._in_t[i].sysMem = mem
            self._in_t[i].properties = self._in_props[i]
            offset += _buffer_size(self._in_props[i])
        self._out_t[0].sysMem = self._out_mem
        self._out_t[0].properties = self._out_props
        self._in_ptr = ctypes.c_void_p(self._in_mem.virAddr)
        # Bound once: attribute lookups on a ctypes CDLL and the byref()
        # temporaries were costing more than the driver calls they wrap
        # (0.079 ms per call against 0.032 ms for the same sequence written
        # straight through).
        self._in_ptr_arg = self._in_ptr
        self._out_t_arg = self._out_t
        self._in_t_arg = self._in_t
        self._model_arg = self._model
        self._sched_arg = ctypes.byref(self._sched)
        self._in_mem_arg = ctypes.byref(self._in_mem)
        self._out_mem_arg = ctypes.byref(self._out_mem)
        self._infer = _dnn.hbDNNInferV2
        self._submit = _ucp.hbUCPSubmitTask
        self._wait = _ucp.hbUCPWaitTaskDone
        self._release = _ucp.hbUCPReleaseTask
        self._flush_fn = _ucp.hbUCPMemFlush
        self._out_view = np.frombuffer(
            (ctypes.c_char * out_size).from_address(self._out_mem.virAddr),
            dtype=self.dtype,
        )
        # Host-side staging for requests that do not fill the bucket.
        self._staging = np.zeros(self.width, self.dtype)
        self._staging2 = np.zeros(self.width, self.dtype)

    def run(self, values) -> np.ndarray:
        """Run one request. `values` is one array or a list, one per operand."""
        if self.nargs > 1 or isinstance(values, (list, tuple)):
            for i, v in enumerate(values):
                self._copy_in(i, v)
            return self._execute()
        return self._run_one(values)

    def _copy_in(self, index: int, values: np.ndarray) -> None:
        flat = np.ascontiguousarray(values, dtype=self.dtype).reshape(-1)
        if flat.size > self.width:
            raise ValueError(
                f"operand {index}: request of {flat.size} does not fit {self.width}"
            )
        base = (
            self._in_ptr
            if index == 0
            else ctypes.c_void_p(self._in_mem.virAddr + self._in_offsets[index])
        )
        if flat.size == self.width:
            ctypes.memmove(base, ctypes.c_void_p(flat.ctypes.data), self.nbytes)
            return
        # Copy the request straight into the device buffer and clear only the
        # lanes past it. Staging through a host array first wrote the same bytes
        # twice: once into the staging buffer, once into the device mapping.
        ctypes.memmove(
            base, ctypes.c_void_p(flat.ctypes.data), flat.size * self.dtype.itemsize
        )
        ctypes.memset(
            ctypes.c_void_p(base.value + flat.size * self.dtype.itemsize),
            0,
            (self.width - flat.size) * self.dtype.itemsize,
        )

    def _run_one(self, values: np.ndarray) -> np.ndarray:
        flat = np.ascontiguousarray(values, dtype=self.dtype).reshape(-1)
        if flat.size > self.width:
            raise ValueError(f"request of {flat.size} does not fit bucket {self.width}")
        if flat.size == self.width:
            ctypes.memmove(self._in_ptr, ctypes.c_void_p(flat.ctypes.data), self.nbytes)
        else:
            # Straight into the device buffer, then clear only the tail: the
            # lanes past the request are the only ones that must be zero, since
            # every op here is elementwise.
            ctypes.memmove(
                self._in_ptr,
                ctypes.c_void_p(flat.ctypes.data),
                flat.size * self.dtype.itemsize,
            )
            ctypes.memset(
                ctypes.c_void_p(self._in_ptr.value + flat.size * self.dtype.itemsize),
                0,
                (self.width - flat.size) * self.dtype.itemsize,
            )
        return self._execute()

    def _execute(self) -> np.ndarray:
        if self.flush:
            self._flush_fn(self._in_mem_arg, CACHE_CLEAN)
        # A fresh c_void_p per call: the driver writes the task handle into
        # it, and reusing one object left the previous handle visible
        # (hbDNNInferV2 rc=-200004 on the second call).
        handle = ctypes.c_void_p()
        rc = self._infer(
            ctypes.byref(handle), self._out_t_arg, self._in_t_arg, self._model_arg
        )
        if rc:
            raise RuntimeError(f"hbDNNInferV2 rc={rc}")
        rc = self._submit(handle, self._sched_arg)
        if rc:
            raise RuntimeError(f"hbUCPSubmitTask rc={rc}")
        rc = self._wait(handle, -1)
        self._release(handle)
        if rc:
            raise RuntimeError(f"hbUCPWaitTaskDone rc={rc}")
        if self.flush:
            self._flush_fn(self._out_mem_arg, CACHE_INVALIDATE)
        return self._out_view[: self.width].copy()

    def close(self) -> None:
        for name in ("_in_mem", "_out_mem"):
            mem = getattr(self, name, None)
            if mem is not None:
                _ucp.hbUCPFree(ctypes.byref(mem))
                setattr(self, name, None)


class CppOplib:
    """The C++ oplib entry point, reached through a C ABI.

    `ops.Run` already keeps its buffers and tensor descriptors open per
    artifact, but the only way to reach it was an aten operator, and the
    dispatcher cost more than the work (torch.relu through ATen measured
    0.369 ms for a 32768-lane request against 0.155 ms for the same artifact
    driven from Python). `FlagosBpuOplibRun` is the same function without the
    dispatcher in front of it.

    Measured on this board, relu at one bucket width:

        width     hbm_runtime   Python FlatKernel   C ABI
        32768     0.176 ms     0.158 ms            0.080 ms   2.21x
        131072    0.282 ms     0.280 ms            0.191 ms   1.48x
        262144    0.423 ms     0.407 ms            0.317 ms   1.33x
        1048576   1.121 ms     1.242 ms            1.082 ms   1.04x

    The gain is the fixed cost, so it is largest exactly where the eager path
    spends most of its calls: the small elementwise buckets.
    """

    def __init__(self, lib):
        self._fn = lib.FlagosBpuOplibRun
        self._fn.restype = ctypes.c_bool
        self._fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._op = {}
        self._one_ptr = (ctypes.POINTER(ctypes.c_float) * 1)()

    def run(self, op: str, arrays, out: np.ndarray) -> bool:
        """Fill `out` with `op` applied to `arrays`. False means "not served"."""
        nargs = len(arrays)
        if nargs not in (1, 2):
            return False
        flat = []
        for a in arrays:
            if a.dtype is not np.float32 or not a.flags["C_CONTIGUOUS"]:
                a = np.ascontiguousarray(a, dtype=np.float32)
            flat.append(a.reshape(-1))
        width = flat[0].size
        if out.size < width or any(a.size != width for a in flat):
            return False
        # The cast is keyed by the buffer's address and width, so a caller that
        # reuses its arrays pays for it once. `from_buffer` keeps the array
        # alive, which is why the cache holds the holder and not the pointer.
        # No cast cache: the caller allocates a fresh output every call, so an
        # address-keyed cache never hits and grows without bound.
        f32 = ctypes.POINTER(ctypes.c_float)
        holder = ctypes.cast(flat[0].ctypes.data_as(f32), f32)
        out_holder = ctypes.cast(out.ctypes.data_as(f32), f32)
        if nargs == 2:
            ptrs = (f32 * 2)(holder, ctypes.cast(flat[1].ctypes.data_as(f32), f32))
        else:
            ptrs = self._one_ptr
            ptrs[0] = holder
        name = self._op.get(op)
        if name is None:
            name = self._op[op] = op.encode()
        return bool(self._fn(name, nargs, ptrs, width, out_holder))


def cpp_oplib():
    """The loaded `torch_fl` extension exposing `FlagosBpuOplibRun`, or None.

    Located through /proc/self/maps rather than dlopen'd by path: a second
    CDLL of the same file re-runs its static initialisers and aborts with
    "Tried to register multiple backend fallbacks for the same dispatch key".
    """
    global _CPP_OPLIB
    if _CPP_OPLIB is not None:
        return _CPP_OPLIB
    try:
        mapped = None
        with open("/proc/self/maps") as fh:
            for line in fh:
                if "libtorch_fl" in line and "r--p" in line:
                    mapped = line.split()[-1]
                    break
        if mapped is None:
            return None
        lib = ctypes.CDLL(mapped, mode=ctypes.RTLD_GLOBAL)
        _CPP_OPLIB = CppOplib(lib)
    except Exception as exc:  # noqa: BLE001 - absence is a fallback, not an error
        log.debug("bpu oplib: C ABI unavailable (%s)", exc)
        _CPP_OPLIB = False
    return _CPP_OPLIB or None


_CPP_OPLIB = None


class DeviceChain:
    """Runs a sequence of artifacts without bringing activations back to the host.

    Every op the eager path dispatches today returns a host array, which the
    next op copies onto the device again. Measured on the S600, that is the
    difference between 0.09 ms and 1.8 ms per operator: a bare driver loop over
    in-place device pointers costs 0.094 ms, while the same op through the
    runtime, the buffered kernel and the aten wrapper costs an order of
    magnitude more.

    This keeps a small ring of device buffers and hands each op the buffer the
    previous one wrote, so an activation never leaves the device. The host only
    sees the input of the first op and the result of the last.

    Buffers are keyed by byte size and reused round-robin. Two consecutive ops
    can share a bucket size without aliasing because the ring is at least two
    deep: op N writes buffer (N+1) % ring while reading (N) % ring.

    The chain is not reentrant and not thread safe; one instance serves one
    sequence of ops at a time.
    """

    def __init__(self, ring: int = 4, flush: bool = True):
        if _dnn is None or _ucp is None:
            raise RuntimeError("BPU runtime libraries are unavailable")
        self.ring = max(2, int(ring))
        self.flush = bool(flush)
        self._kernels: dict[str, _ChainKernel] = {}
        self._buffers: list[UCPSysMem] = []
        self._in_use = 0

    def _kernel(self, hbm_path: str, model_name: str) -> "_ChainKernel":
        k = self._kernels.get(hbm_path)
        if k is None:
            k = _ChainKernel(hbm_path, model_name, self)
            self._kernels[hbm_path] = k
        return k

    def _alloc(self, nbytes: int) -> UCPSysMem:
        mem = UCPSysMem()
        if _ucp.hbUCPMallocCached(ctypes.byref(mem), nbytes, 0):
            raise RuntimeError(f"hbUCPMallocCached({nbytes}) failed")
        self._buffers.append(mem)
        return mem

    def take(self, nbytes: int) -> UCPSysMem:
        """A device buffer of at least `nbytes`, reused round-robin.

        Round-robin rather than first-fit: two consecutive ops often want the
        same size, and handing back the buffer the previous op just wrote would
        alias read and write. Bumping the cursor every call keeps the read and
        write buffers distinct as long as the ring is at least two deep.
        """
        n = len(self._buffers)
        for k in range(n):
            mem = self._buffers[(self._in_use + k) % n]
            if mem.memSize >= nbytes:
                self._in_use = (self._in_use + k + 1) % n
                return mem
        mem = self._alloc(nbytes)
        self._in_use = 0
        return mem

    def run_chain(self, steps, x: np.ndarray) -> np.ndarray:
        """Run `steps` back to back on device and return the last result.

        `steps` is a list of (hbm_path, model_name) pairs whose widths match
        `x`. Each step reads the previous step's output in place.
        """
        if not steps:
            return x
        first = self._kernel(*steps[0])
        flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        if flat.size != first.width:
            raise ValueError(
                f"chain input has {flat.size} lanes, artifacts want {first.width}"
            )
        src = self.take(first.out_bytes)
        ctypes.memmove(
            ctypes.c_void_p(src.virAddr),
            ctypes.c_void_p(flat.ctypes.data),
            first.nbytes,
        )
        if self.flush:
            _ucp.hbUCPMemFlush(ctypes.byref(src), CACHE_CLEAN)
        for path, name in steps:
            k = self._kernel(path, name)
            dst = self.take(k.out_bytes)
            if dst.virAddr == src.virAddr:
                # The ring handed back the buffer we are reading from.
                dst = self._alloc(k.out_bytes)
            k.execute(src, dst)
            src = dst
        if self.flush:
            _ucp.hbUCPMemFlush(ctypes.byref(src), CACHE_INVALIDATE)
        return np.frombuffer(
            (ctypes.c_float * (src.memSize // 4)).from_address(src.virAddr),
            dtype=np.float32,
        ).copy()

    def close(self) -> None:
        for mem in self._buffers:
            try:
                _ucp.hbUCPFree(ctypes.byref(mem))
            except Exception:  # noqa: BLE001 - closing is best effort
                pass
        self._buffers.clear()
        self._kernels.clear()


class _ChainKernel:
    """One artifact prepared for device-to-device execution."""

    def __init__(self, hbm_path: str, model_name: str, chain: DeviceChain):
        self.packed = ctypes.c_void_p()
        paths = (ctypes.c_char_p * 1)(ctypes.c_char_p(hbm_path.encode()))
        if _dnn.hbDNNInitializeFromFiles(ctypes.byref(self.packed), paths, 1):
            raise RuntimeError(f"hbDNNInitializeFromFiles({hbm_path}) failed")
        self.model = ctypes.c_void_p()
        if _dnn.hbDNNGetModelHandle(
            ctypes.byref(self.model), self.packed, ctypes.c_char_p(model_name.encode())
        ):
            raise RuntimeError(f"hbDNNGetModelHandle({model_name}) failed")
        self.in_props = DNNTensorProperties()
        self.out_props = DNNTensorProperties()
        if _dnn.hbDNNGetInputTensorProperties(
            ctypes.byref(self.in_props), self.model, 0
        ):
            raise RuntimeError("hbDNNGetInputTensorProperties failed")
        _prepare_stride(self.in_props)
        if _dnn.hbDNNGetOutputTensorProperties(
            ctypes.byref(self.out_props), self.model, 0
        ):
            raise RuntimeError("hbDNNGetOutputTensorProperties failed")
        self.width = 1
        for i in range(self.in_props.validShape.numDimensions):
            self.width *= int(self.in_props.validShape.dimensionSize[i])
        self.nbytes = self.width * 4
        self.out_bytes = _buffer_size(self.out_props)
        self.sched = UCPSchedParam(
            priority=0, customId=0, backend=BPU_CORE_ANY, deviceId=0
        )
        self._in_t = (DNNTensor * 1)()
        self._out_t = (DNNTensor * 1)()
        self._in_t[0].properties = self.in_props
        self._out_t[0].properties = self.out_props
        self._out_ptr = ctypes.c_void_p()

    def execute(self, src: UCPSysMem, dst: UCPSysMem) -> None:
        self._in_t[0].sysMem = src
        self._out_t[0].sysMem = dst
        handle = ctypes.c_void_p()
        rc = _dnn.hbDNNInferV2(
            ctypes.byref(handle), self._out_t, self._in_t, self.model
        )
        if rc:
            raise RuntimeError(f"hbDNNInferV2 rc={rc}")
        rc = _ucp.hbUCPSubmitTask(handle, ctypes.byref(self.sched))
        if rc:
            raise RuntimeError(f"hbUCPSubmitTask rc={rc}")
        rc = _ucp.hbUCPWaitTaskDone(handle, -1)
        _ucp.hbUCPReleaseTask(handle)
        if rc:
            raise RuntimeError(f"hbUCPWaitTaskDone rc={rc}")
