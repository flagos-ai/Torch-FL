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
from collections import deque

import numpy as np

log = logging.getLogger("torch_fl.bpu")

HB_DNN_MAX_DIMS = 10
CACHE_INVALIDATE = 1
CACHE_CLEAN = 2
BPU_CORE_ANY = 0x80

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
