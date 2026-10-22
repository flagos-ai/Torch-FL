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

"""Tensor-level wrapper over hbm_runtime.

Bridges torch.Tensor and the numpy-based hbm_runtime.run() interface, applying
the quantization parameters baked into the compiled artifact.

The hbm_runtime API surface used here (verified against runtime 3.13.6 / HBRT
4.7.5): model metadata is exposed as *properties* keyed by model name, not
methods, and QuantParams carries (axis, quant_type, scale, zero_point).

A tensor already on the flagos device does not need a device-to-host copy to
become a numpy array: UCP memory is mapped into this process, so `data_ptr()` is
host-dereferenceable and a numpy array can wrap it in place. See `_device_view`.
"""

from __future__ import annotations

import ctypes
import logging

import numpy as np
import torch

log = logging.getLogger("torch_fl.bpu")

# torch dtype -> (ctypes scalar, numpy dtype), for wrapping device storage in a
# numpy array without copying. Only fixed-width scalar types the BPU can carry.
_CTYPES_OF = {
    torch.float32: (ctypes.c_float, np.float32),
    torch.float16: (ctypes.c_uint16, np.float16),  # no ctypes half; reinterpret
    torch.int8: (ctypes.c_int8, np.int8),
    torch.uint8: (ctypes.c_uint8, np.uint8),
    torch.int16: (ctypes.c_int16, np.int16),
    torch.int32: (ctypes.c_int32, np.int32),
    torch.int64: (ctypes.c_int64, np.int64),
    torch.bool: (ctypes.c_bool, np.bool_),
}


def _device_view(t: torch.Tensor) -> np.ndarray | None:
    """Zero-copy numpy view over a flagos tensor's storage, or None.

    UCP device memory is host-mapped (hbUCPMallocCached hands back a writable
    `virAddr`), so the storage can be read and written in place rather than
    copied to the host first. Returns None whenever a view would be wrong --
    a non-flagos tensor, a non-contiguous layout, or a dtype with no fixed-width
    ctypes spelling -- and the caller falls back to `.cpu()`.

    The returned array borrows `t`'s storage and must not outlive it. Every
    caller here uses it within one `__call__`, where the inputs are held alive by
    the argument list.
    """
    if t.device.type != "flagos" or not t.is_contiguous() or t.numel() == 0:
        return None
    spec = _CTYPES_OF.get(t.dtype)
    if spec is None:
        return None
    ctype, np_dtype = spec
    buf = (ctype * t.numel()).from_address(t.data_ptr())
    arr = np.frombuffer(buf, dtype=np_dtype, count=t.numel())
    return arr.reshape(tuple(t.shape))


def _as_numpy(t: torch.Tensor) -> np.ndarray:
    """Read a tensor as numpy, without a device-to-host copy where possible."""
    t = t.detach()
    view = _device_view(t)
    if view is not None:
        return view
    return t.contiguous().cpu().numpy()


# hbDNNDataType enum name -> numpy dtype.
_NP_DTYPE = {
    "F32": np.float32,
    "F16": np.float16,
    "S8": np.int8,
    "U8": np.uint8,
    "S16": np.int16,
    "U16": np.uint16,
    "S32": np.int32,
    "U32": np.uint32,
    "S64": np.int64,
    "BOOL8": np.bool_,
}


def _dtype_of(enum_val) -> np.dtype:
    """Map an hbDNNDataType enum member to a numpy dtype."""
    name = getattr(enum_val, "name", str(enum_val).rsplit(".", 1)[-1])
    return _NP_DTYPE.get(name, np.float32)


def _scale_array(quant) -> np.ndarray | None:
    """Extract a quantization scale as an array, or None if unquantized."""
    if quant is None:
        return None
    scale = getattr(quant, "scale", None)
    if scale is None:
        return None
    arr = np.atleast_1d(np.asarray(scale, dtype=np.float32))
    if arr.size == 0 or not np.any(arr):
        return None
    return arr


def _broadcast(scale: np.ndarray, ndim: int, axis: int) -> np.ndarray:
    """Shape a per-channel scale for broadcasting against an ndim tensor."""
    if scale.size == 1:
        return scale.reshape(())
    shape = [1] * ndim
    shape[axis if -ndim <= axis < ndim else -1] = -1
    return scale.reshape(shape)


class BPURuntime:
    """Executes one compiled .hbm graph, taking and returning torch tensors."""

    def __init__(
        self,
        hbm_path: str,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ):
        from hbm_runtime import HB_HBMRuntime

        self._rt = HB_HBMRuntime(str(hbm_path))
        self._hbm_path = str(hbm_path)
        self._model = self._rt.model_names[0]
        m = self._model

        # Trust the artifact's own IO names over the ONNX export names: hbdk4
        # may rename or reorder during compilation.
        self._in_names: list[str] = list(self._rt.input_names[m])
        self._out_names: list[str] = list(self._rt.output_names[m])
        self._in_dtypes = self._rt.input_dtypes[m]
        self._out_dtypes = self._rt.output_dtypes[m]
        self._in_shapes = self._rt.input_shapes[m]
        self._in_quants = self._rt.input_quants[m]
        self._out_quants = self._rt.output_quants[m]

        if input_names is not None and len(input_names) != len(self._in_names):
            log.warning(
                "%s: artifact takes %d input(s) %s, graph supplies %d",
                hbm_path,
                len(self._in_names),
                self._in_names,
                len(input_names),
            )

    # -- conversion ------------------------------------------------------

    def _to_device(self, name: str, t: torch.Tensor) -> np.ndarray:
        """Convert a torch tensor into the artifact's expected array.

        A flagos tensor whose dtype already matches the artifact needs no work at
        all: the array returned by `_as_numpy` *is* its UCP storage. Anything
        that changes dtype or scale (the quantization branch below, or the final
        astype) copies, which is unavoidable -- the artifact wants int8 where the
        graph holds float32.
        """
        arr = _as_numpy(t)
        target = _dtype_of(self._in_dtypes.get(name))

        scale = _scale_array(self._in_quants.get(name))
        if scale is not None and np.issubdtype(target, np.integer):
            quant = self._in_quants[name]
            axis = getattr(quant, "axis", 1) or 1
            zp = getattr(quant, "zero_point", None)
            arr = arr.astype(np.float32) / _broadcast(scale, arr.ndim, axis)
            if zp is not None:
                zp_arr = np.atleast_1d(np.asarray(zp, dtype=np.float32))
                if np.any(zp_arr):
                    arr = arr + _broadcast(zp_arr, arr.ndim, axis)
            info = np.iinfo(target)
            arr = np.clip(np.rint(arr), info.min, info.max)

        # asarray, not astype: astype always copies, so a zero-copy device view
        # whose dtype already matches would be duplicated for nothing.
        return np.ascontiguousarray(np.asarray(arr, dtype=target))

    def _from_device(
        self, name: str, arr: np.ndarray, device: torch.device | None = None
    ) -> torch.Tensor:
        """Dequantize an output array back to a float tensor on `device`.

        The dequantization branch copies because it has arithmetic to do. The
        float path does not: `run()` hands back a freshly allocated buffer on
        every call rather than reusing one, so the array can be wrapped in place.
        `asarray` rather than `astype` for the same reason as `_to_device` --
        astype copies unconditionally, and on a 6 MB output that copy cost more
        than a fifth of the whole inference.
        """
        out = np.asarray(arr, dtype=np.float32)
        scale = _scale_array(self._out_quants.get(name))
        if scale is not None and np.issubdtype(arr.dtype, np.integer):
            quant = self._out_quants[name]
            axis = getattr(quant, "axis", 1) or 1
            zp = getattr(quant, "zero_point", None)
            if zp is not None:
                zp_arr = np.atleast_1d(np.asarray(zp, dtype=np.float32))
                if np.any(zp_arr):
                    out = out - _broadcast(zp_arr, out.ndim, axis)
            out = out * _broadcast(scale, out.ndim, axis)
        t = torch.from_numpy(np.ascontiguousarray(out))
        # Outputs follow the inputs' device, so a partition spliced into a graph
        # of flagos tensors hands flagos tensors to its successors rather than
        # forcing the rest of the graph onto the CPU.
        return t if device is None or device.type == "cpu" else t.to(device)

    # -- execution -------------------------------------------------------

    def __call__(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        if len(inputs) != len(self._in_names):
            raise ValueError(
                f"{self._hbm_path}: expected {len(self._in_names)} input(s) "
                f"{self._in_names}, got {len(inputs)}"
            )
        feed = {
            name: self._to_device(name, t) for name, t in zip(self._in_names, inputs)
        }
        result = self._rt.run(feed)
        out_map = result[self._model]
        device = inputs[0].device if inputs else None
        return [self._from_device(n, out_map[n], device) for n in self._out_names]

    # -- introspection ---------------------------------------------------

    @property
    def input_names(self) -> list[str]:
        return list(self._in_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._out_names)

    def describe(self) -> str:
        """Multi-line summary of the artifact's IO signature."""
        lines = [f"{self._hbm_path} [{self._model}]"]
        for n in self._in_names:
            q = _scale_array(self._in_quants.get(n))
            lines.append(
                f"  in  {n}: {list(self._in_shapes.get(n, []))} "
                f"{_dtype_of(self._in_dtypes.get(n)).__name__}"
                + (f" scale[{q.size}]" if q is not None else "")
            )
        for n in self._out_names:
            q = _scale_array(self._out_quants.get(n))
            lines.append(
                f"  out {n}: "
                f"{_dtype_of(self._out_dtypes.get(n)).__name__}"
                + (f" scale[{q.size}]" if q is not None else "")
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BPURuntime({self._hbm_path!r}, in={self._in_names}, "
            f"out={self._out_names})"
        )
