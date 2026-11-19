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

"""CUDA/HIP graph capture for the flagos device.

Two backends, selected at import time by whether ``torch.cuda.CUDAGraph`` is a
*real* binding:

* **Platforms with a real vendor-torch CUDA binding** (MetaX, where torch_fl
  relinks the vendor libtorch_cuda).  ``torch._C._CUDAGraph`` is a real
  pybind class, so we reuse ``torch.cuda.CUDAGraph`` / ``torch.cuda.graph``
  unchanged.  The flagos device's kernels are submitted through that same
  binding's capture stream (``flagos.current_stream`` is a vendor stream shim),
  so ``with flagos.graph(g):`` records flagos-device work exactly like eager
  dispatch into the vendor libtorch.

* **Hygon DCU (decoupled mode)**, where the CPU torch wheel's
  ``torch.cuda.CUDAGraph`` is a dummy and torch_fl's ``_C`` cannot link the
  hipified ``at::cuda::CUDAGraph`` (it lives in DTK's ``libtorch_hip.so`` while
  ``_C`` links the CPU torch).  Here we drive the HIP graph API directly through
  ctypes, on top of the capture-aware runtime
  (``SetStreamCapturing``, added in csrc/runtime/accelerator/nvidia).

Usage mirrors ``torch.cuda.graph``::

    g = flagos.CUDAGraph()
    with flagos.graph(g):
        out = model(x)          # captured
    g.replay()                  # re-runs the captured region
"""

import ctypes
import re

import torch

__all__ = ["CUDAGraph", "graph"]


def _cuda_graph_binding_available() -> bool:
    """True when ``torch.cuda.CUDAGraph`` is a real binding, not a dummy class.

    The CPU torch wheel injects a placeholder ``_CUDAGraph`` whose ``__module__``
    is ``torch._utils`` and which raises "Tried to instantiate dummy base class
    CUDAGraph" on construction.  A real binding has ``__module__ == "torch._C"``.
    """
    try:
        import torch.cuda.graphs  # noqa: F401  -- defines torch._C._CUDAGraph
    except Exception:
        return False
    graph_cls = getattr(torch._C, "_CUDAGraph", None)
    return graph_cls is not None and getattr(graph_cls, "__module__", "") == "torch._C"


def cudagraph_available() -> bool:
    """True when ``torch.cuda.CUDAGraph`` is a usable graph class right now.

    The CPU torch wheel ships ``torch.cuda.CUDAGraph`` as a dummy base class
    whose ``__module__`` is ``torch._utils`` and which raises on construction.
    Usable means it is either the vendor torch's real pybind class (MetaX)
    or this module's ctypes-driven HIP graph (DCU decoupled mode, installed by
    ``torch_fl.flagos._alias_torch_cuda_graph``).  Inductor gates
    ``mode="reduce-overhead"`` on exactly this.
    """
    graph_cls = getattr(torch.cuda, "CUDAGraph", None)
    return (
        graph_cls is not None and getattr(graph_cls, "__module__", "") != "torch._utils"
    )


def cudagraph_is_native() -> bool:
    """True when the graph class is the torch build's own CUDAGraph binding.

    ``cudagraph_available`` answers "can a graph be constructed here"; this
    answers "is it the runtime's native implementation". The distinction matters
    to ``torch.compile``: on a native binding its generated kernels capture and
    instantiate normally, while on the emulated ctypes HIP graph instantiating a
    graph of Inductor-generated kernels aborts inside the vendor runtime (see
    ``hip_graph_instantiate_crash.md``). So ``mode="reduce-overhead"`` is enabled
    automatically only on the native path.
    """
    return _cuda_graph_binding_available()


# Inductor's generated code snapshots the raw-stream getter at module import
# time: ``from torch._C import _cuda_getCurrentRawStream as get_raw_stream``. So
# the function object must never be reassigned -- only the mutable state it reads.
# Reassigning it (as _route_stream did) leaves the snapshot on the default-stream
# handle and compiled kernels never enter the capture. Keep a stable getter over
# a module global: None means "not capturing", in which case the getter defers to
# the native raw-stream getter the accelerator compat layer installed (which is
# the current queue on native builds, and 0 on the decoupled CPU-torch build).
_capture_raw_stream = None

# The raw-stream getter snapshot at compile time, before the stable getter
# replaces it. Read by _raw_stream_getter when nothing is being captured.
_native_raw_stream = None


def _raw_stream_getter(idx=0):
    if _capture_raw_stream is not None:
        return _capture_raw_stream
    if _native_raw_stream is not None:
        return _native_raw_stream(idx)
    return 0


def install_raw_stream_getter() -> None:
    """Point ``torch._C._cuda_getCurrentRawStream`` at the stable getter.

    Inductor's generated code snapshots this name at module import time
    (``from torch._C import _cuda_getCurrentRawStream as get_raw_stream``), so the
    binding must be installed before the first compiled graph is imported. The
    accelerator compat layers (dcu/nvidia/metax) overwrite it with their own
    default-stream lambda during ``import torch_fl``, which is why the module-level
    assignment above is not enough -- this must be re-applied at compile time,
    after those layers have run and before Inductor emits code.
    """
    global _native_raw_stream
    _native_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    torch._C._cuda_getCurrentRawStream = _raw_stream_getter


if _cuda_graph_binding_available():
    # ---- Backend 1: real torch.cuda.CUDAGraph binding (MetaX vendor torch). ----
    CUDAGraph = torch.cuda.CUDAGraph
    graph = torch.cuda.graph

else:
    # ---- Backend 2: ctypes-driven HIP graph (Hygon DCU decoupled mode). ----

    _HIP_LIBS = ("libgalaxyhip.so.5", "libamdhip64.so", "libamdhip64.so.6")

    def _load_hip():
        for name in _HIP_LIBS:
            try:
                return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
        raise ImportError("no HIP runtime library found")

    def _load_loaded(pattern):
        maps = open("/proc/self/maps").read()
        hits = sorted({m for m in re.findall(rf"(\S*{pattern}\S*)", maps) if m})
        if not hits:
            raise ImportError(f"{pattern} is not loaded")
        return ctypes.CDLL(hits[0])

    class CUDAGraph:
        """A CUDA/HIP graph, captured and replayed on a private stream.

        Not thread-safe and single-graph-per-instance, mirroring
        torch.cuda.CUDAGraph.
        """

        def __init__(self) -> None:
            self._hip = _load_hip()
            self._bind_hip()
            # stream + flagos capture-flag bindings (see csrc/runtime/accelerator/nvidia)
            torch_hip = _load_loaded(r"libtorch_hip\.so")
            self._set_cuda_stream = torch_hip.torch_set_current_cuda_stream
            self._set_cuda_stream.restype = ctypes.c_int
            self._set_cuda_stream.argtypes = [ctypes.c_void_p, ctypes.c_int32]
            flagos = _load_loaded(r"libflagos\.so")
            self._set_capturing = flagos.SetStreamCapturing
            self._set_capturing.restype = None
            self._set_capturing.argtypes = [ctypes.c_void_p]

            self._stream = ctypes.c_void_p()
            rc = self._hip.hipStreamCreate(ctypes.byref(self._stream))
            if rc != 0 or not self._stream.value:
                raise RuntimeError(f"hipStreamCreate failed: {rc}")
            self._graph = None  # hipGraph_t (during capture)
            self._exec = None  # hipGraphExec_t (after instantiate)

        def _bind_hip(self) -> None:
            sig = {
                "hipStreamCreate": ([ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
                "hipStreamBeginCapture": (
                    [ctypes.c_void_p, ctypes.c_int],
                    ctypes.c_int,
                ),
                "hipStreamEndCapture": (
                    [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
                    ctypes.c_int,
                ),
                "hipGraphInstantiate": (
                    [
                        ctypes.POINTER(ctypes.c_void_p),
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_void_p),
                        ctypes.c_char_p,
                        ctypes.c_size_t,
                    ],
                    ctypes.c_int,
                ),
                "hipGraphLaunch": ([ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
                "hipGraphExecDestroy": ([ctypes.c_void_p], ctypes.c_int),
                "hipGraphDestroy": ([ctypes.c_void_p], ctypes.c_int),
                "hipStreamDestroy": ([ctypes.c_void_p], ctypes.c_int),
                "hipStreamSynchronize": ([ctypes.c_void_p], ctypes.c_int),
            }
            for name, (args, restype) in sig.items():
                fn = getattr(self._hip, name)
                fn.restype = restype
                fn.argtypes = args

        def _route_stream(self, stream) -> None:
            """Point vendor + triton launches at `stream` (None = default stream)."""
            global _capture_raw_stream
            if stream is None:
                self._set_cuda_stream(None, 0)
                handle = 0
                # Not capturing: defer the raw-stream getter to the native one.
                _capture_raw_stream = None
            else:
                handle = stream.value
                self._set_cuda_stream(stream, 0)
                _capture_raw_stream = handle
            # Update the mutable state the stable getter reads; do not reassign
            # torch._C._cuda_getCurrentRawStream -- the generated Inductor code
            # already snapshotted the function object at import time.
            try:
                from triton.runtime import driver

                driver.active.get_current_stream = lambda idx=0, h=handle: h
            except Exception:
                pass

        def capture_begin(self, capture_error_mode: str = "global") -> None:
            stream = self._stream
            self._route_stream(stream)
            self._set_capturing(stream)
            # HIP capture modes, mirroring torch.cuda.graph's capture_error_mode:
            #   global (default) / thread_local / relaxed
            mode = {
                "global": 0,  # hipStreamCaptureModeGlobal
                "thread_local": 1,  # hipStreamCaptureModeThreadLocal
                "relaxed": 2,  # hipStreamCaptureModeRelaxed
            }[capture_error_mode]
            rc = self._hip.hipStreamBeginCapture(stream, mode)
            if rc != 0:
                self._set_capturing(None)
                raise RuntimeError(f"hipStreamBeginCapture failed: {rc}")

        def capture_end(self) -> None:
            try:
                graph = ctypes.c_void_p()
                rc = self._hip.hipStreamEndCapture(self._stream, ctypes.byref(graph))
                if rc != 0:
                    raise RuntimeError(f"hipStreamEndCapture failed: {rc}")
                self._graph = graph
            finally:
                self._set_capturing(None)
                self._route_stream(None)

        def instantiate(self) -> None:
            if self._graph is None:
                raise RuntimeError("capture_end() must be called before instantiate()")
            exec_ = ctypes.c_void_p()
            rc = self._hip.hipGraphInstantiate(
                ctypes.byref(exec_), self._graph, None, None, 0
            )
            if rc != 0:
                raise RuntimeError(f"hipGraphInstantiate failed: {rc}")
            self._exec = exec_

        def replay(self) -> None:
            if self._exec is None:
                raise RuntimeError("instantiate() must be called before replay()")
            rc = self._hip.hipGraphLaunch(self._exec, self._stream)
            if rc != 0:
                raise RuntimeError(f"hipGraphLaunch failed: {rc}")
            self._hip.hipStreamSynchronize(self._stream)

        def reset(self) -> None:
            if self._exec is not None:
                self._hip.hipGraphExecDestroy(self._exec)
                self._exec = None
            if self._graph is not None:
                self._hip.hipGraphDestroy(self._graph)
                self._graph = None
            if self._stream.value:
                self._hip.hipStreamDestroy(self._stream)
                self._stream = ctypes.c_void_p()

    class graph:
        """Context manager capturing into a :class:`CUDAGraph`, like torch.cuda.graph.

        ``pool`` and ``stream`` mirror ``torch.cuda.graph``'s signature so Inductor
        can call it unchanged; the flagos graph captures on its own private stream
        (created in ``CUDAGraph.__init__``), so they are accepted but unused.
        """

        def __init__(
            self,
            cuda_graph: CUDAGraph,
            pool=None,
            stream=None,
            capture_error_mode: str = "global",
        ) -> None:
            self._graph = cuda_graph
            self._capture_error_mode = capture_error_mode
            self._stream = stream

        def __enter__(self) -> CUDAGraph:
            self._graph.capture_begin(self._capture_error_mode)
            return self._graph

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            if exc_type is not None:
                self._graph.reset()
                return
            self._graph.capture_end()
            self._graph.instantiate()
