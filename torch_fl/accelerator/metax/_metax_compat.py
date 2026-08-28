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

"""
MetaX compatibility layer for torch.cuda on MetaX (MetaX) hardware.

On MetaX systems, PyTorch's bundled libcudart.so.12 (CUDA 12.x) is ABI-incompatible
with MetaX's cu-bridge (CUDA 11.6 compat). Basic CUDA calls (device_count, is_available)
work via LD_PRELOAD of libsymbol_cu.so, but get_device_properties fails because
PyTorch calls cudaGetDeviceProperties_v2 (CUDA 12 API) which MetaX doesn't provide.

This module monkey-patches torch.cuda.get_device_properties and get_device_name
to use MetaX's native mcruntime API via ctypes, bypassing the incompatible C++ path.

Usage:
    Before importing flag_gems or any code that calls torch.cuda.get_device_properties:

        from torch_fl._metax_compat import patch_torch_cuda_for_metax
        patch_torch_cuda_for_metax()

Environment:
    Requires LD_PRELOAD=/opt/maca/lib/libsymbol_cu.so and MetaX SDK installed.
    Set METAX_PATH env var if MetaX is not at /opt/maca.
"""

import ctypes
import os
import warnings
from dataclasses import dataclass
from typing import Union

import torch


def _find_metax_path():
    """Find the MetaX SDK root path."""
    for env_var in ("METAX_PATH", "METAX_HOME", "MACA_PATH", "MACA_HOME"):
        path = os.environ.get(env_var)
        if path and os.path.isdir(path):
            return path
    for default in ("/opt/maca", "/opt/maca-3.3.0"):
        if os.path.isdir(default):
            return default
    return None


def _load_mcruntime(metax_path):
    """Load MetaX mcruntime library via ctypes."""
    lib_path = os.path.join(metax_path, "lib", "libmcruntime.so")
    if not os.path.isfile(lib_path):
        return None
    try:
        return ctypes.CDLL(lib_path)
    except OSError:
        return None


@dataclass
class _MetaxDeviceProperties:
    """Minimal device properties matching torch.cuda._CudaDeviceProperties interface."""

    name: str = ""
    major: int = 0
    minor: int = 0
    total_memory: int = 0
    multi_processor_count: int = 0
    is_integrated: bool = False
    is_multi_gpu_board: bool = False
    warp_size: int = 64
    max_threads_per_multi_processor: int = 0
    regs_per_multiprocessor: int = 0
    shared_memory_per_block: int = 0
    shared_memory_per_block_optin: int = 0
    shared_memory_per_multiprocessor: int = 0
    L2_cache_size: int = 0
    pci_bus_id: str = ""
    pci_device_id: str = ""
    pci_domain_id: str = ""
    uuid: str = ""
    gcnArchName: str = ""


def _query_metax_device_properties(mcruntime, device_index):
    """Query device properties from MetaX native API."""
    props = _MetaxDeviceProperties()

    # mcDeviceGetName(char *name, int len, int device)
    name_buf = ctypes.create_string_buffer(256)
    ret = mcruntime.mcDeviceGetName(name_buf, 256, device_index)
    if ret == 0:
        props.name = name_buf.value.decode("utf-8", errors="replace")

    # mcDeviceGetAttribute(int *value, int attr, int device)
    mcDeviceGetAttribute = mcruntime.mcDeviceGetAttribute
    mcDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    mcDeviceGetAttribute.restype = ctypes.c_int

    val = ctypes.c_int(0)

    def get_attr(attr_id):
        ret = mcDeviceGetAttribute(ctypes.byref(val), attr_id, device_index)
        return val.value if ret == 0 else 0

    props.multi_processor_count = get_attr(16)
    props.warp_size = get_attr(10)
    props.max_threads_per_multi_processor = get_attr(39)
    props.shared_memory_per_block = get_attr(8)
    props.shared_memory_per_multiprocessor = get_attr(81)

    # Major/minor from MetaX may not map to CUDA compute capability.
    # Use reasonable defaults for MetaX GPUs.
    raw_major = get_attr(21)
    raw_minor = get_attr(22)
    if raw_major > 100:
        # MetaX returns non-standard values; use a sensible default
        props.major = 8
        props.minor = 0
    else:
        props.major = raw_major
        props.minor = raw_minor

    # Get total memory
    mcruntime.mcSetDevice(device_index)
    free_mem = ctypes.c_size_t(0)
    total_mem = ctypes.c_size_t(0)
    mcMemGetInfo = mcruntime.mcMemGetInfo
    mcMemGetInfo.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    mcMemGetInfo.restype = ctypes.c_int
    ret = mcMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))
    if ret == 0:
        props.total_memory = total_mem.value // (1024 * 1024)  # in MiB

    return props


# Cache for device properties
_metax_props_cache = {}
_mcruntime = None
_patched = False

# device_index -> torch.Generator(device="cuda"), the flaggems RNG source.
_cuda_generators = {}


def _get_cuda_generator(idx):
    """Lazily build one CUDA generator per device (the flaggems RNG source).

    MetaX's flag_gems backend uses ``device_name="cuda"``, so its
    ``philox_backend_seed_offset`` reads ``torch.cuda.default_generators[dev]``
    and unpacks the state as 2x int64 ``(seed, offset)`` -- the CUDA generator's
    philox layout. The flagos PrivateUse1 generator is a CPUGeneratorImpl whose
    state is the ~5KB mt19937 blob (632x int64), which would blow up that unpack
    with "too many values to unpack". So we must expose CUDA generators here,
    NOT the flagos ones, and seed those.
    """
    gen = _cuda_generators.get(idx)
    if gen is None:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(torch.initial_seed())
        _cuda_generators[idx] = gen
    return gen


class _CudaDefaultGenerators:
    """list-like stand-in for ``torch.cuda.default_generators`` (see
    _get_cuda_generator). Indexing yields a per-device CUDA generator; ``len``
    reports the device count so flag_gems' empty-tuple guard is False.

    Upstream declares ``default_generators`` as a *tuple*, so callers are
    entitled to iterate it, slice it or wrap it in ``list()``. Bounds-checking
    ``__getitem__`` is what makes that safe: with no ``__iter__``, Python falls
    back to the legacy protocol of calling ``__getitem__(0, 1, 2, ...)`` until
    IndexError, so an unchecked index turned ``for g in default_generators``
    into an infinite loop that allocated a fresh CUDA generator per step.
    """

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, idx):
        n = len(self)
        if isinstance(idx, slice):
            return tuple(self[i] for i in range(*idx.indices(n)))
        idx = int(idx)
        if idx < 0:  # negative indices wrap, as on the tuple this replaces
            idx += n
        if not 0 <= idx < n:
            raise IndexError(
                f"device index {idx} out of range for {n} flagos device(s)"
            )
        return _get_cuda_generator(idx)

    def __len__(self):
        try:
            _flagos = torch.flagos if hasattr(torch, "flagos") else None
            n = _flagos.device_count() if _flagos is not None else 0
        except Exception:
            n = 0
        return max(n, 1)


def _device_index(device: Union[torch.device, int, str, None]) -> int:
    """Extract a device index from the various forms torch.cuda accepts."""
    if device is None:
        return 0
    if isinstance(device, int):
        return device
    if isinstance(device, torch.device):
        return device.index if device.index is not None else 0
    if isinstance(device, str):
        return int(device.split(":")[-1]) if ":" in device else 0
    return int(device)


class _MetaxStreamShim:
    """Minimal stream object exposing ``.cuda_stream`` for triton-metax.

    Uses the null/default stream (0), consistent with the boxing path where the
    caching allocator is given ``stream=nullptr``. FlagGems' Triton launcher
    reads ``.cuda_stream`` (and torch._C._cuda_getCurrentRawStream) to pick the
    launch stream; the maca cu-bridge treats handle 0 as the default stream.
    """

    def __init__(self, index=0):
        self.cuda_stream = 0
        self.device_index = index

    def synchronize(self):
        _metax_synchronize()


def _metax_synchronize(device=None):
    """Synchronize the current MetaX device via cu-bridge cudaDeviceSynchronize."""
    cudart = _get_cudart()
    if cudart is not None:
        try:
            cudart.cudaDeviceSynchronize()
        except Exception:
            pass


def is_metax_available():
    """Check if MetaX runtime is available."""
    metax_path = _find_metax_path()
    if metax_path is None:
        return False
    return _load_mcruntime(metax_path) is not None


def _patch_inductor_event():
    """Make Inductor events bypass the lightweight CUDA stream shim."""
    flagos = getattr(torch, "flagos", None)
    if flagos is not None and getattr(flagos, "Event", None) is not None:
        torch.cuda.Event = flagos.Event


def patch_torch_cuda_for_metax():
    """
    Monkey-patch torch.cuda functions to work on MetaX hardware.

    This patches:
    - torch.cuda.get_device_properties: uses MetaX native API
    - torch.cuda.get_device_name: uses MetaX native API
    - torch.cuda._lazy_init: skips capability check
    - torch.cuda.get_device_capability: returns sensible defaults

    Must be called before importing flag_gems or other code that uses
    torch.cuda.get_device_properties.
    """
    global _mcruntime, _patched

    if _patched:
        # FLAGOS_METAX_COMPAT can call us once before torch.flagos is registered
        # and again afterward. Keep this late-bound patch outside the one-time
        # runtime setup so the second call can install the real Event wrapper.
        _patch_inductor_event()
        return True

    metax_path = _find_metax_path()
    if metax_path is None:
        warnings.warn("MetaX SDK not found, skipping torch.cuda patches")
        return False

    _mcruntime = _load_mcruntime(metax_path)
    if _mcruntime is None:
        warnings.warn(f"Cannot load libmcruntime.so from {metax_path}/lib/")
        return False

    # Skip PyTorch's CUDA capability check (it fails because MetaX reports
    # CUDA 11.6 but PyTorch expects CUDA 12.0+ runtime)
    if hasattr(torch.cuda, "_queued_calls"):
        torch.cuda._queued_calls.clear()

    def _patched_get_device_properties(
        device: Union[torch.device, int, str, None] = None,
    ):
        """Get device properties from MetaX native API."""
        if device is None:
            device_index = torch.cuda.current_device()
        elif isinstance(device, torch.device):
            device_index = device.index if device.index is not None else 0
        elif isinstance(device, str):
            device_index = int(device.split(":")[-1]) if ":" in device else 0
        else:
            device_index = int(device)

        if device_index not in _metax_props_cache:
            _metax_props_cache[device_index] = _query_metax_device_properties(
                _mcruntime, device_index
            )
        return _metax_props_cache[device_index]

    def _patched_get_device_name(
        device: Union[torch.device, int, str, None] = None,
    ) -> str:
        """Get device name from MetaX native API."""
        props = _patched_get_device_properties(device)
        return props.name

    def _patched_get_device_capability(
        device: Union[torch.device, int, str, None] = None,
    ):
        """Get device capability from MetaX (returns sensible defaults)."""
        props = _patched_get_device_properties(device)
        return (props.major, props.minor)

    torch.cuda.get_device_properties = _patched_get_device_properties
    torch.cuda.get_device_name = _patched_get_device_name
    torch.cuda.get_device_capability = _patched_get_device_capability

    # Patch basic CUDA tensor operations that use NVIDIA-specific kernels.
    # On MetaX, PyTorch's built-in CUDA kernels (fill_, zero_, copy_) fail
    # because they are compiled for NVIDIA GPUs. We replace them with
    # implementations using MetaX's cu-bridge runtime API (cudaMemset, cudaMemcpy).
    _patch_cuda_tensor_ops(metax_path)

    # Availability + stream shims (mirrors cuda/_cuda_compat.py). The pip
    # torch+cpu wheel is compiled WITHOUT CUDA, so torch.cuda.is_available() is
    # False and _lazy_init() raises. FlagGems' Triton path needs these to report
    # a usable device and hand out a launch stream; the actual compute still runs
    # through maca's libtorch_cuda.so via the boxing kernels.
    _flagos = torch.flagos if hasattr(torch, "flagos") else None

    def _current_device():
        if _flagos is not None:
            try:
                return _flagos.current_device()
            except Exception:
                pass
        return 0

    def _set_device(device):
        if _flagos is not None:
            try:
                _flagos.set_device(_device_index(device))
            except Exception:
                pass

    torch.cuda.is_available = lambda: True
    torch.cuda._lazy_init = lambda: None
    if hasattr(torch.cuda, "_initialized"):
        torch.cuda._initialized = True
    if hasattr(torch.cuda, "_queued_calls"):
        torch.cuda._queued_calls.clear()

    # Route torch.cuda.set_device / current_device to the maca runtime.
    #
    # Stock torch.cuda.set_device calls torch._C._cuda_setDevice, which on the
    # CPU wheel only bumps torch's own device counter and never reaches maca's
    # mcSetDevice -- so torch.cuda.set_device(rank) leaves the maca runtime on
    # device 0. NCCL(mccl) hides this because ProcessGroupNCCL wraps every op in
    # a CUDAGuard(tensor.device()); FlagCX does NOT -- it creates its collective
    # stream (mcStreamCreateWithFlags) on the *current* maca device before
    # setting the tensor's device, so on rank!=0 the stream lands on device 0
    # while the communicator/tensor are on device r, giving
    # "CUDA error: invalid resource handle". Make set_device actually move the
    # maca runtime (via _flagos.set_device) and read the true current device
    # back from it, so the FlagCX path binds a consistent device.
    _orig_cuda_setDevice = getattr(torch._C, "_cuda_setDevice", None)

    def _patched_set_device(device):
        idx = _device_index(device)
        if _orig_cuda_setDevice is not None:
            try:
                _orig_cuda_setDevice(idx)  # keep torch's own counter aligned
            except Exception:
                pass
        _set_device(idx)  # actually move the maca runtime (mcSetDevice)

    torch.cuda.set_device = _patched_set_device
    torch.cuda.current_device = _current_device

    torch.cuda.synchronize = _metax_synchronize
    torch.cuda.current_stream = lambda device=None: _MetaxStreamShim(
        _device_index(device)
    )
    torch.cuda.default_stream = lambda device=None: _MetaxStreamShim(
        _device_index(device)
    )

    # Inductor's autotuner constructs torch.cuda.Event directly. Its default
    # record path rejects the lightweight stream shim above, while flagos.Event
    # resolves the real underlying stream before recording the same MetaX event.
    _patch_inductor_event()

    # Device context exchange: extract index for flagos/privateuseone tensors.
    def _exchange_device(idx):
        if idx < 0:
            return -1
        prev = _current_device()
        _set_device(idx)
        return prev

    torch.cuda._exchange_device = _exchange_device
    torch.cuda._maybe_exchange_device = _exchange_device

    # triton reads torch._C._cuda_getCurrentRawStream(idx) -> raw handle.
    try:
        torch._C._cuda_getCurrentRawStream = lambda idx=0: 0
    except Exception:
        pass

    # Seeding: with is_available()=True, torch.manual_seed() calls
    # torch.cuda.manual_seed_all(), which walks torch.cuda.default_generators
    # -- an empty tuple on the CPU wheel -> IndexError. Install per-device CUDA
    # generators (philox 2x int64 state, exactly what MetaX's cuda-named
    # flag_gems philox_backend_seed_offset unpacks) and route cuda seeding to
    # them, so torch.manual_seed() makes flaggems RNG reproducible. NOTE: this
    # must be CUDA generators, NOT flagos.default_generators (CPUGeneratorImpl,
    # ~5KB mt19937 state) -- pointing default_generators at those would make
    # gems' `c0, c1 = state.view(int64)` fail with "too many values to unpack".
    def _manual_seed_all(seed):
        seed = int(seed)
        try:
            n = _flagos.device_count() if _flagos is not None else 1
            for i in range(max(n, 1)):
                _get_cuda_generator(i).manual_seed(seed)
        except Exception:
            pass

    def _manual_seed(seed):
        seed = int(seed)
        idx = _current_device()
        try:
            _get_cuda_generator(idx).manual_seed(seed)
        except Exception:
            pass

    torch.cuda.manual_seed_all = _manual_seed_all
    torch.cuda.manual_seed = _manual_seed
    try:
        torch.cuda.default_generators = _CudaDefaultGenerators()
    except Exception:
        pass

    # The public backend is flagos. Keep its seed/state API synchronized with
    # the CUDA-shaped Philox generators required by the MetaX FlagGems path.
    if _flagos is not None:
        native_manual_seed = _flagos.manual_seed
        native_manual_seed_all = _flagos.manual_seed_all

        def _flagos_manual_seed(seed):
            native_manual_seed(seed)
            _manual_seed(seed)

        def _flagos_manual_seed_all(seed):
            native_manual_seed_all(seed)
            _manual_seed_all(seed)

        def _flagos_get_rng_state(device="flagos"):
            return _get_cuda_generator(_device_index(device)).get_state()

        def _flagos_set_rng_state(state, device="flagos"):
            _get_cuda_generator(_device_index(device)).set_state(state)

        _flagos.manual_seed = _flagos_manual_seed
        _flagos.manual_seed_all = _flagos_manual_seed_all
        _flagos.get_rng_state = _flagos_get_rng_state
        _flagos.set_rng_state = _flagos_set_rng_state

    # FlagGems/Triton autotuners benchmark kernels with torch.cuda.Event, which
    # fails on the CPU torch wheel ("invalid device ordinal"). Time with a wall
    # clock instead -- affects only autotune config selection, not correctness.
    _patch_triton_do_bench()

    _patched = True
    return True


def patch_flaggems_device_name() -> bool:
    """Retarget FlagGems' device *identity* to ``flagos``, keep ``torch.cuda``.

    MetaX's FlagGems descriptor declares ``device_name="cuda"`` so its runtime
    picks ``torch.cuda`` as ``torch_device_fn`` (RNG, streams, and device guard
    all run against maca's libtorch_cuda via the shims above). But torch_fl's
    C++ ``flagos_python`` path hands the generic ops ``flagos``-typed
    (PrivateUse1) tensors, so every op's identity gate
    ``device.type != _DEVICE_NAME`` sees ``'flagos' != 'cuda'`` and wrongly
    falls through to the aten reference path. That path is fine for ops that
    have a CompositeExplicitAutograd fallback, but ``mul.out`` (used by the
    generic ``mul_``) does not, so ``loss.backward()`` dies with
    ``NotImplementedError: ... no fallback ... aten::mul.out``.

    Only the *identity* name is retargeted here. ``runtime._state.device_name``
    -- which ``set_torch_backend_device_fn``/``gen_torch_device_object`` read to
    build ``torch.cuda``/``torch.backends.cuda`` -- is left at ``cuda``, so the
    Triton path keeps launching through torch.cuda.

    Must run *after* ``import flag_gems`` (which binds the ``_DEVICE_NAME`` /
    ``device`` module globals) and before any flagos op runs. Idempotent.

    Returns False if FlagGems is not installed or is not on the metax vendor.
    """
    import importlib.util
    import sys

    if importlib.util.find_spec("flag_gems") is None:
        return False
    try:
        from flag_gems.runtime.backend.device_finder import DeviceDetector
    except ImportError:
        return False

    detector = DeviceDetector()
    if detector.vendor_name != "metax":
        return False

    stale = detector.name  # "cuda"
    if stale == "flagos":
        return True
    detector.name = "flagos"

    # Rewrite every flag_gems module global that captured the stale name at
    # import time. Two bindings exist in the generic ops:
    #   _DEVICE_NAME = runtime_device.name   (mul.py, ...)
    #   device      = device.name            (minimum/maximum/eq/..., via
    #                                         from flag_gems.runtime import device)
    # flag_gems/__init__.py also binds `device = runtime.device.name`.
    # Restrict to flag_gems namespaces: unlike GCU there are no bare-key vendor
    # modules here, and "cuda" is far too common a string to match globally.
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if not name.startswith("flag_gems"):
            continue
        for attr in ("_DEVICE_NAME", "device"):
            if getattr(module, attr, None) == stale:
                setattr(module, attr, "flagos")
    return True


def _patch_triton_do_bench():
    """Replace triton.testing.do_bench to avoid CUDA Event timing (metax).

    Mirrors cuda/_cuda_compat._patch_triton_do_bench: triton's autotuner uses
    torch.cuda.Event(enable_timing=True), which raises on the CPU torch wheel.
    We use a wall clock; kernels still run on the real MetaX GPU.
    """
    try:
        import triton
        import triton.testing
    except ImportError:
        return

    import statistics
    import time

    def _do_bench(
        fn,
        warmup=25,
        rep=100,
        grad_to_none=None,
        quantiles=None,
        return_mode="mean",
        **kwargs,
    ):
        fn()
        _metax_synchronize()
        n_rep = 5
        times = []
        for _ in range(n_rep):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            t0 = time.perf_counter()
            fn()
            _metax_synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)  # ms

        if quantiles is not None:
            times_sorted = sorted(times)

            def _quantile(q):
                pos = q * (len(times_sorted) - 1)
                lo = int(pos)
                hi = min(lo + 1, len(times_sorted) - 1)
                frac = pos - lo
                return times_sorted[lo] * (1 - frac) + times_sorted[hi] * frac

            ret = [_quantile(q) for q in quantiles]
            return ret[0] if len(ret) == 1 else ret

        if return_mode == "min":
            return min(times)
        if return_mode == "max":
            return max(times)
        if return_mode == "median":
            return statistics.median(times)
        if return_mode == "all":
            return times
        return statistics.mean(times)

    triton.testing.do_bench = _do_bench
    # Some triton versions cache the benchmarker on the driver; refresh it.
    try:
        triton.runtime.driver.active.get_benchmarker = lambda: _do_bench
    except Exception:
        pass


_cudaMemcpyHostToDevice = 1

# Cached cudart handle
_cudart = None


def _get_cudart():
    """Get the MetaX-compatible CUDA runtime library (cached)."""
    global _cudart
    if _cudart is not None:
        return _cudart
    # libsymbol_cu.so provides cuda* API functions compatible with MetaX
    try:
        _cudart = ctypes.CDLL("libsymbol_cu.so")
    except OSError:
        metax_path = _find_metax_path()
        if metax_path:
            _cudart = ctypes.CDLL(os.path.join(metax_path, "lib", "libsymbol_cu.so"))
    return _cudart


def _patch_cuda_tensor_ops(metax_path):
    """
    Patch PyTorch's CUDA tensor ops to use MetaX runtime API instead of
    NVIDIA-specific CUDA kernels.

    Patched operations:
    - Tensor.zero_(): uses cudaMemsetAsync (contiguous only)
    - Tensor.fill_(scalar): uses cudaMemsetAsync for 0, cudaMemcpy for others (contiguous only)
    """
    cudart = _get_cudart()
    if cudart is None:
        return

    # Setup cudaMemsetAsync
    cudaMemsetAsync = cudart.cudaMemsetAsync
    cudaMemsetAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    cudaMemsetAsync.restype = ctypes.c_int

    # Setup cudaMemcpyAsync
    cudaMemcpyAsync = cudart.cudaMemcpyAsync
    cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    cudaMemcpyAsync.restype = ctypes.c_int

    # Setup cudaStreamSynchronize
    cudaStreamSynchronize = cudart.cudaStreamSynchronize
    cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    cudaStreamSynchronize.restype = ctypes.c_int

    def _metax_copy_to_cuda(dst, src):
        """Copy src (CPU, contiguous) into dst (CUDA, contiguous) via cudaMemcpyAsync."""
        nbytes = src.nelement() * src.element_size()
        ret = cudaMemcpyAsync(
            dst.data_ptr(), src.data_ptr(), nbytes, _cudaMemcpyHostToDevice, None
        )
        if ret != 0:
            warnings.warn(f"cudaMemcpyAsync failed with error {ret}")
        cudaStreamSynchronize(None)

    _orig_zero = torch.Tensor.zero_

    def _metax_zero_(self):
        """zero_() using cudaMemsetAsync for CUDA tensors on MetaX."""
        if not self.is_cuda:
            return _orig_zero(self)
        if not self.is_contiguous():
            raise NotImplementedError(
                "MetaX compat: zero_() on non-contiguous CUDA tensors is not supported. "
                "Call .contiguous() first."
            )
        ptr = self.data_ptr()
        nbytes = self.nelement() * self.element_size()
        ret = cudaMemsetAsync(ptr, 0, nbytes, None)
        if ret != 0:
            cpu_zeros = torch.zeros(self.shape, dtype=self.dtype, device="cpu")
            _metax_copy_to_cuda(self, cpu_zeros)
        return self

    torch.Tensor.zero_ = _metax_zero_

    _orig_fill = torch.Tensor.fill_

    def _metax_fill_(self, value):
        """fill_() for CUDA tensors on MetaX."""
        if not self.is_cuda:
            return _orig_fill(self, value)
        if value == 0 and self.is_contiguous():
            return _metax_zero_(self)
        # Create on CPU and copy to CUDA
        cpu_t = torch.full(self.shape, value, dtype=self.dtype, device="cpu")
        if not self.is_contiguous():
            raise NotImplementedError(
                "MetaX compat: fill_() on non-contiguous CUDA tensors is not supported. "
                "Call .contiguous() first."
            )
        _metax_copy_to_cuda(self, cpu_t)
        return self

    torch.Tensor.fill_ = _metax_fill_
