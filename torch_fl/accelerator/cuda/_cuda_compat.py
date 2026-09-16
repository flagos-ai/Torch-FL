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
Generic NVIDIA CUDA compatibility layer for torch.cuda under the flagos backend.

Under the external-libtorch scheme the pip torch is CPU-only
(``torch.__version__`` ends with ``+cpu``): its ``torch.cuda`` Python bindings
were compiled WITHOUT CUDA, so ``torch.cuda.is_available()`` is ``False`` and
``torch.cuda._lazy_init()`` raises "Torch not compiled with CUDA enabled". That
cannot be fixed by ``LD_PRELOAD``-ing ``libtorch_cuda.so`` -- the Python layer
was frozen at compile time.

Triton, however, does NOT use torch's CUDA Python layer to compile/launch
kernels: it uses its own C extension plus the system ``libcuda.so`` (the NVIDIA
driver). So FlagGems' Triton kernels can run correctly as long as we make
torch.cuda's *probe* functions report a real device. This module monkey-patches
those probes, sourcing real values from the CUDA Driver API (``libcuda.so``,
always present alongside an NVIDIA driver) via ctypes -- no CUDA runtime, no
torch CUDA build required.

Reporting a real device is necessary but not sufficient: FlagGems also compares
the *name* of the device on its inputs against the name its own nvidia backend
declares (``"cuda"``), while torch_fl registers the accelerator as ``flagos``.
The ops that make that comparison fell back to ATen instead of running their
kernel -- see ``patch_flaggems_device_name``, which realigns the two.

Enabled by default from ``torch_fl.__init__`` when a generic NVIDIA GPU is
detected (not MetaX, not Ascend). Disable with ``FLAGOS_DISABLE_CUDA_SHIM=1``.

Modeled on ``torch_fl/accelerator/metax/_metax_compat.py``.
"""

import ctypes
import os
import warnings
from dataclasses import dataclass
from typing import Union

import torch


_patched = False
_cuda = None  # cached libcuda.so handle
_cudart = None  # cached libcudart.so handle (for synchronize)
_props_cache = {}

# device_index -> torch.Generator(device="cuda"), one per device. See
# _get_cuda_generator / _CudaDefaultGenerators below.
_cuda_generators = {}


def _get_cuda_generator(idx):
    """Lazily build one CUDA generator per device (the flaggems RNG source).

    flag_gems' ``philox_backend_seed_offset`` (nvidia ``device_name="cuda"``)
    reads ``torch.cuda.default_generators[device]`` and unpacks its 16-byte
    state as 2x int64 ``(seed, offset)`` -- the CUDA generator's philox layout.
    We install these as ``torch.cuda.default_generators`` so gems' generator-less
    RNG ops find a real, seedable generator instead of the empty tuple the
    CPU-torch wheel ships (which used to force the philox monkeypatch).

    Lazy because at import time the external ``libtorch_cuda.so`` is not yet
    wired into ATen -- ``torch.Generator(device="cuda")`` raises "Cannot get
    CUDA generator without ATen_cuda library". By the time any RNG op runs, cuda
    is live and construction succeeds. Seeded from ``torch.initial_seed()`` so a
    ``torch.manual_seed(...)`` issued before first use is honoured.
    """
    gen = _cuda_generators.get(idx)
    if gen is None:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(torch.initial_seed())
        _cuda_generators[idx] = gen
    return gen


class _CudaDefaultGenerators:
    """list-like stand-in for ``torch.cuda.default_generators``.

    Indexing yields a real (lazily created) per-device CUDA generator; ``len``
    reports the device count so flag_gems' ``len(default_generators) == 0``
    guard is False and it uses the generator instead of erroring.

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
            raise IndexError(f"device index {idx} out of range for {n} device(s)")
        return _get_cuda_generator(idx)

    def __len__(self):
        try:
            n = torch.cuda.device_count()
        except Exception:
            n = 0
        return max(n, 1)


# ---- CUDA Driver API (libcuda.so) constants ----
# CUdevice_attribute enum values (cuda.h). Confirmed against A100 (sm_80).
_ATTR_CC_MAJOR = 75
_ATTR_CC_MINOR = 76
_ATTR_MP_COUNT = 16
_ATTR_L2_CACHE_SIZE = 38
_ATTR_MAX_THREADS_PER_MP = 39
_ATTR_WARP_SIZE = 10


@dataclass
class _CudaDeviceProperties:
    """Minimal stand-in for torch.cuda._CudaDeviceProperties.

    Exposes the fields FlagGems reads: ``name``, ``major``, ``minor``,
    ``multi_processor_count``, ``L2_cache_size``, ``total_memory``,
    ``warp_size``, ``max_threads_per_multi_processor``.
    """

    name: str = ""
    major: int = 8
    minor: int = 0
    total_memory: int = 0
    multi_processor_count: int = 108
    L2_cache_size: int = 40 * 1024 * 1024
    warp_size: int = 32
    max_threads_per_multi_processor: int = 2048
    is_integrated: bool = False
    is_multi_gpu_board: bool = False
    gcnArchName: str = ""

    def __repr__(self):
        return (
            f"_CudaDeviceProperties(name='{self.name}', "
            f"major={self.major}, minor={self.minor}, "
            f"total_memory={self.total_memory // (1024 * 1024)}MB, "
            f"multi_processor_count={self.multi_processor_count})"
        )


def _load_libcuda():
    """Load the NVIDIA driver library (libcuda.so) via ctypes, cached."""
    global _cuda
    if _cuda is not None:
        return _cuda
    for name in ("libcuda.so", "libcuda.so.1"):
        try:
            _cuda = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if _cuda is not None:
        # cuInit(0) is idempotent and required before other Driver API calls.
        try:
            _cuda.cuInit(0)
        except Exception:
            _cuda = None
    return _cuda


def is_nvidia_cuda_available() -> bool:
    """True if a generic NVIDIA GPU is reachable via the driver API."""
    cuda = _load_libcuda()
    if cuda is None:
        return False
    count = ctypes.c_int(0)
    try:
        if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return False
    except Exception:
        return False
    return count.value > 0


def cuda_shim_active() -> bool:
    """True once ``patch_torch_cuda_for_flagos`` has wired torch.cuda to the driver.

    Callers that need to know whether a real CUDA runtime backs this build cannot
    test ``torch._C._cuda_getCurrentStream`` for it: that symbol exists only when
    torch itself was compiled with CUDA, and the CPU torch wheel this shim exists
    for does not have it. The shim's own state is the accurate signal.
    """
    return _patched


def cuda_graph_supported() -> bool:
    """True when torch can capture and replay CUDA graphs on this build.

    ``torch.cuda.graphs`` binds ``_CUDAGraph`` as a dummy base class when the
    wheel has no CUDA graph binding, and the dummy raises ``RuntimeError: Tried to
    instantiate dummy base class CUDAGraph`` on construction. A CPU torch wheel
    driving an external libtorch_cuda.so is that case: the CUDA runtime underneath
    is real, but no C++ graph object is reachable from Python.

    ``hasattr`` cannot be the test. torch injects the placeholders into
    ``torch._C.__dict__`` as soon as ``torch.cuda.streams`` is imported -- ``cuda``
    is not part of any real API surface -- so the attribute exists on the CPU wheel
    too. What separates them is where the class came from: ``_dummy_type`` builds
    its placeholders with a bare ``type()`` call inside ``torch._utils``, while a
    real binding is a built-in type whose ``__module__`` is ``torch._C``.
    """
    try:
        # Importing the module is what defines torch._C._CUDAGraph.
        import torch.cuda.graphs  # noqa: F401
    except Exception:
        return False
    graph_cls = getattr(torch._C, "_CUDAGraph", None)
    return graph_cls is not None and graph_cls.__module__ == "torch._C"


def _device_index(device: Union[torch.device, int, str, None]) -> int:
    if device is None:
        return 0
    if isinstance(device, torch.device):
        return device.index if device.index is not None else 0
    if isinstance(device, str):
        return int(device.split(":")[-1]) if ":" in device else 0
    return int(device)


def _query_device_properties(device_index: int) -> _CudaDeviceProperties:
    """Query device properties from the CUDA Driver API."""
    props = _CudaDeviceProperties()
    cuda = _load_libcuda()
    if cuda is None:
        return props

    dev = ctypes.c_int(0)
    if cuda.cuDeviceGet(ctypes.byref(dev), device_index) != 0:
        return props

    # Name
    name_buf = ctypes.create_string_buffer(256)
    if cuda.cuDeviceGetName(name_buf, 256, dev) == 0:
        props.name = name_buf.value.decode("utf-8", errors="replace")

    val = ctypes.c_int(0)

    def attr(attr_id, default):
        if cuda.cuDeviceGetAttribute(ctypes.byref(val), attr_id, dev) == 0:
            return val.value
        return default

    props.major = attr(_ATTR_CC_MAJOR, props.major)
    props.minor = attr(_ATTR_CC_MINOR, props.minor)
    props.multi_processor_count = attr(_ATTR_MP_COUNT, props.multi_processor_count)
    props.L2_cache_size = attr(_ATTR_L2_CACHE_SIZE, props.L2_cache_size)
    props.warp_size = attr(_ATTR_WARP_SIZE, props.warp_size)
    props.max_threads_per_multi_processor = attr(
        _ATTR_MAX_THREADS_PER_MP, props.max_threads_per_multi_processor
    )

    total = ctypes.c_size_t(0)
    try:
        if cuda.cuDeviceTotalMem_v2(ctypes.byref(total), dev) == 0:
            props.total_memory = total.value
    except Exception:
        pass

    return props


def _get_props(device=None) -> _CudaDeviceProperties:
    idx = _device_index(device)
    if idx not in _props_cache:
        _props_cache[idx] = _query_device_properties(idx)
    return _props_cache[idx]


def _load_cudart():
    """Load libcudart.so for cudaDeviceSynchronize, cached."""
    global _cudart
    if _cudart is not None:
        return _cudart
    try:
        _cudart = ctypes.CDLL("libcudart.so")
    except OSError:
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        try:
            _cudart = ctypes.CDLL(f"{cuda_home}/lib64/libcudart.so")
        except OSError:
            _cudart = None
    return _cudart


def _real_stream(device_index=0):
    """Resolve the true current stream as a real ``torch.cuda.Stream``.

    ``torch.flagos.current_stream`` reads the stream tuple from the C++ runtime
    (``_cuda_getCurrentStream``), which the shim below never overwrote, so it
    denotes the same physical stream the boxing kernels submit to. Returns
    ``None`` on a runtime with no CUDA stream API to read from.
    """
    flagos = getattr(torch, "flagos", None)
    if flagos is None:
        return None
    try:
        real = flagos.current_stream(device_index)
    except Exception:
        return None
    return real if isinstance(real, torch.cuda.Stream) else None


class _StreamShim:
    """Stream stand-in for the CPU torch wheel, where ``torch.cuda.Stream`` is a
    dummy base class that raises on construction.

    Triton's launcher reads ``.cuda_stream`` to pick the launch stream; that is 0,
    the null/default stream, until something selects another one, matching the
    boxing path where the caching allocator is given ``stream=nullptr``.

    The event/ordering half of the stream API delegates to the real stream
    (``_real_stream``) rather than being stubbed out. ``torch.cuda.Stream``
    implements ``wait_stream`` as ``self.wait_event(stream.record_event())``, and
    ``torch.cuda.StreamContext`` compares ``current_stream().device`` on entry
    while restoring through ``current_stream().stream_id`` on exit -- so a shim
    missing any of those makes the whole ``with torch.cuda.stream(...)`` protocol
    unusable. FlagTree's Triton benchmarks every autotuned FlagGems kernel that
    way (``triton.testing.do_bench_cudagraph``), and it is what the shim's own
    ``_synchronize``-only, `.cuda_stream`-only surface broke on PPU.
    """

    def __init__(self, index=0):
        self.cuda_stream = 0
        self.device_index = index
        # Snapshot the stream that is current *now*, rather than reading it on
        # every attribute access. A stream object has to keep denoting the same
        # stream for as long as it is held: torch.cuda.StreamContext saves
        # current_stream() on entry and restores it on exit, so a live view would
        # restore whatever became current in between -- i.e. never restore at all.
        self._real = _real_stream(index)

    @property
    def device(self):
        """The device this stream belongs to, spelled as torch.cuda spells it.

        ``torch.cuda.StreamContext.__enter__`` compares ``current_stream().device``
        against the target stream's, so a shim without this attribute breaks every
        ``with torch.cuda.stream(...)`` block.
        """
        return torch.device("cuda", self.device_index)

    @property
    def stream_id(self):
        return 0 if self._real is None else self._real.stream_id

    @property
    def handle(self):
        """Raw stream handle, under the name the vendor stream classes use.

        ``torch_fl.flagos.Stream`` reads ``.handle`` off whatever it wraps, so
        accepting this shim as a delegate needs the alias to exist. It is the
        same value as ``cuda_stream``.
        """
        return self.cuda_stream

    @property
    def device_type(self):
        # DeviceType::CUDA; only meaningful when there is a real CUDA stream.
        return 1 if self._real is None else self._real.device_type

    def record_event(self, event=None):
        if self._real is None:
            raise RuntimeError(
                "cannot record an event on the cuda stream shim: "
                "torch.flagos is not initialized"
            )
        if event is None:
            # Not ``real.record_event(None)``: that allocates the ``Event`` bound
            # inside torch/cuda/streams.py, which is the dummy base class on the
            # CPU torch wheel. ``torch.cuda.Event`` is the working flagos wrapper.
            event = torch.cuda.Event()
        event.record(self._real)
        return event

    def wait_event(self, event):
        if self._real is None:
            return None
        return self._real.wait_event(event)

    def wait_stream(self, other):
        if self._real is None:
            return None
        return self._real.wait_stream(other)

    def query(self):
        return True if self._real is None else self._real.query()

    def synchronize(self):
        _synchronize()


def _synchronize(device=None):
    cudart = _load_cudart()
    if cudart is not None:
        try:
            cudart.cudaDeviceSynchronize()
            return
        except Exception:
            pass
    # Fall back to driver API context sync.
    cuda = _load_libcuda()
    if cuda is not None:
        try:
            cuda.cuCtxSynchronize()
        except Exception:
            pass


def patch_torch_cuda_for_flagos():
    """Monkey-patch torch.cuda probes to report a real NVIDIA GPU.

    Must be called before importing flag_gems (which reads
    ``torch.cuda.get_device_name()`` at import).
    """
    global _patched
    if _patched:
        return True

    if not is_nvidia_cuda_available():
        warnings.warn(
            "torch_fl: no NVIDIA GPU reachable via libcuda.so; skipping torch.cuda shim"
        )
        return False

    _flagos = torch.flagos if hasattr(torch, "flagos") else None

    def _device_count():
        cuda = _load_libcuda()
        if cuda is None:
            return 0
        count = ctypes.c_int(0)
        if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return 0
        return count.value

    def _current_device():
        # Route through the flagos runtime so the notion of "current device"
        # stays consistent with the PrivateUse1 backend.
        if _flagos is not None:
            try:
                return _flagos.current_device()
            except Exception:
                pass
        return 0

    def _set_device(device):
        idx = _device_index(device)
        if _flagos is not None:
            try:
                _flagos.set_device(idx)
            except Exception:
                pass

    # --- probes ---
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = _device_count
    torch.cuda.current_device = _current_device
    torch.cuda.set_device = _set_device
    torch.cuda.get_device_properties = _get_props
    torch.cuda.get_device_name = lambda device=None: _get_props(device).name
    torch.cuda.get_device_capability = lambda device=None: (
        _get_props(device).major,
        _get_props(device).minor,
    )
    torch.cuda.synchronize = _synchronize

    # _lazy_init must be a no-op; the real one raises on CPU torch.
    torch.cuda._lazy_init = lambda: None
    if hasattr(torch.cuda, "_initialized"):
        torch.cuda._initialized = True
    if hasattr(torch.cuda, "_queued_calls"):
        torch.cuda._queued_calls.clear()

    # Device context: extract index for flagos/privateuseone; forward to driver.
    _orig_device_init = torch.cuda.device.__init__

    def _patched_device_init(self, device):
        if hasattr(device, "type") and hasattr(device, "index"):
            if device.type in ("privateuseone", "flagos"):
                device = device.index if device.index is not None else 0
        try:
            return _orig_device_init(self, device)
        except Exception:
            # CPU torch's device ctx may reject; store index for our exchange.
            self.idx = _device_index(device)
            self.prev_idx = -1

    torch.cuda.device.__init__ = _patched_device_init

    def _exchange_device(idx):
        if idx < 0:
            return -1
        prev = _current_device()
        _set_device(idx)
        return prev

    torch.cuda._exchange_device = _exchange_device
    torch.cuda._maybe_exchange_device = _exchange_device

    # Streams for triton raw-stream lookup.
    torch.cuda.current_stream = lambda device=None: _StreamShim(_device_index(device))
    torch.cuda.default_stream = lambda device=None: _StreamShim(_device_index(device))

    # torch.cuda.set_stream -- and therefore StreamContext, i.e. every
    # ``with torch.cuda.stream(...)`` block -- writes through
    # torch._C._cuda_setStream, absent from the CPU wheel. The shims above always
    # denote stream 0, so "switch to the shim's stream" is switching to the stream
    # already selected; dropping the call is not an approximation of the switch,
    # it is the switch. Triton's do_bench_cudagraph enters a torch.cuda.stream
    # block before benchmarking any candidate config.
    if not hasattr(torch._C, "_cuda_setStream"):
        torch.cuda.set_stream = lambda stream: None

    # torch.cuda.Event/Stream are dummy base classes in the CPU wheel and raise
    # on construction. flagos ships working ones over the same physical GPU
    # (its Event does real elapsed_time), so hand those out instead. inductor's
    # kernel benchmarking constructs torch.cuda.Event(enable_timing=True).
    if _flagos is not None:
        if getattr(_flagos, "Event", None) is not None:
            torch.cuda.Event = _flagos.Event
        if getattr(_flagos, "Stream", None) is not None:
            torch.cuda.Stream = _flagos.Stream

    # Memory stats. Every torch.cuda.memory_* query goes through
    # torch._C._cuda_memoryStats, which the CPU wheel does not build, so they all
    # raise AttributeError. flagos delegates its allocator to
    # c10::cuda::CUDACachingAllocator, so its own stats describe the very same
    # pool -- route the CUDA queries there. inductor's autotuner needs these to
    # size its benchmark scratch budget (copy_args_to_cpu_if_needed).
    if _flagos is not None:
        torch.cuda.memory_allocated = lambda device=None: _flagos.memory_allocated(
            _device_index(device)
        )
        torch.cuda.memory_reserved = lambda device=None: _flagos.memory_reserved(
            _device_index(device)
        )

        def _memory_stats(device=None):
            """flagos stats under the nested keys torch.cuda callers expect.

            flagos reports flat names (``peak_allocated_bytes``); torch.cuda's
            schema is ``allocated_bytes.all.peak``. Emit both so either style of
            lookup resolves.
            """
            stats = dict(_flagos.memory_stats(_device_index(device)))
            for flat, nested in (
                ("allocated_bytes", "allocated_bytes.all.current"),
                ("peak_allocated_bytes", "allocated_bytes.all.peak"),
                ("reserved_bytes", "reserved_bytes.all.current"),
                ("peak_reserved_bytes", "reserved_bytes.all.peak"),
            ):
                if flat in stats:
                    stats[nested] = stats[flat]
            return stats

        torch.cuda.memory_stats = _memory_stats
        torch.cuda.max_memory_allocated = lambda device=None: _memory_stats(device).get(
            "peak_allocated_bytes", 0
        )
        torch.cuda.max_memory_reserved = lambda device=None: _memory_stats(device).get(
            "peak_reserved_bytes", 0
        )

        # Emptying and resetting go through the same C++ module as the stats
        # above, for the same reason: the CPU wheel builds neither
        # torch._C._cuda_emptyCache nor torch._C._cuda_resetPeakMemoryStats, so
        # every diffusers call site (``empty_device_cache`` at the end of each
        # ``from_pretrained``) and every ``torch.cuda.reset_peak_memory_stats``
        # raised AttributeError. flagos has a single reset that clears both
        # peaks; torch.cuda splits it into reset_peak_memory_stats (both) and
        # reset_max_memory_allocated (allocation peak only), so the narrower name
        # is mapped to the same call rather than left undefined.
        torch.cuda.empty_cache = _flagos.empty_cache
        torch.cuda.reset_peak_memory_stats = lambda device=None: (
            _flagos.reset_peak_memory_stats(_device_index(device))
        )
        torch.cuda.reset_max_memory_allocated = torch.cuda.reset_peak_memory_stats

    # triton reads torch._C._cuda_getCurrentRawStream(idx) -> raw handle.
    try:
        torch._C._cuda_getCurrentRawStream = lambda idx=0: 0
    except Exception:
        pass
    try:
        torch._C._cuda_synchronize = lambda: _synchronize()
    except Exception:
        pass

    # Seeding / RNG source: the CPU-torch wheel ships an EMPTY
    # torch.cuda.default_generators, and torch.manual_seed() -> [nothing on
    # cuda], so flag_gems' generator-less RNG ops (rand/randn/uniform_/...) had
    # no seedable source and were not reproducible. Install per-device CUDA
    # generators (philox 2x int64 state, exactly what gems'
    # philox_backend_seed_offset unpacks) and route cuda seeding to them, so
    # torch.manual_seed(s) -> torch.cuda.manual_seed_all(s) reseeds them and
    # gems RNG becomes reproducible. This is the SAME shape the metax branch
    # uses; it replaces the old _patch_flaggems_philox monkeypatch.
    def _manual_seed(seed):
        seed = int(seed)
        idx = _current_device()
        try:
            _get_cuda_generator(idx).manual_seed(seed)
        except Exception:
            pass

    def _manual_seed_all(seed):
        seed = int(seed)
        try:
            for i in range(max(_device_count(), 1)):
                _get_cuda_generator(i).manual_seed(seed)
        except Exception:
            pass

    torch.cuda.manual_seed = _manual_seed
    torch.cuda.manual_seed_all = _manual_seed_all
    try:
        torch.cuda.default_generators = _CudaDefaultGenerators()
    except Exception:
        pass

    # The user-facing backend is flagos even when FlagGems needs CUDA-shaped
    # Philox state internally. Keep both representations synchronized so the
    # public flagos seed/state API describes the stream consumed by the kernels.
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

    _patch_triton_do_bench()

    _patched = True
    return True


# FlagGems' own name for this accelerator, and the one its nvidia backend
# declares. See patch_flaggems_device_name for why they have to agree.
_FLAGGEMS_DEVICE_NAME = "flagos"
_VENDOR_DEVICE_NAME = "cuda"


def _flag_gems_package_root():
    """Absolute path of the installed flag_gems package directory, or None."""
    try:
        import flag_gems
    except Exception:
        return None
    paths = getattr(flag_gems, "__path__", None)
    if not paths:
        return None
    return os.path.abspath(paths[0]) + os.sep


def _is_flag_gems_module(module, root) -> bool:
    """True for anything loaded out of the flag_gems tree.

    The name test alone is not enough: FlagGems appends its backend directory to
    ``sys.path`` and imports architecture overrides under bare names (``ampere``,
    ``hopper``, ``turing`` for nvidia), so those carry no ``flag_gems`` prefix in
    ``__name__``. They still live inside the tree, which the ``__file__`` test
    catches.
    """
    name = getattr(module, "__name__", "")
    if name == "flag_gems" or name.startswith("flag_gems."):
        return True
    path = getattr(module, "__file__", None)
    return bool(root and path and os.path.abspath(path).startswith(root))


def patch_flaggems_device_name() -> bool:
    """Point FlagGems' device identity at the name torch_fl registered.

    FlagGems decides per op between running its Triton kernel and handing the
    call back to ATen by comparing the input's device type against its own
    backend's device string::

        device = _select_device(a, b)
        if device.type != _DEVICE_NAME:
            return torch.ops.aten.mul.Tensor.redispatch(_FALLBACK_KEYSET, a, b)

    For nvidia that string is ``"cuda"``, because its VendorDescriptor declares
    ``device_name="cuda"``, while a flagos tensor reports ``"flagos"``. The two
    never compare equal, so a module that makes that comparison takes the
    fallback branch for every flagos input rather than only for the ones it means
    to exclude, and its Triton kernel never runs. Eleven of the ``flaggems``
    routes in ``backends_cuda.conf`` land on such a module. The fallback is also
    where the CUDA test failure came from: it redispatches to the ``Tensor``
    overload, which cannot accept the Python scalar that wrapped numbers are now
    handed over as, so ``mask * 1.3333333333333333`` raised "Expected a value of
    type 'Tensor' for argument 'other' but instead found type 'float'".

    Renaming the device is preferable to rewriting the comparison because it
    leaves FlagGems' own bookkeeping intact: ``torch_device_fn`` stays
    ``torch.cuda``, which the rest of this module already points at the flagos
    device, and every ``torch.empty(..., device=device)`` inside FlagGems then
    allocates through the same registered device as the inputs it was handed.
    GCU applies the same remedy in
    ``torch_fl.accelerator.gcu._gcu_compat.patch_flaggems_device_name``.

    Must run after ``import flag_gems``: ``DeviceDetector`` is a singleton and
    copies the name out of the vendor descriptor at construction, and the op
    modules capture it into a module global as they are imported (``device =
    device.name`` in 13 of the generic-path modules, ``_DEVICE_NAME`` in
    ``ops/mul.py``). Correcting the singleton alone would leave each of those
    literals stale, so every already-loaded flag_gems module is rewritten as
    well. Idempotent, and a no-op for any vendor other than nvidia.
    """
    import importlib.util
    import sys

    if importlib.util.find_spec("flag_gems") is None:
        return False
    try:
        from flag_gems.runtime.backend.device_finder import DeviceDetector
    except ImportError:
        try:
            from flag_gems.runtime.backend.device import DeviceDetector
        except ImportError:
            return False

    detector = DeviceDetector()
    if detector.vendor_name != "nvidia":
        return False

    stale = detector.name
    if stale == _FLAGGEMS_DEVICE_NAME:
        return True
    if stale != _VENDOR_DEVICE_NAME:
        # Another descriptor is in charge; not ours to rewrite.
        return False

    detector.name = _FLAGGEMS_DEVICE_NAME

    root = _flag_gems_package_root()
    for module in list(sys.modules.values()):
        if module is None or not _is_flag_gems_module(module, root):
            continue
        for attr in ("device", "_DEVICE_NAME"):
            if getattr(module, attr, None) == stale:
                setattr(module, attr, _FLAGGEMS_DEVICE_NAME)
    return True


def _patch_triton_do_bench():
    """Replace triton.testing.do_bench to avoid CUDA Event timing.

    triton's autotuner benchmarks kernels with ``torch.cuda.Event(
    enable_timing=True)`` and ``torch.empty(device='cuda')``, both of which fail
    on CPU torch. We time with a wall clock instead. Timing only affects
    autotune config *selection*, not kernel correctness -- kernels still run on
    the real GPU via the system libcuda.so.
    """
    try:
        import triton
        import triton.testing
    except ImportError:
        return

    import time
    import statistics

    def _do_bench(
        fn,
        warmup=25,
        rep=100,
        grad_to_none=None,
        quantiles=None,
        return_mode="mean",
        **kwargs,
    ):
        # Warmup
        fn()
        _synchronize()
        # A few timed reps with a wall clock.
        n_rep = 5
        times = []
        for _ in range(n_rep):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            t0 = time.perf_counter()
            fn()
            _synchronize()
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

    # FlagGems' autotuner benchmarks each candidate config through a second
    # triton entry point, ``triton.testing.do_bench_cudagraph``, which is the
    # default for the replay protocol -- see
    # flag_gems/utils/libentry.py's ``_select_benchmark_mode`` and the protocol
    # resolver it calls, which picks replay because the active triton driver is
    # the nvidia one. That helper captures a ``torch.cuda.CUDAGraph``, which this
    # wheel cannot construct (see ``cuda_graph_supported``), so every candidate
    # raises:
    #
    #     RuntimeError: Tried to instantiate dummy base class CUDAGraph
    #
    # ``LibEntry`` catches that per config, prints "[libentry] config ... failed to
    # compile" and records an ``inf`` timing, so the noise is the visible half and
    # the silent half is that no config is ever selected from measurement. Timing
    # it with the same wall clock instead keeps autotuning doing what it is for --
    # picking a config by comparing kernels -- which needs a stable relative
    # number, not graph replay. Only on a build without the graph binding: where
    # torch can capture, replay measures better and stays.
    if not cuda_graph_supported():
        triton.testing.do_bench_cudagraph = _do_bench
