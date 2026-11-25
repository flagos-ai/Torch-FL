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

"""Device surfaces needed by FlagGems and its Triton backends."""

import ctypes
import sys

import torch

from ... import _env
from . import runtime


_MUSA_MEM_GET_INFO = []


def _musa_mem_get_info():
    """The MUSA runtime's ``musaMemGetInfo``, resolved through ctypes.

    torch_fl links the MUSA runtime, so the soname resolves without a path --
    the same lookup the MetaX compat shim uses for mcMemGetInfo. The returned
    callable takes two out-parameters and reports the *current* device; callers
    bind the device they want first.
    """
    if not _MUSA_MEM_GET_INFO:
        runtime = ctypes.CDLL("libmusart.so")
        fn = runtime.musaMemGetInfo
        fn.argtypes = [ctypes.POINTER(ctypes.c_size_t)] * 2
        fn.restype = ctypes.c_int
        _MUSA_MEM_GET_INFO.append(fn)
    return _MUSA_MEM_GET_INFO[0]


def _install_musa_flaggems_compat(flagos) -> None:
    """Expose the MUSA surface expected by FlagGems on top of flagos.

    FlagGems 5.x selects its MThreads backend from ``torch.musa`` and imports
    ``current_device``/``get_device_capability`` from ``torch_musa``. The
    native torch_musa plugin cannot claim PrivateUse1 in the same process, so
    provide only the small compatibility surface required during FlagGems
    discovery. The actual tensor device remains ``flagos``.

    The surface is also read by third-party feature detection. Publishing a
    ``torch_musa`` entry in ``sys.modules`` makes
    ``importlib.util.find_spec("torch_musa")`` succeed, and libraries gate on
    exactly that -- transformers' ``is_torch_musa_available()``, then
    ``set_seed()`` -> ``torch.musa.manual_seed_all``; accelerate's
    ``is_musa_available()``, then ``set_module_tensor_to_device()`` ->
    ``torch.musa.empty_cache``. Whatever the injected module makes detectable
    has to actually answer, so the CUDA-shaped device API is carried below
    rather than left to raise ``AttributeError`` inside an unrelated caller.
    """

    import functools
    import importlib.machinery
    import importlib.util
    import types

    musa = types.ModuleType("torch.musa")
    for name in (
        "device",
        "device_count",
        "current_device",
        "set_device",
        "synchronize",
        "is_available",
        "get_device_properties",
        "current_stream",
        "stream",
        "Event",
        "default_generators",
        "manual_seed",
        "manual_seed_all",
        "initial_seed",
        "get_rng_state",
        "set_rng_state",
    ):
        setattr(musa, name, getattr(flagos, name))

    def _get_device_capability(device=None):
        if device is None:
            device = flagos.current_device()
        props = flagos.get_device_properties(device)
        return props.major, props.minor

    musa.get_device_capability = _get_device_capability
    musa.get_device_name = lambda device=None: (
        flagos.get_device_properties(
            flagos.current_device() if device is None else device
        ).name
    )

    # Device management calls other accelerators route here. FlagGems needs
    # only the device and stream names above, so without these the MUSA branch
    # those libraries open from the detection described above ends in an
    # AttributeError inside a caller that never mentioned MUSA.
    musa.empty_cache = flagos.empty_cache
    musa.memory_allocated = flagos.memory_allocated
    musa.memory_reserved = flagos.memory_reserved
    musa.reset_peak_memory_stats = flagos.reset_peak_memory_stats
    # torch.cuda spells "read the peak watermark" max_memory_allocated and
    # "drop it" reset_max_memory_allocated; flagos tracks one peak per pool, so
    # both reset spellings are the same call. A disabled caching allocator
    # reports no stats at all, hence the default rather than a KeyError.
    musa.max_memory_allocated = lambda device=None: flagos.memory_stats(device).get(
        "peak_allocated_bytes", 0
    )
    musa.max_memory_reserved = lambda device=None: flagos.memory_stats(device).get(
        "peak_reserved_bytes", 0
    )
    musa.reset_max_memory_allocated = flagos.reset_peak_memory_stats
    musa.is_bf16_supported = lambda: torch.bfloat16 in flagos.get_amp_supported_dtype()

    def _mem_get_info(device=None):
        """(free, total) device bytes, matching `torch.cuda.mem_get_info`.

        musaMemGetInfo takes no index and reads whichever device is current, so
        bind the requested one for the call and put the previous one back.
        """
        if device is None:
            device = flagos.current_device()
        elif not isinstance(device, int):
            index = torch.device(device).index
            device = flagos.current_device() if index is None else index
        previous = flagos.current_device()
        restore = device != previous
        if restore:
            flagos.set_device(device)
        try:
            free = ctypes.c_size_t(0)
            total = ctypes.c_size_t(0)
            status = _musa_mem_get_info()(ctypes.byref(free), ctypes.byref(total))
            if status != 0:
                raise RuntimeError(
                    f"musaMemGetInfo failed on MUSA device {device} (error {status})"
                )
            return free.value, total.value
        finally:
            if restore:
                flagos.set_device(previous)

    musa.mem_get_info = _mem_get_info

    def _get_rng_state_all():
        return [flagos.get_rng_state(index) for index in range(flagos.device_count())]

    def _set_rng_state_all(states):
        for index, state in enumerate(states):
            flagos.set_rng_state(state, index)

    musa.get_rng_state_all = _get_rng_state_all
    musa.set_rng_state_all = _set_rng_state_all

    # transformers' Trainer reads these off `torch.<device>.random`, not off the
    # device module itself.
    musa.random = types.ModuleType("torch.musa.random")
    musa.random.get_rng_state = flagos.get_rng_state
    musa.random.set_rng_state = flagos.set_rng_state
    musa.random.get_rng_state_all = _get_rng_state_all
    musa.random.set_rng_state_all = _set_rng_state_all

    # torch.cuda.amp's entry points, bound to the real autocast/GradScaler
    # implementations under this backend's registered device name.
    musa.amp = types.ModuleType("torch.musa.amp")
    musa.amp.autocast = functools.partial(torch.amp.autocast, "flagos")
    musa.amp.GradScaler = functools.partial(torch.amp.GradScaler, "flagos")
    musa.amp.custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="flagos")
    musa.amp.custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="flagos")

    # `torch_musa` lands `Module.musa` the way the CUDA build lands `Module.cuda`,
    # and accelerate's `dispatch_model` reads it off the model unconditionally on
    # its MUSA branch. `generate_methods_for_privateuse1_backend` above produced
    # the same wrapper under this backend's real name, so alias it rather than
    # write a second mover.
    torch.nn.Module.musa = torch.nn.Module.flagos

    musa.__spec__ = importlib.machinery.ModuleSpec(
        name="torch.musa", loader=None, origin="torch_fl_shim"
    )
    torch.musa = musa

    # FlagTree's MThreads benchmark helper allocates its cache with the literal
    # device name "musa" while running under a context that bypasses
    # TorchFunctionMode. Stock PyTorch cannot parse that vendor device name, so
    # translate this one factory call to the renamed PrivateUse1 backend.
    original_empty = torch.empty

    def _musa_empty(*args, **kwargs):
        device = kwargs.get("device")
        if isinstance(device, str) and (device == "musa" or device.startswith("musa:")):
            suffix = device[4:]
            kwargs = {**kwargs, "device": f"flagos{suffix}"}
        return original_empty(*args, **kwargs)

    torch.empty = _musa_empty

    if "torch_musa" not in sys.modules:
        # Preserve access to the installed package's distributed submodule for
        # MCCL fallback without importing torch_musa.__init__, which would claim
        # the process-global PrivateUse1 hooks before torch_fl can own them.
        search_locations = None
        try:
            spec = importlib.util.find_spec("torch_musa")
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            search_locations = spec.submodule_search_locations
        torch_musa = types.ModuleType("torch_musa")
        torch_musa.current_device = flagos.current_device
        torch_musa.set_device = flagos.set_device
        torch_musa.current_stream = flagos.current_stream
        torch_musa.get_device_properties = flagos.get_device_properties
        torch_musa.get_device_name = musa.get_device_name
        torch_musa.get_device_capability = musa.get_device_capability
        torch_musa._MUSAC = types.SimpleNamespace(
            _musa_getCurrentRawStream=flagos._C._get_musa_current_raw_stream
        )
        torch_musa.musa = musa
        torch_musa.__path__ = list(search_locations or ())
        torch_musa.__spec__ = importlib.machinery.ModuleSpec(
            name="torch_musa",
            loader=None,
            origin="torch_fl_shim",
            is_package=True,
        )
        torch_musa.__spec__.submodule_search_locations = torch_musa.__path__
        sys.modules["torch_musa"] = torch_musa

    # ProcessGroupMCCL may already be registered by a linked vendor library.
    # Publishing it through the shim keeps process_group.py's fallback lookup
    # useful even when importing torch_musa.distributed itself is unavailable.
    mccl_cls = getattr(torch.distributed, "ProcessGroupMCCL", None)
    if mccl_cls is not None and "torch_musa.distributed" not in sys.modules:
        distributed = types.ModuleType("torch_musa.distributed")
        distributed.ProcessGroupMCCL = mccl_cls
        distributed.__spec__ = importlib.machinery.ModuleSpec(
            name="torch_musa.distributed", loader=None, origin="torch_fl_shim"
        )
        sys.modules["torch_musa.distributed"] = distributed
        sys.modules["torch_musa"].distributed = distributed


def _install_tsingmicro_flaggems_compat(flagos) -> None:
    """Expose the ``torch.txda`` surface FlagGems' TSINGMICRO backend needs.

    Only built for a TsingMicro wheel that routes at least one op to FlagGems
    (``backends_tsingmicro.conf`` is entirely ``flagos_python``), and inert
    otherwise, mirroring the MUSA shim above.

    torch_fl must NOT import ``torch_txda``: that package hard-links its own
    ``libtorch.so``/``libtorch_cpu.so`` and is ABI-bound to the vendor's torch
    minor, so it fails at import on any other torch line
    (``undefined symbol: c10::TensorImpl::compute_strides_like_channels_last_3d``
    on 2.10) -- and it would additionally claim the process-global PrivateUse1
    key as ``txda``, colliding with torch_fl's ``flagos``. FlagGems only needs a
    few attributes off ``torch.txda`` for its device discovery, device guard and
    Triton launch, so synthesize exactly those from ``torch_fl.flagos`` -- the
    same surface, backed by the TX runtime torch_fl already owns.
    """

    import importlib.machinery
    import types

    txda = types.ModuleType("torch.txda")

    # Straight proxies to the flagos device module.
    for name in (
        "device",
        "device_count",
        "current_device",
        "set_device",
        "synchronize",
        "is_available",
        "get_device_properties",
        "current_stream",
        "stream",
        "Stream",
        "Event",
        "default_generators",
        "empty_cache",
        "mem_get_info",
        "memory_allocated",
        "memory_reserved",
        "manual_seed",
        "initial_seed",
        "get_rng_state",
        "set_rng_state",
    ):
        value = getattr(flagos, name, None)
        if value is not None:
            setattr(txda, name, value)

    # flag_gems' codegen config and its Triton launch grid read
    # `.multi_processor_count` / `.major` / `.minor` off the object, and
    # torch_fl.flagos._DeviceProperties supplies name/total_memory already.
    def _get_device_capability(device=None):
        if device is None:
            device = flagos.current_device()
        props = flagos.get_device_properties(device)
        return props.major, props.minor

    txda.get_device_capability = _get_device_capability
    txda.get_device_name = lambda device=None: (
        flagos.get_device_properties(
            flagos.current_device() if device is None else device
        ).name
    )

    # triton.backends.tsingmicro.driver imports torch_txda at module scope, a
    # requirement torch_fl deliberately does not satisfy. Publish a STUB under
    # that name so the import succeeds and returns the same torch.txda object;
    # the driver then only ever reads torch.txda.{current_device,set_device,
    # current_stream,is_available}, which this shim provides.
    if "torch_txda" not in sys.modules:
        stub = types.ModuleType("torch_txda")
        stub.torch = torch
        stub.txda = txda
        stub.__getattr__ = lambda name: getattr(torch.txda, name)
        stub.__spec__ = importlib.machinery.ModuleSpec(
            name="torch_txda", loader=None, origin="torch_fl_shim"
        )
        sys.modules["torch_txda"] = stub

    txda.__spec__ = importlib.machinery.ModuleSpec(
        name="torch.txda", loader=None, origin="torch_fl_shim"
    )
    torch.txda = txda

    # Qwen-Image-2.1's rope puts a complex64 operand on the device, and FlagGems'
    # Triton kernels have no complex pointer type for the cat/mul it feeds. Patch
    # the rotation to stay real-valued. Inert when diffusers or the 2.1
    # transformer is absent, and skippable via FLAGOS_DISABLE_QWENIMAGE_ROPE.
    from torch_fl.accelerator.tsingmicro._tsingmicro_compat import (
        patch_diffusers_qwenimage21_rope,
    )

    patch_diffusers_qwenimage21_rope()


def install_device_shims(accelerator: str, flagos) -> None:
    """Expose the device modules required by FlagGems' vendor detectors."""
    if (
        accelerator not in ("mthreads", "tsingmicro")
        or not runtime.routes_to_flaggems()
    ):
        return
    if accelerator == "mthreads":
        _install_musa_flaggems_compat(flagos)
    else:
        _install_tsingmicro_flaggems_compat(flagos)


def install_gcu_device_probes(flagos) -> None:
    """Point torch.cuda's device queries at torch_fl on a GCU build.

    The torch wheel here is CPU-only, but two things make torch believe a CUDA
    device exists: torch_fl reports the flagos device through
    `torch.cuda.is_available` (that is how FlagTree's enflame backend finds it)
    and inductor imports `torch.cuda` to write kernels. The next CUDA call then
    runs `_lazy_init()` and raises "Torch not compiled with CUDA enabled" -- from
    dynamo's trace, from inductor's autotune cache, or from a Triton probe, each
    of which the GCU lane has shown in turn.

    `compat.cuda.install_alias` answers part of this, but only when it decides there
    is no real CUDA, and on this lane it decides there is (a probe was flipped to
    True by something else before the decision), so it stands down. This installs
    the same answers unconditionally: the set a CUDA build's device module
    provides, plus a no-op `_lazy_init` so nothing below can reach the CPU wheel's.

    It deliberately leaves `torch.utils._triton` alone. Priming that module's
    probes to False fixes the CUDA errors but hides the working FlagTree enflame
    backend, and inductor then raises `TritonMissing` (`scheduler.create_backend`)
    -- measured on this lane.

    Called at the end of `_phase_claim`, after `torch._register_device_module
    ("flagos", flagos)` and before `_phase_vendor_compat` and the first
    `import flag_gems`, both of which can reach these queries. Remove once torch's
    CUDA getters answer here without `_lazy_init` raising.
    """

    def device_capability(device=None):
        properties = flagos.get_device_properties(device)
        return properties.major, properties.minor

    torch.cuda.is_available = lambda: flagos.device_count() > 0
    torch.cuda.device_count = flagos.device_count
    torch.cuda.current_device = flagos.current_device
    torch.cuda.set_device = flagos.set_device
    torch.cuda.synchronize = flagos.synchronize
    torch.cuda.get_device_properties = flagos.get_device_properties
    torch.cuda.get_device_capability = device_capability
    torch.cuda._lazy_init = lambda: None


def install_metax_cuda_shim() -> None:
    """Expose CUDA queries on the optional MetaX compatibility path."""
    from ...accelerator.metax._metax_compat import (
        is_metax_available,
        patch_torch_cuda_for_metax,
    )

    if is_metax_available():
        patch_torch_cuda_for_metax()


def prepare_cuda(accelerator: str) -> None:
    """
    Prepare torch.cuda for the FlagGems Python runtime and TorchDynamo.

    FlagGems binds ``torch_device_fn = torch.cuda`` on CUDA-shaped vendors and
    enters ``with torch_device_fn.device(tensor.device):`` across its operator
    implementations. Eager execution reaches ``torch.cuda.device.__init__``,
    whereas TorchDynamo's ``CUDADeviceVariable.create`` calls
    ``torch.cuda._get_device_index(device, optional=True)`` directly and emits
    ``torch.cuda._exchange_device`` / ``_maybe_exchange_device`` into the FX
    graph. Extend ``_get_device_index`` (and CPU-only exchange stubs when no
    vendor CUDA shim owns them) to accept ``flagos`` / ``privateuseone``
    devices cleanly.
    """
    import torch.cuda._utils as cuda_utils

    if not getattr(cuda_utils._get_device_index, "_flagos_patched", False):
        _original_cuda_get_device_index = cuda_utils._get_device_index

        def _patched_cuda_get_device_index(
            device, optional: bool = False, allow_cpu: bool = False
        ) -> int:
            if isinstance(device, int):
                return _original_cuda_get_device_index(device, optional, allow_cpu)
            if isinstance(device, str):
                device = torch.device(device)
            if isinstance(device, torch.device) and device.type in (
                "privateuseone",
                "flagos",
            ):
                if device.index is not None:
                    return device.index
                if optional:
                    flagos_mod = getattr(torch, "flagos", None)
                    if flagos_mod is not None and flagos_mod.device_count() > 0:
                        return flagos_mod.current_device()
                    return -1
                raise ValueError(
                    f"Expected a torch.device with a specified index "
                    f"or an integer, but got:{device}"
                )
            return _original_cuda_get_device_index(device, optional, allow_cpu)

        _patched_cuda_get_device_index._flagos_patched = True
        cuda_utils._get_device_index = _patched_cuda_get_device_index
        torch.cuda._get_device_index = _patched_cuda_get_device_index
        for submodule_name in ("memory", "random", "streams"):
            submodule = getattr(torch.cuda, submodule_name, None)
            if submodule is not None and hasattr(submodule, "_get_device_index"):
                submodule._get_device_index = _patched_cuda_get_device_index

    if not getattr(torch.cuda.device.__init__, "_flagos_patched", False):
        _original_cuda_device_init = torch.cuda.device.__init__

        def _patched_cuda_device_init(self, device):
            if isinstance(device, str) and (
                device in ("privateuseone", "flagos")
                or device.startswith(("privateuseone:", "flagos:"))
            ):
                device = torch.device(device)
            if hasattr(device, "type") and hasattr(device, "index"):
                if device.type in ("privateuseone", "flagos"):
                    device = cuda_utils._get_device_index(device, optional=True)
            return _original_cuda_device_init(self, device)

        _patched_cuda_device_init._flagos_patched = True
        torch.cuda.device.__init__ = _patched_cuda_device_init

    if (
        not hasattr(torch._C, "_cuda_exchangeDevice")
        and getattr(torch.cuda._exchange_device, "__module__", "") == "torch.cuda"
    ):

        def _flagos_exchange_device(device: int) -> int:
            if device < 0:
                return -1
            flagos_mod = getattr(torch, "flagos", None)
            if flagos_mod is None or flagos_mod.device_count() == 0:
                return -1
            prev = flagos_mod.current_device()
            if prev != device:
                flagos_mod.set_device(device)
            return prev

        def _flagos_maybe_exchange_device(device: int) -> int:
            if device < 0:
                return -1
            return _flagos_exchange_device(device)

        torch.cuda._exchange_device = _flagos_exchange_device
        torch.cuda._maybe_exchange_device = _flagos_maybe_exchange_device

    # Initialize CUDA only for builds that use its FlagGems Python path.
    # A DCU self-contained wheel selects a hipified libtorch for a stock +cpu torch,
    # which makes is_available() return True even though the CUDA runtime libs are
    # absent, and torch.cuda.init() would fail with "libcaffe2_nvrtc.so: not found".
    # PPU is included: its torch is a real CUDA-13 build with the CUDA runtime libs
    # bundled, so the init works and is what its FlagGems/Triton path relies on.
    if (
        not _env.flag("FLAGOS_DISABLE_FLAGGEMS_PY")
        and accelerator in ("nvidia", "", "thead")
        and torch.cuda.is_available()
    ):
        torch.cuda.init()
