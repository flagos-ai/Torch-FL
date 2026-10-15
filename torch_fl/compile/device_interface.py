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
Device runtime interface that lets TorchInductor treat flagos as a GPU.

Inductor asks a `DeviceInterface` for everything it needs at codegen and
autotune time: current device, streams, hardware properties, Triton capability.
Registering one for flagos is what makes inductor generate Triton kernels for
flagos tensors *directly*, instead of us rewriting the graph to cuda first.

That rewrite is not a cosmetic difference. `at::getAccelerator()` returns
PrivateUse1 (flagos) in this build, and `torch::autograd::Node::stream()` only
returns a stream when a node's input device type equals the accelerator. So a
graph whose inputs were rewritten to cuda produces autograd nodes with no
stream, and the engine trips `opt_ready_stream && opt_parent_stream` (see
engine.cpp:1085) as soon as AOT autograd traces the backward. Keeping the graph
on flagos avoids that entirely -- and avoids a copy-in/copy-out per call.

On CUDA-like builds hardware queries proxy to torch.cuda: flagos runs on the
same physical GPU and its allocator delegates to c10::cuda::CUDACachingAllocator,
so device properties, compute capability and raw streams are the CUDA ones.
Device indices line up too (flagos.set_device(i) moves the CUDA current device).

Ascend has no CUDA runtime at all, so that proxying does not apply there: its
properties come from torch.flagos and its raw stream from the ACL stream registry
(the same aclrtStream torch_fl's own aclnn ops run on). See platform_profile.py
for which build gets which.
"""

from typing import Any, Optional, Union

import torch

from torch._dynamo.device_interface import (
    DeviceInterface,
    caching_worker_current_devices,
    caching_worker_device_properties,
)

from torch_fl.compile.platform_profile import platform_profile


DEVICE_TYPE = "flagos"


# Accelerators whose runtime is *not* CUDA-derived: torch.cuda has no working
# implementation in the process, so every hardware query has to go through
# torch.flagos and the vendor Triton driver instead of proxying to CUDA.
#
# MUSA reaches Triton through MThreads FlagTree; GCU reaches it through
# FlagTree's enflame backend (or Enflame's older triton_gcu plugin), already
# redirected onto flagos by
# torch_fl.accelerator.gcu._gcu_compat.patch_gcu_triton_for_flagos(). Kunlunxin
# reaches it through FlagTree's XPU backend and, like the other two, runs with no
# CUDA runtime at all. The distinction that matters to this module is the same
# for all of them, so keep the checks keyed on this set rather than on a vendor
# name.
_NATIVE_ACCELERATORS = frozenset({"musa", "gcu", "kunlunxin"})


def is_native_accelerator() -> bool:
    """Whether the active accelerator runs without a usable CUDA runtime."""
    from torch_fl._build_config import ACCELERATOR

    return ACCELERATOR in _NATIVE_ACCELERATORS


def _triton_backend() -> tuple[str, str]:
    """Return ``(device_type, triton_backend)`` for the active accelerator.

    flagos has no triton backend of its own, so inductor must report the backend
    of the *underlying hardware*. The type must match the ``GPUTarget.backend``
    understood by the installed compiler: MThreads FlagTree calls this ``musa``
    (its Python backend package is named ``mthreads``), while MetaX and NVIDIA
    use their respective vendor-neutral target names. See platform_profile.py
    for the mapping and the measured evidence behind each entry.
    """
    profile = platform_profile()
    return profile.triton_device_type, profile.triton_backend_key


def _native_warp_size(props: Any) -> int:
    """Warp size for a native accelerator, preferring the vendor Triton driver.

    Inductor's own default is CUDA's 32, which is wrong on GCU (12 on gcu300, 8
    on gcu400, 128 on gcu500) and would misshape every generated kernel. The
    authority is the Triton driver, since it is what compiles the kernel:
    ``GPUTarget.warp_size`` is the same number the backend codegen uses. Fall
    back to the device properties, then to 32, so a driver without target
    resolution still compiles.
    """
    warp_size = getattr(props, "warp_size", None)
    if warp_size:
        return int(warp_size)
    try:
        from triton.runtime import driver

        target = driver.active.get_current_target()
        if getattr(target, "warp_size", None):
            return int(target.warp_size)
    except Exception:
        pass
    return 32


def _triton_current_target() -> Any:
    """The active Triton driver's ``GPUTarget``, or None.

    Split out so tests can substitute a target: ``triton.runtime.driver.active``
    is a read-only property on the driver config object, so it cannot be
    monkeypatched at the use site.
    """
    try:
        from triton.runtime import driver

        return driver.active.get_current_target()
    except Exception:
        return None


def _triton_driver_props(props: Any) -> dict:
    """Layout fields no torch binding supplies, read off the Triton driver.

    ``DeviceProperties.create`` tells Triton how to shape the kernel it is about
    to compile, and inductor's own implementation fills the layout fields from
    ``torch.cuda`` -- `cc` from ``get_compute_capability`` (which reports the
    *arch*, as in ``GPUTarget.arch``) and the rest as optional ``getattr``s off
    the device properties. A build whose torch has no CUDA bindings has neither,
    so the fields have to come from somewhere that does. The Triton driver is
    the authority for exactly this reason: it is what compiles the kernel, and
    its own ``get_current_target()`` already reports ``GPUTarget``, whose
    ``arch`` is the same sm number inductor would have passed.

    Only the fields ``torch.cuda`` would have supplied are returned, so a driver
    that answers nothing degrades to the upstream defaults rather than to a
    wrong number.
    """
    layout: dict = {}
    target = _triton_current_target()
    if target is not None:
        arch = getattr(target, "arch", None)
        if arch is not None:
            # In 64-bit mode arch is "sm_80" rather than 80; keep the number
            # inductor would have handed Triton.
            digits = str(arch).rsplit("_", 1)[-1]
            if digits.isdigit():
                layout["cc"] = int(digits)
        warp_size = getattr(target, "warp_size", None)
        if warp_size is not None:
            layout["warp_size"] = int(warp_size)

    # Prefer the device properties when they carry the value; they are a field
    # of the device itself, the driver target is global.
    for src, dst in (
        ("multi_processor_count", "multi_processor_count"),
        ("major", "major"),
        ("warp_size", "warp_size"),
        ("regs_per_multiprocessor", "regs_per_multiprocessor"),
        ("max_threads_per_multi_processor", "max_threads_per_multi_processor"),
        ("max_threads_per_block", "max_threads_per_block"),
    ):
        value = getattr(props, src, None)
        if value is not None:
            layout[dst] = value
    return layout


def _device_index(device: Any) -> Optional[int]:
    """Normalize str / torch.device / int into a plain device index."""
    if device is None:
        return None
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        return device.index
    return int(device)


def _vendor_device(device: Any = None) -> Any:
    """Device index for the vendor runtime, which takes ints only."""
    idx = _device_index(device)
    if idx is None:
        return torch.flagos.current_device()
    return idx


def _hardware_module():
    """Return the module that answers *hardware* queries for this build.

    CUDA-like builds share the physical GPU with torch.cuda, which reports real
    properties, capability and raw streams. Ascend has no CUDA runtime, so
    torch.flagos is the only source -- routing there would call into a shim that
    is itself backed by flagos, or raise outright.
    """
    if platform_profile().is_cuda_like:
        return torch.cuda
    return torch.flagos


def _raw_stream(device_idx: int) -> int:
    """Return the raw stream handle generated launch code hands to the kernel.

    On Ascend this is the aclrtStream from torch_fl's own stream registry. It
    must not fall back to 0: rt stream 0 is not ordered against the aclnn ops
    producing the kernel's inputs, which silently corrupts results rather than
    failing (see scripts/vendor/patch_triton_ascend.py for the nan-loss regression).
    """
    profile = platform_profile()
    if profile.is_cuda_like:
        return torch._C._cuda_getCurrentRawStream(device_idx)
    if profile.vendor == "musa":
        from torch_fl.compile.flagtree_shim import get_musa_current_raw_stream

        return get_musa_current_raw_stream(device_idx)
    if profile.vendor == "gcu":
        from torch_fl.compile.flagtree_shim import get_gcu_current_raw_stream

        return get_gcu_current_raw_stream(device_idx)
    if profile.vendor == "kunlunxin":
        from torch_fl.compile.flagtree_shim import get_kunlunxin_current_raw_stream

        return get_kunlunxin_current_raw_stream(device_idx)

    from torch_fl.accelerator.ascend.acl_stream import current_acl_raw_stream

    return current_acl_raw_stream(device_idx)


def _set_stream(stream: Any) -> None:
    profile = platform_profile()
    if profile.is_cuda_like:
        torch.cuda.set_stream(stream)
        return
    native = getattr(stream, "_stream", stream)
    if profile.vendor == "musa" and not hasattr(native, "set_current"):
        # Native MUSA currently exposes the default stream only. Reject a
        # foreign stream instead of silently switching CUDA state.
        if getattr(stream, "musa_stream", None) is not None:
            return
        torch.cuda.set_stream(stream)
        return
    if profile.vendor == "kunlunxin":
        # The XPU runtime has one queue per device, denoted by handle 0, and the
        # stream stand-in carries it; there is no other stream to switch to.
        return
    native.set_current()


def _set_stream_by_id(stream_id: int, device_index: int, device_type: int) -> None:
    profile = platform_profile()
    if profile.is_cuda_like or profile.vendor == "musa":
        torch.cuda._set_stream_by_id(
            stream_id=stream_id, device_index=device_index, device_type=device_type
        )
        return
    if profile.vendor == "gcu":
        from torch_fl.accelerator.gcu.tops_stream import TopsStream

        TopsStream.borrowed(stream_id, device=device_index).set_current()
        return
    if profile.vendor == "kunlunxin":
        # Single-queue device (handle 0): there is nothing else to switch to and
        # no torch stream-id registry to switch it through.
        return
    # Native ACL streams have no torch stream-id registry; the id *is* the
    # aclrtStream handle (see AclStream.stream_id), so switch on it directly.
    from torch_fl.accelerator.ascend.acl_stream import AclStream

    AclStream.borrowed(stream_id, device=device_index).set_current()


def _exchange_device(device_idx: int) -> int:
    # `torch.cuda._exchange_device` is only the same thing as switching the
    # flagos device where there is a CUDA runtime for them to agree on. A
    # CUDA-like platform can run the CPU-only pip torch (the default
    # FlagGems-primary wheel), where the call raises "PyTorch was compiled
    # without CUDA support" -- so gate on the torch, not the platform, and go
    # through the flagos runtime otherwise.
    if platform_profile().is_cuda_like and _torch_has_cuda_kernels():
        return torch.cuda._exchange_device(device_idx)
    if device_idx < 0:
        return -1
    previous = torch.flagos.current_device()
    torch.flagos.set_device(device_idx)
    return previous


class FlagOSDeviceInterface(DeviceInterface):
    """Inductor's device runtime interface for the flagos device.

    Mirrors torch._dynamo.device_interface.CudaInterface. Device *state* always
    goes to torch.flagos. Where *hardware* queries go depends on the platform
    profile: torch.cuda on CUDA-like builds (same physical GPU), torch.flagos
    plus the ACL runtime on Ascend, which has no CUDA at all.
    """

    device = torch.flagos.device  # type: ignore[assignment]

    # Inductor captures these through dynamo; flagos ships its own shims that
    # proxy the CUDA streams/events of the same physical GPU.
    Event = torch.flagos.Event  # type: ignore[assignment]
    Stream = torch.flagos.Stream  # type: ignore[assignment]

    class Worker:
        """Property queries that must work in forked compile workers.

        Workers cannot touch the GPU, so properties are recorded in the parent
        process and read from the cache here (same contract as CudaInterface).
        """

        @staticmethod
        def set_device(device: int) -> None:
            caching_worker_current_devices[DEVICE_TYPE] = device

        @staticmethod
        def current_device() -> int:
            if DEVICE_TYPE in caching_worker_current_devices:
                return caching_worker_current_devices[DEVICE_TYPE]
            return torch.flagos.current_device()

        @staticmethod
        def get_device_properties(device: Any = None) -> Any:
            idx = _device_index(device)
            if idx is None:
                idx = FlagOSDeviceInterface.Worker.current_device()

            if DEVICE_TYPE not in caching_worker_device_properties:
                hardware = _hardware_module()
                caching_worker_device_properties[DEVICE_TYPE] = [
                    hardware.get_device_properties(i)
                    for i in range(hardware.device_count())
                ]

            return caching_worker_device_properties[DEVICE_TYPE][idx]

    # --- device state: flagos ------------------------------------------------
    current_device = staticmethod(torch.flagos.current_device)
    device_count = staticmethod(torch.flagos.device_count)
    synchronize = staticmethod(torch.flagos.synchronize)

    # --- streams -------------------------------------------------------------
    # torch.flagos ships the stream/event shims; on CUDA-like builds they proxy
    # the CUDA stream of the same GPU, on Ascend they wrap real ACL streams, and
    # on MThreads they wrap the vendor musa stream.
    stream = staticmethod(torch.flagos.stream)  # type: ignore[assignment]
    current_stream = staticmethod(torch.flagos.current_stream)  # type: ignore[assignment]
    set_stream = staticmethod(_set_stream)  # type: ignore[assignment]
    _set_stream_by_id = staticmethod(_set_stream_by_id)  # type: ignore[assignment]
    # Generated Triton launch code passes this raw stream handle to the kernel.
    get_raw_stream = staticmethod(_raw_stream)  # type: ignore[assignment]

    # --- hardware ------------------------------------------------------------
    memory_allocated = staticmethod(torch.flagos.memory_allocated)
    exchange_device = staticmethod(_exchange_device)  # type: ignore[assignment]
    maybe_exchange_device = staticmethod(_exchange_device)  # type: ignore[assignment]

    @staticmethod
    def get_device_properties(device: Any = None) -> Any:
        return _hardware_module().get_device_properties(_vendor_device(device))

    # Retained for callers that reached for it on the vendor path directly.
    _vendor_device = staticmethod(_vendor_device)  # type: ignore[assignment]

    @staticmethod
    def set_device(device: Any) -> None:
        torch.flagos.set_device(device)

    @staticmethod
    def is_bf16_supported(including_emulation: bool = True) -> bool:
        if platform_profile().is_cuda_like:
            return torch.cuda.is_bf16_supported()
        # Native runtimes advertise their supported autocast dtypes through
        # torch.flagos rather than torch.cuda.
        return torch.bfloat16 in torch.flagos.get_amp_supported_dtype()

    @staticmethod
    def get_compute_capability(device: Any = None) -> Union[int, str]:
        profile = platform_profile()
        if profile.is_cuda_like and _torch_has_cuda_kernels():
            major, minor = torch.cuda.get_device_capability(_device_index(device))
            return major * 10 + minor
        if profile.is_cuda_like:
            # CUDA-like platform on the CPU-only pip torch: `torch.cuda` has no
            # capability to report. The Triton driver is the authority for the
            # arch the kernels are actually compiled for, and on this build the
            # active backend *is* NVIDIA's, so its target carries the same sm
            # number. Fall back to the shared `cc`-from-driver read used for the
            # other vendors that report their capability natively.
            target = _triton_current_target()
            arch = getattr(target, "arch", None)
            digits = str(arch).rsplit("_", 1)[-1] if arch is not None else ""
            if digits.isdigit():
                return int(digits)
            props = torch.flagos.get_device_properties(_vendor_device(device))
            return props.major * 10 + props.minor
        if profile.vendor in ("musa", "gcu"):
            props = torch.flagos.get_device_properties(_vendor_device(device))
            return props.major * 10 + props.minor
        if profile.vendor == "kunlunxin":
            # The XPU device reports no compute-capability pair of its own -- its
            # properties fall back to the Ascend-shaped stand-in -- so read the
            # arch FlagTree's XPU backend compiles for from its driver, the same
            # number that backend puts in GPUTarget.arch.
            return _kunlunxin_arch()
        # Ascend: `cc` reaches Triton as GPUTarget.arch, and the Ascend backend
        # expects a SoC name there (e.g. "Ascend910_9382"), not a number. That is
        # what its own driver reports, so read it from the same place.
        return _ascend_arch()

    @staticmethod
    def is_triton_capable(device: Any = None) -> bool:
        if not platform_profile().is_cuda_like:
            # No CUDA compute-capability floor applies: the vendor Triton
            # backend decides whether it supports the target device.
            return True
        return torch.cuda.get_device_properties(_device_index(device)).major >= 7

    @staticmethod
    def raise_if_triton_unavailable(device: Any = None) -> None:
        import inspect
        import triton.backends

        if not FlagOSDeviceInterface.is_triton_capable(device):
            from torch._inductor.exc import GPUTooOldForTriton

            raise GPUTooOldForTriton(
                FlagOSDeviceInterface.get_device_properties(device),
                inspect.currentframe(),
            )

        _, triton_backend = _triton_backend()
        if triton_backend not in triton.backends.backends:
            from torch_fl._build_config import ACCELERATOR

            hint = {
                "musa": "install the MThreads FlagTree runtime for MUSA",
                "gcu": (
                    "install Enflame's triton_gcu plugin and the /opt/triton_gcu "
                    "toolchain (see docs/vendors/gcu/installation.md)"
                ),
            }.get(ACCELERATOR, "install the vendor Triton runtime")
            raise RuntimeError(
                f"triton not built with the '{triton_backend}' backend; {hint}"
            )

    @staticmethod
    def is_available() -> bool:
        return torch.flagos.device_count() > 0


def _ascend_arch(_cache: list = []) -> str:
    """SoC name for this chip, as the Ascend Triton backend expects in arch.

    Queried once from the installed backend's own driver so the value cannot
    drift from what ``driver.active.get_current_target()`` would report. On a
    real 910 this returns e.g. ``Ascend910_9382``.
    """
    if not _cache:
        from triton.backends.ascend.driver import NPUUtils

        _cache.append(str(NPUUtils().get_arch()))
    return _cache[0]


def _kunlunxin_arch(_cache: list = []) -> int:
    """Arch the FlagTree XPU backend compiles for, as ``GPUTarget.arch``.

    Queried once from the installed backend's own driver so the value cannot
    drift from what ``driver.active.get_current_target()`` would report. The XPU
    device reports no compute-capability pair to derive it from, so the driver is
    the only authority.
    """
    if not _cache:
        from triton.runtime import driver

        _cache.append(int(driver.active.get_current_target().arch))
    return _cache[0]


def _register_gpu_type() -> None:
    """Teach inductor that flagos is a GPU, not a CPU-like device.

    Two separate things read GPU_TYPES:

    * `is_gpu()` -- a membership test. Without flagos in the list inductor picks
      the C++/CPU codegen path and never emits Triton. The append must be in
      place; callers captured this exact list object at import time.

    * `get_gpu_type()` -- picks *the* single GPU type, asserting at most one
      entry of GPU_TYPES is available. torch_fl's torch.cuda shim reports
      available alongside flagos, so that assert would fire (post_grad's
      ConstructorMoverPass hits it). It is `functools.cache`d, so we prime the
      cache with flagos while the list is temporarily narrowed, and every later
      caller gets the memoized answer.
    """
    from torch._inductor import utils as inductor_utils

    if DEVICE_TYPE in inductor_utils.GPU_TYPES:
        return

    saved = list(inductor_utils.GPU_TYPES)
    try:
        inductor_utils.GPU_TYPES[:] = [DEVICE_TYPE]
        inductor_utils.get_gpu_type()
    finally:
        inductor_utils.GPU_TYPES[:] = saved
        inductor_utils.GPU_TYPES.append(DEVICE_TYPE)


def _patch_device_properties() -> None:
    """Report flagos devices to the Triton layer under the hardware's backend.

    ``DeviceProperties.type`` is what inductor forwards to Triton as
    ``GPUTarget.backend`` (hints.py -> triton_heuristics.py:718 -> triton's
    make_backend). flagos has no triton backend of its own, so we rewrite the
    type to the underlying hardware's name: ``maca`` on MetaX (triton-metax),
    ``cuda`` elsewhere (triton's NVIDIA backend). The interface lookup and every
    other property stay on torch.cuda -- flagos shares the same physical GPU.

    On a CUDA-like platform whose torch has no CUDA bindings (the default
    FlagGems-primary wheel) torch.cuda cannot answer those property queries at
    all, so the record is built from ``FlagOSDeviceInterface`` instead.

    We wrap ``create`` and rewrite the ``type`` of the result (rather than only
    editing the read sites) so the functools cache and every other field keep
    working unchanged.
    """
    from torch._inductor.runtime.hints import DeviceProperties

    if getattr(DeviceProperties.create, "_flagos_patched", False):
        return

    original = DeviceProperties.create
    device_type, _ = _triton_backend()
    profile = platform_profile()
    cuda_like = profile.is_cuda_like
    vendor = profile.vendor

    def create(device: Any) -> Any:
        is_flagos = device is not None and getattr(device, "type", None) == DEVICE_TYPE
        if is_flagos and cuda_like and not _torch_has_cuda_kernels():
            # A CUDA-like platform can still link the CPU-only pip torch -- the
            # default FlagGems-primary wheel does. `torch.cuda` answers nothing
            # there while still looking available (torch_fl aliases the probe to
            # the flagos device count, see accelerator/cuda/_cuda_compat.py), so
            # the translation below sends inductor into `torch.cuda`'s own
            # property path. That path only ever works on a CUDA-enabled torch:
            # `_lazy_init` raises "Torch not compiled with CUDA enabled" before
            # it defines the module-level `_get_device_properties` that
            # `get_device_properties` then calls -- and torch_fl itself sets
            # `torch.cuda._initialized`, so the guard that would have raised is
            # skipped and the call instead dies with
            # "NameError: name '_get_device_properties' is not defined".
            #
            # Build the record from our own interface, which reads the same
            # physical device through torch.flagos, exactly as the native
            # runtimes below do. Only the layout fields differ from what
            # torch.cuda would have returned on a CUDA torch: `cc` and the
            # warp/register counts come off the Triton driver instead.
            interface = FlagOSDeviceInterface
            props = interface.get_device_properties(device)
            layout = _triton_driver_props(props)
            return DeviceProperties(
                type=device_type,
                index=device.index,
                # Required, unlike the layout fields below: a device that does
                # not report it still has to produce a record, and 1 is the
                # value inductor's own CpuProperties uses.
                multi_processor_count=layout.pop("multi_processor_count", 1),
                cc=layout.pop("cc", interface.get_compute_capability(device)),
                **layout,
            )
        if is_flagos and cuda_like:
            # Interface/property lookup goes through torch.cuda (same GPU); only
            # the type reported onward becomes `maca`/`cuda`. The non-CUDA
            # runtimes keep the flagos device here: there is no torch.cuda to
            # look up, and our own interface is registered for flagos.
            device = torch.device("cuda", device.index or 0)
        elif is_flagos and vendor in ("musa", "gcu", "kunlunxin"):
            # These vendor implementations of DeviceProperties.create cannot
            # read a flagos device, so build the record from our interface.

            interface = FlagOSDeviceInterface
            props = interface.get_device_properties(device)
            return DeviceProperties(
                type=device_type,
                index=device.index,
                multi_processor_count=props.multi_processor_count,
                cc=interface.get_compute_capability(device),
                major=getattr(props, "major", None),
                regs_per_multiprocessor=getattr(props, "regs_per_multiprocessor", None),
                max_threads_per_multi_processor=getattr(
                    props, "max_threads_per_multi_processor", None
                ),
                warp_size=_native_warp_size(props),
            )
        result = original(device)
        if is_flagos:
            result = result._replace(type=device_type)
        return result

    create._flagos_patched = True  # type: ignore[attr-defined]
    DeviceProperties.create = create  # type: ignore[method-assign, assignment]


def _patch_inductor_benchmark_device() -> None:
    """Teach inductor's benchmarker that the Triton backend name is not a device.

    ``_patch_device_properties`` reports ``DeviceProperties.type`` as the *Triton*
    backend name so triton picks the right backend -- ``maca`` on MetaX. But
    inductor also forwards that same string to its benchmarker as a **torch**
    device (triton_heuristics.py:933 -> benchmarking.py `torch.device(device)`),
    and ``maca`` is not a torch device type, so autotuning dies with
    "Expected one of cpu, cuda, ... at start of device string: maca".

    The vendor MetaX torch build patches this in-tree (a ``# USE_MACA`` branch
    mapping ``maca`` -> ``cuda`` in ``Benchmarker.benchmark``). The official CPU
    wheel we ship against has no such patch, so we apply the same mapping here.
    Benchmarking on cuda is correct on a CUDA-like build: flagos and cuda are the
    same physical GPU. On Ascend and MUSA there is no CUDA runtime to name, so the
    replacement is ``flagos`` -- the device the tensors are actually on. Either
    way the argument only selects cpu- versus gpu-style benchmarking.

    No-op when the Triton backend name is already a valid torch device (the CUDA
    build reports ``cuda``), so this costs nothing off MetaX.
    """
    device_type, _ = _triton_backend()
    if device_type == "cuda":
        return

    torch_device = "cuda" if platform_profile().is_cuda_like else DEVICE_TYPE

    from torch._inductor.runtime import benchmarking

    benchmarker_cls = benchmarking.Benchmarker
    original = benchmarker_cls.benchmark
    if getattr(original, "_flagos_patched", False):
        return

    def benchmark(
        self: Any,
        fn: Any,
        fn_args: Any = None,
        fn_kwargs: Any = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> float:
        if isinstance(device, str) and device == device_type:
            device = torch_device
        return original(self, fn, fn_args, fn_kwargs, device=device, **kwargs)

    benchmark._flagos_patched = True  # type: ignore[attr-defined]
    benchmarker_cls.benchmark = benchmark  # type: ignore[method-assign]


def _repair_stock_cuda_interface() -> None:
    """Make the stock `CudaInterface` usable on a build with no CUDA runtime.

    inductor's *runtime* launcher resolves its device interface by the
    *reported* device type -- `triton_heuristics.get_device_interface` does
    `get_interface_for_device(device_props.type)`. flagos reports that type as
    "cuda" (see `_patch_device_properties`), so inductor reaches the stock
    `CudaInterface` rather than `FlagOSDeviceInterface`.

    The platforms that report "cuda" without being able to use it are Kunlunxin
    and a CUDA-like platform whose torch has no CUDA bindings -- the default
    FlagGems-primary wheel. There `register_flagos_device_interface` claims the
    name for `FlagOSDeviceInterface`, and this function is the safety net for the
    paths that hold a direct reference to the stock class instead of going
    through the registry. Two of its members are bound at import time from a
    `torch.cuda` probe:

    * `get_raw_stream` -> None when `torch.cuda._is_compiled()` is false, so the
      autotuner raises "'NoneType' object is not callable";
    * `exchange_device` / `maybe_exchange_device` -> `torch.cuda._exchange_device`,
      whose CPU-wheel fallback raises "PyTorch was compiled without CUDA
      support" -- and `DeviceGuard.__enter__` calls exactly that before every
      kernel launch.

    The two repairs are gated differently, because the underlying probes differ.
    The raw-stream getter is missing on *every* flagos build, the CUDA lane
    included: torch_fl ships the CPU torch wheel and installs its own
    `torch._C._cuda_getCurrentRawStream` from `accelerator/cuda/_cuda_compat.py`,
    but `torch._dynamo.device_interface` binds `get_raw_stream` from the
    torch.cuda probe at import time -- before that shim lands -- so it is left at
    None even where a CUDA runtime exists. It is therefore repaired
    unconditionally. `exchange_device` is only wrong where there is no CUDA
    runtime to dispatch to: on a CUDA-enabled torch the stock binding is already
    correct and is left alone, and the gate is therefore the torch, not the
    platform.
    """
    import torch._dynamo.device_interface as di

    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if getter is not None and di.CudaInterface.get_raw_stream is None:
        di.get_cuda_stream = getter
        di.CudaInterface.get_raw_stream = staticmethod(getter)  # type: ignore[assignment]

    # `get_device_properties` shares the raw-stream binding's shape one step
    # further along: `torch._dynamo.device_interface` snapshots
    # `torch.cuda.get_device_properties` into the class at import time, and on a
    # CUDA-like build torch_fl then replaces the *module* attribute with a vendor
    # getter that does not call `_lazy_init` (metax's
    # `_patched_get_device_properties`, dcu/ppu's equivalent). When dynamo is
    # imported before torch_fl -- the normal order in any pytest/transformers run
    # -- the class keeps the stock getter, which on a vendor torch reaches
    # `_lazy_init()` and then an undefined `_get_device_properties`
    # (NameError). `_patch_device_properties` reports a flagos device as
    # `torch.device("cuda", i)`, so `DeviceProperties.create` (and through it
    # flex_attention's lowering, hints.py) resolves exactly this class. Repair it
    # unconditionally -- the two bindings agree on an untouched torch, so this is
    # a no-op there.
    if di.CudaInterface.get_device_properties is not torch.cuda.get_device_properties:
        di.CudaInterface.get_device_properties = staticmethod(  # type: ignore[assignment]
            torch.cuda.get_device_properties
        )

    if platform_profile().is_cuda_like and _torch_has_cuda_kernels():
        return

    di.CudaInterface.exchange_device = staticmethod(_exchange_device)  # type: ignore[assignment]
    di.CudaInterface.maybe_exchange_device = staticmethod(_exchange_device)  # type: ignore[assignment]


def _prime_has_triton() -> None:
    """Make `torch.utils._triton.has_triton()` see the flagos device.

    `has_triton()` walks a hard-coded table of device types -- cuda, xpu, cpu,
    mtia -- and asks each one's interface whether it is available. flagos is not
    in that table, so the answer depends on something unrelated: torch_fl aliases
    `torch.cuda.is_available` to the flagos device count, which happens to make
    the "cuda" row pass. With `FLAGOS_ALIAS_CUDA=0` it does not, and then
    `Scheduler.create_backend` raises `TritonMissing` for a flagos graph even
    though Triton is installed and working.

    The table is a local inside `has_triton`, so it cannot be extended. The
    function is `functools.cache`d instead, so we prime that cache while the
    table's rows temporarily resolve to flagos, and every later caller --
    including the module-level `from torch.utils._triton import has_triton`
    bindings inductor already made -- gets the memoized answer.

    The row we borrow is "xpu", whose extra check is `_return_true`: the whole
    condition is then `has_triton_package() and flagos.device_count() > 0`,
    which is the same contract as xpu/mtia and all that holds on Ascend (there
    is no compute-capability floor; the toolchain accepts the SoC or refuses it
    at compile time). "cuda" is masked out for the duration so the probe cannot
    fall into `CudaInterface.Worker.get_device_properties`, which walks *every*
    device.

    Only the answer is primed, not forced: with no Triton installed the real
    check still returns False, and that False is what gets memoized.
    """
    if platform_profile().is_cuda_like:
        return

    from torch.utils import _triton

    # On a native accelerator this function has already been replaced outright by
    # the `has_native_triton` override installed at the end of
    # register_flagos_device_interface(), which is a plain function and answers
    # True unconditionally. There is no cache left to prime, and nothing to gain
    # from priming one: `cache_info` is absent, so probing it raises. This is the
    # second and every later call -- `import torch_fl` registers once, and the
    # compile backend re-registers before each compile_fx.
    cache_info = getattr(_triton.has_triton, "cache_info", None)
    if cache_info is None:
        return

    if cache_info().currsize:
        return

    import torch._dynamo.device_interface as di

    class _Unavailable(DeviceInterface):
        @staticmethod
        def is_available() -> bool:
            return False

    # Populate the built-in table first; otherwise has_triton's own lookup runs
    # init_device_reg() and overwrites the entries installed below.
    di.get_interface_for_device("cuda")

    saved = {name: di.device_interfaces[name] for name in ("cuda", "xpu")}
    di.device_interfaces["cuda"] = _Unavailable
    di.device_interfaces["xpu"] = FlagOSDeviceInterface
    try:
        _triton.has_triton()
    finally:
        di.device_interfaces.update(saved)


def _torch_has_cuda_kernels() -> bool:
    """Whether this torch can actually allocate a CUDA tensor.

    `torch._C._cuda_getDeviceCount` is compiled in only for a CUDA-enabled torch
    build, so it answers the one question `platform_profile().is_cuda_like` cannot:
    where the *torch* stands, not where the platform stands. A CUDA-like platform
    can still link the CPU-only pip wheel -- the default FlagGems-primary build
    does, on the cuda platform -- and there `aten::empty` has no CUDA kernel at
    all, while `torch.cuda.is_available()` still answers True from torch_fl's own
    shim (accelerator/cuda/_cuda_compat.py) because FlagGems' Triton kernels never
    go through torch's CUDA layer and the probe has to report a real device for
    them.

    The probe is deliberately read off torch rather than off
    `torch_fl._env.build_kernels()`: a source-tree run has no build record, so an
    empty kernel set would read as "boxing off" on a wheel that does have it. The
    same question is asked by `inductor_backend._patch_cuda_rng_for_cpu_torch`,
    which keys on the sibling `torch._C._cuda_getDevice` binding.
    """
    return hasattr(torch._C, "_cuda_getDeviceCount")


def _patch_joint_graph_pattern_init() -> None:
    """Pre-trace inductor's replacement patterns on CPU, not on absent CUDA.

    `joint_graph.lazy_init()` traces the pad_mm, SDPA and misc replacement
    patterns before any lowering runs. All three choose their trace device the
    same way -- `device = "cuda" if torch.cuda.is_available() else "cpu"` -- and
    torch_fl aliases that probe to the flagos device count (__init__.py
    _alias_cuda_to_flagos), so on a build with no CUDA runtime they ask for cuda
    anyway and fail two different ways:

    * `FakeTensor.__new__` -> `init_gpu_context` does a real
      `torch.empty(1, device="cuda")`, which raises "Torch not compiled with CUDA
      enabled" in `torch.cuda._lazy_init`;
    * `torch.tensor(2.0, device="cuda")` (the SDPA inv_scale) raises "PyTorch is
      not linked with support for cuda devices" straight from the factory.

    Either one aborts *every* compile_fx on Ascend before codegen is reached.

    The device only decides where a throwaway tracing tensor lives -- upstream
    picks cuda solely as a workaround for pytorch#97894, and the patterns
    themselves are device-independent (they are matched against the real graph
    afterwards, and cpu is what upstream uses on any non-CUDA build). So the fix
    is to make the probe answer honestly for the duration of that one call.

    The gate is `_torch_has_cuda_kernels()`, not `platform_profile()
    .is_cuda_like`: the CUDA platform on the default FlagGems-primary wheel is
    CUDA-like (the flagos device is the same physical GPU) while the torch it
    links is the CPU-only pip wheel, so the trace dies there too, now with
    `NotImplementedError: Could not run 'aten::empty.memory_format' with
    arguments from the 'CUDA' backend` out of `_pad_mm_init`.

    `lazy_init` is `functools.cache`d, so this is a one-shot priming: after it,
    the patterns are registered and the real probe is back in place.
    """
    if _torch_has_cuda_kernels():
        # A torch that can allocate a CUDA tensor: the upstream cuda trace works,
        # on the CUDA-ABI vendor wheels as well.
        return

    from torch._inductor.fx_passes import joint_graph

    if joint_graph.lazy_init.cache_info().currsize:
        return

    saved = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        joint_graph.lazy_init()
    finally:
        torch.cuda.is_available = saved


def _patch_benchmarker() -> None:
    """Point inductor's autotune benchmarking at APIs this build actually has.

    Gated on `_torch_has_cuda_kernels()`, the same probe the joint-graph priming
    uses: a CUDA-like build that links a CUDA-enabled torch keeps the stock
    behaviour, while a CUDA-like platform on the default FlagGems-primary wheel
    needs the patches as much as Ascend does. Two things break otherwise:

    * The `benchmarking.benchmarker` singleton is chosen at import time as
      `InductorBenchmarker() if use_experimental_benchmarker else
      TritonBenchmarker()`, and that flag is `config.use_experimental_benchmarker
      and torch.cuda.is_available()` -- True here, because torch_fl aliases
      `torch.cuda.is_available` to the flagos device count (__init__.py
      _alias_cuda_to_flagos). `InductorBenchmarker` then wants two things a
      CUDA-less build cannot give it: `torch.cuda.Event(enable_timing=True)`,
      which is a dummy base class that raises on construction on the CPU torch
      wheel, and the `torch.empty(L2_size // 4, device="cuda")` it flushes the
      L2 cache with, which no shim can cover for -- a CUDA-like platform's shim
      does install a working Event from torch.flagos (_cuda_compat.py), and no
      more. `TritonBenchmarker` instead defers to triton's own `do_bench`, which
      takes its events from
      `driver.active.get_device_interface()` (torch.flagos, so real ACL events).

    * `CachingAutotuner.bench` passes `device=self.device_props.type`, which is
      the *Triton* backend name -- "npu" -- not a torch device type, so
      `torch.device("npu")` raises. That argument only selects cpu- versus
      gpu-style benchmarking, so flagos is the right translation.
      `_patch_inductor_benchmark_device` already rewrites that name on the
      `Benchmarker` *class*; this one repeats it on the singleton because
      swapping `benchmark_gpu` above only takes effect for calls that reach this
      instance, and the instance attribute shadows the class method.

    Both are patched on the singleton instance, not the module attribute:
    inductor modules do `from .runtime.benchmarking import benchmarker`, so
    rebinding the module attribute would miss every binding already imported.
    """
    if platform_profile().is_cuda_like and _torch_has_cuda_kernels():
        return

    from torch._inductor.runtime import benchmarking

    obj = benchmarking.benchmarker
    if getattr(obj, "_flagos_patched", False):
        return

    if isinstance(obj, benchmarking.InductorBenchmarker):
        obj.benchmark_gpu = benchmarking.TritonBenchmarker.benchmark_gpu.__get__(obj)

    triton_device_type, _ = _triton_backend()
    original_benchmark = obj.benchmark

    def benchmark(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("device") == triton_device_type:
            kwargs = {**kwargs, "device": DEVICE_TYPE}
        return original_benchmark(*args, **kwargs)

    obj.benchmark = benchmark
    obj._flagos_patched = True


def register_flagos_device_interface() -> None:
    """Register the flagos device interface + GPU type with inductor.

    Idempotent; called by the compile backend before each compile_fx.
    """
    import torch._dynamo.device_interface as dyn_di
    from torch._dynamo.device_interface import register_interface_for_device

    # Populate inductor's table *before* publishing anything into it.
    # `init_device_reg` writes the built-in rows unconditionally --
    # `register_interface_for_device("cuda", CudaInterface)` among them -- and
    # runs lazily on the first `get_interface_for_device`/`get_registered_...`
    # call. Registering first and letting that run later silently discards the
    # claim: the table holds FlagOSDeviceInterface for "cuda" right after
    # registration and CudaInterface by the time the runtime launcher asks. It is
    # only ever entered when the flag is unset, so this is a no-op once inductor
    # has been initialised (`_prime_has_triton` relies on the same property).
    dyn_di.get_interface_for_device("cpu")

    _register_gpu_type()
    _patch_device_properties()
    _patch_inductor_benchmark_device()
    _repair_stock_cuda_interface()
    _patch_joint_graph_pattern_init()
    _patch_benchmarker()
    register_interface_for_device(DEVICE_TYPE, FlagOSDeviceInterface)
    _prime_has_triton()
    # The triton-reported device type ("maca" on MetaX) is what the runtime
    # launcher resolves via `triton_heuristics.get_device_interface`, so that
    # name needs an interface too. It proxies the same hardware as flagos.
    #
    # On a CUDA-enabling torch that name is "cuda" and inductor has already
    # registered the stock CudaInterface for it, which is correct there and must
    # be left alone. The collision is between the *platform* and the *torch*:
    # flagos reports the target as "cuda" wherever the vendor Triton backend is
    # NVIDIA's -- Kunlunxin's XPU backend, and a CUDA-like platform whose torch is
    # the CPU-only pip wheel (the default FlagGems-primary build). Neither has a
    # CUDA runtime, so the stock CudaInterface is wrong for both: it binds a dozen
    # members straight off `torch.cuda` at import time (current_device, synchronize,
    # get_device_properties, exchange_device, ...), and the runtime launcher calls
    # several of them before every kernel launch -- `_make_launchers` alone does
    # `DeviceGuard(device_interface, ...)` (exchange_device) followed by
    # `synchronize(current_device())`. On a CPU torch those die with "PyTorch was
    # compiled without CUDA support" / "module 'torch._C' has no attribute
    # '_cuda_getDevice'". The gate is therefore the torch, not the platform.
    device_type, _ = _triton_backend()
    if device_type != "cuda" or not (
        platform_profile().is_cuda_like and _torch_has_cuda_kernels()
    ):
        register_interface_for_device(device_type, FlagOSDeviceInterface)

    # torch.utils._triton caches this result and may have been queried during
    # torch import, before torch_fl registered PrivateUse1. Invalidate it after
    # publishing the native MUSA interface so Inductor accepts the vendor Triton
    # package for the custom GPU device.
    try:
        from torch.utils import _triton as triton_utils

        triton_utils.has_triton.cache_clear()

        if is_native_accelerator() and triton_utils.has_triton_package():
            # PyTorch's helper only knows CUDA/XPU/PrivateUse1 at import time;
            # the CPU wheel's CUDA probe is false even though the vendor runtime
            # owns PrivateUse1. Inductor imports this helper into several
            # modules, so update those bound references as well as the canonical
            # helper.
            def has_native_triton() -> bool:
                return True

            triton_utils.has_triton = has_native_triton
            import sys

            for module_name in (
                "torch._inductor.scheduler",
                "torch._inductor.compile_fx",
                "torch._inductor.async_compile",
            ):
                module = sys.modules.get(module_name)
                if module is not None and hasattr(module, "has_triton"):
                    module.has_triton = has_native_triton
    except (ImportError, AttributeError):
        pass
