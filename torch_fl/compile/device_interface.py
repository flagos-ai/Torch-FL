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

Hardware queries proxy to torch.cuda: flagos runs on the same physical GPU and
its allocator delegates to c10::cuda::CUDACachingAllocator, so device
properties, compute capability and raw streams are the CUDA ones. Device
indices line up too (flagos.set_device(i) moves the CUDA current device).
"""

from typing import Any, Optional, Union

import torch

from torch._dynamo.device_interface import (
    DeviceInterface,
    caching_worker_current_devices,
    caching_worker_device_properties,
)


DEVICE_TYPE = "flagos"


def _triton_backend() -> tuple[str, str]:
    """Return ``(device_type, triton_backend)`` for the active accelerator.

    The type is the target name passed to Triton through Inductor's
    ``DeviceProperties``.  It must match the ``GPUTarget.backend`` understood by
    the installed compiler: MThreads FlagTree calls this ``musa`` (the Python
    backend package is named ``mthreads``), while MetaX and NVIDIA use their
    respective vendor-neutral target names.
    """
    from torch_fl._build_config import ACCELERATOR

    if ACCELERATOR == "musa":
        return "musa", "mthreads"
    if ACCELERATOR == "metax":
        return "maca", "metax"
    return "cuda", "nvidia"


def _device_index(device: Any) -> Optional[int]:
    """Normalize str / torch.device / int into a plain device index."""
    if device is None:
        return None
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        return device.index
    return int(device)


class FlagOSDeviceInterface(DeviceInterface):
    """Inductor's device runtime interface, backed by flagos + torch.cuda.

    Mirrors torch._dynamo.device_interface.CudaInterface. Anything touching
    *hardware* goes to torch.cuda (same GPU); anything touching *device state*
    goes to torch.flagos so the two stay in sync.
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
                from torch_fl._build_config import ACCELERATOR

                if ACCELERATOR == "musa":
                    caching_worker_device_properties[DEVICE_TYPE] = [
                        torch.flagos.get_device_properties(i)
                        for i in range(torch.flagos.device_count())
                    ]
                else:
                    caching_worker_device_properties[DEVICE_TYPE] = [
                        torch.cuda.get_device_properties(i)
                        for i in range(torch.cuda.device_count())
                    ]

            return caching_worker_device_properties[DEVICE_TYPE][idx]

    # --- device state: flagos ------------------------------------------------
    current_device = staticmethod(torch.flagos.current_device)
    device_count = staticmethod(torch.flagos.device_count)
    synchronize = staticmethod(torch.flagos.synchronize)

    # --- streams: use the vendor stream for native MUSA/FlagTree, and proxy
    # CUDA's stream state for the boxing-backed accelerators. -----------------
    stream = staticmethod(torch.flagos.stream)  # type: ignore[assignment]
    current_stream = staticmethod(torch.flagos.current_stream)  # type: ignore[assignment]
    _set_stream_by_id = staticmethod(torch.cuda._set_stream_by_id)  # type: ignore[assignment]

    @staticmethod
    def get_raw_stream(device_idx: int) -> int:
        from torch_fl._build_config import ACCELERATOR

        if ACCELERATOR == "musa":
            from torch_fl.compile.flagtree_shim import get_musa_current_raw_stream

            return get_musa_current_raw_stream(device_idx)
        return torch._C._cuda_getCurrentRawStream(device_idx)

    # --- hardware: same physical GPU as cuda for boxing, native properties
    # for MUSA where torch.cuda is deliberately unavailable. ------------------
    @staticmethod
    def get_device_properties(device: Any = None) -> Any:
        from torch_fl._build_config import ACCELERATOR

        if ACCELERATOR == "musa":
            return torch.flagos.get_device_properties(
                FlagOSDeviceInterface._vendor_device(device)
            )
        return torch.cuda.get_device_properties(device)

    memory_allocated = staticmethod(torch.flagos.memory_allocated)

    @staticmethod
    def _vendor_device(device: Any = None) -> Any:
        """Device index for the vendor runtime, which takes ints only."""
        if device is None:
            return torch.flagos.current_device()
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(device, torch.device):
            if device.index is None:
                return torch.flagos.current_device()
            return device.index
        return int(device)

    @staticmethod
    def set_stream(stream: Any) -> None:
        from torch_fl._build_config import ACCELERATOR

        if ACCELERATOR == "musa":
            native = getattr(stream, "_stream", stream)
            if hasattr(native, "set_current"):
                native.set_current()
                return
            # Native MUSA currently exposes the default stream only.  Reject a
            # foreign stream instead of silently switching CUDA state.
            if getattr(stream, "musa_stream", None) is not None:
                return
        torch.cuda.set_stream(stream)

    @staticmethod
    def set_device(device: Any) -> None:
        torch.flagos.set_device(device)

    @staticmethod
    def exchange_device(device: int) -> int:
        previous = torch.flagos.current_device()
        torch.flagos.set_device(device)
        return previous

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        return FlagOSDeviceInterface.exchange_device(device)

    @staticmethod
    def is_bf16_supported(including_emulation: bool = True) -> bool:
        from torch_fl._build_config import ACCELERATOR

        if ACCELERATOR == "musa":
            return torch.bfloat16 in torch.flagos.get_amp_supported_dtype()
        return torch.cuda.is_bf16_supported()

    @staticmethod
    def get_compute_capability(device: Any = None) -> Union[int, str]:
        from torch_fl._build_config import ACCELERATOR

        if ACCELERATOR == "musa":
            props = torch.flagos.get_device_properties(
                FlagOSDeviceInterface._vendor_device(device)
            )
            return props.major * 10 + props.minor
        major, minor = torch.cuda.get_device_capability(_device_index(device))
        return major * 10 + minor

    @staticmethod
    def is_triton_capable(device: Any = None) -> bool:
        from torch_fl._build_config import ACCELERATOR

        if ACCELERATOR == "musa":
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
            raise RuntimeError(
                f"triton not built with the '{triton_backend}' backend; "
                "install the MThreads FlagTree runtime for MUSA"
            )

    @staticmethod
    def is_available() -> bool:
        return torch.flagos.device_count() > 0


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

    We wrap ``create`` and rewrite the ``type`` of the result (rather than only
    editing the read sites) so the functools cache and every other field keep
    working unchanged.
    """
    from torch._inductor.runtime.hints import DeviceProperties

    if getattr(DeviceProperties.create, "_flagos_patched", False):
        return

    original = DeviceProperties.create
    device_type, _ = _triton_backend()

    def create(device: Any) -> Any:
        is_flagos = device is not None and getattr(device, "type", None) == DEVICE_TYPE
        if is_flagos:
            from torch_fl._build_config import ACCELERATOR

            if ACCELERATOR != "musa":
                # Boxing-backed accelerators expose CUDA properties; only the
                # target type changes at the Triton boundary.
                device = torch.device("cuda", device.index or 0)
                result = original(device)
                return result._replace(type=device_type)

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
                warp_size=getattr(props, "warp_size", 32),
            )
        return original(device)

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
    In a boxing build the target is right for the direct reason that flagos and
    cuda are the same physical GPU. On native MUSA there is no CUDA runtime, and
    the mapping is only harmless because torch_fl's FLAGOS_ALIAS_CUDA mode
    resolves the ``cuda`` spelling back to flagos as well; MUSA does not reach
    this path in practice, since coordinate-descent tuning is disabled there.

    No-op when the Triton backend name is already a valid torch device (the CUDA
    build reports ``cuda``), so this costs nothing off MetaX.
    """
    device_type, _ = _triton_backend()
    if device_type == "cuda":
        return

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
            device = "cuda"
        return original(self, fn, fn_args, fn_kwargs, device=device, **kwargs)

    benchmark._flagos_patched = True  # type: ignore[attr-defined]
    benchmarker_cls.benchmark = benchmark  # type: ignore[method-assign]


def _repair_cuda_interface_raw_stream() -> None:
    """Give the stock CudaInterface its raw-stream getter back.

    Because DeviceProperties reports flagos as cuda (see _patch_device_properties),
    inductor's *runtime* launcher resolves the stock `CudaInterface`, and its
    autotuner calls `get_raw_stream(current_device())`. That attribute is bound at
    import time from `torch._C._cuda_getCurrentRawStream`, but only when
    `torch.cuda._is_compiled()` -- which is False for the CPU torch wheel, so it
    lands on None and the autotuner raises "'NoneType' object is not callable".

    The binding itself is present: torch_fl's torch.cuda shim installs it
    (accelerator/cuda/_cuda_compat.py). So this just re-attaches what the import
    time probe missed.
    """
    import torch._dynamo.device_interface as di

    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if getter is None or di.CudaInterface.get_raw_stream is not None:
        return

    di.get_cuda_stream = getter
    di.CudaInterface.get_raw_stream = staticmethod(getter)  # type: ignore[assignment]


def register_flagos_device_interface() -> None:
    """Register the flagos device interface + GPU type with inductor.

    Idempotent; called by the compile backend before each compile_fx.
    """
    from torch._dynamo.device_interface import register_interface_for_device

    _register_gpu_type()
    _patch_device_properties()
    _patch_inductor_benchmark_device()
    _repair_cuda_interface_raw_stream()
    register_interface_for_device(DEVICE_TYPE, FlagOSDeviceInterface)
    # The triton-reported device type ("maca" on MetaX) is what the runtime
    # launcher resolves via `triton_heuristics.get_device_interface`, so that
    # name needs an interface too. It proxies the same hardware as flagos.
    device_type, _ = _triton_backend()
    if device_type != "cuda":  # "cuda" is already registered by inductor
        register_interface_for_device(device_type, FlagOSDeviceInterface)

    # torch.utils._triton caches this result and may have been queried during
    # torch import, before torch_fl registered PrivateUse1. Invalidate it after
    # publishing the native MUSA interface so Inductor accepts the vendor Triton
    # package for the custom GPU device.
    try:
        from torch.utils import _triton as triton_utils

        triton_utils.has_triton.cache_clear()
        from torch_fl._build_config import ACCELERATOR

        if ACCELERATOR == "musa" and triton_utils.has_triton_package():
            # PyTorch's helper only knows CUDA/XPU/PrivateUse1 at import time;
            # the CPU wheel's CUDA probe is false even though MUSA owns
            # PrivateUse1. Inductor imports this helper into several modules,
            # so update those bound references as well as the canonical helper.
            def has_musa_triton() -> bool:
                return True

            triton_utils.has_triton = has_musa_triton
            import sys

            for module_name in (
                "torch._inductor.scheduler",
                "torch._inductor.compile_fx",
                "torch._inductor.async_compile",
            ):
                module = sys.modules.get(module_name)
                if module is not None and hasattr(module, "has_triton"):
                    module.has_triton = has_musa_triton
    except (ImportError, AttributeError):
        pass
