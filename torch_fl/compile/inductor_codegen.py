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
Inductor codegen backend registration for the flagos device.

Two things get registered here:

1. `DeviceOpOverrides` -- the snippets inductor splices into generated code for
   device guards, stream lookup and synchronization. On CUDA-like builds flagos
   reuses the CUDA ones almost verbatim: a flagos tensor's storage *is* CUDA
   memory (the flagos allocator delegates to c10::cuda::CUDACachingAllocator),
   and torch_fl ships a torch.cuda shim over the same physical GPU. Only the
   Python-level device guard and set_device switch to `torch.flagos`, so
   generated code moves the flagos current device rather than desyncing the two.

   All builds query the current stream through the flagos device runtime.
   Ascend uses Python wrapper snippets because the CUDA C++ and AOTI wrappers
   depend on CUDA types.

2. The scheduling + wrapper codegen classes. These are the stock CUDA ones --
   flagos wants exactly the Triton/CUDA pipeline -- registered under the
   "flagos" device key.

Registration goes through inductor's official PrivateUse1 path where possible:
`init_backend_registration()` looks up `Scheduling`, `PythonWrapperCodegen`,
`CppWrapperCodegen` and `WrapperFxCodegen` on the device module named by
`torch._C._get_privateuse1_backend_name()` (i.e. `torch.flagos`). We publish
those four names *and* call `register_backend_for_device` directly, since
`init_backend_registration` may already have run before torch_fl was imported.
"""

from torch._inductor.codegen.common import (
    DeviceOpOverrides,
    get_scheduling_for_device,
    register_backend_for_device,
    register_device_op_overrides,
)
from torch._inductor.codegen.cuda.device_op_overrides import CUDADeviceOpOverrides

from torch_fl.compile.platform_profile import platform_profile


DEVICE_TYPE = "flagos"


class _FlagOSDeviceQueries:
    """Python snippets backed by the flagos device runtime."""

    def set_device(self, device_idx: int) -> str:
        return f"torch.flagos.set_device({device_idx})"

    def device_guard(self, device_idx: int) -> str:
        # torch.flagos has no _DeviceGuard; torch.flagos.device(idx) is the
        # equivalent context manager and keeps flagos/cuda current device in
        # sync (flagos.set_device moves the CUDA device too).
        return f"torch.flagos.device({device_idx})"

    def synchronize(self) -> str:
        return "torch.flagos.synchronize()"

    def import_get_raw_stream_as(self, name: str) -> str:
        return f"from torch_fl.flagos._streams import get_raw_stream as {name}"


class FlagOSDeviceOpOverrides(_FlagOSDeviceQueries, CUDADeviceOpOverrides):
    """Use the CUDA driver and C++ wrapper snippets for CUDA-shaped runtimes."""


class FlagOSPythonDeviceOpOverrides(_FlagOSDeviceQueries, DeviceOpOverrides):
    """Device snippets for backends with a Python wrapper."""


def _device_op_overrides() -> DeviceOpOverrides:
    """Select the C++ wrapper capabilities for the build."""
    profile = platform_profile()
    if profile.vendor == "ascend" or profile.compiler == "metal":
        return FlagOSPythonDeviceOpOverrides()
    return FlagOSDeviceOpOverrides()


def _init_builtin_device_op_overrides() -> None:
    """Register torch's built-in device-op overrides before flagos.

    ``get_device_op_overrides`` (``torch/_inductor/codegen/common.py``) imports
    the built-in override modules inside an ``if not device_op_overrides_dict``
    guard, and those imports are what populate the ``cpu``, ``cuda``, ``mps``,
    ``mtia`` and ``xpu`` entries. Registering flagos first makes the dictionary
    non-empty, so the guard never runs again and every later lookup of a device
    this process has not touched -- ``cpu`` above all -- raises
    ``KeyError: 'cpu'``. ``init_backend_registration()`` does not help: it
    fills the *codegen* table, not this one.

    Importing the modules here is what matters; the guard in
    ``get_device_op_overrides`` cannot be made to run once the dictionary is
    non-empty, which is why this imports the registration modules directly
    instead of calling that function as a side effect.

    ``cpu`` is imported unguarded because it is the device the issue is about
    and the one every inductor build has.  The others are guarded: this has to
    keep working on an older or vendor-patched torch whose codegen tree has a
    different set of accelerator modules, and losing one of them would only
    matter to a process that then compiles for that device.
    """
    import importlib

    from torch._inductor.codegen import (  # noqa: F401
        cpu_device_op_overrides,
        mps_device_op_overrides,
    )

    for module in (
        "torch._inductor.codegen.cuda.device_op_overrides",
        "torch._inductor.codegen.mtia.device_op_overrides",
        "torch._inductor.codegen.xpu.device_op_overrides",
    ):
        try:
            importlib.import_module(module)
        except ImportError:  # pragma: no cover - torch without that device
            continue


def register_flagos_codegen() -> None:
    """Register flagos scheduling/wrapper codegen + device op overrides.

    Idempotent: safe to call before every compile_fx.
    """
    from torch._inductor.codegen.common import (
        device_op_overrides_dict,
        init_backend_registration,
    )

    from torch_fl.native_host import _prepared

    if _prepared:
        from torch_fl.native_host.load import bind_inductor_launcher

        bind_inductor_launcher()

    # Populate the built-in registrations first, so our checks below see the
    # real state and so the CUDA overrides module is imported.
    init_backend_registration()
    _init_builtin_device_op_overrides()

    if DEVICE_TYPE not in device_op_overrides_dict:
        register_device_op_overrides(DEVICE_TYPE, _device_op_overrides())

    if get_scheduling_for_device(DEVICE_TYPE) is None:
        scheduling, wrapper, cpp_wrapper, fx_wrapper = _codegen_classes()
        register_backend_for_device(
            DEVICE_TYPE, scheduling, wrapper, cpp_wrapper, fx_wrapper
        )
    if platform_profile().compiler == "metal":
        from torch_fl.compile.metal import install_scheduler

        install_scheduler()


def _codegen_classes() -> tuple:
    """Return ``(scheduling, wrapper, cpp_wrapper, fx_wrapper)`` for this build.

    CUDA-like builds take the full CUDA table, including the CUDA C++ template
    scheduling that `CUDACombinedScheduling` delegates to. Ascend takes plain
    `TritonScheduling`: the extra delegates in the combined scheduler are CUTLASS
    / ROCm / CuteDSL template paths that have no Ascend equivalent, and
    `CppWrapperGpu` emits CUDA-runtime C++, so the cpp-wrapper slot is left
    unregistered (None) rather than pointing at something that cannot build.
    """
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen

    profile = platform_profile()
    if profile.compiler == "metal":
        from torch_fl.compile.metal import MetalScheduling

        return MetalScheduling, PythonWrapperCodegen, None, WrapperFxCodegen

    if not profile.is_cuda_like:
        from torch._inductor.codegen.triton import TritonScheduling

        return TritonScheduling, PythonWrapperCodegen, None, WrapperFxCodegen

    from torch._inductor import config
    from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
    from torch._inductor.codegen.cuda_combined_scheduling import (
        CUDACombinedScheduling,
    )
    from torch._inductor.codegen.halide import HalideScheduling
    from torch._inductor.codegen.simd import SIMDScheduling

    # Same table CUDA uses; flagos wants the Triton pipeline.
    backends = {
        "triton": CUDACombinedScheduling,
        "halide": HalideScheduling,
    }
    return (
        lambda scheduling: backends.get(config.cuda_backend, SIMDScheduling)(
            scheduling
        ),
        PythonWrapperCodegen,
        CppWrapperGpu,
        WrapperFxCodegen,
    )


def publish_codegen_on_device_module() -> None:
    """Expose the codegen classes on `torch.flagos`.

    This is inductor's sanctioned PrivateUse1 hook: `init_backend_registration`
    reads `Scheduling` / `PythonWrapperCodegen` / `CppWrapperCodegen` /
    `WrapperFxCodegen` off the device module. Setting them means a fresh
    inductor process registers flagos on its own, without our backend having to
    run first.

    Note that hook requires all of Scheduling/PythonWrapperCodegen/
    CppWrapperCodegen to be present, so on Ascend -- where there is no usable C++
    wrapper -- it does nothing and `register_flagos_codegen` is the only path.
    """
    import torch

    scheduling, wrapper, cpp_wrapper, fx_wrapper = _codegen_classes()
    if cpp_wrapper is None:
        return

    mod = getattr(torch, DEVICE_TYPE, None)
    if mod is None:
        return
    for name, cls in (
        ("Scheduling", scheduling),
        ("PythonWrapperCodegen", wrapper),
        ("CppWrapperCodegen", cpp_wrapper),
        ("WrapperFxCodegen", fx_wrapper),
    ):
        if not hasattr(mod, name):
            setattr(mod, name, cls)
