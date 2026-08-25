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
   device guards, stream lookup and synchronization. flagos reuses the CUDA
   ones almost verbatim: a flagos tensor's storage *is* CUDA memory (the flagos
   allocator delegates to c10::cuda::CUDACachingAllocator), and torch_fl ships a
   torch.cuda shim over the same physical GPU. Only the Python-level device
   guard and set_device switch to `torch.flagos`, so generated code moves the
   flagos current device rather than desyncing the two.

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
    get_scheduling_for_device,
    register_backend_for_device,
    register_device_op_overrides,
)
from torch._inductor.codegen.cuda.device_op_overrides import CUDADeviceOpOverrides


DEVICE_TYPE = "flagos"


class FlagOSDeviceOpOverrides(CUDADeviceOpOverrides):
    """Code snippets inductor emits into generated flagos kernels.

    Inherits the CUDA implementation for everything C++/driver related (kernel
    headers, stream types, TMA helpers, AOTI guards) because those operate on
    the underlying CUDA runtime, which is exactly what flagos runs on. Only the
    Python-level device manipulation is overridden, to go through torch.flagos
    so generated code doesn't desync the flagos and CUDA current device.
    """

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
        from torch_fl._build_config import ACCELERATOR

        if ACCELERATOR == "musa":
            # MThreads FlagTree's launcher consumes a raw musaStream_t.  Do not
            # import the CUDA binding on native MUSA, where CPU PyTorch has no
            # CUDA runtime symbols.
            return (
                "from torch_fl.compile.flagtree_shim import "
                f"get_musa_current_raw_stream as {name}"
            )
        if ACCELERATOR == "gcu":
            # Same on GCU: triton_gcu's launcher takes a topsStream_t and there
            # is no CUDA runtime behind torch.flagos.
            return (
                "from torch_fl.compile.flagtree_shim import "
                f"get_gcu_current_raw_stream as {name}"
            )
        # Boxing-backed flagos streams are CUDA streams on the same device.
        return f"from torch._C import _cuda_getCurrentRawStream as {name}"


def register_flagos_codegen() -> None:
    """Register flagos scheduling/wrapper codegen + device op overrides.

    Idempotent: safe to call before every compile_fx.
    """
    from torch._inductor.codegen.common import (
        device_op_overrides_dict,
        init_backend_registration,
    )

    # Populate the built-in registrations first, so our checks below see the
    # real state and so the CUDA overrides module is imported.
    init_backend_registration()

    if DEVICE_TYPE not in device_op_overrides_dict:
        register_device_op_overrides(DEVICE_TYPE, FlagOSDeviceOpOverrides())

    if get_scheduling_for_device(DEVICE_TYPE) is None:
        from torch._inductor import config
        from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
        from torch._inductor.codegen.cuda_combined_scheduling import (
            CUDACombinedScheduling,
        )
        from torch._inductor.codegen.halide import HalideScheduling
        from torch._inductor.codegen.simd import SIMDScheduling
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen
        from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen

        # Same table CUDA uses; flagos wants the Triton pipeline.
        backends = {
            "triton": CUDACombinedScheduling,
            "halide": HalideScheduling,
        }
        register_backend_for_device(
            DEVICE_TYPE,
            lambda scheduling: backends.get(config.cuda_backend, SIMDScheduling)(
                scheduling
            ),
            PythonWrapperCodegen,
            CppWrapperGpu,
            WrapperFxCodegen,
        )


def publish_codegen_on_device_module() -> None:
    """Expose the four codegen classes on `torch.flagos`.

    This is inductor's sanctioned PrivateUse1 hook: `init_backend_registration`
    reads `Scheduling` / `PythonWrapperCodegen` / `CppWrapperCodegen` /
    `WrapperFxCodegen` off the device module. Setting them means a fresh
    inductor process registers flagos on its own, without our backend having to
    run first.
    """
    import torch

    from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
    from torch._inductor.codegen.cuda_combined_scheduling import (
        CUDACombinedScheduling,
    )
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen

    mod = getattr(torch, DEVICE_TYPE, None)
    if mod is None:
        return
    for name, cls in (
        ("Scheduling", CUDACombinedScheduling),
        ("PythonWrapperCodegen", PythonWrapperCodegen),
        ("CppWrapperCodegen", CppWrapperGpu),
        ("WrapperFxCodegen", WrapperFxCodegen),
    ):
        if not hasattr(mod, name):
            setattr(mod, name, cls)
