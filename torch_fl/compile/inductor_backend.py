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
Inductor-based compile backend for the flagos device.

flagos is registered with inductor as a first-class GPU device (see
device_interface.py and inductor_codegen.py), so the traced graph is handed to
`compile_fx` *as is* -- still on flagos. Inductor generates Triton kernels for
it directly.

Why not rewrite the graph to cuda (as an earlier version did): `at::getAccelerator()`
is PrivateUse1/flagos here, and `torch::autograd::Node::stream()` only yields a
stream when a node's input device type matches the accelerator. A cuda-rewritten
graph therefore produces stream-less autograd nodes, and AOT autograd's backward
trace inside compile_fx trips `opt_ready_stream && opt_parent_stream`
(engine.cpp:1085). Staying on flagos also removes a copy-in/copy-out per call.
"""

import os
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.fx
import torch.cuda


def _patch_musa_triton_autotune() -> None:
    """Select the first compiled MUSA config without CUDA benchmarking.

    Inductor's generic runtime autotuner allocates a CUDA L2-cache buffer and
    records ``torch.cuda.Event`` pairs. Neither API exists on native MUSA, and
    the MThreads driver already compiles a valid launch configuration for each
    kernel. Selecting the first compiled configuration keeps execution native;
    it does not claim autotuning performance parity.
    """
    from torch_fl._build_config import ACCELERATOR

    if ACCELERATOR != "musa":
        return
    try:
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    except ImportError:
        return
    if getattr(CachingAutotuner, "_flagos_musa_patched", False):
        return

    def autotune_to_one_config(self, *args, **kwargs):
        del args, kwargs
        if not self.launchers:
            self.precompile()
        if not self.launchers:
            raise RuntimeError("MThreads FlagTree produced no valid MUSA launchers")
        self.launchers = [self.launchers[0]]

    CachingAutotuner.autotune_to_one_config = autotune_to_one_config
    CachingAutotuner._flagos_musa_patched = True


def _patch_native_musa_cuda_probe() -> None:
    """Make Inductor's CUDA-shaped FakeTensor probe see native MUSA devices.

    CPU-only PyTorch reports ``torch.cuda.is_available() == False`` even when
    the PrivateUse1 MUSA runtime is active. Inductor uses that probe to decide
    whether FakeTensor should initialize a GPU context; without this bridge it
    skips the registered flagos device and later attempts CPU-torch CUDA lazy
    initialization. The compiler still receives a ``flagos`` device and never
    executes a CUDA kernel.
    """
    from torch_fl._build_config import ACCELERATOR

    if ACCELERATOR != "musa" or getattr(torch.cuda, "_flagos_musa_patched", False):
        return
    if not torch.flagos.is_available():
        return

    original = torch.cuda.is_available
    torch.cuda.is_available = torch.flagos.is_available
    torch.cuda._flagos_original_is_available = original
    torch.cuda._flagos_musa_patched = True


def _bind_musa_flagtree_runtime() -> None:
    """Bind MThreads FlagTree to torch_fl's MUSA runtime, not to torch_musa.

    The vendor driver reads its device/stream/capability from the separate
    ``torch_musa`` plugin, whose ``__init__`` claims the process-global
    PrivateUse1 hooks torch_fl must own. ``flagtree_shim`` rebinds those lookups
    onto torch_fl, so the compiler and the native mudnn kernels share one device
    and one stream. This must run before any Triton driver instance exists, hence
    module load time.
    """
    from torch_fl._build_config import ACCELERATOR

    if ACCELERATOR != "musa":
        return
    try:
        from torch_fl.compile.flagtree_shim import bind_flagtree_musa_driver

        bind_flagtree_musa_driver()
    except ImportError:
        pass


def _patch_cuda_rng_for_cpu_torch():
    """
    Workaround for CPU torch + external libtorch_cuda.so setup.

    dynamo tries to capture torch.cuda.get_rng_state() during tracing, but
    CPU torch doesn't have torch._C._cuda_getDevice() binding. We patch
    torch.cuda to provide stub implementations that prevent the crash.

    Only applied when torch._C lacks CUDA bindings (CPU torch build).
    """
    import torch as torch_module

    if hasattr(torch_module._C, "_cuda_getDevice"):
        return  # Native CUDA torch, no patch needed

    # CPU torch detected - patch CUDA RNG functions
    import torch.cuda as cuda_module

    # Stub implementations that won't be called (dynamo just needs them callable)
    def _stub_get_rng_state(device=None):
        # Return empty tensor as placeholder (dynamo won't execute this)
        return torch_module.tensor([], dtype=torch_module.uint8)

    def _stub_set_rng_state(new_state, device=None):
        pass  # No-op

    cuda_module.get_rng_state = _stub_get_rng_state
    cuda_module.set_rng_state = _stub_set_rng_state


# Apply runtime probes at module load time, before FakeTensor is imported by
# the first compile. Native MUSA is not CUDA, but Inductor's GPU probe is the
# shared gate for CUDA-shaped GPU devices.
_patch_native_musa_cuda_probe()
_patch_musa_triton_autotune()
_bind_musa_flagtree_runtime()
_patch_cuda_rng_for_cpu_torch()


def _resolve_config_patches(
    mode: Optional[str],
    options: Optional[Dict[str, Any]],
    dynamic: Optional[bool],
) -> Dict[str, Any]:
    """Turn torch.compile's mode/options into an inductor config patch dict.

    Same expansion `_TorchCompileInductorWrapper` does, plus the flagos-specific
    overrides this build needs. Passing these to compile_fx as `config_patches`
    scopes them to this compile, instead of mutating inductor's global config.
    """
    patches: Dict[str, Any] = {}

    if mode and mode != "default":
        from torch._inductor import list_mode_options

        patches.update(list_mode_options(mode, dynamic))
    if options:
        patches.update({k.replace("-", "_"): v for k, v in options.items()})

    from torch_fl._build_config import ACCELERATOR

    # Native MUSA has no CUDA implementation for Inductor's CUDA-only pattern
    # registration (for example SDPA replacement construction). Keep the
    # general joint-graph passes, but skip that optional pattern bundle until
    # FlagTree/MUSA supplies an equivalent lowering.
    if ACCELERATOR == "musa":
        patches["use_joint_graph_passes"] = False
        patches["max_autotune"] = False
        patches["max_autotune_pointwise"] = False
        patches["max_autotune_gemm"] = False
        patches["max_autotune_gemm_backends"] = "TRITON"
        # Coordinate-descent tuning times candidate configs through Inductor's
        # CUDA benchmarker (torch.cuda.synchronize plus torch.cuda.Event), which
        # the CPU torch wheel cannot provide. mode="max-autotune" enables it, so
        # turn it back off; the kernels still compile and run, just untuned.
        patches["coordinate_descent_tuning"] = False

    # CUDA graphs need torch.cuda.CUDAGraph, a dummy base class in the CPU torch
    # wheel ("Tried to instantiate dummy base class CUDAGraph"). mode=
    # "max-autotune" turns them on, so force them back off.
    patches["triton.cudagraphs"] = False

    # The static launcher needs torch._C._StaticCudaLauncher, which the CPU
    # torch wheel does not build (we supply libtorch_cuda.so externally). Fall
    # back to the regular Triton launch path.
    if not hasattr(torch._C, "_StaticCudaLauncher"):
        patches["use_static_cuda_launcher"] = False

    return patches


def _patch_vendor_flagtree_compile_workers(config_patches: Dict[str, Any]) -> None:
    """Avoid unsafe vendor-driver initialization in compile workers.

    PPU and MThreads FlagTree drivers query a vendor runtime while creating
    compiler hints. Keeping the first implementation conservative is important
    for MUSA: the vendor runtime owns the current device and stream, while
    Inductor workers may be forked after the parent has initialized it. Explicit
    per-compile and environment settings remain authoritative.
    """
    if "compile_threads" in config_patches:
        return
    if os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") is not None:
        return

    from torch_fl._build_config import ACCELERATOR
    from torch_fl.compile.flagtree_shim import flagtree_backend

    backend = flagtree_backend()
    if backend == "ppu" or (ACCELERATOR == "musa" and backend == "mthreads"):
        config_patches["compile_threads"] = 1


# Keep the old helper name for downstream callers and the PPU regression tests.
def _patch_ppu_flagtree_compile_workers(config_patches: Dict[str, Any]) -> None:
    _patch_vendor_flagtree_compile_workers(config_patches)


def flagos_compile_backend(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    mode: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    dynamic: Optional[bool] = None,
) -> Callable:
    """
    torch.compile backend for flagos device.

    Registers flagos as an inductor GPU device, then delegates the graph to
    inductor unchanged. Generated Triton kernels run on flagos tensors directly.

    `mode` / `options` arrive as kwargs from dynamo's `_TorchCompileWrapper`
    (torch/__init__.py) whenever torch.compile is called with them on a named
    backend; they are expanded into inductor config patches.

    Usage:
        model = torch.compile(model, backend="flagos")
        # or
        model = torch.compile(model, backend="flagos", mode="max-autotune")

    Environment:
        FLAGOS_USE_FLAGTREE=1 : Require that the active triton be FlagTree
        FLAGOS_COMPILE_FALLBACK_EAGER=1 : Fall back to eager on compile errors
    """
    # Import inductor lazily (not all torch builds have it)
    try:
        from torch._inductor.compile_fx import compile_fx
    except ImportError as e:
        if os.environ.get("FLAGOS_COMPILE_FALLBACK_EAGER", "0") == "1":
            return gm.forward
        raise RuntimeError(
            "torch._inductor not available. Install torch with inductor support "
            "or set FLAGOS_COMPILE_FALLBACK_EAGER=1 to fall back to eager."
        ) from e

    config_patches = _resolve_config_patches(mode, options, dynamic)

    # FlagTree substitutes itself for triton at install time, so if it is
    # installed inductor already compiles with it and there is nothing to switch
    # on here. This only asserts that, so the flag cannot silently no-op.
    if os.environ.get("FLAGOS_USE_FLAGTREE", "0") == "1":
        from torch_fl.compile.flagtree_shim import require_flagtree

        require_flagtree()

    # Idempotent: module load already tried this, but a process that imported
    # the backend before FlagTree was importable gets its second chance here.
    _bind_musa_flagtree_runtime()
    _patch_vendor_flagtree_compile_workers(config_patches)

    # Make inductor treat flagos as a GPU device. Order matters: is_gpu() must
    # answer True and the device interface must be resolvable before the
    # codegen backend registration reads them.
    from torch_fl.compile.device_interface import register_flagos_device_interface
    from torch_fl.compile.inductor_codegen import (
        publish_codegen_on_device_module,
        register_flagos_codegen,
    )

    register_flagos_device_interface()
    publish_codegen_on_device_module()
    register_flagos_codegen()

    # Hand the graph to inductor untouched -- it is on flagos and stays there.
    try:
        return compile_fx(gm, example_inputs, config_patches=config_patches)
    except Exception as e:
        if os.environ.get("FLAGOS_COMPILE_FALLBACK_EAGER", "0") == "1":
            import warnings

            warnings.warn(f"Inductor compilation failed: {e}. Falling back to eager.")
            return gm.forward
        raise


def register_backend():
    """
    Register the flagos backend with torch._dynamo.

    Called automatically on import torch_fl if torch 2.0+ detected.
    """
    try:
        import torch._dynamo

        torch._dynamo.register_backend(
            name="flagos", compiler_fn=flagos_compile_backend
        )
    except (ImportError, AttributeError):
        # torch._dynamo not available (torch < 2.0)
        pass
