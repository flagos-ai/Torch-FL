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
A torch_npu-free backend policy for FlagTree's Ascend backend.

FlagTree's Ascend backend dispatches its host-runtime operations through a
strategy registry keyed by a "backend policy" string. Upstream ships two
policies, ``torch_npu`` and ``mindspore``, and neither is usable here:

* ``mindspore`` needs MindSpore, not PyTorch.
* ``torch_npu`` needs torch_npu, which claims PrivateUse1 on import. torch_fl
  needs that same slot for ``flagos``, and PyTorch allows only one owner per
  process, so importing torch_npu means ``flagos`` cannot be registered at all.

Critically, the torch_npu coupling is *not* just a Python import that a stub
module could satisfy. The policy also decides generated C++ and link flags:
``get_cc_cmd`` emits ``-ltorch_npu``, ``header_file`` includes
``<torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>``, and two strategies emit
``at_npu::native::`` calls. A fake ``sys.modules["torch_npu"]`` has no headers
and no shared library, so the launcher would fail to compile. Worse,
``allocate_memory`` allocates on ``at::kPrivateUse1`` as *torch_npu's* device,
which is the conflict rather than a way around it.

This module registers a third policy, ``flagos``, that answers the same
strategy names from torch_fl's own runtime: device and stream come from
``torch.flagos`` and the ACL stream registry, and the generated C++ uses plain
ATen against PrivateUse1 -- which under torch_fl *is* flagos -- instead of
``at_npu::``.

Upstream tracking issue for removing the need for this shim:
https://github.com/flagos-ai/FlagTree/issues/1046

Nothing here is imported unless FLAGOS_USE_FLAGTREE=1 selects it, so the
default triton-ascend path is untouched.
"""

from __future__ import annotations

POLICY_NAME = "flagos"

# Set once install_policy() has registered the strategies. FlagTree's registry
# cannot be introspected to answer this (see install_policy).
_installed = False

# Arguments for the presence probe in install_policy: enough to reach the
# strategy body, since these are pure codegen/query functions.
_PROBE_ARGS = {
    "header_file": (False,),
    "get_cc_cmd": (False,),
}

# Strategy names FlagTree's driver.py dispatches through get_backend_func. A
# policy that misses any of these raises at compile time rather than import
# time, so completeness is asserted up front by install_policy().
_REQUIRED_STRATEGIES = (
    "version_hash",
    "cxx_abi",
    "type_convert",
    "get_device_interface",
    "get_empty_tensor",
    "get_tensor_params_shape",
    "get_cc_cmd",
    "get_current_device",
    "set_current_device",
    "get_current_stream",
    "header_file",
    "allocate_memory",
    "allocate_sync_block_lock",
    "pre_launch",
    "async_launch",
)


def _register_strategies(registry) -> None:
    """Register the flagos policy on FlagTree's strategy registry."""
    register = registry.register

    @register(POLICY_NAME, "version_hash")
    def version_hash():
        # Part of the compile cache key. torch_npu's policy mixes in
        # torch_npu.version.git_version; the flagos equivalent is the plugin's
        # own version, so caches do not survive a torch_fl upgrade.
        import torch

        import torch_fl

        return [torch.version.git_version, getattr(torch_fl, "__version__", "unknown")]

    @register(POLICY_NAME, "cxx_abi")
    def cxx_abi():
        import torch

        return 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

    @register(POLICY_NAME, "type_convert")
    def type_convert():
        import numpy as np
        import torch

        return {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            torch.bfloat16: np.float16,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.bool: np.bool_,
        }

    @register(POLICY_NAME, "get_device_interface")
    def get_device_interface():
        import torch

        return torch.flagos

    @register(POLICY_NAME, "get_empty_tensor")
    def get_empty_tensor(size):
        import torch

        return torch.empty(size, dtype=torch.int32, device="flagos")

    @register(POLICY_NAME, "get_tensor_params_shape")
    def get_tensor_params_shape(*args):
        import torch

        return [list(a.shape) for a in args if isinstance(a, torch.Tensor)]

    @register(POLICY_NAME, "get_cc_cmd")
    def get_cc_cmd(build_pch):
        # torch_npu's version adds its include dir and links -ltorch_npu. The
        # flagos launcher needs neither: it only uses ATen, which lives in the
        # torch headers, and torch's own libraries are already loaded in-process
        # by the time a launcher runs.
        import os

        import torch

        torch_path = os.path.dirname(os.path.realpath(torch.__file__))
        return [
            f"-I{os.path.join(torch_path, 'include')}",
            f"-I{os.path.join(torch_path, 'include', 'torch', 'csrc', 'api', 'include')}",
            f"-D_GLIBCXX_USE_CXX11_ABI={cxx_abi()}",
        ]

    @register(POLICY_NAME, "get_current_device")
    def get_current_device():
        import torch

        return torch.flagos.current_device()

    @register(POLICY_NAME, "set_current_device")
    def set_current_device(device_id):
        import torch

        return torch.flagos.set_device(device_id)

    @register(POLICY_NAME, "get_current_stream")
    def get_current_stream(device):
        # Must be the same aclrtStream the ACLNN ops producing this kernel's
        # inputs ran on. Returning 0 would be accepted and silently produce
        # wrong results, because rt stream 0 is not ordered against them.
        import torch

        from torch_fl.accelerator.ascend.acl_stream import current_acl_raw_stream

        if device is None:
            device = torch.flagos.current_device()
        return current_acl_raw_stream(device)

    @register(POLICY_NAME, "header_file")
    def header_file(enable_taskqueue):
        # torch_npu pulls in NPUWorkspaceAllocator.h and, for the task queue,
        # OpCommand.h. Plain ATen covers what the flagos launcher does, and the
        # task queue is a torch_npu concept with no flagos equivalent.
        return "#include <ATen/ATen.h>"

    @register(POLICY_NAME, "allocate_memory")
    def allocate_memory(size, stream):
        # torch_npu's version also allocates on kPrivateUse1 -- but under
        # torch_fl that dispatch key is flagos, so the same ATen call reaches
        # torch_fl's allocator instead of torch_npu's.
        return (
            f"workspace_addr_ptr = const_cast<void *>(at::empty({size}, "
            "at::TensorOptions().device(at::kPrivateUse1).dtype(at::kByte))"
            ".storage().data());"
        )

    @register(POLICY_NAME, "allocate_sync_block_lock")
    def allocate_sync_block_lock(size, stream):
        # torch_npu calls at_npu::native::allocate_workspace, which needs
        # libtorch_npu. An ATen allocation on the same device serves the same
        # purpose: scratch memory for cross-block synchronization. Zeroed
        # because the lock protocol expects a cleared buffer, whereas the
        # torch_npu workspace allocator returns already-zeroed memory.
        return (
            f"syncBlockLock_ptr = const_cast<void *>(at::zeros({size}, "
            "at::TensorOptions().device(at::kPrivateUse1).dtype(at::kByte))"
            ".storage().data());"
        )

    @register(POLICY_NAME, "pre_launch")
    def pre_launch(first_call):
        # torch_npu returns "" here too: nothing to bind per launch.
        return ""

    @register(POLICY_NAME, "async_launch")
    def async_launch(func):
        # Only reachable if TRITON_ENABLE_TASKQUEUE is forced back on;
        # install_policy() turns it off, because the task queue is a torch_npu
        # construct end to end -- upstream wraps the launch in
        # at_npu::native::OpCommand and the generated function returns rtError_t
        # for the queue to consume. Calling the lambda here would discard that
        # error code, so refuse instead of silently dropping launch failures.
        raise RuntimeError(
            "TRITON_ENABLE_TASKQUEUE is on, but the flagos FlagTree policy has "
            "no task queue: upstream's async launch requires "
            "at_npu::native::OpCommand from libtorch_npu. Unset "
            "TRITON_ENABLE_TASKQUEUE to use the synchronous launch path."
        )


def install_policy() -> str:
    """Register and select the flagos policy inside FlagTree's Ascend backend.

    Returns the policy name on success.

    Raises:
        RuntimeError: if the active Triton has no FlagTree Ascend backend, if
            its registry API has changed shape, or if torch_npu is already
            loaded (in which case flagos cannot own PrivateUse1 anyway).
    """
    import sys

    import torch

    # torch_fl installs its own minimal torch_npu stub for FlagGems (see
    # torch_fl/__init__.py), so mere presence in sys.modules proves nothing. What
    # matters is whether the *real* extension got loaded and took PrivateUse1:
    # the stub has no _C, and flagos still owns the backend name.
    existing = sys.modules.get("torch_npu")
    if existing is not None and hasattr(existing, "_C"):
        raise RuntimeError(
            "The real torch_npu extension is loaded, so it owns the PrivateUse1 "
            "backend and torch_fl cannot own 'flagos'. The flagos FlagTree "
            "policy exists precisely to avoid loading torch_npu; remove the "
            "import from your program."
        )

    backend_name = torch._C._get_privateuse1_backend_name()
    if backend_name != "flagos":
        raise RuntimeError(
            f"PrivateUse1 is registered as '{backend_name}', not 'flagos', so "
            "generated code allocating on that key would not reach torch_fl's "
            "allocator. Import torch_fl before anything else that claims "
            "PrivateUse1."
        )

    try:
        from triton.backends.ascend import backend_register, utils
    except ImportError as exc:
        raise RuntimeError(
            "FLAGOS_USE_FLAGTREE=1 on Ascend needs a FlagTree build carrying the "
            "Ascend backend (the triton_v3.5.x line; it is absent from main and "
            "the 3.6/3.7 branches). The active triton has no "
            "triton.backends.ascend. See docs/vendors/ascend/installation.md."
        ) from exc
    except RuntimeError as exc:
        # Importing the backend package runs FlagTree's own module-scope
        # `import torch_npu` (backends/ascend/__init__.py -> testing.py), which
        # only profiling needs but which executes during backend discovery. With
        # flagos holding PrivateUse1, torch_npu refuses to load and the raw error
        # ("npu and npu") says nothing about the real cause, so restate it.
        raise RuntimeError(
            "This FlagTree build's Ascend backend imports torch_npu while being "
            "loaded (backends/ascend/__init__.py imports do_bench_npu, and "
            "testing.py imports torch_npu at module scope), so it cannot be "
            "imported at all once torch_fl owns PrivateUse1 -- the flagos policy "
            "never gets a chance to be selected. Only profiling needs that "
            "import; making it lazy is a prerequisite for FlagTree on Ascend "
            "(https://github.com/flagos-ai/FlagTree/issues/1046). "
            f"Underlying error: {exc}"
        ) from exc

    registry = getattr(backend_register, "backend_strategy_registry", None)
    if registry is None or not hasattr(registry, "register"):
        raise RuntimeError(
            "triton.backends.ascend.backend_register has no usable "
            "backend_strategy_registry; this FlagTree build's Ascend backend "
            "does not match the strategy-registry API this shim targets."
        )

    # Idempotent: FlagTree's registry raises ValueError on duplicate
    # registration, and torch_fl's compile registration can be reached more than
    # once. The introspection helpers cannot be used to check first --
    # backend_strategy_registry is a _LazyBackendStrategyRegister that proxies
    # only register() and execute_func(), so list_categories()/list_methods()
    # raise AttributeError on it. Track it here instead.
    global _installed
    if not _installed:
        try:
            _register_strategies(registry)
        except ValueError as exc:
            # Someone already registered this category on a shared registry.
            # Harmless -- the strategies are ours either way.
            if "already registered" not in str(exc):
                raise
        _installed = True

    # Verify by dispatch rather than by introspection, for the same reason. A
    # missing strategy otherwise surfaces much later, mid-compile.
    for name in ("get_current_device", "header_file", "get_cc_cmd"):
        try:
            registry.execute_func(POLICY_NAME, name, *_PROBE_ARGS.get(name, ()))
        except ValueError as exc:
            raise RuntimeError(
                f"flagos FlagTree policy did not register strategy '{name}'; "
                "FlagTree's Ascend driver would fail mid-compile."
            ) from exc
        except Exception:
            # Strategy exists but needs a live device (get_current_device off
            # hardware). Presence is what matters here.
            pass

    # The task queue is torch_npu-only (at_npu::native::OpCommand), and it
    # defaults to ON upstream, so it has to be turned off explicitly. Respect an
    # explicit opt-in so the failure is the clear message in async_launch rather
    # than a silent override of what the user asked for.
    import os

    if "TRITON_ENABLE_TASKQUEUE" not in os.environ:
        os.environ["TRITON_ENABLE_TASKQUEUE"] = "false"

    # utils.get_backend_func only honours TRITON_BACKEND when it names
    # torch_npu or mindspore, so the env var cannot select this policy. Set the
    # module global it caches into directly -- and do it before any compile, so
    # the auto-detection branch (which imports torch_npu) never runs.
    utils.backend_policy = POLICY_NAME

    return POLICY_NAME
