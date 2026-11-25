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

import os
import sys

# Single access point for every FLAGOS_* variable (see torch_fl/_env.py). Imported
# for its side effect as well: it scans the environment once and warns about a
# FLAGOS_* name torch_fl does not recognise, which is the only way a typo like
# FLAGOS_LOG_DISPACH is ever noticed -- the misspelled variable is simply never
# read and the setting silently does nothing.
from torch_fl import _env  # noqa: F401
from torch_fl import _platform
from torch_fl.accelerator import _loader
from torch_fl.compat import parallel


_build_accelerator = _platform.build_accelerator


# The conf _select_backend_config() resolved, or "" when the user named one
# through FLAGOS_BACKEND_CONFIG (nothing to resolve) or none was found. Handed to
# the C++ routing table right after _C loads; read through backend_config_path().
_BACKEND_CONFIG_PATH = ""
_STARTUP_PROFILE = "full"


def _mps_shutdown_allocator() -> None:
    """Drain Metal and free its cached blocks before interpreter teardown."""
    try:
        if "torch_fl._C" in sys.modules:
            torch_fl._C._synchronize()
            torch_fl._C._shutdown_allocator()
    except (NameError, AttributeError):
        # Python may clear this module before atexit runs. The runtime's own
        # final-buffer hook still releases the cache.
        pass


def _select_backend_config() -> None:
    """Resolve the routing file before the first operator dispatch.

    A user-supplied path wins, then the generated build record and vendor default.
    """
    global _BACKEND_CONFIG_PATH

    # Recorded per call, not accumulated. backend_config_path() reports what this
    # resolution picked *first*, mirroring the C++ reader's setter-before-env
    # order, so a path left over from an earlier call would outrank the user's
    # FLAGOS_BACKEND_CONFIG and the early return below would leave it in place.
    # At import time the function runs once and the reset is a no-op; it is what
    # makes a direct second call -- which tests/unit/test_ascend_platform_marker.py
    # makes, against a different fake tree each time -- answer for the
    # environment in front of it rather than for the previous test's.
    _BACKEND_CONFIG_PATH = ""

    if _env.value("FLAGOS_BACKEND_CONFIG"):
        # An override that cannot be read is fatal, not a fallback. The C++
        # loader leaves its table empty when the file will not open, and an empty
        # table resolves every op to Dispatcher::GetFn's default
        # (Backend::kFlagGems) -- so a typo here silently routes the whole
        # workload down the slowest path behind a single stderr line, and it also
        # switches off the FlagGems process-level setup that reads this same file
        # (see compat.flaggems.runtime.routes_to_flaggems). Measured on A100: a nonexistent
        # path sends every dispatch to flagos_python.
        #
        # Resolved to an absolute path because the C++ side reads the variable
        # lazily, at the first op dispatch, which a caller may reach after a
        # chdir -- and a relative path is resolved against the cwd at that
        # moment, not at import.
        #
        # Recorded through _BACKEND_CONFIG_PATH (which _set_backend_config_path
        # hands to the C++ reader), not written back to os.environ: an
        # environment write makes the wheel's own selection indistinguishable
        # from the user's for the rest of the process -- see backend_config_path().
        override = _env.value("FLAGOS_BACKEND_CONFIG")
        resolved = os.path.abspath(override)
        try:
            with open(resolved):
                pass
        except OSError as err:
            raise RuntimeError(
                f"FLAGOS_BACKEND_CONFIG={override!r} cannot be read "
                f"(resolved to {resolved!r}): {err}. Unset it to let torch_fl "
                f"select the conf matching this build, or point it at an "
                f"existing conf file."
            ) from err
        _BACKEND_CONFIG_PATH = resolved
        return

    build = _platform.build_info()
    vendor = build.vendor if build.platform else "nvidia"
    package = os.path.dirname(__file__)
    conf_dir = os.path.join(package, "configs")
    conf_path = (
        os.path.join(package, build.backend_config)
        if build.backend_config
        else os.path.join(conf_dir, f"backends_{vendor}.conf")
    )
    if os.path.exists(conf_path):
        _BACKEND_CONFIG_PATH = conf_path
        return
    if build.backend_config:
        raise RuntimeError(
            f"native-host wheel is missing its generated config: {conf_path}"
        )

    # Older packages shipped the conf under the platform ID.
    if build.platform and build.platform != vendor:
        alias_path = os.path.join(conf_dir, f"backends_{build.platform}.conf")
        if os.path.exists(alias_path):
            _BACKEND_CONFIG_PATH = alias_path


def backend_config_path() -> str:
    """Path of the conf the op-routing table is read from ("" if none).

    Answers in the same order the C++ reader resolves it
    (csrc/aten/common.cc:ResolveBackendConfigPath): what _select_backend_config()
    picked and handed over, then the user's FLAGOS_BACKEND_CONFIG, then nothing --
    the last case being a path C++ derives from its own library location and
    Python has no reason to duplicate.

    It used to be answered by reading os.environ["FLAGOS_BACKEND_CONFIG"], which
    torch_fl wrote at import time. That made "the user overrode the conf" and
    "the wheel chose its own conf" the same observable state for the rest of the
    process, and it leaked the path into every child process the wheel spawns.
    """
    return _BACKEND_CONFIG_PATH or _env.value("FLAGOS_BACKEND_CONFIG") or ""


def _check_privateuse1_unclaimed() -> None:
    """Fail with an actionable message if a vendor plugin already took the key.

    PrivateUse1 admits exactly one backend name. `import torch` autoloads any
    `torch.backends` entry point -- torch_musa has one -- so when torch is
    imported before torch_fl, the name is already "musa" and our rename raises a
    bare "already been set!". The autoload guard only covers the
    torch_fl-first order, since by the time we run in the other order torch has
    already been initialised.
    """
    current = torch._C._get_privateuse1_backend_name()
    if current in ("privateuseone", "flagos"):
        return
    raise RuntimeError(
        f"PrivateUse1 is already claimed by the '{current}' backend, so torch_fl "
        "cannot register 'flagos'. A vendor plugin was autoloaded by `import "
        "torch` before torch_fl. Either import torch_fl first, or export "
        "TORCH_DEVICE_BACKEND_AUTOLOAD=0 before starting Python."
    )


# Python registration is empty because operators use the C++ dispatcher.
_registered_ops = []


def get_registered_ops():
    """Return list of registered FlagGems operators for flagos device."""
    return list(_registered_ops)


def is_flaggems_enabled():
    """Check if FlagGems operators are registered for flagos device."""
    return len(_registered_ops) > 0


def fused_kernels_available(device=None) -> bool:
    """Whether a fused Triton kernel can be compiled for this device.

    The predicate dispatch code should gate on instead of ``tensor.is_cuda``.
    ``is_cuda`` is False on flagos, so a caller choosing between
    ``torch.compile(flex_attention)`` and eager ``flex_attention`` by hand takes
    the eager branch and silently materializes the full scores matrix -- correct,
    and several times slower (issue #515).

    This asks the question the caller means: can inductor generate a Triton
    kernel for a tensor on this device, here, now. It is deliberately *not*
    ``optional_integration_status()["compile"] == "active"``: that state records
    only that the backend-registration imports succeeded, and nothing on that
    path probes Triton, so it reports ``active`` on a build whose vendor Triton
    backend is missing. ``raise_if_triton_unavailable`` is the honest check --
    it applies the CUDA compute-capability floor where one applies and otherwise
    lets the vendor backend decide -- so that is what runs here, with its
    exception turned into the boolean a branch needs.

    A False answer is a statement about this build and device, not about the
    operator: the eager path still computes the right result.
    """
    if _optional_integration_state.get("compile") != "active":
        return False
    try:
        from torch_fl.compile.device_interface import FlagOSDeviceInterface

        if not FlagOSDeviceInterface.is_available():
            return False
        FlagOSDeviceInterface.raise_if_triton_unavailable(device)
    except Exception:  # noqa: BLE001 - any failure to prove it means "no"
        return False
    return True


def device_capabilities(device=None) -> dict:
    """Capability flags for this build, for code that cannot import torch_fl APIs.

    One dict so a downstream framework can read every gate it needs in a single
    call, rather than discovering one predicate at a time. Keys are stable; new
    ones may be added, so read by key and do not unpack.
    """
    return {
        "compile": _optional_integration_state.get("compile") == "active",
        "fused_kernels": fused_kernels_available(device),
        "fused_attention": (
            fused_kernels_available(device)
            and _optional_integration_state.get("flex_attention") == "active"
        ),
        "flaggems": is_flaggems_enabled(),
    }


# ---------------------------------------------------------------------------
# Distributed: register "flagos" ProcessGroup backend for privateuseone
# ---------------------------------------------------------------------------


def _register_distributed_backend():
    """Register ProcessGroupFlagOS as the 'flagos' torch.distributed backend.

    After this call:
      - ``torch.distributed.init_process_group("flagos")`` works
      - ``torch.distributed.init_process_group(
            device_id=torch.device("privateuseone:0"))`` auto-selects "flagos"
      - All ``torch.distributed.*`` collectives work on flagos tensors without
        any monkeypatching — the ProcessGroup itself does the view conversion.
    """
    from torch_fl.comm import register_flagos_backend

    register_flagos_backend()


# Register torch.compile backend for flagos device (torch 2.0+)
def _register_compile_backend():
    """Register the 'flagos' backend with torch._dynamo if available."""
    try:
        from torch_fl.compile.inductor_backend import (
            register_backend,
            register_inductor_device,
        )

        register_backend()

        # The default Inductor backend also needs the FlagOS device entry.
        register_inductor_device()
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("torch.compile integration is unavailable") from exc


def _register_bpu_compile_backend() -> None:
    """Register torch.compile(backend="bpu") on a BPU build.

    The RDK BPU executes whole compiled graphs (a .hbm produced by hbdk4), not
    individual operators, so it has no per-op kernels: eager ops reach
    cpu_fallback and all acceleration comes through this backend. That is the
    opposite of every other platform here, where the compile path is incidental
    and the kernels do the work.

    The optional integration runner reports import failures independently. A
    board without the ONNX toolchain can still use its eager fallback.
    """
    if _platform.build_info().platform != "bpu":
        raise RuntimeError("BPU compile backend requires a BPU build")
    from torch_fl.accelerator import bpu

    bpu.register()


_OPTIONAL_INTEGRATIONS = (
    "apex",
    "ddp",
    "parallel_comm",
    "dataparallel",
    "data_parallel",
    "compile",
    "flex_attention",
    "bpu",
)
_optional_integration_state = {}


def _install_apex_compat() -> bool:
    from torch_fl.compat.apex import install_apex_compat

    return install_apex_compat()


def _install_flex_attention_compat() -> bool:
    from torch_fl.compat.flex_attention import install_flex_attention_compat

    return install_flex_attention_compat()


def optional_integration_status() -> dict:
    """Report each optional hook as inactive, active, or failed with a reason."""
    return {
        name: _optional_integration_state.get(name, "inactive")
        for name in _OPTIONAL_INTEGRATIONS
    }


def activate_optional_integrations(*names: str, strict: bool = True) -> dict:
    """Install optional framework hooks after the mandatory device bootstrap.

    Repeated calls do not wrap PyTorch objects twice. A failed hook is not
    retried in the same process because it may have partly changed global
    state. ``strict=False`` reports failures and continues with other hooks;
    the import-time compatibility profile uses that mode. Explicit calls raise
    so applications know when a requested integration was unavailable.
    """
    selected = names or tuple(
        name
        for name in _OPTIONAL_INTEGRATIONS
        if name != "bpu" or _platform.build_info().platform == "bpu"
    )
    unknown = sorted(set(selected) - set(_OPTIONAL_INTEGRATIONS))
    if unknown:
        raise ValueError(f"Unknown torch_fl integration(s): {', '.join(unknown)}")

    installers = {
        "apex": _install_apex_compat,
        "ddp": parallel.install_ddp,
        "parallel_comm": parallel.install_comm,
        "dataparallel": parallel.install_dataparallel,
        "data_parallel": parallel.install_data_parallel,
        "compile": _register_compile_backend,
        "flex_attention": _install_flex_attention_compat,
        "bpu": _register_bpu_compile_backend,
    }
    prerequisites = {
        "dataparallel": ("parallel_comm",),
        "data_parallel": ("parallel_comm", "dataparallel"),
        "flex_attention": ("compile",),
    }

    def activate(name: str) -> None:
        status = _optional_integration_state.get(name)
        if status == "active":
            return
        if status is not None:
            if strict:
                raise RuntimeError(f"torch_fl integration {name} previously {status}")
            return
        for prerequisite in prerequisites.get(name, ()):
            activate(prerequisite)
            if _optional_integration_state.get(prerequisite) != "active":
                message = f"failed: prerequisite {prerequisite} is unavailable"
                _optional_integration_state[name] = message
                if strict:
                    raise RuntimeError(f"torch_fl integration {name} {message}")
                _env.warn(f"Optional integration {name} {message}")
                return
        try:
            installed = installers[name]()
        except Exception as exc:  # noqa: BLE001 - independent optional hooks
            message = f"failed: {type(exc).__name__}: {exc}"
            _optional_integration_state[name] = message
            if strict:
                raise RuntimeError(f"torch_fl integration {name} {message}") from exc
            _env.warn(f"Optional integration {name} {message}")
        else:
            if installed is False:
                if strict:
                    raise RuntimeError(f"torch_fl integration {name} is unavailable")
            else:
                _optional_integration_state[name] = "active"

    for name in selected:
        activate(name)
    return optional_integration_status()


def _phase_conf() -> None:
    """Pick the op-routing conf and stage the MetaX cudart shim.

    One phase of the import-time pipeline below; the order is
    load-bearing, so the constraints are documented at the runner.
    """
    global _STARTUP_PROFILE
    _STARTUP_PROFILE = _env.choice(
        "FLAGOS_STARTUP_PROFILE", ("full", "minimal"), "full"
    )
    _select_backend_config()

    # Optional: PyTorch wheels may require libcudart.so.12 version tags on MetaX.
    if _env.flag("FLAGOS_METAX_CUDART_SHIM"):
        from torch_fl.accelerator.metax._metax_cudart_shim import ensure_cudart_shim

        ensure_cudart_shim()


def _phase_preload() -> None:
    """Select/preload the vendor libtorch and CUDA assets before `import torch`.

    One phase of the import-time pipeline below; the order is
    load-bearing, so the constraints are documented at the runner.
    """
    _loader.preload()


def _phase_claim() -> None:
    """Import torch, claim PrivateUse1, load _C and install the device module.

    One phase of the import-time pipeline below; the order is
    load-bearing, so the constraints are documented at the runner.
    """
    global torch, flagos
    import torch  # noqa: E402

    # Immediately after `import torch`, and before anything relies on CUDA dispatch:
    # confirm the DTK device libraries actually bound to the official core.
    _loader.validate_dcu_runtime()

    # A self-contained PPU build may front its bundled CUDA-enabled libtorch with
    # the official torch+cpu Python wheel. The actual runtime then supports CUDA
    # dispatch while torch/version.py still reports cuda=None. Restore that build
    # metadata before optional packages inspect it and select a native library.
    if _platform.build_info().platform == "ppu":
        from torch_fl.accelerator.thead._ppu_libtorch_link import (  # noqa: E402
            restore_ppu_cuda_version,
        )

        restore_ppu_cuda_version()

    if sys.platform == "win32":
        from ._utils import _load_dll_libraries

        _load_dll_libraries()
        del _load_dll_libraries

    # Optional FlagGems-on-MetaX compat (does not patch torch.cuda unless enabled).
    if _env.flag("FLAGOS_METAX_COMPAT"):
        from .compat.flaggems.device import install_metax_cuda_shim

        install_metax_cuda_shim()

    _loader.load_extension_dependencies(torch.__file__)

    # Checked *before* loading _C, not just before the rename: libtorch_fl.so
    # registers the AutogradPrivateUse1 fallback at dlopen time, and a vendor plugin
    # that already registered one makes that a std::terminate ("Tried to register
    # multiple backend fallbacks for the same dispatch key") -- an abort we cannot
    # catch or report. Running the check first turns that into the actionable
    # message below.
    _check_privateuse1_unclaimed()

    import torch_fl._C  # type: ignore[misc]  # noqa: E402, F401

    # Hand the conf _select_backend_config() resolved to the C++ reader, which builds
    # the routing table on the first op dispatch -- still well after this point.
    # This is a call rather than an os.environ write so the wheel's own choice stays
    # distinguishable from the user's FLAGOS_BACKEND_CONFIG; see
    # backend_config_path(). Nothing to hand over when the user set the variable or
    # nothing was found, in which case the C++ reader resolves it itself.
    if _BACKEND_CONFIG_PATH:
        torch_fl._C._set_backend_config_path(_BACKEND_CONFIG_PATH)

    from . import flagos  # noqa: E402

    if _platform.build_info().platform == "mps":
        # Finish the allocator before torch's module teardown clears Storage
        # objects. A DataPtr can then be deleted without the caching allocator
        # reinitialising during static destruction.
        import atexit

        atexit.register(_mps_shutdown_allocator)

    torch.utils.rename_privateuse1_backend("flagos")
    torch._register_device_module("flagos", flagos)
    torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

    if _platform.build_info().platform == "gcu":
        from .compat.flaggems.device import install_gcu_device_probes

        install_gcu_device_probes(flagos)
        # `has_triton` has to answer for the FlagTree enflame backend before
        # inductor's `create_backend` asks (it raises `TritonMissing` otherwise,
        # measured on this lane). `_prime_has_triton` is the routine that makes
        # it answer -- it borrows a table row and memoizes True -- and it is
        # normally first reached from `register_flagos_device_interface`, which
        # only runs when the compile backend starts, i.e. too late here.
        from torch_fl.compile.device_interface import _prime_has_triton

        _prime_has_triton()


def _phase_vendor_compat() -> None:
    """Install the vendor runtime shims and resolve GEMS_VENDOR (fail-loud).

    One phase of the import-time pipeline below; the order is
    load-bearing, so the constraints are documented at the runner.
    """
    from . import native_host
    from .compat.flaggems import device, runtime

    native_host._bootstrap_launcher()
    device.install_device_shims(_platform.build_info().vendor, flagos)

    # torch::utils::device_lazy_init(PrivateUse1) imports the module named
    # `torch_<backend_name>` and calls its _lazy_init(). It only does so once some
    # library has called set_requires_device_init(PrivateUse1, true) -- which
    # some vendor libraries do, so the very first flagos factory call can raise
    # "No module named 'torch_flagos'". Publishing the device module under that name
    # satisfies the lookup; flagos._lazy_init is the real initializer, so this is a
    # rename, not a stub. Harmless on backends that never trigger lazy init.
    sys.modules.setdefault("torch_flagos", flagos)

    # Apex's amp_C extension bypasses the ATen dispatcher and therefore cannot use
    # the normal DeviceBoxingGuard. Install an optional, CUDA-alias-only shim at the
    # common MultiTensorApply boundary; it remains lazy when Apex is not installed
    # and supports applications that import Apex either before or after torch_fl.
    if _STARTUP_PROFILE == "full":
        activate_optional_integrations("apex", strict=False)

    # Patch FlagGems codegen config before any FlagGems code is imported
    runtime.configure_vendor(_platform.build_info().vendor, flagos)
    # patch_philox() is deliberately NOT called here: it imports
    # flag_gems, and on a build where the vendor's own device name has to be
    # aliased first (TsingMicro) that import needs the CUDA alias to have
    # run. See its call site in _phase_ecosystem.


def _phase_ecosystem() -> None:
    """Mandatory FlagGems/distributed setup, then profile-selected framework hooks.

    One phase of the import-time pipeline below; the order is
    load-bearing, so the constraints are documented at the runner.
    """
    from .compat import cuda, triton
    from .compat.flaggems import device, ops, random, runtime

    accelerator = _platform.build_info().vendor
    device.prepare_cuda(accelerator)

    cuda.install_alias(flagos)
    if _env.flag("FLAGOS_COSMOS_CUDA_COMPAT"):
        from .compat import cosmos_attention

        cosmos_attention.install()

    # Both of these have to precede the flag_gems import below, which happens
    # here rather than in _phase_vendor_compat for exactly that reason. That
    # import reaches Triton's hint manager, which asks the driver factory to
    # resolve -- and the factory refuses while two backends claim to be active.
    # On a FlagTree wheel the alias installed just above is what makes the stock
    # nvidia backend claim that (its probe is `torch.cuda.is_available()`), so
    # the foreign driver has to be stood down first; `flagos._lazy_init` does
    # this too, but only on the first device operation, which is after this.
    # FlagTree's TsingMicro driver then answers `torch.device("txda", ...)`,
    # which resolves only once the alias has mapped "txda" to flagos.
    #
    # Getting either wrong makes the import raise, and a failed import leaves
    # every flag_gems submodule that *did* load in sys.modules bound to the
    # abandoned half-initialised package. FlagGems' own tensor_wrapper and
    # pointwise_dynamic read `flag_gems.vendor_name` off that binding, so ops
    # going through them (`cat` among them) then fail with `module 'flag_gems'
    # has no attribute 'vendor_name'` for the life of the process.
    triton.stand_down_foreign_drivers(cuda.is_active())
    random.patch_philox(accelerator, flagos)

    # Correct flag_gems' layer-norm backward for 3-D+ inputs; must precede the
    # first op call, which is where the generated kernel resolves the callable.
    ops.patch_layer_norm_backward(accelerator)

    runtime.prepare_dispatch(accelerator)
    # Dispatch preparation can import op modules with their own copy of the
    # seed function the bridge replaced above.
    random.refresh_philox_bindings()

    from . import quantization  # noqa: E402, F401

    _register_distributed_backend()

    if _STARTUP_PROFILE == "full":
        activate_optional_integrations(
            "ddp",
            "parallel_comm",
            "dataparallel",
            "data_parallel",
            "compile",
            "flex_attention",
            strict=False,
        )
        if _platform.build_info().platform == "bpu":
            activate_optional_integrations("bpu", strict=False)

    from . import native_host

    native_host._activate()
    if _env.flag("FLAGOS_COSMOS_CUDA_COMPAT"):
        from .compat.cosmos_numerics import install_cuda_numerics

        install_cuda_numerics()


def __getattr__(name: str):
    # Retain the status probe used by existing applications and diagnostics.
    if name == "_cuda_alias_active":
        from .compat.cuda import is_active

        return is_active()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ===========================================================================
# Import-time phase pipeline.
#
# torch_fl runs a fixed sequence of side effects at import. The order is
# load-bearing -- a wrong order produces a dlopen abort or a wrong-vendor build,
# not a Python exception -- so it lives in exactly one place here instead of
# being implied by where each statement happens to sit in the file.
#
#   1. conf          pick the op-routing config and stage the MetaX shim
#   2. preload       select/preload the vendor libtorch and CUDA assets
#   3. claim         import torch, free PrivateUse1, load _C, install the device
#   4. vendor_compat install the vendor runtime shims and resolve GEMS_VENDOR
#   5. ecosystem     mandatory FlagGems/distributed setup, then optional hooks
#
# Constraints, each next to the phase it constrains:
#   * preload before claim: the vendor libtorch has to be in place before
#     `import torch`, and the CUDA assets before _C is dlopened.
#   * within claim, the PrivateUse1 check precedes `import torch_fl._C`: the .so
#     registers the AutogradPrivateUse1 fallback at dlopen time, and a vendor
#     plugin that already claimed the key turns that into an uncatchable abort.
#   * claim before vendor_compat: FlagGems reads GEMS_VENDOR at its import, so
#     the shims and the vendor resolution must precede the ecosystem hooks.
#   * vendor_compat before ecosystem: the FlagGems/dist/compile hooks act on the
#     vendor surface the previous phase installed.
# ===========================================================================
_phase_conf()
_phase_preload()
_phase_claim()
_phase_vendor_compat()
_phase_ecosystem()


__all__ = [
    "flagos",
    "distributed",
    "get_registered_ops",
    "is_flaggems_enabled",
    "fused_kernels_available",
    "device_capabilities",
    "activate_optional_integrations",
    "optional_integration_status",
    "quantization",
]
