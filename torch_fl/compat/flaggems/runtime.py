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

"""Prepare the FlagGems runtime for the selected wheel and device."""

import os
import re
import sys
from pathlib import Path

from ... import _env, _vendor


def routes_to_flaggems() -> bool:
    """True when the selected conf routes at least one op to a FlagGems path.

    The FlagGems Python path needs process-level setup before flag_gems is
    imported (the torch.musa surface its MThreads backend selects on, the Philox
    seed bridge). That setup used to be gated on FLAGOS_USE_FLAGGEMS=1, which no
    longer decides anything: a generated vendor conf is FlagGems-first by
    construction, so the opt-in is gone and the gate would never open -- leaving
    flag_gems to fail its own backend discovery on the very ops the conf routes
    to it. Read the conf instead, which is the thing that actually decides.
    """
    from ... import backend_config_path

    conf = backend_config_path()
    if not conf or not os.path.exists(conf):
        return False
    try:
        with open(conf) as f:
            for line in f:
                value = line.split("#")[0].partition("=")[2].strip()
                backends = [
                    part.rsplit(":", 1)[-1].strip() for part in value.split(";")
                ]
                if any(
                    backend in ("flaggems", "flaggems_python", "flagos_python")
                    for backend in backends
                ):
                    return True
    except OSError:
        return False
    return False


def _restore_dcu_hip_version() -> None:
    """Set torch.version.hip/rocm for a self-contained DCU wheel.

    See the DCU branch of configure_vendor() for why this matters:
    the bundled libtorch is DTK's HIP build, but torch/version.py comes from the
    stock torch+cpu wheel in front and reports hip=None, which switches triton's
    hcu backend off. scripts/vendor/bundle_hygon_libtorch.sh copies the vendor torch's own
    version.py next to the bundled .so as vendor_version.py; read the strings
    back from there. No-op when a real vendor torch is in front.
    """
    import torch

    if getattr(torch.version, "hip", None):
        return  # a real DTK torch is in front; leave its values alone.

    hip_ver = _env.value("FLAGOS_DCU_HIP_VERSION", "")
    rocm_ver = ""
    if not hip_ver:
        ver_py = os.path.join(
            Path(__file__).parents[2], "lib_hygon", "vendor_version.py"
        )
        try:
            with open(ver_py, encoding="utf-8") as f:
                for line in f:
                    m = re.match(r"\s*hip\s*(?::[^=]*)?=\s*'([^']+)'", line)
                    if m:
                        hip_ver = m.group(1)
                        continue
                    m = re.match(r"\s*rocm\s*(?::[^=]*)?=\s*'([^']+)'", line)
                    if m:
                        rocm_ver = m.group(1)
        except OSError:
            return  # not a bundled build (source checkout); nothing to restore.
    if hip_ver:
        torch.version.hip = hip_ver
        if rocm_ver:
            torch.version.rocm = rocm_ver


def configure_vendor(accelerator: str, flagos) -> None:
    """
    Configure FlagGems' vendor + torch.cuda shim for the flagos device.

    FlagGems uses GEMS_VENDOR env var to detect the hardware vendor.

    - Generic NVIDIA CUDA (default when a real NVIDIA GPU is reachable via
      libcuda.so and MetaX compat is not requested): set GEMS_VENDOR=nvidia and
      shim torch.cuda so FlagGems' Triton kernels can compile/run under CPU
      torch + external libtorch_cuda.so. GEMS_VENDOR=nvidia is REQUIRED so
      FlagGems' tl_extra_shim resolves triton.language.extra.cuda.libdevice
      (which has `pow`); otherwise it falls back to tl.math (no `pow`).
      Disable with FLAGOS_DISABLE_CUDA_SHIM=1.

    - MetaX (boxing + FlagGems): set GEMS_VENDOR=metax so FlagGems uses the
      MetaX codegen config (triton-metax backend, prefer_block_pointer=False to
      avoid the triton-metax block-pointer bug), and patch torch.cuda (device
      props + stream/availability) so FlagGems' Triton kernels run on the
      CPU-frozen torch wheel against maca's libtorch_cuda.so. Auto-selected on a
      MetaX build or with FLAGOS_METAX_COMPAT=1 when a MetaX
      card is present.

    - Hygon DCU (DTK): set GEMS_VENDOR=hygon so FlagGems uses its _hygon
      codegen config (triton hcu backend, triton_extra_name="hip"). No
      torch.cuda shim is needed -- DTK ships a hipified torch with a real,
      working torch.cuda. This branch must precede the generic-NVIDIA one:
      is_nvidia_cuda_available() is False on DTK (there is no libcuda.so, only
      libgalaxyhip), so without it DCU would reach the ascend fallback and get
      GEMS_VENDOR=ascend -- which also breaks the comm layer, since that vendor
      selects the HCCL profile (see comm/process_group.py _VENDOR_PROFILES).

    - Ascend: set GEMS_VENDOR=ascend so FlagGems uses the ASCEND codegen
      config (prefer_block_pointer=False, avoiding the Ascend Triton backend's
      tl.make_block_ptr bug), and register torch.flagos as a torch.npu shim so
      FlagGems' gen_torch_device_object('ascend') resolves correctly. This
      branch is taken only for an Ascend build or an explicit
      GEMS_VENDOR=ascend; any other accelerator reaching this point had vendor
      detection fail, which raises rather than silently selecting Ascend.

    - Iluvatar (corex): set GEMS_VENDOR=iluvatar so FlagGems uses its _iluvatar
      codegen tree. Iluvatar is FlagGems-primary -- every covered op routes to
      flaggems in backends_iluvatar.conf and nothing boxes to CUDA. The branch
      must precede the generic-NVIDIA one: corex's runtime is CUDA-shaped, so a
      runtime probe cannot separate iluvatar from a real NVIDIA GPU and
      GEMS_VENDOR=nvidia would select FlagGems' NVIDIA backend instead.
    """
    import torch

    # An explicitly exported GEMS_VENDOR that torch_fl cannot configure is a
    # mistake to surface now, not to hand to FlagGems and the comm layer: the
    # branches below would leave it in place (set_foreign never overrides an
    # explicit export) and the failure would appear as a wrong-vendor crash
    # later. Unknown values raise; unset/blank returns None.
    _explicit_vendor = _vendor.validate_explicit(os.environ.get("GEMS_VENDOR"))

    # --- Apple MPS branch ---
    # The Metal backend ships no FlagGems route at all (the platform table turns
    # FLAGOS_BUILD_FLAGGEMS off and backends_apple.conf is empty; there is no Metal
    # Triton line to build against), so there is no vendor to name and no codegen
    # config to patch. Keyed on the build accelerator because, unlike every other
    # branch below, MPS has no vendor runtime probe that could have failed --
    # reaching the ascend fallback would mislabel the vendor and abort the import.
    if accelerator == "apple":
        return

    # --- Moore Threads MUSA branch ---
    if accelerator == "mthreads":
        _env.set_foreign("GEMS_VENDOR", "mthreads")
        return

    # --- Kunlunxin XPU branch (native vendor kernels) ---
    # Keyed on the build accelerator, like MUSA. The de-boxed Kunlunxin wheel
    # talks to the vendor SDK (libxpurt + xdnn_pytorch) over the flagos
    # PrivateUse1 device rather than the CUDA key, so the generic NVIDIA branch
    # below must not claim it: its CUDA shim reports itself available through the
    # vendor's CUDA-compatible runtime headers and then installs CUDA-backed
    # generators, rebinding torch.flagos.get_rng_state/set_rng_state onto
    # torch.Generator(device="cuda") -- which this CPU wheel cannot construct
    # ("Cannot get CUDA generator without ATen_cuda library"). The native
    # torch_fl.flagos.random module (PrivateUse1-backed _C._get_default_generator)
    # is the correct implementation and must be left in place, so this branch
    # only labels the vendor and returns before any shim runs.
    if accelerator == "kunlunxin":
        _env.set_foreign("GEMS_VENDOR", "kunlunxin")
        # ... and install the shim this platform's own FlagGems routes need.
        # See torch_fl/accelerator/kunlunxin/_klx_compat.py for what each piece
        # removes; both were measured on a P800. This is deliberately NOT the
        # generic-NVIDIA shim the comment above rejects -- that one installs
        # CUDA-backed generators, which this CPU wheel cannot construct. This
        # one forwards the handful of torch.cuda entry points FlagGems calls to
        # torch.flagos, and leaves the generators alone, so the native
        # PrivateUse1 RNG stays in place.
        from torch_fl.accelerator.kunlunxin._klx_compat import (
            install_klx_cuda_shim,
            patch_klx_flag_gems_device_name,
        )

        install_klx_cuda_shim()
        try:
            import flag_gems  # noqa: F401
        except Exception:
            pass
        else:
            try:
                patch_klx_flag_gems_device_name()
            except Exception:
                pass
        return

    # --- MetaX branch (boxing + FlagGems) ---
    # Auto-detects MetaX hardware (like DCU does) or triggered by explicit
    # FLAGOS_METAX_COMPAT=1. Must come before the ascend
    # fallback so MetaX never wrongly gets GEMS_VENDOR=ascend.
    _metax_requested = accelerator == "metax" or _env.flag("FLAGOS_METAX_COMPAT")
    if _metax_requested and os.environ.get("GEMS_VENDOR") not in ("nvidia", "ascend"):
        from torch_fl.accelerator.metax._metax_compat import (
            is_metax_available,
            patch_torch_cuda_for_metax,
        )

        if is_metax_available():
            _env.set_foreign("GEMS_VENDOR", "metax")
            patch_torch_cuda_for_metax()
            return

    # --- Hygon DCU branch (DTK) ---
    # Keyed on the build accelerator rather than probing the runtime: DTK's torch
    # is hipified, so torch.cuda/torch.version.hip look "cuda-ish" and no
    # libcuda.so probe can tell the two apart. Must come before both the generic
    # NVIDIA branch (which no-ops here anyway -- no libcuda.so) and the ascend
    # fallback. setdefault so an explicit GEMS_VENDOR still wins.
    if accelerator == "hygon" and os.environ.get("GEMS_VENDOR") != "ascend":
        _env.set_foreign("GEMS_VENDOR", "hygon")
        # torch.version is pure Python (torch/version.py), generated when the
        # wheel is built -- swapping the bundled DTK .so files cannot change it.
        # A self-contained DCU wheel therefore front-ends a stock torch+cpu whose
        # torch.version.hip is None, while the DTK torch it replaces reports
        # e.g. "6.3.26113". triton's hcu backend gates on exactly that value
        # (backends/hcu/driver.py is_active(): torch.cuda.is_available() and
        # torch.version.hip is not None), so with None the driver never activates
        # and any flag_gems op dies in triton's driver factory with
        # "0 active drivers ([]). There should only be one." Restore the attribute
        # from the bundled libtorch's own version so triton sees a HIP torch,
        # matching what the vendor wheel reported.
        _restore_dcu_hip_version()
        from torch_fl.accelerator.hygon._dcu_compat import (
            install_dcu_rng_bridge,
            patch_torch_cuda_for_dcu,
        )

        # Order matters. A decoupled DCU wheel runs on the official torch+cpu
        # wheel, whose torch.cuda reports is_available()=False and raises from
        # _lazy_init() ("Torch not compiled with CUDA enabled") -- the other half
        # of triton's hcu gate, alongside the torch.version.hip restored above. So
        # patch torch.cuda first (a no-op when a real DTK torch is in front), then
        # bridge the flagos RNG onto the CUDA generators it now exposes.
        patch_torch_cuda_for_dcu()
        install_dcu_rng_bridge()
        return

    # --- TsingMicro branch ---
    # Keyed on the build accelerator: the TX8 host has no libcuda.so and no
    # /dev/davinci*, so without this branch the vendor falls through to the
    # ascend fallback below and gets GEMS_VENDOR=ascend. flag_gems picks its
    # backend from that variable, and only TSINGMICRO resolves the torch.txda
    # device object the shim above publishes -- so the fallback does not merely
    # mislabel the vendor, it makes device discovery fail outright.
    if accelerator == "tsingmicro" and os.environ.get("GEMS_VENDOR") != "ascend":
        os.environ.setdefault("GEMS_VENDOR", "tsingmicro")
        return

    # --- Enflame GCU branch ---
    # Keyed on the build accelerator for the same reason as DCU: no runtime probe
    # distinguishes GCU here, and the tops stack has no libcuda.so, so without
    # this branch GCU would reach the ascend fallback and get GEMS_VENDOR=ascend
    # (which also picks the wrong comm profile). FlagGems' Triton kernels need a
    # vendor Triton backend for the GCU -- FlagTree's enflame backend, or
    # Enflame's older triton_gcu plugin with its /opt/triton_gcu toolchain. If
    # neither is installed, patch_gcu_triton_for_flagos() returns False and we
    # leave GEMS_VENDOR unset so the topsaten kernels and cpu_fallback stay in
    # charge.
    if accelerator == "enflame" and os.environ.get("GEMS_VENDOR") != "ascend":
        from torch_fl.accelerator.enflame._gcu_compat import (
            install_gcu_rng_generators,
            patch_diffusers_qwenimage_rope,
            patch_gcu_triton_for_flagos,
        )

        install_gcu_rng_generators()
        if patch_gcu_triton_for_flagos():
            _env.set_foreign("GEMS_VENDOR", "enflame")
        # diffusers keys the Qwen-Image rotation on device type, so without this a
        # flagos tensor multiplies by a complex exponential the topsaten stack
        # serves slowly -- 57.5% of a transformer forward against 34.0% for the
        # real-valued entry diffusers provides for a device with no complex dtype.
        # It has to run here rather than at the call site: the operand half is
        # cached per device on first use, and this is the last point that is
        # reliably before the pipeline exists. Costs a diffusers import -- the
        # call is a no-op when diffusers is not installed.
        patch_diffusers_qwenimage_rope()
        return

    # --- Cambricon MLU branch (vendor-native, FlagGems through a shim) ---
    # Keyed on the build accelerator like MUSA/Kunlunxin. The wheel runs on the
    # stock torch+cpu; torch_mlu is deliberately not installed, so nothing in the
    # process publishes torch.mlu for FlagGems' cambricon backend or for the
    # Triton mlu driver underneath it. install_cambricon_compat() synthesizes
    # that surface from torch_fl.flagos -- the TsingMicro shim's pattern applied
    # to a vendor whose plugin cannot be loaded at all.
    #
    # Returning here also keeps the generic NVIDIA branch below from claiming the
    # platform: flagos tensors live in cnrt memory, so the CUDA shim would
    # install CUDA-backed generators for them.
    if accelerator == "cambricon":
        from torch_fl.accelerator.cambricon._cambricon_compat import (
            install_cambricon_compat,
        )

        install_cambricon_compat()
        _env.set_foreign("GEMS_VENDOR", "cambricon")
        return

    # --- Iluvatar branch ---
    # Keyed on the build accelerator for the same reason as DCU: the vendor torch
    # is a real CUDA build over corex's CUDA-shaped runtime, so
    # is_nvidia_cuda_available() can report True and the generic-NVIDIA branch
    # below would claim the platform with GEMS_VENDOR=nvidia -- selecting
    # FlagGems' NVIDIA backend instead of _iluvatar. No runtime probe distinguishes
    # the two, so the build record is the only authoritative key.
    if accelerator == "iluvatar":
        _env.set_foreign("GEMS_VENDOR", "iluvatar")
        # FlagGems' _iluvatar descriptor is a copy of its NVIDIA one, so it reads
        # torch.cuda directly in two places the CPU wheel does not back: the
        # philox generators behind every generator-less RNG op, and the device
        # name every op compares its input against.
        from torch_fl.accelerator.iluvatar._ilu_compat import (
            _ensure_corex_headers_on_path,
            install_ilu_cuda_shim,
            install_ilu_rng_generators,
            patch_iluvatar_flag_gems_device_name,
            patch_triton_benchmark_allocator,
        )

        _ensure_corex_headers_on_path()
        install_ilu_cuda_shim()
        install_ilu_rng_generators()
        # Triton's autotuner flushes the L2 with a buffer it allocates on
        # "cuda", which this build does not have; without this every autotuned
        # kernel (layernorm among them) fails before it is benchmarked.
        try:
            patch_triton_benchmark_allocator()
        except Exception:
            pass
        try:
            import flag_gems  # noqa: F401
        except Exception:
            pass
        else:
            try:
                patch_iluvatar_flag_gems_device_name()
            except Exception:
                pass
        return

    # --- Generic NVIDIA CUDA branch (default) ---
    if (
        not _env.flag("FLAGOS_DISABLE_CUDA_SHIM")
        and not _env.flag("FLAGOS_METAX_COMPAT")
        and accelerator != "metax"
        and os.environ.get("GEMS_VENDOR") != "ascend"
    ):
        from torch_fl.accelerator.nvidia._cuda_compat import (
            is_nvidia_cuda_available,
            patch_torch_cuda_for_flagos,
        )

        if is_nvidia_cuda_available():
            _env.set_foreign("GEMS_VENDOR", "nvidia")
            # patch_torch_cuda_for_flagos installs per-device CUDA generators as
            # torch.cuda.default_generators and routes cuda seeding to them, so
            # flag_gems' philox_backend_seed_offset finds a real, seedable
            # generator on its own -- no philox monkeypatch needed.
            patch_torch_cuda_for_flagos()
            return

    # --- Ascend branch, or the detection-failure guard ---
    # An explicit GEMS_VENDOR that reached this far names a vendor none of the
    # branches above claimed. Honor it (set_foreign could not override it) and
    # do not install the Ascend shims for a vendor that is not Ascend.
    if _explicit_vendor is not None and _explicit_vendor != "ascend":
        return

    # BPU has no FlagGems path at all (its kernels are whole compiled graphs),
    # and FLAGOS_DISABLE_CUDA_SHIM=1 is an explicit opt-out of vendor shimming:
    # both leave GEMS_VENDOR unset rather than claiming the Ascend config.
    if accelerator == "bpu" or _env.flag("FLAGOS_DISABLE_CUDA_SHIM"):
        return

    # Ascend is the only remaining legitimate user of the Ascend codegen config.
    # Any other accelerator here means vendor detection failed with GEMS_VENDOR
    # unset -- the silent-ascend fallback this replaces. Fail loud: the previous
    # behavior handed FlagGems the wrong vendor and surfaced the breakage far
    # from its cause (a CUDA wheel with no reachable GPU, a MetaX wheel with no
    # card). An explicit GEMS_VENDOR=ascend still selects this branch.
    if accelerator != "ascend" and _explicit_vendor != "ascend":
        raise RuntimeError(
            f"FlagGems vendor detection failed for FLAGOS_ACCELERATOR="
            f"{accelerator!r}: no vendor runtime was reachable and "
            f"GEMS_VENDOR is unset. Set GEMS_VENDOR to one of "
            f"{sorted(_vendor.KNOWN_VENDORS)} to select a vendor explicitly."
        )

    # Set vendor before FlagGems runtime initializes
    _env.set_foreign("GEMS_VENDOR", "ascend")

    # FlagGems' RNG ops (rand/randn/uniform_/exponential_/bernoulli_/
    # multinomial/native_dropout) unpack the generator state as a CUDA philox
    # (seed, offset) pair. torch_fl's flagos generator is a CPU mt19937, so
    # install per-device philox state objects for them -- the same bridge GCU
    # uses, and for the same reason. Must run before the torch.npu shim below,
    # which copies default_generators off the flagos module.
    from torch_fl.accelerator.ascend._ascend_compat import (
        install_ascend_rng_generators,
    )

    install_ascend_rng_generators()

    # FlagGems' ASCEND backend expects torch.npu to exist (device_name="npu").
    # Provide torch.flagos as a shim so gen_torch_device_object() succeeds.
    # Mark is_available()=False so transformers/accelerate don't think real
    # NPU hardware is present and try to import npu_fusion_attention etc.
    if not hasattr(torch, "npu"):
        import types

        _npu_device_shim = types.ModuleType("torch.npu")
        _npu_device_shim.is_available = lambda: False
        _npu_device_shim.device_count = flagos.device_count
        _npu_device_shim.current_device = flagos.current_device
        _npu_device_shim.set_device = flagos.set_device
        _npu_device_shim.synchronize = flagos.synchronize
        _npu_device_shim.device = flagos.device
        _npu_device_shim.Stream = flagos.Stream
        _npu_device_shim.Event = flagos.Event
        _npu_device_shim.current_stream = flagos.current_stream
        _npu_device_shim.default_generators = flagos.default_generators
        # FlagGems' utils/triton_driver_helper.py captures
        # torch_device_fn.get_device_properties at import time and falls back to
        # triton's driver on AttributeError -- and the Ascend backend's version
        # returns a plain dict, so gems' `get_device_properties(idx).multi_processor_count`
        # raises AttributeError deep inside a kernel launch. cumsum hit this on the
        # (1, 151936) logits of Qwen3's sampler, failing every generate() on the
        # gems path while smaller shapes took a branch that never queried it.
        _npu_device_shim.get_device_properties = flagos.get_device_properties
        torch.npu = _npu_device_shim

    # The Ascend Triton backend (FlagTree and triton-ascend alike) imports
    # torch_npu in _get_vendor_from_quick_cmd. Provide a minimal shim module so
    # the import doesn't fail.
    # Also set __spec__ to satisfy importlib.util.find_spec() checks (used by
    # accelerate.utils.imports.is_npu_available).
    # backend_register.py also checks torch_npu._C for stream APIs.
    if "torch_npu" not in sys.modules:
        import types
        import importlib.machinery

        _npu_shim = types.ModuleType("torch_npu")
        _npu_shim.npu = _npu_device_shim
        _npu_shim.__spec__ = importlib.machinery.ModuleSpec(
            name="torch_npu",
            loader=None,
            origin="torch_fl_shim",
        )
        # backend_register.py checks hasattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait")
        # Provide a minimal _C shim with mock stream functions
        _npu_c_shim = types.ModuleType("torch_npu._C")

        # Mock stream API - return flagos stream handle
        def _mock_get_current_stream(device_index=0):
            return flagos.current_stream(device_index).cuda_stream

        _npu_c_shim._npu_getCurrentRawStreamNoWait = _mock_get_current_stream
        _npu_shim._C = _npu_c_shim
        # Marks this as torch_fl's stub rather than the real extension. The
        # real torch_npu owns PrivateUse1 and would lock flagos out, so code
        # that has to tell them apart (FlagTree's Ascend backend policy, which
        # uses a torch_npu-backed implementation only if torch_npu is real)
        # cannot go by presence or by `_C` alone.
        _npu_shim.__torch_fl_shim__ = True
        sys.modules["torch_npu"] = _npu_shim
        sys.modules["torch_npu._C"] = _npu_c_shim

    # diffusers keys the Qwen-Image rotary embedding on device type and knows
    # only `cuda` (complex exponential) and `neuron` (rotation angles). An
    # unlisted device takes the `cuda` fallback, whose operands are complex --
    # and CANN has no complex compute at all, so `_compute_video_freqs` raises
    # "Unsupported dtype for ACL: ComplexFloat" from the `torch.cat` over
    # `freqs_neg`/`freqs_pos` before the first rotation is applied, and the
    # transformer cannot run. Registering the `flagos` device in both halves of
    # that extension point is what makes the rotation expressible on this backend.
    #
    # It has to run here rather than at the call site: `_get_device_freqs` caches
    # per device on first use, so this is the last point reliably before the
    # pipeline exists. Costs a diffusers import -- the call is a no-op when
    # diffusers is not installed, when it has no Qwen-Image rope table, or when
    # FLAGOS_DISABLE_QWENIMAGE_ROPE is set. See
    # torch_fl/accelerator/ascend/_ascend_compat.py.
    from torch_fl.accelerator.ascend._ascend_compat import (
        patch_diffusers_qwenimage_rope,
    )

    patch_diffusers_qwenimage_rope()


def prepare_dispatch(accelerator: str):
    """
    Prepare FlagGems for the flagos dispatch key.

    No Python-layer FlagGems registration happens here: every FlagGems op is
    dispatched through the C++ stub path, addressed per overload by
    FLAGOS_BACKEND_CONFIG. Only the GCU runtime shims below are installed eagerly.
    """
    if _env.flag("FLAGOS_DISABLE_FLAGGEMS_PY"):
        return

    import importlib.util

    if importlib.util.find_spec("flag_gems") is None:
        # flag_gems not installed, will use cpu_fallback
        return

    # GCU FlagGems uses the C++ dispatcher path. Its generated kernels are
    # compiled when FLAGOS_BUILD_FLAGGEMS=ON and selected per overload by
    # FLAGOS_BACKEND_CONFIG. Calling flag_gems.enable() here would register a
    # competing PrivateUse1 implementation and bypass the shared dispatcher.
    if accelerator == "enflame":
        from torch_fl.accelerator.enflame._gcu_compat import is_gcu_triton_available

        if not is_gcu_triton_available():
            return
        try:
            import flag_gems

            from torch_fl.accelerator.enflame._gcu_compat import (
                bind_vendor_ops_in_generic_modules,
                patch_flaggems_device_name,
            )

            patch_flaggems_device_name()
            bind_vendor_ops_in_generic_modules(flag_gems)
        except Exception as exc:
            import warnings

            warnings.warn(
                f"FlagGems runtime preparation for GCU failed ({type(exc).__name__}: "
                f"{exc}); flagos_python routes may be unavailable.",
                stacklevel=2,
            )


def initialize(accelerator: str, cuda_alias_active: bool) -> None:
    """Install compatibility for the device runtime after its first initialization."""
    from .. import triton
    from . import autotune, launch, ops

    # Before FlagGems is importable, Triton has to agree with torch_fl on which
    # accelerator owns this process: triton's driver factory refuses to resolve
    # while two backends claim to be active, and on a FlagTree wheel the CUDA
    # alias above makes the nvidia backend claim exactly that.
    triton.stand_down_foreign_drivers(cuda_alias_active)

    # FlagGems' kernels are Triton kernels, and on a FlagTree Ascend build the
    # host side of every one of them is generated by FlagTree's policy layer,
    # which needs to be pointed at torch_fl before the first launch.
    triton.install_ascend_policy()

    # FlagGems reads a FlagTree autotuning knob by a name 0.7.0 removed, and
    # raises on the read instead of falling through. Restore the name before the
    # first FlagGems kernel; see the function for the exact build mapping.
    autotune.align_flagtree_knobs(accelerator)

    # FlagGems' Ascend backend imports triton.experimental.tle.language at module
    # scope, and on FlagTree that import cannot complete without AscendSHMEM,
    # which no FlagTree wheel ships. Left alone, `import flag_gems` fails and
    # with it every FlagGems kernel on Ascend; see the module for why a
    # placeholder is the right answer here.
    from torch_fl.compile.flagtree_ascend_tle import ensure_tle_importable

    ensure_tle_importable()

    # FlagGems' autotuned kernels ask Triton for their config on the first call
    # of every process, and Triton's persisted autotune cache is off by default.
    # The knob is read when the @triton.autotune decorators run, i.e. while the
    # import below executes. See the function for the measurement.
    autotune.enable_cache(accelerator)

    # Eagerly import FlagGems to avoid deep import chain during dispatch.
    # FlagGems has a deep lazy import chain (fused → FLA → utils → models → sqlalchemy)
    # that can exceed Python's recursion limit when triggered inside PyTorch dispatch.
    #
    # Not just ImportError: FlagGems' vendor autodetect raises RuntimeError("No
    # device were detected on your machine") when it is installed but cannot
    # identify the backend -- which is the normal state on a vendor box whose
    # Triton plugin is absent, and must not take down device init. The FlagGems
    # kernels are optional everywhere; whatever is not registered runs on the
    # native or CPU path.
    try:
        import flag_gems  # noqa: F401
    except Exception:
        pass  # FlagGems unavailable or undetectable here, skip

    # Triton has to hash its own build before it can read a single compiled
    # kernel out of its on-disk cache, and that hash is the last thing that
    # cannot be warmed by the patches below. Start it here, while the caller is
    # still materialising weights. See the function for the measurement.
    autotune.prewarm_triton_key(accelerator)

    # The tuner repair goes in after the import above, not before it: the
    # mthreads `mm` tune space asks more shared memory on float64 than this
    # device has, so a matmul it refuses has to be repaired rather than
    # retried, and the entries that need it are built by the FlagGems modules
    # this import loads (and, for `mm`, by the lazy import of
    # `flag_gems.ops.mm` on the first call -- see `refresh_tune_space`
    # for that half). See the function for the measurement.
    autotune.patch_tune_space(accelerator)

    # FlagGems locates itself in the device namespace by the vendor's own device
    # string ("cuda" for nvidia), while torch_fl registers this accelerator as
    # "flagos". Its ops compare the two and hand the call back to ATen when they
    # disagree, which they always did -- so the routed ops that make that
    # comparison never ran their kernel. Realigning the names has to happen after
    # the import above, because both the detector singleton and the op modules'
    # module-level copies are fixed by then.
    ops.align_device_identity()

    # Same window, same reason: FlagGems' L2 vector_norm materialises a
    # transposed copy of its input, and the C++ bridge caches the callable it
    # resolved, so this has to be in place before the first routed op runs. See
    # the function for the measurement. No-op outside the vendors it was
    # measured on.
    ops.patch_vector_norm(accelerator)

    # Same window and the same caching rule: these two pick a better FlagGems
    # kernel for a square and for a short last-dim mean, which is what the q/k
    # RMSNorm is made of. See each function for the measurement.
    ops.patch_pow_scalar_square(accelerator)
    ops.patch_mean_last_dim(accelerator)

    # Same window as the two above, and the same requirement: the dispatcher
    # kernel is claimed before the first routed op can reach it. See the
    # function for the measurement.
    ops.patch_index_put(accelerator)

    # Same window again: these replace the launch path every FlagGems pointwise
    # op goes through, so they have to be in place before the first one runs.
    # See the function for the measurement.
    launch.patch_pointwise_dispatch(accelerator)
