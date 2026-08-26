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
ProcessGroupFlagOS — a transparent ProcessGroup backend for flagos (PrivateUse1).

Architecture
------------
flagos tensors share physical GPU memory with CUDA tensors. This ProcessGroup
wraps an underlying comm backend (FlagCX if available, otherwise NCCL/HCCL)
and converts privateuseone tensors to the backend's expected device view inside
each collective virtual method. The Work objects returned are from the inner
backend, so callers (including the DDP Reducer) get properly typed futures.

Vendor selection is table-driven by GEMS_VENDOR (see ``_VENDOR_PROFILES``).
For every vendor the inner-backend priority is FlagCX first, then the vendor's
native backend (NCCL for CUDA-ABI vendors, HCCL for Ascend).

View conversion
    flagos tensors on CUDA-ABI vendors
    (nvidia/metax/iluvatar/kunlunxin/du/thead/hygon)
    share physical GPU memory with CUDA, so they are reinterpreted as CUDA
    tensors via a zero-copy shared-storage view (``_C._flagos_to_cuda_view``)
    before being handed to NCCL / FlagCX-cuda. The underlying buffer is the
    same physical memory, so the backend's write-back is visible to the flagos
    side immediately. Vendors whose flagos tensor is NOT a cuda alias
    (ascend/musa/cambricon) have ``view=None`` in their profile and currently
    require the FlagCX path; see ``_resolve_view``.

GCU device guard
    Enflame GCU streams and pointers are device-scoped. The communicator must
    run on the same device as the tensor, or operations silently no-op. Every
    collective method on enflame applies a device guard that forces the current
    device to match the operand device before dispatch.
"""

import functools
import os
import warnings

import torch
import torch.distributed as dist
from torch._C import _distributed_c10d as _c10d


# ---------------------------------------------------------------------------
# Vendor profiles
#
# GEMS_VENDOR selects the hardware backend (see torch_fl/__init__.py
# _patch_flaggems_codegen_config). Each vendor differs in three comm-relevant
# ways, captured here so _build_inner stays table-driven — adding a new vendor
# means adding one row, not editing branch logic:
#
#   flagcx_dev   The device name FlagCX registers its "flagcx" backend under
#                (must match backend_flagcx.hpp flagcxBackendConstructor). For
#                CUDA-ABI vendors this is "cuda"; Ascend uses "cann", MUSA
#                "musa", Cambricon "mlu", etc.
#   view         Name of the torch_fl._C helper that reinterprets a flagos
#                (privateuseone) tensor as the physical tensor the comm backend
#                expects, or None when the flagos tensor is NOT a zero-copy
#                alias of that device (then no safe view exists yet).
#   native       Callable(self, store, rank, world_size, timeout) -> bool that
#                tries to build the vendor's *native* inner backend (used when
#                FlagCX is unavailable). Returns True on success.
#
# cuda_alias vendors: flagos shares physical GPU memory with CUDA, so a flagos
# tensor can be viewed as a cuda tensor zero-copy (_flagos_to_cuda_view) and
# handed to NCCL/FlagCX-cuda directly. This is the property that makes the
# CPU-torch + external libtorch_cuda scheme work. A hipified torch (hygon)
# qualifies too: its HIP kernels register under the CUDA dispatch key and its
# "NCCL" backend is RCCL, so the same row shape applies.
# ---------------------------------------------------------------------------


#   flagcx_native  True when this vendor's FlagCX adaptor consumes flagos
#                (privateuseone) tensors directly, so no view is needed for the
#                FlagCX path even when `view` is None. This capability is kept
#                separate from `direct` because a native fallback may still need
#                a typed vendor tensor.


class _VendorProfile:
    __slots__ = ("flagcx_dev", "view", "native", "flagcx_native", "direct")

    def __init__(self, flagcx_dev, view, native, flagcx_native=False, direct=False):
        self.flagcx_dev = flagcx_dev
        self.view = view  # attr name on torch_fl._C, or None
        self.native = native  # method name on ProcessGroupFlagOS, or None
        # flagcx_native allows the FlagCX adaptor to consume privateuseone
        # tensors directly. direct applies to all inner backend paths and is
        # used only for vendors whose backend has been measured to support it.
        self.flagcx_native = flagcx_native
        self.direct = direct


# NOTE: `view`/`native` are looked up lazily so importing this module never
# requires torch_fl._C or a specific backend to be present.
_VENDOR_PROFILES = {
    # CUDA-ABI vendors: flagos == cuda alias, native fallback is NCCL.
    "nvidia": _VendorProfile("cuda", "_flagos_to_cuda_view", "_try_build_nccl"),
    "metax": _VendorProfile("cuda", "_flagos_to_cuda_view", "_try_build_nccl"),
    "iluvatar": _VendorProfile("cuda", "_flagos_to_cuda_view", "_try_build_nccl"),
    "kunlunxin": _VendorProfile("cuda", "_flagos_to_cuda_view", "_try_build_nccl"),
    "du": _VendorProfile("cuda", "_flagos_to_cuda_view", "_try_build_nccl"),
    # T-Head PPU (Zhenwu ZW810E). FlagGems reports vendor "thead" when PPU_SDK is
    # set, while torch_fl sets GEMS_VENDOR=nvidia; list it so either value routes
    # the same instead of going through the unknown-vendor warning. PPU_SDK ships
    # a vendor-adapted libnccl.so.2 and its torch wheel is a real CUDA build, so
    # ProcessGroupNCCL works on the zero-copy cuda view unchanged.
    "thead": _VendorProfile("cuda", "_flagos_to_cuda_view", "_try_build_nccl"),
    # Hygon DCU (DTK). torch is a hipified CUDA build, so flagos IS a cuda alias
    # and the zero-copy view applies unchanged. Its ProcessGroupNCCL is RCCL
    # under the hood (dist.is_nccl_available() True, torch.cuda.nccl.version()
    # -> (2, 22, 3)); measured working for all_reduce / broadcast / all_gather /
    # all_gather_into_tensor / reduce_scatter_tensor and DDP on 2 cards.
    "hygon": _VendorProfile("cuda", "_flagos_to_cuda_view", "_try_build_nccl"),
    # Ascend: flagos is NOT a cuda alias; native fallback is HCCL. No flagos->npu
    # view exists, so the HCCL fallback stays unreachable, but FlagCX's cann
    # adaptor takes flagos tensors as-is (it only needs data_ptr(), and flagos
    # memory here is plain aclrtMalloc device memory) -- measured working for
    # allreduce/broadcast/all_gather/reduce_scatter/barrier on 2x910.
    "ascend": _VendorProfile("cann", None, "_try_build_hccl", flagcx_native=True),
    # Enflame GCU: the torch-fl-compatible FlagCX build registers for device
    # "flagos" and consumes privateuseone tensors directly. There is no native
    # fallback: ECCL is reached only through FlagCX, with no separate
    # ProcessGroup. Needs the device guard below (GCU streams/pointers are
    # device-scoped).
    "enflame": _VendorProfile("flagos", None, None, direct=True),
    # MUSA: FlagCX preferred (identity view lets FlagCX's MUSA adaptor receive
    # privateuseone tensors directly), with MCCL native fallback.
    "musa": _VendorProfile("musa", "_flagos_identity_view", "_try_build_mccl"),
    # Cambricon: FlagCX only (no cuda alias, no native fallback wired).
    "cambricon": _VendorProfile("mlu", None, None),
}

_DEFAULT_VENDOR = "nvidia"


def _get_profile(vendor: str) -> "_VendorProfile":
    prof = _VENDOR_PROFILES.get(vendor)
    if prof is None:
        warnings.warn(
            f"[ProcessGroupFlagOS] unknown GEMS_VENDOR={vendor!r}; assuming a "
            f"CUDA-ABI vendor (flagcx devName='cuda', NCCL fallback). Add a "
            f"_VENDOR_PROFILES entry if this is wrong."
        )
        prof = _VENDOR_PROFILES[_DEFAULT_VENDOR]
    return prof


def _configure_flagcx_torch_backend(vendor: str) -> None:
    """Select torch-fl's FlagCX integration before importing the plugin."""
    if vendor == "enflame":
        # The torch-fl-compatible FlagCX build uses its C ABI and does not
        # import or link torch_gcu. Preserve an explicit user selection.
        os.environ.setdefault("FLAGCX_TORCH_BACKEND", "flagos")


# ---------------------------------------------------------------------------
# Tensor view helpers
# ---------------------------------------------------------------------------


def _to_comm(t: torch.Tensor, view_fn) -> torch.Tensor:
    """Convert a single tensor for the comm backend if it is on flagos.

    view_fn is None when the inner backend consumes flagos tensors directly
    (see _VendorProfile.flagcx_native), in which case there is nothing to do.
    """
    if view_fn is not None and t.device.type in ("privateuseone", "flagos"):
        return view_fn(t)
    return t


def _tl(tensors, view_fn):
    """Convert a list of tensors."""
    return [_to_comm(t, view_fn) for t in tensors]


def _tll(tensor_lists, view_fn):
    """Convert a list-of-lists of tensors (used by allgather / reduce_scatter)."""
    return [_tl(tl, view_fn) for tl in tensor_lists]


# ---------------------------------------------------------------------------
# GCU device guard decorator
# ---------------------------------------------------------------------------


def _gcu_device_guard(func):
    """Decorator that sets GCU current device to match the operand before calling func.

    GCU streams and pointers are device-scoped. The communicator must run on the
    same device as the tensor, or operations silently no-op or produce zeros.
    FlagCX may cache streams by device index, so an incorrect first binding can
    permanently poison the communicator.

    Applied to all collective methods when GEMS_VENDOR=enflame. No-op otherwise.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        vendor = os.environ.get("GEMS_VENDOR", _DEFAULT_VENDOR)
        if vendor != "enflame":
            return func(self, *args, **kwargs)

        # Extract device index from the first tensor argument. Tensor-less
        # operations such as barrier reuse the device established by an earlier
        # tensor collective, then fall back to BarrierOptions.device_ids/device.
        device_id = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device_id = arg.device.index
                break
            elif isinstance(arg, list) and arg and isinstance(arg[0], torch.Tensor):
                device_id = arg[0].device.index
                break
            elif isinstance(arg, list) and arg and isinstance(arg[0], list):
                # List[List[Tensor]] case (allgather output, reduce_scatter input)
                if arg[0] and isinstance(arg[0][0], torch.Tensor):
                    device_id = arg[0][0].device.index
                    break

        if device_id is None:
            device_id = getattr(self, "_gcu_device_id", None)
        if device_id is None:
            for arg in args:
                option_device = getattr(arg, "device", None)
                option_index = getattr(option_device, "index", None)
                if option_index is not None:
                    device_id = option_index
                    break
                option_devices = getattr(arg, "device_ids", None)
                if option_devices:
                    device_id = int(option_devices[0])
                    break
        if device_id is None:
            import torch_fl._C as _C

            device_id = _C._get_device()
        else:
            self._gcu_device_id = device_id

        import torch_fl._C as _C

        prev = _C._get_device()
        _C._set_device(device_id)
        try:
            return func(self, *args, **kwargs)
        finally:
            _C._set_device(prev)

    return wrapper


# ---------------------------------------------------------------------------
# ProcessGroupFlagOS
# ---------------------------------------------------------------------------


class ProcessGroupFlagOS(dist.ProcessGroup):
    """ProcessGroup backend for flagos (PrivateUse1).

    Wraps an underlying NCCL or FlagCX ProcessGroup.  Each collective virtual
    method converts privateuseone tensors to the appropriate backend view before
    delegating; the inner backend's Work is returned directly.

    Instantiated by the ``creator_fn`` registered via
    ``dist.Backend.register_backend``.  Do not instantiate directly.
    """

    def __init__(self, store, rank: int, world_size: int, timeout=None):
        super().__init__(rank, world_size)
        self._store = store
        self._timeout = timeout
        self._inner_backend = None
        self._view_fn = self._build_inner(store, rank, world_size, timeout)
        self._register_inner_backend()

    def _register_inner_backend(self) -> None:
        """Expose the inner backend to ProcessGroup's device->backend map.

        Overriding the collective virtuals is enough for plain c10d calls, but
        not for anything that reaches for the *group identity*.
        ``ProcessGroup::setGroupName`` forwards to the registered backends, so
        with none registered the name is never stored and ``pg.group_name``
        raises "ProcessGroup name not set". DeviceMesh reads exactly that
        property, which put DTensor -- and therefore FSDP2 (``fully_shard``) and
        anything else mesh-based -- out of reach.

        Registering under privateuseone with BackendType.CUSTOM gives the name
        somewhere to live. It does not divert the collectives: the dispatcher
        keeps calling this class's overrides, so flagos tensors still go through
        the vendor view conversion before touching the inner backend.

        Best-effort: an inner backend that is not a c10d ``Backend`` (some
        FlagCX builds return a bare ProcessGroup) simply leaves the pre-existing
        behaviour in place rather than failing process-group construction.
        """
        inner = getattr(self, "_inner", None)
        if inner is None:
            return
        try:
            device = torch.device("privateuseone", max(torch.cuda.current_device(), 0))
            self._register_backend(device, dist.ProcessGroup.BackendType.CUSTOM, inner)
        except Exception:  # noqa: BLE001  - never block PG creation on this
            pass

    def _build_inner(self, store, rank, world_size, timeout):
        """Create the inner backend and return the view-conversion function.

        Table-driven by GEMS_VENDOR (see _VENDOR_PROFILES). For every vendor the
        priority is: FlagCX (heterogeneous unified comm) first, then the vendor's
        native backend (NCCL / HCCL). The returned callable maps a flagos
        (privateuseone) tensor to the physical device view the chosen inner
        backend expects; it is the identity for tensors already off flagos.
        """
        vendor = os.environ.get("GEMS_VENDOR", _DEFAULT_VENDOR)
        prof = _get_profile(vendor)

        # --- Try FlagCX first (heterogeneous unified comm) ---
        # FlagCX self-registers the "flagcx" backend for prof.flagcx_dev on
        # ``import flagcx`` (Backend.register_backend, extended_api=True). Its
        # ProcessGroup is created via createFlagcxBackend(opts, extra), NOT a
        # plain (store, rank, world_size) ctor, so we build the
        # _DistributedBackendOptions the same way c10d does.
        if self._try_build_flagcx(store, rank, world_size, timeout):
            self._inner_backend = "flagcx"
            return self._resolve_view(prof, vendor, backend="flagcx")

        # --- Native vendor backend fallback (NCCL / HCCL) ---
        if prof.native is not None:
            native_fn = getattr(self, prof.native)
            if native_fn(store, rank, world_size, timeout):
                self._inner_backend = f"native:{prof.native}"
                return self._resolve_view(prof, vendor, backend="native")

        raise RuntimeError(
            f"ProcessGroupFlagOS: no suitable inner backend for "
            f"GEMS_VENDOR={vendor!r}. Install/import flagcx (heterogeneous), or "
            f"provide the vendor-native backend "
            f"({'none wired' if prof.native is None else prof.native}). For a "
            f"CPU-only torch wheel + external libtorch_cuda, build the "
            f"_flagos_nccl extension (torch_fl/comm/_nccl_ext/build.py)."
        )

    def _resolve_view(self, prof, vendor, backend):
        """Return the flagos->comm-device view callable for this profile.

        cuda-alias vendors (view set) return the zero-copy _flagos_to_cuda_view.
        Vendors marked direct=True need no conversion: the inner backend reads the
        privateuseone tensor itself, so return the identity. Any other vendor
        without a view has no safe zero-copy reinterpretation implemented yet;
        using such a backend would pass a raw flagos tensor to a comm lib that
        cannot handle it, so fail loudly with a pointer to the missing helper
        rather than corrupt memory.
        """
        if prof.view is None:
            if backend == "flagcx" and prof.flagcx_native:
                # FlagCX's adaptor for this vendor takes flagos tensors directly.
                return None
            if prof.direct:
                return lambda t: t
            raise NotImplementedError(
                f"[ProcessGroupFlagOS] GEMS_VENDOR={vendor!r} selected inner "
                f"backend {backend!r}, but no flagos->device view is implemented "
                f"for it (flagos tensors are not a zero-copy alias of "
                f"'{prof.flagcx_dev}'). Implement the corresponding "
                f"_flagos_to_*_view in torch_fl/csrc/module.cc, or use a FlagCX "
                f"adaptor that consumes privateuseone tensors natively."
            )
        import torch_fl._C as _C

        view_fn = getattr(_C, prof.view, None)
        if view_fn is None:
            raise RuntimeError(
                f"[ProcessGroupFlagOS] torch_fl._C.{prof.view} not found "
                f"(required for GEMS_VENDOR={vendor!r})."
            )
        return view_fn

    # ------------------------------------------------------------------
    # Native vendor backends (used when FlagCX is unavailable)
    # ------------------------------------------------------------------

    def _try_build_nccl(self, store, rank, world_size, timeout) -> bool:
        """Build a ProcessGroupNCCL (native binding, else _flagos_nccl ext).

        Returns True and sets self._inner on success. Covers all CUDA-ABI
        vendors (nvidia/metax/iluvatar/kunlunxin/du/thead).
        """
        nccl_cls = getattr(torch.distributed, "ProcessGroupNCCL", None)
        if nccl_cls is not None:
            opts = nccl_cls.Options()
            if timeout is not None:
                opts._timeout = timeout
            self._inner = nccl_cls(store, rank, world_size, opts)
            return True

        # A CPU-only torch wheel does not expose ProcessGroupNCCL (built without
        # USE_C10D_NCCL), but an externally preloaded libtorch_cuda.so still
        # carries the full NCCL backend. The _flagos_nccl extension constructs
        # one and returns it as a c10d.Backend (see torch_fl/comm/_nccl_ext/).
        try:
            from torch_fl.comm._nccl_ext import _flagos_nccl
        except ImportError:
            try:
                import _flagos_nccl  # loose build layout
            except ImportError:
                return False
        timeout_ms = 0
        if timeout is not None:
            timeout_ms = (
                int(timeout.total_seconds() * 1000)
                if hasattr(timeout, "total_seconds")
                else int(timeout)
            )
        self._inner = _flagos_nccl.make_nccl_backend(
            store, rank, world_size, timeout_ms, False
        )
        return True

    def _try_build_hccl(self, store, rank, world_size, timeout) -> bool:
        """Build a ProcessGroupHCCL via torch_npu (Ascend native fallback).

        Returns True and sets self._inner on success, False if torch_npu / HCCL
        is unavailable. Note: the flagos->npu view is not implemented, so
        _resolve_view will still reject this path until that helper lands; the
        FlagCX(cann) path is the supported route on Ascend.
        """
        try:
            import torch_npu.distributed  # noqa: F401
        except ImportError:
            warnings.warn("[ProcessGroupFlagOS] torch_npu not found; cannot use HCCL.")
            return False
        hccl_cls = getattr(torch.distributed, "ProcessGroupHCCL", None) or getattr(
            torch_npu.distributed, "ProcessGroupHCCL", None
        )
        if hccl_cls is None:
            return False
        self._inner = hccl_cls(store, rank, world_size)
        return True

    def _try_build_mccl(self, store, rank, world_size, timeout) -> bool:
        """Build a ProcessGroupMCCL via torch_musa (MUSA native fallback).

        Returns True and sets self._inner on success, False if torch_musa / MCCL
        is unavailable. MCCL is Moore Threads' collective communication library,
        analogous to NCCL for CUDA. Combined with the identity view, this provides
        a pure-MUSA distributed path without requiring FlagCX.
        """
        try:
            import torch_musa.distributed  # noqa: F401
        except ImportError:
            return False
        mccl_cls = getattr(torch.distributed, "ProcessGroupMCCL", None) or getattr(
            torch_musa.distributed, "ProcessGroupMCCL", None
        )
        if mccl_cls is None:
            return False
        self._inner = mccl_cls(store, rank, world_size)
        return True

    def _try_build_flagcx(self, store, rank, world_size, timeout) -> bool:
        """Instantiate a FlagCX inner backend if flagcx is importable.

        Returns True and sets ``self._inner`` on success, False if flagcx is
        unavailable. Any hard failure is surfaced as a
        warning and treated as unavailable so we fall through to NCCL/HCCL.

        Two creator signatures exist upstream. FlagCX only compiles the
        extended_api form for the NVIDIA and MetaX adaptors (see the
        ``#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&
        defined(TORCH_VER_GE_250)`` guards in backend_flagcx.cpp); every other
        adaptor -- ppu, du, kunlunxin, ascend, enflame, ... -- exports the plain
        c10d form instead:

            extended: createFlagcxBackend(_DistributedBackendOptions, Options)
            plain:    createFlagcxBackend(store, rank, size, timeout)

        We try extended first and fall back to plain, otherwise those adaptors
        raise "incompatible function arguments" and silently degrade to NCCL.
        """
        vendor = os.environ.get("GEMS_VENDOR", _DEFAULT_VENDOR)
        _configure_flagcx_torch_backend(vendor)

        try:
            import flagcx  # noqa: F401 — self-registers "flagcx" backend
        except ImportError:
            return False

        # createFlagcxBackend is the extended_api creator exposed by flagcx._C.
        creator = getattr(flagcx, "createFlagcxBackend", None)
        if creator is None:
            # older builds only expose the class; skip and fall back
            warnings.warn(
                "[ProcessGroupFlagOS] flagcx present but "
                "createFlagcxBackend missing; falling back."
            )
            return False

        try:
            from torch._C._distributed_c10d import _DistributedBackendOptions

            opts = _DistributedBackendOptions()
            opts.store = store
            opts.group_rank = rank
            opts.group_size = world_size
            try:
                opts.group_id = self.group_name or ""
            except Exception:
                opts.group_id = ""
            opts.global_ranks_in_group = list(range(world_size))
            if timeout is not None:
                opts.timeout = timeout

            # extra Options: enable_tuner / tune_group_idx (see backend_flagcx.cpp).
            # ProcessGroupFlagCX lives on the flagcx module (flagcx._C), not on
            # torch.distributed.
            pg_cls = getattr(flagcx, "ProcessGroupFlagCX", None) or getattr(
                torch.distributed, "ProcessGroupFlagCX", None
            )
            extra_cls = getattr(pg_cls, "Options", None)
            extra = extra_cls() if extra_cls is not None else None

            self._inner = creator(opts, extra) if extra is not None else creator(opts)
            return True
        except TypeError:
            # Non-NVIDIA/MetaX adaptor: plain (store, rank, size, timeout) form.
            pass
        except Exception as e:
            warnings.warn(
                f"[ProcessGroupFlagOS] FlagCX init failed ({e}); "
                f"falling back to vendor-native backend."
            )
            return False

        try:
            # timeout must be a duration; c10d hands us a datetime.timedelta,
            # which pybind converts to std::chrono automatically. Older flagcx
            # builds ignore the value but still require the argument.
            self._inner = creator(store, rank, world_size, timeout)
            return True
        except Exception as e:
            warnings.warn(
                f"[ProcessGroupFlagOS] FlagCX init failed for both the "
                f"extended_api and plain createFlagcxBackend signatures "
                f"({e}); falling back to vendor-native backend."
            )
            return False

    # ------------------------------------------------------------------
    # Collective virtuals
    #
    # Convention:
    #   _tl(lst)  → convert List[Tensor]
    #   _tll(lst) → convert List[List[Tensor]]
    # Each method converts inputs, delegates to self._inner, returns Work.
    # ------------------------------------------------------------------

    # NOTE: opts default to None (constructed lazily). Options types are not all
    # exposed on ``torch.distributed`` (e.g. AllgatherOptions lives only in
    # ``torch._C._distributed_c10d``), and mutable defaults evaluated at def-time
    # would be shared singletons. c10d always passes opts explicitly when calling
    # these virtuals, so the fallback construction is just a safety net.

    @_gcu_device_guard
    def allreduce(self, tensors, opts=None):
        if opts is None:
            opts = _c10d.AllreduceOptions()
        return self._inner.allreduce(_tl(tensors, self._view_fn), opts)

    @_gcu_device_guard
    def allreduce_coalesced(self, tensors, opts=None):
        if opts is None:
            opts = _c10d.AllreduceCoalescedOptions()
        return self._inner.allreduce_coalesced(_tl(tensors, self._view_fn), opts)

    @_gcu_device_guard
    def allgather(self, output_tensors, input_tensors, opts=None):
        # output_tensors: List[List[Tensor]], input_tensors: List[Tensor]
        if opts is None:
            opts = _c10d.AllgatherOptions()
        return self._inner.allgather(
            _tll(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    @_gcu_device_guard
    def allgather_coalesced(self, output_tensors, input_tensors, opts=None):
        if opts is None:
            opts = _c10d.AllgatherOptions()
        return self._inner.allgather_coalesced(
            _tll(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    @_gcu_device_guard
    def allgather_into_tensor_coalesced(self, output_tensors, input_tensors, opts=None):
        if opts is None:
            opts = _c10d.AllgatherOptions()
        return self._inner.allgather_into_tensor_coalesced(
            _tl(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    @_gcu_device_guard
    def _allgather_base(self, output_tensor, input_tensor, opts=None):
        # Single-tensor allgather, backing the functional
        # dist.all_gather_into_tensor. A distinct virtual from allgather(): if it
        # is not overridden, ProcessGroup's C++ base resolves a Backend for the
        # tensor's device and raises "No backend type associated with device type
        # flagos". FSDP / ZeRO go through this path.
        if opts is None:
            opts = _c10d.AllgatherOptions()
        return self._inner._allgather_base(
            _to_comm(output_tensor, self._view_fn),
            _to_comm(input_tensor, self._view_fn),
            opts,
        )

    @_gcu_device_guard
    def broadcast(self, tensors, opts=None):
        if opts is None:
            opts = _c10d.BroadcastOptions()
        return self._inner.broadcast(_tl(tensors, self._view_fn), opts)

    @_gcu_device_guard
    def reduce(self, tensors, opts=None):
        if opts is None:
            opts = _c10d.ReduceOptions()
        return self._inner.reduce(_tl(tensors, self._view_fn), opts)

    @_gcu_device_guard
    def reduce_scatter(self, output_tensors, input_tensors, opts=None):
        # output: List[Tensor], input: List[List[Tensor]]
        if opts is None:
            opts = _c10d.ReduceScatterOptions()
        return self._inner.reduce_scatter(
            _tl(output_tensors, self._view_fn),
            _tll(input_tensors, self._view_fn),
            opts,
        )

    @_gcu_device_guard
    def reduce_scatter_tensor_coalesced(self, output_tensors, input_tensors, opts=None):
        if opts is None:
            opts = _c10d.ReduceScatterOptions()
        return self._inner.reduce_scatter_tensor_coalesced(
            _tl(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    @_gcu_device_guard
    def _reduce_scatter_base(self, output_tensor, input_tensor, opts=None):
        # Single-tensor reduce_scatter, backing the functional
        # dist.reduce_scatter_tensor; same unoverridden-virtual trap as
        # _allgather_base above. Hot path for FSDP gradient reduction.
        if opts is None:
            opts = _c10d.ReduceScatterOptions()
        return self._inner._reduce_scatter_base(
            _to_comm(output_tensor, self._view_fn),
            _to_comm(input_tensor, self._view_fn),
            opts,
        )

    @_gcu_device_guard
    def alltoall(self, output_tensors, input_tensors, opts=None):
        if opts is None:
            opts = _c10d.AllToAllOptions()
        return self._inner.alltoall(
            _tl(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    def alltoall_base(
        self,
        output_tensor,
        input_tensor,
        output_split_sizes,
        input_split_sizes,
        opts=None,
    ):
        if opts is None:
            opts = _c10d.AllToAllOptions()
        return self._inner.alltoall_base(
            _to_comm(output_tensor, self._view_fn),
            _to_comm(input_tensor, self._view_fn),
            output_split_sizes,
            input_split_sizes,
            opts,
        )

    @_gcu_device_guard
    def gather(self, output_tensors, input_tensors, opts=None):
        # output: List[List[Tensor]], input: List[Tensor]
        if opts is None:
            opts = _c10d.GatherOptions()
        return self._inner.gather(
            _tll(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    @_gcu_device_guard
    def scatter(self, output_tensors, input_tensors, opts=None):
        # output: List[Tensor], input: List[List[Tensor]]
        if opts is None:
            opts = _c10d.ScatterOptions()
        return self._inner.scatter(
            _tl(output_tensors, self._view_fn),
            _tll(input_tensors, self._view_fn),
            opts,
        )

    @_gcu_device_guard
    def send(self, tensors, dst_rank: int, tag: int):
        return self._inner.send(_tl(tensors, self._view_fn), dst_rank, tag)

    @_gcu_device_guard
    def recv(self, tensors, src_rank: int, tag: int):
        return self._inner.recv(_tl(tensors, self._view_fn), src_rank, tag)

    @_gcu_device_guard
    def recv_anysource(self, tensors, tag: int):
        return self._inner.recv_anysource(_tl(tensors, self._view_fn), tag)

    @_gcu_device_guard
    def barrier(self, opts=None):
        # barrier carries no tensors; delegate directly
        if opts is None:
            opts = _c10d.BarrierOptions()
        return self._inner.barrier(opts)

    @_gcu_device_guard
    def monitored_barrier(self, opts=None, wait_all_ranks=False):
        if opts is None:
            opts = _c10d.BarrierOptions()
        return self._inner.monitored_barrier(opts, wait_all_ranks)


# ---------------------------------------------------------------------------
# Backend creator function + public registration helper
# ---------------------------------------------------------------------------


def _create_flagos_pg(store, rank, world_size, timeout):
    """Creator function called by torch.distributed._new_process_group_helper."""
    return ProcessGroupFlagOS(store, rank, world_size, timeout=timeout)


def register_flagos_backend() -> None:
    """Register the ``"flagos"`` backend and set it as default for privateuseone.

    Called once at ``import torch_fl``.  Subsequent calls are no-ops.
    """
    if "flagos" in dist.Backend.backend_list:
        return  # already registered

    dist.Backend.register_backend(
        "flagos",
        _create_flagos_pg,
        extended_api=False,
        devices=["privateuseone"],
    )

    # Make `init_process_group(device_id=torch.device("privateuseone:0"))`
    # auto-select "flagos" without the user specifying a backend string.
    dist.Backend.default_device_backend_map.setdefault("privateuseone", "flagos")
