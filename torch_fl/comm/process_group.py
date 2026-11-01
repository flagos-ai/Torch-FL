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
native backend (NCCL for CUDA-ABI vendors, HCCL for Ascend, MCCL for MUSA), and
finally a host-staged gloo backend, which needs no vendor library at all but pays
a device->host->device copy per collective (``_try_build_staged_gloo``). Set
``FLAGOS_DIST_STAGED_GLOO=0`` to decline that last tier and fail loudly instead.

Gloo requests
    ``"flagos"`` has to be asked for by name, and not every caller does.
    torch.distributed picks gloo for any device type it does not recognise, so
    ``init_process_group(backend="gloo")`` -- what torchrun and the
    transformers FSDP2 tests build on a flagos host -- lands on a
    ProcessGroupGloo, whose C++ collectives reject flagos tensors outright and
    cannot be extended from Python. ``redirect_gloo_requests`` therefore
    interposes on ``init_process_group`` / ``new_group`` and substitutes the
    flagos backend for a gloo request, but only when the process's accelerator
    is the flagos device, i.e. when gloo has nothing else to fall back on. Set
    ``FLAGOS_DIST_REDIRECT_GLOO=0`` to keep the requested backend.

View conversion
    flagos tensors on CUDA-ABI vendors
    (nvidia/metax/iluvatar/du/thead/hygon)
    share physical GPU memory with CUDA, so they are reinterpreted as CUDA
    tensors via a zero-copy shared-storage view (``_C._flagos_to_cuda_view``)
    before being handed to NCCL / FlagCX-cuda. The underlying buffer is the
    same physical memory, so the backend's write-back is visible to the flagos
    side immediately. Vendors whose flagos tensor is NOT a cuda alias
    (ascend/musa/cambricon, and the de-boxed native kunlunxin backend) do not
    take the cuda view -- musa and kunlunxin use ``_flagos_identity_view``, the
    rest have ``view=None`` -- and currently require the FlagCX path; see
    ``_resolve_view``.

Device-scoped vendors
    Enflame GCU and TsingMicro streams and pointers are device-scoped. The
    communicator must run on the same device as the tensor, or operations
    silently no-op. Every collective method for those vendors applies a device
    guard that forces the current device to match the operand device before
    dispatch.
"""

import functools
import os
import warnings

import torch
import torch.distributed as dist
from torch._C import _distributed_c10d as _c10d

from torch_fl import _env


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
    # TsingMicro (TX8 / Kuiper): flagos tensors live in txMalloc'd XPU memory,
    # so they are NOT a cuda alias -- the cuda view would reinterpret XPU storage
    # as CUDA, a correctness break -- and no vendor collective backend is wired
    # to fall back to. Shaped exactly like enflame: the FlagCX build selected by
    # FLAGCX_TORCH_BACKEND=flagos registers for device "flagos" and consumes
    # privateuseone tensors directly (its tsmicro adaptor drives tx_runtime on
    # the tensor's own data_ptr), so no view is needed, direct=True, and without
    # FlagCX the path falls through to host-staged gloo. Needs the device guard
    # below for the same reason as GCU.
    "tsingmicro": _VendorProfile("flagos", None, None, direct=True),
    # MUSA: FlagCX preferred (identity view lets FlagCX's MUSA adaptor receive
    # privateuseone tensors directly), with MCCL native fallback.
    "musa": _VendorProfile("musa", "_flagos_identity_view", "_try_build_mccl"),
    # FlagGems 5.x calls the same Moore Threads vendor "mthreads".
    "mthreads": _VendorProfile("musa", "_flagos_identity_view", "_try_build_mccl"),
    # Kunlunxin (Kunlun P800 XPU), de-boxed from the CUDA ABI to a native
    # backend: its flagos tensors live in XPU memory and are NOT a cuda alias,
    # so ``_flagos_to_cuda_view`` would reinterpret XPU storage as CUDA -- a
    # correctness break -- and NCCL is not its collective. The identity view
    # hands the privateuseone tensor to FlagCX's kunlunxin adaptor unchanged,
    # exactly as ``musa`` does. Native collective builder not yet provided;
    # flagcx is the route (no ``_try_build_*`` for kunlunxin exists), so without
    # FlagCX the path falls through to host-staged gloo.
    "kunlunxin": _VendorProfile("kunlunxin", "_flagos_identity_view", None),
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


def is_cuda_alias_vendor(vendor: str) -> bool:
    """True when this vendor's flagos tensors are a zero-copy CUDA alias.

    The property is already recorded per vendor by ``_VENDOR_PROFILES``: a row
    whose ``view`` is ``_flagos_to_cuda_view`` shares physical GPU memory with
    CUDA, which is exactly what lets a flagos tensor be reinterpreted as a cuda
    one without copying. Exposed here so other compatibility layers that need
    that same guarantee -- the Apex multi-tensor shim in
    ``torch_fl.accelerator.apex_compat`` -- gate on this table rather than
    keeping a second, drift-prone vendor list.

    Unknown vendors are rejected: ``_get_profile`` assumes CUDA-ABI for the comm
    path because that is the only useful default there, but silently handing a
    non-alias tensor to a foreign C++ extension would corrupt memory instead of
    failing, so the conservative answer is the right one here.
    """
    prof = _VENDOR_PROFILES.get(vendor)
    return prof is not None and prof.view == "_flagos_to_cuda_view"


# Vendors whose FlagCX plugin is built in the torch-fl ("flagos") mode. That
# mode is what makes the plugin register the "flagcx" backend for the flagos
# device type and consume privateuseone tensors directly, and it is also a
# *runtime* selector: flagcx's ``_backend_loader`` imports torch_fl for
# ``FLAGCX_TORCH_BACKEND=flagos`` and otherwise probes the vendor packages
# (torch_gcu, torch_txda), which on a flagos-only host either pulls in a second,
# foreign PrivateUse1 provider or fails outright. The two halves of the switch
# therefore have to agree, and this is the half torch_fl can supply.
_FLAGCX_FLAGOS_VENDORS = ("enflame", "tsingmicro")


def _configure_flagcx_torch_backend(vendor: str) -> None:
    """Select torch-fl's FlagCX integration before importing the plugin."""
    if vendor in _FLAGCX_FLAGOS_VENDORS:
        # Preserve an explicit user selection (set_foreign is setdefault).
        _env.set_foreign("FLAGCX_TORCH_BACKEND", "flagos")


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
# Host-staged gloo fallback
# ---------------------------------------------------------------------------

# Both names are the same device: "flagos" is the registered privateuse1 name,
# and torch.device("privateuseone") reports it too once the backend is renamed.
_HOST_STAGED_TYPES = ("privateuseone", "flagos")


def _stage_to_host(obj, staged):
    """Deep-copy every flagos tensor in ``obj`` to host memory.

    Host tensors, split-size lists and option objects are returned untouched.
    Each ``(device_tensor, host_tensor)`` pair is appended to ``staged`` so the
    result can be copied back once the collective has run; that covers output
    operands too, whose host shadow is the buffer gloo writes into.
    """
    if isinstance(obj, torch.Tensor):
        if obj.device.type in _HOST_STAGED_TYPES:
            host = obj.detach().to("cpu")
            staged.append((obj, host))
            return host
        return obj
    if isinstance(obj, list):
        return [_stage_to_host(item, staged) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_stage_to_host(item, staged) for item in obj)
    return obj


def _completed_work():
    """A Work object for a collective this process already finished itself.

    The staging copy-back happens before the caller sees the Work, so the Work
    only has to be waitable; c10d's future-backed Work is exactly that.
    """
    future = torch.futures.Future()
    future.set_result(None)
    return _c10d._create_work_from_future(future)


def _allgather_into_tensor_coalesced(group, output_tensors, input_tensors, opts):
    """gloo's ``_allgather_base`` repeated over the coalesced tensor lists."""
    work = None
    for output, input_ in zip(output_tensors, input_tensors):
        work = group._allgather_base(output, input_, opts)
    return work


def _reduce_scatter_tensor_coalesced(group, output_tensors, input_tensors, opts):
    """gloo's ``_reduce_scatter_base`` repeated over the coalesced tensor lists."""
    work = None
    for output, input_ in zip(output_tensors, input_tensors):
        work = group._reduce_scatter_base(output, input_, opts)
    return work


# gloo implements neither of these two virtuals. Left to ProcessGroup's C++
# base they resolve a backend for the operand's device and raise "No backend
# type associated with device type flagos" -- and they are the path the
# functional collectives (all_gather_into_tensor / reduce_scatter_tensor, and
# therefore DeviceMesh and FSDP2) actually take. Expressing them over gloo's
# single-tensor base calls is what keeps that path working without a vendor
# communicator.
_GLOO_SYNTHESISED_OPS = {
    "allgather_into_tensor_coalesced": _allgather_into_tensor_coalesced,
    "reduce_scatter_tensor_coalesced": _reduce_scatter_tensor_coalesced,
}


class _HostStagedGloo:
    """Run a real gloo group on flagos operands staged through host memory.

    Used as ``ProcessGroupFlagOS._inner`` on hosts where no vendor communicator
    exists. gloo is built into c10d but its collectives are C++ overrides that
    reject anything but cpu/cuda tensors, so a flagos operand has to arrive as
    host memory: every call whose arguments mention one is turned into "copy to
    host, run gloo, copy the result back", and every other call is forwarded
    verbatim so cpu tensors keep gloo's own behaviour unchanged.

    The copy-back is synchronous, before the Work is returned -- callers that
    read the operands after ``wait()`` therefore see the result, and the host
    shadow of an output operand is what gloo filled in.

    ``group`` is the real ProcessGroupGloo behind the staging. It is not a
    detail: it is the only object here that is a c10d ``Backend``, so it is
    what ``ProcessGroupFlagOS._register_inner_backend`` registers for group
    identity (name, rank, size); the facade itself cannot be registered.
    """

    def __init__(self, group):
        self.group = group

    @property
    def c10d_backend(self):
        """The real gloo group, i.e. the only c10d ``Backend`` in this facade."""
        return self.group

    def __getattr__(self, name):
        try:
            op = getattr(self.group, name)
        except AttributeError:
            synthesis = _GLOO_SYNTHESISED_OPS.get(name)
            if synthesis is None:
                raise
            # The synthesised helpers take the group as their first argument
            # because they are also readable on their own; bind it here so _run
            # only has to forward the caller's arguments.
            return functools.partial(
                self._run, functools.partial(synthesis, self.group)
            )
        if not callable(op):
            return op
        return functools.partial(self._run, op)

    def _run(self, op, *args, **kwargs):
        staged = []
        host_args = _stage_to_host(args, staged)
        host_kwargs = _stage_to_host(kwargs, staged)
        if not staged:
            return op(*args, **kwargs)
        op(*host_args, **host_kwargs).wait()
        for device_tensor, host_tensor in staged:
            device_tensor.copy_(host_tensor)
        return _completed_work()

    def __repr__(self):
        return f"_HostStagedGloo({self.group!r})"


# Warned about once per process, not once per group: a job that builds one
# group per rank pair would otherwise repeat the same notice forever.
_STAGED_GLOO_WARNED = False


def _warn_staged_gloo() -> None:
    """Report that collectives are being staged through host memory.

    Not an error -- the group works -- but it is a silent order-of-magnitude
    slowdown if it happens on a host that also has a vendor communicator that
    merely failed to load, so it must be visible in the log.
    """
    global _STAGED_GLOO_WARNED
    if _STAGED_GLOO_WARNED:
        return
    _STAGED_GLOO_WARNED = True
    warnings.warn(
        "[ProcessGroupFlagOS] no vendor communicator (FlagCX/NCCL/HCCL/MCCL) is "
        "available; falling back to host-staged gloo. Collectives still run, but "
        "every flagos operand is copied device->host->device. Set "
        "FLAGOS_DIST_STAGED_GLOO=0 to make this an error instead."
    )


# ---------------------------------------------------------------------------
# Device-scoped vendor guard decorator
# ---------------------------------------------------------------------------

# Vendors whose streams and pointers are device-scoped, so a collective has to
# run with the operand's device current. The hazard is structural, not vendor
# folklore: FlagCX creates its comm stream lazily on whatever device is current
# (``getStreamByIndex`` runs *before* ``initComm`` binds the tensor's device) and
# caches it for the life of the process, so a wrong first binding poisons the
# communicator -- operations then no-op or produce zeros. Enflame GCU and
# TsingMicro both behave this way. The remaining profiles either bind the device
# inside the backend or keep no per-device stream state to poison.
_DEVICE_SCOPED_VENDORS = ("enflame", "tsingmicro")


def _device_scoped_guard(func):
    """Decorator that sets the current device to match the operand before func.

    Applied to all collective methods for a vendor in _DEVICE_SCOPED_VENDORS.
    No-op otherwise.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        vendor = os.environ.get("GEMS_VENDOR", _DEFAULT_VENDOR)
        if vendor not in _DEVICE_SCOPED_VENDORS:
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
            device_id = getattr(self, "_comm_device_id", None)
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
            self._comm_device_id = device_id

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

    Wraps an underlying FlagCX, vendor-native or host-staged-gloo ProcessGroup.
    Each collective virtual method converts privateuseone tensors to the
    appropriate backend view before delegating; the inner backend's Work is
    returned directly.

    Instantiated by the ``creator_fn`` registered via
    ``dist.Backend.register_backend`` -- either by name, or for a plain gloo
    request through ``redirect_gloo_requests``. Do not instantiate directly.
    """

    def __init__(self, store, rank: int, world_size: int, timeout=None):
        super().__init__(rank, world_size)
        self._store = store
        self._timeout = timeout
        # Why the native NCCL fallback declined, filled in by _load_nccl_extension
        # so the "no suitable inner backend" error can carry the real cause
        # instead of leaving the user with a bare NoneType failure. None until
        # that path runs.
        self._nccl_skip_reason = None
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
        # _HostStagedGloo is a facade, not a Backend: what has to be registered
        # for the group identity to have somewhere to live is the gloo group it
        # wraps (see its c10d_backend property).
        inner = getattr(inner, "c10d_backend", inner)
        try:
            device = torch.device("privateuseone", max(torch.cuda.current_device(), 0))
            self._register_backend(device, dist.ProcessGroup.BackendType.CUSTOM, inner)
        except Exception:  # noqa: BLE001  - never block PG creation on this
            pass

    def _build_inner(self, store, rank, world_size, timeout):
        """Create the inner backend and return the view-conversion function.

        Table-driven by GEMS_VENDOR (see _VENDOR_PROFILES). For every vendor the
        priority is: FlagCX (heterogeneous unified comm) first, then the vendor's
        native backend (NCCL / HCCL / MCCL), then host-staged gloo. The returned
        callable maps a flagos (privateuseone) tensor to the physical device view
        the chosen inner backend expects; it is the identity for tensors already
        off flagos, and None for the staged tier, which converts operands itself.
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
            return self._resolve_view(prof, vendor, backend="flagcx")

        # --- Native vendor backend fallback (NCCL / HCCL) ---
        if prof.native is not None:
            native_fn = getattr(self, prof.native)
            if native_fn(store, rank, world_size, timeout):
                return self._resolve_view(prof, vendor, backend="native")

        # --- Host-staged gloo (last resort: needs no vendor library) ---
        if self._try_build_staged_gloo(store, rank, world_size, timeout):
            # No view conversion on this path: _HostStagedGloo stages the flagos
            # operands through host memory itself, so a device view would be both
            # unnecessary and wrong (gloo wants cpu tensors, not cuda ones).
            return None

        # getattr because the unit tests drive _build_inner on __new__ instances
        # that never ran __init__ (see tests/unit/test_vendor_routing.py).
        reason = getattr(self, "_nccl_skip_reason", None)
        staged_reason = getattr(self, "_staged_skip_reason", None)
        raise RuntimeError(
            f"ProcessGroupFlagOS: no suitable inner backend for "
            f"GEMS_VENDOR={vendor!r}. Install/import flagcx (heterogeneous), or "
            f"provide the vendor-native backend "
            f"({'none wired' if prof.native is None else prof.native}). For a "
            f"CPU-only torch wheel + external libtorch_cuda, build the "
            f"_flagos_nccl extension (torch_fl/comm/_nccl_ext/build.py)."
            + (f"\nNative NCCL fallback unavailable: {reason}." if reason else "")
            + (
                f"\nHost-staged gloo unavailable: {staged_reason}."
                if staged_reason
                else ""
            )
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
        vendors (nvidia/metax/iluvatar/du/thead).
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
        ext = self._load_nccl_extension()
        if ext is None:
            return False
        timeout_ms = 0
        if timeout is not None:
            timeout_ms = (
                int(timeout.total_seconds() * 1000)
                if hasattr(timeout, "total_seconds")
                else int(timeout)
            )
        self._inner = ext.make_nccl_backend(store, rank, world_size, timeout_ms, False)
        return True

    def _load_nccl_extension(self):
        """Return the _flagos_nccl extension module, or None when unavailable.

        Sets ``self._nccl_skip_reason`` when it returns None, so _build_inner can
        report the loader failure in the final diagnostic.

        The package attribute is a None sentinel rather than missing when the
        extension is not built, so the absence has to be tested explicitly:
        importing the sentinel succeeds, and the old code then called
        ``make_nccl_backend`` on it, failing as
        ``'NoneType' object has no attribute 'make_nccl_backend'`` while
        discarding the ImportError that explained why the extension was missing.
        See torch_fl/comm/_nccl_ext/__init__.py.
        """
        from torch_fl.comm import _nccl_ext

        ext = _nccl_ext.get_extension()
        if ext is not None:
            return ext

        try:
            import _flagos_nccl as ext  # loose build layout
        except ImportError as loose_exc:
            self._nccl_skip_reason = (
                f"_flagos_nccl is not built "
                f"(package import: {_nccl_ext.load_error()!r}; "
                f"loose module import: {loose_exc!r})"
            )
            return None
        return ext

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

    def _try_build_staged_gloo(self, store, rank, world_size, timeout) -> bool:
        """Build a gloo group and stage flagos operands through host memory.

        The last tier, and the only one that needs no vendor library: gloo ships
        inside c10d, so it is always present. That is what makes
        ``backend="flagos"`` work on a host with no FlagCX, no NCCL and no vendor
        torch -- e.g. a CPU-only wheel on a MUSA box -- and it is therefore also
        what the gloo redirect (``redirect_gloo_requests``) lands on.

        Returns True and sets ``self._inner`` to a ``_HostStagedGloo``. Declines
        when the switch is off, when gloo is absent, when no store was supplied
        (gloo rendezvouses through it) or when gloo refuses to build. Failures
        are recorded rather than raised: this tier runs last, so a hard error
        here would hide the diagnostic that explains why the tiers above it
        declined.
        """
        if not _env.flag("FLAGOS_DIST_STAGED_GLOO", True):
            self._staged_skip_reason = "FLAGOS_DIST_STAGED_GLOO=0"
            return False
        gloo_cls = getattr(torch.distributed, "ProcessGroupGloo", None)
        if gloo_cls is None:
            self._staged_skip_reason = "this torch was built without ProcessGroupGloo"
            return False
        if store is None:
            self._staged_skip_reason = "no store was supplied for the gloo rendezvous"
            return False
        try:
            group = (
                gloo_cls(store, rank, world_size)
                if timeout is None
                else gloo_cls(store, rank, world_size, timeout=timeout)
            )
        except Exception as exc:  # noqa: BLE001 - last tier: report, never raise
            # First line only: pybind's "incompatible function arguments" dump
            # runs for pages and would bury the diagnostic it is appended to.
            detail = str(exc).splitlines()[0][:160] if str(exc) else repr(exc)
            self._staged_skip_reason = f"ProcessGroupGloo construction failed: {detail}"
            return False
        self._inner = _HostStagedGloo(group)
        _warn_staged_gloo()
        return True

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

    @_device_scoped_guard
    def allreduce(self, tensors, opts=None):
        if opts is None:
            opts = _c10d.AllreduceOptions()
        return self._inner.allreduce(_tl(tensors, self._view_fn), opts)

    @_device_scoped_guard
    def allreduce_coalesced(self, tensors, opts=None):
        if opts is None:
            opts = _c10d.AllreduceCoalescedOptions()
        return self._inner.allreduce_coalesced(_tl(tensors, self._view_fn), opts)

    @_device_scoped_guard
    def allgather(self, output_tensors, input_tensors, opts=None):
        # output_tensors: List[List[Tensor]], input_tensors: List[Tensor]
        if opts is None:
            opts = _c10d.AllgatherOptions()
        return self._inner.allgather(
            _tll(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    @_device_scoped_guard
    def allgather_coalesced(self, output_tensors, input_tensors, opts=None):
        if opts is None:
            opts = _c10d.AllgatherOptions()
        return self._inner.allgather_coalesced(
            _tll(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    @_device_scoped_guard
    def allgather_into_tensor_coalesced(self, output_tensors, input_tensors, opts=None):
        if opts is None:
            opts = _c10d.AllgatherOptions()
        out = _tl(output_tensors, self._view_fn)
        inp = _tl(input_tensors, self._view_fn)
        if hasattr(self._inner, "allgather_into_tensor_coalesced"):
            return self._inner.allgather_into_tensor_coalesced(out, inp, opts)
        # Some inner backends (flagcx) only expose _allgather_base; synthesise the
        # coalesced alias over the single-tensor base calls, as _HostStagedGloo does.
        return _allgather_into_tensor_coalesced(self._inner, out, inp, opts)

    @_device_scoped_guard
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

    @_device_scoped_guard
    def broadcast(self, tensors, opts=None):
        if opts is None:
            opts = _c10d.BroadcastOptions()
        return self._inner.broadcast(_tl(tensors, self._view_fn), opts)

    @_device_scoped_guard
    def reduce(self, tensors, opts=None):
        if opts is None:
            opts = _c10d.ReduceOptions()
        return self._inner.reduce(_tl(tensors, self._view_fn), opts)

    @_device_scoped_guard
    def reduce_scatter(self, output_tensors, input_tensors, opts=None):
        # output: List[Tensor], input: List[List[Tensor]]
        if opts is None:
            opts = _c10d.ReduceScatterOptions()
        return self._inner.reduce_scatter(
            _tl(output_tensors, self._view_fn),
            _tll(input_tensors, self._view_fn),
            opts,
        )

    @_device_scoped_guard
    def reduce_scatter_tensor_coalesced(self, output_tensors, input_tensors, opts=None):
        if opts is None:
            opts = _c10d.ReduceScatterOptions()
        out = _tl(output_tensors, self._view_fn)
        inp = _tl(input_tensors, self._view_fn)
        if hasattr(self._inner, "reduce_scatter_tensor_coalesced"):
            return self._inner.reduce_scatter_tensor_coalesced(out, inp, opts)
        # flagcx only exposes _reduce_scatter_base; synthesise the coalesced alias
        # over the single-tensor base calls, as _HostStagedGloo does.
        return _reduce_scatter_tensor_coalesced(self._inner, out, inp, opts)

    @_device_scoped_guard
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

    @_device_scoped_guard
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

    @_device_scoped_guard
    def gather(self, output_tensors, input_tensors, opts=None):
        # output: List[List[Tensor]], input: List[Tensor]
        if opts is None:
            opts = _c10d.GatherOptions()
        return self._inner.gather(
            _tll(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    @_device_scoped_guard
    def scatter(self, output_tensors, input_tensors, opts=None):
        # output: List[Tensor], input: List[List[Tensor]]
        if opts is None:
            opts = _c10d.ScatterOptions()
        return self._inner.scatter(
            _tl(output_tensors, self._view_fn),
            _tll(input_tensors, self._view_fn),
            opts,
        )

    @_device_scoped_guard
    def send(self, tensors, dst_rank: int, tag: int):
        return self._inner.send(_tl(tensors, self._view_fn), dst_rank, tag)

    @_device_scoped_guard
    def recv(self, tensors, src_rank: int, tag: int):
        return self._inner.recv(_tl(tensors, self._view_fn), src_rank, tag)

    @_device_scoped_guard
    def recv_anysource(self, tensors, tag: int):
        return self._inner.recv_anysource(_tl(tensors, self._view_fn), tag)

    @_device_scoped_guard
    def barrier(self, opts=None):
        # barrier carries no tensors; delegate directly
        if opts is None:
            opts = _c10d.BarrierOptions()
        return self._inner.barrier(opts)

    @_device_scoped_guard
    def monitored_barrier(self, opts=None, wait_all_ranks=False):
        if opts is None:
            opts = _c10d.BarrierOptions()
        return self._inner.monitored_barrier(opts, wait_all_ranks)


# ---------------------------------------------------------------------------
# Gloo request redirection
# ---------------------------------------------------------------------------

_FLAGOS_BACKEND = "flagos"

# Marker attribute on the wrappers themselves, so a second import of torch_fl (or
# a reload) cannot stack a second layer of interposition.
_REDIRECT_MARKER = "_flagos_redirects_gloo"


def _gloo_requests_are_redirected() -> bool:
    """True when a gloo request on this host has nothing left to fall back on.

    gloo serves cpu and cuda tensors, so it is only the wrong backend when the
    process's accelerator *is* the flagos device -- exactly the case c10d cannot
    see, because it does not know the device type and maps it to gloo by default
    (``BackendConfig`` expands gloo over ``backend_capability["gloo"]``, and on
    these builds ``torch.device("cuda")`` is itself flagos, so the flagos device
    type ends up with a ProcessGroupGloo registered against it).

    Gated by ``FLAGOS_DIST_REDIRECT_GLOO`` so the requested backend can still be
    honoured verbatim while debugging a distributed problem.
    """
    if not _env.flag("FLAGOS_DIST_REDIRECT_GLOO", True):
        return False
    try:
        accelerator = torch._C._get_accelerator().type
    except Exception:  # noqa: BLE001 - no accelerator query: leave gloo alone
        return False
    return accelerator in _HOST_STAGED_TYPES


def _needs_gloo_redirect(value) -> bool:
    """True for a backend argument that names gloo and must be served by flagos.

    Both halves matter: the request has to be for gloo, and gloo has to be the
    wrong backend here (see ``_gloo_requests_are_redirected``).
    """
    if not _gloo_requests_are_redirected():
        return False
    return isinstance(value, str) and value.lower() == dist.Backend.GLOO


def _redirect_backend_argument(args, kwargs, position, keyword):
    """Rewrite a gloo backend argument to "flagos", wherever it was passed.

    Both spellings have to be handled: torchrun and the transformers FSDP2 tests
    pass ``backend=`` by keyword, while ``new_group`` is usually called
    positionally.
    """
    if len(args) > position:
        if _needs_gloo_redirect(args[position]):
            args = list(args)
            args[position] = _FLAGOS_BACKEND
            args = tuple(args)
    elif keyword in kwargs and _needs_gloo_redirect(kwargs[keyword]):
        kwargs = dict(kwargs)
        kwargs[keyword] = _FLAGOS_BACKEND
    return args, kwargs


def redirect_gloo_requests() -> None:
    """Interpose on ``init_process_group`` / ``new_group`` to serve gloo requests.

    ``torch.distributed`` sends every device type it does not recognise to gloo,
    so on a flagos host ``init_process_group(backend="gloo")`` -- what torchrun
    and the transformers FSDP2 tests build -- lands on a ProcessGroupGloo whose
    C++ collectives reject flagos tensors with "unsupported device type flagos"
    (issue #263).

    That cannot be repaired from the gloo side. gloo's own BackendConfig
    expansion is what registers the group for the flagos device type, its
    creator is asserted to return a ProcessGroupGloo, and ``c10d.Backend``
    cannot be subclassed from Python -- so there is no seam inside the gloo
    branch, and the request has to be answered with the flagos backend *before*
    the backend config is built.

    ``new_group`` is covered as well as ``init_process_group``: a group built
    with ``backend=None`` inherits the default group's backend (already
    "flagos" after a redirected init), but a caller that spells "gloo" out again
    would otherwise get a second, flagos-hostile group on the same tensors.

    Called once at ``import torch_fl``; subsequent calls are no-ops.
    """
    if getattr(dist.init_process_group, _REDIRECT_MARKER, False):
        return

    original_init_process_group = dist.init_process_group
    original_new_group = dist.new_group

    @functools.wraps(original_init_process_group)
    def init_process_group(*args, **kwargs):
        args, kwargs = _redirect_backend_argument(args, kwargs, 0, "backend")
        return original_init_process_group(*args, **kwargs)

    @functools.wraps(original_new_group)
    def new_group(*args, **kwargs):
        # new_group(ranks=None, timeout=..., backend=None, ...): backend is the
        # third positional argument.
        args, kwargs = _redirect_backend_argument(args, kwargs, 2, "backend")
        return original_new_group(*args, **kwargs)

    init_process_group._flagos_redirects_gloo = True
    new_group._flagos_redirects_gloo = True
    dist.init_process_group = init_process_group
    dist.new_group = new_group

    # Same two names on the defining module, so `from torch.distributed
    # .distributed_c10d import init_process_group` also picks up the wrapper.
    c10d_module = getattr(dist, "distributed_c10d", None)
    if c10d_module is not None:
        c10d_module.init_process_group = init_process_group
        c10d_module.new_group = new_group


# ---------------------------------------------------------------------------
# Backend creator function + public registration helper
# ---------------------------------------------------------------------------


def _create_flagos_pg(store, rank, world_size, timeout):
    """Creator function called by torch.distributed._new_process_group_helper."""
    return ProcessGroupFlagOS(store, rank, world_size, timeout=timeout)


def register_flagos_backend() -> None:
    """Register the ``"flagos"`` backend, set it as default, redirect gloo.

    Called once at ``import torch_fl``.  Subsequent calls are no-ops.
    """
    if "flagos" in dist.Backend.backend_list:
        return  # already registered

    # The device type every c10d lookup normalises to.  torch_fl renames
    # PrivateUse1 to "flagos", and even ``torch.device("privateuseone:0")``
    # resolves to type ``"flagos"`` before the map lookup, so the literal
    # ``"privateuseone"`` this registration used to carry was unreachable and
    # ``get_default_backend_for_device`` raised for the device the plugin
    # serves (#506).
    device_type = torch._C._get_privateuse1_backend_name()

    dist.Backend.register_backend(
        "flagos",
        _create_flagos_pg,
        extended_api=False,
        devices=[device_type],
    )

    # Make `init_process_group(device_id=torch.device(f"{device_type}:0"))`
    # auto-select "flagos" without the user specifying a backend string.
    dist.Backend.default_device_backend_map.setdefault(device_type, "flagos")

    # Serve plain gloo requests with this backend instead (see
    # redirect_gloo_requests). Separate try/except on purpose: the backend above
    # is registered either way, so a failure here must not be reported as a
    # failed registration.
    try:
        redirect_gloo_requests()
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"[torch_fl] Failed to redirect gloo requests: {e}")
