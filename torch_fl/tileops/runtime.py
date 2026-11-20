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

"""Runtime for the generated TileOPs routes.

Routes reach aten through the C++ dispatcher on ``Backend::kTileOps``
(``csrc/aten/generated/tileops_python_kernels.cc``), which calls back into
``torch_fl.tileops.generated.shims`` -- TileOPs ships no C++ API, so the kernels
stay in Python. This module is the runtime those shims land in: it owns the
instance cache and turns an aten call into TileOPs constructor arguments.

Dispatch decisions themselves are *not* made here. The dispatcher already
consults the conf files and ``FLAGOS_OP_<op>``, and logs via
``FLAGOS_LOG=dispatch``, so nothing in this module duplicates that. Only the
per-call "can TileOPs actually serve these arguments" test is local, since it
depends on the runtime dtype and shape.

TileOPs ops are stateful Python objects whose constructors commit shape/dtype, and
TileLang kernels need a device and stream their runtime can use. Both are handled here:

  - constructor args are derived from the aten call site per recipe, and instances
    are cached, because a cold constructor costs ~4 s (~54 ms warm) against a
    0.056 ms warm call;
  - a provider device adapter names the hardware TileLang should target and maps
    torch_fl storage and its current stream to the provider's runtime. CUDA-ABI
    providers use the zero-copy CUDA view; a native provider such as MUSA can
    consume the flagos tensor and must register its stream adapter explicitly.

Calls prefer the compiled kernel over the ``Op`` wrapper, which adds 0.05-0.09 ms
of size-independent Python overhead -- enough to erase the gain on every
memory-bound op. The shortcut is verified per instance rather than assumed (see
``_Entry``), since L2 sometimes reshapes before dispatching. Set
``FLAGOS_TILEOPS_USE_L2=1`` to stay on ``Op.__call__`` when debugging.

See ``docs/architecture/tileops/codegen-design.md``.
"""

from __future__ import annotations

import importlib
import inspect
import os
import threading
import types
import warnings
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.utils import _pytree

from torch_fl import _env
from torch_fl.tileops.spec import BINARY, REDUCE, SOFTMAX, UNARY

__all__ = [
    "enable_tileops_for_flagos",
    "is_tileops_available",
    "registered_ops",
    "build_impl",
    "resolve_impl",
    "sample_inputs",
    "warmup",
]

_instances: Dict[tuple, tuple] = {}
_provider: Optional[object] = None
_provider_failure: Optional[BaseException] = None
_provider_lock = threading.RLock()

_ADAPTERS: Dict[str, tuple] = {}
_ADAPTERS_LOCK = threading.RLock()
_CONFIGURED_TARGETS: set[str] = set()
_HAS_SELECT_TARGET: Optional[bool] = None

#: Ceiling on cached Op instances. The cache is keyed on ctor args, which
#: include shape, so a workload with varying shapes (dynamic batch, ragged
#: sequence lengths) would otherwise grow it without bound -- and each entry
#: pins a compiled TileLang kernel, so the leak is device memory rather than
#: just a dict. Beyond the cap, _get stops caching and rebuilds per call: slower
#: (see the cold-ctor cost in the module docstring), but bounded. Raise it with
#: FLAGOS_TILEOPS_CACHE_MAX if you have many static shapes and headroom.
_INSTANCE_CACHE_MAX = int(_env.value("FLAGOS_TILEOPS_CACHE_MAX", "512"))
_cache_full_warned = False


def _use_l2() -> bool:
    return _env.flag("FLAGOS_TILEOPS_USE_L2")


def _disable_frontend_cache() -> bool:
    """Neutralize TileLang's *frontend* disk cache, keeping the kernel cache.

    Ops whose ctor commits a size (the whole UNARY recipe) otherwise reuse the
    first-compiled artifact across shapes and fail with ``violates packed ABI
    constraint; expected: <N>, got: <first N>``.

    The cause is not the kernel disk cache. TileOPs builds these kernels via
    ``functools.lru_cache``-d factories whose inner ``@tilelang.jit`` closure
    captures ``N`` as a free variable, and the frontend cache key
    (``JITImpl._frontend_cache_key_data``) covers only the explicit args -- for
    ReluFwdOp it is ``((('npt_arg', 8), ('threads_arg', 256)), None)`` for every
    N. Instrumenting ``load_frontend_cached`` shows it returning HIT for a
    different N; the per-``JITImpl`` ``_kernel_cache`` objects are distinct, so
    the leak is purely through this on-disk layer. ``GemmOp`` is immune because
    it passes ``(m, n, k)`` explicitly rather than closing over them.

    Upstream tile-ai/tilelang#2358, fixed by #2363 ("Remove frontend disk
    cache") on 2026-06-09 -- one day after 0.1.11 shipped, hence our pin needs
    this. Dropping the same path here is that fix, and it lets the kernel disk
    cache keep working: measured on H800 with three ReluFwdOp shapes, second
    process 0.3 s vs 12.0 s under a blanket ``TILELANG_DISABLE_CACHE=1``, both
    numerically correct. Delete this function when the pin reaches >=0.1.12.
    """
    if os.environ.get("TILELANG_DISABLE_CACHE") == "1":
        return False  # blanket disable already in force; nothing to patch
    patched = False
    _noop_load = lambda *a, **kw: None  # noqa: E731
    for name in ("tilelang.cache", "tilelang.jit"):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for attr in ("load_frontend_cached", "store_frontend_cache"):
            if hasattr(mod, attr):
                setattr(mod, attr, _noop_load)
                patched = True
    if not patched:
        # Future TileLang without these symbols: the bug is fixed upstream, so
        # there is nothing to work around.
        pass
    return patched


# --------------------------------------------------------------------------- #
# availability gate
# --------------------------------------------------------------------------- #
def import_tileops():
    """Import ``FLAGOS_TILEOPS_PROVIDER`` (default ``tileops``), once.

    A missing optional package is remembered so an ordinary torch_fl import
    keeps working. Discovery happens before target detection, allowing a vendor
    distribution to own its detectors and kernel builders.
    """
    global _provider, _provider_failure
    with _provider_lock:
        if _provider is not None:
            return _provider
        if _provider_failure is not None:
            raise _provider_failure
        if _env.flag("FLAGOS_TILEOPS_DISABLE_ALL_CACHE"):
            _env.set_foreign("TILELANG_DISABLE_CACHE", "1")
        try:
            _provider = importlib.import_module(
                _env.value("FLAGOS_TILEOPS_PROVIDER", "tileops")
            )
        except ImportError as exc:
            _provider_failure = exc
            raise
        _disable_frontend_cache()
        return _provider


def _tileops_platform() -> str:
    """Hardware platform of the running torch_fl build."""
    from torch_fl import _platform

    accelerator = _platform.build_accelerator()
    if accelerator:
        return _platform._ACCELERATOR_TO_PLATFORM.get(accelerator, accelerator)
    return _platform.detect_platform()


def _provider_backend():
    """Import backend discovery without making it mandatory for every provider."""
    try:
        backend = importlib.import_module("tileops.backend")
    except ImportError:
        return None
    load = getattr(backend, "ensure_loaded", None)
    if callable(load):
        load()
    return backend


def _provider_registry():
    try:
        registry = importlib.import_module("tileops.backend.registry")
        load = getattr(registry, "ensure_loaded", None)
        if callable(load):
            load()
        return registry
    except Exception:
        # Discovery imports third-party entry points. A broken optional backend
        # must leave torch_fl usable and let the route fall through to aten.
        return None


def _registered_targets():
    """Targets that declared a detector or kernel builder."""
    registry = _provider_registry()
    if registry is None:
        return None
    known = getattr(registry, "known_targets", None)
    if not callable(known):
        return None
    targets = known()
    if not isinstance(targets, (set, list, tuple, frozenset)):
        return None
    return set(targets)


def _detection_device(device):
    """Give provider detectors a device of their own runtime, when possible."""
    if device is None or getattr(device, "type", None) not in ("flagos", "cuda"):
        return device
    kind = "cuda" if device.type == "cuda" else "flagos"
    try:
        return provider_device_object(kind, getattr(device, "index", None))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return device


def resolve_target(device=None, *, allow_unclaimed=False) -> Optional[str]:
    """The TileOPs target serving *device*, or None when no provider claims it.

    A provider detector is authoritative. Without one, a CUDA build uses the
    provider's ``cuda`` target, and MUSA is selected only when the provider has
    explicitly registered it. A device is never assumed to use another ABI.
    """
    global _HAS_SELECT_TARGET
    try:
        import_tileops()
        backend = _provider_backend()
        if backend is None:
            return None
        detect = getattr(backend, "detect_target", None)
        if detect is None:
            try:
                dispatch = importlib.import_module("tileops.backend.dispatch")
            except ImportError:
                dispatch = None
            detect = getattr(dispatch, "detect_target", None)
        if callable(detect):
            _HAS_SELECT_TARGET = True
            target = detect(_detection_device(device))
            if target:
                return str(target)
        known = _registered_targets()
        platform = _tileops_platform()
        kind = getattr(device, "type", None)
        if kind == "flagos" and known is not None:
            # A torch_fl tensor is PrivateUse1 even when the provider names its
            # own device differently. The provider target table is the build-level
            # capability check; its tensor boundary is still exercised per call.
            if platform == "ascend" and "ascend" in known:
                return "ascend"
            if platform == "musa" and "musa" in known:
                return "musa"
            if platform == "metax" and "tileops_maca" in known:
                return "tileops_maca"
            if platform == "cuda" and "cuda" in known:
                return "cuda"
        if platform in ("cuda", "metax") and kind == "cuda":
            for candidate in sorted(known or ()):
                if candidate not in ("cuda", "nvidia", "metax") and _provider_claims(
                    device, candidate
                ):
                    return candidate
        if platform in ("cuda", "metax") and (
            allow_unclaimed
            or known is None
            or "cuda" in known
            or (kind == "meta" and "tileops_maca" in known)
        ):
            return "cuda"
        if (
            platform == "musa"
            and kind in ("flagos", "musa")
            and (allow_unclaimed or (known is not None and "musa" in known))
        ):
            return "musa"
        if (
            kind == "meta"
            and platform == "musa"
            and known is not None
            and "musa" in known
        ):
            return "musa"
    except Exception:
        return None
    return None


def is_tileops_available(device=None) -> bool:
    """Whether an installed provider and device boundary serve this device.

    Never raises: a missing optional dependency must not break ``import torch_fl``.
    Architecture support belongs to the provider; here the target and the
    torch_fl launch boundary must both exist before the generated route is used.
    """
    try:
        if device is not None and getattr(device, "type", None) == "cpu":
            return False
        target = (
            _build_target()
            if device is None
            else resolve_target(device)
            if _HAS_SELECT_TARGET is not False
            else None
        )
        return (
            target is not None and get_device_adapter(target, raising=False) is not None
        )
    except Exception:
        return False


def register_device_adapter(
    target: str,
    kind: str,
    to_device,
    from_device,
    get_device=None,
    get_stream=None,
) -> None:
    """Register a provider's zero-copy tensor and current-stream boundary.

    The provider remains responsible for its kernels and target architecture.
    torch_fl owns flagos storage, its stream, and returned tensor ownership. For
    a hard device, ``get_stream`` is mandatory; a target that merely shares the
    CUDA ABI has no separate adapter.
    """
    if not callable(to_device) or not callable(from_device):
        raise TypeError("TileOPs device adapters need callable tensor conversions")
    if get_stream is not None and not callable(get_stream):
        raise TypeError("TileOPs get_stream must be callable or None")
    if get_device is not None and not callable(get_device):
        raise TypeError("TileOPs get_device must be callable or None")
    if not _cuda_abi(target) and kind != "cuda" and not callable(get_stream):
        raise TypeError("TileOPs hard-device adapters need a stream lookup")
    if getattr(from_device, "__name__", "") == "_unbox":
        raise ValueError("a registered TileOPs adapter must return a flagos tensor")
    with _ADAPTERS_LOCK:
        _ADAPTERS[target] = (kind, to_device, from_device, get_device, get_stream)


def _get_musa_raw_stream(device=None):
    """Read the MUSA stream through torch_fl's C++ runtime."""
    from torch_fl import _C
    from torch_fl.compile.device_interface import FlagOSDeviceInterface

    return int(
        _C._get_musa_current_raw_stream(FlagOSDeviceInterface._vendor_device(device))
    )


def _register_flagos_device_adapters() -> None:
    """Install the MUSA adapter implemented by torch_fl's own runtime."""
    from torch_fl import _C

    if not hasattr(_C, "_get_musa_current_raw_stream"):
        return
    with _ADAPTERS_LOCK:
        _ADAPTERS.setdefault(
            "musa",
            (
                "musa",
                lambda tensor: tensor,
                lambda tensor: tensor,
                lambda index: index,
                _get_musa_raw_stream,
            ),
        )


# --------------------------------------------------------------------------- #
# provider device boundary
# --------------------------------------------------------------------------- #
def _cuda_abi(target: Optional[str]) -> bool:
    """Whether this target's provider uses the CUDA tensor ABI."""
    return target in ("cuda", "nvidia", "metax", "tileops_maca")


def _device_kind(target: str, adapter: tuple) -> str:
    """The device type a target executes on, independently of its vendor name."""
    return adapter[0]


def _box(t):
    """flagos tensor -> CUDA view sharing the same storage (no copy)."""
    if not isinstance(t, torch.Tensor) or t.device.type != "flagos":
        return t
    from torch_fl import _C

    return _C._flagos_to_cuda_view(t)


def _unbox(t, index: Optional[int] = None):
    """CUDA tensor -> flagos view, for values handed back to the caller."""
    from torch_fl import _C

    if index is None:
        index = 0 if t.device.index is None else t.device.index
    return _C._cuda_to_flagos_view(t, index)


def get_device_adapter(target: str, kind=None, *, raising=True):
    """The device boundary for a provider target, or None if unavailable."""
    if kind == "meta":
        # A constructor probe has no tensor; the actual call verifies the target.
        return ("meta", _box, _unbox, lambda index: index, None)
    if kind is None and _cuda_abi(target):
        return ("cuda", _box, _unbox, lambda index: index, None)
    # Vendor and target names differ: tileops_maca runs on CUDA-ABI devices,
    # while torch_fl's flagos storage is not a CUDA tensor to every plugin.
    if kind is not None:
        if kind in ("cuda", "metax"):
            return ("cuda", _box, _unbox, lambda index: index, None)
    elif _cuda_abi(target):
        return ("cuda", _box, _unbox, lambda index: index, None)
    # A provider may have registered a native adapter before the target is
    # queried. Do not replace it with torch_fl's default -- and, on a source-only
    # check without the compiled extension, do not require ``_C`` to exist.
    if target == "musa" and target not in _ADAPTERS and _tileops_platform() == "musa":
        _register_flagos_device_adapters()
    with _ADAPTERS_LOCK:
        adapter = _ADAPTERS.get(target)
        if adapter is not None:
            adapter = (_device_kind(target, adapter), *adapter[1:])
    if adapter is None and raising:
        raise RuntimeError(f"TileOPs target {target!r} has no torch_fl device adapter")
    return adapter


def prepare_tensor(t, target: str):
    """Convert an input without a host copy before provider execution."""
    adapter = get_device_adapter(target)
    return adapter[1](t) if isinstance(t, torch.Tensor) else t


def restore_tensor(t, target: str):
    """Return a provider tensor to flagos storage without a host copy."""
    if not isinstance(t, torch.Tensor):
        return get_device_adapter(target)[2](t)
    if t.device.type == "flagos":
        return t
    return get_device_adapter(target)[2](t)


def configure_device(target: str) -> None:
    """Prepare a provider target without committing its compilation to this route.

    Op construction can encounter an empty input whose output it allocates on a
    concrete device. That output must use this route's device adapter, while its
    hardware diagnostics remain deferred to the actual call's Ops.
    """
    adapter = get_device_adapter(target)
    if adapter[0] == "meta" or (adapter[0] == "cuda" and adapter[4] is None):
        return
    get_stream = adapter[4]
    if get_stream is None:
        raise RuntimeError(f"TileOPs target {target!r} has no stream lookup")
    from tilelang.jit.adapter.base import BaseKernelAdapter

    if target not in _CONFIGURED_TARGETS:
        previous = BaseKernelAdapter.get_current_stream_functor
        BaseKernelAdapter.get_current_stream_functor = staticmethod(
            _make_stream_functor(target, get_stream, previous)
        )
        _CONFIGURED_TARGETS.add(target)


def _make_stream_functor(target: str, get_stream, fallback):
    """Return torch_fl's current raw stream for the target's device kind."""

    def get_current_stream_functor(device=None):
        adapter = get_device_adapter(target, raising=False)
        if device is not None and adapter is not None:
            if getattr(device, "type", None) == "meta":
                return lambda: 0
            if device.type == adapter[0]:
                index = 0 if device.index is None else device.index
                return lambda: get_stream(index)
        return fallback(device)

    return get_current_stream_functor


def _unwrapped_device_constructor(kind: str, index: int):
    """Construct a device without torch_fl's cuda-to-flagos alias mode."""
    from torch.overrides import TorchFunctionMode

    class _NoDeviceAlias(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    with _NoDeviceAlias():
        try:
            return torch.device(kind, index)
        except RuntimeError as exc:
            if kind != "npu":
                raise RuntimeError(
                    f"TileOPs target {kind!r} has no registered torch device; "
                    "install the matching vendor Torch plugin"
                ) from exc
            # torch_npu supplies npu; it is not the flagos allocator.
            return torch.device(kind, index)


def provider_device_object(kind: str, index: Optional[int] = None):
    """Give a provider its own native device object."""
    if kind == "generic":
        # No vendor runtime is required for construction. Every detector rejects
        # this neutral probe; Torch-FL uses it only to select the build's target.
        return torch.device("meta", -1 if index is None else index)
    return _unwrapped_device_constructor(kind, -1 if index is None else index)


def _detection_device_object(device, target: str):
    """Give a detector object needed only when it can still identify its target."""
    adapter = get_device_adapter(target, raising=False)
    if adapter is None or adapter[0] == "cuda":
        return _detection_device(device)
    index = getattr(device, "index", None)
    try:
        vendor_device = provider_device_object(adapter[0], index)
        get_adapter = adapter[3]
        if callable(get_adapter) and target not in ("musa", "tileops_maca"):
            vendor_device = get_adapter(-1 if index is None else index)
        return vendor_device
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return _detection_device(device)


def _provider_claims(device, target: str) -> bool:
    """Ask a target's registered detector whether it owns this device."""
    registry = _provider_registry()
    detect = getattr(registry, "DETECTORS", {}).get(target) if registry else None
    if not callable(detect):
        return True
    try:
        return bool(detect(_detection_device_object(device, target)))
    except Exception:
        return False


def _build_target(platform=None, *, allow_unclaimed=False):
    """Select the provider target of this build without inventing a flagos tensor."""
    platform = _tileops_platform() if platform is None else platform
    try:
        import_tileops()
    except ImportError:
        return None
    known = _registered_targets()
    if platform == "musa" and known is not None and "musa" in known:
        return "musa"
    if platform == "ascend" and known is not None and "ascend" in known:
        # The delivered Ascend provider intentionally leaves its detector off
        # until every op has a builder. It remains reachable by target name.
        return "ascend"
    if platform == "metax" and known is not None and "tileops_maca" in known:
        # Current ChipMind snapshots register known_targets() but also expose a
        # detector that only recognizes native CUDA metadata. Torch-FL owns
        # PrivateUse1 and reaches MetaX through the CUDA ABI view, so select the
        # capability directly instead of dropping to its boxing wrapper set.
        return "tileops_maca"
    if platform == "metax" and known is None:
        # ChipMind's snapshots predate known_targets(); their module owns the
        # detector, which is enough to identify the provider before construction.
        try:
            package = importlib.import_module("tileops_maca")
            if callable(getattr(package, "register_backend", None)):
                package.register_backend()
        except ImportError:
            return None
        return str(getattr(package, "TARGET", "tileops_maca"))
    try:
        backend = _provider_backend()
        if backend is None:
            return None
        has_detect = getattr(backend, "detect_target", None)
        if not callable(has_detect):
            dispatch = importlib.import_module("tileops.backend.dispatch")
            has_detect = getattr(dispatch, "detect_target", None)
    except ImportError:
        has_detect = None
    if not callable(has_detect):
        # Older provider snapshots, including ChipMind MetaX, expose a registry
        # but no detector entry point. The Torch-FL build names its target.
        if platform == "metax":
            return "tileops_maca" if known and "tileops_maca" in known else None
        if platform == "cuda" and (allow_unclaimed or known is None or "cuda" in known):
            return "cuda"
        if platform == "musa" and known is not None and "musa" in known:
            return "musa"
        return None
    device = provider_device_object("generic", 0)
    return resolve_target(device, allow_unclaimed=allow_unclaimed)


def _device_index(t: torch.Tensor) -> int:
    return t.device.index or 0


# --------------------------------------------------------------------------- #
# instance cache
# --------------------------------------------------------------------------- #
def _log_declined(overload: str) -> None:
    """Note a call TileOPs could not serve, when ``FLAGOS_LOG`` lists `dispatch`.

    The dispatcher logs the routing decision, which is made before the arguments
    are known. When a route then declines on dtype or argument shape the real
    kernel is aten's, so this second line is what explains a "-> tileops" log
    line followed by vendor-speed timings.
    """
    if "dispatch" in _env.listed("FLAGOS_LOG"):
        target = resolve_target() or "cpu_fallback"
        print(
            f"[flagos dispatch] {overload} -> {target} (tileops declined)",
            flush=True,
        )


class _Entry:
    """A cached Op, plus its kernel once that shortcut is proven safe.

    Skipping the L2 wrapper is worth 0.05-0.09 ms per call, but it cannot be
    assumed: ``Op.forward`` may reshape or broadcast before handing off, so
    ``op.kernel`` does not necessarily accept the arguments ``Op.__call__`` does.
    Binary ops are exactly that case -- L2 flattens first, and calling the kernel
    with the original 2-D tensors fails an ndim check.

    So the shortcut is verified rather than assumed: on the first call, run L2
    (always correct), then try the kernel on the same inputs and compare. Only an
    exact match enables the fast path; anything else pins this instance to L2 for
    good. One extra invocation is negligible next to the ~4 s constructor.

    A provider may change the target from the flagos device it was given. The
    output is owned by the device adapter, while the provider's kernel remains a
    persistent part of this entry.
    """

    __slots__ = (
        "op",
        "kernel",
        "target",
        "probed",
        "probe_args",
        "consumed_probe",
        "shape_committed",
        "refusal",
    )

    def __init__(self, op, target: str):
        self.op = op
        self.target = target
        # A plugin may commit a shape without having a target selection API.
        # Keep its construction resolution for real calls in that case.
        self.shape_committed = any(
            name in inspect.signature(type(op)).parameters
            for name in ("N_total", "a_shape")
        )
        self.probed = False
        self.probe_args = None
        self.consumed_probe = False
        self.refusal = None
        # None = not probed yet, False = L2 only, callable = verified shortcut.
        self.kernel = None if not _use_l2() else False

    def invoke(self, *args):
        if self.shape_committed:
            previous = getattr(_provider_target, "active", None)
            _provider_target.active = self.target
            try:
                args = tuple(prepare_tensor(arg, self.target) for arg in args)
                return _restore_result(self.op(*args), self.target)
            finally:
                _provider_target.active = previous
        args = tuple(prepare_tensor(a, self.target) for a in args)
        if self.refusal is not None:
            raise RuntimeError from self.refusal
        if self.kernel:
            out = self.kernel(*args)
        else:
            op = self.op
            # A provider can decline this instance and run its in-tree CUDA kernel.
            # Observe the binding, so a call never claims that its vendor target ran.
            seen = getattr(op, "_target_builder", None) is not getattr(
                type(op), "_target_builder", object()
            )
            if seen:
                op._target_builder = getattr(type(op), "_UNRESOLVED", None)
            out = op(*args)
            if seen:
                op._target_builder = None
            target = _provider_serving_target(op)
            if target is not None and str(target) != self.target:
                raise RuntimeError(
                    f"{type(op).__name__} ran target {target!r}, not {self.target!r}"
                )
            if self.kernel is None:
                self.kernel = self._probe(args, out)
        return _restore_result(out, self.target)

    def _probe(self, args, expected):
        """Return the kernel if it reproduces the L2 result exactly, else False."""
        kernel = getattr(self.op, "kernel", None)
        if kernel is None:
            return False
        try:
            got = kernel(*args)
        except Exception:
            return False
        if isinstance(expected, torch.Tensor) and isinstance(got, torch.Tensor):
            if got.shape == expected.shape and torch.equal(got, expected):
                return kernel
        return False


def _restore_result(result, target: str):
    return restore_tensor(result, target)


# --------------------------------------------------------------------------- #
# recipes
# --------------------------------------------------------------------------- #
def _aten_op(overload: str):
    base, _, ov = overload.partition(".")
    packet = getattr(torch.ops.aten, base)
    return getattr(packet, ov) if ov else packet


def _provider_target(op_cls, device, default: str) -> Optional[str]:
    """Resolve the provider that serves this concrete input."""
    bound = getattr(_provider_target, "active", None)
    if bound is not None:
        return bound
    detected = resolve_target(device)
    if detected is not None:
        return detected
    if getattr(device, "type", None) == "flagos" and default is not None:
        # ``default`` comes from the build-time bind target when construction was
        # probed on meta. A torch_fl tensor is PrivateUse1 even when a CUDA-ABI
        # provider names its own target, so resolve_target() above cannot always
        # map it. Let the same registered target validate the first physical call.
        return default
    kind = getattr(device, "type", None)
    if kind == "flagos":
        try:
            platform = _tileops_platform()
            if platform in ("cuda", "metax"):
                kind = "cuda"
            elif platform == "musa":
                kind = "musa"
        except Exception:
            pass
    registered = _registered_targets()
    if registered is not None:
        if kind == "cuda" and "tileops_maca" in registered:
            if _provider_claims(device, "tileops_maca"):
                return "tileops_maca"
        if kind == "musa" and "musa" in registered:
            if _provider_claims(device, "musa"):
                return "musa"
    return default


def _device(kind: str, index: int):
    return torch.device(kind, index)


def _provider_target_impl(impl, target: str):
    """Bind target-free Ops during construction, as an agent before each real call."""

    original = impl

    def guarded(*args, **kwargs):
        device = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
        if (
            device is not None
            and device.device.type != "meta"
            and _provider_target(None, device.device, target) != target
        ):
            raise TypeError("the actual device does not belong to this provider target")
        previous = getattr(_provider_target, "active", None)
        _provider_target.active = target
        try:
            return impl(*args, **kwargs)
        finally:
            _provider_target.active = previous

    guarded.__wrapped__ = original
    return guarded


class _ProviderTargetOp:
    """Run every real call on one target without constructing mock tensors."""

    def __init__(self, op, target: str):
        self.op = op
        self.target = target

    def __call__(self, *args, **kwargs):
        touched = [
            arg
            for arg in (*args, *kwargs.values())
            if isinstance(arg, torch.Tensor) and arg.device.type == "flagos"
        ]
        previous = getattr(_provider_target, "active", None)
        _provider_target.active = self.target
        try:
            for tensor in touched:
                if _provider_target(None, tensor.device, self.target) != self.target:
                    raise TypeError("the actual device is not served by this target")
            prepared = tuple(
                prepare_tensor(arg, self.target)
                if isinstance(arg, torch.Tensor)
                else arg
                for arg in args
            )
            output = self.op(*prepared, **kwargs)
            return _restore_result(output, self.target)
        finally:
            _provider_target.active = previous


def _bind_tensor_devices(args, kwargs, target: str):
    """Let Op constructors read the selected device without entering a fake CUDA path."""
    adapter = get_device_adapter(target, raising=False)
    if adapter is None:
        raise RuntimeError(f"TileOPs target {target!r} has no device adapter")
    index = next(
        (
            arg.device.index or 0
            for arg in (*args, *kwargs.values())
            if isinstance(arg, torch.Tensor) and arg.device.type == "flagos"
        ),
        0,
    )

    def provider_device(tensor):
        return torch.device(adapter[0], index)

    fake = types.MethodType(provider_device, torch.empty(0))
    return (
        tuple(fake if arg is None else arg for arg in args),
        {key: fake if value is None else value for key, value in kwargs.items()},
    )


def _make_fallback(overload: str, target: Optional[str] = None):
    """Build a CPU fallback that cannot re-enter this TileOPs registration.

    The C++ backend fallback is unreachable once the generated shim has been
    selected. Calling the same aten packet with flagos arguments would therefore
    dispatch back here forever. Mirror the boxed fallback at this boundary:
    recursively move tensor arguments to CPU, run aten there, and copy tensor
    results back to the originating flagos device.
    """
    op = _aten_op(overload)

    def fallback(*args, **kwargs):
        _log_declined(overload)
        tree = (args, kwargs)
        tensor_leaves = (
            value
            for value in _pytree.tree_leaves(tree)
            if isinstance(value, torch.Tensor)
        )
        device = next(
            (value.device for value in tensor_leaves if value.device.type == "flagos"),
            None,
        )
        if device is None:
            return op(*args, **kwargs)

        def to_cpu(value):
            if isinstance(value, torch.Tensor) and value.device.type == "flagos":
                return value.cpu()
            return value

        cpu_args, cpu_kwargs = _pytree.tree_map(to_cpu, tree)
        result = op(*cpu_args, **cpu_kwargs)

        def to_flagos(value):
            if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                return value.to(device)
            return value

        return _pytree.tree_map(to_flagos, result)

    return fallback


def open_provider_op(op_cls, target: str, *args, **kwargs):
    """Construct an Op bound to the provider target for this call.

    Current TileOPs takes keyword-only ``target=``. Older vendor copies may take
    it at the end of positional arguments or only expose a kernel map. Instance
    attributes are read only to distinguish a new Op's declared ``target`` from
    an inherited class attribute.
    """
    parameters = inspect.signature(op_cls).parameters
    if inspect.isabstract(op_cls):
        return op_cls(*args, **kwargs)
    if "target" in parameters:
        try:
            op = op_cls(*args, target=target, **kwargs)
        except TypeError:
            op = op_cls(*args, target, **kwargs)
    else:
        op = op_cls(*args, **kwargs)
    if "target" in getattr(op, "__dict__", {}):
        op.target = target
        return op
    vendor_map = getattr(op_cls, "_maca_kernel_map", None)
    default_map = getattr(op, "default_kernel_map", None)
    if callable(vendor_map) and isinstance(default_map, dict) and default_map:
        kernels = vendor_map()
        op.kernel_map = {role: kernels[role] for role in default_map}
        return op
    # The pinned external Op binds through its target-aware kernel constructor.
    return op


def _ctor_param_names(op_cls) -> tuple[str, ...]:
    """Semantic constructor parameters across TileOPs provider generations.

    The delivery-era TileOPs API commits ``N_total`` / ``dtype`` (unary),
    ``a_shape`` / ``b_shape`` / ``dtype`` (binary), or ``dtype`` (reduction) in
    the constructor. The September API moved those facts to the first forward
    call and keeps only tuning and operation parameters in the constructor.
    ``target``, ``tune``, ``kernel_map``, and ``strategy`` are consumer plumbing
    and never identify the semantic variant.
    """
    return tuple(
        name
        for name in inspect.signature(op_cls).parameters
        if name
        not in {"self", "target", "tune", "kernel_map", "strategy", "args", "kwargs"}
    )


def _get(op_cls, target: str, ctor_key: tuple, *args, probe=None, **kwargs):
    """Return a provider Op instance, cached per target and constructor shape."""
    target = str(target)
    key = (target, op_cls, ctor_key)
    entry = _instances.get(key)
    if entry is None:
        global _HAS_SELECT_TARGET
        _HAS_SELECT_TARGET = True
        if get_device_adapter(target) is None:
            raise RuntimeError(f"TileOPs target {target!r} has no device adapter")
        configure_device(target)
        args, kwargs = _bind_tensor_devices(args, kwargs, target)
        op = open_provider_op(op_cls, target, *args, **kwargs)
        entry = _Entry(op, target)
        entry.probe_args = probe
        if len(_instances) < _INSTANCE_CACHE_MAX:
            _instances[key] = entry
        else:
            global _cache_full_warned
            if not _cache_full_warned:
                _cache_full_warned = True
                warnings.warn(
                    f"tileops instance cache hit its {_INSTANCE_CACHE_MAX}-entry "
                    "cap; further shapes rebuild their Op on every call. This "
                    "usually means dynamic shapes. Raise "
                    "FLAGOS_TILEOPS_CACHE_MAX if the shape set is finite.",
                    RuntimeWarning,
                    stacklevel=2,
                )
    elif entry.refusal is not None:
        raise RuntimeError from entry.refusal
    elif entry.probed and len(_instances) < _INSTANCE_CACHE_MAX:
        _instances[key] = entry
    return entry


def verify_provider(entry, args):
    """Prove that the provider target, rather than an in-tree Op, serves a call.

    A target-free Op gets a small target-bound agent while it compiles its
    zero-element kernel. A target-aware Op is checked after its real call.
    """
    if getattr(entry.op, "target", None) is None and not entry.shape_committed:
        # A target that truly declined is pinned too: let its fragment correlation
        # select once, instead of raising outside this call's aten fallback.
        if _HAS_SELECT_TARGET is False and not entry.probe_args:
            return
        if entry.probed:
            return
        entry.op = _provider_target_impl(entry.op, entry.target)
        wrapped = entry.op
        original = entry.op.__wrapped__
        used_probe = not entry.consumed_probe and bool(entry.probe_args)
        try:
            entry.op(*(entry.probe_args if used_probe else args))
        except Exception as exc:
            entry.op = wrapped.__wrapped__
            entry.refusal = exc
            raise RuntimeError from exc
        entry.refusal = None
        entry.probed = True
        entry.consumed_probe = True
        entry.op = _ProviderTargetOp(original, entry.target)
        key = (entry.target, type(original), None)
        if len(_instances) < _INSTANCE_CACHE_MAX:
            for cache_key, cached in tuple(_instances.items()):
                if cached is entry:
                    _instances[key] = entry
                    del _instances[cache_key]
        return
    entry.invoke(*args)
    target = _provider_serving_target(entry.op)
    if target is None or str(target) != entry.target:
        raise RuntimeError(
            f"{type(entry.op).__name__} selected {target!r}, not {entry.target!r}"
        )


def _provider_serving_target(op):
    """Return the provider target recorded by current and legacy TileOPs APIs."""
    return getattr(op, "serving_target", None) or getattr(op, "_settled_target", None)


def _check_registered(op_cls, target: str) -> None:
    """Require the provider to cover this Op before constructing it."""
    registry = _provider_registry()
    builders = getattr(registry, "BUILDERS", None) if registry else None
    if isinstance(builders, dict) and (op_cls.__name__, target) not in builders:
        raise ValueError(f"TileOPs target {target!r} does not cover {op_cls.__name__}")


def _dispatch_provider(
    op_cls,
    dtypes,
    extra,
    overload,
    execute,
    device=None,
    fallback_factory=None,
):
    default = resolve_target()
    if default is None and device is not None:
        # Construction is deliberately probed on an unclaimed ``meta`` device.
        # resolve_target() therefore has no physical device to inspect there;
        # the generated route's build target is the part that must survive until
        # the first ops call. The per-call _provider_target() below rechecks it
        # against that call's actual device.
        default = _build_target()
    fallback = (
        fallback_factory(default)
        if fallback_factory is not None
        else _make_fallback(overload, default)
    )

    def guard(*args, **kwargs):
        target = _provider_target(op_cls, args[0], default)
        if target is not None and get_device_adapter(target) is None:
            raise RuntimeError(f"TileOPs target {target!r} has no device adapter")
        try:
            if target is not None:
                _check_registered(op_cls, target)
                for arg in (*args, *kwargs.values()):
                    if isinstance(arg, torch.Tensor):
                        prepare_tensor(arg, target)
                return execute(op_cls, target, *args, device=device, **kwargs)
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            if _env.flag("FLAGOS_TILEOPS_DEBUG"):
                warnings.warn(
                    f"TileOPs target {target!r} declined {overload}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return fallback(*args, **kwargs)
        return fallback(*args, **kwargs)

    return guard


def _unary_impl(op_cls, dtypes, extra, overload, device=None):
    ctor_extra = (False,) if extra.get("inplace") else ()
    ctor_params = _ctor_param_names(op_cls)
    commits_layout = "N_total" in ctor_params

    def execute(op_cls, target, x, *rest, device=None):
        if x.dtype not in dtypes or rest:
            raise TypeError("dtype or arguments not served by the unary recipe")
        xb = x.contiguous()
        # Delivery-era providers commit N_total in the constructor and take a
        # flat buffer. Current providers infer it from the first forward call.
        flat = xb.reshape(-1) if commits_layout else xb
        if commits_layout:
            ctor_args = (flat.numel(), x.dtype) + ctor_extra
            ctor_kw = {}
        else:
            ctor_args = ()
            ctor_kw = {"inplace": False} if extra.get("inplace") else {}
        entry = _get(
            op_cls,
            target,
            (flat.numel(), x.dtype) + ctor_extra if commits_layout else ctor_extra,
            *ctor_args,
            probe=provider_probe_args(UNARY, x.dtype, device, overload)
            if device
            else None,
            **ctor_kw,
        )
        return entry.invoke(flat)

    return _dispatch_provider(op_cls, dtypes, extra, overload, execute, device)


def _binary_impl(op_cls, dtypes, extra, overload, device=None):
    takes_alpha = extra.get("alpha", False)
    ctor_params = _ctor_param_names(op_cls)
    commits_shapes = {"a_shape", "b_shape", "dtype"} <= set(ctor_params)

    def execute(op_cls, target, a, b, *rest, device=None, **kw):
        alpha = kw.pop("alpha", 1)
        if (
            not isinstance(a, torch.Tensor)
            or not isinstance(b, torch.Tensor)
            or a.dtype != b.dtype
            or a.dtype not in dtypes
            or rest
            or kw
            or (alpha != 1 and not takes_alpha)
        ):
            raise TypeError("dtype or arguments not served by the binary recipe")
        ab, bb = a.contiguous(), b.contiguous()
        shapes = (tuple(ab.shape), tuple(bb.shape))
        ctor_args = shapes + (a.dtype,) if commits_shapes else ()
        ctor_kw = {"alpha": alpha} if takes_alpha else {}
        key = ctor_args + ((alpha,) if takes_alpha else ())
        entry = _get(
            op_cls,
            target,
            key,
            *ctor_args,
            probe=provider_probe_args(BINARY, a.dtype, device, overload)
            if device
            else None,
            **ctor_kw,
        )
        return entry.invoke(ab, bb)

    return _dispatch_provider(op_cls, dtypes, extra, overload, execute, device)


def _reduce_impl(op_cls, dtypes, extra, overload, device=None):
    takes_keepdim = extra.get("keepdim", False)
    takes_correction = extra.get("correction", False)
    ctor_params = _ctor_param_names(op_cls)
    commits_dtype = "dtype" in ctor_params

    def execute(op_cls, target, x, dim=None, keepdim=False, *rest, device=None, **kw):
        correction = kw.get("correction")
        if (
            x.dtype not in dtypes
            or rest
            or "dtype" in kw
            or set(kw) - {"correction"}
            or (correction is not None and not takes_correction)
        ):
            raise TypeError("dtype or arguments not served by the reduction recipe")
        if isinstance(dim, (list, tuple)):
            if len(dim) != 1:
                raise TypeError("TileOPs serves one reduction axis")
            dim = dim[0]
        elif dim is None:
            raise TypeError("TileOPs needs an explicit reduction axis")
        xb = x.contiguous()

        ctor_args = (x.dtype, dim) if commits_dtype else (dim,)
        ctor_kw = {}
        if takes_keepdim:
            ctor_kw["keepdim"] = keepdim
        if takes_correction and correction is not None:
            ctor_kw["correction"] = correction
        key = (x.dtype, dim, tuple(sorted(ctor_kw.items())))
        entry = _get(
            op_cls,
            target,
            key,
            *ctor_args,
            probe=provider_probe_args(REDUCE, x.dtype, device, overload)
            if device
            else None,
            **ctor_kw,
        )
        return entry.invoke(xb)

    def impl(x, dim=None, keepdim=False, *rest, **kw):
        # The generated shim accepts aten's keyword-only keepdim in its
        # positional slot. Normalize that here and let the common guard own the
        # target check, provider call, debug warning, and CPU fallback.
        provider_keepdim = rest[0] if rest else keepdim
        provider_rest = rest[1:] if rest else rest

        def fallback_factory(target):
            fallback = _make_fallback(overload, target)

            def call_fallback(x, dim, provider_keepdim, *provider_rest, **kw):
                if overload in KEYWORD_ONLY_REDUCE:
                    return fallback(
                        x,
                        dim,
                        *provider_rest,
                        keepdim=provider_keepdim,
                        **kw,
                    )
                if overload in NO_POSITIONAL_KEEPDIM:
                    return fallback(x, dim, *provider_rest, **kw)
                return fallback(x, dim, provider_keepdim, *provider_rest, **kw)

            return call_fallback

        return _dispatch_provider(
            op_cls,
            dtypes,
            extra,
            overload,
            lambda op_cls, target, x, *args, device=None: execute(
                op_cls,
                target,
                x,
                *args,
                device=device,
            ),
            device,
            fallback_factory=fallback_factory,
        )(x, dim, provider_keepdim, *provider_rest, **kw)

    return impl


def _softmax_impl(op_cls, dtypes, extra, overload, device=None):
    def execute(op_cls, target, x, dim=-1, half_to_float=False, *rest, device=None):
        if x.dtype not in dtypes or half_to_float or rest:
            raise TypeError("dtype or arguments not served by the softmax recipe")
        entry = _get(
            op_cls,
            target,
            (dim,),
            dim,
            probe=provider_probe_args(SOFTMAX, x.dtype, device, overload)
            if device
            else None,
        )
        return entry.invoke(x.contiguous())

    return _dispatch_provider(op_cls, dtypes, extra, overload, execute, device)


_BUILDERS = {
    UNARY: _unary_impl,
    BINARY: _binary_impl,
    REDUCE: _reduce_impl,
    SOFTMAX: _softmax_impl,
}


def build_impl(recipe, module, cls_name, dtype_names, extra, overload, device=None):
    """Build the callable for one generated route, or None if unavailable.

    Returns None when the Op class cannot be imported; callers that need a
    working callable regardless should use ``resolve_impl``, which substitutes
    the aten fallback. No dispatch logging is added here -- the C++ dispatcher
    already logs the decision, and only a decline needs explaining (see
    ``_log_declined``).
    """
    try:
        op_cls = _resolve_op_class(module, cls_name)
    except Exception:
        return None
    dtypes = tuple(getattr(torch, name) for name in dtype_names)
    try:
        impl = _BUILDERS[recipe](op_cls, dtypes, extra, overload, device)
        impl.target_bound = True
        return impl
    except (ImportError, AttributeError, TypeError, RuntimeError):
        return None


def _resolve_op_class(module: str, cls_name: str):
    """Resolve an Op across TileOPs provider generations.

    The routing manifest comes from the September official tree, which split
    logical reductions and predicates into their own modules. The delivered
    Ascend and MetaX providers keep the older compact layout. Public package
    exports remain stable across both generations, so use the generated module
    first and the nearest package export as a compatibility fallback.
    """
    try:
        return getattr(importlib.import_module(module), cls_name)
    except (ImportError, AttributeError) as original:
        package = module.rsplit(".", 1)[0]
        try:
            return getattr(importlib.import_module(package), cls_name)
        except (ImportError, AttributeError):
            raise original


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def resolve_impl(overload, recipe, module, cls_name, dtype_names, extra):
    """Callable for one generated route, for the C++ stub to call.

    A provider is detected once against the runtime's device before an Op is
    constructed. Zero-element target probes let a detector reject an unsupported
    provider without inventing a flagos tensor, and keep an in-tree fallback from
    being mistaken for a vendor target.
    """
    platform = _tileops_platform()
    target = _build_target(platform)
    if target is not None:
        device = provider_device_object("generic", 0)
        impl = build_impl(
            recipe, module, cls_name, dtype_names, extra, overload, device
        )
        if impl is not None:
            return impl
    return _make_fallback(overload, target)


def enable_tileops_for_flagos(include=None, exclude=None) -> int:
    """Deprecated; TileOPs now registers in C++ and needs no Python call.

    Kernels register on ``Backend::kTileOps`` when the extension module loads,
    and routing is decided by the conf files, so there is nothing to enable.
    Retained because it was the documented entry point; returns the number of
    routes so an existing truthiness check still behaves.

    ``include``/``exclude`` no longer have anything to act on -- use
    ``FLAGOS_OP_<overload>=cuda`` to divert a single op, which the dispatcher
    honors for every backend uniformly.
    """
    if include is not None or exclude is not None:
        warnings.warn(
            "enable_tileops_for_flagos(include=/exclude=) no longer filters "
            "registration; use FLAGOS_OP_<overload>=<backend> instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if not is_tileops_available():
        return 0
    from torch_fl.tileops.generated.routes import ROUTES

    return len(ROUTES)


def registered_ops() -> Dict[str, str]:
    """aten overload -> TileOPs class name for every generated route.

    Reports what the build can route, not what a given process has routed:
    whether a call lands on TileOPs is a per-op conf decision the dispatcher
    makes, and the kernels are registered unconditionally.
    """
    if not is_tileops_available():
        return {}
    from torch_fl.tileops.generated.routes import ROUTES

    return {overload: cls_name for overload, _, _, cls_name, _, _ in ROUTES}


# --------------------------------------------------------------------------- #
# helpers for tests and warmup
# --------------------------------------------------------------------------- #
#: Overloads whose result is only real for positive inputs. Comparing against
#: aten on negative values yields nan on both sides, which says nothing about the
#: kernel, so tests and warmup feed these strictly positive data.
POSITIVE_DOMAIN = frozenset(
    {
        "log",
        "log1p",
        "log2",
        "log10",
        "sqrt",
        "rsqrt",
        "pow.Tensor_Tensor",
        "remainder.Tensor",
    }
)


#: Reduce overloads whose schema declares ``int[1] dim`` rather than a bare int.
#: Passing the wrong one is a hard schema-match error, so it is spelled out.
DIM_IS_LIST = frozenset(
    {
        "sum.dim_IntList",
        "mean.dim",
        "amax",
        "amin",
        "logsumexp",
        "count_nonzero.dim_IntList",
        "var.correction",
        "std.correction",
        "var_mean.correction",
    }
)

#: Reduce overloads that take no ``keepdim`` positionally (or at all).
NO_POSITIONAL_KEEPDIM = frozenset({"count_nonzero.dim_IntList"})

#: Reduce overloads whose ``correction``/``keepdim`` are keyword-only.
KEYWORD_ONLY_REDUCE = frozenset(
    {"var.correction", "std.correction", "var_mean.correction"}
)


def sample_inputs(
    recipe, shape: Sequence[int], dtype, device="flagos", overload=None
) -> Tuple:
    """Positional args matching an overload's schema, for tests and warmup."""
    return _sample_inputs(recipe, shape, dtype, device, overload)


def _sample_inputs(recipe, shape, dtype, device, overload):
    """Allocate one CPU reference or native-device workload."""
    shape = tuple(shape)

    def make():
        if dtype == torch.bool:
            return torch.randint(0, 2, shape, device=device, dtype=torch.bool)
        if not dtype.is_floating_point:
            # Small positive ints keep bitwise/comparison results exact and stay
            # within range of the narrow int dtypes.
            return torch.randint(1, 9, shape, device=device, dtype=dtype)
        t = torch.randn(shape, dtype=dtype).to(device)
        return t.abs() + 0.5 if overload in POSITIVE_DOMAIN else t

    x = make()
    if recipe == BINARY:
        return (x, make())
    if recipe == SOFTMAX:
        return (x, -1, False)
    if recipe == REDUCE:
        dim = [-1] if overload in DIM_IS_LIST else -1
        if overload in KEYWORD_ONLY_REDUCE or overload in NO_POSITIONAL_KEEPDIM:
            return (x, dim)
        return (x, dim, False)
    return (x,)


def provider_probe_args(recipe, dtype, device, overload):
    """Zero-element inputs a provider Op can compile from an unclaimed device."""
    dim = [-1] if overload in DIM_IS_LIST else -1
    x = torch.empty(0, dtype=dtype, device=device)
    if recipe == BINARY:
        return (x, x.clone())
    if recipe == SOFTMAX:
        return (x, -1, False)
    if recipe == REDUCE:
        return (x, dim) if overload in KEYWORD_ONLY_REDUCE else (x, dim, False)
    return (x,)


def warmup(overloads=None, dtype=torch.float16) -> int:
    """Pre-build kernels on manifest workload shapes.

    Useful for a first run on a cold ``TILELANG_CACHE_DIR``, and mandatory under
    ``FLAGOS_TILEOPS_DISABLE_ALL_CACHE=1``, where every process re-pays the full
    compile per op. With the default path (see ``_disable_frontend_cache``) the
    kernel disk cache survives, so later processes start warm and this is only a
    convenience.
    """
    if not is_tileops_available():
        return 0
    from torch_fl.tileops.generated.routes import ROUTES, WORKLOADS

    want = str(dtype).replace("torch.", "")
    built = 0
    for overload, recipe, module, cls_name, dtype_names, extra in ROUTES:
        if overloads is not None and overload not in overloads:
            continue
        shape = WORKLOADS.get(overload)
        if shape is None or want not in dtype_names:
            continue
        impl = build_impl(recipe, module, cls_name, dtype_names, extra, overload)
        if impl is None:
            continue
        try:
            impl(*sample_inputs(recipe, shape, dtype, overload=overload))
            built += 1
        except Exception:
            continue
    return built
