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

"""PyTorch device aliases for runtime stacks without a CUDA device."""

import torch

from torch_fl import _env, _platform


def _keep_device_identity_checks_working(real_device, shim):
    """Repair the two torch registries that compare against ``torch.device``.

    Rebinding ``torch.device`` to a Python class (above) is invisible to almost
    everything -- ``isinstance`` still works, and every consumer that only
    *calls* it gets a genuine device back. Two places compare the attribute by
    identity instead, and both silently change behaviour:

    ``torch.fx.graph.add_global`` carves out ``obj != torch.device`` so that a
    device constant is emitted as the bare name ``device(type='cpu')`` and
    resolved through the ``device`` custom builtin. With the attribute swapped,
    the real device class no longer equals it, so the branch falls through to
    the qualified-name HACK path meant for custom ops -- the name is never added
    to the module's globals, and the generated forward dies with
    ``NameError: name 'device' is not defined`` the moment it runs. It is
    ``torch.arange(..., device=x.device)`` that emits such a constant, which is
    how every HF model builds its position ids, so this took out essentially
    every transformer under ``torch.compile``.

    ``torch._dynamo.utils.common_constant_types`` holds the types Dynamo may
    wrap as a ``ConstantVariable``, and it is tested with ``type(obj) in ...``.
    Reading ``tensor.device`` while tracing then asserts with "Cannot construct
    ``ConstantVariable`` for value of type ``torch.device``".

    Both registries are keyed on objects rather than rebuilt per call, so they
    can be corrected in place once, here. Re-running this is harmless.
    """
    from torch.fx.graph import _register_custom_builtin

    # Point the builtin at the shim: `add_global` skips the qualified-name path
    # for anything not defined under `torch`, so the shim reaches the normal
    # branch and the name lands in the generated module's globals. Calling it
    # still yields a real device, which is all the generated code needs.
    _register_custom_builtin("device", "from torch import device", shim)

    # Dynamo compares `type(obj)`, and a constructed device is always the real
    # C type whatever `torch.device` currently names -- so it is the real class
    # that has to be in the set.
    from torch._dynamo.utils import common_constant_types

    common_constant_types.add(real_device)


# Whether install_alias has taken over torch.cuda, i.e. whether every
# "cuda" this process reports is the flagos device wearing a CUDA name. Read by
# torch_fl.compat.triton.stand_down_foreign_drivers. Stays False on a build
# with real CUDA, where cuda means cuda.
_active = False


# Device strings that have to mean the flagos device on this build, keyed by the
# accelerator the build record holds -- the vendor, which is also what the
# lookup below passes. "cuda" is on every row: it is the name the ecosystem
# hardcodes.
# The second entry on a row is the vendor's *own* torch device name, which the
# vendor's Triton/FlagGems integration hands to torch.device but which is not
# registered here -- torch_fl claims PrivateUse1 as "flagos" and never imports
# the vendor package (torch_musa / torch_txda) that would register it.
#
#   mthreads   torch_musa's device, and FlagGems' own device.name.
#   tsingmicro FlagTree's TsingMicro Triton driver resolves its active device as
#              torch.device("txda", ...) (triton/backends/tsingmicro/driver.py),
#              and FlagGems' tsingmicro VendorDescriptor names the device "txda"
#              too. That call happens at *import* of flag_gems, from Triton's
#              hint manager, so leaving "txda" out does not merely mistype a
#              tensor: `import flag_gems` raises and every flaggems-routed op
#              on TsingMicro is unreachable.
#   cambricon  Triton's mlu driver builds its autotune scratch as
#              torch.empty(..., device='mlu') and reports the device interface
#              as torch.device("mlu", torch.mlu.current_device())
#              (triton/backends/mlu/driver.py). torch_mlu would have registered
#              the name; this build does not install torch_mlu, so the alias is
#              what keeps the benchmark step of every autotuned FlagGems kernel
#              from dying with "Expected one of cpu, cuda, ... device type at
#              start of device string: mlu". Deliberately NOT "cuda": MLU has no
#              CUDA runtime, and a flagos tensor here is cnrt memory, so
#              reinterpreting a device string as CUDA would be wrong rather than
#              merely unhelpful.
_VENDOR_DEVICE_ALIASES = {
    "mthreads": ("cuda", "musa"),
    "tsingmicro": ("cuda", "txda"),
    "cambricon": ("mlu",),
}

# The CUDA probe as it stands when the alias decision is made: torch's own
# answer, or the constant-True probe a CUDA-ABI vendor compat layer installed
# (`patch_torch_cuda_for_flagos` and the MetaX/DCU equivalents), and before
# `install_alias` itself replaces the attribute. `has_real_cuda`
# answers from this captured probe.
_cuda_probe = None


def has_real_cuda() -> bool:
    """Whether a genuine CUDA runtime is reachable through ``torch.cuda``.

    That is torch's own answer on a CUDA build, and the CUDA-ABI vendor compat
    layers' constant-True probe on a CPU torch wheel paired with the vendor's
    externally loaded ``libtorch_cuda.so``. It is *not* the answer once torch_fl
    itself owns the attribute: `install_alias` writes
    ``lambda: flagos.device_count() > 0`` and `_patch_native_cuda_probe` writes
    ``torch.flagos.is_available``, both of which answer True on any build with a
    flagos device, whether or not a CUDA runtime exists.

    Reading the alias's own probe instead answers yes on exactly the builds the
    alias exists for. Measured on MUSA with the redirect in place and this guard
    reading it: nothing past the guard ran, ``torch.cuda.current_device()``,
    ``synchronize``, ``device_count`` and ``get_device_properties`` all stayed
    torch's, and dynamo's ``cuda_extra_check`` -- which calls
    ``torch.cuda.current_device()`` -- landed in ``torch.cuda._lazy_init`` and
    raised ``AssertionError: Torch not compiled with CUDA enabled``, so
    ``torch.compile(model, fullgraph=True)`` could not compile at all.

    The answer is captured while it still means that, not read live: by the time
    anything else asks, torch_fl may own the attribute.
    """
    return bool(_cuda_probe())


def install_alias(flagos) -> None:
    """Make ``device="cuda"`` mean the flagos device when there is no real CUDA.

    Most of the PyTorch ecosystem hardcodes ``"cuda"``: ``model.cuda()``,
    ``device_map="cuda"``, ``torch.device("cuda")`` in example scripts, and
    ``torch.cuda.is_available()`` as the "do I have an accelerator" test. On a
    vendor backend built without CUDA (Ascend), every one of those raises
    ``AssertionError: Torch not compiled with CUDA enabled`` -- so code that runs
    unmodified elsewhere has to be edited to say ``"flagos"``.

    This rewrites ``cuda`` device *arguments* to the flagos device, so that
    hardcoded-``cuda`` code lands on the accelerator that is actually present.

    Deliberately a no-op when *torch* reports a CUDA device: on the CUDA and
    boxing backends ``cuda`` already means a real device, and hijacking it there
    would break the boxing path, which submits genuine CUDA work. Which is not
    the same question as whether ``torch.cuda.is_available()`` says so --
    torch_fl redirects that probe on a build without a CUDA runtime, and reading
    the redirect here is what kept this alias uninstalled; see
    ``has_real_cuda``.

    Opt out with ``FLAGOS_ALIAS_CUDA=0`` -- worth doing if you need
    ``device="cuda"`` to keep failing loudly rather than silently redirecting.
    """
    # This runs after the vendor CUDA facade is ready and before the alias
    # or the compile integration replaces the live probe.
    global _cuda_probe
    _cuda_probe = torch.cuda.is_available
    if has_real_cuda():
        return
    if not _env.flag("FLAGOS_ALIAS_CUDA", True):
        return

    # From here on, every "cuda" this process sees is this alias. Anything that
    # treats torch.cuda as a hardware probe has to be told, because it can no
    # longer distinguish the accelerator from a CUDA device that does not exist;
    # see torch_fl.compat.triton.stand_down_foreign_drivers.
    global _active
    _active = True

    from torch.overrides import TorchFunctionMode

    _flagos_type = torch._C._get_privateuse1_backend_name()  # "flagos"
    _orig_device = torch.device

    def _remap(dev):
        """Vendor accelerator aliases -> flagos; everything else untouched."""
        if isinstance(dev, str):
            aliases = _VENDOR_DEVICE_ALIASES.get(
                _platform.build_info().vendor, ("cuda",)
            )
            for alias in aliases:
                if dev == alias:
                    return _flagos_type
                if dev.startswith(f"{alias}:"):
                    return f"{_flagos_type}:{dev[len(alias) + 1 :]}"
            return dev
        if isinstance(dev, _orig_device) and dev.type == "cuda":
            # `dev.index` is passed through even when it is None, which is the
            # whole point: an index-less target means "the current device", and
            # torch resolves that through the privateuse1 device guard, honouring
            # `set_device`. Forcing 0 here instead sent every rank of a
            # multi-rank job to flagos:0 -- and only through the `torch.device`
            # object, because the str branch above already leaves the index to
            # torch. `nn.Module.to("cuda")` reaches this branch, since
            # `_parse_to` turns its argument into a device object first.
            return _orig_device(_flagos_type, dev.index)
        return dev

    # A TorchFunctionMode, not a wrapper around torch.device.
    #
    # Wrapping torch.device is not enough and was tried first: factory functions
    # parse their `device=` argument in C++ (THPDevice / the argument parser), so
    # `torch.randn(4, device="cuda")` never reaches a Python torch.device call and
    # still dies in torch.cuda._lazy_init. A torch-function mode sits above the
    # C++ parser and sees the keyword before it is resolved, which catches every
    # factory uniformly -- randn, zeros, empty, tensor, arange, and `.to()`.
    #
    # Pushed permanently onto the mode stack at import. That is unusual but
    # intended: the alias has to hold for the whole process, not a `with` block.
    # `__torch_function__` runs on every op, so the body is kept to a dict lookup
    # and a `str.startswith` on the miss path.
    class _CudaAliasMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            dev = kwargs.get("device")
            if dev is not None:
                remapped = _remap(dev)
                if remapped is not dev:
                    kwargs = {**kwargs, "device": remapped}
            # Positional device, as in `Tensor.to("cuda")`.
            elif args and func is torch.Tensor.to:
                remapped = _remap(args[1]) if len(args) > 1 else None
                if remapped is not None and len(args) > 1 and remapped is not args[1]:
                    args = (args[0], remapped) + args[2:]
            return func(*args, **kwargs)

    torch._C._push_on_torch_function_stack(_CudaAliasMode())

    # torch.device("cuda") itself, for code that builds the device object first
    # and only later passes it to a factory (transformers' device_map does this).
    # torch.device is a C type and cannot be subclassed, so wrap the constructor;
    # isinstance(x, torch.device) must keep working, hence __instancecheck__.
    class _DeviceMeta(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, _orig_device)

    class device(metaclass=_DeviceMeta):  # noqa: N801  (mirrors torch.device)
        def __new__(cls, *args, **kwargs):
            if args:
                args = (_remap(args[0]),) + args[1:]
            elif "device" in kwargs:
                kwargs = {**kwargs, "device": _remap(kwargs["device"])}
            return _orig_device(*args, **kwargs)

    # Pickle resolves a class by its qualified module name. Local classes
    # are unreachable by that name, so this class is published as torch.device.
    # That costs more than hand-pickling a device: Inductor pickles the graph
    # module to build its FX graph cache key, and a device constant in the graph
    # brings this class along, so every compile hits BypassFxGraphCache ("Failed
    # to pickle cache key") and recompiles from scratch. Naming it ``torch.device``
    # -- which is exactly what the next line makes it -- makes that lookup find
    # this same object, so the reference round-trips.
    device.__module__ = "torch"
    device.__qualname__ = "device"
    device.__name__ = "device"

    torch.device = device
    _keep_device_identity_checks_working(_orig_device, device)

    # Tensor.cuda() / Module.cuda() -> the flagos device.
    def _tensor_cuda(self, device=None, non_blocking=False, **kwargs):
        # `Tensor.cuda()` with no argument means the current device on CUDA, so
        # it has to mean the current device here too: an index-less target is
        # left index-less and resolved by torch, the same way the `_remap` str
        # branch does it. Pinning 0 instead put every rank of a multi-rank job
        # on flagos:0 and NCCL rejected the duplicate device.
        target = _flagos_type
        if device is not None:
            index = _orig_device(_remap(device)).index
            if index is not None:
                target = f"{_flagos_type}:{index}"
        return self.to(target, non_blocking=non_blocking)

    torch.Tensor.cuda = _tensor_cuda

    # `torch.cuda.is_available()` is the ecosystem's "have I got an accelerator"
    # probe, and gating on it is what sends code down the CPU path. Report the
    # flagos device count so that probe finds the accelerator that is there.
    #
    # Left alone: is_bf16_supported, get_device_capability, and the rest of
    # torch.cuda -- the vendor shim already owns those, and overriding them here
    # would fight it.
    torch.cuda.is_available = lambda: flagos.device_count() > 0
    torch.cuda.device_count = flagos.device_count
    torch.cuda.current_device = flagos.current_device
    torch.cuda.set_device = flagos.set_device
    torch.cuda.synchronize = flagos.synchronize
    # Once is_available() says yes, dynamo's CudaInterface.get_device_properties
    # is called for every device while building compilation metrics, and the
    # stock one goes through torch.cuda._lazy_init -> "Torch not compiled with
    # CUDA enabled". Point it at the flagos properties instead.
    torch.cuda.get_device_properties = flagos.get_device_properties
    capture_query = getattr(flagos, "is_current_stream_capturing", lambda: False)
    torch.cuda.is_current_stream_capturing = capture_query
    graphs_mod = getattr(torch.cuda, "graphs", None)
    if graphs_mod is not None:
        graphs_mod.is_current_stream_capturing = capture_query
        if isinstance(
            getattr(graphs_mod, "_cuda_isCurrentStreamCapturing", None), type
        ):
            graphs_mod._cuda_isCurrentStreamCapturing = capture_query
    if isinstance(
        getattr(torch._C, "_cuda_isCurrentStreamCapturing", None),
        (type, type(None)),
    ):
        try:
            torch._C._cuda_isCurrentStreamCapturing = capture_query
        except Exception:
            pass


def install_ppu_stream_capture_query(flagos) -> None:
    """Replace the CPU wheel's cached CUDA graph query on PPU only."""
    graphs = getattr(torch.cuda, "graphs", None)
    if graphs is None:
        return
    cached_query = getattr(graphs, "_cuda_isCurrentStreamCapturing", None)
    if not (
        isinstance(cached_query, type)
        or getattr(cached_query, "__module__", None) == "torch._utils"
    ):
        return
    query = flagos.is_current_stream_capturing
    torch.cuda.is_current_stream_capturing = query
    graphs.is_current_stream_capturing = query
    graphs._cuda_isCurrentStreamCapturing = query


def is_active() -> bool:
    """Return whether the compatibility layer installed device aliases."""
    return _active
