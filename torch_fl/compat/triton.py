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

"""Device-runtime compatibility for Triton and FlagTree."""


def stand_down_foreign_drivers(cuda_alias_active: bool) -> None:
    """Make Triton's driver factory pick this accelerator, and only it.

    Triton resolves its driver lazily, and refuses to resolve at all when more
    than one installed backend reports itself active:

        RuntimeError: 2 active drivers ([...NPUDriver, ...CudaDriver]).
        There should only be one.

    Which drivers exist is a property of the Triton *build*, not of the host.
    FlagTree ships every backend it was built with in one wheel, so the Ascend
    wheel also carries an ``nvidia`` backend whose ``is_active()`` is
    ``torch.cuda.is_available() and torch.version.hip is None``
    (triton/backends/nvidia/driver.py). On Ascend that expression is True only
    because of ``compat.cuda.install_alias``, which repoints
    ``torch.cuda.is_available`` at the flagos device count so that ecosystem
    code hardcoding ``"cuda"`` finds the accelerator that is actually present.
    Triton reads the same attribute as its hardware probe, concludes a CUDA
    device is present too, and then any FlagGems module that resolves a
    benchmarker while being imported -- ``flag_gems.fused`` does -- dies inside
    its own import.

    The Ascend driver is left alone: it activates on its own, through
    ``bishengir-compile -print-targets``. Only the stock-CUDA driver is stood
    down, and only when the CUDA Triton sees is the alias rather than real
    hardware, so a CUDA build never reaches the assignment. Idempotent.

    Must run after ``import torch_fl`` -- importing the Ascend driver at all
    reaches ``triton/backends/ascend/testing.py``, which imports ``torch_npu``
    unconditionally and relies on the stub torch_fl publishes.
    """
    if not cuda_alias_active:
        # torch.cuda is real here, so whatever it reports to Triton is true.
        return
    try:
        import triton.backends
    except ImportError:
        return
    backend = triton.backends.backends.get("nvidia")
    if backend is None:
        return
    backend.driver.is_active = staticmethod(lambda: False)


def bind_kunlunxin_stream(platform: str, device_module) -> None:
    """Give Kunlunxin's Triton backend a stream source that exists here.

    Triton's ``GPUDriver.__init__`` binds its host-runtime accessors to
    ``torch.cuda`` unconditionally::

        try:
            from torch._C import _cuda_getCurrentRawStream
            self.get_current_stream = _cuda_getCurrentRawStream
        except ImportError:
            self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream

    FlagTree's XPU backend subclasses it and overrides the *device* pair only
    (``XPUDriver.__init__``, under its ``FLAGTREE XPU SYNC MARK``); nothing under
    ``triton/backends/xpu/`` assigns ``get_current_stream``. On the CPU-torch
    wheel this backend runs on there is no ``_cuda_getCurrentRawStream``, so the
    fallback lambda is installed and the first kernel launch dies with
    ``AssertionError: Torch not compiled with CUDA enabled`` raised from
    ``torch.cuda._lazy_init``.

    The handle kernels must submit to is the one torch_fl's own stream carries,
    ``flagos.current_stream(index).handle`` -- the same value
    ``get_kunlunxin_current_raw_stream`` in ``torch_fl.compile.flagtree_shim``
    already reports to inductor. Binding it here is what lets FlagGems, which
    launches eagerly and long before any ``torch.compile`` shim exists, reach
    the device at all.

    Measured on a P800 (torch 2.10.0+cpu, triton 3.6.0): with the lambda in
    place a plain ``tl.load``/``tl.store`` kernel raises in ``_lazy_init``; with
    this binding it launches, synchronizes, and reads back correct values.

    Wrapped at ``DriverConfig.active`` rather than applied once at import,
    because the driver is created lazily on that property's first read and
    resolving it during ``_lazy_init`` imports the backend compiler, which asks
    torch_fl questions device init has not finished answering. Wrapping the
    property catches the driver however late it is built; ``set_active`` (a test
    hook) is wrapped too so a substituted driver is bound the same way.

    Only replaces the parent class's fallback lambda, so a backend that grows a
    real implementation keeps it. No-op off Kunlunxin and when Triton is absent.
    Idempotent.
    """
    if platform != "kunlunxin":
        return
    try:
        import triton.runtime.driver as triton_driver
    except ImportError:
        return

    # ``import triton.runtime.driver as triton_driver`` yields the DriverConfig
    # *instance*, not the module: ``triton/runtime/__init__.py`` rebinds the name
    # to ``driver.driver``. Accept either shape.
    config = None
    if hasattr(triton_driver, "set_active"):
        config = triton_driver
    else:
        for attribute in ("driver", "_driver"):
            candidate = getattr(triton_driver, attribute, None)
            if hasattr(candidate, "set_active"):
                config = candidate
                break
    if config is None:
        return

    cls = type(config)
    if getattr(cls, "_flagos_stream_bound", False):
        return

    # ``getattr`` rather than ``cls.__dict__``: a subclass that only inherits the
    # property still has to be intercepted, and a test that substitutes the
    # class (``instance.__class__ = Sub``) would otherwise slip past.
    original_active = getattr(cls, "active", None)
    if not isinstance(original_active, property):
        return
    get_active = original_active.fget

    def active(self):
        return _bind_stream_on_driver(get_active(self), device_module)

    original_set = getattr(cls, "set_active")

    def set_active(self, driver):
        original_set(self, _bind_stream_on_driver(driver, device_module))

    cls.active = property(active)
    cls.set_active = set_active
    cls._flagos_stream_bound = True


def _bind_stream_on_driver(driver, device_module):
    """Point one Triton driver's stream accessor at torch_fl's stream handle.

    Split out of :func:`bind_kunlunxin_stream` so the lazy wrapper can
    apply it to whichever driver object finally gets built. Replaces the parent
    class's fallback lambda and nothing else; idempotent because the replacement
    no longer matches.
    """
    if driver is None:
        return driver
    current = getattr(driver, "get_current_stream", None)
    owner = getattr(current, "__qualname__", "")
    if "GPUDriver.__init__" not in owner:
        return driver

    # Resolve the accessor on the registered device module when Triton asks
    # for a stream. This also works while the package is still importing.
    def _current_raw_stream(index=0):
        return int(device_module.current_stream(index).handle)

    driver.get_current_stream = _current_raw_stream
    return driver


def install_ascend_policy():
    """Point FlagTree's Ascend backend at torch_fl instead of at torch_npu.

    FlagTree's Ascend backend answers its host-runtime questions -- the version
    hash in the compile cache key, the JIT'd launcher's compiler flags and
    includes, current device and current stream -- from a strategy registry
    keyed by a "backend policy" string, and it selects that policy by trying
    ``import torch_npu``. torch_fl publishes a torch_npu stub for FlagGems'
    benefit, so the import succeeds, the ``torch_npu`` policy is selected, and
    the first FlagGems kernel dies on the stub's missing surface
    (``torch_npu.version``) -- or, one step further in, on
    ``#include <torch_npu/...>`` and ``-ltorch_npu`` in the generated launcher,
    neither of which exists without the real extension.

    torch_fl carries a ``flagos`` policy for exactly this
    (``torch_fl.compile.flagtree_ascend_policy``), but it was only installed on
    the ``torch.compile`` path. FlagGems launches Triton kernels eagerly, so it
    has to be installed here as well, before any kernel runs.

    No-op unless the active Triton is FlagTree's Ascend build *and* FlagGems is
    installed -- the policy exists only for FlagGems' kernels, and a build whose
    registry no longer matches this shim must not take down device init for
    someone who never imports FlagGems.

    Raises:
        RuntimeError: on a FlagTree Ascend build that has FlagGems but whose
            backend cannot take the flagos policy. FlagGems could not run
            there at all, and the message from ``install_policy`` names the
            reason, unlike the torch_npu AttributeError it would otherwise
            surface as, per op.
    """
    import importlib.util

    try:
        if importlib.util.find_spec("flag_gems") is None:
            return
    except (ImportError, ValueError):
        return

    from torch_fl.compile.flagtree_shim import flagtree_backend

    if flagtree_backend() != "ascend":
        return

    from torch_fl.compile.flagtree_ascend_policy import install_policy

    install_policy()


def preload_kunlunxin_runtime(platform: str) -> None:
    """Load the vendor's CUDA-compat runtime before Triton's launcher does.

    FlagTree's XPU launcher is a shared object that resolves `cudaMemcpy` at
    dlopen time:

        liblaunch_shared.so: undefined symbol: cudaMemcpy

    The symbol lives in `/usr/local/xcudart/lib/libcudart.so.12` as a *versioned*
    definition (`cudaMemcpy@@libcudart.so.12`), so it is only findable by a
    later dlopen if that library was already loaded into the global scope.
    Triton loads its launcher lazily on the first kernel, by which point nothing
    has pulled libcudart in, and the launcher ends up half-resolved: the kernel
    is accepted but `xpu_wait(NULL)` never returns. Measured on a P800
    (torch 2.10.0+cpu): without this, every kernel -- vendor, FlagGems and a
    hand-written `@triton.jit` one alike -- hangs on the subsequent sync; with
    it, the same kernels complete and return correct values.

    A hang rather than an error is why this is worth doing at import: the
    dlopen failure is reported once, to stderr, by the launcher's own debug
    path, and nothing downstream looks.

    No-op off Kunlunxin, when `libcudart.so.12` is already loaded, or when the
    file is absent. Idempotent.
    """
    if platform != "kunlunxin":
        return
    import ctypes
    import os

    target = "libcudart.so.12"
    for line in _loaded_shared_objects():
        if os.path.basename(line) == target:
            return

    root = os.environ.get("XPYTORCH_XCUDART_PATH") or os.environ.get("XPU_CUDART_PATH")
    candidates = []
    if root:
        candidates.append(os.path.join(root, "lib", "libcudart.so.12"))
        candidates.append(os.path.join(root, "libcudart.so.12"))
    # The vendor package's default layout; xcudart is what supplies the
    # CUDA-compatible surface the launcher and libxdnn_pytorch both link.
    candidates.append("/usr/local/xcudart/lib/libcudart.so.12")
    candidates.append("/usr/local/xpu/so/libcudart.so.12")
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        return


def _loaded_shared_objects():
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                parts = line.split()
                if len(parts) >= 6:
                    yield parts[5]
    except OSError:
        return
