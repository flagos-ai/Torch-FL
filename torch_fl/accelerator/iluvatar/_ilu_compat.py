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

"""FlagGems and Triton glue for the Iluvatar (corex) backend.

Everything here exists for one structural reason: FlagGems' ``_iluvatar``
descriptor is a copy of its *NVIDIA* one (``device_name="cuda"``), and every
place FlagGems -- and the Triton backend underneath it -- reaches for the CUDA
runtime directly is answered by the CPU torch wheel this build links, which has
no CUDA at all. Each piece below covers one of those reach-throughs:

1. ``torch.cuda.default_generators`` is what
   ``flag_gems.utils.random_utils.philox_backend_seed_offset`` reads for a
   generator-less RNG op. On a CUDA torch that is a tuple of real philox
   generators; on the CPU wheel it is ``()``, so the very first ``randn`` dies
   with ``IndexError: tuple index out of range``. The generators installed here
   hold (seed, offset) in pure Python -- the same object Ascend uses
   (``torch_fl.accelerator.ascend._ascend_compat``), for the same reason -- and
   are reseeded by ``flagos.manual_seed``/``manual_seed_all`` so one seed still
   drives both the native kernels' generator and FlagGems' philox stream.

2. ``DeviceDetector.name`` is ``"cuda"``, and FlagGems' op modules capture it
   into a module global and compare it against the input's device type. A flagos
   tensor reports ``"flagos"``, so the comparison never matches and the op takes
   FlagGems' fallback branch instead of its Triton kernel. Realigning the name is
   what CUDA and DCU already do (``patch_flaggems_device_name``).

3. The handful of ``torch.cuda`` entry points FlagGems calls as ordinary work --
   ``torch_device_fn.device(idx)`` around a launch, ``synchronize()``,
   ``current_device()``, and the capability Triton turns into its ``arch`` --
   raise ``RuntimeError: PyTorch was compiled without CUDA support`` on the CPU
   wheel, so ``install_ilu_cuda_shim`` routes them to ``torch.flagos``.

4. Triton's autotuner allocates its L2-flush buffer on ``"cuda"``, which does not
   exist here (``patch_triton_benchmark_allocator``), and its iluvatar driver
   compiles a C helper against ``cuda.h`` using environment variables only the
   vendor's shell setup exports (``_ensure_corex_headers_on_path``).

All of it is best-effort on the same terms as the other vendors' bridges: a
failure here loses FlagGems' routed kernels, it must not take down device init.
"""

import torch

from torch_fl import flagos


_UINT64_MASK = (1 << 64) - 1

_generators = {}
_installed = False
_cuda_shim_installed = False


def _as_int64_bits(value):
    """Reinterpret a uint64 seed as the signed value with the same bit pattern.

    ``torch.initial_seed()`` and a philox seed are unsigned 64-bit, so a default
    seed routinely exceeds int64 max and ``torch.tensor([...], int64)`` raises
    ``ValueError: Overflow when unpacking long long`` on it.
    """
    value &= _UINT64_MASK
    return value - (1 << 64) if value >= (1 << 63) else value


class _IluvatarPhiloxGenerator:
    """Minimal generator implementing FlagGems' seed/offset protocol."""

    def __init__(self, seed):
        self._seed = int(seed) & _UINT64_MASK
        self._offset = 0

    def get_state(self):
        # 16 bytes of seed+offset, the layout FlagGems unpacks as (c0, c1).
        return torch.tensor(
            [_as_int64_bits(self._seed), _as_int64_bits(self._offset)],
            dtype=torch.int64,
            device="cpu",
        ).view(torch.uint8)

    def set_state(self, state):
        values = state.reshape(-1)
        if values.dtype != torch.int64:
            values = values.view(torch.int64)
        if values.numel() != 2:
            raise ValueError("Iluvatar philox state must contain seed and offset")
        self._seed, self._offset = (
            int(values[0].item()) & _UINT64_MASK,
            int(values[1].item()) & _UINT64_MASK,
        )

    def manual_seed(self, seed):
        self._seed = int(seed) & _UINT64_MASK
        self._offset = 0
        return self

    def seed(self):
        return self._seed

    def initial_seed(self):
        return self._seed


class _IluvatarDefaultGenerators:
    """Tuple-like per-device generators, as ``torch.cuda.default_generators``.

    Bounds-checked so iteration terminates: with no ``__iter__``, Python's legacy
    protocol calls ``__getitem__(0, 1, 2, ...)`` until IndexError.
    """

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, index):
        n = len(self)
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(n)))
        index = int(index)
        if index < 0:  # negative indices wrap, as on a tuple
            index += n
        if not 0 <= index < n:
            raise IndexError(f"device index {index} out of range for {n} device(s)")
        return _generator(index)

    def __len__(self):
        return max(flagos.device_count(), 1)


def _generator(index):
    generator = _generators.get(index)
    if generator is None:
        generator = _IluvatarPhiloxGenerator(torch.initial_seed())
        _generators[index] = generator
    return generator


def install_ilu_rng_generators():
    """Give FlagGems' RNG ops a per-device seed/offset source. Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True

    generators = _IluvatarDefaultGenerators()
    torch.cuda.default_generators = generators

    native_manual_seed = flagos.manual_seed
    native_manual_seed_all = flagos.manual_seed_all

    def manual_seed(seed):
        seed = int(seed)
        native_manual_seed(seed)
        _generator(flagos.current_device()).manual_seed(seed)

    def manual_seed_all(seed):
        seed = int(seed)
        native_manual_seed_all(seed)
        for index in range(len(generators)):
            _generator(index).manual_seed(seed)

    flagos.manual_seed = manual_seed
    flagos.manual_seed_all = manual_seed_all


def _device_index(device):
    if device is None:
        return flagos.current_device()
    if isinstance(device, torch.device):
        return device.index if device.index is not None else flagos.current_device()
    if isinstance(device, str):
        return int(device.split(":")[-1]) if ":" in device else flagos.current_device()
    return int(device)


def _ensure_corex_headers_on_path():
    """Put corex's headers and SDK root where Triton's driver build finds them.

    FlagTree's iluvatar driver compiles a small C module against ``cuda.h`` and
    resolves ``-lcuda`` from ``$COREX_ROOT/...``; it reads both from environment
    variables that the vendor's own shell setup would have exported. Neither is
    set for a conda interpreter that was merely activated, so the first kernel
    launch dies in ``gcc`` with ``cuda.h: No such file or directory``.

    Never overwrites a value the user set: on a machine whose corex lives
    somewhere else, their ``COREX_ROOT``/``CPATH`` is the authoritative one.
    """
    import os

    root = os.environ.get("COREX_ROOT") or "/usr/local/corex"
    if not os.path.isdir(os.path.join(root, "include")):
        return
    if not os.environ.get("COREX_ROOT"):
        os.environ["COREX_ROOT"] = root
    include = os.path.join(root, "include")
    path = os.environ.get("CPATH", "")
    if include not in path.split(os.pathsep):
        os.environ["CPATH"] = os.pathsep.join(filter(None, [path, include]))


def patch_triton_benchmark_allocator():
    """Point Triton's autotune benchmark buffer at this build's device. Idempotent.

    ``get_empty_cache_for_benchmark`` allocates the 256 MB buffer Triton flushes
    the L2 cache with before timing each candidate config, and it hardcodes
    ``device='cuda'``. That is correct on the vendor's stack, where torch *is* a
    CUDA torch; here torch is the CPU wheel, so the allocation has no CUDA kernel
    and every autotuned kernel dies with ``NotImplementedError: Could not run
    'aten::empty.memory_format' with arguments from the 'CUDA' backend`` before
    it is ever benchmarked. On bert that is 125 of the failures -- layernorm is
    autotuned, so every training test hit it.

    The replacement keeps the vendor's size and dtype and only changes the
    device, so autotuning measures what it was written to measure.
    """
    try:
        from triton.backends.iluvatar import driver as driver_module
    except ImportError:
        return False

    # Named `IluvatarDriver` on the 3.2 wheel and `CorexDriver` on 3.6, so find
    # the class by the method it defines rather than by name.
    target = None
    for name in dir(driver_module):
        candidate = getattr(driver_module, name)
        if isinstance(candidate, type) and "get_empty_cache_for_benchmark" in vars(
            candidate
        ):
            target = candidate
            break
    if target is None:
        return False

    def get_empty_cache_for_benchmark(self):
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="flagos")

    target.get_empty_cache_for_benchmark = get_empty_cache_for_benchmark
    return True


def install_ilu_cuda_shim():
    """Make the ``torch.cuda`` calls FlagGems makes answer for flagos. Idempotent.

    FlagGems' ``torch_device_fn`` is ``torch.cuda`` for every descriptor that
    names its device ``cuda``, and its op wrappers use it for ordinary work:
    ``with torch_device_fn.device(idx)`` around a launch, ``synchronize()``,
    ``current_device()``. On this build those are the CPU wheel's, which raise
    ``RuntimeError: PyTorch was compiled without CUDA support`` -- so a shim is
    required, not an optimization. Only the handful FlagGems reaches for are
    replaced; everything else on ``torch.cuda`` is left alone so a genuine
    "no CUDA here" answer is still visible to everybody else.

    The shim is installed here rather than by pointing FlagGems at
    ``torch.flagos`` because ``torch_device_fn`` is bound at import time and
    several op modules reach it as a module attribute.
    """
    global _cuda_shim_installed
    if _cuda_shim_installed:
        return
    _cuda_shim_installed = True

    cuda = torch.cuda
    cuda.current_device = lambda: flagos.current_device()
    cuda.set_device = lambda device: flagos.set_device(_device_index(device))
    cuda.device_count = lambda: flagos.device_count()
    cuda.is_available = lambda: True
    cuda.synchronize = lambda device=None: flagos.synchronize()
    cuda.get_device_properties = lambda device=None: flagos.get_device_properties(
        _device_index(device)
    )

    def _capability(device=None):
        """The real (major, minor), which Triton turns into its ``arch``.

        Triton asks the device interface for this and compiles for
        ``major * 10 + minor``; BI-V150 is sm_71. A stand-in value does not fail
        loudly -- the kernel loads, then hangs the device on launch, so the
        whole machine has to have its driver reloaded.
        """
        props = flagos.get_device_properties(_device_index(device))
        return int(props.major), int(props.minor)

    cuda.get_device_capability = _capability
    cuda.get_device_name = lambda device=None: (
        flagos.get_device_properties(_device_index(device)).name
    )
    cuda.current_stream = lambda device=None: flagos.current_stream(
        _device_index(device)
    )
    cuda.default_stream = lambda device=None: flagos.current_stream(
        _device_index(device)
    )

    # CPU torch's DeviceContext calls torch.cuda._exchange_device on entry, which
    # is exactly the call that raises. Swapping the device and reporting the
    # previous one is the whole of what it is asked for.
    def _exchange_device(idx):
        if idx is None or int(idx) < 0:
            return -1
        previous = flagos.current_device()
        flagos.set_device(int(idx))
        return previous

    cuda._exchange_device = _exchange_device
    cuda._maybe_exchange_device = _exchange_device
    cuda._lazy_init = lambda: None

    # flagos owns real event and stream objects over the same runtime.
    cuda.Event = flagos.Event
    cuda.Stream = flagos.Stream


def patch_iluvatar_flag_gems_device_name():
    """Realign FlagGems' device name with the one torch_fl registered. Idempotent.

    ``DeviceDetector`` copies ``device_name`` out of the vendor descriptor at
    construction, and the op modules capture it into a module global as they are
    imported, so both are rewritten -- the same two-step the CUDA and DCU fix
    performs, minus its ``get_device_properties`` wrapper: that one exists
    because ``torch.cuda.get_device_properties`` rejects a foreign name, and
    here the descriptors' lookups go to FlagGems' own ``get_device_properties``,
    which already accepts a flagos device.
    """
    import sys

    try:
        from flag_gems.runtime.backend.device_finder import DeviceDetector
    except ImportError:
        try:
            from flag_gems.runtime.backend.device import DeviceDetector
        except ImportError:
            return False

    detector = DeviceDetector()
    if detector.vendor_name != "iluvatar":
        return False
    if detector.name == "flagos":
        return True
    if detector.name != "cuda":
        # Another descriptor is in charge; not ours to rewrite.
        return False
    detector.name = "flagos"

    try:
        import flag_gems

        root = list(flag_gems.__path__ or [])
    except ImportError:
        return False
    root = str(root[0]) if root else None
    if root is None:
        return False

    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path or not str(path).startswith(root):
            continue
        namespace = vars(module)
        for attr in ("device", "_DEVICE_NAME"):
            if namespace.get(attr) == "cuda":
                namespace[attr] = "flagos"
    return True
