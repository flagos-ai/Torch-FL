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

"""Ascend RNG bridge for FlagGems' generator-less RNG ops.

FlagGems' Triton RNG kernels never draw from ATen's generator: they call
``flag_gems.utils.random_utils.philox_backend_seed_offset``, which does

    generator = torch_device_fn.default_generators[device]
    c0, c1 = generator.get_state().view(torch.int64)

i.e. it unpacks the state as exactly two int64s -- a CUDA philox state of
(seed, offset) -- then advances the offset and writes the state back. On this
vendor ``torch_device_fn`` is ``torch.npu``, which torch_fl provides as a shim
over ``torch.flagos``, so the generator it reads is
``torch.flagos.default_generators[device]``.

Every other vendor arranges for that object to be philox-shaped: CUDA, MetaX and
DCU hand out real ``torch.Generator(device="cuda")`` objects, and GCU installs
``_GcuPhiloxGenerator`` (see accelerator/gcu/_gcu_compat.py). Ascend has no CUDA
to borrow from, so without an equivalent the object is torch_fl's own
``at::CPUGeneratorImpl`` -- CPU mt19937, a 5056-byte state -- and the unpack
raises ``ValueError: too many values to unpack (expected 2)``. Every FlagGems RNG
op on Ascend (rand/randn/rand_like/randn_like/uniform_/exponential_/
bernoulli_.float/multinomial/native_dropout) dies on it.

The objects below are pure Python and touch no device: they hold (seed, offset)
and implement the three methods the kernels use. The random numbers are still
generated on device by the Triton philox kernel -- this only supplies and
advances the stream coordinates.

They stay in step with the native flagos generator (which the aclnn RNG kernels
consume through ``ReserveSeed``) because every public seeding entry point,
``torch.manual_seed`` -> ``torch.flagos.manual_seed_all`` as well as direct
``torch.flagos.manual_seed`` calls, reseeds both. One seed therefore drives an
interleaved sequence of FlagGems and native draws.
"""

import torch

from torch_fl import flagos


_ascend_generators = {}

_UINT64_MASK = (1 << 64) - 1

_installed = False


def _as_int64_bits(value):
    """Reinterpret a uint64 seed as the signed value with the same bit pattern.

    ``torch.initial_seed()`` and CUDA's philox seed are unsigned 64-bit, so a
    default seed routinely exceeds ``int64`` max. ``torch.tensor(..., int64)``
    raises ``ValueError: Overflow when unpacking long long`` on those values,
    which would break every RNG call on the device.
    """
    value &= _UINT64_MASK
    return value - (1 << 64) if value >= (1 << 63) else value


class _AscendPhiloxGenerator:
    """Minimal generator implementing FlagGems' seed/offset protocol."""

    def __init__(self, seed):
        self._seed = int(seed) & _UINT64_MASK
        self._offset = 0

    def get_state(self):
        # 16 bytes of seed+offset, the same layout a CUDA generator exposes.
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
            raise ValueError("Ascend philox state must contain seed and offset")
        # The state carries the unsigned bit pattern in a signed container.
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


def _get_ascend_generator(index):
    generator = _ascend_generators.get(index)
    if generator is None:
        generator = _AscendPhiloxGenerator(torch.initial_seed())
        _ascend_generators[index] = generator
    return generator


class _AscendDefaultGenerators:
    """List-like per-device philox generators for FlagGems' RNG kernels.

    Bounds-checked so iteration terminates: with no ``__iter__``, Python's
    legacy protocol calls ``__getitem__(0, 1, 2, ...)`` until IndexError.
    """

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, index):
        n = len(self)
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(n)))
        index = int(index)
        if index < 0:  # negative indices wrap, as on a list
            index += n
        if not 0 <= index < n:
            raise IndexError(f"device index {index} out of range for {n} device(s)")
        return _get_ascend_generator(index)

    def __len__(self):
        return max(flagos.device_count(), 1)


def install_ascend_rng_generators():
    """Point FlagGems' RNG source at per-device philox state objects."""
    global _installed
    if _installed:
        return
    _installed = True

    generators = _AscendDefaultGenerators()
    native_proxy = flagos.default_generators
    flagos.default_generators = generators

    # torch_fl's torch.npu shim is built with default_generators bound to the
    # native proxy; rebind it when it is still pointing there. A real torch_npu
    # (never installed by us, and forbidden as a dependency) would not be, and
    # is left alone.
    npu = getattr(torch, "npu", None)
    if npu is not None and getattr(npu, "default_generators", None) is native_proxy:
        npu.default_generators = generators

    native_manual_seed = flagos.manual_seed
    native_manual_seed_all = flagos.manual_seed_all

    def manual_seed(seed):
        seed = int(seed)
        native_manual_seed(seed)
        _get_ascend_generator(flagos.current_device()).manual_seed(seed)

    def manual_seed_all(seed):
        seed = int(seed)
        native_manual_seed_all(seed)
        for index in range(len(generators)):
            _get_ascend_generator(index).manual_seed(seed)

    flagos.manual_seed = manual_seed
    flagos.manual_seed_all = manual_seed_all
