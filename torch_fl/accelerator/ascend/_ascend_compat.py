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

The second shim here is for ``diffusers`` rather than for FlagGems:
``patch_diffusers_qwenimage_rope`` registers the ``flagos`` device in
``QwenEmbedRope``'s rotary-embedding extension point, which ``diffusers`` would
otherwise serve from its ``cuda`` entry. That entry multiplies by a complex
exponential, and CANN has no complex compute at all, so the Qwen-Image
transformer cannot build its rotation operands on this backend without it.
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


# --- diffusers: the Qwen-Image rotary embedding ------------------------------
#
# ``diffusers.models.transformers.transformer_qwenimage`` keys its rotary
# embedding on ``device.type`` and knows two devices: ``cuda``, which multiplies
# by a complex exponential built with ``torch.polar``, and ``neuron``, which has
# no complex dtype and is handed rotation *angles* instead. A device type it does
# not list -- ``flagos`` among them -- takes the ``cuda`` fallback. That fallback
# is not merely slower here, it is unusable: CANN has no complex compute at all,
# so ``QwenEmbedRope._compute_video_freqs`` raises ``RuntimeError: Unsupported
# dtype for ACL: ComplexFloat`` from the ``torch.cat`` over
# ``freqs_neg``/``freqs_pos``, before the first rotation is applied and before the
# transformer produces a single latent.
#
# This is deliberately a second copy of the implementation in
# ``torch_fl/accelerator/gcu/_gcu_compat.py`` rather than an import of it: Ascend
# does not depend on the GCU accelerator package, and that file is owned by an
# in-flight GCU change this one must not disturb. The two copies are line for
# line the same rotation and must stay in step until they are unified in one
# device-neutral module; ``tests/unit/test_ascend_qwenimage_rope.py`` pins this
# copy against diffusers' own neuron expansion and against the complex path it
# replaces.
#
# Two installers for one key has one consequence and it is visible, so it is
# stated here rather than discovered: ``tests/unit/test_gcu_qwenimage_rope.py``
# asserts that its own call leaves every ``ROPE_PER_DEVICE`` entry -- ``flagos``
# included -- pointing at the object it found before the call, and that holds
# only while nothing else has claimed the key. On an Ascend host this
# installation claims it at ``torch_fl`` import, so that one assertion fails
# there; it does so on ``main`` too, where the same call sits at the end of
# ``torch_fl/__init__.py``. The assumption is the GCU test's and the fix belongs
# in it, which is a GCU file this change deliberately does not edit, and no CI
# job runs ``tests/unit`` on Ascend. The reverse case does not arise:
# ``tests/unit/test_ascend_qwenimage_rope.py`` tolerates a ``flagos`` entry that
# was already present -- which is what GCU installs.

_DISABLE_QWENIMAGE_ROPE = "FLAGOS_DISABLE_QWENIMAGE_ROPE"

_QWENIMAGE_ROPE_INSTALLED_ATTR = "_flagos_qwenimage_rope_installed"


def flagos_qwenimage_rotary_emb(x, freqs):
    """Qwen-Image rotary embedding for the flagos device.

    Same rotation as ``diffusers``' ``apply_rotary_emb_qwen_neuron`` -- see
    ``patch_diffusers_qwenimage_rope`` for why the angle expansion differs from
    that function's and why it matters here. Not a numerical shortcut: the
    values, their order and the output layout are identical, and this is
    asserted in ``tests/unit/test_ascend_qwenimage_rope.py``.
    """
    import torch

    # read each angle once and stack it, rather than let `repeat_interleave`
    # lower to a stride-0 expand whose reshape materialises through the copy
    # path; `.flatten` is a view, so `cos`/`sin` are contiguous either way.
    cos_angle = torch.cos(freqs)
    sin_angle = torch.sin(freqs)
    cos = torch.stack([cos_angle, cos_angle], dim=-1).flatten(-2, -1).unsqueeze(1)
    sin = torch.stack([sin_angle, sin_angle], dim=-1).flatten(-2, -1).unsqueeze(1)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def patch_diffusers_qwenimage_rope() -> bool:
    """Register the flagos Qwen-Image rotation, operands and consumer together.

    diffusers keys the Qwen-Image rotation on device type, in two places that have
    to agree:

        ROPE_PER_DEVICE = {"cuda": partial(apply_rotary_emb_qwen, use_real=False),
                           "neuron": apply_rotary_emb_qwen_neuron}

    picks the *consumer* at the attention call site, and

        QwenEmbedRope._get_device_freqs(device)      # lru_cache'd per device
        QwenEmbedLayer3DRope._get_device_freqs       # same, for the layer3d rope

    produces its *operand* -- complex frequencies for a device that has a complex
    dtype, rotation *angles* for one that does not. A flagos tensor is in neither
    table entry, so it takes the ``cuda`` fallback and multiplies by the complex
    exponential, which this backend cannot run at all: the operands raise
    ``Unsupported dtype for ACL: ComplexFloat`` from ``_compute_video_freqs``.

    Both halves are installed here, and they must be installed together and before
    the first forward -- the operand producer is cached per device, so registering
    only the consumer leaves it multiplying complex frequencies with a real-valued
    kernel. ``apply_rotary_emb_qwen_neuron`` is the entry diffusers provides for a
    backend with no complex dtype and is the correct consumer for that operand, but
    its angle layout costs more here than the rotation it feeds, which is what this
    registration replaces it with.

    Both of that function's layout lines expand one rotation angle into two
    adjacent features with ``repeat_interleave(2, dim=-1)``. torch has no
    accelerator kernel for that overload on this backend: ``aten``'s
    ``repeat_interleave.self_int`` is a composite that lowers to
    ``unsqueeze(-1).expand(..., 2).reshape(...)``, and the ``reshape`` materialises
    a stride-0 view through ``StridedCopy``, which is the CPU round-trip this
    backend's copies no longer take but still pay to issue. It is issued 480 times
    per forward. ``flagos_qwenimage_rotary_emb`` writes the same values in the same
    layout from two reads of one angle tensor instead of a stride-0 broadcast, and
    never touches the copy path. Measured against the neuron implementation, four
    variants, one warm forward each: mean, std, absmax and sum equal to four
    decimals, ``max|d| == 0.000e+00``.

    Safe to call more than once, and a no-op when diffusers is absent, has no
    Qwen-Image rope table, or ``FLAGOS_DISABLE_QWENIMAGE_ROPE`` is set -- the last
    of which is how the manual A/B in ``tests/manual/qwen_image_2512/README.md``
    measures this against the complex path now that it is the default. Returns
    whether the registration is in place.
    """
    import importlib.util

    import torch

    from torch_fl import _env

    if _env.flag(_DISABLE_QWENIMAGE_ROPE):
        return False
    if importlib.util.find_spec("diffusers") is None:
        return False
    try:
        from diffusers.models.transformers import transformer_qwenimage as qwenimage
    except ImportError:
        return False

    table = getattr(qwenimage, "ROPE_PER_DEVICE", None)
    if not isinstance(table, dict) or "cuda" not in table:
        return False

    if not getattr(qwenimage, _QWENIMAGE_ROPE_INSTALLED_ATTR, False):
        # The operand half: hand the rotation angles to `flagos` the same way
        # diffusers hands them to `neuron`, keeping the method's own cache for
        # every other device by delegating to it.
        def _device_freqs(original):
            def wrapped(self, device):
                if device is not None and device.type == "flagos":
                    return (
                        torch.angle(self.pos_freqs).to(device),
                        torch.angle(self.neg_freqs).to(device),
                    )
                return original(self, device)

            return wrapped

        for cls in (qwenimage.QwenEmbedRope, qwenimage.QwenEmbedLayer3DRope):
            cls._get_device_freqs = _device_freqs(cls._get_device_freqs)
        setattr(qwenimage, _QWENIMAGE_ROPE_INSTALLED_ATTR, True)

    table["flagos"] = flagos_qwenimage_rotary_emb
    return True
