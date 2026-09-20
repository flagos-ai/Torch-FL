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

"""Unit coverage for the Qwen-Image rotation torch_fl installs on GCU.

``patch_diffusers_qwenimage_rope`` installs two halves that have to agree, and
both are correctness-critical in a way an on-hardware benchmark would not notice:
a wrong rotation still runs and still produces an image.

* the *consumer*, ``flagos_qwenimage_rotary_emb``, which has to apply the same
  rotation as ``diffusers``' ``apply_rotary_emb_qwen_neuron`` -- same values, same
  order, same layout -- because it exists only to write those values without the
  stride-0 broadcast that function's ``repeat_interleave`` lowers to;
* the *operand*, the ``_get_device_freqs`` override that hands that consumer
  rotation angles for the ``flagos`` device, which is what the complex
  exponential path is replaced by.

The consumer is checked against both references: the neuron expansion it
replaces, bit for bit, and the complex exponential path it makes unnecessary,
which is the claim in its docstring. These run on any host -- the functions are
elementwise, so CPU tensors exercise them end to end. The registration itself
needs ``diffusers`` and is skipped without it.

The on-hardware measurements are in
``tests/manual/qwen_image_2512/README.md`` and
``docs/reference/operator-support.md``.
"""

import math

import pytest
import torch

from torch_fl.accelerator.gcu._gcu_compat import (
    flagos_qwenimage_rotary_emb,
    patch_diffusers_qwenimage_rope,
)

# The shapes the model uses: 1024x1024 is 4096 image tokens, the text stream is
# a couple of dozen, 24 heads of head_dim 128, batch 1.
SHAPES = [
    (1, 4096, 24, 128),
    (1, 18, 24, 128),
]


def _angles(batch_seq, head_dim):
    """The operand ``diffusers`` hands the neuron expansion: [S, D // 2].

    A real rotation state, not a random one: ``cos``/``sin`` of a random angle
    would make a wrong expansion agree with the right one more often.
    """
    seq = batch_seq[1]
    positions = torch.arange(seq, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.arange(head_dim // 2, dtype=torch.float32).unsqueeze(0)
    return torch.remainder(positions * 0.7 + frequencies * 0.3, 2 * math.pi) - math.pi


def _tensors(shape, dtype):
    generator = torch.Generator().manual_seed(20260920)
    return torch.randn(shape, dtype=dtype, generator=generator)


def _neuron_expansion(x, freqs):
    """``diffusers.models.transformers.transformer_qwenimage``, verbatim.

    Not imported from ``diffusers`` so the layout this replaces stays pinned here
    even when ``diffusers`` is absent or changes; the diffusers copy is compared
    separately below when it is installed.
    """
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1).unsqueeze(1)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def _complex_path(x, freqs):
    """``apply_rotary_emb_qwen(..., use_real=False)``, the entry flagos falls to."""
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_rotated * freqs_cis.unsqueeze(1)).flatten(3)
    return x_out.type_as(x)


class TestTheRotationIsTheSameRotation:
    """The expansion has to be the one it replaces, not merely a rotation."""

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matches_the_neuron_expansion_bit_for_bit(self, shape, dtype):
        x = _tensors(shape, dtype)
        freqs = _angles(shape, shape[-1])

        expected = _neuron_expansion(x, freqs)
        got = flagos_qwenimage_rotary_emb(x, freqs)

        assert got.dtype == dtype
        assert got.shape == expected.shape
        assert torch.equal(got, expected), (
            f"max|d| = {(got.float() - expected.float()).abs().max().item():.3e}"
        )

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matches_the_complex_path_it_replaces(self, shape, dtype):
        """The reference is the multiply, not another real-valued spelling of it.

        Not bit-exact, and it does not need to be: the complex multiply forms the
        two products of a pair in a different order and the two spellings can land
        one bf16 ulp apart. The bit-for-bit contract is with the neuron expansion
        above, which is what this replaces.
        """
        x = _tensors(shape, dtype)
        freqs = _angles(shape, shape[-1])

        expected = _complex_path(x, freqs)
        got = flagos_qwenimage_rotary_emb(x, freqs)
        tolerance = 1e-4 if dtype == torch.float32 else 4 * torch.finfo(dtype).eps

        assert torch.allclose(
            got.float(), expected.float(), atol=tolerance, rtol=tolerance
        ), f"max|d| = {(got.float() - expected.float()).abs().max().item():.3e}"

    def test_matches_the_installed_diffusers_copy(self):
        """The pinned expansion above has to be the expansion diffusers ships."""
        qwenimage = pytest.importorskip(
            "diffusers.models.transformers.transformer_qwenimage"
        )
        neuron = getattr(qwenimage, "apply_rotary_emb_qwen_neuron", None)
        if neuron is None:
            pytest.skip("this diffusers has no neuron rotation")

        shape = SHAPES[0]
        x = _tensors(shape, torch.bfloat16)
        freqs = _angles(shape, shape[-1])

        assert torch.equal(flagos_qwenimage_rotary_emb(x, freqs), neuron(x, freqs))

    def test_a_zero_angle_is_the_identity(self):
        """Every feature pair shares one angle, and angle 0 has to rotate by none."""
        shape = (1, 5, 2, 8)
        x = _tensors(shape, torch.float32)
        freqs = torch.zeros(shape[1], shape[-1] // 2)

        assert torch.equal(flagos_qwenimage_rotary_emb(x, freqs), x)

    def test_rotation_preserves_the_pair_magnitudes(self):
        """A rotation moves mass between a pair (2k, 2k+1) and never past it."""
        shape = (1, 7, 3, 16)
        x = _tensors(shape, torch.float32)
        freqs = _angles(shape, shape[-1])

        got = flagos_qwenimage_rotary_emb(x, freqs)
        pairs = got.reshape(*shape[:-1], -1, 2)
        source = x.reshape(*shape[:-1], -1, 2)

        assert torch.allclose(
            pairs.pow(2).sum(-1), source.pow(2).sum(-1), atol=1e-5, rtol=1e-5
        )


class TestTheRegistration:
    """The table entry and the operand producer, installed together."""

    @pytest.fixture
    def qwenimage(self):
        module = pytest.importorskip(
            "diffusers.models.transformers.transformer_qwenimage"
        )
        if not isinstance(getattr(module, "ROPE_PER_DEVICE", None), dict):
            pytest.skip("diffusers has no Qwen-Image rope table")
        # The patch mutates module state; put it back so the rest of the session
        # (and any other diffusers test) sees the stock tables.
        table = dict(module.ROPE_PER_DEVICE)
        freq_methods = {
            cls: cls._get_device_freqs
            for cls in (module.QwenEmbedRope, module.QwenEmbedLayer3DRope)
        }
        installed = getattr(module, "_flagos_qwenimage_rope_installed", None)
        yield module
        module.ROPE_PER_DEVICE.clear()
        module.ROPE_PER_DEVICE.update(table)
        for cls, method in freq_methods.items():
            cls._get_device_freqs = method
        if installed is None:
            delattr(module, "_flagos_qwenimage_rope_installed")
        else:
            module._flagos_qwenimage_rope_installed = installed

    def test_registers_the_consumer_and_keeps_the_other_devices(self, qwenimage):
        before = dict(qwenimage.ROPE_PER_DEVICE)

        assert patch_diffusers_qwenimage_rope() is True

        assert qwenimage.ROPE_PER_DEVICE["flagos"] is flagos_qwenimage_rotary_emb
        for device, entry in before.items():
            assert qwenimage.ROPE_PER_DEVICE[device] is entry
        assert qwenimage._flagos_qwenimage_rope_installed is True

    def test_is_idempotent(self, qwenimage):
        assert patch_diffusers_qwenimage_rope() is True

        wrapped = {
            cls: cls._get_device_freqs
            for cls in (qwenimage.QwenEmbedRope, qwenimage.QwenEmbedLayer3DRope)
        }

        assert patch_diffusers_qwenimage_rope() is True

        for cls, method in wrapped.items():
            assert cls._get_device_freqs is method, "the operand half was wrapped twice"

    def test_returns_false_when_the_table_is_missing(self, qwenimage, monkeypatch):
        monkeypatch.delattr(qwenimage, "ROPE_PER_DEVICE")

        assert patch_diffusers_qwenimage_rope() is False

    def test_honours_the_disable_switch(self, qwenimage, monkeypatch):
        """``FLAGOS_DISABLE_QWENIMAGE_ROPE`` is the A/B's off leg."""
        monkeypatch.setenv("FLAGOS_DISABLE_QWENIMAGE_ROPE", "1")
        table = dict(qwenimage.ROPE_PER_DEVICE)
        methods = {
            cls: cls._get_device_freqs
            for cls in (qwenimage.QwenEmbedRope, qwenimage.QwenEmbedLayer3DRope)
        }

        assert patch_diffusers_qwenimage_rope() is False

        assert set(qwenimage.ROPE_PER_DEVICE) == set(table)
        for device, entry in table.items():
            assert qwenimage.ROPE_PER_DEVICE[device] is entry
        for cls, method in methods.items():
            assert cls._get_device_freqs is method

    @pytest.mark.parametrize("cls_name", ["QwenEmbedRope", "QwenEmbedLayer3DRope"])
    def test_hands_the_flagos_device_angles(self, qwenimage, monkeypatch, cls_name):
        patch_diffusers_qwenimage_rope()
        rope = getattr(qwenimage, cls_name)(theta=10000, axes_dim=[16, 56, 56])
        device = type("FakeFlagosDevice", (), {"type": "flagos"})()

        # A flagos device cannot be constructed off hardware, so the move is
        # intercepted: what is under test is which operand the method returns for
        # that device, not the transfer.
        class _Angles:
            def __init__(self, tag):
                self.tag = tag

            def to(self, target):
                return (self.tag, target)

        class _Tagged:
            def __init__(self, tag):
                self.tag = tag

        monkeypatch.setattr(torch, "angle", lambda tensor: _Angles(tensor.tag))

        rope.pos_freqs = _Tagged("pos")
        rope.neg_freqs = _Tagged("neg")

        assert rope._get_device_freqs(device) == (("pos", device), ("neg", device))

    @pytest.mark.parametrize("cls_name", ["QwenEmbedRope", "QwenEmbedLayer3DRope"])
    def test_leaves_every_other_device_alone(self, qwenimage, cls_name):
        patch_diffusers_qwenimage_rope()
        rope = getattr(qwenimage, cls_name)(theta=10000, axes_dim=[16, 56, 56])

        got = rope._get_device_freqs(torch.device("cpu"))

        # The stock method's own cache and its complex operands, untouched.
        assert got[0] is rope.pos_freqs
        assert got[1] is rope.neg_freqs
        assert got[0].is_complex()
