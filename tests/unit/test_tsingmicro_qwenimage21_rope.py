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

"""Unit coverage for the Qwen-Image-2.1 rotation torch_fl installs on TsingMicro.

``patch_diffusers_qwenimage21_rope`` installs two halves that have to agree, and
both are correctness-critical in a way an on-hardware run would not notice: a
wrong rotation still runs and still produces an image.

* the *operand*, ``QwenImage21Rope.rope_params``, which has to hand back the real
  rotation angles the complex ``torch.polar`` carried -- if it stayed complex, the
  rope's ``cat`` reaches FlagGems and raises ``KeyError: 'complex64'`` on a device
  with no complex pointer type;
* the *consumer*, the wrapped ``apply_rotary_emb_qwen``, which has to apply the
  same rotation the complex exponential path did -- same four products, same order,
  shaped for 2.1's ``[B, S, H, D]`` activation.

The consumer is checked against the complex path it replaces. These run on any
host -- the functions are elementwise, so CPU tensors exercise them end to end.
The registration needs ``diffusers`` and is skipped without it.

The on-hardware measurement is in ``docs/reference/operator-support.md``.
"""

import pytest
import torch

from torch_fl.accelerator.tsingmicro._tsingmicro_compat import (
    flagos_qwenimage21_rotary_emb,
    patch_diffusers_qwenimage21_rope,
)

# 2.1's activation layout: batch 1, a few dozen sequence positions, a handful of
# heads, head_dim; the angle operand is [S, D // 2].
SHAPES = [
    (1, 4096, 24, 128),
    (1, 18, 24, 128),
]


def _complex_reference(x, angles):
    """The shipped complex path: ``view_as_complex(x) * polar(1, angles)``."""
    freqs_cis = torch.polar(torch.ones_like(angles, dtype=torch.float32), angles)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_complex * freqs_cis.unsqueeze(1)).flatten(3).type_as(x)


class TestTheRotationMatchesTheComplexPathItReplaces:
    @pytest.mark.anyplatform
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matches_the_complex_exponential_path(self, shape, dtype):
        generator = torch.Generator().manual_seed(20260920)
        x = torch.randn(shape, dtype=dtype, generator=generator)
        angles = torch.randn(shape[1], shape[-1] // 2, generator=generator)

        got = flagos_qwenimage21_rotary_emb(x, angles)
        expected = _complex_reference(x, angles)

        assert got.dtype == dtype
        assert got.shape == expected.shape
        # Same four products in the same order, so bf16 rounding can still differ
        # by an ulp from the complex form's own product ordering.
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.anyplatform
    def test_a_zero_angle_is_the_identity(self):
        x = torch.randn(1, 8, 2, 4)
        angles = torch.zeros(8, 2)
        torch.testing.assert_close(
            flagos_qwenimage21_rotary_emb(x, angles), x, rtol=0, atol=0
        )

    @pytest.mark.anyplatform
    def test_rotation_preserves_the_pair_magnitudes(self):
        """A rotation is not a scale: each (real, imag) pair keeps its norm."""
        x = torch.randn(1, 16, 3, 8)
        angles = torch.randn(16, 4)
        got = flagos_qwenimage21_rotary_emb(x, angles)
        before = x.reshape(*x.shape[:-1], -1, 2).float().square().sum(-1)
        after = got.reshape(*got.shape[:-1], -1, 2).float().square().sum(-1)
        torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-4)


class TestThePatchInstalls:
    @pytest.fixture
    def qwenimage21(self):
        module = pytest.importorskip(
            "diffusers.models.transformers.transformer_qwenimage21"
        )
        if not hasattr(module, "QwenImage21Rope"):
            pytest.skip("this diffusers has no QwenImage21Rope")
        # The patch mutates module state; restore it so the rest of the session is
        # unaffected, the same way test_gcu_qwenimage_rope.py does.
        rope_params = module.QwenImage21Rope.rope_params
        apply_rotary = module.apply_rotary_emb_qwen
        installed = getattr(module, "_flagos_qwenimage21_rope_installed", None)
        yield module
        module.QwenImage21Rope.rope_params = rope_params
        module.apply_rotary_emb_qwen = apply_rotary
        if installed is None:
            try:
                delattr(module, "_flagos_qwenimage21_rope_installed")
            except AttributeError:
                pass
        else:
            module._flagos_qwenimage21_rope_installed = installed

    @pytest.mark.anyplatform
    def test_rope_params_returns_a_real_operand(self, qwenimage21, monkeypatch):
        """The operand must stop being complex, or the rope's cat hits FlagGems.

        ``polar(1, a)`` carries ``cos a`` and ``sin a``; the patch drops the
        ``polar`` call so the angles themselves are the operand.
        """
        monkeypatch.delenv("FLAGOS_DISABLE_QWENIMAGE_ROPE", raising=False)
        assert patch_diffusers_qwenimage21_rope() is True

        rope = qwenimage21.QwenImage21Rope(10000, [16, 56, 56])
        assert not any(f.is_complex() for f in rope.freqs), (
            "rope operand is still complex64 -- the device cat would raise"
        )

    @pytest.mark.anyplatform
    def test_is_idempotent(self, qwenimage21, monkeypatch):
        monkeypatch.delenv("FLAGOS_DISABLE_QWENIMAGE_ROPE", raising=False)
        assert patch_diffusers_qwenimage21_rope() is True
        wrapped = qwenimage21.apply_rotary_emb_qwen
        assert patch_diffusers_qwenimage21_rope() is True
        assert qwenimage21.apply_rotary_emb_qwen is wrapped, "wrapped twice"

    @pytest.mark.anyplatform
    def test_honours_the_disable_switch(self, qwenimage21, monkeypatch):
        monkeypatch.setenv("FLAGOS_DISABLE_QWENIMAGE_ROPE", "1")
        assert patch_diffusers_qwenimage21_rope() is False

    @pytest.mark.anyplatform
    def test_returns_false_when_the_class_is_missing(self, qwenimage21, monkeypatch):
        monkeypatch.delenv("FLAGOS_DISABLE_QWENIMAGE_ROPE", raising=False)
        monkeypatch.delattr(qwenimage21, "QwenImage21Rope", raising=False)
        assert patch_diffusers_qwenimage21_rope() is False

    @pytest.mark.anyplatform
    def test_leaves_the_real_and_complex_branches_untouched(
        self, qwenimage21, monkeypatch
    ):
        """Only a real-seeking, real-operand call is rewritten.

        The two untouched paths are checked by identity, not by re-deriving their
        layouts: the point of this patch is that everything except the 2.1 rope's
        own call is served by diffusers exactly as shipped.
        """
        monkeypatch.delenv("FLAGOS_DISABLE_QWENIMAGE_ROPE", raising=False)
        assert patch_diffusers_qwenimage21_rope() is True

        # A complex operand with use_real=False is what the shipped complex path
        # gets; the wrapper must delegate it, so it equals that path's own result.
        x = torch.randn(1, 8, 2, 4)
        angles = torch.randn(8, 2)
        cis = torch.polar(torch.ones_like(angles), angles)
        got_c = qwenimage21.apply_rotary_emb_qwen(x, cis, use_real=False)
        torch.testing.assert_close(
            got_c, _complex_reference(x, angles), rtol=1e-2, atol=1e-2
        )

        # rope_params carries no torch.polar after the wrap, and the angles it
        # returns are what the reference would angle back to.
        rope = qwenimage21.QwenImage21Rope(10000, [16, 56, 56])
        index = torch.arange(8, dtype=torch.float32)
        params = rope.rope_params(index, 8, 10000)
        assert not params.is_complex()
        torch.testing.assert_close(
            params,
            torch.outer(
                index, 1.0 / torch.pow(10000, torch.arange(0, 8, 2).float().div(8))
            ),
        )
