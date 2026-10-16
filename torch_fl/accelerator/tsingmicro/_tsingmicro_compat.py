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

"""Qwen-Image-2.1 rotary embedding for the flagos device.

``QwenImage21Rope`` builds its rotation operand as a **complex64** tensor
(``torch.polar`` in ``rope_params``), moves it to the device, then advanced-indexes
it and concatenates the three axes. All of those route to FlagGems on TsingMicro,
whose Triton kernels have no complex pointer type:

    File "diffusers/models/transformers/transformer_qwenimage21.py", line 710
        return torch.cat([self.freqs[0][frame_index], ...], dim=-1)
    File ".../flag_gems/runtime/backend/_tsingmicro/ops/cat.py", line 108
    File ".../triton/_utils.py", line 105, in canonicalize_dtype
    KeyError: 'complex64'

Measured on a TX8 which complex ops survive: ``polar``, ``view_as_complex``,
``view_as_real``, ``angle``, ``real`` and a plain complex ``mul`` all work, but
``cat``, ``stack`` and the rope's own broadcast multiply
(``freqs.unsqueeze(1) * x``, the ``use_real=False`` consumer) do not. So this is
not a per-op gap to patch: the complex tensor has to stop reaching the device.

Two halves, and both are required:

* the **operand** -- ``QwenImage21Rope.rope_params`` is wrapped to return the raw
  rotation *angles* instead of ``torch.polar(ones, angles)``. Since
  ``polar(1, a) == cos a + i sin a``, the angles are exactly the values the complex
  tensor carried; dropping the ``polar`` call leaves ``self.freqs`` real for
  ``__init__`` and for ``forward``'s ``.to(device)``, so nothing complex is ever
  moved onto the device.
* the **consumer** -- ``apply_rotary_emb_qwen`` is wrapped so a ``use_real=False``
  call handed real angles is served by a real multiply. The shipped
  ``use_real=True`` branch cannot be reused: it broadcasts ``cos[None, None]``,
  which is flux's heads-before-sequence layout, not 2.1's ``[B, S, H, D]``.

This is not a numerical shortcut. ``freqs_cis`` was ``polar(1, a)``, so its real
and imaginary parts *are* ``cos a`` and ``sin a``; the rotation is the same four
products in the same order. Measured against the complex
``view_as_complex(x) * polar(1, a)`` path: ``max|d| == 2.4e-07`` -- one fp32 ulp.

The Qwen-Image-2512 model solves the same problem in
``torch_fl.accelerator.{ascend,gcu}._*_compat.patch_diffusers_qwenimage_rope``, but
that path keys off the 2512 module's ``ROPE_PER_DEVICE`` table and its
``QwenEmbedRope`` classes, none of which the 2.1 module has -- the 2.1 call site
names ``use_real=False`` outright. Hence a separate patch for a separate module.
"""

import importlib.util

_DISABLE = "FLAGOS_DISABLE_QWENIMAGE_ROPE"

#: Module flag on ``transformer_qwenimage21`` marking the operand+consumer wrap as
#: installed. Mirrors ``_flagos_qwenimage_rope_installed`` on the 2512 module.
_INSTALLED_ATTR = "_flagos_qwenimage21_rope_installed"


def flagos_qwenimage21_rotary_emb(x, freqs):
    """Qwen-Image-2.1 rotary embedding from real rotation angles.

    The consumer of the real operand ``rope_params`` now produces. Writes the same
    four products the complex branch's real and imaginary parts carry::

        real:  xr*cos - xi*sin
        imag:  xi*cos + xr*sin

    Shaped for 2.1's ``[B, S, H, D]`` activation with a ``[S, D]`` angle operand,
    so ``cos``/``sin`` gain a head axis with ``unsqueeze(1)``. Not a numerical
    shortcut: the values and their order are the complex path's, asserted in
    ``tests/unit/test_tsingmicro_qwenimage21_rope.py``.
    """
    import torch

    cos = torch.cos(freqs).unsqueeze(1)
    sin = torch.sin(freqs).unsqueeze(1)
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    out = torch.stack(
        [x_real * cos - x_imag * sin, x_imag * cos + x_real * sin], dim=-1
    )
    return out.flatten(-2).type_as(x)


def _wrap_rope_params(original):
    """``QwenImage21Rope.rope_params`` -> the real angles, no ``torch.polar``."""

    def wrapped(self, index, dim, theta=10000):
        import torch

        return torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
        )

    return wrapped


def _wrap_apply_rotary(original):
    """``apply_rotary_emb_qwen`` -> a real rotation for a real operand.

    Only a call that is *both* real-seeking (``use_real=False``) and handed a
    real operand is rewritten; one already on the real branch, or handed a complex
    operand, is delegated unchanged so the untouched path is bit-for-bit what
    diffusers shipped.
    """

    def wrapped(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
        import torch

        if use_real or torch.is_complex(freqs_cis):
            return original(
                x,
                freqs_cis,
                use_real=use_real,
                use_real_unbind_dim=use_real_unbind_dim,
            )
        return flagos_qwenimage21_rotary_emb(x, freqs_cis)

    return wrapped


def patch_diffusers_qwenimage21_rope() -> bool:
    """Point the Qwen-Image-2.1 rotation at real angles on a flagos device.

    Safe to call more than once, and a no-op when diffusers is absent, has no
    Qwen-Image-2.1 transformer, or ``FLAGOS_DISABLE_QWENIMAGE_ROPE`` is set --
    the last of which restores the complex path is how the A/B is measured.
    Returns whether the patch is in place.

    The 2.1 classes are unreleased (diffusers 0.41.0 here), so every patch point
    is ``hasattr``-checked and a build without it returns ``False`` rather than
    raising.
    """
    import torch  # noqa: F401  - fail fast if torch itself is absent

    from torch_fl import _env

    if _env.flag(_DISABLE):
        return False
    if importlib.util.find_spec("diffusers") is None:
        return False
    try:
        from diffusers.models.transformers import (
            transformer_qwenimage21 as qwenimage21,
        )
    except ImportError:
        return False

    rope_cls = getattr(qwenimage21, "QwenImage21Rope", None)
    if rope_cls is None or not hasattr(rope_cls, "rope_params"):
        return False
    if not hasattr(qwenimage21, "apply_rotary_emb_qwen"):
        return False

    if not getattr(qwenimage21, _INSTALLED_ATTR, False):
        rope_cls.rope_params = _wrap_rope_params(rope_cls.rope_params)
        qwenimage21.apply_rotary_emb_qwen = _wrap_apply_rotary(
            qwenimage21.apply_rotary_emb_qwen
        )
        setattr(qwenimage21, _INSTALLED_ATTR, True)

    return True
