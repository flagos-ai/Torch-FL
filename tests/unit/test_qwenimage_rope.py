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

"""Shared unit coverage for Qwen-Image real-valued rotary embeddings.

The registration supplies the ``flagos`` consumer in ``ROPE_PER_DEVICE`` and
the angle producer used by both ``QwenEmbedRope`` classes. It is exercised on
CPU so every platform runs the same contract.
"""

import importlib.util
import math
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "qwenimage_rope_under_test", ROOT / "torch_fl" / "accelerator" / "qwenimage_rope.py"
)
assert SPEC is not None and SPEC.loader is not None
qwenimage_rope = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qwenimage_rope)
_ROTATION_TABLES = qwenimage_rope._ROTATION_TABLES
_ROTATION_TABLES_MAX = qwenimage_rope._ROTATION_TABLES_MAX
_rotation_table = qwenimage_rope._rotation_table
flagos_qwenimage_rotary_emb = qwenimage_rope.flagos_qwenimage_rotary_emb
patch_diffusers_qwenimage_rope = qwenimage_rope.patch_diffusers_qwenimage_rope

SHAPES = [(1, 4096, 24, 128), (1, 18, 24, 128)]
ROPE_CLASSES = ("QwenEmbedRope", "QwenEmbedLayer3DRope")


def _load_installed_qwenimage(monkeypatch):
    """Load the rope definitions without diffusers' unrelated lazy exports."""

    site_packages = Path(torch.__file__).resolve().parent.parent
    source = (
        site_packages
        / "diffusers"
        / "models"
        / "transformers"
        / "transformer_qwenimage.py"
    )
    if not source.is_file():
        pytest.fail("the shared dependency cohort must install diffusers")
    parent = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(
            "diffusers", site_packages / "diffusers" / "__init__.py"
        )
    )
    parent.__path__ = [str(site_packages / "diffusers")]
    monkeypatch.setitem(sys.modules, "diffusers", parent)
    stubs = {
        "diffusers.configuration_utils": ("ConfigMixin", "register_to_config"),
        "diffusers.loaders": ("FromOriginalModelMixin", "PeftAdapterMixin"),
        "diffusers.utils": ("apply_lora_scale", "logging"),
        "diffusers.utils.torch_utils": (
            "lru_cache_unless_export",
            "maybe_allow_in_graph",
        ),
        "diffusers.models._modeling_parallel": (
            "ContextParallelInput",
            "ContextParallelOutput",
        ),
        "diffusers.models.attention": ("AttentionMixin", "FeedForward"),
        "diffusers.models.attention_dispatch": ("dispatch_attention_fn",),
        "diffusers.models.attention_processor": ("Attention",),
        "diffusers.models.cache_utils": ("CacheMixin",),
        "diffusers.models.embeddings": ("TimestepEmbedding", "Timesteps"),
        "diffusers.models.modeling_outputs": ("Transformer2DModelOutput",),
        "diffusers.models.modeling_utils": ("ModelMixin",),
        "diffusers.models.normalization": (
            "AdaLayerNormContinuous",
            "RMSNorm",
        ),
    }
    for name, exports in stubs.items():
        classes = tuple(type(value, (), {}) for value in exports)
        module = types.SimpleNamespace(**dict(zip(exports, classes)))
        if name.endswith("_modeling_parallel"):
            module.ContextParallelInput = lambda **kwargs: kwargs
            module.ContextParallelOutput = lambda **kwargs: kwargs
        if name == "diffusers.utils":
            module.logging = types.SimpleNamespace(get_logger=lambda *args: mock.Mock())
            module.apply_lora_scale = lambda *args, **kwargs: lambda func: func
        if name.endswith("torch_utils"):
            module.lru_cache_unless_export = lambda *args, **kwargs: lambda func: func
            module.maybe_allow_in_graph = lambda value: value
        if name.endswith("configuration_utils"):
            module.register_to_config = lambda value: value
        monkeypatch.setitem(sys.modules, name, module)
    loader = importlib.machinery.SourceFileLoader(
        "diffusers.models.transformers.transformer_qwenimage", str(source)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, loader.name, module)
    loader.exec_module(module)
    return module


def _rope_classes(module):
    return tuple(
        cls
        for cls in (getattr(module, name, None) for name in ROPE_CLASSES)
        if isinstance(cls, type)
    )


def _angles(batch_seq, head_dim):
    seq = batch_seq[1]
    positions = torch.arange(seq, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.arange(head_dim // 2, dtype=torch.float32).unsqueeze(0)
    return torch.remainder(positions * 0.7 + frequencies * 0.3, 2 * math.pi) - math.pi


def _tensors(shape, dtype):
    generator = torch.Generator().manual_seed(20260920)
    return torch.randn(shape, dtype=dtype, generator=generator)


def _complex_path(x, freqs):
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_rotated * freqs_cis.unsqueeze(1)).flatten(3).type_as(x)


def _neuron_expansion(x, freqs):
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1).unsqueeze(1)
    real, imaginary = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([-imaginary, real], dim=-1).flatten(3)
    return (x.float() * cos + rotated.float() * sin).to(x.dtype)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotation_matches_the_neuron_expansion(shape, dtype):
    x = _tensors(shape, dtype)
    freqs = _angles(shape, shape[-1])

    expected = _neuron_expansion(x, freqs)
    got = flagos_qwenimage_rotary_emb(x, freqs)

    assert got.dtype == dtype
    assert got.shape == expected.shape
    # A vendor's trigonometric implementation may round float32 operands
    # differently; bf16 absorbs that difference at the output.
    tolerance = 1e-3 if dtype == torch.float32 else 4 * torch.finfo(dtype).eps
    assert torch.allclose(
        got.float(), expected.float(), atol=tolerance, rtol=tolerance
    ), f"max|d| = {(got.float() - expected.float()).abs().max().item():.3e}"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotation_matches_the_complex_path(shape, dtype):
    x = _tensors(shape, dtype)
    freqs = _angles(shape, shape[-1])
    got = flagos_qwenimage_rotary_emb(x, freqs)
    tolerance = 1e-4 if dtype == torch.float32 else 4 * torch.finfo(dtype).eps

    assert torch.allclose(
        got.float(), _complex_path(x, freqs).float(), atol=tolerance, rtol=tolerance
    )


def test_rotation_matches_installed_diffusers():
    neuron = _neuron_expansion
    shape = SHAPES[0]
    x = _tensors(shape, torch.bfloat16)
    freqs = _angles(shape, shape[-1])
    expected = neuron(x, freqs)
    got = flagos_qwenimage_rotary_emb(x, freqs)
    assert torch.allclose(got.float(), expected.float(), atol=1e-3, rtol=1e-3)


def test_zero_angle_is_the_identity():
    shape = (1, 5, 2, 8)
    x = _tensors(shape, torch.float32)
    freqs = torch.zeros(shape[1], shape[-1] // 2)
    assert torch.equal(flagos_qwenimage_rotary_emb(x, freqs), x)


def test_rotation_preserves_pair_magnitudes():
    shape = (1, 7, 3, 16)
    x = _tensors(shape, torch.float32)
    got = flagos_qwenimage_rotary_emb(x, _angles(shape, shape[-1]))
    assert torch.allclose(
        got.reshape(*shape[:-1], -1, 2).pow(2).sum(-1),
        x.reshape(*shape[:-1], -1, 2).pow(2).sum(-1),
        atol=1e-5,
        rtol=1e-5,
    )


class TestTheRotationTableIsMemoised:
    @pytest.fixture(autouse=True)
    def empty_cache(self):
        _ROTATION_TABLES.clear()
        yield
        _ROTATION_TABLES.clear()

    def test_a_repeated_operand_reuses_the_tensors(self):
        freqs = _angles(SHAPES[0], SHAPES[0][-1])
        first = _rotation_table(freqs)
        second = _rotation_table(freqs)
        assert first[0] is second[0]
        assert first[1] is second[1]

    def test_a_different_operand_of_the_same_shape_is_not_reused(self):
        shape = SHAPES[0]
        one = _angles(shape, shape[-1])
        other = one.clone()
        other[0, 0] += 1.0
        assert _rotation_table(one)[0] is not _rotation_table(other)[0]
        x = _tensors(shape, torch.bfloat16)
        assert not torch.equal(
            flagos_qwenimage_rotary_emb(x, one),
            flagos_qwenimage_rotary_emb(x, other),
        )

    def test_an_in_place_write_invalidates_the_entry(self):
        shape = SHAPES[0]
        freqs = _angles(shape, shape[-1])
        x = _tensors(shape, torch.bfloat16)
        before = flagos_qwenimage_rotary_emb(x, freqs).clone()
        freqs.add_(0.5)
        assert not torch.equal(flagos_qwenimage_rotary_emb(x, freqs), before)

    def test_the_cache_is_bounded(self):
        shape = SHAPES[1]
        for _ in range(_ROTATION_TABLES_MAX + 3):
            cos, _ = _rotation_table(_angles(shape, shape[-1]))
            assert cos.shape == (shape[1], 1, shape[-1])
        assert len(_ROTATION_TABLES) == _ROTATION_TABLES_MAX


class TestTheRegistration:
    @pytest.fixture
    def qwenimage(self, monkeypatch):
        module = _load_installed_qwenimage(monkeypatch)

        package = sys.modules.get("torch_fl")
        if package is None:
            package = types.ModuleType("torch_fl")
            monkeypatch.setitem(sys.modules, "torch_fl", package)

        def flag(_name):
            return False

        monkeypatch.setattr(
            package, "_env", types.SimpleNamespace(flag=flag), raising=False
        )
        table = module.ROPE_PER_DEVICE
        assert "cuda" in table
        assert hasattr(module, "QwenEmbedLayer3DRope")
        classes = _rope_classes(module)
        assert classes
        original_aliases = {name: getattr(module, name) for name in ROPE_CLASSES}
        original_methods = {cls: cls.__dict__["_get_device_freqs"] for cls in classes}
        original_table = dict(table)
        installed = getattr(module, "_flagos_qwenimage_rope_installed", None)
        table.clear()
        table.update({"cuda": lambda x, f: None, "cpu": object()})
        module._flagos_qwenimage_rope_installed = False
        yield module
        table.clear()
        table.update(original_table)
        for name, cls in original_aliases.items():
            module.__dict__[name] = cls
        for cls, method in original_methods.items():
            cls._get_device_freqs = method
        if installed is None:
            delattr(module, "_flagos_qwenimage_rope_installed")
        else:
            module._flagos_qwenimage_rope_installed = installed
        qwenimage_rope._ENV = None

    def test_registers_the_consumer_and_keeps_the_other_devices(self, qwenimage):
        before = dict(qwenimage.ROPE_PER_DEVICE)
        assert patch_diffusers_qwenimage_rope() is True
        assert qwenimage.ROPE_PER_DEVICE["flagos"] is flagos_qwenimage_rotary_emb
        for device, entry in before.items():
            if device != "flagos":
                assert qwenimage.ROPE_PER_DEVICE[device] is entry
        assert qwenimage._flagos_qwenimage_rope_installed is True

    def test_is_idempotent(self, qwenimage):
        assert patch_diffusers_qwenimage_rope() is True
        wrapped = {cls: cls._get_device_freqs for cls in _rope_classes(qwenimage)}
        assert patch_diffusers_qwenimage_rope() is True
        for cls, method in wrapped.items():
            assert cls._get_device_freqs is method

    def test_returns_false_when_the_table_is_missing(self, qwenimage, monkeypatch):
        monkeypatch.delattr(qwenimage, "ROPE_PER_DEVICE")
        assert patch_diffusers_qwenimage_rope() is False

    def test_honours_the_disable_switch(self, qwenimage, monkeypatch):
        package = types.ModuleType("torch_fl")
        package.__path__ = [str(ROOT / "torch_fl")]
        package.__file__ = str(ROOT / "torch_fl" / "__init__.py")
        monkeypatch.setitem(sys.modules, "torch_fl", package)
        monkeypatch.setattr(qwenimage_rope, "_ENV", None)
        monkeypatch.setenv("FLAGOS_DISABLE_QWENIMAGE_ROPE", "1")
        table = dict(qwenimage.ROPE_PER_DEVICE)
        methods = {cls: cls._get_device_freqs for cls in _rope_classes(qwenimage)}
        assert patch_diffusers_qwenimage_rope() is False
        assert set(qwenimage.ROPE_PER_DEVICE) == set(table)
        for device, entry in table.items():
            assert qwenimage.ROPE_PER_DEVICE[device] is entry
        for cls, method in methods.items():
            assert cls._get_device_freqs is method

    @pytest.mark.parametrize("cls_name", ROPE_CLASSES)
    def test_hands_the_flagos_device_angles(self, qwenimage, monkeypatch, cls_name):
        patch_diffusers_qwenimage_rope()
        rope = getattr(qwenimage, cls_name)(theta=10000, axes_dim=[16, 56, 56])
        device = type("FakeFlagosDevice", (), {"type": "flagos"})()
        moved = []

        class _Angles:
            def __init__(self, angle):
                self.angle = angle

            def to(self, target):
                moved.append(target)
                return self.angle

        pos_freqs = torch.tensor([0.1 + 0.2j], dtype=torch.complex64)
        neg_freqs = torch.tensor([-0.3 + 0.4j], dtype=torch.complex64)
        monkeypatch.setattr(rope, "pos_freqs", pos_freqs)
        monkeypatch.setattr(rope, "neg_freqs", neg_freqs)
        original_angle = torch.angle
        monkeypatch.setattr(
            torch, "angle", lambda value: _Angles(original_angle(value))
        )
        pos, neg = rope._get_device_freqs(device)
        assert torch.equal(pos, original_angle(pos_freqs))
        assert torch.equal(neg, original_angle(neg_freqs))
        assert moved == [device, device]

        got = rope._get_device_freqs(torch.device("cpu"))
        assert got[0] is rope.pos_freqs
        assert got[1] is rope.neg_freqs

    @pytest.mark.parametrize("cls_name", ROPE_CLASSES)
    def test_leaves_every_other_device_alone(self, qwenimage, cls_name):
        patch_diffusers_qwenimage_rope()
        rope = getattr(qwenimage, cls_name)(theta=10000, axes_dim=[16, 56, 56])
        got = rope._get_device_freqs(torch.device("cpu"))
        assert got[0] is rope.pos_freqs
        assert got[1] is rope.neg_freqs
        assert got[0].is_complex()
