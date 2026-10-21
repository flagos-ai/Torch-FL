"""Qwen-Image rotary embedding shared by the non-complex accelerator paths."""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

# The angle expansion is a pure function of ``freqs``, and the model asks for
# the same 62 angle tensors 240 times in one transformer forward. Holding the
# tensor by reference keeps its identity stable; ``_version`` catches in-place
# writes. Four entries retain the reuse the model's call pattern needs.
_ROTATION_TABLES = []  # [(freqs, version, cos, sin)], oldest first
_ROTATION_TABLES_MAX = 4
_ENV = None


def _env():
    """Load torch_fl's stdlib-only environment reader without importing torch_fl."""
    global _ENV

    if _ENV is None:
        package = sys.modules.get("torch_fl")
        found = getattr(package, "_env", None) if package is not None else None
        if found is not None:
            _ENV = found
        else:
            loader = importlib.machinery.SourceFileLoader(
                "torch_fl_env_for_qwenimage",
                str(Path(__file__).resolve().parents[1] / "_env.py"),
            )
            spec = importlib.util.spec_from_loader(loader.name, loader)
            if spec is None:
                return None
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            _ENV = module
    return _ENV


def _rotation_table(freqs):
    """The (cos, sin) pair `flagos_qwenimage_rotary_emb` multiplies by."""
    import torch

    version = freqs._version
    for held, held_version, cos, sin in reversed(_ROTATION_TABLES):
        if held is freqs and held_version == version:
            return cos, sin

    # Keep diffusers' expansion order: each angle covers both values of one
    # rotated pair. Stacking adjacent angles changes the result for real-valued
    # inputs, whose final component is paired by dimension rather than by axis.
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1).unsqueeze(1)

    if len(_ROTATION_TABLES) >= _ROTATION_TABLES_MAX:
        del _ROTATION_TABLES[0]
    _ROTATION_TABLES.append((freqs, version, cos, sin))
    return cos, sin


def flagos_qwenimage_rotary_emb(x, freqs):
    """Qwen-Image rotation for a device without a complex dtype.

    Uses the same pair layout and float32 multiply/add as diffusers'
    ``apply_rotary_emb_qwen_neuron``. The angle tables are memoised, and the
    guard is the angle tensor's identity and ``_version``. The values can differ
    in the last bits when a vendor's trigonometric implementation does.
    """
    import torch

    cos, sin = _rotation_table(freqs)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def patch_diffusers_qwenimage_rope() -> bool:
    """Register the flagos Qwen-Image rotation, its angles and consumer.

    ``ROPE_PER_DEVICE`` selects the consumer for each device type, while
    ``QwenEmbedRope._get_device_freqs`` and ``QwenEmbedLayer3DRope``'s copy of
    it produce its operands. An unlisted device takes diffusers' complex path.
    That path is slow on GCU and unavailable on Ascend, so both halves are
    installed before the operand producer is first cached. Other devices keep
    their existing consumer and delegated operand method.

    Safe to call repeatedly. Returns ``False`` when diffusers, the rope table or
    the registration is unavailable, or when ``FLAGOS_DISABLE_QWENIMAGE_ROPE``
    opts out.
    """
    import torch

    environment = _env()
    if environment is not None and environment.flag("FLAGOS_DISABLE_QWENIMAGE_ROPE"):
        return False
    qwenimage = sys.modules.get("diffusers.models.transformers.transformer_qwenimage")
    if qwenimage is None:
        if importlib.util.find_spec("diffusers") is None:
            return False
        try:
            from diffusers.models.transformers import transformer_qwenimage as qwenimage
        except ImportError:
            return False

    table = getattr(qwenimage, "ROPE_PER_DEVICE", None)
    if not isinstance(table, dict) or "cuda" not in table:
        return False

    if not getattr(qwenimage, "_flagos_qwenimage_rope_installed", False):

        def _device_freqs(original):
            def wrapped(self, device):
                if device is not None and device.type == "flagos":
                    return (
                        torch.angle(self.pos_freqs).to(device),
                        torch.angle(self.neg_freqs).to(device),
                    )
                return original(self, device)

            return wrapped

        classes = [
            getattr(qwenimage, name, None)
            for name in ("QwenEmbedRope", "QwenEmbedLayer3DRope")
        ]
        if not any(
            isinstance(cls, type) and "_get_device_freqs" in cls.__dict__
            for cls in classes
        ):
            return False
        for cls in classes:
            if isinstance(cls, type) and "_get_device_freqs" in cls.__dict__:
                cls._get_device_freqs = _device_freqs(cls._get_device_freqs)
        qwenimage._flagos_qwenimage_rope_installed = True

    table["flagos"] = flagos_qwenimage_rotary_emb
    return True
