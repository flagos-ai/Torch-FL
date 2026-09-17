"""Exercise common.py's real-valued-RoPE hook without a GCU.

`_select_real_valued_rope` only touches diffusers and whatever torch object it is
handed, so a CPU torch is enough to prove the registration: the two
`_get_device_freqs` methods get wrapped, `ROPE_PER_DEVICE` gains the running
device type, and a second call is a no-op.

The numeric half -- that the wrapped method really returns rotation angles for
the registered type and the complex frequencies for every other one -- calls
``.to(device)``, so it needs a device type this interpreter can construct. That
is decided by trying, not by a table: a CPU-only torch refuses ``gcu`` at the
device-string parse, while the vendor ``torch_gcu`` build accepts it and runs the
same closure on real accelerator tensors. Pass a device *with* an index when the
default card is busy (``QWEN_IMAGE_ROPE_TEST_DEVICE=gcu:7``); only the type half
of the string is what the hook compares.

Run with `QWEN_IMAGE_REAL_ROPE` unset and again set to 1, in each environment.
See /tmp/test_rope_hook.sh.
"""

import os
import sys

import common
import torch
from diffusers.models.transformers import transformer_qwenimage as qwenimage

DEVICE = os.environ.get("QWEN_IMAGE_ROPE_TEST_DEVICE", "gcu")
DEVICE_KIND = DEVICE.partition(":")[0]
enabled = os.environ.get("QWEN_IMAGE_REAL_ROPE", "0") not in ("", "0")

try:
    numeric_device = torch.device(DEVICE)
    numeric = DEVICE_KIND != "meta"
except RuntimeError:
    numeric_device = None
    numeric = False

failures = []


def check(name, condition, detail=""):
    print(f"  {name:52s} {'OK' if condition else 'FAIL'} {detail}")
    if not condition:
        failures.append(name)


before_rope = dict(qwenimage.ROPE_PER_DEVICE)
before_qwen = qwenimage.QwenEmbedRope._get_device_freqs
before_layer = qwenimage.QwenEmbedLayer3DRope._get_device_freqs

common._select_real_valued_rope(torch, DEVICE_KIND)

print(
    f"device {DEVICE}, QWEN_IMAGE_REAL_ROPE={'1' if enabled else 'unset'}, "
    f"numeric={'yes' if numeric else 'no'}"
)

check(
    "ROPE_PER_DEVICE has the device type",
    (DEVICE_KIND in qwenimage.ROPE_PER_DEVICE) == enabled,
    f"keys={sorted(qwenimage.ROPE_PER_DEVICE)}",
)
check(
    "ROPE_PER_DEVICE did not lose an entry",
    set(before_rope) <= set(qwenimage.ROPE_PER_DEVICE),
)
if enabled:
    check(
        "ROPE_PER_DEVICE points at the neuron kernel",
        qwenimage.ROPE_PER_DEVICE[DEVICE_KIND]
        is qwenimage.apply_rotary_emb_qwen_neuron,
    )
check(
    "_get_device_freqs wrapped",
    (qwenimage.QwenEmbedRope._get_device_freqs is not before_qwen) == enabled,
)
check(
    "QwenEmbedLayer3DRope._get_device_freqs wrapped",
    (qwenimage.QwenEmbedLayer3DRope._get_device_freqs is not before_layer) == enabled,
)

if enabled and numeric:

    class Fake:
        """The two attributes `_get_device_freqs` reads, in the shapes it reads."""

        pos_freqs = torch.polar(
            torch.ones(4, 2), torch.arange(8, dtype=torch.float32).reshape(4, 2)
        )
        neg_freqs = pos_freqs.clone()

    angles, _ = qwenimage.QwenEmbedRope._get_device_freqs(Fake(), numeric_device)
    check(
        "closure returns a real tensor",
        not angles.is_complex(),
        f"dtype={angles.dtype} device={angles.device}",
    )
    check(
        "angles equal torch.angle of pos_freqs",
        torch.allclose(angles.cpu(), torch.angle(Fake.pos_freqs)),
    )
    other, _ = qwenimage.QwenEmbedRope._get_device_freqs(Fake(), torch.device("meta"))
    check("closure defers for another device type", other.is_complex())

# Idempotency: a second call must not wrap the wrapper again.
second_qwen = qwenimage.QwenEmbedRope._get_device_freqs
common._select_real_valued_rope(torch, DEVICE_KIND)
check(
    "second call is a no-op", qwenimage.QwenEmbedRope._get_device_freqs is second_qwen
)

print()
if failures:
    print(f"ROPE_HOOK=FAIL {failures}")
    sys.exit(1)
print("ROPE_HOOK=PASS")
