"""Exercise common.py's real-valued-RoPE hook without a GCU.

`_select_real_valued_rope` only touches diffusers and whatever torch object it is
handed, so a CPU torch is enough to prove the registration: the two
`_get_device_freqs` methods get wrapped, `ROPE_PER_DEVICE` gains the running
device type, and a second call is a no-op.

**The interpreter decides which of two installers is under test.** In the flagos
environment `import torch_fl` has already called `patch_diffusers_qwenimage_rope`,
which wraps the same two methods for the `flagos` device type and registers a
cheaper expansion of the same rotation for it; `_select_real_valued_rope` sees
that and deliberately does not wrap a second time or overwrite the entry. So the
checks here are phrased per installer rather than per environment: what ends up
on the class is required to be a wrapper from one factory or the other, the
table entry this function adds is required to be the neuron kernel only when
this function is what added it, and the plugin's pair and entry are required to
come through this call untouched. Two of the arms are only reachable on one
interpreter -- the plugin's pair exists only where `torch_fl` was imported, and
it holds the `flagos` device type only -- and the run prints which one it found
so a skipped check is never mistaken for a passing one. What the plugin installs
is also covered, without an accelerator, by
`tests/unit/test_gcu_qwenimage_rope.py`.

The numeric half -- that the wrapped method really returns rotation angles for
the registered type and the complex frequencies for every other one -- calls
``.to(device)``, so it needs a device type this interpreter can construct. That
is decided by trying, not by a table: a CPU-only torch refuses ``gcu`` at the
device-string parse, while the vendor ``torch_gcu`` build accepts it and runs the
same closure on real accelerator tensors. Pass a device *with* an index when the
default card is busy (``QWEN_IMAGE_ROPE_TEST_DEVICE=gcu:7``); only the type half
of the string is what the hook compares.

Run with `QWEN_IMAGE_REAL_ROPE` unset and again set to 1, for each device kind
that matters: `flagos` (the plugin's installer, and the only kind that pulls
`torch_fl` in), `gcu` and `cpu` (this file's own installer). See
/tmp/test_rope_hook.sh.
"""

import os
import sys

import common
from diffusers.models.transformers import transformer_qwenimage as qwenimage

DEVICE = os.environ.get("QWEN_IMAGE_ROPE_TEST_DEVICE", "gcu")
DEVICE_KIND = DEVICE.partition(":")[0]
enabled = os.environ.get("QWEN_IMAGE_REAL_ROPE", "0") not in ("", "0")


# `flagos` is the one device kind that also pulls `torch_fl` in, and torch_fl's
# GCU branch is what installs the plugin's pair -- so the device kind asked for
# here is what decides which of the two installers this run is looking at. Any
# other kind imports torch alone and exercises common.py's own installer.
#
# Not `common.import_torch`, which is otherwise how this directory imports torch:
# it calls `_select_real_valued_rope` on the way out, so the install this file
# exists to measure would already have happened before the snapshot below. The
# conditional import order is load-bearing, and putting it in a function is what
# keeps it from reading as a stray import in the middle of the module.
def import_torch():
    """Import torch, and torch_fl first on the device kind that has a plugin."""

    if DEVICE_KIND == "flagos":
        import torch_fl  # noqa: F401  - must precede `import torch`
    import torch

    return torch


torch = import_torch()

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


def from_a_factory(method):
    """Whether `method` is a factory's wrapper, not diffusers' own method.

    Identity against a snapshot cannot answer this on a flagos interpreter,
    where the plugin's pair is already installed by the time the snapshot is
    taken; the shape of what is on the class can, and it needs no device.
    `_gcu_compat._device_freqs` and `common._get_device_freqs` both hand back a
    nested `wrapped` and neither applies `functools.wraps`, so the qualname
    ends in `_device_freqs.<locals>.wrapped` for both, while diffusers' own
    method -- behind an lru_cache either way -- keeps its class-qualified name.
    """
    return getattr(method, "__qualname__", "").endswith(
        "_device_freqs.<locals>.wrapped"
    )


before_rope = dict(qwenimage.ROPE_PER_DEVICE)
before_qwen = qwenimage.QwenEmbedRope._get_device_freqs
before_layer = qwenimage.QwenEmbedLayer3DRope._get_device_freqs
# Whether torch_fl's GCU branch installed its own pair before this script ran.
# Read here, not after the call: what the call does depends on it.
by_plugin = bool(getattr(qwenimage, "_flagos_qwenimage_rope_installed", False))
had_key = DEVICE_KIND in before_rope

common._select_real_valued_rope(torch, DEVICE_KIND)

print(
    f"device {DEVICE}, QWEN_IMAGE_REAL_ROPE={'1' if enabled else 'unset'}, "
    f"numeric={'yes' if numeric else 'no'}, "
    f"torch_fl installed the pair={by_plugin}"
)

check(
    "ROPE_PER_DEVICE has the device type",
    (DEVICE_KIND in qwenimage.ROPE_PER_DEVICE) == (enabled or had_key),
    f"keys={sorted(qwenimage.ROPE_PER_DEVICE)}",
)
check(
    "ROPE_PER_DEVICE did not lose an entry",
    set(before_rope) <= set(qwenimage.ROPE_PER_DEVICE),
)
if enabled and not had_key:
    # Only this function's own setdefault can have added it; where it already
    # stood -- the plugin's `flagos` entry, or a previous run -- it is left alone.
    check(
        "ROPE_PER_DEVICE points at the neuron kernel",
        qwenimage.ROPE_PER_DEVICE[DEVICE_KIND]
        is qwenimage.apply_rotary_emb_qwen_neuron,
    )
# Either installer counts, and each is reported: the operand producer has to
# come out wrapped for the running device type by whoever got there first, and
# unwrapped when neither did. On the one arm where the plugin is the installer,
# the wrap predates the snapshot, so what this call owes is to leave it alone --
# same wrapper object, same table entry.
after_qwen = qwenimage.QwenEmbedRope._get_device_freqs
after_layer = qwenimage.QwenEmbedLayer3DRope._get_device_freqs
installed = enabled or by_plugin
check(
    "operand producer is wrapped by the right installer",
    from_a_factory(after_qwen) == installed,
    f"by {'torch_fl' if by_plugin else 'common.py'}" if installed else "",
)
check(
    "QwenEmbedLayer3DRope._get_device_freqs came out the same way",
    from_a_factory(after_layer) == installed,
)
if by_plugin:
    check(
        "the plugin's wrap was not doubled by this call",
        after_qwen is before_qwen and after_layer is before_layer,
    )
    check(
        "the plugin's table entry is the one still installed",
        qwenimage.ROPE_PER_DEVICE.get("flagos") is before_rope.get("flagos"),
    )

if enabled and numeric and by_plugin:
    print(
        "  numeric half skipped: torch_fl's plugin holds the running device type,\n"
        "  so this function's closure is not the one in place -- see\n"
        "  tests/unit/test_gcu_qwenimage_rope.py for what the plugin installs"
    )

if enabled and numeric and not by_plugin:

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
