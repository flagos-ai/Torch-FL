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

"""Backend-agnostic plumbing shared by the Qwen-Image-2.1 manual tests.

Three things every script in this directory needs, and which live here so they
cannot drift apart: the diffusers checkout that defines the 2.1 classes, the
import order the flagos backend requires, and the placement of the three
components across the cards a chip actually has.

Nothing here hardcodes a machine or a backend. ``--device`` names a torch device
module (``flagos``, ``cuda``, and any vendor name a plugin registered), the
placement is derived from the devices that module reports, and every part of it
can be overridden from the command line.

``torch`` is a parameter, never a module-scope import. On the flagos path
``import torch_fl`` has to precede ``import torch``, and importing torch here
would take that decision away from the caller.
"""

import os
import sys
import time

DEFAULT_DEVICE = "flagos"

# Where the Qwen-Image-2.1 pipeline classes live. They are in no released
# diffusers -- 0.40.0 is the newest on PyPI and its QwenImage21* names do not
# exist -- so this flow reads them out of a source checkout supplied from
# outside. The default is this development host's; a chip brings its own with
# the environment variable below.
DIFFUSERS_SRC_ENV = "QWEN_IMAGE_21_DIFFUSERS"
DIFFUSERS_SRC_DEFAULT = "/nfs/lvyufeng/Qwen-Image-2.1-ref-qwen-image-2.1/src"

# Where the weights are. A directory, not a hub id: the published model is on
# ModelScope under Qwen/Qwen-Image-2.1 and the copy used here was fetched to
# disk, so the default is a path and every script takes --model to point
# anywhere else -- another disk, or a hub id when the box has a proxy.
MODEL_ENV = "QWEN_IMAGE_21_MODEL"
MODEL_DEFAULT = "/nfs/lvyufeng/Qwen-Image-2.1"

# The three components are 33 GB of bf16 weights (17.5 text encoder, 14.2
# transformer, 1.35 VAE), which one 40 GB card holds. A run therefore uses ONE
# card by default, and this is deliberately the opposite of the Qwen-Image-2512
# flow, whose transformer alone is 40.9 GB and has to be split. Spreading 2.1
# across cards costs a hidden-state round trip on every forward and buys nothing
# unless a card cannot hold the whole model -- ``--transformer-devices`` is there
# for the chips where one cannot.
MAX_DEVICES = 3


def model_ref():
    """The model to load: ``QWEN_IMAGE_21_MODEL``, else the local default.

    Returned as a plain string because it is passed straight to
    ``from_pretrained``, which accepts either a directory or a hub id -- a chip
    that fetched the weights to a different disk sets the variable, and a box
    with a proxy and no local copy can point it at ``Qwen/Qwen-Image-2.1``.
    """
    return os.environ.get(MODEL_ENV, MODEL_DEFAULT)


def diffusers_src():
    """The Qwen-Image-2.1 diffusers checkout, or a fatal error naming the variable."""
    src = os.environ.get(DIFFUSERS_SRC_ENV, DIFFUSERS_SRC_DEFAULT)
    if not os.path.isdir(src):
        raise SystemExit(
            f"Qwen-Image-2.1's diffusers classes are not in any released diffusers, "
            f"so this flow needs a source checkout. Set {DIFFUSERS_SRC_ENV} to one "
            f"(it must contain diffusers/pipelines/qwenimage21/); "
            f"{src!r} has no such directory."
        )
    if not os.path.isdir(os.path.join(src, "diffusers", "pipelines", "qwenimage21")):
        raise SystemExit(
            f"{DIFFUSERS_SRC_ENV}={src!r} does not look like a diffusers source "
            f"tree: diffusers/pipelines/qwenimage21/ is missing."
        )
    return src


def import_diffusers():
    """Import diffusers from the 2.1 checkout, whatever the interpreter has installed.

    Prepending to ``sys.path`` rather than installing: the checkout is a
    0.41.0.dev0 tree, and installing it over an environment's diffusers 0.40.0
    would replace the version every other test in this repository runs against.
    Prepending also means the checkout wins over an installed 0.40.0 without
    touching it, so one interpreter can serve both.
    """
    import_diffusers_src()
    import diffusers

    if not hasattr(diffusers, "QwenImage21Pipeline"):
        raise SystemExit(
            f"diffusers imported from {os.path.dirname(diffusers.__file__)} has no "
            f"QwenImage21Pipeline. {DIFFUSERS_SRC_ENV} must precede it on sys.path."
        )
    return diffusers


def import_diffusers_src():
    src = diffusers_src()
    if src not in sys.path:
        sys.path.insert(0, src)


def import_torch(device_kind):
    """Import torch, and torch_fl first when the run is on flagos.

    ``_preload_cuda_assets()`` in torch_fl/__init__.py dlopens the bundled
    libtorch_cuda.so before PyTorch caches its CUDA hooks. Importing torch first
    leaves those assets unloaded, and the run then fails at the first generator
    with "Cannot get CUDA generator without ATen_cuda library".
    """
    if device_kind == "flagos":
        import torch_fl  # noqa: F401  - must precede `import torch`
    import torch

    return torch


def device_module(torch, device_kind):
    """The ``torch.<kind>`` namespace, or a fatal error that names the problem.

    Every device family torch can drive exposes the same handful of entry
    points under its own name -- ``torch.cuda``, ``torch.flagos``,
    ``torch.musa`` -- so reading it dynamically is what lets one script cover
    any backend instead of carrying a table of vendor import names.
    """
    module = getattr(torch, device_kind, None)
    if module is None or not hasattr(module, "device_count"):
        raise SystemExit(
            f"torch has no '{device_kind}' device module. Check that this is the "
            f"right torch build for the backend and that its plugin was imported."
        )
    return module


def device_count(torch, device_kind):
    """How many devices the backend reports, refusing a run on none."""
    count = int(device_module(torch, device_kind).device_count())
    if count <= 0:
        raise SystemExit(f"'{device_kind}' reports {count} devices; nothing to run on")
    return count


def default_devices(torch, device_kind):
    """The cards a run uses when none are named: the first ``MAX_DEVICES`` visible."""
    count = min(device_count(torch, device_kind), MAX_DEVICES)
    return [f"{device_kind}:{index}" for index in range(count)]


def _checked(devices, device_kind, count):
    """Reject a device string the backend cannot have, before any memory is spent."""
    for name in devices:
        prefix, _, index = name.partition(":")
        if prefix != device_kind or not index.isdigit() or int(index) >= count:
            raise SystemExit(
                f"{name!r} is not a '{device_kind}' device: expected one of "
                f"{device_kind}:0..{device_kind}:{count - 1}"
            )
    return list(devices)


def resolve_placement(torch, args):
    """Work out what goes where, as ``(encoder, transformer_devices, vae)``.

    The rules, in order:

    1. ``--encoder-device``/``--transformer-devices``/``--vae-device`` win
       outright.
    2. Otherwise everything goes on ONE card -- the first device of
       ``--devices``, or ``<device>:0``. See the module docstring: 2.1 is 33 GB
       and one 40 GB card holds it, so a split would cost a cross-device round
       trip per forward and buy nothing. This is where the rule differs from the
       2512 flow, which splits because it has to.
    3. The text encoder and the VAE share that card because the pipeline builds
       its latents and decodes them on ``self._execution_device``, which falls
       back to ``self.device`` -- the first non-CPU component in
       ``_get_signature_keys`` order, and that order is *sorted*, so
       ``text_encoder`` outranks ``transformer`` and ``vae``. Whatever card holds
       the encoder is therefore where the latents are created and where the
       decode input has to be, so the VAE belongs there too. Putting the VAE
       elsewhere without moving the encoder as well fails on the first decode
       with a cross-device error.

    A transformer split across cards is supported (``--transformer-devices``
    takes two or more) and is what a chip whose card is smaller than the model
    needs; it is placed through accelerate's ``dispatch_model``, so the
    cross-device boundary is accelerate's hooks rather than this script's.
    """
    count = device_count(torch, args.device)
    if args.devices:
        chosen = _checked(args.devices, args.device, count)
    else:
        chosen = None

    if args.transformer_devices:
        transformer = _checked(args.transformer_devices, args.device, count)
    else:
        first = chosen[0] if chosen else f"{args.device}:0"
        transformer = [first]

    staging = chosen[-1] if chosen else f"{args.device}:0"
    encoder = args.encoder_device or staging
    return encoder, transformer, args.vae_device or encoder


def block_boundaries(num_blocks, num_shards, blocks_per_device=None):
    """How many transformer blocks each shard holds.

    Even by default, remainder to the earlier shards: the blocks are uniform, so
    splitting the count splits the weight footprint. An explicit
    ``--blocks-per-device`` wins, and is checked here rather than trusted -- a
    device map that leaves out a block fails inside accelerate's
    ``check_device_map`` with a message that never mentions the flag.
    """
    if blocks_per_device:
        counts = [int(count) for count in blocks_per_device]
        if len(counts) != num_shards:
            raise SystemExit(
                f"--blocks-per-device takes {num_shards} counts for "
                f"{num_shards} transformer devices, got {len(counts)}"
            )
        if sum(counts) != num_blocks:
            raise SystemExit(
                f"--blocks-per-device sums to {sum(counts)}, but this "
                f"transformer has {num_blocks} blocks"
            )
        return counts

    base, remainder = divmod(num_blocks, num_shards)
    return [base + (1 if index < remainder else 0) for index in range(num_shards)]


def split_transformer(transformer, devices, blocks_per_device=None):
    """Place the transformer across ``devices``, splitting between blocks.

    ``devices`` are ``torch.device`` objects; ``resolve_placement`` returns
    strings because they also go into printouts and the sweep's manifest, and the
    caller converts.

    One device is a plain ``.to()``. More than one goes through accelerate's
    ``dispatch_model`` with a hand-built device map; the map is not built with
    ``device_map="balanced"`` because that route probes free memory through
    CUDA-specific introspection to answer a question the block count answers.
    """
    blocks = list(transformer.transformer_blocks)
    unique = list(dict.fromkeys(devices))

    if len(unique) == 1:
        print(f"  transformer: all {len(blocks)} blocks -> {unique[0]} (no split)")
        return transformer.to(unique[0])

    counts = block_boundaries(len(blocks), len(devices), blocks_per_device)

    # Enumerate the top-level children rather than naming them. dispatch_model's
    # first statement is check_device_map, which raises on any parameter the map
    # does not cover -- a hardcoded name list that drifts from the model fails as
    # an opaque "does not give any device for the following parameters".
    # named_children() is exhaustive by construction, so nothing can be missed.
    device_map = {}
    for name, _ in transformer.named_children():
        if name != "transformer_blocks":
            device_map[name] = devices[0]

    start = 0
    for device, count in zip(devices, counts):
        for index in range(start, start + count):
            device_map[f"transformer_blocks.{index}"] = device
        start += count

    print(
        "  transformer: blocks "
        + ", ".join(
            f"{sum(counts[:i])}..{sum(counts[: i + 1]) - 1} -> {device}"
            for i, device in enumerate(devices)
        )
    )

    # Every remaining top-level module stays on the first device, including
    # norm_out/proj_out. That is not cosmetic: accelerate's root hook is
    # io_same_device, so the forward's return value is only moved back to the
    # caller's device if it is produced there. Leaving proj_out on the last
    # shard returns the noise prediction on a device the scheduler's latents are
    # not on, and scheduler.step then fails on a cross-device add.
    from accelerate import dispatch_model

    return dispatch_model(transformer, device_map=device_map)


def add_placement_args(parser):
    """The placement flags both scripts accept, with their documented defaults."""
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=(
            "torch device module to run on: flagos (imports torch_fl first), "
            "cuda, or any vendor name a torch plugin has registered"
        ),
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help=(
            "the cards this run may use; default: the first card only, because "
            "2.1 fits one 40 GB card and a split costs cross-device traffic"
        ),
    )
    parser.add_argument(
        "--encoder-device",
        default=None,
        help="default: the last entry of --devices, which is the staging card",
    )
    parser.add_argument(
        "--transformer-devices",
        nargs="+",
        default=None,
        help=(
            "default: the transformer shares the staging card. Pass two or more "
            "to split it across cards on a chip whose card cannot hold 14.2 GB"
        ),
    )
    parser.add_argument(
        "--vae-device",
        default=None,
        help="default: the encoder device, which is the pipeline's execution device",
    )
    parser.add_argument(
        "--blocks-per-device",
        nargs="+",
        type=int,
        default=None,
        help=(
            "transformer blocks per shard, in --transformer-devices order; "
            "default: an even split"
        ),
    )


def report_memory(torch, device_kind):
    """Footprint per card, from the backend's own counters.

    Read as reserved rather than allocated: the allocator's reserved total is
    what has to fit in HBM. There is no ``max_memory_allocated`` on some vendor
    surfaces to read a true peak from, so reserved is the number to record.
    """
    module = getattr(torch, device_kind, None)
    if module is None or not hasattr(module, "memory_reserved"):
        print(f"  {device_kind}: this backend exposes no memory counters")
        return

    for index in range(getattr(module, "device_count", lambda: 0)()):
        reserved = module.memory_reserved(index) / (1024**3)
        if reserved == 0:
            continue
        allocated = module.memory_allocated(index) / (1024**3)
        print(
            f"  {device_kind}:{index} reserved {reserved:.2f} GiB "
            f"allocated {allocated:.2f} GiB"
        )


def sync(torch, device_kind):
    """Block until this backend's queued work is done.

    Its own function because three call sites need it for the same reason: a
    number read without it is a measurement of how fast the host enqueued work,
    not of how long the device took. ``device_module`` raises a message naming
    the backend when there is no such module, which is the failure worth having.
    """
    device_module(torch, device_kind).synchronize()


def peak_memory(torch, device_kind, device):
    """Peak memory on one card, as ``(gib, source)``; ``(None, None)`` if unknown.

    ``device`` is the resolved placement -- ``flagos:2``, not ``flagos``. The
    counters are per card and an index-less query answers for the backend's
    current device, which silently reports another card's number.

    The peak allocated watermark is preferred, because that is what the
    diffusers harness reports and it is the number that says whether a larger
    batch would fit. Three surfaces are tried in order, because torch.cuda's
    spelling is not the only one in use:

    ``max_memory_allocated``
        ``torch.cuda`` and the vendor shims that mirror it.
    ``memory_stats(...)["peak_allocated_bytes"]``
        ``torch.flagos``, whose stats dict is flat and has no separate
        ``max_memory_allocated``.
    ``memory_reserved``
        Last resort. It is a different measurement -- the allocator's whole pool
        rather than what was live -- so the source is returned alongside and a
        table that mixes the two says so.
    """
    module = getattr(torch, device_kind, None)
    if module is None:
        return None, None
    index = torch.device(device).index

    reader = getattr(module, "max_memory_allocated", None)
    if reader is not None:
        try:
            return reader(index) / (1024**3), "peak allocated"
        except Exception:  # noqa: BLE001 - a counter that exists but refuses is not fatal
            pass

    stats = getattr(module, "memory_stats", None)
    if stats is not None:
        try:
            blob = stats(index)
            if isinstance(blob, dict) and "peak_allocated_bytes" in blob:
                return blob["peak_allocated_bytes"] / (1024**3), "peak allocated"
        except Exception:  # noqa: BLE001
            pass

    reader = getattr(module, "memory_reserved", None)
    if reader is not None:
        try:
            return reader(index) / (1024**3), "reserved"
        except Exception:  # noqa: BLE001
            pass
    return None, None


def reset_peak_memory(torch, device_kind, device):
    """Drop this card's peak watermark, so the next read is this call's own.

    Best effort by design: a backend that does not implement it still produces a
    running maximum, which is a peak of the run rather than of the call. That is
    a weaker number, not a wrong one, and refusing to benchmark would be worse.
    """
    module = getattr(torch, device_kind, None)
    if module is None or not hasattr(module, "reset_peak_memory_stats"):
        return False
    try:
        module.reset_peak_memory_stats(torch.device(device).index)
        return True
    except Exception:  # noqa: BLE001
        return False


class PhaseTimer:
    """Attribute the pipeline's wall time to its three components.

    A whole-run number cannot say whether the chip is slow in the text encoder,
    the denoise loop or the decode, and the three fail for different reasons --
    so every full run reports them.

    The loop is timed from ``callback_on_step_end``, which the pipeline calls
    after each step and which nothing else here needs. The text encoder and the
    VAE are timed by wrapping them rather than by editing the pipeline: the VAE
    is reached as ``self.vae.decode(...)``, a method call rather than
    ``__call__``, so a forward hook never fires for it and the bound method has
    to be wrapped instead. Both are synchronised on the way in and out, so the
    number is device-attributed rather than merely "the host returned".

    One sample per ``reset``/``sample`` pair, rather than one accumulator for the
    whole run, because a benchmark needs each measured call's own phases: only
    independent samples can be given a mean and a spread. ``infer.py`` resets
    once and samples once, which is the same table it printed before this class
    existed.

    Usage::

        timer = PhaseTimer(torch, pipe, device_kind)
        ...
        timer.reset()
        pipe(..., callback_on_step_end=timer.on_step_end)
        phases = timer.sample()
        ...
        timer.close()
    """

    #: The four keys ``sample`` returns, in the order infer.py prints them.
    KEYS = ("text encoder", "denoise loop", "loop per step", "vae decode")

    def __init__(self, torch, pipe, device_kind):
        self._pipe = pipe
        self._sync = lambda: sync(torch, device_kind)
        self._marks = []
        self._timings = {}
        self._handles = []

        encoder = getattr(pipe, "text_encoder", None)
        if encoder is not None:
            self._handles.append(encoder.register_forward_pre_hook(self._before))
            self._handles.append(encoder.register_forward_hook(self._after))

        self._original_decode = pipe.vae.decode

        def timed_decode(*call_args, **call_kwargs):
            self._sync()
            started = time.perf_counter()
            out = self._original_decode(*call_args, **call_kwargs)
            self._sync()
            self._timings["vae decode"] = self._timings.get("vae decode", 0.0) + (
                time.perf_counter() - started
            )
            return out

        pipe.vae.decode = timed_decode

    def _before(self, module, args, kwargs=None):
        self._sync()
        module._phase_t0 = time.perf_counter()

    def _after(self, module, args, output):
        self._sync()
        self._timings["text encoder"] = self._timings.get("text encoder", 0.0) + (
            time.perf_counter() - module._phase_t0
        )

    def reset(self):
        """Begin a new sample: forget the previous one's marks and totals."""
        self._marks = []
        self._timings = {}

    def on_step_end(self, _pipe, index, _timestep, kwargs):
        """The ``callback_on_step_end`` the pipeline calls between steps."""
        self._marks.append((index, time.perf_counter()))
        return kwargs

    def sample(self):
        """This sample's phases, all in seconds.

        A phase that did not run is absent rather than zero: a stage that never
        reached the VAE must not read as a 0.000 s decode. ``loop per step``
        divides by ``len(marks) - 1`` because N callbacks bound N-1 intervals --
        dividing by the step count instead would understate the per-step cost by
        one step's worth.

        Seconds for every phase, including the one that is *printed* in
        milliseconds: the unit a measurement is stored in should not depend on
        how it is rendered, and ``print_phases`` owns the rendering.
        """
        timings = dict(self._timings)
        if len(self._marks) > 1:
            first, last = self._marks[0][1], self._marks[-1][1]
            timings["denoise loop"] = last - first
            timings["loop per step"] = (last - first) / (len(self._marks) - 1)
        return timings

    def close(self):
        """Remove the hooks and put ``pipe.vae.decode`` back."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._pipe.vae.decode = self._original_decode


def print_phases(timings):
    """The phase table ``infer.py`` and ``bench.py`` both print.

    Formatting lives here so the two cannot drift into two opinions about what
    the same measurement looks like. Ordered by ``PhaseTimer.KEYS`` rather than
    by insertion, so a phase that happens to be recorded first does not move
    rows around between runs.
    """
    print()
    print("---- phases ----")
    ordered = [name for name in PhaseTimer.KEYS if name in timings]
    ordered += [name for name in timings if name not in ordered]
    for name in ordered:
        seconds = timings[name]
        if name.endswith("per step"):
            print(f"  {name:<16}: {seconds * 1e3:8.1f} ms")
        else:
            print(f"  {name:<16}: {seconds:8.3f} s")
