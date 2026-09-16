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

"""Backend-agnostic plumbing shared by the Qwen-Image-2512 manual tests.

The staged runner and the model-card sweep both need the same three things:
torch imported in the order the flagos backend requires, a placement derived
from the cards the run may use, and the transformer split across them. They
live here rather than being repeated so the two scripts cannot drift -- a
placement that differed between them would make a paired run incomparable,
and the pairing is the point of the comparison.

Nothing here hardcodes a machine. ``--device`` names a torch device module
(``flagos``, ``cuda``, and any vendor name a plugin registered), the placement
is derived from how many devices that module reports, and every part of it can
be overridden from the command line.

``torch`` is a parameter, never a module-scope import. On the flagos path
``import torch_fl`` has to precede ``import torch``, and importing torch here
would take that decision away from the caller.
"""

DEFAULT_DEVICE = "flagos"

# A run uses at most this many cards. Three is the most that buys anything:
# the transformer is 40.9 GB of bf16 weights, which no 40 GB card holds alone,
# so it is split -- but every extra shard adds a hidden-state round trip on
# every forward, and there are 100 forwards in a 50-step true-CFG run.
MAX_DEVICES = 3


def import_torch(device_kind):
    """Import torch, and torch_fl first when the run is on flagos.

    ``_preload_cuda_assets()`` in torch_fl/__init__.py dlopens the bundled
    libtorch_cuda.so before PyTorch caches its CUDA hooks. Importing torch
    first leaves those assets unloaded, and the run then fails at the first
    generator with "Cannot get CUDA generator without ATen_cuda library".
    """
    if device_kind == "flagos":
        import torch_fl  # noqa: F401  - must precede `import torch`
    import torch

    return torch


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
            f"the cards this run may use; default: the first {MAX_DEVICES} visible ones"
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
        help="default: every card of --devices except the staging one",
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
       outright. They exist for the shapes the rules below cannot guess.
    2. Otherwise the run takes a device list: ``--devices`` when given, else
       :func:`default_devices`.
    3. The last device of that list stages the text encoder and the VAE.
       Those two travel together because the pipeline builds its latents and
       decodes them on ``self._execution_device``, which falls back to
       ``self.device`` -- the first non-CPU component in ``_get_signature_keys``
       order, and that order is *sorted*, so ``text_encoder`` outranks
       ``transformer`` and ``vae``. Whatever card holds the encoder is
       therefore where the latents are created and where the decode input has
       to be, so the VAE belongs there too.
    4. The transformer is split across the rest. With exactly two cards there
       is no "rest" -- a 40.9 GB transformer does not fit on one 40 GB card --
       so both are shards and the staging card shares.
    """
    count = device_count(torch, args.device)
    if args.devices:
        chosen = _checked(args.devices, args.device, count)
    else:
        chosen = default_devices(torch, args.device)

    if len(chosen) == 1:
        transformer, staging = chosen, chosen[0]
    elif len(chosen) == 2:
        transformer, staging = chosen, chosen[1]
    else:
        transformer, staging = chosen[:-1], chosen[-1]

    if args.transformer_devices:
        transformer = _checked(args.transformer_devices, args.device, count)

    encoder = args.encoder_device or staging
    return encoder, transformer, args.vae_device or encoder


def block_boundaries(num_blocks, num_shards, blocks_per_device=None):
    """How many transformer blocks each shard holds.

    Even by default, remainder to the earlier shards: the blocks are uniform,
    so splitting the count splits the weight footprint. An explicit
    ``--blocks-per-device`` wins, and is checked here rather than trusted --
    a device map that leaves out a block fails inside accelerate's
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
    strings because they also go into printouts and the sweep's manifest, and
    the caller converts.

    One device is a plain ``.to()``. More than one goes through accelerate's
    ``dispatch_model`` with a hand-built device map; the design document
    records why the map is not built with ``device_map="balanced"`` -- that
    route probes free memory through CUDA-specific introspection to answer a
    question the block count already answers.
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


def report_memory(torch, device_kind):
    """Footprint per card, from the backend's own counters.

    Read as reserved rather than allocated: the allocator's reserved total is
    what has to fit in HBM. There is no ``max_memory_allocated`` on the flagos
    surface to read a true peak from, so reserved is the number to record.
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
