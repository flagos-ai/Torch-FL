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

import functools
import os
import sys
import time

DEFAULT_DEVICE = "flagos"

# An optional source tree for environments whose installed diffusers does not
# provide the Qwen-Image-2.1 classes. A compatible backport package is equally
# valid and does not need a second source checkout.
DIFFUSERS_SRC_ENV = "QWEN_IMAGE_21_DIFFUSERS"

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
# unless a card cannot hold the whole model -- name several with ``--devices``,
# or name the shards with ``--transformer-devices``, for a chip where one
# cannot.
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
    """The explicitly supplied Qwen-Image-2.1 diffusers checkout."""
    src = os.environ.get(DIFFUSERS_SRC_ENV)
    if not src:
        raise SystemExit(
            f"the installed diffusers has no QwenImage21Pipeline. Set "
            f"{DIFFUSERS_SRC_ENV} to a source tree containing "
            "diffusers/pipelines/qwenimage21/."
        )
    if not os.path.isdir(src):
        raise SystemExit(
            f"{DIFFUSERS_SRC_ENV}={src!r} is not a directory; it must contain "
            "diffusers/pipelines/qwenimage21/."
        )
    if not os.path.isdir(os.path.join(src, "diffusers", "pipelines", "qwenimage21")):
        raise SystemExit(
            f"{DIFFUSERS_SRC_ENV}={src!r} does not look like a diffusers source "
            f"tree: diffusers/pipelines/qwenimage21/ is missing."
        )
    return src


def import_diffusers():
    """Import a diffusers build that provides the Qwen-Image-2.1 classes.

    A compatible installed package wins unless ``QWEN_IMAGE_21_DIFFUSERS`` is
    explicitly set. If the installed package lacks the classes, retry from that
    source tree without installing it over the environment's stable diffusers.
    """
    if os.environ.get(DIFFUSERS_SRC_ENV):
        import_diffusers_src()
    import diffusers

    if hasattr(diffusers, "QwenImage21Pipeline"):
        return diffusers

    # The package was imported before discovering that it lacks the required
    # class. Remove that module tree before prepending the explicit checkout;
    # otherwise Python returns the cached, incompatible package on the retry.
    for name in tuple(sys.modules):
        if name == "diffusers" or name.startswith("diffusers."):
            del sys.modules[name]
    import_diffusers_src()
    import diffusers

    if hasattr(diffusers, "QwenImage21Pipeline"):
        return diffusers
    raise SystemExit(
        f"diffusers imported from {os.path.dirname(diffusers.__file__)} has no "
        f"QwenImage21Pipeline. {DIFFUSERS_SRC_ENV} must precede it on sys.path."
    )


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

    _select_real_valued_rope(torch, device_kind)
    _pin_vendor_vae_attention(torch, device_kind)

    return torch


def _select_real_valued_rope(torch, device_kind):
    """Point the 2.1 transformer at a real-valued rotation.

    ``apply_rotary_emb_qwen`` in ``transformer_qwenimage21`` carries both
    rotations and picks between them with ``use_real``. The forward pass hardcodes
    ``use_real=False`` in ``_qwenimage21_prepare_qkv``, and
    ``QwenImage21Rope.rope_params`` builds the operand with ``torch.polar``, so
    the shipped path is a complex multiply of the reshaped query/key against a
    complex exponential.

    A chip without a complex dtype cannot take that path. Enflame GCU is the
    measured case: ``topsaten`` has no complex kernel at all, so the vendor run
    reaches the multiply with a ``ComplexFloat`` operand and the process aborts on
    ``TOPSATEN_STATUS_NOT_SUPPORT`` -- not a Python exception:

        dtype: ComplexFloat
        sizes: [1, 4122, 32, 64]

    This is not a numerical shortcut. ``freqs_cis`` is ``polar(1, a)``, so its
    ``.real`` and ``.imag`` *are* ``cos a`` and ``sin a``, and the rotation is the
    same four products in the same order the complex branch's real and imaginary
    parts carry::

        real:  xr*cos - xi*sin
        imag:  xi*cos + xr*sin

    The shipped ``use_real=True`` branch cannot be reused to express this: it
    shapes the operand ``cos[None, None]``, which broadcasts against a
    heads-before-sequence activation (flux's layout) and not against 2.1's
    ``[B, S, H, D]``. So the wrapper writes the rotation itself, and only rewrites
    a call that is both real-seeking and complex-operand -- a call already on the
    real branch, or one handed angles, is passed through untouched.

    Unlike the 2512 flow, which selects a *consumer* by device type through
    diffusers' ``ROPE_PER_DEVICE`` table, the 2.1 transformer has no such table
    to extend: the call site names ``use_real=False`` outright. So this wraps the
    module-level function that call site looks up.

    Off by default, and reported both ways when it is on, because it changes what
    the run measures: with it off, a chip without complex RoPE cannot run the
    shipped path at all. Set ``QWEN_IMAGE_REAL_ROPE=1`` to turn it on.
    """
    if os.environ.get("QWEN_IMAGE_REAL_ROPE", "0") in ("", "0"):
        return

    try:
        from diffusers.models.transformers import transformer_qwenimage21 as qwenimage21
    except ImportError:
        return

    if getattr(qwenimage21, "flagos_real_rope_installed", False):
        return

    original = qwenimage21.apply_rotary_emb_qwen

    def wrapped(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
        if use_real or not torch.is_complex(freqs_cis):
            return original(
                x, freqs_cis, use_real=use_real, use_real_unbind_dim=use_real_unbind_dim
            )

        cos = freqs_cis.real.contiguous().unsqueeze(1)
        sin = freqs_cis.imag.contiguous().unsqueeze(1)
        x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
        out = torch.stack(
            [x_real * cos - x_imag * sin, x_imag * cos + x_real * sin], dim=-1
        )
        return out.flatten(-2).type_as(x)

    qwenimage21.apply_rotary_emb_qwen = wrapped
    qwenimage21.flagos_real_rope_installed = True


def _pin_vendor_vae_attention(torch, device_kind):
    """Decode the latents on the math SDPA backend where the vendor's is refused.

    Qwen-Image-2.1's VAE self-attention is a single 1152-channel head over 4096
    spatial positions. ``autoencoder_kl_qwenimage21`` builds q, k and v as three
    ``[1, 1, 4096, 1152]`` views of one ``[1, 3C, H*W]`` buffer -- strides
    ``[14155776, 14155776, 3456, 1]``, so the row stride is ``3C`` and the three
    share a base pointer.

    The vendor's flash op takes that shape and then refuses it *inside* the
    kernel, which is an abort rather than a decline::

        op_aten_sdp_efficient_attention.cc:318: add check args err: 3
        status id : 3, status name : TOPSATEN_STATUS_NOT_SUPPORT
        gcu_utils.cpp:74 : Check failed: 0            (exit 134)

    ``_fused_sdp_choice`` had already picked the efficient backend by then, so
    nothing falls back and the leg cannot decode at all -- ``infer.py --stage
    vae`` reproduces it in a minute. Pinning ``SDPBackend.MATH`` around the block
    moves the call to the decomposition the flagos leg already uses: its SDPA
    route declines the same call, which is why that leg decodes as ``bmm`` +
    ``_softmax`` and completes.

    This is the vendor leg's own limitation being worked around, not a change to
    what is measured, and it is scoped accordingly. The pin covers the VAE
    attention block only -- the denoise loop's attention is untouched, so the two
    legs still differ there and nowhere else. It is applied only for ``gcu``,
    the device module whose kernel this is; on any other device the shipped path
    is left alone.
    """
    if device_kind != "gcu":
        return

    try:
        from diffusers.models.autoencoders import autoencoder_kl_qwenimage21 as vae21
    except ImportError:
        return

    if getattr(vae21, "flagos_vendor_sdpa_pinned", False):
        return

    block = vae21.QwenImage21AttentionBlock
    original = block.forward

    @functools.wraps(original)
    def forward(self, x):
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            return original(self, x)

    block.forward = forward
    vae21.flagos_vendor_sdpa_pinned = True


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
    2. Otherwise the run takes the cards named by ``--devices``, or one card --
       ``<device>:0`` -- when none are named. See the module docstring: 2.1 is
       33 GB and one 40 GB card holds it, so a split costs a cross-device round
       trip per forward and buys nothing. This is where the rule differs from the
       2512 flow, which splits because it has to.
    3. The transformer gets every card of that list. Naming one card -- the
       default, and what a 40 GB part needs -- therefore puts all 32 blocks on
       it; naming several is how a chip whose card cannot hold 14.2 GB asks for
       a split, and is the same placement as naming the same set with
       ``--transformer-devices``.
    4. The text encoder and the VAE go on the *first* card of the list, and the
       transformer's first shard is there too. The encoder is where the pipeline
       builds its latents and decodes them -- ``self._execution_device`` falls
       back to ``self.device``, the first non-CPU component in
       ``_get_signature_keys`` order, and that order is *sorted*, so
       ``text_encoder`` outranks ``transformer`` and ``vae`` -- so the decode
       input is created there, and :func:`colocated` refuses a full run whose
       ``--vae-device`` would put the weights on a different card.

    A transformer split across cards is placed through accelerate's
    ``dispatch_model``, so the cross-device boundary is accelerate's hooks
    rather than this script's.
    """
    count = device_count(torch, args.device)
    chosen = _checked(args.devices, args.device, count) if args.devices else None
    first = chosen[0] if chosen else f"{args.device}:0"

    if args.transformer_devices:
        transformer = _checked(args.transformer_devices, args.device, count)
    elif chosen:
        # Naming several cards is the request for a split; naming one is not.
        transformer = chosen
    else:
        transformer = [first]

    encoder = args.encoder_device or first
    return encoder, transformer, args.vae_device or encoder


def colocated(encoder, vae):
    """``vae``, or a stop if it is not the card the encoder is on.

    A full run cannot spread the encoder and the VAE across two cards even
    though both flags exist, because the pipeline decodes on the encoder's card
    (see :func:`resolve_placement`, rule 4): the latents are built there, and
    ``vae.decode`` would be handed a tensor on a card its weights are not on.
    The failure is a cross-device error in the middle of a run that took minutes
    to load, so it is refused here instead -- before the weights are read.

    An isolated ``--stage vae`` places the VAE alone and is not checked by this;
    ``infer.py`` does not call it on that path.
    """
    if encoder == vae:
        return vae
    raise SystemExit(
        f"--vae-device {vae} is not the encoder's card ({encoder}). A full run "
        f"decodes on the encoder's card, so the VAE has to be there: move the "
        f"encoder with --encoder-device {vae} instead, or drop --vae-device and "
        f"leave the VAE on {encoder}."
    )


def place_components(torch, pipe, encoder, transformer, vae, blocks_per_device=None):
    """Move every component onto the cards ``resolve_placement`` named.

    The one place a full run's placement is applied, so the placement a caller
    reports is the placement that happened: bench.py and sweep.py both record
    what this returned rather than what was asked for, which is the difference
    between a record and a claim.

    Returns ``(transformer, placement)`` -- the transformer comes back because a
    caller may want the object accelerate dispatched, not because this changes
    it.
    """
    vae = colocated(encoder, vae)
    placed = split_transformer(
        pipe.transformer,
        [torch.device(name) for name in transformer],
        blocks_per_device,
    )
    pipe.text_encoder.to(torch.device(encoder))
    pipe.vae.to(torch.device(vae))
    print(f"  text_encoder -> {encoder}   vae -> {vae}")
    return placed, {"encoder": encoder, "transformer": transformer, "vae": vae}


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
            "2.1 fits one 40 GB card and a split costs cross-device traffic. "
            "Naming more than one splits the transformer across them, which is "
            "what a chip whose card cannot hold 14.2 GB needs"
        ),
    )
    parser.add_argument(
        "--encoder-device",
        default=None,
        help="default: the first entry of --devices, which is the staging card",
    )
    parser.add_argument(
        "--transformer-devices",
        nargs="+",
        default=None,
        help=(
            "default: the cards of --devices, so one card unless several were "
            "named. Pass two or more explicitly to split the transformer across "
            "a set --devices does not describe"
        ),
    )
    parser.add_argument(
        "--vae-device",
        default=None,
        help=(
            "default: the encoder device. A full run must leave it there -- the "
            "pipeline decodes on the encoder's card -- so this flag is for "
            "infer.py's --stage vae, which places the VAE alone"
        ),
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
    parser.add_argument(
        "--omit-all-valid-prompt-mask",
        action="store_true",
        help=(
            "replace an all-valid prompt attention mask with None after prompt "
            "encoding. This is semantically a no-op, keeps padded masks, and can "
            "make the denoising attention eligible for Flash SDPA"
        ),
    )


def enable_all_valid_prompt_mask_elision(pipe):
    """Drop only prompt masks whose every entry is valid.

    Qwen-Image-2.1 removes padding from the encoded prompt, but still returns an
    all-one mask. Carrying that no-op mask into the transformer makes PyTorch's
    SDPA selector reject Flash attention on some backends. The small mask check
    happens once per prompt encoding, before denoising. Any mask containing a
    padded entry is returned unchanged.
    """
    marker = "_qwen21_all_valid_prompt_mask_elision"
    if getattr(pipe, marker, False):
        return

    original_encode_prompt = pipe.encode_prompt
    reported = False

    def encode_prompt(*args, **kwargs):
        nonlocal reported
        prompt_embeds, prompt_mask, image_pad_mask = original_encode_prompt(
            *args, **kwargs
        )
        if prompt_mask is not None:
            all_valid = bool(prompt_mask.detach().to("cpu").bool().all().item())
            if all_valid:
                if not reported:
                    print(
                        "prompt mask: all entries valid; omitting the no-op mask "
                        "to permit Flash SDPA"
                    )
                    reported = True
                prompt_mask = None
        return prompt_embeds, prompt_mask, image_pad_mask

    pipe.encode_prompt = encode_prompt
    setattr(pipe, marker, True)


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

    A peak read after ``reset_peak_memory`` is a maximum over what is live at
    the reset plus whatever the call added, so a resident model's weights are in
    the number. That is what ``torch.cuda`` reports and what makes the two legs
    comparable; a backend whose reset also drops the live total reports the
    call's own activations instead, smaller by the size of the model.
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


def release_memory(torch, device_kind, device):
    """Hand this card's cached blocks back, so the next call starts cold.

    Best effort, and separate from ``reset_peak_memory``: that one drops a
    watermark, this one drops the pool. The two are not interchangeable -- a
    backend that releases but does not report still gets the release.

    Needed because a backend's allocator is allowed to keep its pool across
    calls and some of them cannot carve a large block back out of it. On the GCU
    flagos backend the second decode of a 2.1 run asks the driver for 576 MiB
    with 9.1 GiB already reserved and cannot get it, so a leg that repeats a call
    (every bench leg, by construction) dies after the first one. Releasing
    between calls makes each call start from the same state, which is also the
    more honest reading of a per-call latency.
    """
    module = getattr(torch, device_kind, None)
    if module is None or not hasattr(module, "empty_cache"):
        return False
    try:
        module.empty_cache()
        return True
    except Exception:  # noqa: BLE001
        return False


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

    The denoise interval starts at the first transformer pre-hook and ends at
    the synchronized boundary immediately before VAE decode.  If decode is not
    reached, :meth:`sample` closes the same boundary.  Step callbacks count the
    completed iterations but do not synchronize each one, avoiding a timing
    probe that would serialize the loop.  The text encoder and VAE remain timed
    by wrapping their boundaries -- the VAE is reached as
    ``self.vae.decode(...)``, a method call rather than ``__call__``, so a
    forward hook never fires for it and the bound method has to be wrapped.

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
        self._loop_start = None
        self._loop_end = None
        self._marks = []
        self._timings = {}
        self._handles = []

        encoder = getattr(pipe, "text_encoder", None)
        if encoder is not None:
            self._handles.append(encoder.register_forward_pre_hook(self._before))
            self._handles.append(encoder.register_forward_hook(self._after))

        transformer = getattr(pipe, "transformer", None)
        if transformer is not None:
            self._handles.append(
                transformer.register_forward_pre_hook(self._before_denoise)
            )

        self._original_decode = pipe.vae.decode

        def timed_decode(*call_args, **call_kwargs):
            self._finish_denoise()
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

    def _before_denoise(self, _module, _args):
        if self._loop_start is None:
            self._sync()
            self._loop_start = time.perf_counter()

    def _finish_denoise(self):
        if self._loop_start is not None and self._loop_end is None:
            self._sync()
            self._loop_end = time.perf_counter()

    def reset(self):
        """Begin a new sample: forget the previous one's marks and totals."""
        self._loop_start = None
        self._loop_end = None
        self._marks = []
        self._timings = {}

    def on_step_end(self, _pipe, index, _timestep, kwargs):
        """The ``callback_on_step_end`` the pipeline calls between steps."""
        self._marks.append(index)
        return kwargs

    def sample(self):
        """This sample's phases, all in seconds.

        A phase that did not run is absent rather than zero: a stage that never
        reached the VAE must not read as a 0.000 s decode. ``loop per step``
        divides by the number of completed step callbacks.  The start is captured
        by the first transformer pre-hook and the end is synchronized immediately
        before VAE decode, so the interval includes both the first and final
        denoising steps without synchronizing every step.

        Seconds for every phase, including the one that is *printed* in
        milliseconds: the unit a measurement is stored in should not depend on
        how it is rendered, and ``print_phases`` owns the rendering.
        """
        timings = dict(self._timings)
        self._finish_denoise()
        if self._loop_start is not None and self._loop_end is not None and self._marks:
            elapsed = self._loop_end - self._loop_start
            timings["denoise loop"] = elapsed
            timings["loop per step"] = elapsed / len(self._marks)
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
