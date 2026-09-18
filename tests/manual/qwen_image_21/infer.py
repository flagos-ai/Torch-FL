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

"""Qwen-Image-2.1 text-to-image inference on torch-fl, with a vendor reference.

The ``--stage`` switch exists so that a failure is attributable to one component.
A full run is a 40-step loop with one transformer forward per step (two when
true CFG is on), and localising a crash inside that is far more expensive than
localising it in an isolated text-encoder call.

Stages:
    text-encoder      prompt -> prompt_embeds + mask + img_mask
    transformer-step  one forward at fixed latents
    vae               decode fixed latents -> PNG
    full              the whole pipeline -> PNG

Three things about 2.1 are not obvious and each has bitten:

**Its diffusers classes are in no release.** ``QwenImage21Pipeline``,
``QwenImage21Transformer2DModel`` and ``AutoencoderKLQwenImage21`` exist only in
a 0.41.0.dev0 source checkout -- PyPI's 0.40.0 is the newest release and has
none of them. ``common.import_diffusers()`` therefore prepends that checkout to
``sys.path`` (``QWEN_IMAGE_21_DIFFUSERS``, see ``common``), which leaves any
installed diffusers untouched.

**The prompt embeddings cannot be injected.** ``__call__`` has no
``image_pad_mask`` parameter, and the local ``append_target_slots`` that builds
one does ``mask.new_ones(...)`` unconditionally -- so the path where prompt
embeddings are supplied directly, and the mask is therefore never built, dies
with ``'NoneType' object has no attribute 'new_ones'``. Reproduced on stock CUDA
torch as well, so it is upstream and not a backend fault. A paired comparison can
therefore inject the **initial latents only**; each backend encodes the prompt
itself. ``--save-embeds`` writes them for inspection, and there is deliberately
no flag that loads them back.

**The whole model fits one card.** 33 GB of bf16 weights against 14.2 GB for
2512's transformer alone, so the default placement is one card. A chip whose card
is smaller splits the transformer with ``--transformer-devices``; that path goes
through accelerate's ``dispatch_model`` and is measured working (A100, blocks
0-15 on one card and 16-31 on another).

``torch_fl`` must be imported before ``torch`` on the flagos path: it preloads
the bundled ``libtorch_cuda.so`` before PyTorch caches its CUDA hooks.

Run the whole pipeline on flagos, into one PNG:
    python tests/manual/qwen_image_21/infer.py --stage full --output flagos.png

Run one stage on another backend, with the latents the reference run saved:
    python tests/manual/qwen_image_21/infer.py --device cuda --stage full \
        --load-latents lat.pt --output cuda.png

The full procedure, the per-stage expectations, and the readings to record:
    tests/manual/qwen_image_21/README.md
"""

import argparse
import sys

import common
import prompts

STAGES = ("text-encoder", "transformer-step", "vae", "full")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Qwen-Image-2.1 inference on any torch device backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=STAGES, default="full")
    common.add_placement_args(parser)
    parser.add_argument(
        "--model",
        default=common.model_ref(),
        help=f"a directory or a hub id; default: ${common.MODEL_ENV} or its fallback",
    )
    parser.add_argument(
        "--prompt",
        default=prompts.canonical()["prompt"],
        help=(
            "default: the canonical prompt in prompts.py, which is also the one "
            "bench.py measures with; change it here only to reach a surface the "
            "default does not"
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        default=" ",
        help=(
            "only used when --true-cfg-scale > 1; 2.1 is meant to be sampled "
            "without guidance, which is why the scale defaults to 1.0"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="the pipeline's own default; 2.1 has no published step count",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=1.0,
        help="1.0 disables guidance; >1 needs --negative-prompt",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help=(
            "disable the prefix KV cache. See the pipeline's help: toggling it "
            "gives an equally valid but visibly different sample, so fix it per run"
        ),
    )
    parser.add_argument("--output", default=None, help="PNG path")
    parser.add_argument(
        "--save-embeds",
        default=None,
        help=(
            "write the prompt embeddings to this .pt for inspection. They cannot "
            "be injected back -- see the module docstring"
        ),
    )
    parser.add_argument(
        "--save-latents", default=None, help="write the initial latents to this .pt"
    )
    parser.add_argument("--load-latents", default=None)
    return parser.parse_args(argv)


def load_pipeline(torch, args):
    """The 2.1 pipeline, from the source checkout that defines it."""
    diffusers = common.import_diffusers()
    print(f"diffusers {diffusers.__version__} from {diffusers.__file__}")
    print(f"loading {args.model} (this materialises ~33 GB on the host)")
    return diffusers.QwenImage21Pipeline.from_pretrained(
        args.model, dtype=torch.bfloat16
    )


def place_pipeline(torch, pipe, args):
    """Move every component the current stage reaches onto the cards it needs.

    Unlike the 2512 flow this is trivial by default -- one card holds the whole
    model -- so the interesting part is the split case, and both go through
    ``common.split_transformer``. Stages that never call the transformer must not
    pay to load 14.2 GB of it onto a card.

    ``--stage full`` places the encoder and the VAE together through
    ``common.colocated``, which refuses a VAE on another card before the weights
    are read rather than letting the decode fail minutes later. The two isolated
    stages reach one component each, so they place it alone and are not checked.
    """
    encoder, transformer, vae = common.resolve_placement(torch, args)

    if args.stage in ("text-encoder", "vae"):
        if args.stage == "vae":
            pipe.vae.to(torch.device(vae))
            print(f"  vae          -> {vae}")
        else:
            pipe.text_encoder.to(torch.device(encoder))
            print(f"  text_encoder -> {encoder}")
        return None

    if args.stage in ("transformer-step", "full"):
        pipe.text_encoder.to(torch.device(encoder))
        print(f"  text_encoder -> {encoder}")
    if args.stage == "full":
        pipe.vae.to(torch.device(common.colocated(encoder, vae)))
        print(f"  vae          -> {vae}")

    return common.split_transformer(
        pipe.transformer,
        [torch.device(name) for name in transformer],
        args.blocks_per_device,
    )


def _encode(torch, pipe, prompt, device):
    """``encode_prompt`` for one prompt, as the pipeline calls it.

    Returns its three values rather than two: the third is the image pad mask,
    which the pipeline appends the target slots to and hands the transformer as
    ``img_mask``. Dropping it is what makes the injected-embeddings path fail, so
    this flow carries it through.
    """
    with torch.no_grad():
        return pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
        )


def stage_text_encoder(torch, pipe, args):
    """Encode the prompt, and the negative prompt when guidance is enabled."""
    encoder = torch.device(common.resolve_placement(torch, args)[0])
    pipe.text_encoder.to(encoder)

    prompt_embeds, prompt_embeds_mask, image_pad_mask = _encode(
        torch, pipe, args.prompt, encoder
    )
    negative_embeds = negative_mask = negative_image_mask = None
    if args.true_cfg_scale > 1.0 and args.negative_prompt:
        negative_embeds, negative_mask, negative_image_mask = _encode(
            torch, pipe, args.negative_prompt, encoder
        )

    print(
        f"  prompt_embeds {tuple(prompt_embeds.shape)} {prompt_embeds.dtype} "
        f"{prompt_embeds.device}"
    )
    print(
        f"  img_mask slots {int(image_pad_mask.sum())} True of "
        f"{image_pad_mask.numel()} (a text-only prompt has none)"
    )
    if negative_embeds is not None:
        print(
            f"  negative      {tuple(negative_embeds.shape)} {negative_embeds.device}"
        )

    if prompt_embeds.device != encoder:
        print(f"FAIL: prompt_embeds on {prompt_embeds.device}, expected {encoder}")
    else:
        print("  device check : OK")

    if args.save_embeds:
        blob = {"prompt_embeds": prompt_embeds.detach().cpu()}
        if prompt_embeds_mask is not None:
            blob["prompt_embeds_mask"] = prompt_embeds_mask.detach().cpu()
        torch.save(blob, args.save_embeds)
        print(f"  saved embeds -> {args.save_embeds} ({', '.join(blob)})")
        print(
            "  (inspection only: this pipeline cannot take them back, see the docstring)"
        )

    return prompt_embeds, prompt_embeds_mask, image_pad_mask


def make_latents(torch, pipe, args, device):
    """Build the packed initial latents exactly as ``__call__`` does.

    ``prepare_latents`` packs on its generator branch and returns an injected
    tensor untouched, so both branches hand back the packed
    ``(batch, num_patches, in_channels)`` form -- ``(1, 4096, 64)`` at 1024x1024.
    Saving the return value of the generator branch therefore produces a tensor
    the other backend can inject, which is the only injection this pipeline
    supports.
    """
    generator = torch.Generator(device=device).manual_seed(args.seed)
    loaded = None
    if args.load_latents:
        loaded = torch.load(args.load_latents, map_location="cpu")
        print(f"  loaded latents <- {args.load_latents} {tuple(loaded.shape)}")

    latents, _ = pipe.prepare_latents(
        None,
        1,
        pipe.transformer.config.in_channels,
        args.height,
        args.width,
        pipe.transformer.dtype,
        device,
        generator,
        latents=loaded,
    )
    print(f"  latents {tuple(latents.shape)} {latents.dtype} {latents.device}")
    if args.save_latents:
        torch.save(latents.detach().cpu(), args.save_latents)
        print(f"  saved latents -> {args.save_latents}")
    return latents


def prepare_timesteps(torch, pipe, args, latents, device):
    """Reproduce the timestep schedule ``__call__`` builds.

    ``FlowMatchEulerDiscreteScheduler`` is configured with
    ``use_dynamic_shifting``, so ``set_timesteps(num_inference_steps)`` alone
    raises: the shift depends on the packed sequence length and has to be passed
    as ``mu``. Both helpers are imported from the pipeline module rather than
    reimplemented -- a hand-copied shift formula drifts the moment diffusers
    changes it.

    ``device`` is the pipeline's execution device, which is what ``__call__``
    passes here. It matters: the schedule lands on that device, and a timestep
    that stays on the CPU while the weights are on a card dies inside
    ``timestep_embedder`` with "mat2 is on cuda:0, different from other tensors
    on cpu" -- reported against a ``linear`` in the embedding stack, which is a
    long way from the line that chose the device.
    """
    import numpy as np
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
        calculate_shift,
        retrieve_timesteps,
    )

    sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
    config = pipe.scheduler.config
    mu = calculate_shift(
        latents.shape[1],
        config.get("base_image_seq_len", 256),
        config.get("max_image_seq_len", 4096),
        config.get("base_shift", 0.5),
        config.get("max_shift", 0.9),
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, args.steps, device, sigmas=sigmas, mu=mu
    )
    print(f"  timesteps {len(timesteps)} mu={mu:.4f} first={timesteps[0].item():.1f}")
    return timesteps


def img_shapes_for(args, pipe):
    """The layout the transformer reads: one frame, image tokens only.

    Two levels of nesting -- a list per batch item holding a tuple per frame --
    and the size is in *latent* tokens, which is why it divides by
    ``vae_scale_factor`` and not by 8.
    """
    return [
        [(1, args.height // pipe.vae_scale_factor, args.width // pipe.vae_scale_factor)]
    ]


def target_img_mask(image_pad_mask, latents, torch):
    """``append_target_slots`` from the pipeline, on the real encode output.

    The transformer's ``img_mask`` spans the joint sequence: the text/condition
    slots from ``encode_prompt`` plus one slot per 2x2 group of target latents.
    Taken from the pipeline's own expression so a change upstream surfaces here
    as a shape error rather than as silently wrong attention.
    """
    return torch.cat(
        [image_pad_mask, image_pad_mask.new_ones(1, latents.shape[1] // 4)], dim=1
    )


def stage_transformer_step(torch, pipe, args):
    """One denoising forward at fixed latents, true CFG off."""
    _, transformer_devices, _ = common.resolve_placement(torch, args)
    latent_device = torch.device(transformer_devices[0])

    prompt_embeds, prompt_embeds_mask, image_pad_mask = stage_text_encoder(
        torch, pipe, args
    )
    latents = make_latents(torch, pipe, args, latent_device)

    # encode_prompt only moves embeddings on the path that computes them, and the
    # text encoder can be on another card than the first transformer shard.
    # Nothing downstream does this for us.
    prompt_embeds = prompt_embeds.to(latent_device)
    if prompt_embeds_mask is not None:
        prompt_embeds_mask = prompt_embeds_mask.to(latent_device)
    image_pad_mask = image_pad_mask.to(latent_device)

    timesteps = prepare_timesteps(
        torch,
        pipe,
        args,
        latents,
        torch.device(common.resolve_placement(torch, args)[0]),
    )
    timestep = timesteps[0].expand(latents.shape[0]).to(latent_device).to(latents.dtype)

    with torch.no_grad():
        joint = pipe.transformer(
            hidden_states=latents.to(pipe.transformer.dtype),
            timestep=timestep / 1000,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes_for(args, pipe),
            img_mask=target_img_mask(image_pad_mask, latents, torch),
            return_dict=False,
        )[0]

    # The transformer returns the *joint* sequence -- text slots first, then the
    # image tokens it was asked to predict -- and the pipeline takes the tail.
    # Reported rather than silently sliced, because the joint width is the
    # clearest single number that says the mask and the layout line up: at
    # 1024x1024 with the default prompt it is 26 text + 4096 image.
    print(f"  joint sequence {tuple(joint.shape)} {joint.dtype} {joint.device}")
    noise = joint[:, -latents.size(1) :]
    print(f"  transformer noise {tuple(noise.shape)} {noise.dtype} {noise.device}")
    expected = torch.device(transformer_devices[-1])
    print(
        "  device check   :",
        "OK" if noise.device == expected else f"FAIL (expected {expected})",
    )
    return noise


def save_images(images, path):
    if path is None:
        return
    first = images[0]
    first.save(path)
    import numpy as np

    arr = np.asarray(first, dtype=np.float32) / 255.0
    print(
        f"  image {first.size} saved -> {path}  mean={arr.mean():.4f} std={arr.std():.4f}"
    )


def stage_vae(torch, pipe, args):
    """Decode latents through the causal VAE and write a PNG."""
    _, _, vae_device = common.resolve_placement(torch, args)
    vae_dev = torch.device(vae_device)
    pipe.vae.to(vae_dev)

    latents = make_latents(torch, pipe, args, vae_dev)

    # Copied from ``__call__``'s decode path. The pipeline normalises with the
    # VAE's own latents_mean / latents_std before calling decode -- this VAE does
    # not divide by a scaling_factor -- so a decoder driven outside __call__ has
    # to repeat it.
    with torch.no_grad():
        latents = pipe._unpack_latents(
            latents, args.height, args.width, pipe.vae_scale_factor
        )
        latents = latents.to(pipe.vae.dtype)
        z_dim = pipe.vae.config.z_dim
        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
            1, z_dim, 1, 1, 1
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, z_dim, 1, 1, 1
        )
        latents = latents / latents_std.to(latents.device, latents.dtype)
        latents = latents + latents_mean.to(latents.device, latents.dtype)
        print(f"  unpacked {tuple(latents.shape)} {latents.dtype} {latents.device}")

        image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    print(f"  decoded {tuple(image.shape)} {image.dtype} {image.device}")

    images = pipe.image_processor.postprocess(image, output_type="pil")
    save_images(images, args.output)
    return images


def stage_full(torch, pipe, args):
    """The whole pipeline: prompt -> latents -> denoise -> decode."""
    _, transformer_devices, _ = common.resolve_placement(torch, args)
    latent_device = torch.device(transformer_devices[0])

    latents = make_latents(torch, pipe, args, latent_device)
    generator = torch.Generator(device=latent_device).manual_seed(args.seed)

    # One sample, because this stage exists to localise a failure rather than to
    # measure: a single full run with its phases is what says which component
    # broke. Repeating it and reporting a spread is bench.py's job.
    timer = common.PhaseTimer(torch, pipe, args.device)
    timer.reset()

    with torch.no_grad():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.true_cfg_scale > 1.0 else None,
            true_cfg_scale=args.true_cfg_scale,
            latents=latents,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            generator=generator,
            output_type="pil",
            use_kv_cache=not args.no_kv_cache,
            callback_on_step_end=timer.on_step_end,
        )

    timings = timer.sample()
    timer.close()
    common.print_phases(timings)
    save_images(result.images, args.output)
    return result.images


STAGE_FUNCS = {
    "text-encoder": stage_text_encoder,
    "transformer-step": stage_transformer_step,
    "vae": stage_vae,
    "full": stage_full,
}


def main(argv=None):
    args = parse_args(argv)
    torch = common.import_torch(args.device)

    print(f"stage: {args.stage}   device: {args.device}   torch: {torch.__version__}")
    encoder, transformer, vae = common.resolve_placement(torch, args)
    print(f"  encoder/vae -> {encoder}   transformer -> {' '.join(transformer)}")
    pipe = load_pipeline(torch, args)
    place_pipeline(torch, pipe, args)

    STAGE_FUNCS[args.stage](torch, pipe, args)
    common.report_memory(torch, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
