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

"""Qwen-Image-2512 text-to-image inference on torch-fl, with a CUDA reference.

The ``--stage`` switch exists so that a failure is attributable to one
component. A full run is a 50-step denoising loop with two transformer forwards
per step; localising a crash inside that is far more expensive than localising
it in an isolated text-encoder call.

Stages:
    text-encoder      prompt -> prompt_embeds + prompt_embeds_mask
    transformer-step  one forward at fixed latents, true CFG disabled
    vae               decode fixed latents -> PNG
    full              the whole pipeline -> PNG

Nothing here is tied to one machine. ``--device`` names the backend by torch
device module, the number of visible devices decides the placement (see
``common`` for the rule), and every placement decision can be
overridden: ``--devices``, ``--encoder-device``, ``--transformer-devices``,
``--vae-device``, ``--blocks-per-device``.

``torch_fl`` MUST be imported before ``torch`` on the flagos path: it preloads
the bundled libtorch_cuda.so before PyTorch caches its CUDA hooks. The import is
therefore driven by ``--device`` after argument parsing and before anything else
touches torch.

Run the whole pipeline on flagos, into one PNG:
    python tests/manual/qwen_image_2512/infer.py --stage full \
        --prompt "a red panda eating bamboo" --output flagos.png

Run one stage on another backend, with the inputs the reference run saved:
    python tests/manual/qwen_image_2512/infer.py --device cuda --stage full \
        --load-inputs inp.pt --load-latents lat.pt --output cuda.png

The full procedure, the per-stage expectations, and the readings to record:
    tests/manual/qwen_image_2512/README.md
"""

import argparse
import sys

import numpy as np

import common

STAGES = ("text-encoder", "transformer-step", "vae", "full")

DEFAULT_PROMPT = "a red panda eating bamboo, soft morning light, photorealistic"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Qwen-Image-2512 inference on any torch device backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=STAGES, default="full")
    common.add_placement_args(parser)
    parser.add_argument("--model", default="Qwen/Qwen-Image-2512")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--negative-prompt",
        default=" ",
        help="an empty value disables true CFG, which visibly degrades the image",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--output", default=None, help="PNG path")
    parser.add_argument(
        "--save-inputs",
        default=None,
        help="write prompt embeds, mask, and the negative pair to this .pt",
    )
    parser.add_argument("--load-inputs", default=None)
    parser.add_argument(
        "--save-latents", default=None, help="write the initial latents to this .pt"
    )
    parser.add_argument("--load-latents", default=None)
    return parser.parse_args(argv)


def load_pipeline(torch, args):
    from diffusers import QwenImagePipeline

    return QwenImagePipeline.from_pretrained(args.model, dtype=torch.bfloat16)


def pipeline_timestep_helpers():
    """The two private-ish helpers QwenImagePipeline.__call__ uses for its schedule.

    They are plain functions in the pipeline module, not methods, so they are
    imported rather than reimplemented -- a hand-copied shift formula would drift
    from the pipeline the moment diffusers changes it.
    """
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
        calculate_shift,
        retrieve_timesteps,
    )

    return calculate_shift, retrieve_timesteps


def place_pipeline(torch, pipe, args):
    """Move the components this stage needs, and dispatch the transformer.

    The transformer is 40 GB, so stages that never call it must not pay for
    loading it onto the devices at all.
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
        pipe.vae.to(torch.device(vae))
        print(f"  vae          -> {vae}")

    return common.split_transformer(
        pipe.transformer,
        [torch.device(name) for name in transformer],
        args.blocks_per_device,
    )


def stage_text_encoder(torch, pipe, args):
    """Encode the prompt, and the negative prompt when true CFG is enabled.

    Both are saved together: the paired comparison must inject the negative
    embeddings too, otherwise the reference run would recompute them on its own
    backend and the comparison would no longer isolate the denoising loop.
    """
    encoder = torch.device(common.resolve_placement(torch, args)[0])
    pipe.text_encoder.to(encoder)

    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt=args.prompt,
            device=encoder,
            num_images_per_prompt=1,
            max_sequence_length=args.max_sequence_length,
        )
        negative_embeds = negative_mask = None
        if args.negative_prompt:
            negative_embeds, negative_mask = pipe.encode_prompt(
                prompt=args.negative_prompt,
                device=encoder,
                num_images_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
            )

    print(
        f"  prompt_embeds {tuple(prompt_embeds.shape)} {prompt_embeds.dtype} "
        f"{prompt_embeds.device}"
    )
    if prompt_embeds_mask is not None:
        print(
            f"  mask          {tuple(prompt_embeds_mask.shape)} "
            f"{prompt_embeds_mask.dtype} {prompt_embeds_mask.device}"
        )
    else:
        print("  mask          None (every token valid)")
    if negative_embeds is not None:
        print(
            f"  negative      {tuple(negative_embeds.shape)} {negative_embeds.device}"
        )

    if prompt_embeds.device != encoder:
        print(f"FAIL: prompt_embeds on {prompt_embeds.device}, expected {encoder}")
    else:
        print("  device check : OK")

    if args.save_inputs:
        blob = {}
        for key, tensor in (
            ("prompt_embeds", prompt_embeds),
            ("prompt_embeds_mask", prompt_embeds_mask),
            ("negative_prompt_embeds", negative_embeds),
            ("negative_prompt_embeds_mask", negative_mask),
        ):
            if tensor is not None:
                blob[key] = tensor.detach().cpu()
        torch.save(blob, args.save_inputs)
        print(f"  saved inputs -> {args.save_inputs} ({', '.join(blob)})")

    return prompt_embeds, prompt_embeds_mask, negative_embeds, negative_mask


def _prompt_inputs(torch, pipe, args):
    """Prompt embeds + mask and the negative pair, from --load-inputs or the encoder."""
    if args.load_inputs:
        blob = torch.load(args.load_inputs, map_location="cpu")
        encoder = torch.device(common.resolve_placement(torch, args)[0])

        def take(key):
            tensor = blob.get(key)
            return None if tensor is None else tensor.to(encoder)

        print(
            f"  loaded inputs <- {args.load_inputs} {tuple(blob['prompt_embeds'].shape)}"
        )
        return (
            take("prompt_embeds"),
            take("prompt_embeds_mask"),
            take("negative_prompt_embeds"),
            take("negative_prompt_embeds_mask"),
        )
    return stage_text_encoder(torch, pipe, args)


def make_latents(torch, pipe, args, device):
    """Build the packed initial latents exactly as QwenImagePipeline.__call__ does.

    ``prepare_latents`` packs on its generator branch but returns an injected
    tensor untouched, so both branches end up handing back the packed
    ``(batch, num_patches, in_channels)`` form. Saving the return value of the
    generator branch therefore produces a tensor the other backend can inject.
    """
    generator = torch.Generator(device=device).manual_seed(args.seed)
    loaded = None
    if args.load_latents:
        loaded = torch.load(args.load_latents, map_location="cpu")
        print(f"  loaded latents <- {args.load_latents} {tuple(loaded.shape)}")

    latents = pipe.prepare_latents(
        1,
        pipe.transformer.config.in_channels // 4,
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


def prepare_timesteps(torch, pipe, args, latents):
    """Reproduce the timestep schedule QwenImagePipeline.__call__ builds.

    ``FlowMatchEulerDiscreteScheduler`` is configured with ``use_dynamic_shifting``,
    so ``set_timesteps(num_inference_steps)`` alone raises: the shift depends on
    the packed sequence length and has to be passed as ``mu``. Driving the
    scheduler outside ``__call__`` therefore means repeating the sigma ramp and
    the shift computation as well.
    """
    sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
    calculate_shift, retrieve_timesteps = pipeline_timestep_helpers()
    mu = calculate_shift(
        latents.shape[1],
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    scheduler = pipe.scheduler
    # The ramp is a fixed numpy array here, so pass it through the same public
    # entry point the pipeline uses rather than poking at scheduler internals.
    timesteps, _ = retrieve_timesteps(
        scheduler, args.steps, torch.device("cpu"), sigmas=sigmas, mu=mu
    )
    print(f"  timesteps {len(timesteps)} mu={mu:.4f} first={timesteps[0].item():.1f}")
    return timesteps


def stage_transformer_step(torch, pipe, args):
    """One denoising forward at fixed latents, with true CFG disabled."""
    _, transformer_devices, _ = common.resolve_placement(torch, args)
    latent_device = torch.device(transformer_devices[0])

    prompt_embeds, prompt_embeds_mask, _, _ = _prompt_inputs(torch, pipe, args)
    latents = make_latents(torch, pipe, args, latent_device)

    # encode_prompt only moves embeddings on the path that computes them, and the
    # text encoder lives on a different device than the first transformer shard.
    # Nothing downstream will do this for us.
    prompt_embeds = prompt_embeds.to(latent_device)
    if prompt_embeds_mask is not None:
        prompt_embeds_mask = prompt_embeds_mask.to(latent_device)

    # The scheduler turns a sigma into the timestep the transformer is called with.
    timesteps = prepare_timesteps(torch, pipe, args, latents)
    timestep = timesteps[0].expand(latents.shape[0]).to(latents.dtype)

    img_shapes = [
        [
            (
                1,
                args.height // pipe.vae_scale_factor // 2,
                args.width // pipe.vae_scale_factor // 2,
            )
        ]
    ]

    with torch.no_grad():
        noise = pipe.transformer(
            hidden_states=latents.to(pipe.transformer.dtype),
            timestep=timestep / 1000,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]

    print(f"  transformer noise {tuple(noise.shape)} {noise.dtype} {noise.device}")
    expected = latent_device
    print(
        "  device check   :",
        "OK" if noise.device == expected else f"FAIL (expected {expected})",
    )
    return noise


def save_images(images, path):
    if path is None:
        print("  no --output given, so the image was not written anywhere")
        return
    first = images[0]
    first.save(path)

    arr = np.asarray(first, dtype=np.float32) / 255.0
    print(
        f"  image {first.size} saved -> {path}  mean={arr.mean():.4f} std={arr.std():.4f}"
    )


def stage_vae(torch, pipe, args):
    """Decode latents through the causal Conv3d VAE and write a PNG."""
    _, _, vae_device = common.resolve_placement(torch, args)
    vae_dev = torch.device(vae_device)
    pipe.vae.to(vae_dev)

    latents = make_latents(torch, pipe, args, vae_dev)

    # Copied from QwenImagePipeline.__call__'s decode path. The pipeline
    # normalises with the VAE's own latents_mean / latents_std before calling
    # decode, so a decoder driven outside __call__ has to repeat it.
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
    """The whole pipeline: prompt -> prompt embeds -> latents -> denoise -> decode."""
    _, transformer_devices, vae_device = common.resolve_placement(torch, args)
    latent_device = torch.device(transformer_devices[0])

    prompt_embeds, prompt_embeds_mask, negative_embeds, negative_mask = _prompt_inputs(
        torch, pipe, args
    )

    latents = make_latents(torch, pipe, args, latent_device)
    generator = torch.Generator(device=latent_device).manual_seed(args.seed)

    # See stage_transformer_step: the embeddings have to reach the device the
    # first transformer shard lives on before __call__ forwards them.
    prompt_embeds = prompt_embeds.to(latent_device)
    if prompt_embeds_mask is not None:
        prompt_embeds_mask = prompt_embeds_mask.to(latent_device)
    if negative_embeds is not None:
        negative_embeds = negative_embeds.to(latent_device)
    if negative_mask is not None:
        negative_mask = negative_mask.to(latent_device)

    with torch.no_grad():
        result = pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_embeds,
            negative_prompt_embeds_mask=negative_mask,
            true_cfg_scale=4.0 if negative_embeds is not None else 1.0,
            latents=latents,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            max_sequence_length=args.max_sequence_length,
            generator=generator,
            output_type="pil",
        )

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
    print(f"loading {args.model} (this materialises ~57 GB on the host)")
    pipe = load_pipeline(torch, args)
    place_pipeline(torch, pipe, args)

    STAGE_FUNCS[args.stage](torch, pipe, args)
    common.report_memory(torch, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
