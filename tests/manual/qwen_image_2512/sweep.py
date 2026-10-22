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

"""Qwen-Image-2512 text-to-image, driven by every prompt on the model card.

The generation settings are the model card's Quick Start snippet as written: the
generic ``DiffusionPipeline`` entry point, ``torch_dtype=torch.bfloat16``, the
card's negative prompt, its 16:9 canvas, 50 steps, ``true_cfg_scale=4.0``, and
seed 42. The prompts are read out of the card itself rather than transcribed --
its Quick Start snippet plus every showcase blockquote -- so there is no risk of
a copy drifting from what the model authors published.

Two things are not the card's, and both are hardware:

``.to(device)`` is replaced by explicit placement. The card assumes the whole
pipeline fits on one device. It does not on a 40 GB card: the transformer is
40.9 GB in bf16, which is the entire card before any activation, and the text
encoder is another 15.6 GB. The placement is derived from the device count and
can be overridden -- see ``common`` and ``--help``.

``torch.Generator(device="cuda")`` takes the backend's device string instead --
``cuda`` and ``flagos`` are two names for the same card, and on the torch_fl
side the name is what ``rename_privateuse1_backend`` registered. On a card big
enough to hold the whole pipeline, one ``--devices`` entry is all the placement
needs.

Nothing else branches on the backend: prompts, resolution, sampler settings and
seed are shared, so a pair of runs is directly comparable image by image. Note
that two backends do not share an RNG stream, so the same seed does not give the
same initial noise -- compare what the images depict, not their pixels, or
inject the same latents with ``infer.py``.

``torch_fl`` must be imported before ``torch`` on the flagos side: it preloads
the bundled ``libtorch_cuda.so`` that the CPU torch wheel dlopens.

Run:
    python tests/manual/qwen_image_2512/sweep.py --device cuda --output-dir out/cuda
    python tests/manual/qwen_image_2512/sweep.py --device flagos --output-dir out/flagos

The full procedure and the readings to record per chip:
    tests/manual/qwen_image_2512/README.md
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import common

# The model card's "16:9" aspect-ratio preset, and the rest of its Quick Start.
WIDTH, HEIGHT = 1664, 928
STEPS = 50
TRUE_CFG_SCALE = 4.0
SEED = 42
MODEL_NAME = "Qwen/Qwen-Image-2512"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Qwen-Image-2512 over every model-card prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common.add_placement_args(parser)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output-dir", required=True, help="one PNG per prompt")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="run just these prompt ids, e.g. --only 01 07",
    )
    return parser.parse_args(argv)


def model_card_prompts(model):
    """Every prompt the model card states, in card order, deduplicated.

    Read from the card rather than transcribed: the Quick Start snippet holds one
    prompt and the showcase sections hold the rest as markdown blockquotes, and
    the Quick Start one is repeated as a showcase, so the pair has to be
    collapsed. Reading the file keeps all of that a property of the card instead
    of a property of this script.
    """
    from huggingface_hub import hf_hub_download

    text = Path(hf_hub_download(repo_id=model, filename="README.md")).read_text(
        encoding="utf-8"
    )

    quick_start = re.search(r"prompt = '''(.*?)'''", text, re.S)
    negative = re.search(r'negative_prompt = "(.*?)"\n', text, re.S)
    if quick_start is None or negative is None:
        raise RuntimeError("model card no longer has the Quick Start snippet")

    candidates = [quick_start.group(1)]
    candidates += re.findall(r"^> (.+)$", text, re.M)

    prompts, seen = [], set()
    for candidate in candidates:
        candidate = candidate.strip()
        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        prompts.append({"id": f"{len(prompts) + 1:02d}", "prompt": candidate})

    return prompts, negative.group(1)


def release(torch, device_kind):
    """Hand the card back between prompts.

    The weights are 57.9 GiB of a 64 GiB card -- transformer, text encoder and
    VAE together -- so there is no page to spare and the sweep is a memory test
    as much as a quality one. Without this the second prompt of a sweep can die
    in the transformer's attention having run at a resolution the first prompt
    cleared, with the card reporting 0 bytes free and only 276 MiB of the
    allocator's own pool unallocated. Collecting and emptying the cache costs a
    few hundred milliseconds against a prompt that takes minutes, and it keeps
    the per-prompt footprints in the log comparable.
    """
    import gc

    gc.collect()
    module = common.device_module(torch, device_kind)
    if hasattr(module, "empty_cache"):
        module.empty_cache()


def footprint(torch, device):
    """Reserved and allocated on the card the prompt actually ran on.

    ``device`` is the resolved placement -- ``flagos:6``, not ``flagos``. The
    counters are per card, and an index-less query answers for the backend's
    current device rather than the one the placement named: the first sweep
    recorded 0.00 GiB for a prompt that had just allocated 57.9 GiB, because
    the pipeline was placed on card 6 while the query landed on card 0.
    """
    resolved = torch.device(device)
    module = common.device_module(torch, resolved.type)
    if not hasattr(module, "memory_reserved"):
        return None
    reserved = module.memory_reserved(resolved.index) / (1024**3)
    allocated = module.memory_allocated(resolved.index) / (1024**3)
    return {"reserved_gib": round(reserved, 2), "allocated_gib": round(allocated, 2)}


def main(argv=None):
    args = parse_args(argv)

    torch = common.import_torch(args.device)
    from diffusers import DiffusionPipeline

    prompts, negative_prompt = model_card_prompts(args.model)
    if args.only:
        wanted = set(args.only)
        prompts = [p for p in prompts if p["id"] in wanted]
        if not prompts:
            raise SystemExit(f"no prompt matched --only {args.only}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder, transformer_devices, _ = common.resolve_placement(torch, args)
    print(f"device: {args.device}   torch: {torch.__version__}")
    print(f"{len(prompts)} model-card prompts, {WIDTH}x{HEIGHT}, {STEPS} steps, ")
    print(f"true_cfg_scale={TRUE_CFG_SCALE}, seed={SEED} -> {out_dir}")

    print(f"loading {args.model} (this materialises ~57 GB on the host)")
    started = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    print(f"  loaded in {time.perf_counter() - started:.1f}s")

    # Read the split before dispatching: accelerate wraps the blocks in hooks
    # rather than replacing them, but the count belongs to the loaded model and
    # is what the manifest has to record, so take it while it is unhooked.
    blocks_per_device = common.block_boundaries(
        len(pipe.transformer.transformer_blocks),
        len(transformer_devices),
        args.blocks_per_device,
    )

    pipe.text_encoder.to(torch.device(encoder))
    pipe.vae.to(torch.device(encoder))
    print(f"  text_encoder -> {encoder}")
    print(f"  vae          -> {encoder} (the pipeline's execution device)")
    pipe.transformer = common.split_transformer(
        pipe.transformer,
        [torch.device(name) for name in transformer_devices],
        args.blocks_per_device,
    )

    manifest = []
    for item in prompts:
        # A fresh generator per prompt, always from the same seed: the card's
        # snippet builds one for its single image, so seeding once and letting
        # the stream run on would make image N depend on how many ran before it.
        generator = torch.Generator(device=args.device).manual_seed(SEED)
        path = out_dir / f"{item['id']}.png"

        print(f"[{item['id']}] {len(item['prompt'])} chars: {item['prompt'][:64]}...")
        started = time.perf_counter()
        image = pipe(
            prompt=item["prompt"],
            negative_prompt=negative_prompt,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            generator=generator,
        ).images[0]
        elapsed = time.perf_counter() - started
        image.save(path)
        memory = footprint(torch, encoder)
        print(f"  -> {path}  {image.size}  {elapsed:.1f}s")
        if memory is not None:
            print(
                f"     {encoder} reserved {memory['reserved_gib']:.2f} GiB "
                f"allocated {memory['allocated_gib']:.2f} GiB"
            )

        manifest.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "file": path.name,
                "seconds": round(elapsed, 1),
                "memory": memory,
            }
        )
        (out_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "device": args.device,
                    "torch": torch.__version__,
                    "model": args.model,
                    "negative_prompt": negative_prompt,
                    "width": WIDTH,
                    "height": HEIGHT,
                    "num_inference_steps": STEPS,
                    "true_cfg_scale": TRUE_CFG_SCALE,
                    "seed": SEED,
                    # The placement is part of the record: the same prompt on a
                    # different card count is a different measurement, and a
                    # manifest that omits it cannot be compared with a later run.
                    "placement": {
                        "encoder": encoder,
                        "vae": encoder,
                        "transformer": transformer_devices,
                        "blocks_per_device": blocks_per_device,
                    },
                    "images": manifest,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        release(torch, args.device)

    total = sum(entry["seconds"] for entry in manifest)
    print(
        f"{len(manifest)} images, {total:.1f}s total, {total / len(manifest):.1f}s mean"
    )
    common.report_memory(torch, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
