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

"""Qwen-Image-2.1 over a fixed prompt cohort, one PNG per prompt plus a manifest.

The cohort is in ``prompts.py`` with its provenance. Unlike Qwen-Image-2512 --
whose sweep reads the prompts out of the model card -- 2.1's card publishes none,
so the cohort is written down rather than discovered, and
``--prompts-file`` is the escape hatch for testing a different one.

Nothing here branches on the backend: prompts, resolution, steps and seed are
shared, so a pair of runs (vendor torch and flagos) is directly comparable. Two
backends do not share an RNG stream, so the same seed does not give the same
initial noise -- compare what the images depict, not their pixels, or inject the
same latents with ``infer.py --load-latents``, which is the only injection 2.1
supports.

``torch_fl`` must be imported before ``torch`` on the flagos side: it preloads
the bundled ``libtorch_cuda.so`` that the CPU torch wheel dlopens.

Run:
    python tests/manual/qwen_image_21/sweep.py --device cuda --output-dir out/cuda
    python tests/manual/qwen_image_21/sweep.py --device flagos --output-dir out/flagos

Procedure: tests/manual/qwen_image_21/README.md
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import common
import prompts

STEPS = 40
TRUE_CFG_SCALE = 1.0
SEED = 42


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Qwen-Image-2.1 over the prompt cohort",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common.add_placement_args(parser)
    parser.add_argument(
        "--model",
        default=common.model_ref(),
        help=f"a directory or a hub id; default: ${common.MODEL_ENV} or its fallback",
    )
    parser.add_argument("--output-dir", required=True, help="one PNG per prompt")
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="JSON list of {id, prompt}; default: the cohort in prompts.py",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="run just these prompt ids, e.g. --only 01 07",
    )
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=TRUE_CFG_SCALE,
        help="1.0 disables guidance, which is how 2.1 is meant to be sampled",
    )
    parser.add_argument("--negative-prompt", default=prompts.NEGATIVE_PROMPT)
    return parser.parse_args(argv)


def release(torch, device_kind):
    """Hand the card back between prompts.

    The weights are 33 GB and a 40 GB card has little to spare once activations
    and the allocator's own overhead are counted, so a sweep is a memory test as
    much as a quality one. Collecting and emptying the cache costs a few hundred
    milliseconds against a prompt that takes a minute, and it keeps the
    per-prompt footprints in the log comparable.
    """
    gc.collect()
    module = common.device_module(torch, device_kind)
    if hasattr(module, "empty_cache"):
        module.empty_cache()


def footprint(torch, devices):
    """Reserved and allocated on every card the placement uses, keyed by card.

    Each name is a resolved device -- ``flagos:2``, not ``flagos``. The counters
    are per card, and an index-less query answers for the backend's current
    device rather than the one the placement named, which silently records
    0.00 GiB for a prompt that just allocated 33 GB.

    Every card, rather than the transformer's first: the encoder and the VAE are
    18.85 GB of the 33, and a run whose transformer is split across cards can
    have those on a card the transformer's first shard is not on. Reporting one
    card would then describe a different one than the reading came from.
    """
    readings = {}
    for device in devices:
        resolved = torch.device(device)
        module = common.device_module(torch, resolved.type)
        if not hasattr(module, "memory_reserved"):
            return None
        reserved = module.memory_reserved(resolved.index) / (1024**3)
        allocated = module.memory_allocated(resolved.index) / (1024**3)
        readings[str(resolved)] = {
            "reserved_gib": round(reserved, 2),
            "allocated_gib": round(allocated, 2),
        }
    return readings


def main(argv=None):
    args = parse_args(argv)

    torch = common.import_torch(args.device)
    diffusers = common.import_diffusers()

    cohort = prompts.load(args.prompts_file)
    if args.only:
        wanted = set(args.only)
        cohort = [entry for entry in cohort if entry["id"] in wanted]
        if not cohort:
            raise SystemExit(f"no prompt matched --only {args.only}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder, transformer, vae = common.resolve_placement(torch, args)
    print(f"device: {args.device}   torch: {torch.__version__}")
    print(f"diffusers {diffusers.__version__}")
    print(f"model: {args.model}")
    print(
        f"placement: encoder -> {encoder}   transformer -> "
        f"{' '.join(transformer)}   vae -> {vae}"
    )
    print(
        f"{len(cohort)} prompts, {args.width}x{args.height}, {args.steps} steps, "
        f"true_cfg_scale={args.true_cfg_scale}, seed={args.seed}"
        + ("" if args.true_cfg_scale > 1.0 else " (guidance off)")
    )
    print(f"-> {out_dir}")

    print(f"loading {args.model} (this materialises ~33 GB on the host)")
    pipe = diffusers.QwenImage21Pipeline.from_pretrained(
        args.model, dtype=torch.bfloat16
    )
    _, placement = common.place_components(
        torch, pipe, encoder, transformer, vae, args.blocks_per_device
    )
    if args.omit_all_valid_prompt_mask:
        common.enable_all_valid_prompt_mask_elision(pipe)

    manifest = {
        "device": args.device,
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "model": args.model,
        "placement": placement,
        "prompts_file": args.prompts_file,
        "prompts_sha256": {entry["id"]: entry["sha256"] for entry in cohort},
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale,
        "omit_all_valid_prompt_mask": args.omit_all_valid_prompt_mask,
        "seed": args.seed,
        "images": [],
    }

    started = time.time()
    # The cards the components were placed on, encoder first, each once: the
    # manifest reports one footprint per card rather than one per run, because
    # a split run has components on more than one.
    cards = list(dict.fromkeys([placement["encoder"], *placement["transformer"]]))
    for entry in cohort:
        # A generator per prompt, seeded identically: the pipeline does not
        # reset one between prompts, so a shared generator would make every
        # prompt after the first depend on its predecessors.
        generator = torch.Generator(device=torch.device(transformer[0])).manual_seed(
            args.seed
        )
        prompt_started = time.time()
        with torch.no_grad():
            result = pipe(
                prompt=entry["prompt"],
                negative_prompt=args.negative_prompt
                if args.true_cfg_scale > 1.0
                else None,
                true_cfg_scale=args.true_cfg_scale,
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                generator=generator,
                output_type="pil",
            )
        seconds = round(time.time() - prompt_started, 1)

        path = out_dir / f"{entry['id']}.png"
        result.images[0].save(path)
        record = {
            "id": entry["id"],
            "prompt": entry["prompt"],
            "prompt_sha256": entry["sha256"],
            "file": path.name,
            "seconds": seconds,
            "memory": footprint(torch, cards),
        }
        manifest["images"].append(record)
        print(f"  -> {path}  {result.images[0].size}  {seconds}s")

        # Written after every prompt so an interrupted sweep still describes what
        # it produced, which is what makes a partial run usable.
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        release(torch, args.device)

    total = round(time.time() - started, 1)
    # The mean of these is not reported: eight heterogeneous prompts whose first
    # run pays for a cold Triton cache do not have a mean latency, and quoting
    # one would invite a comparison against a real benchmark number. Each
    # prompt's own seconds is in the manifest; `run.sh bench` is what produces a
    # number that can be quoted.
    print(f"{len(cohort)} images, {total}s total")
    print("these seconds are recorded, not measured -- see the README, §5")
    common.report_memory(torch, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
