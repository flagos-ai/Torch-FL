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

"""Where does the peak accelerator memory go? Per-phase allocator accounting.

Runs the Qwen-Image-2512 pipeline with the same placement as
``tests/manual/qwen_image_2512/infer.py`` but resets the allocator's peak
counters around every phase, so each entry in the report is that phase's own
peak rather than a running maximum. Run it on flagos and on stock CUDA torch and
diff the two tables.

Both sides read the same counters: flagos delegates to the CUDA caching
allocator, so ``torch.flagos.memory_stats`` and ``torch.cuda.memory_stats`` are
two views of one pool. Reporting anything else -- nvidia-smi, for instance --
would fold the CUDA context and the cuBLAS/cuDNN workspaces into one of the two
numbers and make them incomparable.

The last section of the report covers memory the phase tables cannot see: what
FlagGems' autotuner is holding. That is what made torch-fl run 1.38 GiB above
stock CUDA on this workload, and it is fixed in flagos-ai/FlagGems#6386 -- so on
a current FlagGems the section reports "nothing pinned", and a number there means
the installed FlagGems predates the fix. See the README's "Where the extra memory
went" section.

    python tests/manual/qwen_image_2512/memprobe.py --device flagos --stage vae
    python tests/manual/qwen_image_2512/memprobe.py --device cuda   --stage vae
"""

import argparse
import gc
import sys

# Bound in main() before the pipeline runs: the stats reader and the peak
# resetter, which are the only two allocator calls this file makes and the only
# two that differ between the two backends.
STATS = None
RESET_PEAK = None


def normalize(stats):
    """One shape for both backends' stats dicts.

    flagos returns a flat dict of scalars; ``torch.cuda.memory_stats`` returns a
    flat dict whose keys are dotted ``<metric>.<stat type>`` paths. Both come
    from the same CUDACachingAllocator counters, which is what makes the two
    tables comparable.
    """
    if "allocated_bytes" in stats and isinstance(stats["allocated_bytes"], int):
        return {
            "allocated": stats["allocated_bytes"],
            "reserved": stats["reserved_bytes"],
            "peak_allocated": stats["peak_allocated_bytes"],
            "peak_reserved": stats["peak_reserved_bytes"],
            "alloc_calls": stats["num_alloc_calls"],
            "device_mallocs": stats["num_device_malloc"],
            "retries": stats["num_alloc_retries"],
        }
    return {
        "allocated": stats["allocated_bytes.all.current"],
        "reserved": stats["reserved_bytes.all.current"],
        "peak_allocated": stats["allocated_bytes.all.peak"],
        "peak_reserved": stats["reserved_bytes.all.peak"],
        "alloc_calls": stats["allocation.all.allocated"],
        "device_mallocs": stats["num_device_alloc"],
        "retries": stats["num_alloc_retries"],
    }


def read(device_index):
    return normalize(STATS(device_index))


def reset(device_index):
    RESET_PEAK(device_index)


GIB = 1024.0**3


class Report:
    """Records one row per phase per device, prints one table per device.

    Every row is a delta over that phase alone: peak counters are reset when the
    phase opens, so ``peak_alloc`` is what that phase needed, not the running
    maximum. ``cached`` is reserved-minus-allocated at the phase's end -- the
    allocator's retained free pool, which is the number that shows fragmentation.

    A phase spanning several devices (the denoise loop, whose forwards are split
    across shards) records a row on each, so a phase's cost can be attributed to
    the card that paid it rather than only to the one that was asked first.
    """

    def __init__(self, device_indices):
        self.device_indices = list(device_indices)
        self.rows = {index: [] for index in self.device_indices}

    def phase(self, name, fn):
        before = {index: read(index) for index in self.device_indices}
        for index in self.device_indices:
            reset(index)
        result = fn()
        for index in self.device_indices:
            after = read(index)
            self.rows[index].append(
                {
                    "phase": name,
                    "end_alloc": after["allocated"],
                    "peak_alloc": after["peak_allocated"],
                    # Reserved peaked at some point during the phase; the
                    # current value would hide a phase that grew the pool and
                    # gave it back.
                    "reserved": after["reserved"],
                    "peak_reserved": after["peak_reserved"],
                    "cached": after["reserved"] - after["allocated"],
                    "d_alloc_calls": after["alloc_calls"]
                    - before[index]["alloc_calls"],
                    "d_mallocs": after["device_mallocs"]
                    - before[index]["device_mallocs"],
                    "retries": after["retries"],
                }
            )
        return result

    def print_report(self, label, device_index=None):
        print()
        print(f"===== {label} =====")
        head = (
            f"{'phase':<26}{'peak_alloc':>11}{'end_alloc':>11}"
            f"{'reserved':>11}{'cached':>10}{'calls':>10}{'mallocs':>9}{'retry':>7}"
        )
        print(head)
        print("-" * len(head))
        indices = self.device_indices if device_index is None else [device_index]
        for row in self.rows[indices[0]]:
            print(
                f"{row['phase']:<26}"
                f"{row['peak_alloc'] / GIB:>11.3f}"
                f"{row['end_alloc'] / GIB:>11.3f}"
                f"{row['reserved'] / GIB:>11.3f}"
                f"{row['cached'] / GIB:>10.3f}"
                f"{row['d_alloc_calls']:>10}"
                f"{row['d_mallocs']:>9}"
                f"{row['retries']:>7}"
            )
        print("(all GiB)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="flagos")
    parser.add_argument("--stage", default="full", choices=("vae", "full"))
    parser.add_argument("--model", default="Qwen/Qwen-Image-2512")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--encoder-device",
        default=None,
        help="default: the pipeline's execution device, as in infer.py",
    )
    parser.add_argument("--transformer-devices", nargs="+", default=None)
    parser.add_argument("--split-at", type=int, default=30)
    parser.add_argument(
        "--phase-steps",
        type=int,
        default=5,
        help="how many denoise steps to time individually before the rest",
    )
    return parser.parse_args(argv)


def main(argv=None):
    global STATS, RESET_PEAK
    args = parse_args(argv)

    if args.device == "flagos":
        import torch_fl  # noqa: F401  - must precede `import torch`
    import torch

    if args.device == "flagos":
        import torch_fl.flagos as flagos_module

        STATS = flagos_module.memory_stats
        RESET_PEAK = flagos_module.reset_peak_memory_stats
        device_count = torch.flagos.device_count()
    else:
        STATS = torch.cuda.memory_stats
        RESET_PEAK = torch.cuda.reset_peak_memory_stats
        device_count = torch.cuda.device_count()

    base = args.device
    count = min(device_count, 3)
    encoder_device = args.encoder_device or f"{base}:{count - 1}"
    transformer_devices = args.transformer_devices or [
        f"{base}:{i}" for i in range(count - 1)
    ]
    encoder_dev = torch.device(encoder_device)
    transformer_devs = [torch.device(d) for d in transformer_devices]

    print(f"device={args.device} torch={torch.__version__}")
    print(f"encoder/vae -> {encoder_device}   transformer -> {transformer_devices}")

    # Every device this run will touch has to have its allocator context up
    # before the first stats read: an uninitialized device reports an EMPTY
    # stats dict on the CUDA path, so the opening row of the report would fail
    # rather than read zero.
    for index in {encoder_dev.index, *(d.index for d in transformer_devs)}:
        with torch.device(f"{base}:{index}"):
            torch.zeros(1, dtype=torch.float32, device=f"{base}:{index}")
        reset(index)

    # One Report over every card the run touches: a phase that spans the split
    # then records a row on each shard, not only on the one asked for its stats.
    used_devices = [encoder_dev, *transformer_devs]
    used_indices = sorted({dev.index for dev in used_devices})
    rep = Report(used_indices)

    from diffusers import QwenImagePipeline

    pipe = rep.phase(
        "from_pretrained",
        lambda: QwenImagePipeline.from_pretrained(args.model, dtype=torch.bfloat16),
    )

    def move_encoder():
        pipe.text_encoder.to(encoder_dev)
        pipe.vae.to(encoder_dev)

    rep.phase("encoder+vae -> device", move_encoder)

    # The vae stage never calls the transformer, so it must not pay to put 40 GB
    # of weights on the cards either -- same rule as infer.py's placement.
    if args.stage != "vae":

        def place_transformer():
            from accelerate import dispatch_model

            blocks = list(pipe.transformer.transformer_blocks)
            device_map = {}
            for name, _ in pipe.transformer.named_children():
                if name != "transformer_blocks":
                    device_map[name] = transformer_devs[0]
            for i in range(len(blocks)):
                device_map[f"transformer_blocks.{i}"] = (
                    transformer_devs[0] if i < args.split_at else transformer_devs[1]
                )
            return dispatch_model(pipe.transformer, device_map=device_map)

        pipe.transformer = rep.phase("dispatch transformer", place_transformer)

    def encode():
        with torch.no_grad():
            return pipe.encode_prompt(
                prompt="a red panda eating bamboo, soft morning light",
                device=encoder_dev,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )

    prompt_embeds, prompt_embeds_mask = rep.phase("text encoder forward", encode)

    # Where the latents are built follows the stage, matching infer.py: the vae
    # stage decodes on the encoder's card, the full stage denoises on the first
    # transformer shard and lets the pipeline's own hooks move things.
    latent_device = encoder_dev if args.stage == "vae" else transformer_devs[0]
    generator = torch.Generator(device=latent_device).manual_seed(args.seed)

    def make_latents():
        return pipe.prepare_latents(
            1,
            pipe.transformer.config.in_channels // 4,
            args.height,
            args.width,
            pipe.transformer.dtype,
            latent_device,
            generator,
        )

    latents = rep.phase("prepare latents", make_latents)

    if args.stage == "vae":
        return finish(torch, pipe, args, latents, rep)

    prompt_embeds = prompt_embeds.to(latent_device)
    if prompt_embeds_mask is not None:
        prompt_embeds_mask = prompt_embeds_mask.to(latent_device)

    # The loop keeps the scheduler state on the encoder's card and moves the
    # hidden states across the split explicitly, which is what accelerate's
    # io_same_device hook does in the real pipeline -- otherwise this probe
    # would measure a layout the actual run never has.
    def run_loop():
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
            calculate_shift,
            retrieve_timesteps,
        )

        import numpy as np

        sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
        mu = calculate_shift(
            latents.shape[1],
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler, args.steps, torch.device("cpu"), sigmas=sigmas, mu=mu
        )
        img_shapes = [
            [
                (
                    1,
                    args.height // pipe.vae_scale_factor // 2,
                    args.width // pipe.vae_scale_factor // 2,
                )
            ]
        ]
        # The pipeline builds its latents on the execution device, which is the
        # encoder's card; that is where the scheduler steps them and where the
        # transformer's output hook moves each noise prediction back to.
        state = latents.to(encoder_dev)

        def one_step(t):
            def one_forward():
                return pipe.transformer(
                    hidden_states=state.to(transformer_devs[0]).to(
                        pipe.transformer.dtype
                    ),
                    timestep=t.expand(state.shape[0]).to(state.dtype) / 1000,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]

            noise = one_forward().to(encoder_dev)
            return pipe.scheduler.step(noise, t, state, return_dict=False)[0]

        with torch.no_grad():
            for index, t in enumerate(timesteps):
                if index < args.phase_steps:
                    # Recorded on every card, so the per-step rows of each
                    # device's table show what that step cost there.
                    state = rep.phase(f"step {index}", lambda: one_step(t))
                else:
                    state = one_step(t)
        return state

    latents = rep.phase("denoise loop", run_loop)

    return finish(torch, pipe, args, latents, rep)


def finish(torch, pipe, args, latents, rep):
    """Unpack, normalise and decode -- the tail every stage shares.

    ``latents`` is the packed ``(B, S, C)`` form, which is what
    ``prepare_latents`` returns and what the denoise loop carries, so both the
    vae stage and the full stage hand the same thing in.
    """

    def unpack():
        return pipe._unpack_latents(
            latents, args.height, args.width, pipe.vae_scale_factor
        )

    unpacked = rep.phase("unpack latents", unpack)

    def decode():
        with torch.no_grad():
            z_dim = pipe.vae.config.z_dim
            x = unpacked.to(pipe.vae.dtype)
            # The pipeline, not the VAE, applies these two; a decode driven
            # outside __call__ has to repeat them.
            x = x / (
                1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, z_dim, 1, 1, 1)
            ).to(x.device, x.dtype)
            x = x + torch.tensor(pipe.vae.config.latents_mean).view(
                1, z_dim, 1, 1, 1
            ).to(x.device, x.dtype)
            return pipe.vae.decode(x, return_dict=False)[0][:, :, 0]

    image = rep.phase("vae decode", decode)

    def postprocess():
        return pipe.image_processor.postprocess(image, output_type="pil")

    rep.phase("postprocess", postprocess)

    report_pinned_tensors(rep)

    for index in sorted(rep.device_indices):
        rep.print_report(f"{args.device}:{index}", device_index=index)
    return 0


def report_pinned_tensors(rep):
    """Account for memory the FlagGems autotuner is holding on to.

    ``LibTuner.run`` keeps the argument tuple of the call it just ran so a later
    ``benchmark_config`` can replay it. Up to flagos-ai/FlagGems#6386 it kept
    them strongly, which pinned the tensors of every tuned kernel's most recent
    call for the life of the process. A phase table cannot show that -- nothing
    allocates while it happens -- so it is measured here instead: count what is
    held, drop it, and read the allocator again. The difference is that cost, and
    the rows the tables print afterwards are what they would read without it.

    On a FlagGems carrying that fix this prints "nothing pinned", so a number
    here means the installed FlagGems predates it and is worth upgrading. Where
    FlagGems is not importable at all there is nothing to find either.
    """
    gc.collect()
    tuned = _tuner_instances()
    if tuned is None:
        print("\nflag_gems LibTuner is not importable here; nothing to measure")
        return

    pinned = _pinned_tensors(tuned)
    print()
    print("===== memory pinned by the FlagGems autotuner =====")
    if not pinned:
        print("  nothing pinned")
        return
    for index, entries in sorted(pinned.items()):
        total = sum(size for _, _, size in entries)
        print(f"  device {index}: {total / GIB:.3f} GiB over {len(entries)} tensors")
        for kernel, shape, size in sorted(entries, key=lambda e: -e[2])[:6]:
            print(f"      {kernel:<22}{str(shape):<24}{size / GIB:>7.3f} GiB")
    print("  (FlagGems' LibTuner._last_benchmark_args; released next)")

    rep.phase("drop pinned autotuner args", lambda: _clear_tuner_args(tuned))
    for index in sorted(pinned):
        row = rep.rows[index][-1]
        print(
            f"  device {index} after release: allocated "
            f"{row['end_alloc'] / GIB:.3f} GiB, reserved {row['reserved'] / GIB:.3f} GiB"
        )


def _tuner_instances():
    """Every live LibTuner, or None when flag_gems is not importable."""
    try:
        import importlib

        libentry = importlib.import_module("flag_gems.utils.libentry")
    except Exception:
        return None
    found = []
    for obj in gc.get_objects():
        if isinstance(obj, libentry.LibTuner) or type(obj).__name__.endswith(
            "LibTunerImpl"
        ):
            found.append(obj)
    return found


def _kernel_name(tuner):
    """The Triton kernel a tuner wraps, for the report's left-hand column."""
    fn = getattr(tuner, "fn", None)
    for _ in range(4):
        name = getattr(fn, "__name__", None)
        if name:
            return name
        fn = getattr(fn, "fn", None)
        if fn is None:
            break
    return type(tuner).__name__


def _pinned_tensors(tuners):
    """Device tensors the tuners hold, per device index, as (shape, bytes).

    The total over-counts what is actually *extra*: a weight passed into a
    kernel would be alive anyway, and the same tensor can appear in more than
    one tuner's argument tuple. The allocator delta measured by releasing them
    is the honest figure; this list is for seeing which kernels are responsible.
    """
    import torch

    pinned = {}
    for tuner in tuners:
        for arg in getattr(tuner, "_last_benchmark_args", ()) or ():
            if not isinstance(arg, torch.Tensor) or arg.device.type == "cpu":
                continue
            pinned.setdefault(arg.device.index, []).append(
                (
                    _kernel_name(tuner),
                    tuple(arg.shape),
                    arg.numel() * arg.element_size(),
                )
            )
    return pinned


def _clear_tuner_args(tuners):
    for tuner in tuners:
        tuner._last_benchmark_args = ()
        tuner._last_benchmark_meta = {}
    gc.collect()


if __name__ == "__main__":
    sys.exit(main())
