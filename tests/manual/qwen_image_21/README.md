# Qwen-Image-2.1 manual test flow

`Qwen/Qwen-Image-2.1` is a text-to-image diffusion model: a Qwen3-VL text
encoder, a 32-block single-stream MMDiT transformer with a causal KV cache, and a
causal-Conv3d VAE, driven by `QwenImage21Pipeline`. Running it on an accelerator
is the broadest check in this repository after Qwen-Image-2512 — it reaches
operator surfaces no Transformers test does (Qwen3-VL instead of Qwen2.5-VL,
gated MLPs, causal attention over a prefix, 3-D convolution) and it fails loudly
rather than subtly when a route is missing.

Nothing under `tests/manual/` runs in CI, so this is a bring-up procedure, not a
gate. It is written to be run on a chip that has never run it: every stage is
isolated, the placement derives from the cards it finds, and the readings a new
chip should produce are stated so a result can be judged rather than guessed at.

## Files

| File | Role |
| --- | --- |
| `infer.py` | The staged runner. Four stages, latents save/load for paired runs. |
| `bench.py` | The measurement: one JSON of numbers, and `--table` to render several. §7.1. |
| `sweep.py` | The prompt cohort, one PNG per prompt plus a manifest. |
| `numerics.py` | Per-operator comparison against a reference backend. §8. |
| `prompts.py` | The cohort, its provenance, and the one canonical prompt. §5. |
| `common.py` | Diffusers checkout, import order, placement, memory reporting, phase timing. |
| `run.sh` | Wrapper: sets the environment, runs one mode, summarises the log. |

`infer.py` and `sweep.py` answer *whether* a chip runs the pipeline and *where*
it fails. `bench.py` answers *how fast*, and it is the only thing here that
produces a number worth quoting — §7.1 states the protocol it has to satisfy for
the number to mean anything.

`compare.py` and `side_by_side.py` are model-agnostic — they read PNGs and a
manifest and never import torch — so they live once, in
`tests/manual/qwen_image_2512/`, and `run.sh` dispatches to them. Two copies
would drift.

## 1. Prerequisites

| Requirement | Detail |
| --- | --- |
| torch_fl | Built for the chip and importable. The device is always `flagos` — `rename_privateuse1_backend("flagos")` is not vendor-dependent. |
| A diffusers source checkout | **A release will not do.** See §2. |
| Python packages | `huggingface-hub>=1.26,<2`, `torchvision`, `accelerate` (only for a split), `pillow`. |
| Model | ~33 GB of bf16 weights over 7 files, in a directory or the Hub cache. |
| Host RAM | ~35 GB free. `from_pretrained` materialises the pipeline on the host before any component moves to a card. |
| Accelerator memory | **38.6 GiB reserved** for the whole pipeline on one card, of 39.5 GiB usable on a 40 GB part. See §1.2 — this is tight enough that the allocator's fragmentation decides it. |

### 1.2 Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Not a tuning suggestion: on a 40 GB card it is the difference between running and
not. Measured on A100, same pipeline and same two steps: with expandable segments
the run peaks at 38.61 GiB reserved and completes; without it the allocator
reserves essentially the whole card (39.38 GiB) and still cannot find 576 MiB, so
the run dies in the denoise loop with `CUDA out of memory` while reporting
1.62 GiB "reserved but unallocated". The reserved-but-unallocated figure in that
message is the signature: it is fragmentation, not a shortage of memory.

A chip whose card is smaller than 40 GB, or one where reserved memory is measured
differently, should use the split placement in §3 instead of relying on this.

`import torch_fl` must precede `import torch`. torch_fl's `_preload_cuda_assets()`
dlopens the bundled `libtorch_cuda.so` before PyTorch caches its CUDA hooks;
importing torch first leaves the assets unloaded and the run dies at the first
generator with `Cannot get CUDA generator without ATen_cuda library`. The runner
does this for you; a hand-written snippet has to do it too.

### 1.1 The two packages a release does not cover

`huggingface-hub>=1.26`: the 2.1 checkout imports `resolve_revision` from
`huggingface_hub`, which 1.24 (the version several environments pin) does not
have. The failure is an `ImportError` out of `diffusers/utils/hub_utils.py`
during `import diffusers`, and it names neither diffusers nor the checkout.

`torchvision`: the Qwen3-VL processor loads a `Qwen3VLVideoProcessor`, which
requires it. Without it, `from_pretrained` dies with `ImportError: Qwen3VLVideoProcessor
requires the Torchvision library`. Install the build that matches the torch
already in the environment (`--no-deps`, from the matching torch index) — a
plain `pip install torchvision` can pull a different torch with it, which breaks
a CPU-wheel-plus-external-`libtorch_cuda.so` setup.

## 2. The diffusers checkout

`QwenImage21Pipeline`, `QwenImage21Transformer2DModel` and
`AutoencoderKLQwenImage21` are in **no released diffusers**. Verified 2026-09-17:

| Where looked | Result |
| --- | --- |
| diffusers 0.40.0 (newest on PyPI, 2026-08-20) | none of the three |
| diffusers `main` file tree | only `transformer_qwenimage.py` (the 2512 one) |
| GitHub code search, all repositories | 0 hits |
| all 900 diffusers branches | none |

The model declares `_diffusers_version: "0.37.0.dev0"` — an unreleased internal
tree. Point `QWEN_IMAGE_21_DIFFUSERS` at a checkout of it (any `src/` directory
containing `diffusers/pipelines/qwenimage21/`); `common.import_diffusers()`
prepends it to `sys.path`, which is a prepend rather than an install precisely so
that an environment's diffusers 0.40.0 is left alone and one interpreter can
serve both this flow and every other test in the repository.

If the variable is unset or wrong, the run stops with a message naming the
variable and the missing directory rather than an `ImportError` from deep inside
diffusers. If diffusers is importable but has no `QwenImage21Pipeline` — a
prepend that did not take — it says so.

## 3. Placement

`--device` names a torch device module: `flagos` (imports torch_fl first),
`cuda`, or any vendor name a plugin registered.

**Everything goes on one card by default.** 33 GB of bf16 weights (17.5 text
encoder, 14.2 transformer, 1.35 VAE) fits a 40 GB part with 37.6 GiB reserved,
and spreading it across two cards would add a hidden-state round trip on every
forward for nothing. This is deliberately the opposite of the 2512 flow, whose
transformer alone is 40.9 GB and must be split.

The resolved placement is: the transformer gets every card named by `--devices`
(one card, `<device>:0`, when none are named), and the text encoder and the VAE
go on the first of them. So:

```bash
# one card: everything on it. The default.
tests/manual/qwen_image_21/run.sh infer --stage full

# a chip whose card cannot hold 14.2 GB: name several cards and the transformer
# is split across them, encoder and VAE staying on the first
tests/manual/qwen_image_21/run.sh infer --stage full --devices flagos:0 flagos:1

# the same placement, written out. Use this when the shards are not the whole
# of --devices
tests/manual/qwen_image_21/run.sh infer --stage full \
    --devices flagos:0 --transformer-devices flagos:0 flagos:1
```

The split goes through `accelerate.dispatch_model` with a hand-built device map,
and it is measured working on A100: blocks 0–15 land on the first card, 16–31 on
the second, `norm_out` stays on the first, and a two-step run completes.
`--blocks-per-device` moves the boundary and is checked against the model's block
count rather than trusted.

The encoder and the VAE must share a card, and that is enforced rather than
assumed. `DiffusionPipeline._execution_device` falls back to `self.device`, the
first non-CPU component in `_get_signature_keys` order — and that order is
*sorted*, so `text_encoder` outranks `transformer` and `vae`. The pipeline builds
its latents and decodes them on the encoder's card, so the VAE has to be there
too, and `common.colocated()` stops a full run whose `--vae-device` names a
different one — before the weights are read, rather than as a cross-device error
inside the decode several minutes later. `--vae-device` therefore exists for
`infer.py --stage vae`, which places the VAE alone and is not checked.

`bench.py` and `sweep.py` record the placement that was **applied**, from
`common.place_components`, not the one that was requested. Where they disagree,
the record is what the run did.

## 4. The stages

Run them in order. A failure in a 40-step loop with a KV cache is far more
expensive to localise than a failure in one isolated call.

```bash
export PYTHON=/path/to/the/env/with/torch_fl/bin/python

# 0. The backend answers at all.
$PYTHON -c "import torch_fl, torch; print(torch.__version__, torch.flagos.device_count())"

# 1. Text encoder: prompt -> prompt_embeds + mask + img_mask.
tests/manual/qwen_image_21/run.sh infer --stage text-encoder

# 2. One transformer forward at fixed latents.
tests/manual/qwen_image_21/run.sh infer --stage transformer-step

# 3. VAE decode of fixed latents -> PNG.
tests/manual/qwen_image_21/run.sh infer --stage vae

# 4. The whole pipeline: 1024x1024, 40 steps, one PNG.
tests/manual/qwen_image_21/run.sh infer --stage full
```

What each stage prints, and what a pass looks like:

| Stage | Pass |
| --- | --- |
| `text-encoder` | `prompt_embeds (1, 26, 4096) bfloat16 flagos:N`, `img_mask slots 0 True of 26`, then `device check : OK` |
| `transformer-step` | `joint sequence (1, 4122, 64)`, then `transformer noise (1, 4096, 64)` on the last shard and `device check : OK` |
| `vae` | `unpacked (1, 64, 1, 64, 64)`, `decoded (1, 4, 1024, 1024)`, and a PNG whose `std` is not ~0 |
| `full` | a PNG, then per-card `reserved`/`allocated`. A mean pixel value near 0 or a `std` near 0 means the decode produced nothing. |

Two numbers in those rows are worth reading as diagnostics rather than as
decoration. The `joint sequence` width is `text tokens + 4096` — 4122 with the
default prompt — and that single number says the `img_mask`, the latent count and
the layout all line up; a wrong one produces a shape error here instead of a
subtly wrong attention pattern later. `img_mask slots … of 26` being 0 True slots
confirms the prompt carried no condition image, which is the only case this flow
exercises.

Stage 3 is the cheapest way to prove the whole memory layout: it loads no
transformer weights onto a card at all.

### 4.1 The KV cache changes the image, so fix it per run

`use_kv_cache` defaults on, and the pipeline's own help is explicit that toggling
it does **not** reproduce the same image bit-for-bit: caching makes the decode
step attend with a different sequence layout than the prefill step, so the two
tile differently and land on different rounding — both valid, visibly distinct.
`--no-kv-cache` disables it. Fix the flag per comparison; do not mix.

### 4.2 Determinism

Two runs at the same seed on the same backend must be bit-identical. This is a
property of the backend and it is a prerequisite for §6's paired comparison.

```bash
tests/manual/qwen_image_21/run.sh infer --stage full --seed 42 --output a.png
tests/manual/qwen_image_21/run.sh infer --stage full --seed 42 --output b.png
cmp a.png b.png && echo "bit-identical"
```

Measured on torch-fl/A100: two runs, bit-identical, and the same for the vendor's
CUDA torch. (The 2512 flow did **not** satisfy this — its flagos runs produced
two different images across four identical invocations. 2.1 does, which makes the
paired comparison below meaningful here.)

### 4.3 A cold compile cache is a different machine

The FlagGems path JIT-compiles Triton kernels. On a box whose cache is empty the
first run of each component compiles, and the cost is large and one-off: on A100,
going from an empty cache to a warm one took the text encoder from 12.6 s to
2.0 s, the VAE decode from 14.3 s to 0.9 s, and the whole 40-step run from
71.5 s to 26.4 s. **Warm the cache before quoting any timing** — otherwise the
number measures Triton, not the chip.

## 5. The prompt cohort

`sweep.py` runs the cohort in `prompts.py`, one PNG per prompt plus a
`manifest.json` carrying each prompt, its own sha256, its timing and the
placement used:

```bash
tests/manual/qwen_image_21/run.sh sweep
tests/manual/qwen_image_21/run.sh sweep --only 01 07   # a subset, for a smoke run
```

**2.1's model card publishes no prompts** — its ModelScope README is the
auto-generated stub and the model repository carries no example snippet and no
showcase section. There is therefore nothing to read a cohort out of, the way
`qwen_image_2512/sweep.py` reads the 2512 card. `prompts.py` writes the cohort
down and states the provenance of every entry: `01` is the only prompt the 2.1
authors published (the example in `QwenImage21Pipeline`'s own docstring), and the
rest were written for this flow to reach one operator surface each — fine
detail, rendered text, countable geometry, a second resolution, a long prompt, a
compressed dynamic range, and a one-phrase variant of `01`. Pass
`--prompts-file` (JSON, `[{"id": ..., "prompt": ...}]`) to test a different one.

One entry is also `CANONICAL_ID`, and it is `01`: the prompt every performance
run uses. `prompts.canonical()` returns it, `infer.py` takes it as its default,
and `bench.py` has no flag that could replace it — see §7.1.

**The sweep's `seconds` is not a benchmark number.** It is one `time.time()`
delta per prompt, taken with no warmup, so the first prompt carries the whole
Triton compile (measured in §4.3 as a 3x difference on the same hardware), and
the eight prompts are different workloads whose average would not mean anything.
The manifest records it so a run is describable; §7.1 is where a quotable number
comes from.

Images land in one directory per backend under `OUT_DIR`, named after the
backend's device module — `<OUT_DIR>/flagos`, `<OUT_DIR>/cuda` — so the vendor
run and the flagos run of the same sweep sit side by side instead of overwriting
each other.

## 6. Vendor against flagos, side by side

```bash
PYTHON=/path/to/the/vendor/env/bin/python tests/manual/qwen_image_21/run.sh \
    sweep --device cuda           # or musa, npu: whatever the chip's torch calls it
PYTHON=/path/to/the/torch_fl/env/bin/python tests/manual/qwen_image_21/run.sh \
    sweep --device flagos
tests/manual/qwen_image_21/run.sh side-by-side --left cuda --right flagos
```

The last command writes `compare/01.png` … `compare/08.png`, each one prompt with
the vendor's image on the left and flagos' on the right, captioned with the device
and the generation time, plus `compare/all.png` holding every pair at once — that
is the sheet to look at first.

At the cohort's settings the two backends do not share an RNG stream, so the
images differ in composition as well as in detail. What is being judged is
whether flagos' pictures are as coherent as the vendor's. The paired run below is
the one that can be given a number.

This works when the chip has no torch of its own as well: run the left side on
any working backend — an A100 somewhere, or the CPU reference — carry the two
directories over, and compare.

### 6.1 The paired comparison can inject latents only

Two backends do not share an RNG stream, so seeding both with the same number
produces different noise and therefore different images: comparing those measures
the RNG streams, not the operator implementations. On the 2512 cohort that mistake
reads as MAE 61 / PSNR 9.5 dB, which looks like a broken backend and is not one.

**2.1 cannot take injected prompt embeddings.** `__call__` has no
`image_pad_mask` parameter, and the local `append_target_slots` that builds one
does `mask.new_ones(...)` unconditionally — so the path where embeddings are
supplied directly, and the mask is therefore never built, dies with
`'NoneType' object has no attribute 'new_ones'`. Reproduced on the vendor's own
CUDA torch as well, so it is upstream and not a backend fault. `--save-embeds`
writes them for inspection and there is deliberately no flag that loads them
back.

So the paired run injects the **initial latents only**, and each backend encodes
the prompt itself:

```bash
# on the reference (any working backend, any chip):
tests/manual/qwen_image_21/run.sh infer --stage full --device cuda \
    --save-latents latents.pt --output ref.png

# carry latents.pt to the chip under test and inject it:
tests/manual/qwen_image_21/run.sh infer --stage full \
    --load-latents latents.pt --output out.png
tests/manual/qwen_image_21/run.sh compare --a ref.png --b out.png
```

The saved tensor is a CPU tensor of the packed latents, which is why it moves
between machines unchanged. The prompt is re-encoded on each side, so a difference
in the text encoder is *inside* this comparison rather than excluded from it —
worth knowing when reading the number, and the reason §7 exists.

The target is a tolerance, not equality: bf16 accumulation order differs.
**37.82 dB PSNR / MAE 1.194 over 40 steps at 1024x1024** is the paired A100
number.

## 7. Readings to record

The runner prints these at the end of every run, from the log of that run:

```
---- readings ----
cpu_fallback ops   : 0
dispatch records   : 13033
distinct ATen ops  : 62

---- operator calls by backend ----
8532 cuda
4501 flagos_python

---- distinct operators by backend ----
   37 flagos_python
   25 cuda
```

That block is a real two-step A100 run of `infer.py --stage full`, trimmed to its
first lines; the live output also lists which operators went where. A full
40-step run is ~163 k dispatches.

| Reading | Meaning |
| --- | --- |
| `cpu_fallback ops` | Operator calls that left the accelerator for the CPU (`FLAGOS_LOG` containing `fallback`). Every one is a route to fix; the op names are in the log. |
| `distinct ATen ops` | How many different operators the workload reached, i.e. the size of the cohort the routing has to cover. |
| `operator calls by backend` | Which backend each call took. `cuda` is the boxing path to the vendor kernel; `flagos_python` and `flagos` are FlagGems, the first the Python dispatch and the second the C++ one. |
| `distinct operators by backend` | The same split counted as operators rather than calls — the number to quote as "this workload uses N FlagGems operators". |
| seconds per image | Whole-pipeline wall time, from the sweep manifest. One shot, no warmup — see §5, and use §7.1 for a number you intend to quote. |
| reserved GiB per card | Peak footprint, from `memory_reserved`. |

`census` re-reads a saved log, so a run only has to be done once:

```bash
tests/manual/qwen_image_21/run.sh census /path/to/run.log
```

**Time a run with the dispatch logging off.** `run.sh` puts `dispatch` in
`FLAGOS_LOG` for `infer` and `sweep`, and it is not free: it writes one unbuffered
line per operator call, which on the 2512 workload was ~1.2 M write syscalls per
image and 12% of the loop. `run.sh bench` leaves it out for exactly this reason,
and records what the environment actually held. Set `FLAGOS_LOG` explicitly for
any other number you intend to quote. Note that the per-diagnostic variables
`FLAGOS_LOG_DISPATCH`, `FLAGOS_LOG_FALLBACK` and `FLAGOS_CACHE_STATS` are retired
— an exported value is inert, so setting one produces no logging at all and a
summary that reports zero operators as though the workload had not run any.

The per-op census is a measurement of one run, not a property of the routing
tables. Record it with the log path and the hardware, and re-measure rather than
reusing the numbers on another chip.

### 7.1 Performance measurement

`bench.py` is the measurement. `infer.py` reports what happened once; `bench.py`
repeats it and reports the spread, and it is the only thing in this directory
whose output is a number a chip's result can be quoted from.

```bash
run.sh bench --device cuda                                      # -> out/cuda/bench.json
run.sh bench --device flagos                                    # -> out/flagos/bench.json
run.sh table out/cuda/bench.json out/flagos/bench.json --baseline cuda
```

`table` reads JSON and needs no torch and no device, so the two files can be
carried off the chip and rendered anywhere.

#### The prompt is fixed

`bench.py` has **no `--prompt` flag**. It measures `prompts.canonical()` — entry
`01`, the only prompt the 2.1 authors published, and the same prompt `infer.py`
defaults to. Two runs of it are therefore always the same measurement, and the
JSON records the prompt's sha256 so a number names the exact bytes behind it. To
reach an operator surface the canonical prompt does not, use `infer.py` or
`sweep.py`; neither is a benchmark.

The choice is deliberate rather than arbitrary: `01` is 11 words, the same order
as the COCO captions MLPerf's `text_to_image` benchmark samples, so the text
encoder's share of the latency is representative instead of an artefact of a
prompt written for this flow.

#### The metrics, and where they come from

| Metric | Definition | Industry analogue |
| --- | --- | --- |
| `latency.per_image_s` | Wall time of one `pipe(...)` call at the run's batch, divided by the batch. Includes the text encoder, the denoise loop, the VAE decode and the postprocess to PIL. Excludes model load, placement and the PNG write. | MLPerf SingleStream latency |
| `phases["*"]` | The same call split into text encoder / denoise loop / loop per step / VAE decode. Every edge is synchronised, so each is device time rather than host-return time. | component breakdown |
| `phases["loop per step"]` | Denoise loop ÷ the number of steps in it. Steps-normalised, so two chips at different step counts still compare. | s/iteration |
| `throughput.images_per_s` | `batch / median per-call time`, reported only when `--batch N` (N > 1) runs. | MLPerf Offline samples/s |
| `memory.peak_gib` | Peak allocated watermark, reset before each measured call and read after it. Falls back to `memory_reserved`, and names which it used. | diffusers `mem_plain_GB` |
| `determinism.identical` | Whether the first and last measured calls produced byte-identical images. | prerequisite for §6.1 |
| `vs <baseline>` (in the table) | Latency ratio against a named column; throughput ratio as its reciprocal. | vendor speedup ratio |

The protocol is the diffusers harness' (`benchmarks/benchmarking_utils.py`):
discarded warmup rounds, `torch.utils.benchmark.Timer` with `num_threads=1`, and
peak allocated memory as the memory figure. Two things are deliberately
different, and both are flags whose used value is recorded:

- `--min-run-time` defaults to **60 s** rather than diffusers' 0.2 s. That
  default is sized for a ~50 ms component forward; this pipeline is ~26 s, and
  0.2 s would buy a single sample.
- `--warmup` defaults to **2** rather than 5. Five rounds of a 26 s pipeline is
  2.2 minutes spent before any measurement at all, and two still covers the
  Triton JIT and autotune convergence. At least one is always run even if
  `--warmup 0` is passed, and the JSON says so.

At batch 1 **no throughput is derived from the latency**. A rate computed from a
batch-1 latency is the latency restated, and restating it invites a comparison
against a real throughput number. `--batch N` passes `num_images_per_prompt=N`,
so the images still come from the canonical prompt; if the batch does not fit,
the JSON records the allocator's message in `throughput.error` and `null` for the
number, which is itself the result a 40 GB card gives.

#### Trap: the dispatch log costs 12% of the loop

§7's census is not free. `run.sh bench` therefore leaves `dispatch` out of
`FLAGOS_LOG` while `infer` and `sweep` keep it in, and the JSON records what the
environment actually held — a row taken with it on says so, in the table's
caveats.

#### Trap: a cold compile cache is a different machine

§4.3 measures the same 40-step run at 23.8 s warm and 71.5 s cold. `bench.py`
never clears the cache; it records `TRITON_CACHE_DIR`. `--cold-cache` points it at
a fresh directory, which is how "a box that has never run this" is measured, and
the JSON records that it was used. **Both sides of any cross-chip comparison must
be warm**, and the recorded cache path is what makes that checkable.

#### What a chip's result should look like

The JSON is the record; `table` is the view. A run is worth quoting when:

- `protocol.runs` is at least 2, so `latency.per_image_s.std_s` means something.
  The table warns when it is not.
- `determinism.identical` is true. If it is false the latency still stands, but a
  paired comparison in §6.1 does not, because the two runs are not sampling the
  same trajectory.
- `environment.dispatch_log` is off (or is quoted with the number).
- `memory.source` is stated, because a peak allocated and a reserved total are
  not the same measurement and a table that mixes them is worse than one that
  says which it used.

### 7.2 Reference measurement

The measured pair, taken on 2026-09-18 on 8x NVIDIA A100-SXM4-40GB, one card
(`flagos:0`), 40 steps, 1024x1024, seed 42, `true_cfg_scale=1.0`, batch 1, warm
Triton cache, `FLAGOS_LOG=fallback`. Command, for both columns:

```bash
run.sh bench --device <cuda|flagos> --out <out>.json
run.sh table out/cuda.json out/flagos.json --baseline cuda
```

The table below is that command's output, unedited. `torch cuda` is
2.10.0+cu128 in the vendor environment; `torch-fl` is 2.10.0+cpu with the
external CUDA 12.8 assets, the 2.1 checkout at `0.41.0.dev0`.

| metric | cuda | flagos |
| --- | --- | --- |
| prompt | 01 / `bb383cba` | 01 / `bb383cba` |
| latency, median s/image | 17.42 | 22.25 |
| latency, mean s/image | 17.42 | 22.25 |
| latency, min s/image | 17.41 | 22.25 |
| latency std, s | 0.008 | 0.010 |
| measured calls | 4 | 3 |
| text encoder, median s | 0.04 | 0.21 |
| denoise loop, median s | 17.10 | 21.63 |
| loop per step, median ms | 427.5 | 540.6 |
| vae decode, median s | 0.23 | 0.34 |
| throughput, images/s | n/a | n/a |
| peak GiB | 36.80 | 37.81 |
| memory counter | peak allocated | peak allocated |
| determinism | identical | identical |
| `FLAGOS_LOG` | fallback | fallback |
| triton cache | default | default |
| **vs cuda, latency** | **1.00x** | **1.28x** |

The three phase rows now add up to the latency row on both sides — 0.04 + 17.10
+ 0.23 = 17.37 against 17.42, and 0.21 + 21.63 + 0.34 = 22.18 against 22.25 — the
remainder being the latent preparation and the postprocess to PIL, which are not
attributed to a phase. They did not add up before the loop's start was bracketed;
see the note below.

**1.28x is the end-to-end figure on this chip.** The denoise loop — the only part
that scales with step count — is 1.26x (17.10 s against 21.63 s), and the same
1.26x per step (427.5 ms against 540.6 ms); the text encoder is 5x and the VAE
1.5x, but those are one call each and a small share of the total, so the ratio a
chip reports depends on how many images a run produces.

`throughput` is `n/a` on both because both are batch 1, and no rate is derived
from a batch-1 latency (§7.1). A batch-4 run on this 40 GB card does not fit at
all — `bench.py --batch 4` writes a record whose `throughput.error` is the
allocator's own message and whose latency fields are `null`, which is the
truthful answer rather than an estimate. A real throughput column needs a card
that holds four images at once; none was available here, and this document does
not guess at one.

**What changed between measurements.** An earlier revision of this table read
1.26x, from a loop phase one step short: `callback_on_step_end` fires *after*
each step, so N callbacks bound only N−1 intervals, and the loop total omitted
the first step. `loop per step` was right in both revisions — N−1 intervals
divided by N−1 — but the total was 2.5% short and the phase rows visibly failed
to add up to the latency. `PhaseTimer` now brackets the loop from the transformer
forward that begins it and synchronises inside each callback, so the total covers
all 40 steps and each edge is device-attributed rather than host-return. Both
sides moved, which is why the ratio moved with them.

The synchronise is the part that is not merely cosmetic. Nothing in the pipeline
blocks the host between steps, so an unsynchronised callback stamps when work was
enqueued. Measured on A100 by running the same loop twice in one process: 20.88 s
unsynchronised against 20.91 s synchronised, or 0.15%. The two agree *here*
because the 40 GB card's allocator is nearly full and blocks the host on every
step — a property of this chip's memory pressure, not of the measurement.

A single-shot run of the same pipeline with injected latents, taken a day earlier
with `infer.py --stage full`, gave **17.5 s against 23.8 s (1.36x)** with a phase
split of 0.51 / 16.53 / 0.45 against 2.08 / 20.76 / 0.92. The numbers differ from
the table above because that run was one sample with no warmup, and because its
two single-call components (the text encoder at 2.08 s against 0.21 s) had not
been repeated. That is the difference §7.1 exists to remove, and it is why the
table above, not that figure, is what this flow quotes.

The per-operator readings that go with this chip were taken on the same 2026-09-17
run and are unchanged:

| Reading | torch-fl |
| --- | --- |
| `cpu_fallback` ops | **0** |
| distinct ATen operators | **62** |
| — routed to FlagGems (`flagos_python`) | **37** |
| — routed to the boxing path (`cuda`) | **25** |
| paired PSNR / MAE vs cuda | **37.82 dB / 1.194** |

Inside the loop, 1863 of 4067 dispatches per step are FlagGems and the rest are
the boxing path. The gap to the vendor is mostly host-side: routing those calls
costs about 44 ms/step and the boxing hop another 67 ms/step, against 424 ms of
device work on the vendor side.

**Cold compile cache.** The FlagGems path JIT-compiles Triton kernels, so a box
whose cache is empty pays for it once. Measured by pointing `TRITON_CACHE_DIR` at
an empty directory: the text encoder went from 2.0 s to 12.6 s, the VAE decode
from 0.9 s to 14.3 s, and the whole 40-step run from 23.8 s to 71.5 s — a 3x
difference on the same hardware from the cache alone. Warm the cache before
quoting any timing, and quote the cache state with the number.

The raw log is machine-local; regenerate it with a flagos run of `infer.py
--stage full` or `run.sh bench --device flagos` — both print the phase table
themselves, so no dispatch logging is needed for it (and it should be off,
see §7).

The single-shot figure above is kept only as the record of what the staged runner
reports. For anything quoted, use the `bench.py` pair at the top of this section:
it runs the same pipeline with a discarded warmup and a self-sizing repeat count,
so every phase above comes with a spread instead of a point.

## 8. Per-operator comparison against a reference

`sweep.py` tells you whether the model produces a picture. `numerics.py` tells
you *which operator* is wrong when it does not, and it is the cheapest thing to
run first — before the model is even worth loading.

```bash
# run it on the reference (a vendor torch, or the CPU) and on the chip:
python tests/manual/qwen_image_21/numerics.py --device cuda   --out ref.pt
python tests/manual/qwen_image_21/numerics.py --device flagos --out chip.pt

# then diff, on any interpreter, no device needed:
python tests/manual/qwen_image_21/numerics.py --compare ref.pt chip.pt
```

It checks three things per operator, because they fail independently:

**Numerics.** Max absolute and mean relative error against the reference.
Anything above `5e-3` relative is not bf16 rounding — these are one- or
two-step accumulation chains. On the A100 cohort 35 of the 36 operators agree
bit-for-bit, and the one that does not is `cumsum`, where torch-fl is *more*
accurate than the vendor: FlagGems' Triton kernel accumulates in fp32
(`get_scan_accum_type` in `flag_gems/ops/cumsum.py`) while cuBLAS/CUB accumulates
in the input dtype. On a drifting input — values with a non-zero mean, so the
running sum grows to ~4000× the step, `torch.randn(1, 4096, 64) * 2 - 1` in bf16
— the two differ by `max_abs` **16.0 on torch-fl against 2338.5 on CUDA**, both
measured against an fp32 reference. It is a real difference and an improvement;
do not "fix" it without deciding which accumulator the chip should have. The
checker flags it, which is the point — the tool reports a difference and a human
judges it.

**Aliasing.** For the view/reshape family, whether the result shares storage
with its input. This is the check that earned its place: on this workload
torch-fl's `_unsafe_view` was routed to a FlagGems implementation that is
`self.reshape(size)`, which returns a **materialised copy** where ATen returns a
strided view or raises. Values identical, so no numerics comparison could see
it, and the copy silently doubled that tensor's memory. It was found by probing
`untyped_storage().data_ptr()` directly. **Run this check on any chip whose views
are routed somewhere other than the vendor's own kernel.**

Two things about how it is judged, both of which it got wrong before and both of
which would have reported a broken backend as a pass:

- A view the candidate could **not run** is a failure, not agreement. The earlier
  comparison only looked at the `aliases` flag when *both* sides had one, so a
  backend whose `permute` raised produced `{"error": ...}` on the candidate side,
  matched nothing, and exited 0 printing `view aliasing: all 5 agree`. A
  reference-side error is still skipped, because there is nothing to compare
  against and the candidate is not at fault — the same rule the value checks
  above use. `run()` also counts checks that actually produced a reading rather
  than entries in the file, so its own summary cannot overstate the coverage.
- The `where` and `masked_fill` inputs use a **both-valued** mask. A `randn`
  cast to `bool` is 100% True — a float is `False` only when it is exactly
  `0.0` — so `where` returned its `x` branch on every element and `masked_fill`
  filled nothing, and a backend with a broken false branch agreed with the
  reference by never reaching it. Compared against zero, roughly half the
  entries are True; the CPU generator still makes both backends start from the
  same bytes.

**Cost.** Host microseconds to *issue* one call, with no synchronisation — the
device queue absorbs the work, so the loop is bounded by the host. This is the
number that separates "this kernel is slower here" from "this call costs more to
issue", and the second is the failure mode of a dispatch layer. On A100 the
FlagGems Python route costs 98 µs to issue `mul` where CUDA boxing costs 9.7 µs,
against 102 µs of device time — which is precisely why that route was
host-bound.

Inputs are built on the CPU from a fixed seed and moved to the device, never
drawn on the device: two backends do not share an RNG stream, so a device-side
`randn` would give the two runs different inputs and every operator would
"differ" for the wrong reason.

## 9. Failure triage

| Symptom | Cause | Where the fix belongs |
| --- | --- | --- |
| `Cannot get CUDA generator without ATen_cuda library` | `import torch` ran before `import torch_fl` | The script: call `common.import_torch` first. |
| `ImportError: cannot import name 'resolve_revision' from 'huggingface_hub'` | `huggingface-hub` is older than 1.26 | The environment: upgrade it. |
| `Qwen3VLVideoProcessor requires the Torchvision library` | `torchvision` is not installed | The environment: install the build matching the installed torch, `--no-deps`. |
| `ImportError: cannot import name 'QwenImage21Pipeline' from 'diffusers'` | `QWEN_IMAGE_21_DIFFUSERS` is unset or points at the wrong directory | The environment: point it at the checkout's `src/`. |
| `'NoneType' object has no attribute 'new_ones'` | Something passed `prompt_embeds` to the pipeline directly | Upstream: 2.1 cannot take injected embeddings. Inject latents instead. |
| `mat2 is on cuda:0, different from other tensors on cpu`, from a `linear` in `timestep_embedder` | The timestep schedule was built on the CPU while the weights are on a card | The script: build the schedule on the pipeline's execution device. |
| `mat2 is on cuda:1, different from other tensors on cuda:0`, from `img_in` | The transformer is on another card without accelerate hooks | The script: place it through `common.split_transformer`. |
| `LocalEntryNotFoundError` out of `hf_hub_download` | The model is a hub id and `HF_HOME` is not the cache that holds it | The environment: export `HF_HOME`, or point `--model` at the weights. |
| `PrivateUse1 is already claimed by the '<name>' backend` | A vendor torch plugin autoloaded and took the key | `TORCH_DEVICE_BACKEND_AUTOLOAD=0` before python, or import torch_fl first. |
| OOM while dispatching a shard | The split boundary sits where a shard is too big | `--blocks-per-device`, or more `--transformer-devices`. |
| `CUDA out of memory` on a card the model fits, with `reserved but unallocated` in the message | Allocator fragmentation, not a shortage | The environment: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. See §1.2 — on a 40 GB card this decides whether the run completes. |
| Cross-device error inside `scheduler.step` | `proj_out` is not on the first shard | torch_fl: the block map must leave `norm_out`/`proj_out` on the first shard, because accelerate's root hook only moves the forward's output back if it is produced there. |
| An op is missing entirely | No route for it on this chip | torch_fl: fix the route or the kernel, then re-measure the census. Do not patch `diffusers` or `transformers` — a user must be able to `pip install diffusers` and run this unchanged. |
| Every op reports `flagos_python` but the run behaves like plain CUDA, and `cumsum` agrees with the vendor exactly | The build has no FlagGems Python path compiled, so every `flaggems` route silently degrades to the boxing kernel — `Dispatcher::GetFn` is documented to do that so one conf stays correct across builds | The build: check `FLAGGEMS_KERNEL=ON` in `build/CMakeCache.txt`. A **stale cache** is the trap: a previous `FLAGGEMS_KERNEL=OFF` build persists in `CMakeCache.txt` and later invocations that do not name the variable keep it off. `rm -rf build` before rebuilding. This is worth knowing because the degradation is silent — the run succeeds, and only the phase timings and the census reveal it. |

| `RuntimeError: Backend doesn't support synchronizing all streams on device.` out of `torch.accelerator.synchronize()` | `GuardImpl` did not implement `DeviceGuardImplInterface::synchronizeDevice`, which the interface's default refuses | torch_fl: `csrc/runtime/guard.h`. Anything built on `torch.utils.benchmark.Timer` reaches it, because the timer synchronises that way around every timed call. |

Attribution is the point of the stages: a failure in `text-encoder` is the
encoder's operator surface, a failure in `vae` is the 3-D convolution, and only a
failure in `full` that survives both is a problem with the loop itself.

## 10. Known limits

- The text encoder stays resident for the whole run. It is the pipeline's
  execution device, so it cannot be moved off the card between encoding and
  denoising without moving where the latents and the decode live.
- `--save-embeds` is inspection only. There is no path that injects embeddings,
  for the upstream reason in §6.1, so a paired run cannot exclude the text
  encoder.
- The cohort is ours, not the model authors'. §5 says why and what each prompt is
  for.
- Timing is recorded, not optimised. Nothing here is a performance claim except
  the `bench.py` pair in §7.2, and that table quotes the vendor beside it.
- Reserved memory is not comparable across backends unless both numbers come from
  the allocator. `nvidia-smi` folds the CUDA context and the cuBLAS/cuDNN
  workspaces into one of the two numbers.
- §7.2's throughput row is empty on this hardware. Four images do not fit one
  40 GB card, and no larger part was available; the column needs one, and nothing
  here estimates it.
- Image quality is judged by eye. The paired PSNR is a backend-agreement measure,
  and it says nothing about whether the pictures are any good.
- Nothing under `tests/manual/` is in `.github/configs/*.yml`, so none of this
  runs in CI. Adding it would mean mounting a ~33 GB model into the test
  container, installing `diffusers` from a source checkout, and providing a
  GPU runner.
