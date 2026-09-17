# Qwen-Image-2512 manual test flow

`Qwen/Qwen-Image-2512` is a text-to-image diffusion model: a Qwen2.5-VL text
encoder, a 60-block MMDiT transformer, and a causal-Conv3d VAE, driven by
`diffusers`' `QwenImagePipeline`. Running it on an accelerator is the broadest
check this repository has for a new chip — it exercises operator surfaces no
Transformers test reaches (causal `Conv3d`, complex-dtype rotary embeddings, a
50-step loop with two transformer forwards per step), and it fails loudly rather
than subtly when a route is missing.

Nothing under `tests/manual/` runs in CI, so this is a bring-up procedure, not a
gate. It is written to be run on a chip that has never run it: every stage is
isolated, the placement adapts to the cards it finds, and the readings a new
chip should produce are stated so a result can be judged rather than guessed at.

## Files

| File | Role |
| --- | --- |
| `infer.py` | The staged runner. Four stages, input save/load for paired runs. |
| `sweep.py` | Every prompt on the model card, in the card's own settings. The cohort run. |
| `side_by_side.py` | Pairs two cohort directories into per-prompt sheets plus one contact sheet. |
| `compare.py` | PSNR and MAE between two PNGs. Imports no torch, so it runs anywhere. |
| `memprobe.py` | Per-phase allocator accounting, and what FlagGems is holding. §7. |
| `common.py` | Import order, placement, split, memory reporting. Both runners use it. |
| `run.sh` | Wrapper: sets the environment, runs one mode, summarises the log. |

## 1. Prerequisites

| Requirement | Detail |
| --- | --- |
| torch_fl | Built for the chip and importable. The device is always `flagos` — `rename_privateuse1_backend("flagos")` is not vendor-dependent. |
| Python packages | `diffusers` (0.40.0 verified) and `pillow`. Neither is a test dependency of the repo. |
| Model | `Qwen/Qwen-Image-2512`, ~54 GB over 30 files, in `$HF_HOME/hub`. |
| Host RAM | ~57 GB free. `from_pretrained` materialises the pipeline on the host before any component moves to a card. |
| Accelerator memory | 38.1 GiB of transformer weights plus activations. One card holds that only on a 64 GiB part; two 40 GiB cards do not. |

`import torch_fl` must precede `import torch`. torch_fl's `_preload_cuda_assets()`
dlopens the bundled `libtorch_cuda.so` before PyTorch caches its CUDA hooks;
importing torch first leaves the assets unloaded and the run dies at the first
generator with `Cannot get CUDA generator without ATen_cuda library`. The runner
does this for you; a hand-written snippet has to do it too.

Set `HF_HOME` to the cache that holds the model before running anything. A
shared cache is usually not at the `huggingface_hub` default, and pointing
`HF_HOME` at an empty directory produces the same error as a missing model. The
runner deliberately does not guess a cache path.

Set `HF_HUB_OFFLINE=1` too, once the cache is complete and the box has no proxy
— the sweep reads the model card through `hf_hub_download`, so it needs either
the cache or the network.

## 2. Placement

`--device` names a torch device module: `flagos` (imports torch_fl first),
`cuda`, or any vendor name a torch plugin registered. The number of devices the
module reports decides the shape of the run:

| Visible devices | Text encoder + VAE | Transformer |
| --- | --- | --- |
| 1 | that card | that card, no split |
| 2 | the second card | split across both |
| 3 or more | the third card | split across the first two |

Only the first three are used by default. Three is the most that buys anything:
each extra shard adds a hidden-state round trip on every forward, and a 50-step
true-CFG run has 100 of them.

Why the encoder and the VAE share a card: `DiffusionPipeline._execution_device`
falls back to `self.device`, the first non-CPU component in
`_get_signature_keys` order — and that order is *sorted*, so `text_encoder`
outranks `transformer` and `vae`. The pipeline therefore builds its latents and
decodes them on the encoder's card, and the VAE has to be there or the decode
hands it an input from a card its weights are not on.

Every decision above can be overridden, and the overrides are what to reach for
when the default does not fit: `--devices`, `--encoder-device`,
`--transformer-devices`, `--vae-device`, `--blocks-per-device`. Two examples:

```bash
# One 64 GiB card: everything on it, no split, no cross-device traffic.
tests/manual/qwen_image_2512/run.sh infer --stage full --devices flagos:0

# Three 40 GiB cards, encoder first and the transformer on the other two
# (the layout the A100 reference measurement used).
tests/manual/qwen_image_2512/run.sh infer --stage full \
    --encoder-device flagos:0 --transformer-devices flagos:1 flagos:2
```

`--blocks-per-device 20 40` moves the split boundary when one shard runs out of
room; the counts must sum to the model's block count and it is checked, not
assumed.

## 3. The stages

Run them in order. A failure in a 50-step loop with two forwards per step is far
more expensive to localise than a failure in one isolated call, and the stages
are ordered so that the two components with new operator surfaces — complex
rotary embeddings in the encoder path, causal `Conv3d` in the VAE — are checked
before the loop that uses both.

```bash
export PYTHON=/path/to/the/env/with/torch_fl/bin/python

# 0. The backend answers at all.
$PYTHON -c "import torch_fl, torch; print(torch.__version__, torch.flagos.device_count())"

# 1. Text encoder: prompt -> prompt_embeds + mask.
tests/manual/qwen_image_2512/run.sh infer --stage text-encoder

# 2. One transformer forward at fixed latents, true CFG disabled.
tests/manual/qwen_image_2512/run.sh infer --stage transformer-step

# 3. VAE decode of fixed latents -> PNG.
tests/manual/qwen_image_2512/run.sh infer --stage vae

# 4. The whole pipeline: 1024x1024, 50 steps, one PNG.
tests/manual/qwen_image_2512/run.sh infer --stage full
```

What each stage prints, and what a pass looks like:

| Stage | Pass |
| --- | --- |
| `text-encoder` | `prompt_embeds (1, L, 3584) bfloat16 flagos:N`, then `device check : OK` |
| `transformer-step` | `transformer noise (1, 4096, 64)` on the first shard, `device check   : OK` |
| `vae` | `decoded (1, 3, H, W)` and a PNG whose `std` is not ~0 |
| `full` | a PNG, then per-card `reserved`/`allocated`. A mean pixel value near 0 or a `std` near 0 means the decode produced nothing. |

Stage 3 is the cheapest way to prove the whole memory layout: it loads no
transformer weights onto a card at all.

### 3.1 Determinism

Two runs at the same seed on the same backend must be bit-identical. This is a
property of the backend, and it is a prerequisite for the paired comparison
below — if a repeat run drifts, the backend has a nondeterminism to find before
any cross-backend number means anything.

```bash
tests/manual/qwen_image_2512/run.sh infer --stage full --seed 42 --output a.png
tests/manual/qwen_image_2512/run.sh infer --stage full --seed 42 --output b.png
cmp a.png b.png && echo "bit-identical"
```

### 3.2 The model-card cohort

`sweep.py` runs every prompt the model card states — its Quick
Start snippet plus every showcase blockquote, read out of the card rather than
transcribed, deduplicated — in the card's own settings: 1664x928, 50 steps,
`true_cfg_scale=4.0`, seed 42, the card's negative prompt. It writes one PNG per
prompt plus a `manifest.json` carrying each prompt, its timing, and the
placement used.

```bash
tests/manual/qwen_image_2512/run.sh sweep
tests/manual/qwen_image_2512/run.sh sweep --only 01 07   # a subset, for a smoke run
```

Images land in one directory per backend under `OUT_DIR`, named after the
backend's device module — `<OUT_DIR>/flagos`, `<OUT_DIR>/musa`,
`<OUT_DIR>/cuda`. The vendor run and the flagos run of the same sweep therefore
sit side by side instead of overwriting each other, which is what makes the
comparison in §4 possible. Pass `--output-dir` explicitly to name a directory
anything else.

## 4. Vendor against flagos, side by side

The check that needs no interpretation is two directories of the same prompts,
one per backend, looked at together. Run the sweep twice:

```bash
PYTHON=/path/to/the/vendor/env/bin/python tests/manual/qwen_image_2512/run.sh \
    sweep --device cuda           # or musa, npu: whatever the chip's own torch calls it
PYTHON=/path/to/the/torch_fl/env/bin/python tests/manual/qwen_image_2512/run.sh \
    sweep --device flagos
tests/manual/qwen_image_2512/run.sh side-by-side --left cuda --right flagos
```

The last command writes `compare/01.png` … `compare/12.png`, each one prompt with
the vendor's image on the left and flagos' on the right, captioned with the
device and the generation time, plus `compare/all.png` holding every pair at
once — that is the sheet to look at first. `--left` and `--right` take a name
under `OUT_DIR` or a path, so a reference directory kept on another machine works
too.

At the cohort's settings the two backends do not share an RNG stream (§5), so the
images differ in composition as well as in detail. What is being judged is
whether flagos' pictures are as coherent as the vendor's: same subject, same
structure, no smearing, no missing content, comparable timings. The paired run in
§5 is the one that can be given a number.

This works when the chip has no torch of its own as well: run the left side on
any working backend — an A100 somewhere, or the CPU reference — carry the two
directories over, and compare.

## 5. Paired comparison against a reference

The comparison that isolates the backend injects the same inputs on both sides.
Two backends do not share an RNG stream, so seeding both with the same number
produces different noise and therefore different images: comparing those
measures the RNG streams, not the operator implementations. On the A100 cohort
that mistake reads as MAE 61 and PSNR 9.5 dB, which looks like a broken backend
and is not one.

The runner has both halves. On the reference (any working backend, any chip):

```bash
tests/manual/qwen_image_2512/run.sh infer --stage full --device cuda \
    --save-inputs inputs.pt --save-latents latents.pt --output ref.png
```

Carry `inputs.pt` and `latents.pt` to the chip under test and inject them:

```bash
tests/manual/qwen_image_2512/run.sh infer --stage full \
    --load-inputs inputs.pt --load-latents latents.pt --output out.png
tests/manual/qwen_image_2512/run.sh compare --a ref.png --b out.png
```

The saved tensors are CPU tensors of the packed latents and of both prompt
embedding pairs, which is why they move between machines unchanged. Injecting
them removes the generator from the comparison entirely: what is left is the
operator implementations on the two backends. bf16 accumulation order differs,
so the target is a tolerance, not equality — 35.4 dB PSNR / MAE 1.7 over 50
steps at 1024x1024 is the paired A100 number.

Comparing two independently seeded runs is still worth doing, but read it as a
quality check on the images, not as a number.

## 6. Readings to record

The runner prints these at the end of every run, from the log of that run:

```
---- readings ----
cpu_fallback ops   : 0
libentry failures  : 0
dispatch records   : 14479582
distinct ATen ops  : 55

---- operator calls by backend ----
9519382 cuda
4960200 flagos_python

---- distinct operators by backend ----
     31 flagos_python
     24 cuda
```

The block above is the real A100 run of §6.1, trimmed to its first lines; the
live output also lists which operators went where.

| Reading | Meaning |
| --- | --- |
| `cpu_fallback ops` | Operator calls that left the accelerator for the CPU (`FLAGOS_LOG_FALLBACK=1`). Every one is a route to fix; the op names are in the log. |
| `distinct ATen ops` | How many different operators the workload reached, i.e. the size of the cohort the routing has to cover. |
| `operator calls by backend` | Which backend each call took. `cuda` is the boxing path to the vendor kernel; `flagos_python` and `flagos` are FlagGems, the first being the Python dispatch and the second the C++ one. |
| `distinct operators by backend` | The same split counted as operators rather than calls — the number to quote as "this workload uses N FlagGems operators". |
| `libentry failures` | Triton kernel build failures in the FlagGems path. Any of these means a FlagGems op silently fell back. |
| seconds per image | Whole-pipeline wall time, from the sweep manifest. |
| reserved GiB per card | Peak footprint, from `memory_reserved`. |

`census` re-reads a saved log, so a run only has to be done once:

```bash
tests/manual/qwen_image_2512/run.sh census /path/to/run.log
```

The per-op census is a measurement of one run, not a property of the routing
tables: `FLAGOS_LOG_DISPATCH=1` logs the backend actually chosen for every
dispatch, and only ops the workload reaches appear. Record it with the log path
and the hardware, and re-measure rather than reusing the numbers on another chip.

### 6.1 Reference measurement

Taken on 2026-09-16, 8x NVIDIA A100-SXM4-40GB, torch 2.10.0+cpu with the
external CUDA 12.8 assets, `diffusers` 0.40.0, placement `flagos:2` for the text
encoder and VAE with the transformer split 30/30 over `flagos:0` and `flagos:1`.
The full twelve-prompt model-card cohort of §3.2.

| Reading | Value |
| --- | --- |
| Succeeded | 12/12 images, exit 0 |
| Time per image | 72.6 s mean (71.5–73.9 s) |
| Reserved per card | 20.01 GiB / 19.93 GiB / 25.02 GiB on `flagos:0,1,2` |
| `cpu_fallback` ops | **0** — every operator the pipeline reaches has a route |
| `libentry` compile failures | **0** — no FlagGems kernel fell back at build time |
| Distinct ATen operators dispatched | **55** |
| — routed to FlagGems (`flagos_python`) | **31** |
| — routed to the boxing path (`cuda`) | **24** |
| Dispatch calls | 14,479,582 (9,519,382 cuda / 4,960,200 FlagGems) |

The 31 FlagGems operators: `_unsafe_view`, `add.Tensor`, `all`, `arange`,
`arange.start`, `bmm`, `clamp`, `clamp_min`, `constant_pad_nd`, `cos`,
`div.Tensor`, `eq.Scalar`, `exp`, `gelu`, `linalg_vector_norm`, `mean.dim`,
`mm`, `mul.Tensor`, `neg`, `ones`, `pow.Tensor_Scalar`, `reciprocal`, `rsqrt`,
`silu`, `sin`, `sub.Tensor`, `sum`, `transpose.int`, `unsqueeze`, `zero_`,
`zeros`.

The 24 on the boxing path: `_upsample_nearest_exact2d`, `addmm`, `alias`, `cat`,
`convolution`, `detach`, `embedding`, `empty_like`, `index.Tensor`,
`lift_fresh`, `native_layer_norm`, `permute`, `randn.generator`, `select.int`,
`slice.Tensor`, `split.Tensor`, `split_with_sizes`, `squeeze.dim`, `stack`,
`sum.dim_IntList`, `t`, `unbind.int`, `view_as_complex`, `view_as_real`.

Two of those are worth calling out for a chip that has to reproduce them.
`view_as_complex` and `view_as_real` are the complex-dtype rotary embeddings of
§3; they are on the boxing path because `torch_fl/configs/backends_cuda.conf`
routes them there (`view_as_complex = cuda`, `view_as_real = cuda`), and they are
called 24,000 and 120,000 times per image respectively. `convolution` is the
VAE's causal `Conv3d`, 37 calls per image.

None of this is a runtime fallback: `cpu_fallback` is 0, so every one of the 55
operators was served by the backend the configuration chose. On another chip the
same names have to resolve to that chip's own kernel — the call counts above are
what it takes to reproduce the layout, and `native_layer_norm` at 24,100 calls
per image and `addmm` at 84,768 are the two that dominate the dispatch traffic.

The raw log is machine-local and was 508 MB; it is not committed. Regenerate it
by running §4's flagos sweep with `FLAGOS_LOG_DISPATCH=1`, which `run.sh` exports
by default.

## 7. Where the extra memory went

A chip that runs this flow and reports more reserved memory than the CUDA
reference should be measured before it is explained. `memprobe.py` does that: it
runs the same pipeline and placement as `infer.py`, resets the allocator's peak
counters around each phase so every row is that phase's own peak, and prints one
table per card.

```bash
PYTHON=/path/to/the/torch_fl/env/bin/python \
    python tests/manual/qwen_image_2512/memprobe.py --device flagos --stage vae
PYTHON=/path/to/the/cuda/env/bin/python \
    python tests/manual/qwen_image_2512/memprobe.py --device cuda   --stage vae
```

Both sides read the same counters — flagos delegates its allocations to the CUDA
caching allocator, so `torch.flagos.memory_stats` and `torch.cuda.memory_stats`
are two views of one pool. `nvidia-smi` is not a substitute: it folds the CUDA
context and the cuBLAS/cuDNN workspaces into one of the two numbers and makes
them incomparable.

### 7.1 The measurement, on the A100 cohort

Peak allocated during each phase, 1024x1024, encoder and VAE on the third card:

| Phase | stock CUDA torch | torch-fl, before | torch-fl, after |
| --- | --- | --- | --- |
| load weights | 15.736 GiB | 15.736 | 15.736 |
| text encoder forward | 15.771 | 15.778 | 15.771 |
| denoise loop | 15.748 | 15.770 | 15.748 |
| **VAE decode (peak)** | **19.877** | **20.460** | **19.877** |
| **VAE decode (reserved)** | **20.482** | **21.686** | **20.482** |
| device mallocs during decode | 20 | 25 | 20 |

Everything the transformer touches agreed to within 0.001 GiB, so the allocator,
the boxing path and the routing were all fine. The whole difference was in the
VAE decode, and the "after" column is what the allocator reports once FlagGems
stops holding those tensors — it agrees with CUDA byte for byte, malloc count
included.

### 7.2 The cause

`flag_gems.utils.libentry.LibTuner.run` kept the argument tuple of the call it
had just run, so that a later `benchmark_config` could replay it without the
caller passing `args` again. The store was unconditional and strong, which meant
**every `@libtuner`-decorated kernel pinned the tensors of its most recent call
for the life of the process**, whether or not anyone ever asked for a replay. The
pinned size is the largest activation that kernel ever saw, nothing in user code
can reach it, `empty_cache()` does not touch it, and each additional tuned
operator adds its own set.

On this workload it cost 1.38 GiB across three cards, of which 334 MiB was the
VAE decoder's own activations — the two `(1, 96, 1, 1024, 1024)` buffers
`mul_generic_nd_kernel` holds. It does not grow with the number of steps: each
kernel holds exactly one set, so it is a fixed overhead, and it scales with
resolution rather than time. At the cohort's 1664x928 it is 2.4 GiB on the
encoder card.

Two things made it findable, and both are worth repeating on another chip:

- A `TorchDispatchMode` that samples live bytes after every ATen op shows the gap
  opening in exact doublings at the VAE decoder's upsampling levels
  (`+24`, `+48`, `+96`, `+192` MiB on `add.Tensor`, each followed by a doubled
  step on `silu`).
- Walking `gc.get_objects()` for a tensor that should have been freed names its
  holder directly — it came back as
  `LibEntry.__dict__['fn'].<LibTuner>.__dict__['_last_benchmark_args'][1]`.

`memprobe.py` does the first of those only for its own phase boundaries, but it
prints the second up front: the `memory pinned by the FlagGems autotuner` section
counts what is held, releases it, and reads the allocator again. On a FlagGems
carrying the fix it prints `nothing pinned`.

### 7.3 The fix

Fixed upstream in [flagos-ai/FlagGems#6386](https://github.com/flagos-ai/FlagGems/pull/6386):
tensor arguments and Triton TMA descriptors are now retained as `weakref`s, and
`benchmark_config` dereferences them when it falls back to the stored context. If
the tensors have been freed by then, the context resolves to `None` and the call
raises the error it already had for a missing prior context.

Until that lands in the FlagGems a chip is using, the section above is the check:
a non-zero total means the installed FlagGems still retains the arguments.

## 8. Failure triage

| Symptom | Cause | Where the fix belongs |
| --- | --- | --- |
| `Cannot get CUDA generator without ATen_cuda library` | `import torch` ran before `import torch_fl` | The script: call `common.import_torch` first. |
| `LocalEntryNotFoundError` or `OfflineModeIsEnabled` out of `hf_hub_download` | `HF_HOME` is not the cache that holds the model, or `HF_HUB_OFFLINE=1` is set against one that is incomplete | The environment: export `HF_HOME`, and drop `HF_HUB_OFFLINE` for the first run on a box that has to download. |
| `PrivateUse1 is already claimed by the '<name>' backend` | A vendor torch plugin autoloaded and took the key | `TORCH_DEVICE_BACKEND_AUTOLOAD=0` before python, or import torch_fl first. |
| OOM while dispatching a shard | The split boundary sits where a shard is too big | `--blocks-per-device`, or more `--transformer-devices`. |
| `does not give any device for the following parameters` | The device map missed a top-level module | torch_fl: `split_transformer` enumerates `named_children()` on purpose and nothing here should need a name list. |
| Cross-device error inside `scheduler.step` | `proj_out` is not on the first shard | torch_fl: the block map must leave `norm_out`/`proj_out` on the first shard, because accelerate's root hook only moves the forward's output back if it is produced there. |
| The decode fails on a device mismatch | The VAE is not on the encoder's card | The script: the VAE must be on the execution device (see §2). |
| An op is missing entirely | No route for it on this chip | torch_fl: fix the route or the kernel, then re-measure the census. Do not patch `diffusers` or `transformers` — a user must be able to `pip install diffusers` and run this unchanged. |

Attribution is the point of the stages: a failure in `text-encoder` is the
encoder's operator surface, a failure in `vae` is `Conv3d`, and only a failure
in `full` that survives both is a problem with the loop itself.

## 9. Known limits

- The text encoder stays resident for the whole run. It is the pipeline's
  execution device, so it cannot be moved off the card between encoding and
  denoising without moving where the latents and the decode live. On a two-card
  chip that card also holds half the transformer, so peak memory there is
  roughly the encoder plus a shard: 16.6 GB + 20.4 GB of weights before any
  activation. Prefer three cards.
- Timing is recorded, not optimised. Nothing here is a performance claim.
- Reserved memory is not comparable across backends unless both numbers come from
  the allocator. §7 is how to take that measurement, and what it cost here.
- Image quality is judged by eye. The paired PSNR is a backend-agreement
  measure, and it says nothing about whether the pictures are any good.
- Nothing under `tests/manual/` is in `.github/configs/*.yml`, so none of this
  runs in CI. Adding it would mean mounting a ~54 GB model into the test
  container and installing `diffusers` there.
