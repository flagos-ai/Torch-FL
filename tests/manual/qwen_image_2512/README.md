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

## 7. Failure triage

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

## 8. Known limits

- The text encoder stays resident for the whole run. It is the pipeline's
  execution device, so it cannot be moved off the card between encoding and
  denoising without moving where the latents and the decode live. On a two-card
  chip that card also holds half the transformer, so peak memory there is
  roughly the encoder plus a shard: 16.6 GB + 20.4 GB of weights before any
  activation. Prefer three cards.
- Timing is recorded, not optimised. Nothing here is a performance claim.
- Image quality is judged by eye. The paired PSNR is a backend-agreement
  measure, and it says nothing about whether the pictures are any good.
- Nothing under `tests/manual/` is in `.github/configs/*.yml`, so none of this
  runs in CI. Adding it would mean mounting a ~54 GB model into the test
  container and installing `diffusers` there.
