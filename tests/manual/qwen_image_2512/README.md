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

`Qwen/Qwen-Image-2.1` has its own flow next to this one, at
`tests/manual/qwen_image_21/`, and it is worth running as well on a chip that
passes here: it reaches a different encoder (Qwen3-VL rather than Qwen2.5-VL),
gated MLPs, a causal prefix KV cache and a different VAE, so a route that is
correct on one model is not evidence for the other. Its README states where the
two differ — the placement rule is the opposite of this one's, and it cannot
inject prompt embeddings. `compare.py` and `side_by_side.py` are shared: they
are model-agnostic, so they live here and the 2.1 `run.sh` dispatches to them
rather than keeping a second copy that could drift.

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

### 3.3 Chips without a complex dtype

`diffusers.models.transformers.transformer_qwenimage` builds its rotary
embedding by multiplying a complex exponential, and it keys that choice on
`device.type`. It knows two devices: `cuda`, which takes the complex path, and
`neuron`, which has no complex dtype and is handed rotation *angles* instead. A
device type it does not list falls back to the complex path.

Enflame GCU is the measured case of a chip that cannot take it. `topsaten` has
no complex kernel at all, so the vendor run reaches `topsatenMul`/`topsatenCos`
with a `ComplexFloat` operand and the process dies on
`TOPSATEN_STATUS_NOT_SUPPORT` — not a Python exception, which is why §7's table
cannot catch it. The flagos leg used to survive the same rotation by leaving the
two `view_as_complex`/`view_as_real` calls to `cpu_fallback`, which is 24,000 and
120,000 host round trips per image at §6.1's layout; on this build they have
native `gcu` kernels instead (the complex-view entry in
`docs/reference/operator-support.md`), so the complex path is served on the
accelerator rather than on the host. Serving it there is not the same as serving
it quickly — see "Why the complex multiply costs what it does" below, where the
same multiply turns out to be 91x the cost of copying its own operands.

**On GCU the angle path now ships as the default.** `torch_fl`'s GCU branch calls
`patch_diffusers_qwenimage_rope` at import, which registers the `flagos` device in
both halves of the extension point upstream provides for exactly this case:
`ROPE_PER_DEVICE` at the attention call site, and `_get_device_freqs` (on
`QwenEmbedRope` and `QwenEmbedLayer3DRope`) for the operand — which has to be
angles, because that method is cached per device and returns the complex
exponential for every device but `neuron`. The rotation it registers is
numerically the same rotation the complex path performs, i.e.
`apply_rotary_emb_qwen_neuron`'s contraction, asserted bit for bit in
`tests/unit/test_gcu_qwenimage_rope.py`; what differs is the spelling, which drops
the stride-0 `repeat_interleave` broadcast that function expands each angle with.
`FLAGOS_DISABLE_QWENIMAGE_ROPE=1` opts out and leaves the table as `diffusers`
ships it.

`QWEN_IMAGE_REAL_ROPE=1` is what is left for a device kind the plugin does not
cover: it makes `common.import_torch` do the same registration for the running
device type. On GCU it is now redundant with the shipped default — and it is
still meaningful, because `common.py` deliberately does not overwrite an entry
that is already set, so the shipped consumer survives it. The switch stays
documented because the harness runs on more than one backend and because the A/B
below is stated in its terms.

Report the leg both ways regardless, because the state changes what the run
measures: on the complex path the census carries the
`view_as_complex`/`view_as_real` traffic; on the angle path the rotation is
real-valued on the accelerator and those calls disappear. On GCU the two legs are
now `FLAGOS_DISABLE_QWENIMAGE_ROPE=1` and the default.

#### The switch is worth a third of the step

The paragraph above is about *routing*: the complex path no longer leaves the
card, so the switch is not the difference between a working run and a dead one.
It is still very much a difference in speed, and an earlier version of this
section overstated the case by calling it "no longer the difference between a
fast and a slow run". Measured on an S60 (2026-09-20), one transformer forward,
1024x1024 (4096 image tokens and 18 text tokens, 24 heads of head_dim 128, 60
blocks), transformer on `flagos:1`, the rope entry point wrapped at the call site
so each call is timed the way the model's own drain times it:

| | `QWEN_IMAGE_REAL_ROPE=0` | `QWEN_IMAGE_REAL_ROPE=1` |
| --- | --- | --- |
| forward | 6.629 / 6.639 s | 4.236 / 4.278 s |
| rope calls per forward | 240 | 240 |
| rope total | 3812.6 / 3852.8 ms | 1441.4 / 1469.6 ms |
| — mean | 15.886 / 16.053 ms | 6.006 / 6.123 ms |
| — share of the forward | 57.5 / 58.0 % | 34.0 / 34.3 % |
| `freqs` dtype at the call | `complex64` | `float32` |

So the rotation alone is 57.5 % of the forward on the complex path and 34.0 % on
the angle path, and turning it on takes the forward down by 36 %. The two rows
that identify the leg are the `freqs` dtype and the `ROPE_PER_DEVICE` keys
(`['cuda', 'neuron']` against `['cuda', 'flagos', 'neuron']`); a leg that reports
the wrong dtype was instrumented in the wrong place, which is easy to do because
the call site reads `ROPE_PER_DEVICE.get(img_query.device.type, ROPE_PER_DEVICE["cuda"])`
and only one of those two entries is live.

Do not measure this by registering the angle path *after* the pipeline is loaded.
The rotation is cached per device, so a late registration measures the complex
path under an angle-path label — an early attempt at exactly that A/B reported
the angle path 4.6x *slower* and its output checksum did not match the one a
fresh `QWEN_IMAGE_REAL_ROPE=1` process produced. The table above is two fresh
processes, each with the registration ordered as `common.import_torch` does it.

#### What the shipped registration adds on top of the switch

The table above is the harness switch alone. The plugin now installs the angle
operand *and* a cheaper expansion of the same rotation, so three legs are worth
measuring rather than two — and the third is what a user gets without setting
anything. One warm forward each, one process each, 1024x1024, transformer on
`flagos`, 3 timed forwards after 1 warm-up:

| | complex (opts out) | neuron (`QWEN_IMAGE_REAL_ROPE=1`) | shipped (default) |
| --- | --- | --- | --- |
| `FLAGOS_DISABLE_QWENIMAGE_ROPE` | `1` | `1` | unset |
| `ROPE_PER_DEVICE['flagos']` | *absent* — `cuda` fallback | `apply_rotary_emb_qwen_neuron` | `flagos_qwenimage_rotary_emb` |
| forward | 5.411 s | 3.392 s | **2.455 s** |
| rope total | 3818.7 ms | 1448.2 ms | **546.8 ms** |
| — share of the forward | **70.6 %** | 42.7 % | **22.3 %** |
| `freqs` dtype at the call | `complex64` | `float32` | `float32` |

240 rope calls per forward on all three legs, each timed with its own drain. The
shipped leg is **2.956 s per forward (54.6 %) faster than the fallback it
removes** and **0.937 s (27.6 %) faster than the angle path it replaces**; the
harness switch alone accounts for 2.019 s of the first number and the expansion
for the rest. The three forwards were saved and compared as full model outputs,
`(1, 4096, 64)` bf16:

| pair | `torch.equal` | max abs delta | elements differing |
| --- | --- | --- | --- |
| neuron vs shipped | **True** | `0.000e+00` | **0 / 262144** |
| complex vs shipped | False | `3.125e-02` | 142636 / 262144 |
| complex vs neuron | False | `3.125e-02` | 142636 / 262144 |

So the shipped consumer is exactly the neuron rotation end to end, and both real
legs sit two bf16 ulps from the complex path at the output's own magnitude
(`2^-6` at `absmax 5.3125`) — the accumulated effect of a complex multiply that
forms the two products of a pair in a different order. The three leg-identifying
rows are the table entry, its `__name__`, and the `freqs` dtype; the probe prints
all three before it loads the pipeline, so a leg labelled wrongly is visible
before any timing is taken.

Probe: `/tmp/probe_rope_shipped.py`, driver `/tmp/probe_rope_shipped.sh`,
`TOPS_VISIBLE_DEVICES=0,1,2`, saved outputs `/tmp/rope_out_{complex,neuron,shipped}.pt`.

#### What the switch costs in image quality, and what it is worth end to end

A third of the step is worth taking only if two things hold: the rotation has to be
the same rotation, and a real 50-step run has to be shorter and not merely a
wrapped forward.

**It is the same rotation.** On the model's own frequency table
(`QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56])`, `pos_freqs` `(4096, 64)`
`complex64`, its `torch.angle` `(4096, 64)` `float32`) and the model's own shape
`(1, 4096, 24, 128)` bf16 on `flagos:0`, both spellings measured against a
float64 reference of `(xr + i.xi)(cos + i.sin)`:

| | max abs delta | mean abs delta |
| --- | --- | --- |
| complex (default) | `1.561453e-02` | `1.121671e-03` |
| real (`QWEN_IMAGE_REAL_ROPE=1`) | `1.561453e-02` | `1.121671e-03` |

Identical to six digits against `|ref|max 5.429e+00`, so neither side is the more
accurate one. Against each other the two agree **bit for bit on 12,582,343 of
12,582,912 elements (99.9955 %)**, differ by at most one bf16 ulp (`1.5625e-02`
at magnitude ~5, which is 2^-6) and have mean abs delta `3.2e-08`; on the 569
elements that differ at all, the complex side is closer to the float64 rotation
on 178 of them and the angle side on 391. The switch is therefore not a numerical
shortcut — it is the same rotation reached through a different spelling, which is
what makes the residue below the size it is.

**And the loop is shorter.** Both legs of a 50-step `--stage full` run,
1024x1024, true CFG, seed 0, laid out as the vendor baseline is — encoder and VAE
on the third visible card, transformer blocks 0..29 and 30..59 on the first two —
one environment variable apart, back to back on the same cards:

| | `QWEN_IMAGE_REAL_ROPE=0` | `QWEN_IMAGE_REAL_ROPE=1` |
| --- | --- | --- |
| denoise loop | 50/50 in 11:15, **13.51 s/it** | 50/50 in 06:58, **8.38 s/it** |
| output | `/tmp/full_rope0.png` | `/tmp/full_rope1.png` |
| the two outputs against each other | `MAE 3.637/255`, `PSNR 28.58 dB`, 69.61 % of bytes differing | |
| the vendor baseline's `4.01 s/it` | 3.37x | **2.09x** |

The `1` column is the harness switch, not what ships. Neither `QWEN_IMAGE_REAL_ROPE`
is set by default and the plugin's own registration is faster than that switch, so
the shipped leg reads **`5.01 s/it` against the vendor's `3.81` — `1.31x`**; see
§5.3, which supersedes the `2.09x` on this row.

That is 38 % off the whole run, the same order as the 36 % the wrapped forward
measured, so the wrapper was not manufacturing it. The two images are not
bit-identical and should not be: a one-ulp difference on 0.0045 % of one op's
elements is amplified by 50 steps of a chaotic loop. For scale, the same pair of
spellings over 8 steps is `PSNR 33.86 dB` (the 2026-09-18 table above), and a
paired A100 run over 50 steps is 35.4 dB (§5) — so the two spellings differ by
about as much as two backends differ at all, and the number that moves with the
rollout length is the amplification, not the arithmetic.

The vendor baseline's `perf/gcu_full.png` is *not* a usable anchor for either leg:
it differs from both by `MAE ~47/255` (`PSNR 11.97` and `11.91 dB`), which is an
image-level difference and not a rounding one, so it is an independently seeded
generation. §5 says to read two independently seeded runs as a quality check on
the images and not as a number, and that is what this is.

#### Why the complex multiply costs what it does

On this build a `complex64` multiply does dispatch to `gcu` — `FLAGOS_LOG=dispatch`
prints `mul.Tensor -> gcu` and `FLAGOS_LOG=fallback` prints nothing at all, so it
is not a host round trip. It is simply very slow, and the cost is the arithmetic
rather than the bytes. Same card, same shapes, ten calls each, minimum reported:

| op | operands | ms/call |
| --- | --- | --- |
| `complex64` mul, `(1,4096,24,64)` x itself | 50.3 MB each | 50.166 |
| `complex64` mul, `(1,4096,24,64)` x `(4096,1,64)` | 50.3 MB | 36.412 |
| `complex64` `clone` | 50.3 MB in, 50.3 MB out | 0.547 |
| `complex64` `copy_` | 50.3 MB in, 50.3 MB out | 0.583 |
| `float32` mul, `(1,4096,24,128)` x itself | 25.2 MB each | 0.495 |
| `out_real = ar*wr - ai*wi` | 25.2 MB | 1.557 |
| `out_imag = ar*wi + ai*wr` | 25.2 MB | 1.571 |
| `torch.angle(complex64)` | 50.3 MB | 134.091 |

Copying the identical bytes takes 0.55 ms and the same rotation written as two
real multiplies takes 3.1 ms, so the complex multiply is about 91x a copy of the
same size and about 16x the equivalent real arithmetic. That ratio is what the
angle path collects, and it is a property of this build's complex kernel rather
than of the rotation: a real-valued rewrite of it is *cheaper*, not merely
equivalent. The `torch.angle` row is why `_get_device_freqs` matters — it is
called once per forward and its result is reused by all 240 rotations, so its
134 ms is amortised; a version of the wrapper that dropped the original
`lru_cache` would call it far more often and would measure the host rather than
the kernel.

Per-op CPU time from `torch.profiler` does not show any of this. The same step
profiled with `record_shapes=True` reports `aten::mul` at 2.6 ms a call on the
complex path, because a GCU kernel's wait is attributed to
`topsStreamSynchronize` rather than to the op that launched it. Time this branch
at the call site, or with a profiler that reports wall time per op.

Measured on an S60 (2026-09-18), flagos transformer over `flagos:6,7`, encoder and
VAE on `flagos:3`, 1024x1024, 3 steps, seed 42, and the same paired prompt embeds
and initial latents on both sides:

| | `QWEN_IMAGE_REAL_ROPE=0` | `QWEN_IMAGE_REAL_ROPE=1` |
| --- | --- | --- |
| dispatch records (3 steps) | 42,336 | 48,096 |
| — native `gcu` | 31,774 | 37,534 |
| — FlagGems (`flagos_python`) | 10,562 | 10,562 |
| `view_as_complex` calls | 1,440 | **0** |
| `view_as_real` calls | 1,440 | **0** |
| `cpu_fallback` | 0 | 0 |

So the switch does what the census claim above says: 2,880 view calls are replaced
by 5,760 further `gcu` calls, which is the angle path's explicit `cos`/`sin` and
the multiplies they need, and the FlagGems count is untouched. Both images were
then compared (`run.sh compare`): `MAE 2.02/255`, `PSNR 33.86 dB` over the three
steps. The two paths reorder the same rotation rather than computing different
ones, so they are close but not bit-identical, and a diffusion loop amplifies the
residue; at 50 steps that is the same order as the difference between two
backends at all (§5). Neither side reached `cpu_fallback`: on this build
`view_as_complex` and `view_as_real` have native `gcu` kernels of their own, so
the complex path no longer costs host round trips either. What the switch buys on
this build is the ratio priced below — 36 % of the forward — and not a fallback.

A vendor run takes the same switch, because the registration is keyed on
`--device` rather than on the flagos backend: `torch_gcu` renames PrivateUse1 to
`gcu`, so a vendor run registers `gcu` and the rotation becomes real-valued
there too. `test_rope_hook.py` covers both interpreters, and which one it is
looking at is decided by `QWEN_IMAGE_ROPE_TEST_DEVICE`: on a `flagos` run the
plugin has already wrapped the operand producer and set the table entry by the
time the script starts, so what is checked there is that the module switch and
this file's installer leave that pair alone, while every other device kind
checks the registration this file makes. Where the interpreter can construct the
device (the vendor build can; a CPU-only torch refuses `gcu` at the device-string
parse) it also checks that the wrapped method returns `torch.angle` of the
frequencies placed on that device. Each run prints which installer it found, so
an arm that skipped a check cannot read as one that passed.

```bash
QWEN_IMAGE_REAL_ROPE=1 tests/manual/qwen_image_2512/run.sh infer --stage transformer-step
QWEN_IMAGE_REAL_ROPE=1 tests/manual/qwen_image_2512/run.sh infer --stage transformer-step --device gcu
```

On GCU a `--device flagos` run no longer needs the variable: `import torch_fl`
installs the same rotation, minus the `repeat_interleave` expansion, and
`FLAGOS_DISABLE_QWENIMAGE_ROPE=1` is the opt-out. The variable is still what a
`--device gcu` run takes, and still what the A/B in §3.3 is stated in.

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

### 5.1 Three legs on the S60: one kernel, measured end to end

The S60 workstream rerouted `_scaled_dot_product_efficient_attention` from ATen's
math decomposition to the vendor flash op. The route is fixed at build time — no
runtime flag can move it — so this is the shape of the measurement: **three legs,
two builds of the same tree**, and the paired inputs from this section carrying
between them.

| Leg | Configuration | `s/it` at 8 steps | vs the vendor |
| --- | --- | --- | --- |
| before | the math decomposition (op out of `HANDWRITTEN_OPS`) | **19.33** | 5.07x |
| after | the vendor flash op (the state at `486db58`) | **8.48**, and 8.81 on a repeat | **2.23x** |
| reference | `torch_gcu` + diffusers | **3.81** | 1.00x |

Leg 1 wrote its latents and prompt embeds with `--save-latents` / `--save-inputs`
and legs 2 and 3 read them back, so all three consumed identical inputs. Every leg
set `QWEN_IMAGE_REAL_ROPE=1` (the vendor leg cannot run without it: `topsaten` has
no complex kernel) and every leg had encoder and VAE on the third visible card
with transformer blocks 0..29 and 30..59 on the first two, `TOPS_VISIBLE_DEVICES=
0,1,2`.

The `after` row is the state at `486db58` and the `8.48`/`2.23x` on it are no
longer what a user gets: it is the harness switch, and §3.3's registration is
faster than that switch. **For the shipped configuration read §5.3's `5.01 s/it`
and `1.31x` instead** — this table is kept because the two legs below it are a
build pair and the shape of that measurement is the point.

**19.33 -> 8.48 is 2.28x, and 56.1 % off the denoising loop**; against the repeat,
2.19x and 54.4 %. The repeat is the same build run a second time and it produced a
byte-identical image (`md5 d47974b1…`), which is the evidence that the leg is
deterministic and the first number is a reading rather than a lucky one.

The two `torch_fl` images differ by `MAE 3.568/255` (`PSNR 30.10 dB`), and both sit
about the same distance from the vendor — 26.96 dB for the shipped leg, 26.91 dB
for the math path, differing by 0.05 dB in the shipped leg's favour. So the flash
kernel is a real numerical change at the image level (eight steps amplify a bf16
noise floor into 30 dB), and it is not a step away from the reference; the ~27 dB
that remains to `torch_gcu` is the rest of its graph.

For the record, the route cannot be A/B'd at runtime and two attempts to do so are
worth not repeating: `FLAGOS_OP__scaled_dot_product_efficient_attention=none` is
refused outright because the conf route and the PrivateUse1 registration must
agree, and `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])` is a silent no-op
here — that run returned a byte-identical image to the shipped leg and still
logged 960 `-> gcu` dispatches, since torch_fl's `__torch_function__` picks the
route before ATen's backend pin is consulted.

### 5.2 Four more commits on the same tree

The branch then added four commits, none of which moves a route: each one changes
how an operand is described, or reads a marker file once instead of once per
stream-helper call. Measured the same way as §5.1 — one build pair at a time,
control and patched legs back to back on the same cards, the paired inputs
carrying through:

| Leg | Configuration | `s/it` at 8 steps |
| --- | --- | --- |
| control | clean at `486db58` | 8.80, 8.85 |
| + | the accelerator marker read once per process | 8.05, 8.07 |
| + | the addmm weight handed over as a transpose view | 7.44, 7.55 |
| + | the `cat` inputs handed over as they are | 7.50, 7.51 |
| + | the addmm bias handed over as a rank-1 vector | **7.13, 7.15, 7.10** |

The `cat` commit is a null result and is kept as one: it removes 240 of the 780
`topsatenCopy` calls a step issues and the step wall does not move (3.627 s ->
3.645 s), because a drain waits for whatever the device has queued rather than
paying a fixed cost of its own. That is also why its row reads 0.03 s/it *higher*
than the row above it: the two are inside the same spread.

Every leg except the last writes `md5 d47974b1…` byte for byte, at every repeat.
The last writes `0e262317…`, `PSNR 26.93 dB` from its own control, and it is the
one leg in this document whose image moves. It moves toward the vendor: `29.85 dB`
from `torch_gcu` against the control's `26.96 dB`. Against a float64 reference at
the model's mlp shape the maximum error falls from 0.95 ulp to 0.49 ulp, better
at 25.6 % of elements and worse at 0.0 % of them, because the bias now takes the
route the vendor's own `addmm` takes instead of a zero-stride `(M, N)` view that
costs a second pass over the output. The per-op probe sees the same thing from the
other side: the fused call drops from 2643.2 us to 1575.5 us of device time at
(4096,3072)x(3072,12288) bf16, which is exactly what a bare `mm` costs there.

### 5.3 The rotation registration, measured on the shipped leg

§3.3 times the rope registration on a single warm forward, because that is the
only way to see the operand and the expansion separately. What the harness ships
is the 50-step loop, where a step is two forwards under true CFG, so the same
change should be worth about twice as much per step. Three legs, one session,
back to back on the same cards with a warm page cache, one environment variable
apart, 8 steps at 1024x1024, seed 0:

| Leg | Configuration | `s/it` at 8 steps | stage wall | vs the vendor |
| --- | --- | --- | --- | --- |
| opt out | `FLAGOS_DISABLE_QWENIMAGE_ROPE=1` | **11.81** | 267.0 s | 3.10x |
| shipped | registration installed at import, nothing set | **5.01** | 212.4 s | **1.31x** |
| reference | `torch_gcu` + diffusers | **3.81** | 197.0 s | 1.00x |

The registration is worth **6.80 s per step, 57.6 % of the loop**, or 3.40 s per
forward. §3.3 measured 2.956 s for the same change with each rope call timed
against its own drain, so the step-level gain is larger than twice the isolated
one. That is the direction §5.2's `cat` row already reports: per-copy cost does
not project linearly onto a step, because a drain waits for whatever the device
has queued rather than paying a fixed cost per operation. `5.01` reproduced as
`5.00` on a separate run in the same session, so the reading is stable to 0.2 %.

**This supersedes the `8.38 s/it` / `2.09x` headline in §3.3 and the `8.48 s/it` /
`2.23x` "after" row in §5.1 for the configuration a user actually gets.** Both of
those legs were the harness switch — `QWEN_IMAGE_REAL_ROPE=1`, which selects
diffusers' `apply_rotary_emb_qwen_neuron` — and the shipped registration is
faster than that switch by the 0.937 s per forward §3.3 records.

A companion run tried to split the step further, by timing `text-encoder` and
`vae` on their own so that `wall(vae) - wall(text-encoder)` would isolate the
1024x1024 decode. That subtraction carries no signal and the arm is best not
repeated: both stages pay the same ~155-165 s pipeline load (164.1 s and 161.2 s
on `flagos`, 156.3 s and 153.7 s on the vendor), which is larger than the decode
and moves more between repeats than the decode is worth. The `s/it` line of the
`full` stage is the only timing these two arms can contribute.

Logs: `/tmp/rope_ab_{rope_on,rope_off,vendor}.log`, driver `/tmp/run_rope_ab.sh`; the two
discarded arms are `/tmp/decomp_{flagos,vendor}.log`, driver
`/tmp/run_stage_decomp.sh`. `TOPS_VISIBLE_DEVICES=0,1,2` on the vendor leg, and
the vendor leg runs through its own interpreter as §4 describes. No leg sets
`QWEN_IMAGE_REAL_ROPE`: the opt-out leg uses the switch the change itself ships,
so both `flagos` legs run diffusers' table unless the plugin replaces it.

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
| `cpu_fallback ops` | Operator calls that left the accelerator for the CPU (`FLAGOS_LOG=fallback`). Every one is a route to fix; the op names are in the log. |
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
tables: `FLAGOS_LOG=dispatch` logs the backend actually chosen for every
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
by running §4's flagos sweep with `FLAGOS_LOG=dispatch`, which `run.sh` exports
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
