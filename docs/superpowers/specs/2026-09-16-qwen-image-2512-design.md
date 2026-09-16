# Qwen-Image-2512 on torch-fl — design document

Date: 2026-09-16
Status: Approved for implementation planning

## 1. Goal

Run the `QwenImagePipeline` from the Hugging Face `diffusers` library on the
`flagos` device of torch-fl, and produce a 1024x1024 image from a text prompt on
local NVIDIA A100 hardware.

The deliverable is a reproducible inference entry point plus the record of what
the backend had to fix to get there. This is the first diffusion-model workload
on torch-fl: every previous end-to-end model validation in this repository is
Transformers inference (`tests/integration/test_qwen3_infer.py`), Transformers
training (`tests/integration/test_qwen3_train.py`), or `torch.compile`
(`tests/integration/test_compile.py`). The operator and runtime surface a
diffusion pipeline exercises is different enough — causal `Conv3d` in the VAE,
complex-dtype rotary embeddings in the transformer, a 50-step dual-forward
denoising loop — that the existing coverage record does not predict the outcome.

## 2. Non-goals

The following are explicitly out of scope for this change:

- CI integration. The inference script lands under `tests/manual/`, which the
  CUDA manifest in `.github/configs/cuda.yml` does not execute. Adding it would
  require mounting a ~58 GB model into the CI container and installing
  `diffusers` there; that is a separate decision with its own cost.
- Quantization (bitsandbytes, torchao, GGUF). The memory problem is solved by
  layer splitting, not by reduced precision.
- Other accelerator vendors. The scripts are backend-generic — `--device` names
  a torch device module — so bring-up on another chip is the same procedure, and
  §8's readings are what to compare against. But only CUDA-compatible hardware is
  measured here; no other vendor's numbers are claimed.
- Performance tuning. The pipeline is expected to be correct and reproducible,
  not fast. Step timing is recorded, not optimized.
- Patching `diffusers` or `transformers`. See section 5.

## 3. Environment and hardware

| Field | Value |
| --- | --- |
| Branch | `feat/qwen-image-2512`, based on `flagos/main` |
| PyTorch | 2.10.0+cpu (CPU wheel) plus external CUDA 12.8 assets |
| Conda environment | `torch-fl-210` |
| Hardware | 8x NVIDIA A100-SXM4-40GB, all idle at design time |
| GPUs used | 3 (`flagos:0`, `flagos:1`, `flagos:2`) |
| Model | `Qwen/Qwen-Image-2512`, ~57.7 GB on disk |

`torch-fl-210` matches the repository's `TORCH_PIN = "torch>=2.10,<2.11"`. The
`torch-fl-211` environment exists but installs torch-fl from the 2.11 worktree,
which does not match this branch.

`diffusers` is not installed in either environment and must be added, together
with `Pillow` for image output.

## 4. Component placement and data flow

```text
prompt --> Qwen2Tokenizer --> input_ids
             |
             v  Qwen2_5_VLForConditionalGeneration        flagos:2   16.58 GB
         prompt_embeds [1, L, 3584] + prompt_embeds_mask
             |
latents = randn[1, 16, 128, 128] on flagos:0  --+
             |                                  | 50 steps x 2 forwards (true CFG)
             v                                  |
   QwenImageTransformer2DModel                  |
     |- img_in / txt_in / time_text_embed       |
     |- transformer_blocks[0:30]   flagos:0     |  ~20.4 GB each
     |- transformer_blocks[30:60]  flagos:1     |
     +- norm_out / proj_out                    |
             |                                  |
             v                                  |
   FlowMatchEulerDiscreteScheduler.step  <------+  latents stay on flagos:0
             |
             v  AutoencoderKLQwenImage (causal Conv3d)   flagos:0   253 MB
          image[:, :, 0] --> PNG
```

The split is required, not a preference: the transformer alone is 40.86 GB of
bf16 weights, which exceeds the 40 GB HBM of a single A100. The boundary has to
fall between transformer layers, with `img_in`/`txt_in`/`time_text_embed`
travelling with block 0 and `norm_out`/`proj_out` with block 59.

The text encoder is placed on a third GPU rather than sharing one with the
transformer, so that it can be released from the device between the prompt
embedding step and the denoising loop without competing for HBM.

## 5. Layer splitting mechanism

Use `accelerate.dispatch_model(transformer, device_map)` with an explicitly
constructed device map of the form
`{"transformer_blocks.0": torch.device("flagos:0"), ..., "proj_out": torch.device("flagos:1")}`.

Two decisions are worth recording.

**The device map is built by hand, not with `device_map="balanced"`.** Automatic
balancing runs `infer_auto_device_map`, which probes free memory through
`torch.cuda.mem_get_info`. The split point is known in advance from the layer
count and the parameter count, so there is no reason to route the placement
through a code path that touches CUDA-specific introspection. Fewer moving
parts on the flagos path means fewer places to debug.

**The accelerate dispatch path was checked for device-type assumptions.** The
relevant modules (`accelerate/hooks.py`, `accelerate/utils/operations.py`) move
weights with `.to()` and move cross-module inputs through hooks; the only place
that reads a device is `next(module.parameters()).device` in
`hooks.py:759`. None of it is CUDA-specific. If `check_cuda_p2p_ib_support` or
device-string parsing fails on `flagos`, the fix belongs in torch-fl.

Host RAM is 893 GB available with 256 cores, so the straightforward approach —
load the model fully on CPU, then dispatch — is viable. Streaming loader
intricacies (`low_cpu_mem_usage`) are not worth the complexity here.

## 6. Compatibility fixes belong in torch-fl

If the pipeline hits a compatibility problem, the change is made in torch-fl,
not in a user script and not by patching `diffusers` or `transformers`. The
repository must not carry a private fork of either library; a user should be
able to `pip install diffusers` and run the local inference script unchanged.

The one thing this rule does **not** cover is behavior that is genuinely
diffusers-specific and cannot be expressed as a torch-fl fix. If such a case
appears, it is raised during implementation rather than worked around silently.

### 6.1 Already resolved by existing code

The generator-device problem is already handled inside torch-fl.
`diffusers.utils.torch_utils.randn_tensor` raises
`ValueError("Cannot generate a {device} tensor from a generator of type
{gen_device_type}.")` when a CUDA-typed generator is paired with a non-CUDA
device. `scripts/codegen/codegen_ops.py:_generator_inject_line` already
translates a `flagos` (PrivateUse1) generator into a CUDA one before the inner
ATen redispatch — that code exists precisely for `diffusers randn_tensor`, per
its own comment. No diffusers-side shim is needed as long as the user
constructs the generator on the `flagos` device.

### 6.2 Import order is load-bearing

`import torch_fl` must precede `import torch`. `_preload_cuda_assets()` in
`torch_fl/__init__.py` dlopens the bundled `libtorch_cuda.so` before PyTorch
caches its CUDA hooks; when the order is reversed the assets never load and
`torch.Generator(device="cuda")` fails with

```text
RuntimeError: Cannot get CUDA generator without ATen_cuda library.
```

This was reproduced directly on the target machine. The inference script must
therefore import `torch_fl` first, and the access guide must state the
requirement.

## 7. Known gaps to verify

The design does not predict which of these will require a torch-fl change. They
are listed in the order they are expected to be hit, so that a failure is
attributed to the right stage during component-by-component bring-up.

| # | Gap | Current state | Response |
| --- | --- | --- | --- |
| 1 | Attention implementation chosen by `Qwen2_5_VLForConditionalGeneration` on `flagos` | Transformers defaults to SDPA; `is_torch_sdpa_available()` behavior for PrivateUse1 is unknown | Bring up the text encoder alone; fall back to `attn_implementation="eager"` if needed; fix in torch-fl |
| 2 | Complex-dtype RoPE correctness in bf16 | `ROPE_PER_DEVICE` in `diffusers/models/transformers/transformer_qwenimage.py` has no `flagos` entry, so it silently uses the `"cuda"` entry (`use_real=False`), which goes through `torch.view_as_complex` / `torch.view_as_real` | Both ops are registered; verify numerically |
| 3 | VAE causal `Conv3d` decode at 1024x1024 | `Conv3d` is registered and verified working on `flagos` by direct test | Run VAE encode/decode in isolation |
| 4 | `dispatch_attention_fn` default `NATIVE` backend on `flagos` | Defaults to `F.scaled_dot_product_attention`; the backends guarded by `_check_device_cuda` are unreachable | Verify by running; measure |
| 5 | Cross-GPU boundary transfer | Two `[1, seq, 3584]` `.to()` calls per step across the layer boundary | Small; not optimized in this change |

Facts already established by direct measurement on the target machine, which
remove them from the risk list: `flagos` exposes 8 devices; cross-device
operators work; `Conv3d` forward works on `flagos:1`; a `torch.Generator` works
with a `flagos:5` tensor.

## 8. Verification strategy

### 8.1 Component bring-up

Each component is exercised on its own before the full pipeline is attempted.
This is what makes an unknown gap attributable: a failure in a 50-step loop with
two forwards per step is far more expensive to localize than a failure in an
isolated text-encoder call.

1. Text encoder alone: prompt -> `prompt_embeds` and mask.
2. Transformer single step: fixed latents, `num_inference_steps=1`, guidance
   disabled.
3. VAE alone: encode/decode round-trip.
4. Full pipeline: fixed prompt and seed, 50 steps, image written to PNG.

### 8.2 Reproducibility

Two runs with the same seed must produce bit-identical output. This validates
flagos-internal determinism and is a prerequisite for the comparison in 8.3.

### 8.3 Correctness comparison against real CUDA

The reference run uses the `torch-cuda-210` environment (torch 2.10.0+cu128) on
the same machine and the same `diffusers` version, with **the same initial
latents and the same `prompt_embeds` injected on both sides**. Injecting the
inputs removes the generator from the comparison, so the only variable left is
the operator implementations on the two backends. Comparing whole-pipeline
outputs with independent RNG would test RNG equality instead, which is neither
guaranteed nor what this change is about.

The images are compared with PSNR and mean absolute error, not bitwise. bf16
accumulation order differs between backends and bitwise equality is not
achievable at 50 steps of iterative refinement.

### 8.4 Operator coverage

The operators actually reached during the run are recorded, with those falling
through to `cpu_fallback` called out explicitly. The fallback set is the
concrete follow-up list for future operator work.

## 9. Deliverables

| Artifact | Location | Purpose |
| --- | --- | --- |
| Staged runner | `tests/manual/qwen_image_2512/infer.py` | Runs the pipeline; flags for prompt, seed, steps, device placement, output path |
| Cohort sweep | `tests/manual/qwen_image_2512/sweep.py` | Every prompt on the model card, in the card's own settings, one PNG per prompt plus a manifest |
| Comparison sheets | `tests/manual/qwen_image_2512/side_by_side.py` | Pairs a vendor cohort with a flagos cohort for the eyeball comparison |
| Numeric comparison | `tests/manual/qwen_image_2512/compare.py` | PSNR and MAE between two images |
| Shared plumbing | `tests/manual/qwen_image_2512/common.py` | Import order, placement, split, memory reporting |
| Runner | `tests/manual/qwen_image_2512/run.sh` | Environment, one mode, and the log census |
| Access guide | `tests/manual/qwen_image_2512/README.md` | The procedure below, the per-stage expectations, and the readings to record per chip |
| Access guide (CUDA) | `docs/vendors/cuda/qwen-image-2512.md` | Component placement, measured commands, step timing, known limitations, operator coverage summary |
| torch-fl fixes | wherever the gaps in section 7 land | Unknown in number until the pipeline runs |

`tests/manual/` is the correct home for the script: the CUDA manifest in
`.github/configs/cuda.yml` lists the tests it executes, and nothing under
`tests/manual/` is in that list. A script there runs locally without implying CI
coverage.

Where a fix touches operator registration or routing, section 10 applies.

## 10. Operator support record

If any change in this work adds, enables, removes, disables, or reroutes an
operator, `docs/reference/operator-support.md` is updated in the same change,
with `tests/manual/flaggems_overload_survey.py` rerun against the affected
hardware and the summary, raw evidence, provenance, and update history taken
from the measured results.

The A100 hardware used here is the same hardware the CUDA rows already cover, so
a revalidation is possible and no row needs to be marked *not revalidated*. If
the pipeline reaches an operator that routing configuration alone cannot
characterize, the survey is the instrument for measuring it, not an inference
from the config file.

## 11. Risks

**The number and nature of the required torch-fl fixes is unknown.** That is the
point of approach A: the gaps are found by running, not by static analysis of
the routing tables. The component-by-component bring-up in section 8.1 is the
mitigation — it bounds where an unknown gap can surface, and it front-loads the
text encoder and VAE, which are the two components with new operator surfaces
(complex rotary embeddings and causal `Conv3d` respectively).

**bf16 numerical drift makes the reference comparison a tolerance check, not an
equality check.** This is expected and is a property of the precision, not a
defect in the backend.

**The model download is ~58 GB over a proxy.** Disk space is not a constraint
(1.3 PB available). The download is a one-time cost.

**The two-GPU split may not be enough.** If peak activation memory at
1024x1024 pushes a shard over 40 GB, the response is to move the split boundary
and spread across more of the available 8 GPUs, not to add quantization.

## 12. Success criteria

The change is complete when all of the following hold:

1. The inference script produces a 1024x1024 PNG from a text prompt on `flagos`.
2. Two runs at the same seed produce bit-identical output.
3. The flagos output and the real-CUDA reference output, run with identical
   injected latents and prompt embeddings, are within a stated PSNR tolerance.
4. Every torch-fl fix the work required is committed to `feat/qwen-image-2512`
   and documented in the access guide.
5. If any operator routing changed, `docs/reference/operator-support.md` is
   updated from a fresh hardware measurement in the same change.
