# InNata-0 torch-fl inference parity on Hygon DCU

**Status: measured parity.** On a single Hygon DCU (DTK 6.3) the `flagos`
CUDA-boxing route runs InNata-0 inference at 0.96-1.04x of the vendor torch
build of the same version across the five fixed requests: ACWM 30 denoise
steps take **19.970 s** versus **20.021 s** (three measured rounds, averaged),
the 9-frame Understanding request **4.31 s** versus **4.48 s**, and both WAM
requests within 3% of vendor. Full-run dispatch audits report **zero
`cpu_fallback`** ops. This document records the baseline, the four measured
causes, the delivered levers, and the parts of the recipe that generalise.

Acceptance line: sampling and decode each within 0.95x of vendor torch, with
no CPU fallback.

The wheel measured here is `0.1.0+dtk`, predating the Native Host programs
merged in #917; the route under test is the DCU CUDA-boxing route (generated
PrivateUse1 kernels dispatching into `libtorch_hip.so`), not the generated
host programs. The same accounting of causes applies to both.

## Environment

| Item | Value |
|---|---|
| torch_fl | 0.1.0+dtk, DCU boxing build (`lib_dcu/libtorch_hip.so`) |
| frontend torch | 2.10.0+cpu, DTK device libraries preloaded by torch_fl |
| vendor torch | 2.10.0 (DTK build, `hip 6.3.26113`, `cuda` device) |
| flag_gems | 5.4.0rc2 |
| flash_attn | 2.8.3+das185.dtk2604.torch2100 |
| DTK | 6.3 (dtk2604) |
| Model | InNata-0 Understanding, ACWM, WAM-RoboDojo, WAM-RoboCasa365, bf16, eager |
| Bench | model CLI; built-in `--benchmark` timer for ACWM, 30-step progress timing for the WAM requests, a per-generate hook for Understanding; one card |

Vendor torch runs from the same image (its DTK wheel, `device=cuda`) with the
application source copied and the `flagos` device name rewritten to `cuda`.
Both routes run the same request JSON, seed, and `--no-use-torch-compile`.

## Baseline

Before the change the ACWM request took **117.9 s / 30 steps** (3.93 s/it),
RoboDojo **28.8 s**, RoboCasa365 **41.4 s**, Understanding BLINK **0.414 s**
and the 9-frame request **12.14 s** on the `flagos` route.

## Measured causes

1. **Fused SDPA was never enabled.** The delivery image ships no
   `flash_attn_2_cuda*.so`, so `_dcu_compat._configure_dcu_sdpa_backends()`
   disables the fused backends and `scaled_dot_product_attention` runs the
   math decomposition.
2. **Grouped KV heads were materialised.** The application's portable SDPA
   backend expanded GQA with `repeat_interleave`; at 2048/12576 tokens the
   fused kernel takes 0.44/10.7 ms with `enable_gqa` versus 1.19/34.6 ms with
   the expansion.
3. **Hot ops ran through the FlagGems Python bridge.** A dispatch census
   (`FLAGOS_LOG=dispatch`) of two ACWM steps showed the sampling loop calling
   `mean.dim`, `pow.Tensor_Scalar`, `rsqrt`, `neg`, `silu`, `_unsafe_view`,
   `where.self`, plus VAE-side `native_group_norm`, `sigmoid`, `gelu`,
   `linalg_vector_norm` and friends on the `flagos_python` route. Separately,
   the VAE `RMS_norm` layer's `F.normalize` is a 5-D `vector_norm` that costs
   ~1.48 ms per call through FlagGems at decoder shapes (570 calls per decode).
4. **`torch.equal` has no boxing kernel.** `Action128.validate_contract()` ran
   one `[flagos cpu_fallback] aten::equal` per sampling step.

Per-op dispatch cost itself is not the gap: back-to-back microbenchmarks on
the same card measure `unsqueeze` 2.68 us vs 2.47 us, `transpose` 2.81 us vs
2.64 us, and `empty_like` 3.69 us vs 3.76 us (flagos vs vendor). The residual
was the ops that stayed on the Python bridge and the missing fused kernel.

## Delivered levers

1. **Load the matching flash library and name it explicitly.** Point
   `FA_SO_PATH` at `flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so`. DTK's
   CUTLASS adapter otherwise looks three parents up from the loaded
   `libtorch_hip.so`, which works for a vendor `torch/lib` copy but not for a
   package installed elsewhere.
2. **Use `enable_gqa`** in the SDPA backend instead of expanding KV heads.
3. **Move the measured hot operators to the `cuda` route** with per-op
   overrides. The DCU profile in this workload reroutes 58 operators
   (`mean.dim`, `pow.Tensor_Scalar`, `rsqrt`, `neg`, `silu`, `_unsafe_view`,
   `constant_pad_nd`, `clamp_min`, `scalar_tensor`, `fill_.Scalar`,
   `fill_.Tensor`, `uniform_`, `erfinv_`, `clamp_`, `zero_`, `pow.Scalar`,
   `reciprocal`, `zeros`, `cos`, `sin`, `cos_`, `randn`, `bitwise_not`,
   `addmm`, `masked_fill.Scalar`, `eq.Tensor`, `arange`,
   `arange.start_step`, `embedding`, `cumsum`, `rsub.Scalar`, `nonzero`,
   `lt.Scalar`, `sum`, `index_copy`, `gt.Scalar`, `all`, `ones_like`,
   `ne.Scalar`, `isin.Tensor_Tensor`, `bitwise_and.Tensor`,
   `native_group_norm`, `native_layer_norm`, `sigmoid`, `tanh`, `exp`,
   `gelu`, `sqrt`, `_softmax`, `avg_pool2d`, `clamp`, `index_add`,
   `reflection_pad2d`, `where.self`, `zeros_like`, `any`,
   `repeat_interleave.Tensor`, `sum.dim_IntList`) from `flaggems` to `cuda`.
   The resulting table has 406 flaggems / 1636 cuda routes and no
   `cpu_fallback` entries.
4. **Rewrite two model-side spots that route poorly.** The VAE `RMS_norm`
   layer computes `(x*x).sum(dim, keepdim=True).sqrt().clamp_min(1e-12)`
   instead of `F.normalize` (0.17 ms vs 1.48 ms per call at decoder shapes),
   and the action-contract digest check reads the 16-byte digest once over a
   host read and caches the result instead of calling `torch.equal` per step.

## Results

Same host, same request and seed. Understanding is the per-request
`model.generate` time (mean of 3-5 runs after a warmup), the generation rows
are 30-step denoise sampling; neither includes checkpoint loading or VAE
decode. Relative performance is vendor / version, so values above 1 mean the
version is faster than vendor.

| Fixed request | Old torch-fl | Vendor torch 2.10 | Old vs vendor | Optimized torch-fl | Optimized vs vendor |
|---|---:|---:|---:|---:|---:|
| Understanding, BLINK image | 0.414 s | 0.168 s | 0.41x | 0.174 s | **0.96x** |
| Understanding, MMSI 9 frames | 12.14 s | 4.48 s | 0.37x | 4.31 s | **1.04x** |
| ACWM, 73-frame video | 117.9 s | 20.02 s | 0.17x | 19.97 s | **1.00x** |
| WAM-RoboDojo, 16x14 actions | 28.8 s | 23.7 s | 0.82x | 24.0 s | **0.99x** |
| WAM-RoboCasa365, 16x12 actions | 41.4 s | 35.1 s | 0.85x | 36.2 s | **0.97x** |

Supporting ACWM timer readings: `generate_samples_from_batch` 19.970 s versus
20.021 s, `decode` 3.871 s versus 3.877 s, `generate_batch` 24.547 s versus
24.605 s; end-to-end (including the shared checkpoint load) 325 s versus
321 s. ACWM sampling improved **5.9x** over the old delivery image, and the
old image's Understanding and RoboCasa numbers were re-measured with the same
instrumentation so the columns are comparable.

Dispatch audit over complete runs: ACWM 161148 dispatch lines, RoboDojo
25414, `[flagos cpu_fallback]` = **0** in both. The only operators left on the
Python bridge in the sampling loop are `ge.Tensor` and `abs`, once per step.
The Understanding path still makes about 50 bridge calls per request
(`eq.Scalar`, `floor_divide`, `full`, `bitwise_or.Tensor`, `ones`, `max`,
`argmax` and a few more); moving those to the `cuda` route is the remaining
lever for the short single-image request, which sits at 0.96x.

Functional output is unchanged within bf16/flash rounding: ACWM stays
736x544, 73 frames at 16 fps with mean |delta| 0.657/255 against the
reference, and RoboDojo stays 16x14 with metadata identical and
max |delta| 0.0077.

## Reusable notes

- **One copy of the flash library per process.** The DTK adapter and an
  `import flash_attn` that resolves to a second copy both register the
  `flash_attn2_c_op` namespace, and the second registration aborts with
  `Only a single TORCH_LIBRARY can be used to register the namespace`.
  Either remove the duplicate or shadow the Python package and leave only
  `FA_SO_PATH`; selecting the same file on both paths is safe.
- **`FLAGOS_FORCE_BACKEND=cuda` does not move `flaggems`-only operators.**
  The force path judges availability from the conf annotation, not from the
  compiled kernel set, so operators that only appear as `flaggems` stay there
  even though a boxing kernel exists. Per-op `FLAGOS_OP_<op>=cuda` overrides
  (dots become `__`, e.g. `FLAGOS_OP_mean__dim=cuda`) and a conf copy are the
  effective mechanisms.
- **`linalg_vector_norm` cannot be boxed on this DTK build.** Calling it
  through the `cuda` route raises `hipErrorNoBinaryForGpu`; it works on vendor
  torch and on FlagGems. Leaving it on `flaggems` and rewriting the hot
  call site avoids both costs.
- **`torch.equal` has only a `cpu_fallback` kernel.** Adding it to the conf
  does not help; replace the call site.
- **No `PYTHONPATH` in a `flagos` process.** With any `PYTHONPATH` present,
  device initialization fails with `Cannot initialize CUDA without ATen_cuda
  library`.

## Scope

The 58-operator list is a profile of this model's shapes on this hardware,
not a proposal for the DCU default table; each entry should be re-measured
before becoming a default. Raw benchmark JSON and dispatch logs are kept
outside the repository.
