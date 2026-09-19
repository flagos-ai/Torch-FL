# Fused SDPA on GCU: codegen limitation and kernel design

Status: **implemented and measured.** CLAUDE.md allows a hand-written vendor
kernel only when the platform code generator cannot express the required runtime
behaviour, and requires the concrete limitation to be documented and approved
before the implementation is added. This file is that document, and the approval
was obtained before any code was written. The kernel is
`csrc/aten/backends/gcu/scaled_dot_product_attention.cc`; the route record —
counts, artifact hashes, generator idempotency and the S60 measurement — is in
`docs/reference/operator-support.md` under the 2026-09-20 GCU entry.

The approval requested is narrow. Section 3 documents two things the generator
cannot express: a `REGISTER_PRIVATEUSE1_DISPATCH` registrar into an ATen
`DispatchStub`, and a kernel body returning a 4-tuple. Everything else — the
`m.impl` registration, the dispatcher entry and the `= gcu` route — is produced by
adding one line to `HANDWRITTEN_OPS`. The exception is therefore **one
hand-written translation unit plus that one line**, not a departure from the
codegen route for the op set as a whole. The measured justification is section 1:
4.397 s of an 11.637 s step (37.8 %), on the leaf's real call, 60/60 calls eligible.
Section 6 states what it does not fix.

## 1. Why this op matters

Qwen-Image-2512's transformer issues **60** `F.scaled_dot_product_attention`
calls per forward pass. On an S60 running the torch_fl leg they are all served by
the math decomposition, and that decomposition is the single largest cost in the
step:

| path | ms per call | evidence |
| --- | --- | --- |
| math decomposition (shipped) | 90.4 | 45.7 % of an 11.97 s step; `_safe_softmax` 47.5, `isneginf` 27.1, `where` 7.77, `bmm` 7.24, `_softmax` 5.87, `all` 5.57 |
| `topsatenScaledDotProductFlashAttention` | **4.017** | vendor op on the model's exact non-contiguous view; `sentinel-left 0/12638208`, `6.083e-05` vs an fp64 reference |
| FlagGems SDPA path | 133.84 | slower than the math path (90.54 in the same run) |

### Measured end to end

Op-level numbers do not settle whether the change is worth a hand-written kernel, so
the whole transformer forward was run three ways in one process on one set of
inputs, with the leaf either delegating to `_scaled_dot_product_attention_math` or
calling the vendor op (`probe_vendor_sdpa_step`, cards `flagos:0` +
`flagos:3,flagos:6`):

| leaf mode | step | leaf calls | attention time | declined |
| --- | --- | --- | --- | --- |
| math (shipped) | 11.637 s | 60/60 | 5442.9 ms -> 90.71 ms/call | 0 |
| vendor, drain | **7.240 s** | 60/60 | 331.3 ms -> 5.52 ms/call | 0 |
| vendor, no drain | 7.356 s | 60/60 | 83.1 ms -> 1.39 ms/call | 0 |

`4.397 s` of a `11.637 s` step, **37.8 %**, with `sentinel-left 0` of 12 638 208
elements in every run and byte-identical outputs between the two vendor runs
(`max|d| 3.125e-02`, `mean|d| 2.743e-03`, against `|math|max 5.750e+00`, i.e. a mean
relative deviation of ~4.8e-04). The drain costs 248 ms/step (3.4 %); the probe's
per-call figure overstates the host cost, because it fills a 25 MB output with a
sentinel before every call, which the kernel would not do.

The FlagGems carve-out in the originating request does not apply: FlagGems'
SDPA is *slower* here, not faster.

**The paired end-to-end run.** The probe above is one process; the shipped change
was then measured as two builds of one tree over a whole 8-step 1024x1024
`--stage full` generation, seed 42, `QWEN_IMAGE_REAL_ROPE=1` on every leg and the
same latents and prompt embeds carried between the legs, against
`torch_gcu + diffusers` as the reference:

| leg | leaf | `s/it` at 8 steps | vs `torch_gcu` |
| --- | --- | --- | --- |
| before | ATen math decomposition | **19.33** | 5.07x |
| after | the vendor flash op | **8.48** (8.81 on a repeat) | **2.23x** |
| reference | `torch_gcu + diffusers` | **3.81** | 1.00x |

**2.28x and 56.1 % off the denoising loop.** The repeat leg reproduced its image
byte for byte (`md5 d47974b1…`), so the leg is deterministic and the faster
reading is not a lucky one. The loop delta is larger than the 37.8 % the probe
projects, for a reason the probe cannot see: it priced 60 leaf calls of one
forward shape, while the math decomposition also materialises intermediates
(`_safe_softmax`'s `where`, `isneginf`, `all`) that the vendor leaf never builds,
and those intermediates are inside the step but outside the leaf's self time.

The images move the right way: the shipped leg against the math leg is
`MAE 3.568/255` (`PSNR 30.10 dB`), and against `torch_gcu` the shipped leg is
`5.089/255` (**26.96 dB**) while the math leg is `5.102/255` (26.91 dB). The
kernel is therefore closer to the reference than the path it replaced; the ~27 dB
that remains is the rest of the vendor's graph, not this op.

The route cannot be toggled at runtime, which is why the "before" leg is a second
build. `FLAGOS_OP__scaled_dot_product_efficient_attention=none` is refused
(*"routed to 'none' … but the op is registered on PrivateUse1 -- regenerate the
vendor conf so registration and routing agree"*) and
`torch.nn.attention.sdpa_kernel([SDPBackend.MATH])` is a silent no-op: that leg
came back byte-identical to the shipped leg and still logged 960
`_scaled_dot_product_efficient_attention -> gcu` dispatches, because
`__torch_function__` picks the route before ATen's backend pin is consulted.

"CPU fallback" is not the problem and is not what this fixes — the math
decomposition runs on device through flagos ops. The problem is that it is 22x
the vendor kernel's cost.

## 2. The exact call the model makes

Instrumented over one real transformer step (`probe_attn_sig_step`, 53-line log):

```
query = (1, 24, 4114, 128) bfloat16  stride (12638208, 128, 3072, 1)  NONCONTIG
key   = same shape, same strides
value = same shape, same strides
attn_mask=None  dropout_p=0.0  is_causal=False  scale=None  enable_gqa=False
```

That stride vector is a dense `(1, 4114, 24, 128)` buffer read as `(B, H, S, D)`
— i.e. the transpose of a contiguous BHSD tensor, not the contiguous case. The
vendor op was measured on exactly this description and **honours the strides**
(no staging copy, no transpose: a `(1, 2)` transpose would produce a different,
contiguous tensor and is not needed).

## 3. What the generator can and cannot express

`scripts/codegen/codegen_gcu.py` is the required route. Its registry has three
relevant categories:

* **`OPS` categories** — every template ends in one `topsaten::` call and emits a
  body `at::Tensor {kernel}(...)` plus
  `REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})`.
* **`HANDWRITTEN_OPS`** — emits only `m.impl("<op>", Wrapper...)` into
  `gcu_register.inc` and a `= gcu` route into `backends_gcu.conf`, and expects a
  hand-written translation unit to carry the body (this is how `rng.cc` and
  `native_dropout` are done).
* **`METADATA_OPS`** — same, for view ops whose bodies live in
  `csrc/aten/strided_ops.cc`.

`_scaled_dot_product_efficient_attention` **is already a generated op**: it has a
dispatcher (`generated/ops.h:1335`, `generated/ops.cc:449`), a wrapper
(`generated/register.inc:1331`, `:6657`) and an entry in the generator's `OPS`
map. So the registration and the conf route are expressible. Two things are not:

1. **The `_fused_sdp_choice` DispatchStub slot.** PyTorch's SDPA asks
   `_fused_sdp_choice_stub` — via the stub's `PrivateUse1` slot, filled by
   `REGISTER_PRIVATEUSE1_DISPATCH` — both whether the device is supported and
   which fused backend to use. Every generator template emits
   `REGISTER_IMPL_TO_DISPATCHER(..., Backend::kGcu, ...)` into the flagos
   dispatcher table; none emits a `REGISTER_PRIVATEUSE1_DISPATCH` registrar into
   an ATen DispatchStub.
2. **A body returning a 4-tuple.** The leaf returns
   `(output, log_sumexp, philox_seed, philox_offset)`. Every generator template
   writes one out-parameter `at::Tensor`.

Measured, and the reason (1) is not an implementation detail but a hard
requirement — three runs of the same probe, differing only in whether the stub
slot is filled:

| stub slot | observed |
| --- | --- |
| filled, returns 2 | `_fused_sdp_choice REACHED -> 2`; leaf fired **1** time |
| filled, returns 0 | `_fused_sdp_choice REACHED -> 0`; leaf fired **0** times |
| empty | `aten::_fused_sdp_choice(...)` still returns 2 and both `PrivateUse1` kernels exist, yet the leaf fires **0** times |

The third row is the interesting one: `csrc/aten/register.cc:421` registers the
*ATen op* `_fused_sdp_choice` on PrivateUse1 unconditionally and returns
`efficient_attention`, and that is what makes a direct call answer 2 — but the
composite does not consult it. It consults the stub. With the stub slot empty,
`is_device_supported(PrivateUse1)` is false, the composite takes the math branch,
and a registered leaf is unreachable. This matches the comment already in
`csrc/aten/backends/ascend/scaled_dot_product_attention.cc:13-19`.

The second row is what makes the stub useful rather than merely necessary: its
return value *is* the backend selection, so it can decline calls the vendor
kernel cannot serve and leave them on today's correct math path, with no host
round trip and no CPU fallback.

## 4. What was built

**One new translation unit**, `csrc/aten/backends/gcu/scaled_dot_product_attention.cc`,
picked up automatically by the CMake glob (`csrc/CMakeLists.txt:138` excludes the
whole `backends/<vendor>/` tree only when the build is not for that vendor), plus
**one line** adding `_scaled_dot_product_efficient_attention` to `HANDWRITTEN_OPS`
in `codegen_gcu.py`, plus regenerated artifacts.

The file mirrors `csrc/aten/backends/ascend/scaled_dot_product_attention.cc` and
`csrc/aten/backends/gcu/rng.cc`:

1. A namespace-scope `REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, ...)`
   (the macro expands a static registrar, so it cannot live in an anonymous
   namespace) returning `efficient_attention` (2) when the vendor kernel can
   serve the call and `math` (0) otherwise.
2. `PrivScaledDotProductEfficientAttentionKernelGcu`, issuing one
   `topsatenScaledDotProductFlashAttention` (the "4.0" overload, the one that
   takes `topsatenPhiloxState_t`) through `EXEC_TOPSATEN_CMD` with
   `TopsatenTensorWrapper` for q/k/v/out, registered via
   `REGISTER_IMPL_TO_DISPATCHER(PrivScaledDotProductEfficientAttentionFn,
   priv_scaled_dot_product_efficient_attention_dispatcher, Backend::kGcu, ...)`.
   No backward kernel: `_scaled_dot_product_efficient_attention_backward` is
   already registered on PrivateUse1 by
   `generated/gcu_flaggems_register.inc:261` and routed to `flaggems`.

### Eligibility predicate (shared by the stub and the kernel)

Return `efficient_attention` only when **all** hold; every other call returns
`math` and gets today's behaviour:

* 4-D `q`/`k`/`v` (`[B, H, S, D]`), all bf16, all on `flagos` (bf16 is the only
  dtype measured with the vendor op);
* `k`/`v` agree in shape; `k.size(1) == q.size(1)` (no GQA repeat — GQA goes to
  the math path, which handles `enable_gqa` itself); `k.size(0) == q.size(0)`,
  `k.size(3) == q.size(3)`;
* a dense layout under *some* permutation of the shape's non-singleton dims
  (accepts both the model's BSHD-strided view and the plain contiguous case;
  rejects anything else, since only those two were measured);
* `attn_mask` absent, `dropout_p == 0.0`, `is_causal == false`;
* `at::GradMode::is_enabled() == false`.

`is_causal` is declined deliberately: the vendor op exposes the flag, but its
mask-origin convention for `q_len != kv_len` is unmeasured, and "unmeasured" is
not a basis for a kernel. Causal calls keep the math path. This costs the target
workload nothing (the model is not causal) and is the honest boundary.

The kernel additionally declines when `compute_log_sumexp` is true and delegates
to `at::_scaled_dot_product_attention_math`, which the dispatch-key census shows
is a pure composite with no kernel on `AutogradPrivateUse1`, `PrivateUse1`,
`Autograd` or `CPU` — so the delegation stays on device and returns exactly what
the composite's math branch returns today. Two things were measured before
settling that, rather than read off the header. First, the "4.0" overload does
fill its logsumexp output slot, so the value is available; it is simply never
validated against anything, and this is the only branch where the caller reads it
(`compute_log_sumexp` is set for the backward, which this kernel does not serve).
Second, and decisively, ATen's own convention for that slot is not a logsumexp at
all: called directly on the model's envelope, the math path puts the
`(B, H, S, S)` bf16 **attention matrix** in slot 2, not a `(B, H, S)` fp32
logsumexp. Delegating returns that value verbatim instead of inventing a third
answer. The `philox_seed` / `philox_offset` slots it does not produce are returned
as empty tensors; they are only consumed by dropout, which the predicate forbids.

### Grad scope

The census shows the leaf carries `Autograd=True`: ATen has a derivative formula
for it, so under grad a call would reach the *backward* leaf. On GCU that leaf is
FlagGems, while the forward here would be the vendor op — an unvalidated
forward/backward pairing. Gating the stub on `!at::GradMode::is_enabled()` keeps
training exactly as it is today (math forward, math graph) and confines the
change to inference. The diffusion loop runs under `torch.no_grad()`.

## 5. Routing and documentation consequences

* Adding `_scaled_dot_product_efficient_attention` to `HANDWRITTEN_OPS` puts an
  `m.impl` into `gcu_register.inc`, so
  `gen_vendor_confs.route()` stops returning `none` for it. The op is listed in
  `FLAGGEMS_PENDING_NATIVE_OPS`, which `build_all` subtracts from `py_here` for
  `gcu`, so it will not be captured by the FlagGems route and the generated conf
  line becomes `_scaled_dot_product_efficient_attention = gcu` (today `none` at
  `backends_gcu.conf:486`). The backward line (`flaggems`, `:487`) is unchanged,
  and `gcu_flaggems_register.inc` must not gain a forward entry or the vendor
  kernel would stop being reached.
* `docs/reference/operator-support.md` must be updated for GCU in the same
  change, with `tests/manual/flaggems_overload_survey.py` rerun on the affected
  hardware; anything not revalidated is marked as such.
* The generator must be run twice with an empty second diff, and the generator
  change committed with its artifacts.

## 6. What this change does not fix

The same transformer step profiled three ways, same inputs and same harness
(`prof_step.py` for the torch_gcu leg, `prof_vendor_step.py` for the two flagos
legs; profiler self-time per step in ms, `dev` column is zero because neither leg
reports device time):

| op | flagos, math leaf | flagos, vendor leaf | torch_gcu |
| --- | --- | --- | --- |
| **step** | 13.74 s | 7.99 s | 2.60 s |
| `topsStreamSynchronize` | 6529.8 (21086) | 1887.8 (15426) | — |
| `aten::_to_copy` | 2639.7 (3295) | 1886.0 (2995) | 3.1 (1304) |
| `topsMemcpy` | 1227.1 (1208) | 1317.8 (1208) | — |
| `aten::contiguous` | 1208.6 (3619) | 550.2 (1569) | — |
| attention | 2378 | 214.2 (60) | 2.4 (60) |
| `aten::addmm` | 410.9 (846) | 429.4 (846) | 34.0 (846) |
| `aten::mul` | 701.1 (1566) | 393.6 (1446) | 38.5 (1446) |
| `aten::mean` | 176.5 (241) | 181.4 (241) | 4.8 (241) |

The attention fix removes the attention term and, with it, the `fill_` / `where` /
`isneginf` / `all` ops `_safe_softmax` needs (`fill_` 781 -> 20 ms). What is left is
a host-side per-op cost that the torch_gcu leg does not have at all: `_to_copy`,
`topsMemcpy` and `topsStreamSynchronize` are ~0.6 ms, ~1.1 ms and ~0.12 ms per call
on the flagos leg against ~2 µs for the equivalent `GCU::_copy_from`. Two caveats
before those numbers are used: the profiler serialises and inflates runtime calls
(the unprofiled step is 7.24 s against 7.99 s profiled), and the flagos legs ran on
cards `0/3/6` while the torch_gcu leg ran on `0/1/2`, so the two columns are not a
controlled comparison. The within-process A/B in section 1 is the controlled one.

Conclusion: this kernel is a 37.8 % step reduction, **not parity**. Parity needs a
separate census of the host-side copy and sync paths above; none of it is attention.

With the kernel shipped the same harness reads **7.83 s** profiled and **7.50 s**
of summed self time over 64 ops, with the leaf at **170.813 ms / 60 calls** — the
projected rate, reached through the composed route. `topsStreamSynchronize`
(1873.2 / 13980 calls), `aten::_to_copy` (1822.7 / 2995) and `topsMemcpy` (1256.9
/ 1208) are then what remains between the two legs, and the census that attacks
them starts from a same-card, profiler-off pair rather than from the table above.

## 7. Numerical risk

The vendor FlashAttention accumulates in bf16; the math path upcasts to fp32 for
`_safe_softmax` and `bmm`. Output will therefore not be bit-identical to the
shipped leg. The measured deviation of the vendor op against an fp64 reference on
the model's own view was `6.083e-05` at the operands and `max|d| 1.555e-02`,
`mean|d| 1.005e-03` against `|ref|max 5.315e+00` on the shipped kernel's returned
output — the bf16 noise floor, not a defect. End-to-end image comparison against
the torch_gcu leg is the acceptance test, not op-level equality.
