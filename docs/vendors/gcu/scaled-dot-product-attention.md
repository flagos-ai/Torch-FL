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

The scope was extended once, for Qwen-Image-2.1, and section 4 records it: the
same kernel now also serves calls that carry a mask, through a predicate on the
mask's shape and innermost stride. That extension is what admits **98 of the 171**
`F.scaled_dot_product_attention` calls of a 2.1 forward — the 65 target-image
calls whose math score matrix is 2.0127 GiB each, which is the allocation the GCU
driver refuses when the leg is asked to fit on one card. Section 2 has the census.

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

The FlagGems carve-out in the originating request does not apply, and on 2.1 it
does not merely fail to help — it aborts. FlagGems' SDPA is *slower* here, not
faster (133.84 ms per call against the math path's 90.54 in the same run), and a
call that carries a mask takes the process down: `expand`ing the mask onto the
`(B, H, S, KV)` shape gives it a zero-valued stride on those axes, which
Triton-GCU rejects, and the leg dies with `exit 134` rather than falling back.
Supporting the FlagGems route on a masked call would need a FlagGems-side change
(either a constant-folded mask stride or an in-kernel mask path that does not
rely on the expanded view); `csrc/aten/sdp_choice_stub.cc`'s `FlagGemsEligible` /
`RouteMask` pair is where such a gate would go.

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

### The same op on Qwen-Image-2.1, where the calls carry a mask

2.1 is a different workload from 2512 and needed a second scope extension. Its
transformer issues **98** `F.scaled_dot_product_attention` calls per forward and
**every one of them carries a mask**, which the 2512-shaped predicate above
(``attn_mask`` absent) refused outright. Logged from one real forward
(`probe_q21_sdpa_wrap`, 171 calls in the process, of which 73 are outside the
transformer):

| group | n | q | k / v | mask | route |
| --- | --- | --- | --- | --- | --- |
| text encoder | 72 | `(1, 32, 40, 128)` | `(1, 8, 40, 128)` | none, `is_causal=True` | math (GQA and causal are declined by design) |
| target image | 65 | `(1, 32, 4096, 128)` | `(1, 32, 4122, 128)` | `(1, 1, 1, 4122)` bool, stride `(4122, 4122, 4122, 1)` | **fused** |
| prefix segment | 33 | `(1, 32, 26, 128)` | `(1, 32, 26, 128)` | `(1, 1, 26, 26)` bool, stride `(676, 676, 26, 1)` | **fused** |
| VAE decode | 1 | `(1, 1, 4096, 1152)` | same | none | math (layout, see section 4) |

`q`/`k`/`v` are `(1, S, H*D)` bf16 buffers read as `(B, H, S, D)` — the same
non-contiguous stride form as 2512, e.g. `(16883712, 128, 4096, 1)`, and the
vendor op honours it.

The 65 target-image calls are why this mattered enough to extend the predicate.
On the math path each one materialises an fp32 `(1, 32, 4096, 4122)` score matrix
— **2.0127 GiB per call**, 130.8 GiB of allocation churn over one forward — and
the driver refuses it:

```
itopsMalloc required :2164260864      (2.0156 GiB, the target-image score matrix)
```

so the leg could not complete a single denoise step on one card. Priced directly
at that shape (`probe_gcu_attn_target`, `flagos:0`, three calls per row, peak read
from `torch.flagos.memory_stats()["peak_allocated_bytes"]`):

| route | ms/call | peak | note |
| --- | --- | --- | --- |
| fused, no mask | 5.46 | 0.031 GiB | for comparison |
| fused, mask admitted | **13.81** | **0.031 GiB** | the lane this extension opens |
| math, mask declined | **147.14** | **4.785 GiB** | the control: same computation, fp32 mask |

**10.65x** per call and **4.754 GiB** off the peak. A masked call costs 2.5x the
unmasked one on the same op; the cause of that gap is not measured here and is
not needed for the route decision, since both are far below math.

The two mask spellings in that table differ only in dtype, and they are the same
allow-mask: the fused row passes a bf16 additive row of `0` / `-inf`, the math
row the fp32 equivalent. They are therefore a controlled pair — the predicate is
the only thing that sends them to different routes.

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

Qwen-Image-2.1's four attention groups are in section 1 rather than repeated
here. The two that reach the fused lane differ from 2512 in three ways, and each
of the three is load-bearing for the predicate in section 4:

* they carry an `attn_mask`, which 2512 does not;
* their `q_len` and `kv_len` can differ (`4096` against `4122` in the joint
  attention) — the window is the 4096 target-image tokens attending all 4122
  joint keys;
* the mask itself is a rank-4 bool `(1, 1, 1, 4122)` — a contiguous `(4122,)`
  viewed as `(1, 1, 1, 4122)`, stride `(4122, 4122, 4122, 1)` — and the
  per-segment mask of the prefix group is `(1, 1, 26, 26)`.

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

**Two registrations, two owners.** `WrapperFusedSdpChoice`
(`csrc/aten/register.cc:230`, registered at `:421`) answers `efficient_attention`
unconditionally, and it is the one the *dispatcher* holds — `torch._fused_sdp_choice`
reports `efficient_attention` for every spelling, mask or no mask, grad or no
grad, because that is this op. The composite does not read it. One experiment
separates the two (`probe_choice_owner`):

```
grad_enabled=False   choice(none)=efficient  choice(mask)=efficient  real-call=fused
grad_enabled=True    choice(none)=efficient  choice(mask)=efficient  real-call=math
```

The direct call answers `efficient` in both rows because `register.cc`'s op does;
the real call flips to `math` under grad, which is this file's stub taking its
`c10::GradMode::is_enabled()` branch. So the object that decides the route is the
`DispatchStub` in this translation unit, and anything that reads the dispatcher
op to predict this kernel's behaviour — including a `TorchDispatchMode` census,
which sees the stub's C++ calls and no `aten::_fused_sdp_choice` at all — will
get it wrong.

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
* `dropout_p == 0.0`, `is_causal == false`;
* `at::GradMode::is_enabled() == false`;
* if `attn_mask` is present, `SupportedVendorMask` — described next.

`is_causal` is declined deliberately: the vendor op exposes the flag, but its
mask-origin convention for `q_len != kv_len` is unmeasured, and "unmeasured" is
not a basis for a kernel. Causal calls keep the math path. This costs the target
workload nothing (the model is not causal) and is the honest boundary.

#### The mask clause

`attn_mask` was refused outright until Qwen-Image-2.1, whose every transformer
call carries one (section 1: 98 calls, two groups, both now admitted). The clause
is one uniform rule for both dtypes:

* bool, or additive in the query's own dtype (bf16);
* rank 1, 2 or 4, right-aligned on `(batch, q_head_num, q_seq_len, kv_seq_len)`
  per the header's own table, with every dim either 1 or the full extent;
* `mask.size(-1) == 1 || mask.stride(-1) == 1` — the innermost axis is either a
  broadcast or a genuine row.

Nothing else. There is no `DenseUnderSomePermutation` call on the mask: the mask
is never passed to the vendor op as a strided operand that would have to be
honoured, it is applied as an additive row, and the two measured spellings above
(`(1, 1, 1, 4122)` and `(1, 1, 26, 26)`, both innermost stride 1) are the ones the
model asks for.

**The same predicate answers two different questions about two different
tensors.** The stub is asked about the mask the *caller* wrote; the leaf is handed
the additive one the composite built from it, in the query's dtype, on the
caller's shape. That the stub is asked first is measured, not inferred from
reading ATen: two caller spellings pass the shape test and fail *only* on the
innermost stride, and in both cases the conversion would have replaced that
stride with 1 —

```
caller (4096, 4122) bool, `.t()`:                    innermost stride 4096  -> math
caller 1 x 1 x 64 x 80 bool, innermost axis expanded: innermost stride 0    -> math
```

An admitted conversion would have sent both to the leaf, so the stub must be
reading the caller's tensor. Hence the clause is dtype-symmetric: a bool-only rule
would admit at the stub what it refuses at the leaf, and the leaf's refusal is a
hard `RuntimeError`, not a fallback. That failure is not hypothetical — it is what
the first version of this predicate did, and the prefix-segment group
(`(1, 1, 26, 26)`) is the call that raised it.

The two sites can only disagree in the direction "stub admits, leaf refuses", and
the conversion cannot produce a spelling the leaf refuses: it always yields the
caller's shape right-aligned on `(B, H, q, kv)`, size-1-or-full, with an innermost
stride of 1 (`where` materialises, and the innermost axis is padded to a multiple
of eight and sliced back — a 78-wide key axis arrives at the leaf as an 80-wide
row). Verified on sixteen caller spellings (rank 1, 2 and 4; windowed, square,
strided, broadcast on either axis, 8-padded, additive bf16, additive fp32,
transposed, causal): every admitted one returns a correct result from the leaf,
every declined one a correct result from math, and none raises. The regression
test is `tests/integration/ops/test_gcu_sdpa_mask.py`, which asserts the *route*
rather than only the values — every decline here is a correct answer on the math
path, so a values-only test would keep passing on a build where the fused lane had
silently gone away.

Five things in the clause are there for a measured reason:

* **fp32 is refused even though the op accepts it.** An fp32 additive row returns
  `SUCCESS` — not an error — and reads the buffer as bf16 bytes, so the result
  matches no reading of the mask: against the two hypotheses it could be
  implementing, 1.1908 relative for "no mask" and 1.2072 for "this additive row",
  where a correct reading is `4e-3` (`probe_effattn`). It is unreachable through
  `F.scaled_dot_product_attention`, which converts a mask to the query's dtype
  before the leaf sees it, and this clause is what keeps it unreachable.
* **The innermost stride.** An additive bf16 mask reaches the leaf unconverted, so
  its strides are the caller's; a caller-level transpose puts the key count on the
  innermost axis and goes to math. The cost of the clause is a lost fusion, not a
  wrong answer.
* **The additive reading is measured, not assumed.** The masked fused calls agree
  with an fp32 CPU reference to `2.092e-03` relative at the target-image shape
  (`(1, 1, 1, 4122)` windowed) and `5.376e-03` on the prefix group, i.e. the op
  *adds* the row it is given, as ATen's float-mask semantics require, rather than
  reading it as a keep/drop boolean. Measured on the padded-plane spellings, whose
  rows differ from each other: additive `4.302e-03` and `4.134e-03` relative
  (`probe_effattn`), where reading the same tensor as a boolean mask is `4.4e-01`
  to `1.64e+00`.
* **The mask is applied on the key axis.** A rank-2 mask is `(q_seq, kv_seq)` by
  ATen's contract, so a `(64, 80)` mask right-aligns to `(1, 1, 64, 80)` and
  confirms `dim2 = q`, `dim3 = kv`. A rank-3 `(1, S, KV)` and a kv-mismatched
  `(1, 1, S, KV+1)` are declined by the composite to math before any backend is
  chosen, and an fp16 additive mask is rejected by ATen outright
  (`Expected attn_mask dtype to be bool or float or to match query dtype`).
* **A finite constant offset in the mask is not exact.** Softmax is
  shift-invariant, so a mask that adds the same constant to every key is a no-op
  in exact arithmetic and the host answers the maskless question for any constant.
  The op applies the additive row at reduced precision, so on the regression
  test's geometry (2 heads, 64 queries, 80 keys, bf16) the answer moves, and by an
  amount that grows with the constant: `max|d|` against an fp32 CPU reference is
  `1.6e-2` at `-8`, `2.2e-2` at `-16`, `3.5e-2` at `-30`, `1.29e-1` at `-100` and
  `1.33` at `-1000`. The two spellings the model actually produces are exact: an
  all-zero mask is bitwise identical to no mask, and `-inf` — what the composite
  builds from a bool mask — agrees to `0.000e+00`. The clause does not decline it:
  a threshold on the constant's magnitude would be arbitrary, a row that is
  uniformly shifted is a degenerate input, and `-inf` is the spelling for "block
  this key" rather than a finite large negative. It is recorded instead, and
  `tests/integration/ops/test_gcu_sdpa_mask.py` pins the deviation so that a
  vendor op which stops deviating fails the test and forces these numbers to be
  re-measured rather than silently outliving their measurement.

#### The VAE decode call is declined by the layout clause

2.1's VAE self-attention is one `(1, 1, 4096, 1152)` call per image, q/k/v being
three views of one `[1, 3C, H*W]` buffer with strides
`(14155776, 14155776, 3456, 1)`. It is declined — non-singleton dims are stride
`1` size `1152` and stride `3456` size `4096`, so the sorted strides are not the
dense `1, 1152` — and it is important that it is: the vendor op *takes* that shape
and then refuses it inside the kernel, which is an abort rather than a decline,

```
op_aten_sdp_efficient_attention.cc:318: add check args err: 3
status id : 3, status name : TOPSATEN_STATUS_NOT_SUPPORT
gcu_utils.cpp:74 : Check failed: 0            (exit 134)
```

and by then `_fused_sdp_choice` has already picked the backend, so nothing falls
back and the leg cannot decode at all. On the flagos route the layout clause keeps
the call on math, which is why that leg decodes as `bmm` + `_softmax` and
completes. The clause is doing this incidentally rather than by intent, so it is
recorded here: any future spelling of that call that *is* dense under the
predicate would reach an op that aborts on it, and the kernel-level re-check
cannot help because the refusal is inside the vendor kernel.

#### An inert mask is dropped before the vendor op is asked

The mask the composite builds from an all-allow caller mask is a buffer of zeros.
On 2.1's target-image call it reaches the leaf as a `(1, 32, 4096, 4122)` bf16 view
at strides `(4128, 0, 0, 1)` — **one stored row of 4122 values**, all of them zero,
expanded over 540M logical elements — and the vendor op has two entry points: the
mask-taking one it reaches today, and the maskless one
(`topsatenScaledDotProductFlashAttention`) the composite reaches when the caller
passes no mask at all. On the model's own operands the maskless entry point is the
cheaper one by a wide margin, measured in-model with a sync bracket per call:

| target-image call | ms/call |
| --- | --- |
| masked, with the composite's all-zero broadcast | 19.3 |
| maskless, `attn_bias=None` | **5.2** |

The 32 target-image calls of one forward came to **617.3 ms** verbatim; the same
32 calls maskless came to **166.3 ms**.

So the kernel asks the mask whether it is zero before handing it over.
`AdditiveMaskIsAllZero` narrows every axis of size > 1 that has stride 0 down to
index 0 — a view, not a copy, so what is read is the stored row — declines
outright when that changed nothing (a dense mask has no stride-0 axis to narrow
and is left alone), and then evaluates `(narrowed != 0).any()` through the routed
`ne` and `any` kernels. The check costs the flagos leg **0.52 ms** per call at the
target-image geometry and **0.88 ms** at the prefix geometry (0.19 / 0.63 on the
vendor leg, and ~11 ms on the first call while the modules load) — **~45 ms per
forward** against the 451.0 ms those 32 calls were costing, so it pays for itself
about ten times over.

Four bounds on the rule, all deliberate:

* **bf16 additive masks only.** Zero is the additive identity, so an all-zero
  additive mask is exactly the maskless call. It is *not* the identity for a bool
  mask, where `False` means "block", so a bool mask is never treated as inert —
  and what the gate reads is the converted tensor, which is bf16 by the time the
  leaf has it.
* **A dense mask is declined even when it is all zeros.** Narrowing is what makes
  the read cheap; with no stride-0 axis there is nothing to narrow, and reading
  the whole buffer is what the drop exists to avoid.
* **The innermost axis is never narrowed.** It is the axis the check has to read,
  and the only one whose stride is 1 on every admitted spelling.
* **Which keys are blocked does not change.** The mask is read, not discarded: a
  non-zero anywhere in the stored row keeps the masked call.

**The equivalence is measured, not argued.** Off-model, on a nine-geometry grid
spanning `q` from 1 to 4096 and `kv` from 26 to 4122 with the model's own padded
strides (`kv + 6` stored, sliced back), the masked and the maskless answers are
**bitwise equal on all nine — 0 geometries differ** — and each geometry carries a
control, one stored entry set to `-1000.0`, which differs from the maskless answer
in all nine. In-model, replaying every masked call of one real forward three ways
gives, against the result the shipped tree computes:

| replay | bitwise vs verbatim | max abs diff | whole-model output vs the shipped tree |
| --- | --- | --- | --- |
| maskless (`attn_bias=None`) | **32/32** | `0` | relative L2 `0.000000e+00`, `100.00 %` of elements within one bf16 ulp |
| narrowed broadcast | 0/32 | `1.562e-02` | relative L2 `7.102801e-03`, max abs diff `3.125e-01`, `26.25 %` within one ulp |

The second row is the rejected spelling, and it is rejected on this measurement:
handing the stored row in place of the broadcast is cheaper still (548.2 -> 343.0
ms on the leaf) and is bitwise exact at the prefix geometry (32/32), but at the
target-image geometry it is **0/32 bitwise, one bf16 ulp off**, and over the
transformer's 32 layers the model amplifies that ulp to `3.125e-01`. It is not in
the tree. Dropping the mask *is* bitwise exact at both geometries, on every one of
the nine off-model shapes, and at the whole-model output.

**What it is worth.** A census A/B of one tree, one op apart
(`/tmp/op-census-shipped.log` against `/tmp/op-census-maskless.log`), warm-up 1
then 2 timed forwards, `--stage transformer-step`, same prompt and seed, 1024x1024:

| leg | forward | exclusive total | 64 attention leaves | `mul.Tensor` |
| --- | --- | --- | --- | --- |
| before: every admitted mask goes to the vendor op | 1971.8 ms | 2281.9 ms | 551.4 ms | 545.2 ms (357) |
| after: an inert mask is dropped | **1550.9 ms** | 1946.7 ms | **200.8 ms** | 551.6 ms (357) |

**420.9 ms, 21.3 % of the forward**, and the dispatch counts are identical on both
legs (4753 dispatches / 9506 syncs), so the whole of it is the leaf. The bracket
measurement above credits 451.0 ms of that to the 32 target-image calls; it
carries the drain wait that this exclusive-time column excludes, and the two
numbers are not meant to sum.
The census forward's output is bitwise the plain forward's on both legs
(`max|d| 0.000e+00`). The rope-matched pair reads the same way —
`QWEN_IMAGE_REAL_ROPE=1`, 5521 dispatches / 11042 syncs on both legs: 1897.5 ->
**1468.4 ms** with the leaf at 548.2 -> **200.6 ms**.

The same A/B is what the two rejected routes are not: the broadcast-narrowing leg
reads 1720.2 ms (leaf 343.0), i.e. cheaper than the shipped tree and not the
answer the model computes.

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
(The 2.1 forward at the end of this section, measured after the maskless drop, is
the closest the two legs have come: **+254.3 ms**.)

With the kernel shipped the same harness reads **7.83 s** profiled and **7.50 s**
of summed self time over 64 ops, with the leaf at **170.813 ms / 60 calls** — the
projected rate, reached through the composed route. `topsStreamSynchronize`
(1873.2 / 13980 calls), `aten::_to_copy` (1822.7 / 2995) and `topsMemcpy` (1256.9
/ 1208) are then what remains between the two legs, and the census that attacks
them starts from a same-card, profiler-off pair rather than from the table above.

On 2.1 the mask extension leaves **73 of 171** calls on math — the 72
text-encoder calls and the 1 VAE decode call — and that is deliberate rather than
unfinished: the text-encoder calls are GQA (32 query heads to 8 key heads) *and*
`is_causal`, both refused for the reasons in section 4, and they are 40 tokens
long, so their fp32 score matrix is 0.0002 GiB and they are not where the time
goes. The target-image and prefix groups are the whole of the attention cost.

### The remaining gap on 2.1, measured

With both changes in the tree the two legs can finally be compared like for like:
same rope configuration (`QWEN_IMAGE_REAL_ROPE=1`, so neither leg dispatches
`view_as_complex`), same 64 masked attention calls, and a per-op census on each
(`/tmp/op-census-maskless-realrope.log` against `/tmp/op-census-vendor-narrow.log`,
5521 against 5132 dispatches and 11042 against 10264 syncs):

| | flagos | torch_gcu | delta |
| --- | --- | --- | --- |
| forward, steady mean | 1468.4 ms | 1214.1 ms | **+254.3** |
| exclusive total | 1934.0 ms | 1680.7 ms | +253.3 |
| the 64 attention leaves | **200.6 ms** | 407.6 ms | **-207.0** |

The fused kernel is now **207 ms cheaper on the flagos leg than on the vendor leg**
at the same 64 calls — the maskless entry point inverted that column. What is left
is a per-op premium on the ops both legs call the same number of times, and it is
concentrated in one row:

| op | flagos ms (calls) | torch_gcu ms (calls) | delta |
| --- | --- | --- | --- |
| `aten.mul.Tensor` | 403.7 (549) | 241.0 (549) | **+162.7** |
| `aten.add.Tensor` | 117.9 (259) | 67.3 (259) | +50.6 |
| `aten.index_put_.default` | 27.7 (6) | 3.1 (6) | +24.6 |
| `aten._to_copy.default` | 104.8 (264) | 89.7 (264) | +15.1 |
| `aten.mm.default` | 364.3 (232) | 350.4 (232) | +13.9 |
| `aten.native_layer_norm.default` | 32.1 (65) | 19.0 (65) | +13.1 |
| `aten.bitwise_and.Tensor` | 14.9 (32) | 4.1 (32) | +10.8 |
| `aten.unsqueeze.default` | 50.7 (580) | 41.0 (580) | +9.7 |
| `aten.clone.default` | 21.6 (130) | 14.7 (130) | +6.9 |
| `aten.silu.default` | 26.9 (35) | 20.5 (35) | +6.4 |
| `aten.mean.dim` | 23.3 (65) | 17.4 (65) | +5.9 |

Those rows, together with the rest down to `-4.4 ms`, sum to **+248.5 ms** of the
+253.3 ms exclusive-total delta. `mul.Tensor` alone is **65 % of that sum** (162.7
of 248.5 ms) at an identical call count, and it is the same op the complex64
real-view decomposition already took out of the rope path at a different dtype (the
2026-09-20 support row in `docs/reference/operator-support.md`): a complex64
multiply routed through `topsatenMul` was **3405.2 ms / 357 calls / 66.0 %** of one
census forward's exclusive self time and is **545.2 ms / 357 calls / 23.9 %**
afterwards — `-2860.0 ms` at identical call counts, against an exclusive total that
fell `-2879.8 ms`, so that one row is the whole of that change.

Beside it sits a shorter list of ops the flagos graph issues and the vendor graph
does not, and those rows are op-count differences rather than per-call premiums —
the two must not be added into the same column: `where.self` 125.4 ms over **194**
calls against 94.6 over 130, `scalar_tensor.default` 21.4 over **131** against 0.3
over 3, `constant_pad_nd` 14.1 over 64 against none, `expand` 5.8 over 65 against
none, `slice.Tensor` 42.5 over 485 against 30.6 over 421, `view.default` 55.3 over
616 against 45.9 over 615.

**Parity is not claimed.** The gap is 254.3 ms on the forward; the tables above are
its decomposition, not a plan, and the lever they point at is `mul.Tensor` at an
identical call count. Nothing in this section is a route change.

## 7. Numerical risk

The vendor FlashAttention accumulates in bf16; the math path upcasts to fp32 for
`_safe_softmax` and `bmm`. Output will therefore not be bit-identical to the
shipped leg. The measured deviation of the vendor op against an fp64 reference on
the model's own view was `6.083e-05` at the operands and `max|d| 1.555e-02`,
`mean|d| 1.005e-03` against `|ref|max 5.315e+00` on the shipped kernel's returned
output — the bf16 noise floor, not a defect. End-to-end image comparison against
the torch_gcu leg is the acceptance test, not op-level equality.

A masked call does not change that. Against an fp32 CPU reference the masked fused
calls agree to `2.092e-03` relative at the target-image shape and `5.376e-03` on
the prefix group — the same order as the mask-free cells (`3.650e-03`), so adding
the mask costs no accuracy. The one thing the mask clause *does* buy numerically
is that the fp32 spelling, which the op silently misreads, is kept on math
(section 4).

The mask's *values* are a separate matter, and one of them is not exact: a row
that is uniformly shifted by a finite constant is a no-op mathematically, and the
op moves the answer by an amount proportional to the shift (`1.6e-2` at `-8` to
`1.33` at `-1000` on the regression test's geometry). The values 2.1 produces —
zero, and `-inf` from the bool conversion — are both exact, which is the reason
the lane is admissible at all; section 4 records the measured table and why the
clause declines nothing on account of it.
