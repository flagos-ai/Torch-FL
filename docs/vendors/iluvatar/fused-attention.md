# Iluvatar fused attention: the CoreX `ixAttnBkd` C ABI

This note records a measured bring-up of CoreX's fused attention kernel from a
`torch_fl` process on a BI-V150 (CoreX 4.5.0). It exists because the call fails
with an error that points at the driver but is caused entirely by the caller,
and that misdirection cost several days.

**Status:** the kernel is proven to run at vendor-parity speed from a
`torch_fl` process. The `scaled_dot_product_attention` route itself is not
wired yet; [What remains](#what-remains) lists the open items.

## Why this matters

Issue [#667](https://github.com/flagos-ai/Torch-FL/issues/667) measured a
31× gap between `torch_fl` and the vendor torch on a Qwen-family forward
(5933 ms vs 190 ms), with attention the dominant term. On a non-boxing,
FlagGems-primary platform the SDPA composite cannot reach a fused kernel at
all: `csrc/aten/sdp_choice_stub.cc` is the only route that op can take, and its
platform guard excludes `USE_ILUVATAR`, so the call decomposes into the
NxN-score math path.

CoreX ships a fused flash-attention C API. This note covers calling it with
device memory that belongs to `torch_fl`.

## The library

`/usr/local/corex/lib64/libixattnbkd.so.1` (36 KB) is a front door. It
`dlopen`s the kernel images on first use:

| library | role |
|---|---|
| `libixattnbkd.so.1` | front door; resolves `FwdKernelDispatch` and friends |
| `libixattnbkd_fwd.so.1` | forward kernels (~1 GB) |
| `libixattnbkd_bwd.so.1` | backward kernels |
| `libixattnbkd_acc.so.1` | accuracy-first variants |

The 1 GB image is mapped **lazily, during the first forward call** — not at
process start and not when the front door is loaded. A `/proc/self/maps`
sampler thread in the caller shows the front door's 5 maps before the call and
11 maps (both files) for roughly 115 ms around it.

Public header: `/usr/local/corex/include/ixattnbkd.h`.

## The ABI, exactly

`ixAttnBkdConfigInfo` is **104 bytes**. `ixAttnBkdPhiloxState` is **24 bytes**
in C++ — `Payload` is a union of `uint64_t` and `int64_t*`, so the struct is
`u64 seed`, `u64 offset`, `u32 offset_intragraph`, `bool captured`, 3 bytes of
tail padding:

| offset | field | type |
|---:|---|---|
| 0 | `stream` | `void*` |
| 8 | `softmax_scale` | `float` |
| 12 | `dropout_prob` | `float` |
| 16 | `philox_args.seed` | `u64` |
| 24 | `philox_args.offset` | `u64` |
| 32 | `philox_args.offset_intragraph` | `u32` |
| 36 | `philox_args.captured` | `bool` |
| 40 | `extragraph_offset` | `int64_t*` |
| 48 | `seed` | `int64_t*` |
| 56 | `is_causal` | `bool` |
| 60 | `causal_mode` | `u32` |
| 64 | `return_softmax_lse` | `bool` |
| 65 | `is_alibi` | `bool` |
| 68 | `alibi_mode` | `u32` |
| 72 | `alibi_slopeM` | `float*` |
| 80 | `inplace_update` | `bool` |
| 81 | `accuracy_first` | `bool` |
| 82 | `is_unpad` | `bool` |
| 84 | `batch` | `unsigned` |
| 88 | `max_seq_len_src` | `unsigned` |
| 92 | `max_seq_len_trg` | `unsigned` |
| 96 | `softcap` | `float` |
| 100 | `imp_mode` | enum (4 bytes) |

### The trap

A ctypes mirror that models `Payload` as `u64` plus a separate
`u32`/`bool`/`char[7]`, or that copies "112 bytes" from a captured vendor call,
is **8 bytes too large**. Everything after `philox_args` then shifts by +8, and
the library reads `batch` at its real offset +84 from the high half of
`alibi_slopeM`, i.e. **zero**. A zero batch is an invalid launch configuration,
and the failure surfaces as:

```text
CUDA Error in FlashAttnFwdF16Ixmma at
/home/corex/sw_home/sdk/ixattention-backend/src/impl/launch_flash_fwd_template.h:170
- invalid argument
```

followed by an abort inside a pybind11 destructor. Do not treat the trailing
abort as the bug: it is the library's error path unwinding through the
PyTorch extension.

The "112 bytes" seen when dumping a finished vendor call are the 104-byte
struct plus 8 bytes of the caller's stack past it. Never copy them.

## Descriptor and argument conventions

* `ixAttnBkdTensorDesc` is `{u32 n_dims; vector<size_t> dim; vector<size_t>
  stride; enum dataType;}`. `dim` and `stride` are `std::vector`s of
  `size_t`; a ctypes mirror must model `{begin, end, cap}` faithfully.
* Q/K/V/O descriptors use **BSHD order `[B, S, H, D]`** with **element
  strides**. For a contiguous torch tensor laid out `[B, H, S, D]`, pass
  `dim = [B, S, H, D]` and `stride = [stride(0), stride(2), stride(1),
  stride(3)]`.
* An absent mask is a descriptor with `n_dims = 0` **and `dataType` set**
  (the vendor uses bf16). A zero-initialised descriptor fails an earlier
  assertion.
* `lse` is 3-D contiguous fp32 `[B, H, S]`.
* `ixAttnBkdGetFlashAttnFwdBuffSize` returns 0 for the measured shapes, and the
  vendor passes `workSpace = NULL` for them.
* `dataType`: 0 half, 1 bfloat16, 2 float, 3 int32, 4 uint32.
* `imp_mode`: 0 perf, 1 memory, 2 atomic-add (backward), 3 backward-split.

## The working call

Captured from the vendor's own SDPA with an `LD_PRELOAD` shim that forwards the
call and logs its return code (rc = 0):

```text
softmax_scale        = 1/sqrt(head_dim)
is_causal            = 0
causal_mode          = 1
return_softmax_lse   = 1
batch                = B
max_seq_len_src/trg  = S
softcap              = 0
imp_mode             = 1
mask desc            = n_dims 0, dtype bf16
ws                   = NULL
lse                  = fp32 [B, H, S] contiguous
```

Every one of those fields must be written into a 104-byte struct whose
`philox_args` occupies bytes 16–39. With that, the call returns 0 from a
`torch_fl` process and the output matches an fp32 CPU reference.

## Measured results

BI-V150, CoreX 4.5.0, torch 2.10.0+cpu, bf16, contiguous inputs, no mask,
200 iterations after 5 warmups, `cudaDeviceSynchronize` around the loop. The
vendor column is vendor-torch `F.scaled_dot_product_attention` on the same
box for the same shapes.

| shape (B,H,S,D) | causal | CoreX fused | vendor SDPA | max abs diff vs CPU fp32 |
|---|---|---:|---:|---:|
| 1,16,1944,64 | no | **0.284 ms** | 0.292 ms | 2.6e-3 |
| 1,16,2048,64 | no | **0.285 ms** | 0.292 ms | 2.9e-3 |
| 1,32,4096,128 | no | **3.629 ms** | 3.63 ms | 1.5e-3 |
| 1,16,1944,64 | yes | **0.244 ms** | — | 2.4e-2 |

The fused kernel is at parity with the vendor's own attention. The causal
row's diff is larger and deserves a tighter reference check before it is
claimed as correct; the non-causal rows are within bf16 noise.

## Diagnosing a failure

Two non-invasive tools settled the investigation when `LD_PRELOAD`
instrumentation of the launch internals did not (it perturbs the vendor
process: under such a preload even `torch.zeros(2)` can fail the same way):

1. **Maps sampler.** A daemon thread in the calling Python process reads
   `/proc/self/maps` every ~5 ms and appends the matching lines. The C call
   releases the GIL, so the sampler keeps running and survives an abort.
2. **Return-code shim.** An `LD_PRELOAD` of `ixAttnBkdFlashAttnForward` that
   dumps the 104 config bytes, all six descriptors and the return code, then
   forwards through `dlsym`. This is what proved the vendor's call passes
   `ws = NULL`, `imp_mode = 1`, and rc 0 — and that our byte-identical-looking
   struct was not byte-identical.

## What remains

Before the route can be claimed in `backends_iluvatar.conf`:

* wire the kernel through the platform path (an Iluvatar arm of the SDPA
  composite override, or a generated kernel) and change the conf entry;
* verify fp16, GQA (`head_num_kv < head_num_q`), non-contiguous inputs, and
  every `head_dim` the dispatcher accepts (64/96/128/160/192/256 are the
  branches visible in `FwdKernelDispatch`);
* decide the mask policy: the fused kernel takes a mask descriptor, but only
  empty-mask calls have been measured; masked calls should keep the existing
  path until they are measured;
* confirm the causal result against an fp32 reference;
* keep backward out of scope until the forward route is stable.