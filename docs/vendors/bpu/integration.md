# D-Robotics RDK BPU (Horizon BPU) backend

The BPU is the one platform in this repo where acceleration does **not** come
from operator kernels. Its BPU executes whole compiled graphs, so `torch_fl`
provides a real device (UCP-backed memory, device/stream layer) plus a
`torch.compile` backend, and every eager op runs on the CPU.

```bash
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
ACCELERATOR=bpu pip install --no-build-isolation -e .
```

```python
import torch, torch_fl

model = MyNet().eval()
compiled = torch.compile(model, backend="bpu")
out = compiled(torch.randn(1, 3, 224, 224))
```

## torch version

**Pinned to the 2.10 series** (`TORCH_PIN = "torch>=2.10,<2.11"` in `setup.py`,
mirrored in `pyproject.toml`'s build requires). This is the same pin every
platform in this repo carries, and it is enforced, not advisory.

The pin exists because the checked-in `csrc/aten/generated/*` bindings are
generated against one specific ATen surface. A newer torch drifts from them and
fails as a wall of compile errors at build time rather than a clean resolver
error, so the pin is what turns a confusing build break into an install-time
message. Moving to a newer torch is a deliberate act: re-run
`scripts/codegen/codegen_ops.py`, do not hand-edit the generated files.

The verified board environment uses `torch 2.10.0+cpu` with a Python 3.14
aarch64 interpreter (the cp314 aarch64 wheel exists on PyPI).

## Why not per-op kernels

`hbDNNInferV2` takes a model handle and a tensor array — a compiled `.hbm`
artifact, not a Conv2d argument list. There is no entry point that computes one
convolution, so there is nothing for a `PrivateUse1` kernel to call. This
mirrors `tsingmicro`: `csrc/aten/register.cc` skips the generated
`register.inc` under `USE_BPU`, and every aten op reaches `cpu_fallback`.

Convolution is the one exception, and not by choice. `aten::convolution`
dispatches `PrivateUse1` to `convolution_overrideable`, whose only other kernel
is a `CompositeExplicitAutograd` stub that raises `NotImplementedError`. The
boxed fallback cannot rescue it — moving the arguments to CPU and redispatching
the *same* op lands back on the stub — so `register.cc` registers two small
wrappers that cross to CPU and call `at::convolution` instead. Without them
`conv2d` on a flagos tensor raises rather than falling back.

## Compile pipeline

```
Dynamo -> AOTAutograd -> aten FX graph
       -> decompose           (decompose.py: rewrite ops before partitioning)
       -> partition           (torch_fl/accelerator/bpu/partition.py)
       -> freeze weights      (params/buffers become ONNX initializers)
       -> ONNX export         (compiler.py, decompose.py again)
       -> int8 Q/DQ insertion (qdq.py, calibrate.py)
       -> hbdk4               -> .hbm, cached by graph structure
       -> hbm_runtime         (runtime.py)
```

Three parts of this are load-bearing rather than optimizations:

**Quantization is a precondition.** hbdk4's `convert(advice=True)` says it
outright: `"lower to cpu. P.S. The type of hbir.conv's fin is f32, which should
be si8, si16 on bpu."` A float artifact compiles fine and then runs conv on the
CPU, so the BPU sits idle. Q/DQ insertion is what moves the MAC work onto the
device — measured 6.1 ms to 0.64 ms on a 2-conv net. Set
`FLAGOS_BPU_QUANTIZE=0` for bit-exact float artifacts with no speedup.

**Weight freezing is what makes offload a win at all.** AOTAutograd lifts every
parameter and buffer to a graph input, so a 2-conv block crosses the partition
boundary with 13 tensors instead of 1 — copied on every call, and emitted as
ONNX graph *inputs* rather than initializers, which also blocks the int8 fold.
Before freezing, BPU offload measured 3x *slower* than eager (25.7 ms vs
2.8 ms); after, 0.849 ms.

**The rewrites in `decompose.py` decide how much of a network is one graph.**
They run before partitioning, not just before export, because the ops they
target do two kinds of damage. A tuple-returning op (`max_pool2d_with_indices`,
`_native_batch_norm_legit_no_training`) cannot join a partition *and* poisons
the next one's boundary — a `getitem` whose producer's `meta['val']` is a tuple
made `_example_inputs_for` reject the partition outright. Untouched, ResNet-18
split into a 4-node stem and an 84-node body, the body was rejected, and the
whole network ran on the CPU at **31.4 ms — slower than eager**. With the
rewrite it is one 68-node partition at **3.24 ms**.

The others are export blockers. AOTAutograd emits `aten._softmax`, which has no
ONNX symbolic at all (`softmax.int` does), plus `_unsafe_view` and `t`. A
transformer block hit all three: nine partitions, none exportable. Rewritten, it
is one 55-node partition that exports clean.

| aten op | rewritten to | why |
| --- | --- | --- |
| `_native_batch_norm_legit_no_training` | `batch_norm` | tuple output; no ONNX symbolic |
| `max_pool2d_with_indices` | `max_pool2d` | tuple output; BPU emits no indices |
| `_softmax` | `softmax.int` | no ONNX symbolic |
| `_unsafe_view` | `view` | not in the supported set |
| `t` (2-D only) | `transpose(0, 1)` | not in the supported set |

Each rewrite is skipped when its extra output is genuinely consumed — a graph
that uses max-pool indices for `max_unpool` keeps the original node and the old
behaviour.

Partitions that cannot be compiled stay in the graph and run eagerly, so a
missing or failing hbdk4 costs performance and never correctness. Pass
`strict=True` to `bpu_backend` to raise instead.

## Compiling on the board

hbdk4 ships **x86_64-only** wheels (`hbdk4_compiler-4.7.5-cp310/cp311-manylinux_2_17_x86_64.whl`);
the aarch64 wheels are runtime-only. So compiling on the board means running an
x86_64 Python under an emulator — but it does work, on the **stock kernel**, with
no VM and no cross-compile host.

One command sets it up:

```bash
scripts/vendor/setup_bpu_hbdk4.sh --wheels /path/to/oe/wheels
export FLAGOS_BPU_X86_PYTHON=~/hbdk4-x86/python/bin/python3.11
export FLAGOS_BPU_X86_EMULATOR=~/hbdk4-x86/bin/box64
```

Verify with:

```bash
python -c "from torch_fl.accelerator.bpu.compiler import find_hbdk; print(find_hbdk())"
# -> x86-emul
```

The wheels come from the D-Robotics OE package (`oe-package-*.tgz` on
`ftp://oeftp@sdk.d-robotics.cc/`); the script needs `hbdk4_compiler-*cp311*` and
`hbdk4_march-*`.

### Why the setup is not just "apt install box64"

Four things have to line up. Each one fails in a way that looks unrelated to the
real cause, so they are worth naming.

**1. box64 must be built from source; the packaged one is too old.** Debian and
Ubuntu ship 0.2.6 (Jan 2024), which had its page size fixed at build time and
aborts immediately:

```
Error: PageSize configuration is wrong: configured with 4096, but got 65536
```

Current box64 reads the host page size at runtime (`box64_pagesize =
sysconf(_SC_PAGESIZE)`) and maps 4 KB-aligned x86 `PT_LOAD` segments onto 64 KB
pages itself. Verified with **v0.4.5**: the same x86_64 binary that 0.2.6 refuses
runs correctly. **No kernel rebuild is needed** — earlier revisions of this doc
described one, and it is unnecessary.

qemu-user has no equivalent fix. It still fails with `SIGBUS` on any `.so` with
4 KB-aligned segments, and `pip` segfaults under it. box64 is the working path.

**2. `libhbtl.so` needs an explicit `RTLD_GLOBAL` preload.** `_hbdk*.so` expects
`hbtl::Storage::createExternal` and `hbtl::getStrides` to be resolvable, but does
not list `libhbtl.so` in its own `DT_NEEDED` — it inherits them transitively
through `libHBDKPythonCAPI.so`, which box64 does not reproduce. Without the
preload:

```
Error: Symbol _ZN4hbtl7Storage14createExternalE... not found,
       cannot apply R_X86_64_JUMP_SLOT
ImportError: Cannot dlopen(".../_hbdk.cpython-311-x86_64-linux-gnu.so")
```

`compiler.py` handles this: `x86_env()` puts the `_mlir_libs` directory on
`BOX64_LD_LIBRARY_PATH` (box64 resolves guest libraries through its own search
path, so `RUNPATH=$ORIGIN` is not enough) and the compile driver does the
`ctypes.CDLL(..., RTLD_GLOBAL)` preload.

**3. numba must be *absent* and stubbed, not installed.** hbdk4's ONNX entry
point imports numba unconditionally, but real numba imports `llvmlite.binding`,
whose x86 LLVM JIT segfaults under box64. The crash is in llvmlite's JIT setup —
not in numba, and not in hbdk4.

Stubbing it out is safe because hbdk4 only ever *calls* numba for custom/numba
ops: `compile_numba()` returns the module untouched when `has_numba_op()` is
false, which is always true for a graph exported from ONNX standard operators.
The stubs raise if actually invoked, so a graph that genuinely needs numba fails
loudly rather than miscompiling.

**4. torch is stubbed for the same reason.** hbdk4's tracing module imports torch
at module scope but only uses it for `isinstance` checks and `torch.jit.trace` in
the custom-op path. A real x86_64 torch would work but costs several hundred MB
in the emulated environment for code that never runs. Note the stub is the
*emulated* interpreter's torch, entirely separate from the board's aarch64 torch
that torch_fl itself runs on.

The stubs live in `~/hbdk4-x86/stubs` and are appended to `PYTHONPATH`, never
prepended, so a real numba or torch installed in the guest would win.

### One tolerated failure

hbdk4's `compile()` ends by loading the artifact back through hbrt4 to validate
it, which claims BPU device memory. Under emulation that can fail:

```
hbrt4_py.Hbrt4PyError: ... Cannot malloc bpu memory with length 52496 bytes:
                           AllocError { len: 135168 }
```

This happens *after* a complete `.hbm` has been written. The compile driver
tolerates exactly this case — non-empty output file plus an exception from the
validation step — because the artifact is the deliverable and torch_fl then loads
it with the board's native aarch64 runtime, which is a stricter check.

### Caching

`find_hbdk()` tries a native import, then a CLI driver, then the emulator
(`FLAGOS_BPU_X86_EMULATOR` first, then `box64`, `qemu-x86_64-static`,
`qemu-x86_64`), caching the result. When none is reachable the backend logs a
warning and leaves every partition on the CPU.

Emulated compilation is slow, which is why `compile_partition` caches artifacts
under `~/.cache/torch_fl_bpu` keyed by graph structure, input signature, march,
and the activation-scale set. Float and int8 builds never share a cache entry.

## Calibration

Without calibration, activations use `FLAGOS_BPU_ACT_SCALE` (default 0.05,
covering roughly ±6.35 — wide enough for post-BN/ReLU activations). For better
accuracy, measure real ranges:

```python
from torch_fl.accelerator.bpu.calibrate import calibrate_onnx
scales = calibrate_onnx("partition.onnx", samples=[x1, x2, ...])
```

Scales key on **ONNX** tensor names, which have no stable relation to torch
module names, so calibration runs on the exported graph via onnxruntime rather
than on the eager module. `calibrate_module` collects per-module ranges when the
torch-side view is what you want.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLAGOS_BPU_X86_PYTHON` | unset | x86_64 python with `hbdk4-compiler`, run under box64 |
| `FLAGOS_BPU_X86_EMULATOR` | unset | path to a box64 binary; needed because the distro 0.2.6 is too old and a self-built one is not on `PATH` |
| `FLAGOS_BPU_X86_STUBS` | `<x86 python>/../../stubs` | numba/torch import stubs for the emulated interpreter |
| `FLAGOS_BPU_MLIR_LIBS` | unset | directory containing `libhbtl.so`; preloaded `RTLD_GLOBAL` by the emulated interpreter before `hbdk4` is imported |
| `FLAGOS_BPU_MARCH` | `nash-p` | BPU micro-architecture (`nash-p`=BPU, `nash-e`=S100, `nash-m`=S100P) |
| `FLAGOS_BPU_QUANTIZE` | `1` | int8 Q/DQ insertion; `0` compiles float (bit-exact, no BPU speedup) |
| `FLAGOS_BPU_ACT_SCALE` | `0.05` | fallback activation scale for uncalibrated tensors |
| `FLAGOS_BPU_CACHE` | `~/.cache/torch_fl_bpu` | `.hbm` artifact cache |

## Runtime notes

- **Device memory is host-mapped.** `hbUCPMallocCached` returns an
  `hbUCPSysMem` carrying both a host-writable `virAddr` and a `phyAddr`, and it
  works unprivileged. So `Memcpy` is a plain `memcpy` plus cache maintenance
  (invalidate before reading what the BPU wrote, clean after writing what it
  will read) — no bounce buffer, and a zero-copy inference path is possible
  because a tensor's storage can *be* the BPU's memory.
  `FlagosBPUPhysicalAddress()` exposes the virtual-to-physical mapping for
  building an `hbDNNTensor` without a copy.
- **Four BPU cores, one device.** `hb_bpu_core_open()` takes a core *mask* with
  a scheduling policy, and UCP memory is SoC-wide, so cores are a scheduling
  detail rather than separate devices. `GetDeviceCount` returns 1, and 0 when
  the driver is absent (`hb_bpu_core_num() == 0`).
- **Submission is synchronous** (`hbUCPSubmitTask` + `WaitTaskDone`), so there
  is no async copy engine: `MemcpyAsync` is a synchronous copy, streams are a
  single sentinel handle, and event timestamps are real `steady_clock` readings
  — which makes `EventElapsedTime` genuinely meaningful here.
- **Cached blocks leak at process exit, deliberately.** The caching allocator is
  a function-local static, so its destructor runs from `__run_exit_handlers`, by
  which point `libhbucp`'s own `FINI_ARRAY` teardown may have released the heap
  the UCP blocks live in — `hbUCPFree` then aborts with `double free or
  corruption (fasttop)`. `caching_device_allocator.cc` skips the release under
  `USE_BPU` (as it already does for tsingmicro). Calling
  `torch_fl._C._empty_cache()` before exit frees the same blocks cleanly, which
  is what confirms only the exit ordering is at fault. The kernel reclaims the
  carveout when the fd closes.
- **The `cuda` alias must not break FX or Dynamo.** `_alias_cuda_to_flagos`
  rebinds `torch.device` to a Python wrapper so hardcoded-`cuda` code lands on
  the accelerator that is present. Two torch registries compare against that
  attribute by *identity*, so the swap silently changed their behaviour:
  `torch.fx.graph.add_global` special-cases `obj != torch.device` to emit a
  device constant as the bare name `device(type='cpu')`, and with the attribute
  swapped it fell through to the qualified-name path for custom ops — the name
  was never added to the generated module's globals and running the graph raised
  `NameError: name 'device' is not defined`. `torch._dynamo.utils`'s
  `common_constant_types` is membership-tested with `type(obj) in ...`, so
  reading `tensor.device` while tracing asserted instead. Between them these
  took out every model that calls `torch.arange(..., device=...)`, which is how
  HF builds position ids — i.e. every transformer. Both registries are keyed on
  objects, so `_keep_device_identity_checks_working` corrects them in place at
  import. `tests/unit/bpu/test_device_alias.py` covers it.

## Measured performance

### ResNet-18 against D-Robotics' own artifact

The vendor ships `/opt/hobot/model/s600/basic/resnet18_224x224_nv12.hbm` and
drives it from `/app/pydev_demo/classification_sample/resnet18/resnet18.py`.
That is the honest upper bound for this board, so it is what
`benchmarks/bpu_resnet18_bench.py` measures against — eager CPU only proves the
offload happened.

| path | median | p10 | p90 | input |
| --- | ---: | ---: | ---: | --- |
| official `resnet18_224x224_nv12.hbm` | **1.075 ms** | 1.060 | 1.136 | 74 KB NV12 uint8 |
| eager CPU float32 | 26.095 ms | 23.952 | 62.456 | 588 KB f32 NCHW |
| ours, `torch.compile(backend="bpu")` | **1.356 ms** | 1.315 | 1.454 | 588 KB f32 NCHW |

**19.2x vs eager, 1.26x off the vendor artifact.** Accuracy against eager float:
cosine 0.990, top-1 agreement 5/5 over random inputs, relative error 0.14 — the
expected cost of int8 activations at the default scale.

The remaining 0.28 ms is not overhead we can remove by tuning: the official
artifact takes NV12 (two uint8 planes, 74 KB) and ours takes float32 NCHW
(588 KB), an 8x difference in input bandwidth, and the vendor quantized with
HMCT 2.6.5 against real calibration data where ours uses a fixed default scale.
Our own conversion layers are already negligible — measured `_to_device`
0.025 ms and `_from_device` 0.004 ms against a 2.1 ms artifact, with 0.29 ms of
Dynamo/graph dispatch on top.

Two fixes got ResNet-18 from *slower than eager* to this:

1. **31.4 ms -> 3.24 ms** — the `max_pool2d_with_indices` rewrite (see the
   compile pipeline section). Without it the network was two partitions, the
   84-node body was rejected outright, and everything ran on the CPU.
2. **2.63 ms -> 1.36 ms** — quantizing the pool's *input*. `convert(advice=True)`
   named it precisely: `"lower to cpu. P.S. The type of hbir.max_pool's fin is
   f32, which should be si8, si16, f16 on bpu."` The stem max-pool was the one
   op left on the CPU inside the artifact. After the fix hbdk4 reports **zero
   CPU fallbacks**: 21 `b30.conv2d`, 1 `b30.pool2d`, all on the BPU.

`convert(advice=True)` is the tool for this. It reports a backend per op and a
`fallback_reason` for anything it lowers to the CPU, and it runs in a couple of
minutes rather than the ~25 a full emulated compile takes.

### Qwen3-0.6B against the vendor `llm` demo

`benchmarks/bpu_qwen3_bench.py` drives D-Robotics' own two-graph Qwen3-0.6B
`.hbm` through `infer.py`. Same artifact, same four cores, same weights, so the
comparison isolates the runtime rather than the model:

| path | decode | prefill |
| --- | ---: | ---: |
| vendor `oellm_runtime` `llm` demo | 84.8 – 87.7 tok/s | 5626 – 7014 tok/s |
| ours, `infer.Package` | **82.1 tok/s** (12.18 ms/tok) | **6881 tok/s** |

Prefill is quoted per *chunk*: the graph always computes all 512 columns, so a
17-token prompt reads as 219 tok/s for the same 78 ms of work. The vendor
counts the chunk, and so does the benchmark.

Getting there took one path change and three undocumented details.

`hbm_runtime.run()` copies every input on every call. For a CNN that is a
588 KB input and it does not matter; for a decode step the KV cache *is* an
input, 336 MiB of it at 4096 context, and copying it per token costs more than
the inference — **68.4 ms/token** against the vendor's 11.4. `infer.py`
allocates each tensor once in UCP memory and passes pointers, which is what the
vendor's own `libxlm.so` does (its undefined symbols are `hbDNNInferV2`/`V3`
plus `hbUCPMallocCached`).

Then, in the order they mattered:

1. **`hbUCPReleaseTask` must not follow `hbUCPWaitTaskDone` immediately.**
   Releasing inline blocks for **28 ms** — more than twice the inference. The
   per-call breakdown made this unmissable: build task 0.05 ms, submit 3.36 ms,
   wait 10.73 ms, release 28.03 ms. Deferred by one step it costs 1.65 ms and
   the step drops 43.2 ms -> 11.3 ms. Task handles cannot be resubmitted, so
   `_ReleaseRing` builds one per step and releases it one step late. The ring
   must stay shallow: handles are a finite pool, and once it is dry
   `hbDNNInferV2` returns a *null handle* instead of failing, so the step
   appears to run at triple speed while computing nothing. `infer()` raises on
   a null handle for exactly this reason.
2. **`HB_DNN_USER_DEFINED_L2M_SIZES` must be set before the first inference.**
   Unset, an LLM artifact fails with `L2 memory not enough ... user-assigned l2
   memspace size: [0, 0, 0, 0]` and `hbUCPWaitTaskDone` returns -200003. The
   vendor's `run_llm.sh` exports `6:6:6:6`; on this board `8:8:8:8` also fails.
   `ensure_l2_config()` sets it, which is why `Package` must be constructed
   after it runs.
3. **The KV cache is a sliding window, not a buffer you append into.** This one
   is a correctness trap rather than a performance one, and it is the reason
   `KVWindow` exists — see below.

#### The KV cache layout

The vendor runtime never copies the cache: it moves the pointer. Traced through
`LD_PRELOAD` over `hbDNNInferV2`, `sysMem.virAddr` for `layer_i_cache_key`
advances by exactly one slot per decode step while the allocation stays put,
and `layer_i_new_key` is bound one full window ahead at
`cache_base + window * slot_bytes`. Each token's K/V is written once, by the
device, into the slot the next step reads as part of its window.

The consequence for the mask is the part that cannot be guessed. Inside the
graph the attended keys are `concat(window[span:], new_keys)` for a step of
`span` tokens, so mask column `c` refers to window slot `c + span`, and the
last `span` columns are the current step's own tokens. With `pos` tokens of
history and row `r`:

```
open columns = [window - span - pos, window - span + r + 1)
```

which reproduces the trace exactly: prefill row `r` at `pos` 0 opens
`[3584, 3584+r+1)`, and decode at `pos` 24 opens `[4071, 4096)` — width 25, not
24, because the token's own key is the last column. The mask is **additive**
(0 attends, -65504 blocks); a uniform mask is a softmax no-op, which is how you
can confirm the polarity in a single run.

Two things that look like details and are not: the window advances by the
**real token count**, not the padded chunk width (a 512-wide prefill of 17
tokens slides by 17 — sliding by 512 puts padding in the context and generates
loops), and the allocation needs `window + max_tokens + chunk` slots, because a
prefill pass writes `chunk` slots past the window. Undersizing it surfaces as
`hb_bpu_map failed` / `hbUCPSubmitTask rc=-400006`, not a bounds message.

Worth stating plainly, because it cost the most time: **every wrong layout runs
at full speed and returns confident, fluent, wrong text.** No error, no NaN,
sane logit magnitudes. Black-box probing of the mask is ambiguous — opening a
zero-key slot also perturbs the logits, so "did this column matter?" has no
clean answer. Tracing the vendor took about twenty minutes and settled it.
`tests/unit/bpu/test_kv_window.py` pins the geometry to values transcribed from
that trace, so a regression fails loudly instead of producing plausible prose.

This path drives a prebuilt vendor artifact, not a `torch.compile` graph. For
what the compile path does on a stock transformer, see below.

### Stock transformers under `torch.compile`

A HuggingFace `Qwen3ForCausalLM` (2 layers, d=64) under
`torch.compile(backend="bpu")` runs on the board, compiled from the traced graph
rather than from a vendor artifact: **cosine 0.9910 against eager float**, with
the largest partitions offloading whole — 30/30 and 22/22 compute nodes, one
partition each. It is correct, not yet fast: each Dynamo region becomes its own
`.hbm` submission rather than one graph over a device-resident cache, so the
82 tok/s above still belongs to `infer.py` alone.

Three things had to be fixed to get here, and the order they were found in is
the useful part.

**The `cuda` alias was breaking FX codegen.** This looked exactly like a torch
2.10 bug and was not; see the entry under *Runtime notes*. Until it was fixed,
every HF model failed at `NameError: name 'device' is not defined` before any
BPU code ran.

**`pow`, `rsqrt`, `sqrt` and `neg` were missing from the supported set.** Those
four ops *are* RMSNorm, and without them a norm block split in two at 86%
coverage. `convert(advice=True)` shows hbdk4 lowering the lot to b30vpu with
zero CPU fallbacks, so the whitelist was wrong, not the hardware. `slice` joined
for the same reason on the attention side — it is how attention takes its causal
window.

**What actually caps coverage is Dynamo's symbolic shapes, not any missing op.**
This is worth measuring rather than assuming: the obvious suspect,
`aten._scaled_dot_product_flash_attention_for_cpu`, is a fused tuple-returning
node that keeps attention proper on the CPU — but forcing the math SDPA backend
decomposes it and moves whole-model coverage only 56% → 60%. Of the nodes
rejected on a 1-layer Qwen3, 21 are whitelisted ops carrying a symbolic dim
(`slice` x10, `view` x4, `unsqueeze` x3, `cat` x2, `mul` x2) and 5 are
int64/bool, against 8 that are genuinely unsupported (`embedding`, `arange`,
`scalar_tensor`, `lift_fresh_copy`). hbdk4 bakes memspace offsets and SRAM
tiling into the artifact, so a symbolic dim cannot be compiled at all and
`partition.py` rejects it deliberately. Tracing at a fixed sequence length is
what removes them.

One partition also falls back on an `expand` feeding a 5-D broadcast, which
fails hbdk4 shape inference (`'hbir.tile' op ... axis 0 is inferred to be 1, but
got 0`) — the constant-folding gap noted in the limitations. The model still
runs.

Compile time dominates: **~350 s for a 2-layer model**, because every partition
goes through hbdk4 under box64. Cached after the first run.

### Smaller graphs

A 6-layer conv stack at 224x224, quantized with frozen weights, compiled
**on the board** (torch 2.10, box64 v0.4.5): **3.75 ms on the BPU vs 72.06 ms
eager CPU — 19.2x.** An earlier measurement on a slightly different stack gave
4.00 ms vs 94.35 ms (23.6x) with cosine similarity 0.981; the ratio depends on
how much of the graph is convolution.

On a small 2-conv net the relative error against eager float is ~3%, which is the
expected cost of int8 activations — set `FLAGOS_BPU_QUANTIZE=0` for a bit-exact
float artifact, at the price of the conv work falling back to the CPU.

Toy networks are a wash (0.849 ms vs 0.805 ms) — the fixed submission cost
dominates, and there is not enough MAC work to amortize it. The offload is worth
it when the graph is genuinely convolution-heavy.

### Compile time

hbdk4 runs under box64, so compiling is slow: **~25 minutes** for ResNet-18's
68-node partition on this board. The `.hbm` is cached in
`~/.cache/torch_fl_bpu` keyed by graph structure, weight values, input
signature, march and Q/DQ pass version, so this is a first-run cost only.

## Known limitations

- Single device only; the four BPU cores are not scheduled independently.
- Synchronous execution; `hbDNNInferAsync` is unused.
- int8 only. hbdk4 also supports si16, which would trade throughput for accuracy.
- The zero-copy path is partial. `runtime.py` wraps a flagos tensor's UCP
  storage in a numpy array in place (`_device_view`), so there is no
  device-to-host copy on the way in — but quantization changes dtype (float32 in
  the graph, int8 in the artifact), and that conversion copies. Only a
  dtype-matched input is truly copy-free. Outputs always copy: `hbm_runtime.run`
  allocates its own arrays. Driving `hbDNNInferV2` directly with tensors built
  from `FlagosBPUPhysicalAddress()` would close the remaining gap.
- Calibration is opt-in and manual. On ResNet-18 the default scale costs
  cosine 0.990 against eager float, which preserved top-1 on every input tried
  — but the vendor's own artifact was calibrated with HMCT against real data,
  and part of the remaining gap to it is likely quantization quality rather
  than scheduling.
- Input format. The vendor's artifacts take NV12 uint8 (74 KB for 224x224);
  a graph traced from PyTorch carries float32 NCHW (588 KB). The BPU can
  consume NV12 directly, so a preprocessing path that fed it would remove
  8x of input bandwidth, but nothing in an FX graph expresses that.
- `qdq.py` quantizes convolution, matmul and pooling. Anything else — softmax,
  layer norm, elementwise — stays float, which is correct but leaves those ops
  on the BPU's VPU rather than its MAC array. `convert(advice=True)` is how to
  check what a given graph actually lowers to.
- **LLM inference runs a prebuilt vendor artifact, not a compiled graph.**
  `infer.py` is the runtime half only, and it is what the 82 tok/s above
  measures. A `torch.compile`d transformer is a separate path that works but is
  not yet fast — see *Stock transformers under `torch.compile`*. Closing the gap
  needs an ONNX constant-folding pass after export (torch's `aten.expand` emits
  a `Shape->Equal->Where->Expand` chain hbdk4 cannot shape-infer through) and
  mixed-precision Q/DQ — the vendor artifact uses si16 for the K cache, si8 for
  V, and f16 for the mask and logits, where `qdq.py` is si8-only. With folding
  alone, `convert(advice=True)` already puts 54 of 66 ops on the BPU; the 12
  fallbacks all report `"fallback to float, op is not completely quantized"`,
  which is the Q/DQ gap rather than a hardware limit.

