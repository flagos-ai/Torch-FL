# A fast eager wrapper for FlagGems operators on NVIDIA H100

Status: analysis + design proposal, measured on H100. Numbers are host-side edge
times in µs per operator call unless stated otherwise.

## 0. Summary

On NVIDIA H100 the FlagGems Python route costs **22–97 µs of host time per
operator call, depending on the operator** (≈22–25 µs for the leanest
single-kernel reductions, 65–95 µs for pointwise and tuned operators), against
**6–11 µs for the same ATen operator on a real CUDA
tensor** and **4.8 µs for a bare `cuLaunchKernel` with pre-resolved arguments**.
For the host-bound families the GPU side is not the problem: small elementwise
kernels measure ~0.4 µs on both routes, so the 70–96 µs elementwise gap is
entirely host dispatch. The gap is also not the plugin: ~95% of it is
**FlagGems' own Python wrapper plus Triton's Python launch path**; the plugin's
pybind bridge costs 2.8–5.1 µs and the flagos allocator/stream/device plumbing
measures ≈0. Kernel quality is a separate axis — large matmul is where FlagGems
loses badly (`mm[4096²]`: 2211 µs vs cuBLAS 369 µs, 6×) — and is handled by
routing, not by the wrapper.

Three levers can close the gap, in increasing order of effort:

1. **C++ eager route** (`flaggems_cpp`, already in torch-fl): a C++ trampoline —
   **hand-written today** for 18 operators in
   `csrc/aten/flaggems_cpp_kernels.cc`; the plan generates it — calls FlagGems'
   C++ operator library (`liboperators.so` +
   `libtriton_jit`), which launches Triton kernels with no Python in the steady
   state. Historically this was the shortest path off the Python tax — on A100 it
   reached **11 µs vs 10 µs vendor** for the reductions it covers — but see §7:
   on the current H100 cohort route C is **slower than both the shim and today's
   Python wrapper**, so the A100 figure is historical context, not current
   guidance.
   On H100, with the C++ runtime built from source, it recovers **~40% of the
   Python host cost** where it engages (`_softmax` 55→32 µs, `mm[512²]`
   130→74 µs) but still sits **4–6× above the true vendor floor** (8–12 µs),
   with 0.36–3.4 s cold-start JIT per operator and a large-matmul config loss.
   Its limits are structural: runtime compile through an embedded CPython,
   per-launch signature strings and argument marshalling, no autotuning, and an
   operator surface that excludes the pointwise family.

2. **AOT static launcher** (the production target): compile the FlagGems Triton
   kernels for the target architecture at wheel build time, ship cubins plus a
   launch manifest, and generate a typed C++ trampoline per operator that selects
   a config from an **offline-tuned table** and issues one `cuLaunchKernel`. No
   Python, no GIL, no JIT, no per-call strings, no boxing. Budget: 4.8 µs launch
   floor + 1–2 µs marshalling (target: at or below the vendor kernel; the
   marshalling side must replicate the `StridedBuffer` argument layout, §6.4).

3. **Pure-Python fast-launch shim** (the recommended first step, §7 Phase 0.5):
   keep FlagGems' compiled Triton kernel but replace its launch path — cache the
   captured `CudaLauncher` plus a static argument layout, patch only the tensor
   positions per call, add an equal-shape fast path, and hoist the dtype and
   contiguity checks into a memo. Measured below; it needs no C++, no AOT bundle
   and no new dependency, and it lives behind the existing `flaggems` key.

Recommendation: **start with the cheapest measured lever**, which turned out to
be (3), not (1) or (2). A late experiment prototyped a **pure-Python fast-launch
shim** (cache the compiled-kernel launcher and a static argument layout, patch
only the tensor positions per call, skip `LibEntry.run`, replace
`torch.broadcast_shapes` with a per-shape memo). Independently validated on a
second wheel in one process with all arms exact against the CPU reference
(`max_err = 0.0`, flagos, 512²-class, submit µs):

| case | vendor | FlagGems stdlib | shim, reused output | shim, fresh output per call |
|---|---:|---:|---:|---:|
| `add` fp32 | 8.7 | 87.4 | 10.9 | **15.4** |
| `mul` fp32 | 8.5 | 75.3 | **7.7** | **12.1** |
| `relu` fp32 | 9.0 | 73.8 | 9.4 | **13.8** |
| `add` + `(512×1)` (broadcast) | 8.8 | 110.9 | 11.3 | 15.9 |
| `(512×1) × (1×512)` | 8.9 | 88.0 | 8.0 | 12.2 |

The FlagGems Python tax collapses from 74–111 µs to **12–16 µs** at the call
level (5.4–8.0×), against a vendor floor of 8.5–9.0 µs — with **no C++, no AOT,
no `liboperators.so`, no build change**. Note what that is and is not: measured
against the vendor arm on the same wheel it is **1.4–1.8× vendor**, not parity
(`mul` 1.42×, `relu` 1.5–1.8×, `add` 1.76×). The plan should quote the
**fresh-output column** (12–16 µs): the reused-output column (7.7–11.3 µs, where
`mul` even beats vendor) changes the semantics by handing back a warmed buffer and
hides the 3.5–4.5 µs of allocation that an out-of-place op must pay. Reached
through the device it still pays the measured 3–7 µs plugin hop (≈15–22 µs, the
one number still to measure).

A same-wheel head-to-head against the C++ route settles the ordering (§7): on
`_softmax` the shim runs **10.5–13.6 µs against route C's 32.2–32.5 µs (2.8–3.1× faster)**,
and route C sits 3.7× off the vendor floor. The shim works for **both pointwise
and reductions**, at **~1.4–1.8× vendor for pointwise and ~1.05–1.45× for
reductions** (10.5–19.4 µs, fresh output per call — `sum.dim` at 1.06–1.19× is
vendor parity; full-reduce `sum` is 1.17–1.34×). The benefit claim is
**2.3–3.1× faster than the FlagGems wrapper for reductions** (27.4–38.7 → 10.5–12.5
µs) and **5.4–8.0× for pointwise** at the call level — a large win over the
wrapper, but not parity with the vendor kernel.

So the order is: ship the Python fast-launch path first (Phase 0.5 — **pointwise
and reductions**, smallest diff, no new dependencies), then treat C++/AOT as a
separate step whose justification has changed. **Phase 1 is no longer a
host-time win**: route C is slower than the shim on the same hardware, and the
A100 result that made it look 4.5× faster came from a cohort whose Python wrapper
was ~2× heavier than today's. Phase 1's case is exactly three properties Python
cannot provide: **GIL freedom under threading, CUDA-graph-capturable launches, and
AOT kernels (no per-process JIT/capture dependency)**. That is a throughput and
deployability argument, not a latency one.
Keep per-operator routing by measured end-to-end cost so compute-bound operators
whose FlagGems kernel trails the vendor (large `mm`, convolution, fused attention
at some shapes) use the vendor kernel.

## 1. How the numbers were measured

Testbed: NVIDIA H100 80GB HBM3, one GPU per run, idle. Software cohort: torch
2.10.0+cu130, FlagTree Triton 3.6.0, flag_gems 5.4.0, `USE_FLAGTUNE=0`.

Routing on the measured wheel is per-operator and fixed at import, so a
same-process A/B is done with `FLAGOS_OP_<op>=cuda|flaggems` (dots become double
underscores). The wheel used for the A/B is vendor-primary
(`torch_fl/configs/backends_cuda.conf`: 1618 `cuda`, 416 `flaggems`; the conf file
has 2081 assignment lines). Note that `backends_cuda.conf` is the **pre-rename
name that only exists on that installed wheel**; at HEAD the in-tree file is
`torch_fl/configs/backends_nvidia.conf` (renamed in #648), which has entirely
different contents — a reader will not find `backends_cuda.conf` in the tree.

The C++-route numbers in §4 come from a second build in the same environment: a
torch-fl wheel rebuilt from the same commit with `FLAGOS_BUILD_FLAGGEMS_CPP=1`,
`liboperators.so` and `libtriton_jit` built from FlagGems v5.4.0 source, and the
wheel's bundled CUDA assets disabled (`FLAGOS_DISABLE_CUDA_ASSETS=1`). That build
ships no boxing kernels, so its "vendor" column silently degrades to the Python
route; its true vendor floor was therefore measured with the same operators on
real CUDA tensors inside the same process. The two builds' absolute numbers are
not directly comparable to each other; each table states which wheel it used.

* **submit** — N calls back to back, one synchronization at the end. The async
  queue hides GPU time, so this is host dispatch cost.
* **e2e** — synchronize per call. Host dispatch plus kernel time.
* Two vendor baselines are quoted because they answer different questions:
  **ATen on real CUDA tensors** (6–11 µs) is the true vendor floor;
  **vendor kernel through flagos boxing** (9.5–12.7 µs) is what the same device
  costs when the vendor kernel is reached through the plugin's boxing route.

Caveat for anyone re-measuring: at HEAD the *default* NVIDIA conf routes
549 FlagGems / 1485 none / **0 boxing**, so a default HEAD wheel has no vendor
route to compare against. The vendor arm requires a boxing wheel plus
`FLAGOS_BACKEND_CONFIG=.../backends_nvidia_boxing.conf`.

## 2. Baseline on H100

Host cost per call (submit), 512², fp32, same process:

| op | `torch.<op>` on flagos (FlagGems Python route) | `flag_gems.<op>` called directly | `torch.<op>` on real CUDA (vendor) | gap |
|---|---:|---:|---:|---:|
| `add` | 77.4 | 73.2 | **6.4** | +71.0 |
| `mul` | 67.8 | 64.2 | **6.2** | +61.6 |
| `relu` | 66.0 | 61.2 | **6.5** | +59.5 |
| `sum` (full reduce) | 45.3 | 40.3 | **11.1** | +34.2 |
| `softmax(dim=-1)` | 24.8 | 22.0 | **6.0** | +18.8 |
| `mm` | 89.2 | 85.3 | **9.5** | +79.7 |

The same picture holds at 4096² and for bf16: FlagGems stays at 70–97 µs
regardless of size or dtype, i.e. **the cost is per call, not per element**. The
e2e consequences are direct: for small and medium shapes the host cost *is* the
e2e cost (`add` 512² 82.7 µs vs vendor 11.9; `softmax` 33.7 vs 14.8), while large
compute-bound shapes hide it (`mm` 4096²: submit 95 µs, e2e 2255 µs vs vendor
373 µs — there the kernel dominates and must be fixed by kernel routing, not by
the wrapper).

Wider operator matrix (host cost, submit; vendor arm = the same ATen operator
forced onto the vendor/boxing route, FlagGems arm = forced onto the FlagGems
Python route). Selected rows; the full 70-case table is in the measurement
artifacts. This matrix comes from an independent harness/second measurement
round; it agrees with the table above to within ~10% (e.g. `add` 9.3 vs 10.35 µs
vendor, 93.8 vs 94.7 µs FlagGems):

| op | vendor | FlagGems Python | ratio |
|---|---:|---:|---:|
| `add[512²]` | 9.3 | 93.8 | 10.1× |
| `where[512²]` | 11.0 | 110.5 | 10.1× |
| `gelu[512²]` | 8.9 | 79.0 | 8.9× |
| `cat[512²]` | 10.9 | 40.4 | 3.7× |
| `fill_/zero_[512²]` | 7.0–7.3 | 26.5–27.5 | 3.8× |
| `copy_[512²]` | 8.1 | 7.6 | **0.95×** (in-place, no wrapper cost) |
| `layernorm[1024²]` | 14.6 | 59.6 | 4.1× |
| `sum.dim[1024²]` | 10.9 | 34.1 | 3.1× |
| `softmax[1024²]` | 9.2 | 30.7 | 3.3× |
| `mm[512²]` | 12.9 | 107.6 | 8.4× |
| `bmm[8×512²]` | 17.7 | 68.5 | 3.9× |
| `addmm[1024²]` | 26.9 | 85.0 | 3.2× |
| `embedding` | 14.1 | 31.8 | 2.3× |
| `index_select` | 13.5 | 106.8 | 7.9× |
| `arange[1e6]` | 11.6 | 29.9 | 2.6× |
| `sdpa` (decode q4h8s1 d64 kv1024) | 25.9 | 25.2 | **0.97×** (no plugin dispatch logged) |
| `contiguous`, `clone`, `empty_like` | 0.1 / 47.8 / 3.2 | 0.1 / 47.6 / 3.3 | **1.0×** (not on the FlagGems route) |

The pattern is consistent: anything that runs FlagGems' Python wrapper pays a
roughly fixed 20–100 µs; anything that does not (in-place copies, no-dispatch
composites, ops whose configured route is the vendor kernel) is at parity.

### 2.1 Kernel quality is a separate axis

Host cost is one half; kernel quality is the other, and FlagGems is not uniformly
at parity. The defensible claims from the measured data:

* **Small elementwise kernels are at parity.** For `add[512²]` the kernel is
  ~0.4 µs on both routes (from the host/GPU decomposition); the 84 µs gap is
  entirely host. The same holds for `mul`/`relu`/`abs`/`neg`/`gelu`/`silu`.
* **Large matmul is not.** `mm[4096²]` fp32 measures **2211 µs of GPU time on
  the FlagGems kernel vs 369 µs on cuBLAS (6.0×)**, corroborated by an
  independent CUDA-event window (2164.8 vs 339.6 µs). It is not a cold-start
  artifact: the first call took 0.01 s because a warm tuned config came from the
  persistent tuner database. Even at zero host cost this operator would be ~6×
  slower, so it must be routed to the vendor kernel until the FlagGems kernel
  improves. The matmul family shows the same direction at smaller sizes
  (`bmm[8×512²]`, `addmm[1024²]`, `mm[512²]`).
* **Large reductions are mixed and shape-dependent.** At 4096² the submit-time
  ratio for `softmax` and `layernorm` favours FlagGems (0.76× and 0.92×), but
  e2e does not (1.05× and 1.22×), i.e. the result depends on the measurement
  point and the kernel/shape interaction. The routing decision for these has to
  be made per shape by measurement, not generalised from one table.
* **Some operators never reach the routing table.** 24 operators are registered
  as hand-written `m.impl`s that bypass the generated dispatcher entirely
  (`copy_`, `contiguous`, `clone`, `_to_copy`, `view`, `expand`,
  `empty.memory_format`, `scaled_dot_product_attention`, …): no conf key, no
  `FLAGOS_OP_*` override, no `[flagos dispatch]` line. Their measured parity
  says nothing about the wrapper, and they are also why some operators cannot
  be A/B-tested with per-op overrides.

So a fast wrapper is necessary but not sufficient: the matmul family needs
either a genuinely tuned FlagGems kernel (the Python route's autotune does not
close a 6× gap) or the vendor kernel, while elementwise and most reductions are
competitive once host cost is removed. This is why the design keeps a
per-operator, per-shape-class routing decision made by measurement, not a blanket
"FlagGems everywhere".

Two further measured facts matter for the routing design:

* **CUDA graph capture is a hard discriminator between the routes.** The
  vendor/boxing route captures and replays correctly (verified by capturing an
  in-place fill on a private buffer, clobbering it outside the graph, and
  replaying): 2.2 µs/op for `add`, 2.4 `gelu`, 4.2 `sum.dim`, 7.8 `mm[512²]`,
  8.0 `embedding` — matching a pure-CUDA capture. The **FlagGems Python route
  cannot be captured**: an in-place capture fails with
  `cudaErrorStreamCaptureInvalidated`, and out-of-place sequences appear to
  "succeed" with a 0.1–0.2 µs/op replay only because the Triton Python launches
  escape the graph (a false positive). The graph ceiling (~2–8 µs/op) is
  therefore reachable only by a native wrapper that launches the kernels on the
  capture stream — which is a requirement on the fast-wrapper design
  (§6.4, §6.5), not an independent optimisation.
* **Chain crossover**: a chain of N identical ops with a single sync stops being
  host-bound at N≈8 for elementwise, N≈16 for reductions, N≈32 for embedding,
  N≈2 for a 4096² matmul (already GPU-bound). The async queue hides host cost
  only once enough work is queued, which is why decode-style workloads (small N,
  many distinct operators) are the ones that suffer.

Two smaller measured effects are worth recording for follow-up: the plugin hop
costs about +7 µs for `add` (direct `flag_gems.add` 86.2 µs vs 93.8 µs through
the device), and in-place operators pay a larger plugin hop than out-of-place
ones (`zero_[4096²]`: 41.6 µs directly on flagos vs 90.5 µs e2e through the
device). Neither changes the design conclusion, but both point at the same
bridge code the fast path replaces. No `cpu_fallback` occurred for any measured
operator, and `copy_`/`contiguous`/`sdpa` never reach the plugin routing table
(they are composite or already-ATen paths), so they are out of scope for the
wrapper.

Floors measured on the same device:

| quantity | µs/call |
|---|---:|
| plain Python no-op call | 0.07 |
| `torch.empty_like` on flagos | 1.9–3.3 (same as CUDA: no allocator penalty) |
| `CudaLauncher.__call__` replay, pre-bound args (`cuLaunchKernel` floor) | **4.8** |
| `triton.JITFunction[grid](...)` replay, same kernel | 10.7 |
| vendor ATen kernel on real CUDA | **6.0–11.1** |
| vendor kernel through flagos boxing | 9.5–12.7 |

Cold start, same process: first `torch.add` on flagos **383.7 ms** (FlagGems
codegen writes a generated module into `~/.flaggems/code_cache/`, then Triton
compiles the kernel); the next `flag_gems.add` still 316 µs. First `mm` at a new
4096² shape **~2.5 s**, and the autotune inside that one call issued **1717
`cuLaunchKernel`s**; a full 20-config fp32 `mm` tune costs **~5 minutes of CPU
compile** in a fresh process. Tuned selections are cached in SQLite
(`~/.flaggems/config_cache/TunedConfig_<vendor>_triton_<ver>.db`).

## 3. Where the ~70 µs goes

Two wrapper generations exist in FlagGems 5.4 and they cost differently:

* **Generic `pointwise_dynamic` path** (`add`, `relu`, and many elementwise ops):
  a Python code generator plus a dispatcher, executed per call.
* **Hand-written specialised path with `@libentry` (+`@libtuner`)** (`mul`, `sum`,
  `softmax`, `mm`): a per-op Python wrapper around a Triton kernel, with the
  tuner adding a key/cache lookup per call.

Attribution for `add[512²]` fp32 on flagos, total 71.2 µs (sum of measured
layers, cross-checked with cProfile):

| layer | µs | share |
|---|---:|---:|
| `pointwise_dynamic.prepare_args` — schema validation, dtype promotion via `torch._prims_common` (pure-Python table walk, ~7 µs), output allocation, a `StridedBuffer` per tensor, kwargs rebuild | 15.1 | 21% |
| generated wrapper — shape/stride arithmetic, tile-size heuristics, `torch_device_fn.device(...)` context manager | 12.1 | 17% |
| `LibEntry.run` — parameter classification loop, specialization key (`data_ptr() % 16`, dtype, type tags), dict lookups, positional-arg reconstruction | 14.2 | 20% |
| Triton `CudaLauncher` → `cuLaunchKernel` | 4.8 | 7% |
| complex check, second capture check, output-buffer recording, unwrap | 2.2 | 3% |
| residual — CPython bytecode, kwargs rebuild, allocation churn, branch/cache effects | 22.8 | 32% |
| **total** | **71.2** | 100% |

Per-operator split of the gap against the true vendor floor (same ATen operator
on a real CUDA tensor, measured in the same process):

| component | add | mul | sum | softmax | mm |
|---|---:|---:|---:|---:|---:|
| FlagGems wrapper Python (prepare / generated wrapper / LibEntry / tuner) | ~62 | ~53 | ~27 | ~14 | ~73 |
| Triton Python launcher | 4.8 | 4.8 | 9.3 (2 kernels) | 4.7 | 9.7 |
| torch-fl pybind bridge | 4.2 | 3.6 | 5.1 | 2.8 | 3.9 |
| flagos allocation / stream / device shims | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |
| vendor reference | 6.4 | 6.2 | 11.1 | 6.0 | 9.5 |

The components are attribution estimates from the per-layer decomposition and
overlap in places (the launcher is inside the `LibEntry.run` measurement for
some operators); read the table as "where the time is", not as an exact sum.

Three consequences matter for the design:

* **The plugin is not the cost.** The whole plugin dispatch chain (PrivateUse1
  wrapper → `Dispatcher` → generated stub → pybind → GIL → callable lookup)
  measures **3–7 µs** (direct-vs-through-device delta: +3.2 µs for `mul`, +6.8 µs
  for `add`); allocation and stream shims are neutral versus CUDA. Caching the
  callable, dropping the per-call mutex/string lookup and the redundant output
  retag is worth ~1–4 µs — real hygiene, not a fix.
* **The FlagGems wrapper logic must be replaced, not tuned.** ~86% of the gap is
  Python work above the plugin: schema/dtype promotion in Python, per-tensor
  buffer wrappers, per-call specialization keys, tuner cache lookup, grid
  heuristics. No amount of plugin-side work removes it. Corollary: **on flagos
  there is no vendor-competitive FlagGems operator, and the floor is
  operator-dependent** — the leanest measured wrapper (`softmax`) still costs
  22–25 µs (≈3.7× vendor), and the pointwise/tuned operators cost 64–97 µs; the
  fast-looking `mul` row exists only for real CUDA tensors and does not survive
  on flagos. The design target is therefore "escape the FlagGems-on-flagos
  floor", not "copy whatever makes one operator look cheap". About **63% of the
  pointwise floor is removable without touching the kernels**: for a 63.7 µs
  `mul` call, Triton's Python launch machinery is ~50% (`LibEntry.run` 24.6 µs +
  launcher replay 5.2 µs out of the 32.2 µs launch helper),
  `torch.broadcast_shapes` is ~13% (pure `torch._refs` Python, no tensors), and
  allocation/checks are ~10%.
* **The Python launcher is a floor, not a ceiling.** Even a perfectly cached
  FlagGems wrapper pays Triton's Python launcher unless it reaches
  `CudaLauncher` directly (4.8 µs) — and that is already close to the whole
  vendor budget for the operator.

Two measured anomalies are *not* device problems, and are worth recording
because they mislead:

* `flag_gems.add` costs ~75–86 µs on **either** device, while `flag_gems.mul`
  costs ~10–14 µs on real CUDA tensors and 64–67 µs on flagos. The mechanism is
  now settled: both names resolve to plain generic Python functions
  (`flag_gems.mul is flag_gems.ops.mul.mul`; **no vendor rebinding**), but the
  specialised wrappers carry a runtime-device-name gate
  (`mul.py:34 _DEVICE_NAME = runtime_device.name` = `"flagos"` under torch-fl),
  so a real CUDA tensor bails out at `mul_broadcast_func:593` into
  `aten::mul.redispatch` (**8.85 µs** of the 10.4 µs total), i.e. native ATen.
  `add`'s generic `pointwise_dynamic` path has no such gate and pays its full
  Python cost on either device. Note also that the `SpecOpRegistrar` mechanism
  is real but narrow: on nvidia it rebinds only 16 Hopper names (`fill_*`, `mm`,
  `sqrt`, FP8 GEMMs, …) — and **binding to a vendor Python function does not buy
  vendor speed**: `flag_gems.mm` *is* vendor-bound to the Hopper implementation
  and still costs 85–95 µs.
* What **is** load-bearing: on flagos there is no cheap FlagGems operator, but
  the floor is **operator-dependent, not a single number** — `softmax` 22–25 µs
  (leanest: one kernel, four parameters; 30.7 on the boxing wheel used for the
  operator matrix — the two campaigns' absolutes differ ~15–20%, so compare
  same-wheel deltas only), `sum` 40–45 (two kernels), `mul`
  64–67, `relu` 66, `add` 72–78, `mm` 85–97, against a 7–11 µs vendor floor. Even
  the best case is ≈3.7× vendor. For `mul` the per-call decomposition is:
  launch helper 32.2 µs (of which `LibEntry.run` 24.6 and the launcher 5.2),
  `torch.broadcast_shapes` 8.1 µs (pure `torch._refs` Python, avoidable),
  `torch.empty` 3.2, contiguity/`is_complex` checks 3.0, residual ~15.6. The
  plugin hop is 3–7 µs, and the nested ATen calls the wrapper re-enters are only
  1–3 per call totalling ~2–3 µs (so nested re-entry is *not* a cost centre); the
  bulk is Python execution inside the wrapper frame, which is why the fix has to
  leave CPython entirely.

## 4. FlagGems' C++ wrapper, examined

Upstream's C++ route is the closest existing answer to "leave Python":

* `cpp/lib/*.cpp` (~10.7k lines) are generated C++ wrappers per operator, built
  into **`liboperators.so`** (~30 translation units) linking `Torch::Torch` and
  **`libtriton_jit`** (fetched from `resource.flagos.net` at build time).
  Pointwise operators get a C++ port of the codegen:
  `cpp/lib/pointwise_dynamic_cpp/prebuild_kernels.py` runs the Python generator
  at build time to materialise the Triton kernel sources and emit
  `pointwise_manifest.h` + `pointwise_runtime.h`; `add`/`div`/`fill` are behind
  `FLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP` (experimental, default OFF).
* **The runtime still JITs.** `TritonJITFunction::get_instance(<flag_gems .py
  path>, <kernel>)` embeds a CPython interpreter (`Py_InitializeEx(false)`) to
  read the kernel's static signature, and on a cache miss imports a Python
  helper (`standalone_compile.py`) and calls `triton.compile`. Compiled kernels
  are cached per (signature, device) in-process and on disk by Triton.
* **Steady state is GIL-free**, and per launch it does: `memcpy` each argument
  into a parameter buffer; `fmt::format` a signature fragment per tensor and
  constexpr; **join the full signature into one `std::string`**; cache-key
  lookup in an `unordered_map`; `cuCtxGetCurrent` + `cuCtxGetDevice`; a
  `std::mutex` in shared-memory lookup; a `LOG(INFO)` on the launch path; then
  `cuLaunchKernel`. That is ~3–8 µs of avoidable work per launch on top of the
  driver call.
* **No autotuning.** `mm` hardcodes `BLOCK_M=64, BLOCK_N=128, BLOCK_K=64,
  stages=2, warps=4`; the stream-k variant only fires on compute capability 8.x,
  so H100 always takes the fixed config. The Python route instead picks from 20
  H100-tuned configs (`hopper/tune_configs.yaml`, block up to 256, stages 3–5)
  and has a Hopper TMA/persistent variant. This is the measured A100 `mm`
  regression (C++ 110 µs vs Python 65 µs) and it is a *kernel-config* loss, not
  a dispatch loss. libtriton_jit lists autotuning as an unimplemented roadmap
  item.
* **Coverage is a curated subset**: `mm`, `bmm`, `addmm`, `sum`, `softmax`,
  `max`, `argmax`, `nonzero`, `sort`, `topk`, `embedding`, `rms_norm`,
  `fused_add_rms_norm`, `rotary_embedding`, `cat`, `contiguous`, `copy_`,
  `flash_attn_varlen`, FP8 matmuls, `zeros`, and (experimental) `add`/`div`/
  `fill`. A transformer's remaining pointwise surface (`mul`, `sub`, `where`,
  `gelu`, `silu`, `layernorm`, …) is **not** bound.
* **Build/deployment weight**: pybind11 + CMake ≥ 3.25 + C++20 + libtorch + a
  TritonJIT tarball from `resource.flagos.net`; at run time it still needs the
  FlagGems Python sources and `triton`. Upstream's Python-facing `c_operators`
  module also holds the GIL (no `gil_scoped_release`); linking `liboperators.so`
  directly, as torch-fl does, avoids that.

**Measured steady state, A100** (from `benchmarks/README.md`): `sum.dim[1024²]`
and `softmax[1024²]` go from 49/40 µs (Python) to **11 µs** (C++), against a
10/8 µs vendor floor — within 1–3 µs of vendor. This is the existence proof
that a native launcher around FlagGems kernels *can* reach parity.

**Measured steady state, H100** (this task: FlagGems v5.4.0 C++ extensions built
from source, `liboperators.so` + `libtriton_jit`, torch-fl wheel rebuilt with
`FLAGOS_BUILD_FLAGGEMS_CPP=1`; true vendor floor measured in the same process on
real CUDA tensors, because this build ships no boxing kernels and the
vendor-forced column silently degrades to the Python route):

| op | Python route (submit) | C++ route (submit) | true vendor floor | C++ vs floor |
|---|---:|---:|---:|---:|
| `_softmax[1024²]` | 55.3 | **32.1** | 8.4 | 3.8× |
| `mm[512²]` | 129.6 | **74.4** | 12.2 | 6.1× |
| `bmm[32×512²]` | 226.9 | 255.3 | 48.8 | 5.2× |
| `mm[4096²]` | 2164 | 4607 | 375 | 12× |

So on H100 the C++ route as shipped recovers **~40% of the Python host cost on
the operators where it engages** (softmax 55→32, mm512 130→74) but remains
**4–6× above the vendor floor**, and the matmul-family result gets worse at large
sizes (the non-autotuned kernel config costs 2.1× against the Python route's
tuned kernel, on top of the kernel being 5.8× slower than cuBLAS regardless).
Cold start of the C++ route, first call per op: `_softmax` 355 ms, `bmm` 2.05 s,
`mm[512²]` 3.42 s — the embedded-interpreter compile path is expensive.

The remaining C++-vs-floor gap is **not** Python: it is the device/boxing layer
in this build plus per-launch C++ work (signature strings, parameter marshalling,
context queries, mutex, logging) plus the Triton kernel itself. That is the
concrete argument for the AOT static design: C++-in-C++ is necessary but the
shipped runtime leaves 3–11 µs/call on the table and does not fix kernel quality.

**Verdict.** The direction is right and the mechanism works; the engineering is
not yet a production fast path: runtime compile through an embedded interpreter,
per-launch signature strings and marshalling, no autotune table, a curated
operator surface with the pointwise family least mature, and a build that pulls
an external runtime. It removes the Python tax but lands 3–8 µs above a
hand-tuned launcher and loses kernel quality where the Python tuner mattered.

## 5. What torch-fl already has

* **A dispatcher that is not the problem** (`csrc/aten/dispatcher.h`): backend
  resolved once per operator per process, no per-call environment reads, hashes
  or config I/O.
* **A real codegen pipeline** (`scripts/codegen/codegen_ops.py`): torchgen-
  faithful signatures for 2034 operators, category generators, and per-operator
  knowledge (FlagGems qualname, positional/keyword split, dtype-tagged kwargs,
  return arity, measured exceptions). This is exactly the input a fast-trampoline
  generator needs.
* **Boxing machinery** (`csrc/aten/device_boxing.h`) for vendor fallbacks —
  unnecessary for a launcher that takes raw pointers.
* **A measurement-driven routing table** (`torch_fl/configs/backends_*.conf`,
  generated by `scripts/codegen/gen_vendor_confs.py`).
* **Missing**: any AOT kernel compilation, any static launch metadata, any
  generated C++ bridge (the 18 `flaggems_cpp` operators are hand-written), any
  precompiled kernel bundle in the wheel.

## 6. Design: `flaggems_static`, an AOT launch path

### 6.1 Budget

| quantity | µs |
|---|---:|
| vendor ATen kernel on real CUDA | 6.0–11.1 |
| vendor kernel through flagos boxing | 9.5–12.7 |
| `cuLaunchKernel` with pre-resolved args | 4.8 |
| output allocation (`empty_like`) | ~2 |
| **target for the static wrapper** | **≤ 6–8 submit** |

The target is at parity with the native CUDA kernel and below the boxing route,
because the static wrapper skips both the second ATen dispatch and the boxing
retag, and its argument marshalling is a struct copy rather than Python work.

### 6.2 Shape of the solution

```
wheel build (per platform / arch)
  pinned FlagGems source
    → FlagGems' own codegen materialises the Triton kernel sources
    → triton.compile(target=sm90) for each (op, dtype, rank, config)
    → cubin + kernel metadata (args, constexprs, shared mem, warps/stages)
    → offline autotune table for H100 (shape bucket → config)
  → generated per-op C++ trampoline + manifest header
  → static bundle in the wheel

run time (per call)
  ATen dispatcher → PrivateUse1 wrapper → Dispatcher (backend = flaggems_static)
    → generated trampoline:
        read pointers / sizes / strides / dtype straight from at::Tensor
        pick the config from the static table by (dtype, layout class, shape bucket)
        patch a preallocated argument buffer, compute the grid arithmetically
        cuLaunchKernel on the flagos stream with a cached CUfunction
    → return the output tensor (allocator or caller-provided out=)
  no GIL, no Python frame, no FlagGems wrapper, no Triton launcher, no JIT,
  no boxing, no per-call strings, no locks, no logging
```

### 6.3 Build-time kernel bundle

* **Scope.** The operators the target models actually call, listed explicitly in
  a manifest: elementwise and unary, reductions, normalization, rotary,
  embedding, `cat`, `copy_`, small matmuls first; expanding by measurement.
* **Compilation.** Reuse FlagGems' own code generators to materialise Triton
  sources (pointwise operators are rank-specialised Python codegen; reductions
  and matmuls are static sources), then compile offline with
  `triton.compile(target=...)` for the target architecture, emitting cubin plus a
  JSON entry: kernel id, source hash, signature, constexpr values, shared memory,
  `num_warps`, `num_stages`, argument order.
* **Config selection.** `(op, dtype, shape-class) → config` tables produced by an
  offline sweep on H100. This is what removes the C++ route's `mm` regression and
  replaces runtime autotuning. For shape classes not in the table, the fallback
  is the runtime tuner (cold path) or the vendor kernel.
* **Reproducibility.** Pinned FlagGems/Triton versions, a content hash recorded
  in the wheel's build record, and an idempotent regeneration check. A mismatched
  bundle is a load-time error, never a silent wrong launch.
* **Size discipline.** Only compiled kernels are shipped; a fatbin section per
  arch. Operators whose kernel is not precompilable are excluded explicitly.

### 6.4 Generated trampoline

Extend `scripts/codegen/codegen_ops.py` with a generator that emits, per operator
and variant (functional / in-place / out / tuple / factory), a typed C++
trampoline. The generator already knows the argument list, which arguments are
tensors, the FlagGems kernel name per dtype family, the positional/keyword split,
the out-parameter shape and the return arity. Sketch:

```cpp
// generated, one per operator variant
at::Tensor AddTensorKernelStatic(const at::Tensor& self, const at::Tensor& other,
                                 const at::Scalar& alpha) {
  auto out = at::empty(self.sizes(), self.options());       // flagos allocator
  fls::LaunchAddTensor(self.const_data_ptr(), other.const_data_ptr(),
                       out.mutable_data_ptr(), self.numel(),
                       fls::DTypeOf(self), alpha.toDouble());
  return out;
}
```

`LaunchAddTensor` lives with the bundle: it selects the kernel for the dtype and
layout class, patches a preallocated argument buffer, computes the grid and calls
`cuLaunchKernel` on the current flagos stream. Out-variants pass the output in and
skip the allocation; in-place variants write into the operand. The trampoline
must not box: the kernels take raw pointers, so no `DeviceBoxingGuard`, no second
ATen dispatch and no output retag. The stream comes from torch-fl's own runtime,
not `c10::cuda`.

**Budget risk to plan for: the pointwise family does not take bare pointers.**
The generated kernels of the `pointwise_dynamic` family marshal `StridedBuffer`
wrappers: measured, `mul` launches with **16 kernel arguments and its tensors at
positions [9, 10, 11]**, while `add` launches with **24 arguments and
`StridedBuffer` objects at [9, 10, 12]**, with `alpha` passed as a static value.
A native trampoline must replicate that exact argument layout — shapes/strides
per operand, not just a pointer per tensor — which is the part of the 1–2 µs
marshalling budget that is easy to under-estimate. It is also why the Python
shim quotes its win as ~29 µs removable rather than the whole floor: it still
builds those argument objects, just far more cheaply.

**Capture safety is a requirement, not a later optimisation.** The vendor/boxing
route captures correctly in a CUDA graph (2–8 µs/op replay), while the FlagGems
Python route cannot be captured at all (out-of-place sequences only appeared to
work because the Triton Python launches escaped the graph). A fast wrapper that
launches on the current capture stream therefore makes graph capture available
for FlagGems kernels for the first time; a wrapper that calls into Python or
defers the launch keeps the graph path out of reach. The generated launcher must
query the stream once per call from `flagos::CurrentStream()` (which is
capture-aware) and must not synchronise, allocate, or take locks inside capture.

### 6.5 Fallback, routing and cold start

* New conf key `flaggems_static`, generated by `gen_vendor_confs.py` from the
  manifest plus the measurement table.
* Fallback chain per call: `flaggems_static` → `flaggems_cpp` (runtime-JIT C++)
  → `flaggems` (Python) → vendor kernel where the platform has one. Never
  `cpu_fallback` for an operator a target model calls; that stays a build-time
  check, like the existing dead-route tests.
* Cold start is module load (tens of ms) instead of the measured 383 ms
  first-call Triton compile and 2.5 s + 5 min autotune for a new `mm` shape.
* An explicit `prewarm(op, shapes)` API covers residual first-use work so it does
  not land in a latency-sensitive iteration.
* Multi-device: bind the operands' device index only when it differs from the
  current one, as the boxing guard does today.

### 6.6 Correctness and performance gates

Per operator, in CI on H100 (and per platform elsewhere):

1. **Numerics**: static vs FlagGems Python vs (where available) vendor, at the
   shapes/dtypes the manifest claims; tolerance per dtype.
2. **No dead route**: every `flaggems_static` conf entry has a manifest entry and
   a compiled kernel.
3. **No Python in the hot path**: the plugin's Python-op counter must not move for
   `flaggems_static` operators.
4. **Latency**: submit ≤ vendor + 2 µs and ≤ 8 µs absolute for the host-bound
   families; e2e must not regress versus the FlagGems Python route; the harness
   in `benchmarks/` is the instrument.
5. **Idempotent build**: regenerating the bundle and manifest produces an empty
   diff.

### 6.7 Alternatives considered

* **A faster C++ caller that still invokes the FlagGems Python callable** —
  rejected by measurement, and worth stating because it looks like the obvious
  shortcut. The plugin's current Python route already *is* a C++ caller: the
  bridge costs only 3–7 µs, and the remaining ~40 µs is inside the callable
  (Triton's Python launch machinery ~32 µs plus `torch.broadcast_shapes` ~8 µs).
  Making the caller faster recovers the plugin hop only; it inherits the wrapper
  and does not restore graph capturability. Only replacing the launch — with the
  precompiled kernel launched from C++ — recovers the tax.
* **Extend the hand-written C++ route only** — cheapest parity step (already
  measured at parity on A100), kept as Phase 0, rejected as the final
  architecture: runtime JIT, FlagGems-source dependency at run time, 3–8 µs of
  per-launch C++ waste, no autotune, and a curated operator surface.
* **Optimise the Python path** — the cheap hygiene items are worth ~1–4 µs of a
  60–90 µs gap, but one variant is no longer "hygiene": a **Python-side
  pre-bound launcher shim** that bypasses `LibEntry.run` and calls the compiled
  Triton launcher with a prebuilt argument tuple was **prototyped, measured and
  independently re-validated in one process** on a second wheel. Results
  (§0 table): 74–111 µs → **12–16 µs** with a fresh output per call, exact
  against the CPU reference; ~29 µs of the pointwise floor is removable in pure
  Python (`LibEntry.run` ~19 of 23.8, `broadcast_shapes` ~7.2, contiguity/
  `result_type` ~3), leaving the ~5 µs Triton C-launcher plus 3.5–4.5 µs
  allocation as the floor. Its limits are measured, not hypothetical:
  it is **per-op-family** (`mul`'s launcher takes `torch.Tensor`; `add`/`relu`
  take `StridedBuffer` wrappers that must be rebuilt per call, costing +2.2–3.3 µs);
  it is a **per-config JIT cache**, and reusing an equal-shape capture on
  broadcast inputs produced `max_err` 39.9, while a naive stride assumption on
  non-contiguous `add` produced `max_err` 8.93 — so a real
  `(op, dtype, shape, strides, contiguity)` cache **with a fallback to the
  standard path is mandatory**, not a nicety; it does not help reductions on a
  boxed route (`sum.dim` has no Triton launch to capture) nor `mm`'s config
  selection; and it stays in Python, so it keeps the GIL/pybind hop and does
  **not** make the route CUDA-graph-capturable. Cache-key cost is small and
  measured (0.35–0.78 µs per call; first-call miss 250–483 µs). This is the
  cheapest path to a large win, which is why it now precedes the C++ work;
  `liboperators.so` is an optimisation, not a prerequisite.
* **`torch.compile` only** — inductor fuses and removes per-op dispatch, and the
  torch-fl backend exists; it does not fix eager per-op latency, so it
  complements the wrapper.
* **CUDA graphs only** — removes per-op host cost for static-shape loops; a
  model-level tactic, not an eager wrapper.
* **Vendor kernels everywhere (boxing)** — matches vendor by definition but
  abandons FlagGems-primary portability and needs vendor `libtorch` assets.

## 7. Phased plan for H100

**Phase 0 — C++ route coverage and plugin hygiene (days).**
Extend `flaggems_cpp` from the hand-written 18 operators to the FlagGems C++
surface. This needs **new generator emission**: `scripts/codegen/codegen_ops.py`
today emits `ops.h`/`ops.cc`/`cuda_kernels.cc`/`flaggems_python_kernels.cc`/
`register.inc` only — neither the C++ bridge file
(`csrc/aten/flaggems_cpp_kernels.cc`) nor its `REGISTER_IMPL` lines come from a
generator, so the emission path itself is part of the work, not an extension of
an existing one. Also **enable FlagGems' own pointwise C++
codegen** (`FLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP=ON`, which exists in v5.4.0 but
is experimental/off by default and is why `add` has no C++ slot today). Ship the
measured plugin-side fixes (callable caching, `LogDispatch` `const char*`,
redundant retag, dtype-escape inlining, device scan). Note the H100 evidence
above: in the built-and-measured wheel only `mm`, `bmm` and `_softmax` actually
engaged the slot (`sum.dim_IntList`, `addmm`, `embedding` never reached the
dispatcher; elementwise has no slot), so "registered in C++" is not the same as
"routed" — Phase 0's exit criterion is route proof plus measured submit next to
the vendor floor. **Phase 0 alone does not reach parity** (measured 4–6× above
the floor, from the boxing/device layer, per-launch C++ overhead and
non-autotuned configs); it is the parity *step* and the harness for Phase 1.

**Phase 0.5 — Python fast-launch shim for pointwise and reductions (days).**
Promote the measured prototype to a real path, **covering pointwise operators and
reductions in both measured forms**: the single-launch forms (`_softmax` 10.5–11.8 µs,
`sum.dim_IntList` 10.8–11.7 µs on 1024² fp32, **1.06–1.40× the vendor floor**,
parity for `sum.dim`; correct across bf16 / non-square / non-last-dim / `dim=0`) and the **two-launch
full-reduce `sum`** (`ops/sum.py:76`: `contiguous()`, `mid`/`out`, two kernels)
measured on the **lean** replay path at **16.55 / 16.45 µs versus 40.45 / 40.26 µs
for the wrapper and 14.16 / 13.71 µs for the raw-CUDA floor** (1.17–1.20× on that
wheel; 1.17–1.29× across both wheels, see the note below) with a real `out` slot
and `max_err` bit-identical to the wrapper's (fp32 reduction-order rounding, not a
shim defect), and the **multi-output, single-launch** case (`max.dim`, values + indices)
replays correctly at **12.31 µs = 1.03× the floor** with both outputs exact —
while multi-pass tuple ops (`sort`, 21 launches) are kernel-bound and routed to
vendor instead (§8). Three rules came out of
this case and belong in the implementation: it is **2 launches with one distinct
intermediate pointer** (referred to from more than one argument slot; written by
launch 0 and read by launch 1); intermediates are
**persistent but need no initialisation** (zeroing every call costs **+29.7 µs**,
2.7× the whole shim, and changes nothing — launch 0 fully overwrites `mid`); and a
capture **must be keyed by `(shape, dtype, layout)` and a mismatched replay
refused** — replaying the 1024² capture on a 512² input measured `max_err` **31.38**
and, because the baked grid writes past the smaller buffer, **corrupted the flagos
allocator** so a later 1 MB allocation failed. Outside the measured shapes/dtypes, and for multi-kernel fusions
(`layernorm`-class), behaviour is unmeasured and must fall back to the standard
wrapper. Cache each
operator's compiled kernel launcher plus its static argument layout per
`(op, dtype pair, shape, strides, contiguity, out vs fresh)`,
patch only the tensor positions and scalars per call, replay the **captured
per-family allocation mode** (§8: `empty_like` for `pointwise_dynamic`,
contiguous `empty(shape)` for hand-written ops such as `mul`), and replace
`LibEntry.run` + `torch.broadcast_shapes` +
repeated contiguity/`result_type` checks with that memo. It **composes with the
existing `flaggems` route as an internal fast path rather than replacing it**, so
it needs no new conf key (Appendix C does not apply), no build switch and no
change to the routing policy; that is also what makes it deployable as an
intermediate state. Cache keys must cover everything that changes the launch —
`(op, dtype pair, shape, strides, contiguity, out vs fresh)` — and be
**versioned against FlagGems/Triton** so a kernel change invalidates them; every
miss falls back to today's wrapper. That is not paranoia: reusing an equal-shape
capture on broadcast inputs measured `max_err` 39.9, and a naive stride
assumption on non-contiguous `add` measured 8.93. It does **not** help `mm` or any
operator whose cost is config selection — that still needs the generated
launch-config table — and it does not cover reductions on a boxed route
(`sum.dim` has no Triton launch to capture). Measured cache-key cost is
0.35–0.78 µs per call; first-call miss is 250–483 µs.

Exit criteria: **direct-call submit 12–16 µs with a fresh output** (measured on
two independent wheels; the reused-output column reads 7.7–11.3 µs but hides the
3.5–4.5 µs of allocation a real op pays), and **through-the-device submit
≤ 22 µs** (12–16 µs plus the 3–7 µs plugin hop — the one number still unmeasured,
and the acceptance gate). Correctness must be exact against the CPU reference
across broadcast, non-contiguous, out/in-place and bf16 variants, with a
documented fallback for every case the cache cannot serve. This is the highest
value per unit of risk in the whole plan and should land before any C++ or bundle
work.

**Phase 1 — C++ / AOT static bundle for the host-bound families (weeks).**
Elementwise, reductions, normalization, rotary, embedding, `cat`, `copy_`. After
the Python shim, what remains is small but real: ~5 µs of Triton C-launcher,
3.5–4.5 µs of allocation, the GIL/pybind hop, and the CUDA-graph blocker — Phase 1
removes those plus the first-call JIT. It is justified by **GIL freedom, graph
capture and the last ~3–7 µs**, not by "C++ is faster".
Exit: submit ≤ 8 µs for covered ops, no Python frames in steady state, no
first-call Triton compile, capture-capable, bundle regenerable in CI.

**Measured head-to-head, and it moves the plan: the shim beats the C++ route on
H100 host time, and reductions are shim-able after all.** Same hour, same GPU,
same tensors, one process per arm; submit µs, median of 5, fp32; route C
verified genuine (`5152 × [flagos dispatch] _softmax -> flagos`, `0 × -> flagos_python`):

| case | CUDA floor | FlagGems Python | route C (`flaggems_cpp`) | shim, fresh output per call |
|---|---:|---:|---:|---:|
| `mul[512²]` | 8.64 | 53.60 | *no C++ slot registered* | **10.68** (`max_err` 0.0, 1.24×) |
| `_softmax[1024²]` | 8.50 | 27.38 | 32.22 (`max_err` 7.5e-09) | **11.80** (`max_err` 7.5e-09, 1.39×) |
| `sum.dim_IntList[1024²]` | 10.16 | 30.86 | *not reachable from Python* | **11.69** (`max_err` 7.5e-09, 1.15×) |
| `sum` full-reduce[1024²] | 14.16 | 40.45 | — | **16.55** (`max_err` = wrapper's, 1.17×) |
| `max.dim[1024²]` (tuple) | 11.96 | 104.08 | — | **12.31** (both outputs `max_err` 0.0, 1.03×) |

A leaner replay of the same two reduction kernels, implemented in the C++ runtime
route's own reference script and measured on its own wheel, lands at **10.54 µs `_softmax` (1.24×)**
and **10.75 µs `sum.dim` (1.06× — parity)**. Four implementation × wheel
combinations now agree on the ordering, with errors unchanged, and the finding is
worth stating in the plan's own words:

> The earlier 1.4–1.6× reduction figures were **implementation-bound, not
> mechanism-bound**: a naive replay costs 15–20% (~2 µs/op) — about the entire
> remaining vendor gap — so the **lean replay pattern is the baseline**: allocate
> at the result shape (never `.view()` per call) and patch only the output slot
> into a prebuilt argument list.

**Full-reduce `sum` was re-run lean on both wheels and improves only modestly:
16.55 µs (1.17×) and 16.45 µs at 512² (1.20×)** on the first, **18.49 µs (1.29×)**
on the second (whose wrapper baseline is also ~20% heavier, 49.09 vs 40.45 µs), from
17.35 / 17.12 naive. The band is therefore **1.17–1.29×** and it is the most
expensive reduction for the shim by construction: the only measured op that pays
for a second launch. A direct pointer census settled the intermediate count: **exactly one**
persistent intermediate — launch 0 writes `mid` at pointer B and launch 1 reads
that same pointer B, then writes the 0-dim output. The earlier readings of three
and of two were both the occurrence-versus-pointer slip, so no caveat is needed:
**one persistent intermediate, deduped by `data_ptr`, never re-initialised.** That is not an
implementation shortfall but a property of the op: **lever 1 is a no-op here** (the
captured output slot is already the 0-dim result shape, so there is no per-call
`.view()` to remove), only the prebuilt-argument-list lever applies, and with two
launches the bookkeeping it saves is small. The residual is launch-structural —
16.55 ≈ 14.16 floor + ~2.4 µs for the second launch and its `mid` plumbing,
consistent with a ~1.1 µs/launch replay model. So **1.17–1.29× is the honest floor
for this op shape**. The lean run also confirmed that patching only the final
output slot leaves `mid` undisturbed (launch count, intermediate count and
per-launch kinds identical to the naive path) and that `mid` still needs no
initialisation under the lean path. Independent evidence for the scratch rule came
from the second wheel: allocating intermediates **per slot occurrence** instead of
**per distinct `data_ptr`** produced `max_err` **2312.5** on exactly this op,
because launch 2 then read a fresh empty buffer instead of launch 1's partial
sums — deduplicating by pointer fixed it. Pointer-deduped, first-appearance
scratch is therefore a requirement, not an implementation detail.

The last two rows were re-derived in the corrected campaign and are **confirmed**:
the full-reduce arm records a real `out` slot, its `max_err` is bit-identical to
the wrapper's, and the stale-mid experiment holds; the `max.dim` row shows the
**multi-output** case (a tuple) replaying correctly at 1.03× the floor, which was
the last structural case the slot rule had not been exercised on.

**Two of these numbers were withdrawn after cross-checking, and the episode is
worth recording.** The first (now withdrawn, superseded) reduction shim figures
10.28 / 10.35 µs, quoted at the time as "1.02× vendor", came from runs that
classified the kernel's output slot as
reusable **scratch**: the slot matched on `data_ptr`, because FlagGems launches
into an internal buffer and returns `buffer.view(...)` — for `sum.dim` the buffer
is `(1024,1)` and the result is a squeezed view. The freshly allocated output was
therefore **never written**, so those runs were simultaneously fast and wrong
(`max_err` 0.0590 / 101.71). The rule was then re-derived independently and run as
**three implementations on two wheels**: the C++ runtime route's fixed code on its wheel
12.31–12.33 µs (1.22–1.47×), the same code on the second wheel 13.79–13.95 µs
(1.35–1.57×, the expected ~12% cohort delta), and a leaner replay on the second
wheel **11.69–11.80 µs = 1.13–1.40× the vendor floor**. The gap between the first
two readings was *not* allocation (allocation is independently confirmed at
3.9–4.0 µs `alloc_only` vs 2.7–3.2 µs for bare `empty`+drop); it was two
avoidable implementation costs in the reference harness — a per-call `.view()`
(1.75–2.35 µs, because `sum.dim` launches into a `(1024,1)` buffer and returns a
squeezed view) and ~2 µs of per-call replay bookkeeping (list rebuilds, index
lookups). Removing them is worth **~4 µs/op — more than the remaining vendor gap
for `sum.dim`**, so they are recorded as implementation rules, not tuning: allocate
at the **result** shape and let the argument wrapper supply the captured
shape/strides, and keep the replay to pointer patching.

The general lesson belongs in the harness, not just the prose: **an arm that does
not write its output is both wrong and fast**, so for host-bound operators **a shim
number below the vendor floor is a bug signal before it is a result** — the shim
replays a vendor-comparable kernel sequence and still allocates and returns a
tensor, so it cannot legitimately beat a bare ATen launch by that margin. The
inverse holds too, which is why timing alone is not the test: an invalid replay can
also be far *slower* than doing nothing (the `sort` replay is 16× the vendor floor
with no validated output, and a cross-shape reuse costs 719.3 µs against the
wrapper's 40.57). **A suspicious number in either direction is a bug signal; the
reliable check is the slot census** — did the plan produce an `out` slot and
validate an output — because that is what actually located the false negative. The fingerprint is the captured slot list —
a misclassified arm records `launch_arg_kinds: [["scratch","in"]]` where a correct
one records `[["out","in"]]` — and `max_err` alone cannot identify which artifact
a run came from, because every correct run reproduces the same rounding. Any
shim number quoted in a plan must therefore be traceable to an artifact whose
slot list shows a real output, not merely to a passing error check.

One fairness note for the record: the superseded pair that briefly circulated and
is not a planning figure (10.46 / 10.43
µs) did come from a *corrected* script (`shim_diag.json` records
`scratch_buffers: 0` with exact errors at those timings) — the defect was that the
fix never reached the **shared** script, whose published artifact stayed buggy.
That makes it a reproducibility failure rather than a fabricated measurement, but
a number the shared code cannot reproduce still cannot serve as a planning figure;
the independent cross-wheel reimplementation is what the plan cites.

Four consequences, in order of importance for the plan:

* **Route C retains no host-time advantage.** The shim is **2.4–2.6× faster than
  route C** on the same kernel where both were measured (10.54 vs 32.49 µs
  = 3.0×; 2.8× against the less-lean replay) and lands at **1.06–1.40× the
  vendor floor**, where route C is 3.7–3.9× off it and `_softmax` route C is even
  slower than the plain Python wrapper (32.49 vs 25.14). The C++ slot pays its N+1
  `SetTensorImplDevice` retags plus a TritonJIT launch and cannot repay them once
  the Python launcher is bypassed. So Phase 1 must **not** be justified by a
  host-time win; its case is exactly three properties Python cannot provide:
  **GIL freedom under threading, CUDA-graph-capturable launches, and AOT kernels
  (no per-process JIT/capture dependency)**.
* **Reductions are shim-able, with a scope caveat that is a source fact, not a
  guess.** Each of the three captured exactly **1 Triton launch**, and the output
  is simply the single non-input tensor slot — no reduction-specific adapter is
  required. The apparent "1 launch vs 2 kernels" contradiction resolves in the
  source: the full reduction `torch.sum` (`ops/sum.py:76`) uses `contiguous()` +
  `mid`/`out` buffers and **two** kernels (`sum_kernel_1`, `sum_kernel_2`),
  whereas `sum.dim_IntList` — what was measured here — is `sum_dim_comm`
  (`sum.py:262`) with **one**. **Both forms are in scope**: the
  single-launch ones measure **10.5–12.5 µs (1.06–1.45× vendor)** with fresh output
  per call — `sum.dim` at 1.06–1.19× is effectively parity — and are correct and
  ratio-flat across **bf16, non-square and non-last-dim** shapes and `dim=0`
  (the variant rows are lean too: 11.58–11.90 µs, flat — no shape or dtype tax
  beyond the cache key), and the two-launch full-reduce `sum` measures **16.55 µs**
  (confirmed on the lean path, **1.17–1.34×** across both wheels — the most
  cohort-sensitive row, since the wheel gap accrues over two launches instead of
  one — with a real `out` slot), with one persistent `mid` scratch that needs **no
  initialisation** — zeroing it costs +29.7 µs and buys nothing — and a capture
  that **must never be reused across shapes** (a mismatched replay measured
  `max_err` 31.38, corrupted the allocator, and for full-reduce `sum` took
  **719.3 µs** — slower than the untouched wrapper, so a wrong reuse is a
  performance regression as well as a correctness bug). Multi-launch is therefore not a
  boundary of the replay rule. The generality tax of reductions is in the
  per-config cache key, not in per-call cost; what remains unproven is
  multi-kernel fusions (`layernorm`-class). An earlier run appeared to contradict
  this, but that was a **harness bug**: slots were classified by `data_ptr`
  equality with the returned tensor, and FlagGems returns a **view** of the buffer
  it launched into, so the replay targeted a stale scratch buffer. Classify slots
  as *input* (`data_ptr` match) versus *the single non-input tensor slot*.
  Phase 0.5 therefore covers **pointwise and reductions**.
* **The A100 ordering has reversed, so the old evidence must not be quoted as
  current guidance.** On A100 the Python wrapper cost 49 µs and route C 11 µs
  (4.5× faster); on this H100 cohort the wrapper is 24.89 µs and route C 32.22 µs
  (**1.3× slower**). Today's FlagGems Python wrapper is roughly 2× leaner, so
  there is little Python overhead left for C++ to remove, while C++'s fixed
  costs (retags, per-launch marshalling, no autotune) do not shrink.
* **"Registered" is not "reachable", and it is now confirmed mechanically.** With
  `FLAGOS_LOG=dispatch,fallback` the C++ arm prints
  `[flagos cpu_fallback] aten::sum` for `torch.sum(x, dim=1)`, while the control
  `torch.sum(x)` (full reduction) prints `[flagos dispatch] sum -> flagos_python`
  — i.e. that wheel's PrivateUse1 claim list has an `m.impl` for `sum` but not for
  `sum.dim_IntList` (nor `addmm`, `index_select`, which `embedding` decomposes
  into). A route-table entry is only consulted once the PrivateUse1 kernel is
  entered, so `FLAGOS_OP_<op>=flaggems_cpp` prints its override line at load and
  then has **no effect**. This is a **silent performance cliff**: an unclaimed op
  works, but on the CPU round trip — 571 µs versus 31.55 µs for the FlagGems
  wrapper directly (≈18× slower) and 10.22 µs for the vendor kernel. The fix is a
  **claim-list audit**, not per-op latency work (§8).

One comparability trap for whoever repeats this: the shim's advantage is a
**contiguous-equal-shape** advantage, and the captured output slot can differ from
the returned view (`sum.dim` captures `(1024, 1)` and returns `(1024,)`). Allocate
the *captured* shape and expose the caller's view, consistent with the
per-family allocation-mode rule in §8.

**Phase 2 — kernel-quality routing (weeks).**
Measure e2e per operator at target-model shapes; route compute-bound operators
whose FlagGems kernel trails the vendor kernel to the vendor route, with the
measurement recorded in the conf. One case is already decided: **`sort`** (and by
family `topk`) is a multi-pass radix sort whose replay alone costs 644 µs against a
39.4 µs vendor floor (~16×), so it goes to the vendor kernel regardless of wrapper
design. The measured evidence for this phase already
exists: on H100 `mm[4096²]` fp32 runs 2211 µs on the FlagGems kernel against
369 µs on the vendor kernel (6×), `bmm` ~5×, `addmm` ~4×, `index_select` ~9×,
while `softmax[4096²]` (49 vs 64), `layernorm[4096²]` (61 vs 66) and `add` are at
parity or faster on FlagGems. Wrapper parity is not operator parity, and the
routing table has to encode that per operator per shape class.

**Phase 3 — model-level amortisation.** CUDA graph capture was verified to work
on flagos operators for all three routes, with graph replay at 0.1–0.2 µs per
small op (versus 78–94 µs eager). Compare eager vs CUDA graph vs
`torch.compile(backend="flagos")` on the target models at their real shapes; the
eager wrapper stays the correctness baseline, the graph/compile path is the
throughput path for static-shape loops (decode).

## 8. Risks and open questions

* **Triton ABI and kernel drift**: AOT cubins pin Triton and kernel sources; the
  manifest hash must make mismatches loud.
* **Build weight / wheel size**: scope the bundle to the operators that matter.
* **Edge cases**: non-contiguous layouts, broadcasting, dtype promotion,
  out-variants, zero-size tensors, 64-bit index tensors — handle or fall back
  deterministically.
* **Upstream coverage**: `mul`/`sub`/`where`/`gelu`/`silu`/`layernorm` are not
  bound in FlagGems' C++ library. The AOT design does not need those bindings
  (it compiles Triton sources directly), but the Phase-0 `flaggems_cpp` route
  does; until then those operators keep the Python or vendor route.
* **Autotune table coverage**: shapes outside the table need a deterministic
  fallback and a measurement update process.
* **The Python fast-launch shim is the highest-risk/highest-value item.** It is a
  **per-configuration JIT cache**, and its cache key is a correctness boundary, not
  a performance one: reusing an equal-shape capture on broadcast inputs measured
  `max_err` 39.87 with no error raised, and a naive stride assumption on
  non-contiguous `add` measured 8.93. Production **must** key on
  `(op, dtype pair, shape, strides, contiguity, out vs fresh)` and fall back to the
  stdlib wrapper on any miss; a stale or loose key means silently wrong results.
  It must also **replay the allocation policy observed at capture time, per
  operator family** — this is not one global rule. The `pointwise_dynamic`
  family (`add`, `relu`, …) allocates `torch.empty_like(input)`, so its output
  preserves the input's physical layout; allocating contiguous instead is silently
  wrong on non-contiguous inputs (**`max_err` 8.76 `add`, 4.46 `relu`**; 0.0 with
  `empty_like`). The hand-written wrappers (`mul`, which uses `torch.empty(shape)`)
  bake **contiguous** output strides into their captured args: there, `empty_like`
  on a transposed input is silently wrong (**`max_err` 14.07**; 0.0 with
  contiguous). So the cache entry carries the allocation mode, and the safe
  implementation replays whatever the captured launch did rather than applying a
  global preference. The **output slot must also never be classified by
  `data_ptr` equality with the returned tensor**: FlagGems launches into an
  internal buffer and returns `buffer.view(...)` (for `sum.dim` the buffer is
  `(1024,1)`, the result a squeezed view), so pointer equality fails and the slot
  looks like reusable scratch. Classify by **first appearance** instead — a
  non-input slot first seen in the **last** launch is the output, and non-input
  slots first seen earlier are persistent-but-uninitialised intermediates.
  Precondition: the operator's output must be *introduced in the final launch*;
  the rule is op-agnostic but not algorithm-agnostic (this is why full-reduce
  `sum` passes and a ping-pong `sort` cannot be captured at all).
  Misclassifying this is not slow, it is wrong (the freshly allocated output is
  never written), and it is what produced the withdrawn "1.02× vendor" reduction
  figures in §7. Two measured implementation details are worth ~4 µs/op and are
  therefore also rules: allocate at the **result** shape and pass the captured
  shape/strides through the argument wrapper, rather than allocating at the
  captured buffer shape and calling `.view()` on every call (1.75–2.35 µs); and
  keep the replay path to pointer patching, with no per-call list rebuilds or
  index lookups (~2 µs).
  That makes **four silent-wrongness classes** in this design — strides missing
  from the key (broadcast reuse 39.87), a wrong per-family allocation mode
  (8.76 / 4.46 / 14.07), output slot classified by `data_ptr` (the 0.059 / 101.71
  set, whose 10.28–10.35 µs timings must never be quoted), and cross-shape capture reuse (31.38) — and all are
  structural rules (wrong numbers, no exception raised), so key + allocation mode
  + slot rule + shape-keyed captures + fallback must be reviewed together as a
  correctness invariant, not an optimisation.
  Because `mul`-class configs come from `LibTuner`, a cached launcher embeds the
  tuner's choice, so build the cache **per process from a live FlagGems call**,
  and any cross-process cache must include the **config hash** in its key.
  It is also per-op-**family**, not universal: `mul` launches 16 argument slots
  with plain `torch.Tensor` at [9,10,11]; `add` launches 24 slots with
  `StridedBuffer` wrappers at [9,10,12]; `relu` uses 20 slots of
  `StridedBuffer` — the adapter is exactly why `add` costs 2.2–3.3 µs more than
  `mul` in shim form, and it is also the part of the Phase-1 marshalling budget
  most likely to be under-estimated. Finally it duplicates FlagGems internals
  (`LibEntry`/launcher APIs), so it must be pinned, tested against the installed
  cohort, and treated as an upstream candidate rather than a permanent fork. It
  remains a Python path: it does not make the route CUDA-graph-capturable — that
  still requires the native launcher of Phase 1.
* **A native launcher without a config table makes matmul worse.** This is
  measured, not hypothetical: `libtriton_jit` has no autotuning, and that is
  exactly why route C regressed `mm` (110 vs 65 µs on A100) while the Python path
  selects tuned META. Phase 0/1 must ship the generated launch-config table with
  the launcher, or restrict the new route to the elementwise/reduction families
  and leave matmul on its current route.
* **Claim-list gap: registered operators that are not claimed on PrivateUse1.**
  Confirmed by log (`FLAGOS_LOG=dispatch,fallback`): `torch.sum(x, dim=1)` emits
  `[flagos cpu_fallback] aten::sum`, `torch.addmm` emits
  `[flagos cpu_fallback] aten::addmm`, and `torch.embedding` emits
  `[flagos cpu_fallback] aten::index_select` — while the control `torch.sum(x)`
  (full reduction) emits `[flagos dispatch] sum -> flagos_python`. So the wheel
  claims `sum` but not `sum.dim_IntList`/`addmm`/`index_select`, and the forced
  route key has no effect for them. **Action item, quantified**: audit and fix the
  `m.impl` claim list before any route depends on those operators — an unclaimed
  op is ~18× slower than the FlagGems wrapper called directly (571 vs 31.55 µs)
  and 56× the vendor floor (10.22 µs), and no `FLAGOS_OP_*` override can rescue
  it. Note the fallback log prints the **base** ATen name (`aten::sum`), not the
  overload, so the dispatch line is what proves which overload is claimed; and
  because such an op never enters the dispatcher, **all three arms of a routing
  comparison come out identical** — which is easy to misread as "the routes
  perform the same" rather than "no route is in effect". A claim-list audit is
  worth 18× on the affected operators and must precede any per-op routing
  decision.
* **Multi-output operators: split by launch count, not by "tuple".** A tuple is
  not itself a problem: `max.dim` (values + indices, one launch) replays correctly
  at **12.31 µs = 1.03× the floor** with both outputs exact. But `sort` **fails and
  cannot be fixed by an adapter** — FlagGems' sort is a multi-pass radix sort with
  **21 launches ping-ponging over 9 buffers**, so no output slot is ever introduced
  in the last launch and the first-appearance rule has nothing to identify. More
  decisively, sort is **kernel-bound, not host-bound**: replaying those 21 kernels
  into scratch already costs **644 µs ≈ 16× the 39.4 µs vendor floor**, and the
  wrapper adds ~920 µs on top, so even a perfect shim would sit ~16× off vendor.
  The correct treatment is the same as large `mm`: **route `sort` (and `topk`,
  same multi-pass family, unmeasured) to the vendor kernel** and keep them out of
  Phase 0.5. Do not generalise "tuple ops work" from `max.dim`; the discriminator
  is the number of launches and whether the output slot is introduced at the end. The measured `max.dim` arm also confirms the generalisation:
  every non-input slot first appearing in the last launch is an output.
* **Routing policy**: the default HEAD NVIDIA conf has no vendor route; any
  vendor-parity claim must state which wheel/conf it was measured on.
* **Python-route lifetime**: the Python route should remain as the correctness
  reference and the fallback for uncovered operators, not as a hot path.

## 9. Implementation status (Phase 0.5 skeleton)

The replay core is implemented, not just designed:
`torch_fl/fast_launch.py` (provider-agnostic plan/cache/replay, standalone-loadable
— importing it requires no torch, triton or flag_gems) plus
`tests/unit/test_fast_launch.py` (23 hermetic tests over an injected fake provider).
It is **not wired into `torch_fl/__init__`** yet: nothing enables it implicitly, so
landing it cannot change any existing route. `enable(names)` / `disable()` are the
only entry points, and `FlagGemsProvider` imports the device stack lazily.

What the code encodes, one invariant per measured silent-wrongness class:

| rule | where | test |
|---|---|---|
| slots by first appearance, never `data_ptr` of the result | `build_plan` | `test_single_launch_classification`, `test_ping_pong_capture_is_unsupported` |
| one intermediate per **distinct pointer**, never per occurrence | `build_plan` | `test_scratch_is_deduplicated_per_pointer` |
| intermediates persistent and never re-initialised | `Plan.scratch` | `test_scratch_is_persistent_and_never_reinitialised` |
| cache key carries shape + strides + contiguity; a miss re-captures | `signature_of`, `PlanCache` | `test_signature_distinguishes_strides_and_contiguity`, `test_two_shapes_capture_twice_instead_of_reusing` |
| allocation policy replayed per family (`empty_like` vs contiguous) | `OutSpec.flat` | `test_contiguous_output_is_allocated_at_result_shape`, `test_noncontiguous_output_replays_the_captured_layout` |
| output allocated at the result shape, no per-call `.view()` | `Plan.replay` | `test_output_allocated_at_result_shape_without_view` |
| replay patches a prebuilt argument list | `LaunchPlan.arg_list` | `test_replay_reuses_prebuilt_argument_list` |
| unsupported captures fall back to the untouched callable | `wrap_callable` | `test_wrap_callable_falls_back_for_unsupported_ops` |
| **warm before capturing**: the cold call may still be autotuning (~19 launches) | `wrap_callable` state machine | `test_wrap_callable_captures_once_then_replays` |
| inputs identified by identity, never by `data_ptr` (aliases, empty tensors at ptr 0) | `build_plan` | `test_aliased_inputs_are_not_collapsed_by_pointer`, `test_zero_element_tensor_is_refused_not_misclassified` |
| replay returns the same container the callable does (tuple for `max.dim`) | `Plan.returns_tuple` | `test_multi_output_keeps_the_tuple_container` |
| **outputs paired to results by base storage, in call order** — pointer order is unrelated to return order, so pairing `sorted(data_ptr)` with the result list silently swaps `max.dim`'s values and indices | `build_plan` (`storage_ptr`) | `test_multi_output_order_survives_pointer_order`, `test_multi_output_with_different_shapes_is_paired_correctly`, `test_ambiguous_output_pairing_is_refused` |
| a replay-time failure falls back instead of raising | `wrap_callable` | `test_replay_failure_falls_back_instead_of_raising` |

Two implementation details came from review rather than from the measurement campaign
and are worth keeping: the **warm-before-capture** state machine (capturing the cold
call records libtuner's ~19 benchmarking launches, which would blacklist the operator
as an unsupported multi-launch algorithm) and **replaying the captured allocation
template** instead of encoding a per-family rule — replaying the template's layout
subsumes both `empty_like(input)` (pointwise_dynamic) and contiguous `empty(shape)`
(hand-written ops) without naming either family.

The multi-output ordering rule came from a second review pass and is worth stating
because it is invisible to accuracy checks: which *slots* are outputs is decided by
first appearance, but which *result* came from which slot must be established
separately, and the only stable identifier is the **base storage pointer**
(`untyped_storage().data_ptr()`, unaffected by `storage_offset`). Pairing pointer
order with return order returns indices where values belong — int64 for float32,
no exception, and an `max_err` check against a correctly-ordered reference arm
cannot see it. Because only `max.dim` is characterised, an ambiguous pairing (two
results sharing one storage) is refused rather than guessed.

The hardware gate is `benchmarks/fast_launch_integration.py`. It exists because
the three defects this shim is built to avoid are invisible to an accuracy check —
a plan that classifies its output slot as scratch never writes the result, a swapped
multi-output pairing returns indices where values belong, and a per-occurrence
intermediate lets launch 2 read a fresh buffer. So the gate is a slot census plus
exactness (against both the untouched FlagGems callable and a CPU reference), the
`max.dim` ordering check that is the only hardware test of the storage-pointer
pairing assumption, a fail-closed check that `sort` is refused rather than replayed,
and a no-growth check over repeated calls.

Deliberately out of the skeleton's scope: `out=`/in-place variants and keyword
arguments fall back; the environment switch is expected to be a `FLAGOS_*`
variable registered in `torch_fl/_env.py` with its mirror in
`docs/reference/environment-variables.md` (the module reads no environment, which
is what keeps it importable on a machine without the device stack); and the
allocator-growth / leak case from the review matrix needs the real device to
exercise, so it belongs in the integration script rather than the unit tests —
and so does re-measuring the multi-output row: the 12.31 µs / 1.03× figures were
produced by an arm with correct ordering, so the plan path must reproduce the same
`(values, indices)` pairing on hardware before §2's row is taken as covering the
adapter rather than the kernel.

## 10. Reaching vendor parity with FlagGems kernels (mechanism design)

Sections 2 and 9 measure the shim against the FlagGems wrapper and against a raw vendor
kernel floor. This section answers the design question the project actually asks: **with
FlagGems as the compute backend and no routing to vendor kernels, what wrapper mechanism
reaches vendor parity?** Boxing is not an answer here - it is being retired - so every
number below keeps the FlagGems Triton kernel.

### 10.1 Where the host time goes today

`torch.mul[1024²]` on `flagos`, per call, measured by probing each component:

| component | µs |
|---|---:|
| vendor `torch.mul` on `cuda` (same host) | 6.80 |
| Python replay: allocation + arg patch + Triton launcher | 12.96 |
| — of which `at::empty` on the flagos device | 4.04 |
| — of which the Python-side launch | 8.92 |
| signature key (shape/stride/dtype/contiguity) | 4.41 |
| signature + plan lookup | 5.12 |
| full shim end to end through `torch.mul` | 22.73 |

The Python wrapper's own work (keying + allocation + launcher + bridge) is ~14 µs above
the kernel. **None of it can be removed while the replay stays in Python**: the Triton
launcher alone is 11.98 µs of pure Python per launch, versus 6.94 µs for the entire
vendor eager call including its allocation and dispatch.

### 10.2 The mechanism that closes it: capture in Python, replay in C++ by graph

Two measured steps, both keeping the FlagGems kernel:

| path | µs | note |
|---|---:|---|
| Triton eager launch (current FlagGems path) | 11.98 | Python launcher |
| C++ replay: cached `CUfunction` + prebuilt arg array + `cuLaunchKernel` | 10.07 | via pybind, incl. 1.54 µs of binding; exact (`max_err` 0.0) |
| **C++ replay of the same Triton kernel through a CUDA graph** | **4.39** | `max_err` 0.0 |
| vendor `torch.mul` eager (reference) | 6.94 | - |
| graph replay of one vs two kernels | 4.51 / 4.39 | replay cost is flat in launch count |

A graph replay is a single driver call, so the per-launch Python work, the per-call
allocation and the argument marshalling all disappear from the hot path. At 4.39 µs the
mechanism is **cheaper than the vendor's entire eager call**, which leaves headroom for
the PrivateUse1 dispatch hop (~2.5 µs measured: 22.73 e2e versus 20.2 direct) and still
lands at ~6.9 µs - parity.

### 10.3 The design that follows

`flaggems_fast` - a new backend key, with `flaggems` untouched as the fallback:

1. **First call per signature (Python, once).** Invoke the FlagGems entry point normally.
   This is also where libtuner autotunes, so it must be allowed to run (~19 launches).
   Then capture the steady-state launch and export to C++: cubin, kernel name,
   `num_warps`, shared-memory size, the kernel's argument list, and the output allocation
   policy observed in the capture (layout, not a per-family rule).
2. **C++ side.** `cuModuleLoadData` + `cuModuleGetFunction` once; build a
   `cudaGraph` containing one kernel node per captured launch, with prebuilt kernel
   parameters; instantiate the graph executable.
3. **Per call.** Take an output buffer from a per-plan pool (a small ring; the previous
   result's refcount decides when its slot returns), point the graph's kernel-node
   parameters at the current inputs and that slot (`cudaGraphExecKernelNodeSetParams`),
   `cudaGraphLaunch`, and hand back a tensor over the slot with a deleter that returns it
   to the pool. Rotating one captured graph per slot avoids the parameter update on the
   common path entirely.
4. **Fallback contract unchanged.** Anything the capture cannot describe - multi-pass
   ping-pong, ambiguous output pairing, unsupported argument kinds - keeps calling the
   untouched FlagGems entry point.

Two properties fall out that no Python route has: the launch is **GIL-free**, and the
route is **CUDA-graph capturable** (the FlagGems Python wrapper is not, because it
allocates per call).

### 10.4 Prototype status: the pooled C++ replay works

`benchmarks/fast_launch_cpp/` implements it end to end for one Triton kernel (`mul`, 1M fp32): Python registers a
plan once (cubin, kernel name, grid/block/shared,
argument kinds, pool size); C++ then owns a `CUfunction`, a prebuilt argument array and
a pool of output buffers. The per-call path is pointer stores plus `cuLaunchKernel` -
no Python, no allocation, no argument marshalling.

Measured on H100, three consecutive runs, `max_err` 0.0 every time:

| run | pooled C++ replay | vendor `torch.mul` | Triton eager launch |
|---|---:|---:|---:|
| 1 | 4.09 µs | 6.98 | 16.06 |
| 2 | 4.29 µs | 6.96 | 15.86 |
| 3 | 4.22 µs | 6.65 | 15.49 |

The replay is **~4.2 µs against a ~6.9 µs vendor eager call**, and the vendor number
includes ATen's own dispatch while the replay number does not. Adding the PrivateUse1
hop (~2.5 µs, measured as the difference between 22.73 µs end to end and 20.2 µs direct)
projects **~6.7 µs, i.e. 0.96-1.0× the vendor operator** - parity, without graphs.

### 10.5 Two hard-won findings for anyone reusing this

**The driver reads the PTX parameter list, not the Python-level signature.** This kernel
declares six parameters:

```
param_0..2  u64 .ptr   (a, b, out)
param_3     u32        (n)
param_4..5  u64 .ptr   (global scratch, profile scratch)
```

Triton 3.6 emits both scratch pointers **even when their sizes are 0**, and
`metadata.global_scratch_size` reads 0 for a kernel that still declares the parameter.
Launching with four arguments made the driver read uninitialized stack: one run produced
`max_err` 0.0 and an implausibly good 3.53 µs, the next returned
`CUDA_ERROR_INVALID_VALUE`, and reverting did not reproduce either. Reading
`asm["ptx"]`'s `.entry` declaration settled it, and with all six arguments the result is
correct and repeatable. **Derive the argument list from the compiled kernel, never from
the wrapper's Python signature** - and treat a number that appears only in one
configuration as a bug signal, which is how this one was caught.

**Argument storage must be live for the duration of the call.** Passing a pointer array
that lives on the stack, built fresh per call, is what the driver accepts here; the
equivalent heap-vector arrangement crashed. Both are cheap, so the stack form is the
prototype's.

### 10.6 Output pool: refcount-driven, and verified

The pool returns a tensor whose deleter puts the slot back when the last reference drops
(`at::from_blob` with a deleter over a per-slot shared pointer). A slot is never handed
out while a live result holds it; instead the pool grows, so the steady state stays
allocation-free and two live results can never alias. Measured, with the assertions the
design requires:

| check | result |
|---|---|
| two results held at once | different buffers, both exact |
| slot reuse after the first result is dropped | reused, result exact |
| 40 results held simultaneously | 40 distinct buffers, all exact |
| replay cost with the pool | **3.44 µs** (vs 6.83 µs vendor `torch.mul`, 15.84 µs Triton eager) |

Against the FlagGems wrapper as measured on the same host for `torch.mul[1M]`
(78.73 µs end to end), the mechanism is **~23× cheaper**; against the vendor operator it
is ~2× cheaper, before the PrivateUse1 hop.

### 10.7 A real FlagGems operator, end to end

The same code captures `torch.mul[1M fp32]` on the flagos device through the FlagGems
route, exports the compiled kernel to C++, and replays it.
Capture prints the whole mapping, which is also the check that the argument list is right:

```
kernel=mul_kernel cubin=7640B grid=1024 threads=128 shared=0
kinds=['in', 'in', 'out', 'i32', 'zero', 'zero']
ints=[1048576]   ptx=['ptr', 'ptr', 'ptr', 'i32', 'ptr', 'ptr']
```

| path | µs |
|---|---:|
| FlagGems wrapper, `torch.mul` end to end on `flagos` | 80.33 |
| C++ replay of the same kernel, called from Python (pybind + Python call included) | 10.58 |
| **C++ replay inside C++, i.e. what the plugin would pay** | **4.33** |
| vendor `torch.mul` on `cuda` (same host) | ~6.83 |

With the PrivateUse1 hop (~2.5 µs) the projected end-to-end figure is **~6.8 µs against a
~6.8 µs vendor operator - parity, while executing the FlagGems Triton kernel**, and about
**12× faster than today's wrapper** (80.3 µs). `max_err` is 0.0 throughout.

**The argument list must come from the PTX, not from the launcher's arguments.** Recording
`flag_gems.mul`'s launch shows user arguments
`[Tensor, Tensor, Tensor, int, str('float32'), int, bool]`: pointwise kernels forward
constexprs - including a dtype *string* - that have no kernel parameter, and the extra
integer `1024` beside the element count is `BLOCK`, likewise not a parameter. The PTX
`.entry` declaration is the authority: parse its parameter kinds (`ptr` / `i32`), map the
launch's tensors and its first `scalars` integers onto them positionally, drop the
remaining constexprs, and zero-fill the scratch pointers. Getting this wrong is not a
crash but a plausible-looking wrong answer, which is why the capture prints both lists
side by side.

### 10.8 What shipped, and what is left

`benchmarks/fast_launch_cpp/` contains the mechanism described above: the CUDA extension
(`fast_launch_ext.cu`, a `CUfunction` per plan, a prebuilt argument array, a
refcount-driven output pool, `cuLaunchKernel`), the Python capture and replay layer
(`fast_launch_cpp.py`), and the hardware gate (`fast_launch_cpp_gate.py`). It is opt-in:
`enable()` replaces the FlagGems bindings for the operators it can prove and declines the
rest by name.

Two arms of the gate on H100, `max_err` shown as relative error:

| operator | wrapper | replayed from C++ | speed-up | rel. error |
|---|---:|---:|---:|---:|
| `mul` 512x512 | 82.48 us | **15.05 us** | **5.48x** | 0.0 |
| `softmax` 1024x1024 | 30.11 us | **14.89 us** | **2.02x** | 7.4e-09 |
| `add` 512x512 | 93.16 us | declined (keyword arguments) | - | 0.0 |
| `relu` 512x512 | 76.08 us | declined (see below) | - | 0.0 |
| `sum_dim` 1024x1024 | 10.93 us | declined (keyword arguments) | - | 1.7e-07 |
| `max.dim` 1024x1024 | 112.24 us | declined (two outputs) | - | 0.0 |
| `sort` 1024x1024 | 1293 us | declined (21 launches) | - | 0.0 |

The gate passes on both arms, with `hits` 2038, `fallbacks` 0 and every declined
operator keeping its untouched cost.

**A verified-operator allowlist is part of the design, not a limitation of it.** `relu`
captures a seven-parameter layout with three scalars whose replay does not reproduce the
wrapper's result (relative error 1.36), and the cause is not yet explained. A wrong
layout does not crash - it returns a plausible wrong answer - so the module refuses any
operator outside `VERIFIED_OPS` instead of trusting the mapping, and adding a name
requires a gate run that proves exactness. The remaining work is therefore precise:

* **Explain `relu`'s three-scalar layout.** Its PTX has more scalar parameters than the
  pointwise operators do, and the mapping that works for `mul` does not reproduce its
  results. Until that is understood, `relu` stays out of `VERIFIED_OPS`.
* **Key plans on keyword arguments** so `add` (`alpha`) and `sum_dim` (`dtype`) can be
  captured; today a call carrying keywords is passed through, which is why both are
  declined.
* **Multi-output and multi-launch plans.** `max.dim` needs two pooled slots in one
  launch; full-reduce `sum` needs two launches in one plan; `sort` is out of scope
  (kernel-bound, 21 launches).
* **Wire it behind a route name.** The mechanism has the numbers now; the remaining work
  is the dispatch plumbing (`flaggems_fast` key or an internal fast path of `flaggems`),
  which also decides packaging: today the extension is built at first use through
  `torch.utils.cpp_extension`, and a shipped route should build it with the wheel.

## Appendix A — how the measurements were produced

The raw JSON and profiler captures behind these tables were retained on the
measurement host and are deliberately not committed: they are large, machine-specific
and unreviewable in a diff. What is committed is the code that reproduces the
load-bearing numbers, plus the description below.

**Committed reproduction entry point.** `benchmarks/fast_launch_integration.py` runs
the acceptance gate for the Python side of this design on a CUDA host: the slot census,
exactness against an untouched `flag_gems.<op>` and against a CPU reference, the
`max_dim` output-ordering check, the `sort` fail-closed check, plan-growth stability and
timings. It takes `--impl PATH` to load an implementation by path and `--allow-missing`
to skip ops a given wheel does not expose.

**Measurement activities.** Six separate campaigns produced the numbers in this
document. Each is described here so a reviewer can tell what kind of evidence a given
table rests on:

1. **Routing matrix.** A 70-case operator matrix run under three routing modes (default,
   forced vendor, forced FlagGems) with per-operator overrides
   (`FLAGOS_OP_<op>=cuda|flaggems`), plus host/GPU decomposition of each call, a chain
   curve locating the host-bound to GPU-bound crossover, a CUDA-graph capture test (its
   in-place component is what distinguishes a real capture from the Python route's
   false positive), and a profiler audit.
2. **Per-layer decomposition of the Python route.** cProfile and torch.profiler
   attribution, raw launch floors (4.8 µs `CudaLauncher`, 10.7 µs `JITFunction`), the
   two wrapper generations (`pointwise_dynamic` versus `libentry`/`libtuner`), cold
   start and autotune data, and the upstream C++ wrapper analysis.
3. **C++ runtime route.** An H100 build of the FlagGems C++ runtime (`liboperators.so`
   plus `libtriton_jit`) and a torch-fl wheel built with `FLAGOS_BUILD_FLAGGEMS_CPP=1`,
   then steady-state and cold-start A/B against the Python route and against the vendor
   floor.
4. **Python fast-launch shim.** The prototype behind §5-§7, its decomposition
   (`LibEntry.run` 23.8 → 5.0 µs, `broadcast_shapes` 7.4 µs, cache-key cost
   0.35-0.78 µs) and a generality run across broadcast / non-contiguous / bf16,
   including the two silent-wrong-answer cases the cache key must prevent.
5. **Same-process head-to-head.** Raw-CUDA floor / FlagGems Python / C++ runtime route /
   shim on `_softmax` and `sum.dim_IntList`, with the dispatch log as proof that the C++
   route really engaged (`-> flagos`, not `-> flagos_python`).
6. **Two-launch full-reduce `sum`.** Both shapes on the lean path, the per-launch tensor
   kinds `[in, scratch]` / `[scratch, out]`, the stale-versus-zeroed `mid` comparison
   (+29.7 µs, no accuracy change) and the cross-shape reuse failure (`max_err` 31.38,
   allocator corruption - that destructive check must run last or in a throwaway
   process), plus launch counts per reduction variant.

The §2 A/B table was measured independently with per-operator routing overrides
(`FLAGOS_OP_<op>=cuda|flaggems`).

**One artifact is defective and is called out so it is never quoted.** An early shim
run recorded its slot list as `[["scratch","in"]]` and its `max_err_fresh` as
0.0590 / 101.71: it reused a stale scratch buffer, so those runs were simultaneously
fast and wrong. Its timings must not be cited. The corrected rule was re-derived
independently and run as three implementations across two wheels.

Verification status at hand-off, stated as the evidence actually supports:
**pointwise shim** validated on two independent wheels under the fresh-allocation
contract (12-16 µs, 1.4-1.8× vendor, §0); **reductions** re-derived on a second wheel
after the output-slot bug: `_softmax[1024²]` **10.54-11.80 µs (1.24-1.39× floor)** and
`sum.dim_IntList[1024²]` **10.75-11.69 µs (1.06-1.16×, vendor parity - a real measured
value, not a rounding claim)**, correct across bf16 / non-square / non-last-dim /
`dim=0`, with full numerics correct; full-reduce `sum` is **16.45-18.49 µs
(1.17-1.29×) on the lean path** across the two wheels, with a real `out` slot and
bit-identical error - its second launch is structural, so it does not reach parity; the
**multi-output** case is exercised for single-launch tuples: `max.dim` at 12.31 µs =
1.03× floor with both outputs exact; **`sort` is measured and explicitly out of scope**
(21 launches over 9 buffers, replay alone 644 µs ≈ 16× the vendor floor - kernel-bound,
route it to vendor), with `topk` unmeasured but the same family; **the C++ runtime
route** is measurable for `_softmax` only (32.22 µs, correct), not for `sum.dim`
(unreachable from Python). **The two previously unverified non-contiguous pointwise
cells are now closed**: `add` non-contiguous **10.38 µs (7.4×) with `max_err` 0.0** and
`relu` non-contiguous **9.73 µs (6.8×) with `max_err` 0.0**, so the pointwise matrix
(equal-shape / broadcast / non-contiguous / bf16, fp32+bf16) has **no unverified cells**.
One earlier shim run reported wrong reduction values - that was the slot-classification
bug described in §7 (`data_ptr` versus the wrapper's returned **view**), not a mechanism
limit; the corrected run is exact (`max_err` 7.5e-09 softmax, 1.5e-05 `sum.dim`).

## Appendix B — code references

| what | where |
|---|---|
| plugin dispatcher, per-op route memo | `csrc/aten/dispatcher.h` |
| generated Python-route kernels | `csrc/aten/generated/flaggems_python_kernels.cc` |
| hand-written C++ bridge to FlagGems | `csrc/aten/flaggems_cpp_kernels.cc` |
| Python op caller (IValue, pybind, GIL, callable cache) | `csrc/aten/backends/flagos/python_op_caller.cc` |
| boxing / unboxing | `csrc/aten/device_boxing.h` |
| route table, env overrides | `csrc/aten/common.cc` |
| codegen | `scripts/codegen/codegen_ops.py` |
| routing confs and their generator | `torch_fl/configs/backends_*.conf`, `scripts/codegen/gen_vendor_confs.py` |
| existing benchmark harness | `benchmarks/flaggems_dispatch_bench.py`, `benchmarks/flaggems_dispatch_overhead.py` |
## Appendix C — Adding a `flaggems_static` route: touch points and silent failures

Scope note: none of the below is needed if the fast path is an **internal
optimisation behind the existing `flaggems` key** (which is what the measured
Python shim can be: same conf key, same route, faster launch inside it). The
checklist applies only if a *new* conf key is introduced — and adopting the shim
first means Phase 0.5 needs none of it.

Adding a conf key is not a one-file change: `Backend` is an exhaustive enum with
`-Wswitch` on purpose (`csrc/aten/dispatcher.h`), and several adjacent tables are
*not* compiler-checked. The list below is from a source review at the analysis
commit; the three marked **silent** fail without a compile error and would ship a
slow or broken path. Order matters.

| # | where | change | why |
|---|---|---|---|
| 1 | `csrc/aten/common.h` (`enum class Backend`) | new enumerator; keep `kNone`/`kUncached` last (`kUncached` is the per-op memo sentinel) | slot identity |
| 2 | `csrc/aten/common.cc` `ParseBackendName` | parse the new conf spelling | **silent**: an unknown value warns and routes the op to **FlagGems**, i.e. the slow path the route was meant to replace |
| 3 | `csrc/aten/common.cc` `IsFlagGems` | include the new key in the FlagGems family | otherwise `FLAGOS_FORCE_BACKEND=vendor` treats it as a vendor target and repins operators into a slot with no kernels |
| 4 | `csrc/aten/common.cc` `ApplyForcedBackend` | target selection reads `IsFlagGems` | same class as #3 |
| 5 | `csrc/aten/dispatcher.h` `RegisterKernel` | new kernel slot | registration |
| 6 | `csrc/aten/dispatcher.h` `GetFn` | `flaggems_static → flaggems_cpp → flaggems` degradation | one switch arm, no per-call state; same shape the existing routes use, with a "why this order" comment |
| 7 | `csrc/aten/dispatcher.h` `KernelSetOf` | name the kernel set | keeps the "set not compiled into this wheel" diagnostic meaningful |
| 8 | `csrc/aten/dispatcher.h` `NameOf` | log name | `tests/unit/test_route_log_name_tables.py` and the integration `_LOG_NAME` table must map it, or `[flagos dispatch] <op> -> <name>` assertions break |
| 9 | `csrc/aten/dispatcher.h` `VendorSlot`, `RouteAcceptsDecoded`, `DecodedFallbackFn`, local `ParseBackendName` | decide explicitly whether the key is safe on the soft-lowp decode path | decode path enumerates slot kinds |
| 10 | `csrc/aten/register.cc` + per-platform claim lists | claim `m.impl` in both directions | claiming an op the conf routes `none`, or not claiming one the conf routes, is the failure documented at `register.cc:790-840` |
| 11 | `scripts/codegen/codegen_boxing_flaggems_register.py` | regenerate `register_flaggems_<vendor>.inc` | **silent**: an op routed to the new key but absent from the claim list is never claimed on PrivateUse1 → `cpu_fallback` (the PPU bug recorded in the same comment) |
| 12 | `scripts/codegen/gen_vendor_confs.py` | emit the key | `tests/unit/test_vendor_conf_single_writer.py` requires it to be the only conf writer; `tests/unit/test_conf_matches_codegen_routes.py` pins conf == generator route sets |
| 13 | `AGENTS.md` ("Default confs never route boxing" — allowed keys) | amend the allowed-key list in the same PR | documented policy, **review-enforced only**; a new key otherwise violates the stated rule |

If `flaggems_static` is only a *route* reusing an existing kernel set, the build
plumbing is untouched: `setup.py` `KERNEL_SWITCHES`/`KERNEL_SET_NAME`/
`_build_config.KERNELS`, the CMake `FLAGOS_BUILTIN_KERNELS` list and the
`torch_fl/_env.py` registry all stay as they are, and no new environment variable
is introduced. If it ships its own compiled library, all of those come in with it
and the wheel-compatibility guardrail (`torch_fl/compatibility.json`, preflight)
applies.
