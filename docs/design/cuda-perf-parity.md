# Aligning the FlagGems-primary CUDA build with torch-cuda

## Problem

On the OCR-RM Qwen3.5-VL 2B value head (12 samples, bf16, sdpa), steady median
forward on 8xH100:

| build | steady median | peak mem |
| --- | --- | --- |
| torch-cuda | **60 ms** | — |
| torch-fl, boxing (`FLAGOS_BUILD_BOXING=ON`) | **316 ms** | — |
| torch-fl, FlagGems-primary (default) | **3391 ms** | 5166 MB |

The default build is 56x slower than torch-cuda. `memcpy_share` is **49.3%**; the
forward issues **8826 `cudaMemcpy`**; 71 ops reach `cpu_fallback`.

## Root cause

`torch_fl/configs/backends_nvidia.conf` holds 2039 routes: **1630 to `cuda`**, 353 to
`flaggems`. The `cuda` route is how the conf says "run ATen's CUDA kernel here",
and on a boxing build it does exactly that — `DeviceBoxingGuard` retags to
`DispatchKey::CUDA` and ATen's kernel runs.

On the default build the boxing kernels are not compiled (`csrc/CMakeLists.txt`
excludes `aten/generated/cuda_kernels.cc`), so **1630 of 2039 routes (80%)
resolve to nothing** and fall through to `cpu_fallback`, paying a host round trip
per call. `mul` alone: 1522 calls, 8 `cudaMemcpy` each.

Measured contrast, same shapes, same device:

| op | PrivateUse1 kernel present | memcpy per call |
| --- | --- | --- |
| `add.Tensor` | yes | **0** |
| `mul.Tensor` | no | **8** |
| `cat` | no | **8** |

FlagGems 5.4.0 executes `mul`, `sum`, `softmax` correctly on `flagos:0` today. They
are simply not routed there.

Verified end to end, same tensors, `cudaMemcpy` counted over a profiler window:

| call | memcpy | equals the falling-back result |
| --- | --- | --- |
| `torch.mul(a, b)` | 8 | — |
| `flag_gems.mul(a, b)` | **0** | yes |
| `torch.cat([a, b], 1)` | 8 | — |
| `flag_gems.cat([a, b], 1)` | **0** | yes |

Routing these two ops to FlagGems is worth ~13900 `cudaMemcpy` per forward
(`mul` 1522 calls, `cat` 220) before anything else changes.

## Design rule

> **Default every op to FlagGems. Where FlagGems has no implementation, add a
> native CUDA kernel. Switch an op between the two on measured performance. Never
> fall back to the CPU.**

Three properties, in priority order:

1. **No CPU fallback, ever.** An op that cannot be served on device is a build
   failure, not a runtime round trip. This is the invariant the current build
   violates 71 times.
2. **FlagGems is the default.** It is the portable path, it is already installed
   and working, and it is where the project's operator development goes.
3. **Native CUDA kernels are the exception, admitted by measurement.** Only for
   ops FlagGems cannot do, or where the native kernel is measurably faster at the
   shapes that matter. Recorded per op with its evidence.

### Why native kernels are not the primary path

Two kernels were implemented directly against cuBLAS on
`feat/cuda-native-cublas` — correct to rel ≤ 4e-07, 47 tests passing. At the
model's real shapes they do not pay:

| op (model shape) | CPU | native GPU | verdict |
| --- | --- | --- | --- |
| `addmm` `[640,1024,3072]` | 0.28 ms | 0.57 ms | CPU wins |
| `conv1d` depthwise | 0.18 ms | 0.30 ms | CPU wins |
| `cat` | 0.05 ms | 0.24 ms | CPU wins |
| `index_select` `[2]` | 0.003 ms | 0.040 ms | CPU wins |
| `conv3d` patch-embed | 7.54 ms | 0.58 ms | **GPU 12.9x** |
| `conv3d` depthwise | 2.69 ms | 0.29 ms | **GPU 9.3x** |

Hand-writing 1630 kernels would be the wrong answer even if it were feasible: the
gap is 80% of the route table, not a handful of missed ops. Native kernels earn
their place at the two `conv3d` shapes and (with a batched launcher) at
`solve_triangular`.

## Implementation

### Gate 1 — No empty route can ship

The regression that caused this is a conf line naming a backend the build does not
contain, discovered only as a 49% memcpy share in production. It needs a guard:

* A test walks the shipped conf against the compiled kernel set and **fails on any
  route that resolves to nothing**.
* Codegen resolves each route at generation time against the build's kernel set,
  so the shipped conf cannot contain a dead `cuda` line. A `cuda` route that has
  no native kernel and no boxing kernel is rewritten to `flaggems`, or the op is
  reported as unserved.

### Gate 2 — Default to FlagGems

Flip the default in `scripts/codegen/codegen_ops.py`. Today a route is `cuda`
unless something forces it elsewhere; it becomes `flaggems` unless one of these
holds:

* **No FlagGems implementation** (the `FLAGGEMS_PYTHON_OPS` coverage set says so)
  → native or unserved, never silently CPU.
* **FlagGems known-broken** (`flaggems_runtime_broken`: `index_select` fails on
  stream ordering, issue #120 — HF beam search produced NaN) → respect the
  exclusion rather than resweep it.
* **FlagGems recursive fallback** (`flaggems_recursive_fallback`) → same.
* **Contract exception** (`_conj` must stay a lazy view) → stays.
* **Measured native win** → route to the native kernel, with the measurement
  recorded beside the exception.

The 1630 `= cuda` lines are the input to codegen, not its output, so this changes
what the conf *means* rather than hand-editing it.

### Gate 3 — Fill the gaps with native kernels

For each op that Gate 2 reports as unserved, in descending order of measured cost
in the forward:

| priority | op | calls/fwd | current | plan |
| --- | --- | --- | --- | --- |
| 1 | `mul.Tensor` | 1522 | cpu_fallback, 8 memcpy/call | FlagGems (works today) |
| 2 | `cat` | 220 | cpu_fallback, 8 memcpy/call | FlagGems (works today) |
| 3 | `add` / `sub` / `where` | 717 | mixed | FlagGems |
| 4 | `conv3d` | 20 | cpu_fallback | native cuDNN, 12.9x at model shape |
| 5 | `solve_triangular` | 36 | native, 0.867 ms vs 2.085 ms CPU | batched launcher |
| — | `index_select` | 6 | cpu_fallback | blocked by #120; keep CPU-excluded and fix the stream bug |

`addmm` stays unflagged: at the model's shape the CPU wins, so a native kernel
there would be a regression with a maintenance bill.

### Gate 4 — Switch on measured performance

Routing is decided by measurement, per op, recorded in the conf with its evidence
and date. The measurement harness exists:

```bash
python bench_value.py --backend flagos --device flagos:0 \
  --demo <demo> --checkpoint <ckpt> --out out.jsonl --iters 3   # e2e
python audit_fallback.py --backend flagos --demo <demo> \
  --checkpoint <ckpt> --out audit.json                          # per-op
```

## Verification

1. **Accuracy must not move**: 12/12 samples, max|Δ| ≤ 5e-03 vs the torch-cpu
   reference (currently 12/12, 4.86e-03).
2. **`memcpy_share_pct` must fall** at each gate; 49.3% now.
3. **Zero CPU fallbacks**: `ops_cpu_fallback == 0` in the audit. This is the
   primary gate — the others are consequences.
4. **Steady median**: interim checkpoint is boxing's 316 ms; the target is
   torch-cuda's 60 ms.
5. **Codegen idempotency**: a second regeneration produces an empty diff.

## Risks

* **FlagGems correctness on this device.** Every newly routed op needs the
  accuracy gate, not a spot check. `index_select` is the worked example of a
  FlagGems path that looks fine and is not.
* **The audit is a profiler sample, not a census.** Figures above come from one
  forward on one machine and must be regenerated for whatever hardware a claim is
  made on.
* **Native-kernel maintenance.** Each one is a per-op commitment; that is why the
  bar for admitting one is a measured win at a real shape.

## Status

Nothing in this document is implemented. `feat/cuda-native-cublas` holds the
phase-3 groundwork (cuBLAS handle, device guard, stream accessors, two correct
kernels, 47 passing tests); of those two, `addmm` should be left unregistered
because it loses to the CPU at model shapes.