# Ascend NPU integration plan and operator coverage

> Drafted: 2026-07-20
> Machine: Ascend 910 (8×, CANN 9.0.0, aarch64)
> Target: **both inference and training**
> Primary approach: **aclnn codegen first**, with FlagGems/triton-ascend and a CPU fallback as backup

## 0. Conclusions up front

- The CUDA route — "load an external `libtorch_cuda.so` and get a free fallback" — **does not
  hold on NPU** (torch_npu and flagos both claim the PrivateUse1 key; see
  [external-libtorch-npu.md](external-libtorch-npu.md)). NPU must **provide its own operators**.
- The best foundation for those operators is the **public C ABI of CANN aclnn
  (`libopapi.so`)**. The existing `csrc/aten/backends/ascend/` (33 hand-written operators)
  already proves the path works.
- **The decisive current blocker**: the codegen cleanup in #10/`285f52c` removed the per-operator
  headers under `csrc/aten/*.h` and consolidated them into `csrc/aten/generated/ops.h`, but the
  33 hand-written ascend `.cc` files still `#include "../../mm.h"` and similar **headers that no
  longer exist**. **The Ascend backend does not compile on current main** — the first gate for
  any NPU work.
- Feasibility was verified on real hardware with a **standalone prototype**: the two-stage
  `aclnn<Op>GetWorkspaceSize + aclnn<Op>` call plus an `aclTensor` built over raw NPU storage;
  `aclnnSqrt` matched the CPU reference (`max_err=2.85e-07`). See §4.

## 1. Why aclnn codegen

On the CUDA side, `scripts/codegen/codegen_ops.py` generates 3429 boxing fallbacks
(`cuda_kernels.cc`) from `native_functions.yaml`: rewrite a flagos tensor's metadata, then
forward to `at::xxx` (the native CUDA kernel).

Ascend has no separate key to box into, so the counterpart is not boxing but **bulk generation
of aclnn call glue**:

```
aten op (schema)  --codegen-->  KernelAscend(...) { EXEC_ASCEND_CMD(aclnn<Op>, ...); }
                                └─ registered into the kAscend slot of the flagos internal Dispatcher
```

aclnn naming and calling conventions are highly regular:

- Naming: `aclnn` + the CamelCase operator name (`aclnnMm` / `aclnnAdd` / `aclnnCos` /
  `aclnnSqrt`, …).
- Calling: a uniform two-stage `xxxGetWorkspaceSize(inputs..., out, &ws, &exec)` +
  `xxx(ws_addr, ws, exec, stream)` — already abstracted by `EXEC_ASCEND_CMD`
  (`op_api_common.h`).

So elementwise, unary math, part of the reductions, and the matmul family can be generated in
bulk from an `aten→aclnn` mapping table plus per-category templates.

## 2. Existing assets

| Asset | Location | Status |
|---|---|---|
| aclnn call abstraction | `csrc/aten/backends/ascend/op_api_common.h` (`EXEC_ASCEND_CMD` / `AclTensorWrapper` / `AclScalarWrapper` / dtype mapping) | ✅ usable |
| Output tensor allocation | `op_preparation.h` (`apply_tensor_without_format` = `at::empty(device=PrivateUse1)`) | ✅ usable |
| Internal dispatcher | `csrc/aten/dispatcher.h` (`REGISTER_IMPL_TO_DISPATCHER(..., Backend::kAscend, ...)`) | ✅ usable |
| Hand-written operators | `backends/ascend/*.cc` (33: mm/bmm/add/mul/cat/embedding/softmax/sum/nll_loss/index/…) | ⚠️ dangling headers; needs fixing |
| Backend selection config | `torch_fl/configs/backends_ascend.conf` (per-op `flaggems\|ascend`) | ✅ usable |
| codegen framework | `scripts/codegen/codegen_ops.py` + `generated/name_map.json` (authoritative symbol naming) | ✅ skeleton is reusable |
| Runtime (stream/allocator/device) | `csrc/runtime/accelerator/ascend/` | ✅ exists |

## 3. Approach (layered; backend chosen per op via the conf)

Three capability tiers. `backends_ascend.conf` decides which tier each op takes; anything
uncovered falls back to CPU automatically:

1. **aclnn codegen (primary)** — covers the regular operators (elementwise, unary math,
   reductions, the matmul family). Goal: grow the hand-written 33 into the hundreds.
2. **FlagGems / triton-ascend (secondary)** — fused operators and hot spots where Triton both
   compiles and runs faster (the `_patch_flaggems_codegen_config` and `patch_triton_ascend.py`
   infrastructure already exists).
3. **CPU fallback** — the long tail and rarely used operators, explicitly flagged as known
   performance costs.

### Order of work

- **P0 (blocker): make the Ascend backend compile again.** Resolve the dangling per-operator
  headers referenced by the 33 `.cc` files. Two options:
  - (a) have codegen also emit per-operator headers for ascend (resurrect `csrc/aten/*.h`); or
  - (b) change those `.cc` files to a uniform `#include "generated/ops.h"` (better fit for the
    single-header structure after #10; recommended).
  - First get the Ascend backend compiling on current main, `import torch_fl` clean, and the
    33 operators passing regression, as the baseline.
- **P1: aclnn codegen MVP.** Start with the most regular category — one-input/one-output
  elementwise and unary math (sqrt/exp/reciprocal/sigmoid/tanh/floor/ceil/sign/gelu/…). Build
  the `aten→aclnn` mapping table plus one category template, emitting into
  `backends/ascend/generated/`. For this category `AclTensorWrapper(in)/(out)` +
  `EXEC_ASCEND_CMD(aclnn<Op>, in, out)` suffices — lowest risk.
- **P2: more categories.** Binary (add/mul/sub/div — handling broadcast, alpha, and dtype
  promotion; see the existing hand-written `add.cc`), reductions (sum/mean/max — handling
  dim/keepdim), matmul (mm/bmm — cube_math_type). Long-tail ops whose aclnn names are irregular
  or that need special arguments go on a skip list and fall back.
- **P3: training operators.** The backward family (silu_backward / embedding_dense_backward /
  nll_loss_backward are already hand-written; fill in relu/gelu/norm and others), and evaluate
  aclnnForeach* coverage for the optimizer foreach operators.

### Relationship to the CUDA codegen

- Reuse `codegen_ops.py`'s schema parsing (`native_functions.yaml` → signature / category /
  `fn_type` / `dispatcher`) and `name_map.json` as the naming authority.
- **Add** an ascend-specific emitter: it emits an aclnn body rather than a boxing body, and its
  input is an `aten→aclnn` mapping table (a new file, e.g. `torch_fl/ascend_aclnn_map.json`)
  rather than `backends_cuda.conf`.
- Output lands in `csrc/aten/backends/ascend/generated/`, picked up by the `ASCEND_KERNEL` glob
  in `csrc/CMakeLists.txt`.

## 4. Feasibility verification (passed on real hardware)

A standalone prototype (not depending on the torch_fl build, so it sidesteps the P0 blocker)
proved that the **kernel body shape codegen will emit** computes correctly on hardware:

- The prototype source and build script have been deleted (no longer needed once engineered);
  to review them, see `docs/ascend_aclnn_codegen_prototype.cc` and
  `docs/build_ascend_prototype.sh` in git history.
- Method: raw `aclrtMalloc` device memory → `aclCreateTensor` (same as `AclTensorWrapper`) →
  two-stage `aclnnSqrtGetWorkspaceSize` + `aclnnSqrt` (same as `EXEC_ASCEND_CMD`) → copy back
  and verify.
- Result:

  ```
  aclnnSqrt sample: in[3]=4.0 out[3]=2.000000 (ref=2.000000)  max_err=2.850e-07
  PASS: aclnnSqrt on NPU matches CPU reference
  ```

**Conclusion**: the two-stage aclnn call over raw storage aclTensors is a **viable codegen body
shape**. What remains is engineering — the mapping table, the category templates, the P0 compile
fix — with no unresolved low-level uncertainty.

## 5. Risks and costs

- ⚠️ **The P0 compile fix is a hard prerequisite**; without it no NPU operator can be verified.
- ⚠️ Irregularity in the long tail of aclnn names and signatures: codegen absorbs 60–80% of the
  regular operators; the tail still needs hand-writing or fallback, so a skip list must be
  maintained (analogous to `codegen_skip_ops.txt`).
- ⚠️ Broadcast, dtype promotion, alpha, and dim/keepdim semantics for the binary and reduction
  categories must be handled correctly in the templates (the existing hand-written `add.cc` /
  `sum.cc` / `mean.cc` are the reference samples).
- ⚠️ Tight coupling to the CANN version (aclnn interfaces evolve with CANN); a CANN upgrade
  requires a regression pass.

## 6. Next steps (after this round)

1. Decide the P0 fix direction (recommend (b), the unified `generated/ops.h`) and get the
   Ascend baseline running.
2. Build `ascend_aclnn_map.json` plus the P1 unary elementwise category template; generate and
   regress.
3. Progressively extend categories through P2/P3; long-tail ops go on the skip list and fall
   back.
