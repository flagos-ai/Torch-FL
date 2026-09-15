# MUSA FlagGems Integration Test Results

Measured on an eight-device MTT S5000 host. Installation and configuration are
covered in [flaggems-setup.md](flaggems-setup.md).

## Environment Setup

### Python Environment
- **Environment**: conda (musa_test)
- **Python Version**: 3.10.21
- **Location**: `/publi-flash/lvyufeng/env/miniconda3/envs/musa_test`

### Package Versions
- **torch**: 2.10.0+cpu
- **torch_fl**: 0.1.0 (rebuilt for Python 3.10 with MUSA support)
- **flagtree**: 0.6.2a3+mthreads3.6 (provides triton 3.6.0 with mthreads backend)
- **flag_gems**: 5.4.0rc2.post1+g4d9c34775 (master branch, editable install from `/tmp/FlagGems`)
- **triton**: 3.6.0 (bundled by flagtree, with the `mthreads` backend)

### Installation Steps

1. **Create conda environment**:
   ```bash
   conda create -n musa_test python=3.10 -y
   conda activate musa_test
   ```

2. **Install PyTorch**:
   ```bash
   pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install flagtree** (provides mthreads-enabled triton):
   ```bash
   pip install 'flagtree===0.6.2a3+mthreads3.6' \
     --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple
   ```

4. **Build and install torch_fl**:
   ```bash
   ACCELERATOR=musa python -m build --wheel --no-isolation
   pip install dist/torch_fl-0.1.0-cp310-cp310-linux_x86_64.whl
   ```

5. **Install FlagGems master** (without dependencies to preserve flagtree's triton):
   ```bash
   cd /tmp/FlagGems
   pip install --no-deps -e .
   ```

### Environment Variables
```bash
export MUSA_HOME=/usr/local/musa
export ACCELERATOR=musa MUSA_KERNEL=1 FLAGGEMS_PYTHON=1 FLAGGEMS_KERNEL=0
export LD_LIBRARY_PATH=/publi-flash/lvyufeng/env/miniconda3/envs/musa_test/lib:/usr/local/musa/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/publi-flash/lvyufeng/PyTorch-Plugin-FL
```

Rebuild the in-place extension after any change under `csrc/` or `scripts/`:

```bash
cd /publi-flash/lvyufeng/PyTorch-Plugin-FL
/publi-flash/lvyufeng/env/miniconda3/envs/musa_test/bin/python setup.py build_ext --inplace
```
(the interpreter under `/publi-flash/lvyufeng/env/miniconda3/envs/musa_test` is the one with flagtree + FlagGems)

## Test Results

### MUSA Dispatch Tests
**File**: `tests/integration/ops/test_musa_dispatch.py`

**Results**: ✅ **113 passed** (100% pass rate)

**Command**:
```bash
pytest tests/integration/ops/test_musa_dispatch.py -m musa -v
```

**Test Duration**: ~185 seconds

**Key Changes**:
- Updated test expectations to match FlagGems-first routing strategy
- 468 ops route to `flaggems` (FlagGems Triton kernels)
- 49 ops route to `musa` (mudnn native kernels): 36 that FlagGems does not
  cover at all, plus 13 of the 14 `NATIVE_TRITON_GAPS["musa"]` ops that FlagGems
  cannot execute correctly on this stack
- The remaining gap (`_conj`) routes to `none`: it is a contract entry — ATen's
  `conj` is a lazy view that must set the Conjugate bit, FlagGems materializes it,
  and mudnn has no Conjugate-bit path either, so leaving the op unregistered and
  letting ATen's composite implement it is the correct route

**Routing Expectations**:
| Operator | Expected Backend | Reason |
|----------|------------------|--------|
| `mm` | `flaggems` | FlagGems coverage |
| `add.Tensor` | `musa` | FlagGems bf16 wrapped-number lowering fails |
| `mul.Tensor` | `musa` | mudnn native only |
| `div.Tensor` | `musa` | FlagGems bf16 wrapped-number lowering fails |
| `div.Tensor_mode` | `musa` | FlagGems loses the trailing store on integer inputs |
| `floor_divide` | `musa` | FlagGems loses the trailing store on integer inputs |
| `_softmax` | `flaggems` | FlagGems coverage |
| `relu` | `flaggems` | FlagGems coverage |
| `mm.out` | `flaggems` | FlagGems coverage |

### Factory Ops Tests
**File**: `tests/integration/test_factory_ops.py`

**Results**: ✅ **46 passed** (100% pass rate)

**Command**:
```bash
pytest tests/integration/test_factory_ops.py -v
```

**Test Duration**: ~11 seconds

## Test Coverage by Category

### TestMusaDispatch (7 tests)
- ✅ dispatch routing to FlagGems/mudnn backends
- ✅ environment variable overrides (`FLAGOS_OP_*`)
- ✅ dispatch logging verification

### TestMusaEmptyInplace (5 tests)
- ✅ empty tensor fill operations
- ✅ zero-element tensor handling

### TestMusaCorrectness (62 tests)
- ✅ CPU reference matching for all ops
- ✅ broadcast operations
- ✅ strided tensor copies
- ✅ dtype casting
- ✅ reduce operations

### TestMusaConvolution (14 tests)
- ✅ 2D convolution forward/backward
- ✅ grouped convolutions
- ✅ various kernel sizes and strides
- ✅ 3D convolution CPU fallback

### TestMusaMixedDeviceOperandOrder (9 tests)
- ✅ CPU/device tensor mixing
- ✅ scalar operations

### TestMusaDegenerateStride (3 tests)
- ✅ transposed tensors with size-1 dimensions
- ⏭️ BERT regression head (skipped)

### TestMusaAutograd (2 tests)
- ✅ backward pass correctness
- ✅ gradient preservation across device transfers

## Known Issues

### Integer division dropped the trailing element, and `int64 / int64` raised

[Issue #266](https://github.com/flagos-ai/Torch-FL/issues/266). Two independent
defects, both fixed in the platform code generator rather than with handwritten
kernels:

- **FlagGems integer floor division loses its final store on this stack.** On
  integer inputs the kernel writes a wrong last element whenever `numel` is not a
  power of two — wrong at `n = 3, 5, 6, 7, 9, 15, 17, 31, 33, 100`, correct at
  `n = 1, 2, 4, 8, 16, 32, 64, 1024`. Float inputs are correct. This reached
  `a // b`, `a // 2`, `torch.floor_divide(a, b)`, `a.clone().floor_divide_(b)`,
  and the rounding-mode division overloads.
- **mudnn's `TRUEDIV` has no integer overload.** `a / b`, `torch.div(a, b)`, and
  `a.div_(b)` on integer tensors failed with
  `Unsupported binary mode: TRUEDIV, with left data type: INT64`, because the
  generated kernel took `result_dtype` from `at::result_type(self, other)` —
  which is the integer type itself — while ATen's true division promotes integer
  inputs to `float32` through `promote_integer_inputs_to_float`.

- **Resolution**: `div.Tensor_mode`, `div_.Tensor_mode`, `floor_divide`, and
  `floor_divide_.Tensor` are listed in `NATIVE_TRITON_GAPS["musa"]` and route to
  mudnn (`FLOORDIV` / `TRUNCATEDIV` / `TRUEDIV`, selected at run time from ATen's
  `rounding_mode` string). The true-division kernels widen an integral result to
  `at::get_default_dtype_as_scalartype()`, guarded on the absence of a
  `rounding_mode` so `'floor'` and `'trunc'` keep `int64`. FlagGems is not
  patched.
- **Measured**: a 59-case CPU-parity probe over out-of-place, in-place, scalar
  and tensor operands, both rounding modes, negative operands, `out=`,
  broadcasting, and `floor_divide` at `n = 2, 3, 4, 5, 7, 8, 15, 17, 33, 100`
  was run against both this tree and a second worktree built at the base commit
  (`flagos/main`, `6b978c0`):

  | Tree | Result |
  |---|---|
  | before (`6b978c0`) | `39 exact, 7 float-approximate, 0 error-text matches, 13 mismatches (of 59)` |
  | after (this branch) | `43 exact, 14 float-approximate, 1 error-text match, 1 mismatch (of 59)` |

  The 13 base-tree mismatches are exactly the two defects above. On the fixed
  tree every integer floor-division and `rounding_mode` case is exact, and the
  in-place `int64` true-division case `a.div_(b)` keeps ATen's own
  `result type Float can't be cast to the desired output type Long`,
  byte-identical to CPU. Two qualifications, both measured: the 14
  float-approximate cases are `truediv` results differing from CPU by exactly one
  float32 ULP (`5.960e-08`) at `n = 5, 7, 8, 15, 17, 33, 100`, and the pure-float
  spellings — which never touched the FlagGems floor-divide kernel — show the
  identical `5.960e-08` on the base tree, so this is mudnn `TRUEDIV` arithmetic
  versus CPU libm and predates the change; the remaining mismatch is the probe's
  own `out=` helper passing CPU tensors to a Triton path and raising identically
  on both trees, not a property of the operators. Full CI manifest re-run: see
  the table below.
- **The two fixes are independent, and the routing one is causal.** Forcing the
  four rerouted ops back onto FlagGems with `FLAGOS_OP_div__Tensor_mode=flaggems
  FLAGOS_OP_div___Tensor_mode=flaggems FLAGOS_OP_floor_divide=flaggems
  FLAGOS_OP_floor_divide___Tensor=flaggems` reproduces the tail loss exactly and
  nothing else: at `n = 3` with `a = [10, 20, 30]`, `b = [2, 4, 5]`, `a // b`,
  `a // 2`, `torch.floor_divide(a, b)`, `torch.div(a, b, rounding_mode='floor')`,
  `'trunc'`, `a.clone().floor_divide_(b)` and `a.clone().div_(b, rounding_mode=
  'floor')` all return `[5, 5, 0]` where CPU returns `[5, 5, 6]`, while
  `a / b` and `torch.div(a, b)` (true division, which never used the FlagGems
  kernel) stay correct. The same 10 cases are all correct on the shipped route.

### FlagGems bf16 wrapped-number promotion reaches an unsupported LLVM intrinsic

- **Issue**: ATen boxes a Python-float operand into a float64 0-dim tensor
  (`is_wrapped_number`), and FlagGems' pointwise promotion does not honour that
  flag, so it promotes the result to fp64. mthreads' LLVM lowering declares
  `llvm.musa.float2bfloat16(float)` with no double overload, so the kernel fails
  to compile for **bf16 only**.
- **Reproduced locally, and only for the FlagGems route**:
  ```
  $ FLAGOS_OP_add__Tensor=flaggems pytest tests/integration/test_amp_contract.py -m amp
  F  RuntimeError: failed to translate module to LLVM IR
     error: intrinsic call operand #0 has type double but
            "llvm.musa.float2bfloat16" expects float
     ... pointwise_dynamic_*_add_func_kernel_rank_1_bptr_t1024.py:102:36
  1 failed, 26 passed
  ```
  With the shipped route (`add.Tensor = musa`) the same file is **27 passed**.
- **Impact**: every `add`/`sub`/`div` entry point that ends up on the `.Tensor`
  overload in bf16 — including `add_.Scalar`, `_foreach_add_`, and therefore
  AdamW's foreach step. `test_autocast_fp32_policy[dtype1]` was the CI-visible case.
- **Resolution**: those ops are listed in `NATIVE_TRITON_GAPS["musa"]` and route
  to the mudnn native kernel. FlagGems is not patched.

### FlagGems random number generation — resolved

- **Former issue**: `philox_backend_seed_offset` function expects 2 values but
  gets more, so direct `flag_gems.enable()` + `torch.randn` on the flagos device
  crashed. `randn` and `randn_like` were consequently listed in
  `NATIVE_TRITON_GAPS["musa"]` while the FlagGems-side bug stayed open upstream.
- **Re-measured 2026-09-15, no longer reproduces**: with FlagGems `4d9c34775` and
  flagtree `0.6.2a3+mthreads3.6`, `torch.randn` on `flagos:0` is finite,
  reproducible from `torch.manual_seed`, sensitive to a changed seed over 65536
  samples, and correct in float32/float16/bfloat16 (`std ≈ 0.97`); `randn_like`
  inherits shape and dtype. FlagGems now runs off the Philox bridge torch_fl
  installs for MUSA, which is exactly the generator state the old crash was about.
- **Status**: both ops were removed from `NATIVE_TRITON_GAPS["musa"]` and route to
  `flaggems`, with the mudnn/muRAND native kernels retained as the fallback
  (`flaggems  # musa`).

### Triton Backend Warning
```
RuntimeWarning: active Triton backend does not provide a replay benchmarker; 
falling back to event timing
```
- **Impact**: None on correctness, only affects performance measurement
- **Reason**: mthreads backend doesn't implement replay benchmarking API

## Commit History

### Latest commit on this branch

```
fix: promote four MUSA FlagGems gap entries back to FlagGems
```

**Changes**:
- Re-measured every `NATIVE_TRITON_GAPS["musa"]` entry on its recorded failure
  signature against the FlagGems revision the MUSA CI job installs (`4d9c34775`)
  and against the CI pin. Four entries no longer reproduce and leave the set:
  `index_add` and `index_add_` (recorded as "returns all zeros instead of
  accumulating") now route to `flaggems` from `none`, and `randn`/`randn_like`
  (recorded as "crashes unpacking generator state") route to `flaggems` with the
  mudnn/muRAND kernel retained as a fallback
- MUSA routing: 468 `flaggems`, 49 `musa`, 1519 `none`
- The other fourteen entries keep their routes; each was re-probed and reproduces
  its signature exactly, so the provenance comment in
  `scripts/codegen/gen_vendor_confs.py` is updated to the current revision

### Previous commit

```
fix: route MUSA integer division through mudnn and promote true division
```

**Changes**:
- Fixed issue #266 in the generator: integer true division now computes in
  `float32` per ATen's `promote_integer_inputs_to_float` contract instead of
  handing `int64` to mudnn's `TRUEDIV`, and the four integer floor-division
  overloads route to mudnn instead of the FlagGems kernel that loses its
  trailing store
- MUSA routing at that commit: 464 `flaggems`, 51 `musa`, 1521 `none`

### Earlier commit

```
test: update MUSA dispatch tests to accept FlagGems routing
```

**Changes**:
- Restored the MUSA `musa_flaggems_register.inc` generation (`scripts/codegen/codegen_musa_flaggems.py`),
  registering all 482 FlagGems Python ops on PrivateUse1 for MUSA
- Updated test expectations to match FlagGems-first routing: 468 ops route to
  `flaggems`, 47 to `musa`, 1521 to `none`
- Moved the ops FlagGems cannot execute correctly on the mthreads stack into
  `NATIVE_TRITON_GAPS["musa"]`, including the four in-place arithmetic overloads
  that ATen's wrapped-number boxing (`add_.Scalar` -> `add_.Tensor`,
  `_foreach_add_`, AdamW) reaches

## Running Tests

### Quick Test (dispatch only)
```bash
conda activate musa_test
export LD_LIBRARY_PATH=/publi-flash/lvyufeng/env/miniconda3/envs/musa_test/lib:$LD_LIBRARY_PATH
pytest tests/integration/ops/test_musa_dispatch.py::TestMusaDispatch -m musa -v
```

### Full MUSA Suite
```bash
conda activate musa_test
export LD_LIBRARY_PATH=/publi-flash/lvyufeng/env/miniconda3/envs/musa_test/lib:$LD_LIBRARY_PATH
pytest tests/integration/ops/test_musa_dispatch.py -m musa -v
```

### All Integration Tests
```bash
conda activate musa_test
export LD_LIBRARY_PATH=/publi-flash/lvyufeng/env/miniconda3/envs/musa_test/lib:$LD_LIBRARY_PATH
pytest tests/integration/ -v
```

## Configuration

### Backend Priority (backends_musa.conf)
Current MUSA platform uses FlagGems-first routing:
1. **flaggems_cpp** - FlagGems C++ runtime (if available)
2. **flaggems** - FlagGems Python/Triton kernels ← **PREFERRED**
3. **tileops** - TileOPs Triton shims (SM90+ only)
4. **musa** - mudnn native kernels
5. **none** - CPU fallback

### Runtime Overrides
Override individual ops:
```bash
FLAGOS_OP_mm=musa python script.py  # Force mm to use mudnn
```

Force all ops to vendor backend (where available):
```bash
ALL_USE_VENDOR=1 python script.py
```

Force all ops to FlagGems (where available):
```bash
ALL_USE_FLAGGEMS=1 python script.py
```

Enable dispatch logging:
```bash
FLAGOS_LOG_DISPATCH=1 python script.py
```

## Full CI manifest reproduction (local)

Every group in `.github/configs/musa.yml` was run locally on the eight-device
MTT S5000 host against the working tree, in the same order the CI manifest runs
them:

| # | CI group | Local result |
|---|---|---|
| 1 | Isolated MUSA environment preflight | OK — CPU PyTorch 2.10.0+cpu, MUSA devices: 8 |
| 2 | `test_musa_dispatch.py -m musa` | **113 passed** (185.29s) |
| 3 | `test_factory_ops.py` | **46 passed** |
| 4 | `test_amp_contract.py -m amp` | **27 passed** (not re-run for the gap promotion) |
| 5 | `test_math_bits_contract.py -m math_bits` | **12 passed** (not re-run for the gap promotion) |
| 6 | `test_profiler_contract.py -m profiler` | **10 passed, 1 skipped, 1 xpassed** (not re-run for the gap promotion) |
| 7 | `tests/integration/ops/` in a wheel-only workspace | **493 passed, 1 skipped, 521 deselected, 2 xfailed, 1 xpassed** (182.27s), plus the 3 pre-existing failures below |
| 8 | `test_rng_dispatch.py -m main_ops` | **80 passed, 37 deselected** |

Groups 2, 3, 7 and 8 were re-run for the FlagGems gap promotion; groups 4, 5 and 6
do not exercise the promoted ops and their numbers are from the integer-division
change on the same tree.

Group 7's three failures are `test_flaggems_conf_consistency.py`
(`test_every_conf_op_maps_to_a_dispatcher`, `test_no_orphan_flagos_python_kernels`,
`test_counts_match`), which report `mm`/`bmm`/`addmm` dispatcher drift in
`csrc/aten/generated/`. They reproduce byte-identically against the pristine
`backends_musa.conf` on the same tree, so they are pre-existing and unrelated to
integer division or to the promoted routes.

Group 8 needs the manifest's own `-k` filter. Without it the suite reports 8
`TestRngSeedSource` failures on this host, all from the runner's broken muRAND:
`torch.empty(8).normal_()` raises `murandGenerateNormal failed: LAUNCH_FAILURE`,
the generator path raises `murand normal stream sync failed`, and `randperm`
raises err 700 (illegal memory access). The device RNG itself is healthy
(`torch.randn(1024, device='flagos:0')` has `std ≈ 1.0` on all 8 devices), the
same 8 failures appear with the previous conf stashed in, and the upstream MUSA CI
job reports the identical "80 passed, 37 deselected" on the same image. The
manifest already documents this as a driver/toolkit/image regression rather than a
`torch_fl` defect.

Group 7 is run with `FLAGOS_USE_FLAGGEMS=1`, matching
`.github/scripts/set_env_musa.sh`: the group's marker expression does not
exclude the `flaggems` marker, and `tests/integration/ops/conftest.py` gates
`flaggems`-marked tests on that variable, so a run without it skips them
instead of exercising them. `PYTHONPATH` must point at this working tree —
the conda environment also carries a non-editable wheel install of `torch_fl`
from an earlier tree, and without the override `import torch_fl` resolves to
that install and reads its stale `backends_musa.conf`.

Unit tests: `tests/unit/test_gen_vendor_confs.py` — **34 passed, 1 failed**. The
one failure is `test_shipped_confs_are_up_to_date`, which reports
`stale: ['backends_gcu.conf', 'backends_ascend.conf']`; it is pre-existing
generator drift on the two out-of-scope platforms, and the MUSA conf it also
checks is byte-identical to a fresh generation.

### Generator idempotency

- `scripts/codegen/codegen_mudnn.py` re-run produces byte-identical artifacts
  (`musa_flaggems_register.inc`, `musa_register.inc`, `musa_kernels.cc` all match
  on the second run).
- `scripts/codegen/codegen_musa_flaggems.py --check` reports "is up to date".
- `scripts/codegen/gen_vendor_confs.py --check` is clean for MUSA. It is stale for
  `backends_ascend.conf` and `backends_gcu.conf`; that drift predates this work
  and those platforms are out of scope here.

Order matters between the first and third: `gen_vendor_confs.py` reads the
on-disk `musa_flaggems_register.inc` to decide which ops the platform registers,
so `codegen_musa_flaggems.py` must run **before** it. Running them the other way
round leaves `index_add` and `index_add_` at `none`.

## Summary

✅ **Environment configured with flagtree + FlagGems master**  
✅ **All eight MUSA CI groups run locally; group 7 carries the three pre-existing `test_flaggems_conf_consistency.py` failures, and group 8 needs the manifest's `-k` filter for the broken-runner muRAND condition**  
✅ **FlagGems-first routing validated: 468 routed to FlagGems, 49 to mudnn native, 1519 to `none`**  
✅ **Every operator FlagGems cannot execute correctly on this stack falls back to mudnn, without patching FlagGems**

The MUSA platform integrates FlagGems as the primary execution backend, with
mudnn native kernels as the fallback for ops FlagGems does not cover and for the
ops measured to be incorrect or uncompilable on the mthreads Triton stack.
