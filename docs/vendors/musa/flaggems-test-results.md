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

**Results**: ✅ **104 passed, 1 skipped** (100% pass rate)

**Command**:
```bash
pytest tests/integration/ops/test_musa_dispatch.py -m musa -v
```

**Test Duration**: ~84 seconds

**Key Changes**:
- Updated test expectations to match FlagGems-first routing strategy
- 468 ops route to `flaggems` (FlagGems Triton kernels)
- 47 ops route to `musa` (mudnn native kernels): 36 that FlagGems does not cover at all, plus 11 of the 14 `NATIVE_TRITON_GAPS["musa"]` ops that FlagGems cannot execute correctly on this stack
- The remaining 3 gaps (`_conj`, `index_add`, `index_add_`) route to `none`: mudnn has no kernel for them either, so they reach ATen's CPU fallback instead of a registered-but-wrong dispatcher slot

**Routing Expectations**:
| Operator | Expected Backend | Reason |
|----------|------------------|--------|
| `mm` | `flaggems` | FlagGems coverage |
| `add.Tensor` | `musa` | FlagGems bf16 wrapped-number lowering fails |
| `mul.Tensor` | `musa` | mudnn native only |
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

### FlagGems random number generation

- **Issue**: `philox_backend_seed_offset` function expects 2 values but gets more
- **Impact**: Direct `flag_gems.enable()` + `torch.randn` on flagos device fails
- **Workaround**: `randn` and `randn_like` are in `NATIVE_TRITON_GAPS["musa"]`, so
  torch_fl routes them to the mudnn/muRAND native kernels; the FlagGems-side bug
  remains open upstream and is not worked around in this repository.
- **Status**: Bug in FlagGems master with torch 2.10 compatibility

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
test: update MUSA dispatch tests to accept FlagGems routing
```

**Changes**:
- Restored the MUSA `musa_flaggems_register.inc` generation (`scripts/codegen_musa_flaggems.py`),
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
| 2 | `test_musa_dispatch.py -m musa` | **104 passed, 1 skipped** (83.78s) |
| 3 | `test_factory_ops.py` | **46 passed** |
| 4 | `test_amp_contract.py -m amp` | **27 passed** |
| 5 | `test_math_bits_contract.py -m math_bits` | **12 passed** |
| 6 | `test_profiler_contract.py -m profiler` | **10 passed, 1 skipped, 1 xpassed** |
| 7 | `tests/integration/ops/` in a wheel-only workspace | **490 passed, 2 skipped, 512 deselected, 2 xfailed, 1 xpassed** (126.75s) |
| 8 | `test_rng_dispatch.py -m main_ops` | **80 passed, 37 deselected** |

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

- `scripts/codegen_mudnn.py` re-run produces byte-identical artifacts
  (`musa_flaggems_register.inc`, `musa_register.inc`, `musa_kernels.cc` all match
  on the second run).
- `scripts/codegen_musa_flaggems.py --check` reports "is up to date".
- `scripts/gen_vendor_confs.py --check` is clean for MUSA. It is stale for
  `backends_ascend.conf` and `backends_gcu.conf`; that drift predates this work
  and those platforms are out of scope here.

## Summary

✅ **Environment configured with flagtree + FlagGems master**  
✅ **All eight MUSA CI groups pass locally**  
✅ **FlagGems-first routing validated: 468 routed to FlagGems, 47 to mudnn native, 1521 to `none`**  
✅ **Every operator FlagGems cannot execute correctly on this stack falls back to mudnn, without patching FlagGems**

The MUSA platform integrates FlagGems as the primary execution backend, with
mudnn native kernels as the fallback for ops FlagGems does not cover and for the
ops measured to be incorrect or uncompilable on the mthreads Triton stack.
