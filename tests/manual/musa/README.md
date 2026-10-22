# MUSA Distributed Communication Testing Guide

## Overview

This directory contains manual tests for MUSA distributed communication via FlagCX. These tests verify that the identity view integration works correctly with real multi-GPU workloads.

## Prerequisites

1. **Hardware**: 2+ Moore Threads MUSA GPUs (e.g., MTT S5000)
2. **Software**:
   - MUSA toolkit (mudnn + musart runtime)
   - FlagCX built with MUSA adaptor support
   - torch_fl built with `FLAGOS_ACCELERATOR=musa`
   - transformers library (for Qwen3 tests)

## Building FlagCX with MUSA Support

```bash
# Clone FlagCX
git clone https://github.com/FlagOpen/FlagCX.git
cd FlagCX

# Build with MUSA adaptor
mkdir build && cd build
cmake .. -DADAPTOR_MUSA=ON
make -j$(nproc)

# The MUSA adaptor libraries will be in build/lib/
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
```

## Test Suite

### 1. Basic Collectives (`test_comm_musa.py`)

Tests all fundamental collective operations that DDP and FSDP2 depend on:
- `all_reduce` (DDP gradient sync)
- `broadcast` (parameter initialization)
- `all_gather` / `all_gather_into_tensor` (FSDP parameter gather)
- `reduce_scatter` / `reduce_scatter_tensor` (FSDP gradient reduction)
- `barrier` (synchronization)

**Run:**
```bash
LD_LIBRARY_PATH=<flagcx-build/lib> \
    python tests/manual/musa/test_comm_musa.py --world-size 2
```

**Expected output:**
```
[rank 0] inner backend = ProcessGroupFlagCX
[rank 0] all_reduce -> 3.0 (expect 3.0) OK
[rank 0] broadcast -> [0.0, 1.0, 2.0, 3.0] OK
[rank 0] all_gather -> [1.0, 2.0] OK
[rank 0] all_gather_into_tensor -> [0.0, 1.0] OK
[rank 0] reduce_scatter_tensor -> [0.0, 2.0] (expect [0.0, 2.0]) OK
[rank 0] barrier OK
[rank 0] result device flagos:0 OK
=== MUSA FlagCX collectives: all checks passed ===
```

### 2. DDP Training (`test_ddp_musa.py`)

Tests DistributedDataParallel with a 3-layer MLP. Verifies:
- DDP construction on flagos tensors
- Python reducer path activation
- Gradient synchronization across ranks
- Loss convergence

**Run:**
```bash
LD_LIBRARY_PATH=<flagcx-build/lib> \
    python tests/manual/musa/test_ddp_musa.py --world-size 2 --steps 5
```

**Expected output:**
```
[setup] world_size=2 backend=ProcessGroupFlagCX
[rank 0] DDP _use_python_reducer=True hooks=5
[step 0] loss=2.3456 grad_sum=['12.345', '12.345'] synced=OK
[step 1] loss=2.1234 grad_sum=['10.123', '10.123'] synced=OK
...
=== MUSA DDP: 5 steps, losses=['2.346', '2.123', ...], finite=OK decreasing=OK ===
```

### 3. FSDP2 Training (`test_fsdp2_musa.py`)

Tests PyTorch's FSDP2 (Fully Sharded Data Parallel) with a 3-layer MLP. Verifies:
- FSDP2 `fully_shard()` succeeds on flagos
- Forward/backward with sharded parameters
- Correct use of `_allgather_base` and `_reduce_scatter_base`
- Loss convergence

**Run:**
```bash
LD_LIBRARY_PATH=<flagcx-build/lib> \
    python tests/manual/musa/test_fsdp2_musa.py --world-size 2 --steps 10
```

**Expected output:**
```
[setup] world_size=2 backend=ProcessGroupFlagCX
[rank 0] FSDP2 model initialized, total params=332810
[step 0] loss(per-rank)=['2.4567', '2.3210']
[step 1] loss(per-rank)=['2.1234', '2.0987']
...
=== MUSA FSDP2: 10 steps, losses(rank0)=['2.457', '2.123', ...], finite=OK decreasing=OK ===
```

### 4. Qwen3 DDP (`test_qwen3_ddp_musa.py`)

End-to-end test with a real transformer model (Qwen3-0.6B). Verifies that DDP gradient synchronization works correctly with a production-scale model.

**Run:**
```bash
HF_HOME=<your-hf-cache-dir> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
LD_LIBRARY_PATH=<flagcx-build/lib> \
    python tests/manual/musa/test_qwen3_ddp_musa.py --world-size 2 --steps 5
```

**Expected output:**
```
[setup] world_size=2 backend=ProcessGroupFlagCX
[setup] loading Qwen3 model from Qwen/Qwen3-0.6B
[rank 0] DDP _use_python_reducer=True hooks=124
[step 0] loss=10.2345 grad_sum(per-rank)=['123.456', '123.456'] synced=OK
[step 1] loss=9.8765 grad_sum(per-rank)=['98.765', '98.765'] synced=OK
...
=== MUSA Qwen3 DDP: 5 steps, losses=['10.235', '9.877', ...], finite=OK ===
```

## Debugging

### FlagCX Initialization Failure

If you see:
```
RuntimeError: FlagCX init failed: no suitable inner backend
```

**Check:**
1. `LD_LIBRARY_PATH` includes FlagCX build/lib
2. `ldd` on the flagcx Python module shows libflagcx.so resolves
3. `GEMS_VENDOR=musa` is set (auto-detected from `torch_fl/lib/flagos_platform`)

### Device Type Mismatch

If FlagCX complains about device type:
```
RuntimeError: expected CUDA tensor but got PrivateUse1
```

This means the identity view path failed. **Check:**
1. `torch_fl/comm/process_group.py` has `"musa": _VendorProfile("musa", "_flagos_identity_view", None)`
2. `torch_fl._C._flagos_identity_view` is callable: `python -c "import torch_fl; print(hasattr(torch_fl._C, '_flagos_identity_view'))"`

### Stream/Device Index Issues

If you see GPU faults or "invalid resource handle":
```
VMFault at address 0x...
```

This likely means device index pinning failed. **Check:**
1. The test calls `torch_fl.flagos.set_device(rank)`, not just `torch.cuda.set_device(rank)`
2. Run `tests/manual/test_comm_device_index.py` first to verify device pinning works

## Performance Notes

- MUSA FlagCX → MCCL should achieve near-native bandwidth for large tensors (>1MB)
- Small tensor collectives may be slower than NCCL due to launch overhead
- Identity view has zero overhead (no storage conversion)

## Comparison with CUDA/MetaX

The MUSA tests mirror the structure of:
- `tests/manual/test_flagos_dist_live.py` (CUDA baseline)
- `tests/manual/metax/test_*_metax.py` (MetaX FlagCX tests)

Key differences:
- MUSA uses **identity view** (no storage conversion)
- CUDA/MetaX use **zero-copy cuda view** (storage reinterpretation)
- Both paths share the same FlagCX → vendor CCL backend logic
