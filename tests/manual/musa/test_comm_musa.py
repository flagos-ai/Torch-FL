# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Basic collective communication tests for MUSA via FlagCX identity view.

Tests all fundamental collectives that FSDP2 and DDP depend on:
  - all_reduce (DDP gradient sync)
  - broadcast (parameter initialization)
  - all_gather / all_gather_into_tensor (FSDP parameter gather)
  - reduce_scatter / reduce_scatter_tensor (FSDP gradient reduction)
  - barrier (synchronization)

Verifies:
  1. FlagCX (MUSA adaptor) init succeeds with identity view
  2. Tensors stay on flagos device through the round trip
  3. Numerical results match expected collective semantics
  4. Device index is correctly pinned (no cross-device stream hazards)

Usage:
    LD_LIBRARY_PATH=<flagcx build/lib> \
        python tests/manual/musa/test_comm_musa.py --world-size 2
"""

import argparse
import os

# torch_fl MUST be imported before torch (registers MUSA backend)
import torch_fl  # noqa: F401
import torch

try:
    import flagcx  # noqa: F401 -- self-registers "flagcx" backend (MUSA adaptor)
except ImportError:
    flagcx = None
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29618")

    dev = torch.device(f"flagos:{rank}")
    # MUSA: flagos:i is PrivateUse1 device i, shares physical GPU i
    torch_fl.flagos.set_device(rank)

    dist.init_process_group(backend="flagos", rank=rank, world_size=world_size)

    inner = type(getattr(dist.distributed_c10d._get_default_group(), "_inner", None))
    print(f"[rank {rank}] inner backend = {inner.__name__}", flush=True)

    failures = []

    # --- all_reduce (DDP gradient synchronization) ---
    t = torch.ones(4, device=dev, dtype=torch.float32) * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected_sum = float(sum(range(1, world_size + 1)))
    got = t.cpu()[0].item()
    ok_ar = abs(got - expected_sum) < 1e-5
    failures += [] if ok_ar else [f"all_reduce got {got} want {expected_sum}"]
    print(
        f"[rank {rank}] all_reduce -> {got} (expect {expected_sum}) "
        f"{'OK' if ok_ar else 'FAIL'}",
        flush=True,
    )

    # --- broadcast (parameter initialization) ---
    b = torch.arange(4, device=dev, dtype=torch.float32) + rank * 10
    dist.broadcast(b, src=0)
    expected_bc = list(range(4))
    ok_bc = b.cpu().tolist() == expected_bc
    failures += (
        [] if ok_bc else [f"broadcast got {b.cpu().tolist()} want {expected_bc}"]
    )
    print(
        f"[rank {rank}] broadcast -> {b.cpu().tolist()} {'OK' if ok_bc else 'FAIL'}",
        flush=True,
    )

    # --- all_gather (parameter gathering) ---
    src = torch.ones(2, device=dev, dtype=torch.float32) * (rank + 1)
    gathered = [torch.zeros(2, device=dev) for _ in range(world_size)]
    dist.all_gather(gathered, src)
    vals = [g[0].item() for g in gathered]
    expected_ag = [float(i + 1) for i in range(world_size)]
    ok_ag = vals == expected_ag
    failures += [] if ok_ag else [f"all_gather got {vals} want {expected_ag}"]
    print(f"[rank {rank}] all_gather -> {vals} {'OK' if ok_ag else 'FAIL'}", flush=True)

    # --- all_gather_into_tensor (_allgather_base: FSDP parameter gather) ---
    agb = torch.empty(world_size, device=dev, dtype=torch.float32)
    dist.all_gather_into_tensor(agb, torch.full((1,), float(rank), device=dev))
    expected_agb = [float(i) for i in range(world_size)]
    ok_agb = agb.cpu().tolist() == expected_agb
    failures += (
        []
        if ok_agb
        else [f"all_gather_into_tensor got {agb.cpu().tolist()} want {expected_agb}"]
    )
    print(
        f"[rank {rank}] all_gather_into_tensor -> {agb.cpu().tolist()} "
        f"{'OK' if ok_agb else 'FAIL'}",
        flush=True,
    )

    # --- reduce_scatter_tensor (_reduce_scatter_base: FSDP gradient reduction) ---
    # Each rank contributes the same arange, so rank r's output shard is
    # world_size * arange[2*r : 2*r+2]
    rs_in = torch.arange(2 * world_size, dtype=torch.float32, device=dev)
    rs_out = torch.empty(2, device=dev, dtype=torch.float32)
    dist.reduce_scatter_tensor(rs_out, rs_in)
    exp_rs = [float(world_size) * (2 * rank), float(world_size) * (2 * rank + 1)]
    ok_rs = rs_out.cpu().tolist() == exp_rs
    failures += (
        []
        if ok_rs
        else [f"reduce_scatter_tensor got {rs_out.cpu().tolist()} want {exp_rs}"]
    )
    print(
        f"[rank {rank}] reduce_scatter_tensor -> {rs_out.cpu().tolist()} "
        f"(expect {exp_rs}) {'OK' if ok_rs else 'FAIL'}",
        flush=True,
    )

    # --- barrier (synchronization primitive) ---
    dist.barrier()
    print(f"[rank {rank}] barrier OK", flush=True)

    # --- device consistency check ---
    # Result tensors must stay on flagos:rank, not leak to flagos:0 or CPU
    ok_dev = t.device.type == "flagos" and t.device.index == rank
    failures += [] if ok_dev else [f"result on {t.device}, want flagos:{rank}"]
    print(
        f"[rank {rank}] result device {t.device} {'OK' if ok_dev else 'FAIL'}",
        flush=True,
    )

    if failures:
        raise AssertionError(f"[rank {rank}] " + "; ".join(failures))

    dist.destroy_process_group()
    if rank == 0:
        print("=== MUSA FlagCX collectives: all checks passed ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    args = ap.parse_args()
    if torch_fl.flagos.device_count() < args.world_size:
        raise SystemExit(
            f"needs {args.world_size} flagos devices, "
            f"have {torch_fl.flagos.device_count()}"
        )
    mp.spawn(worker, args=(args.world_size,), nprocs=args.world_size, join=True)
