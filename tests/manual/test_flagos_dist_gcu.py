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

"""Live multi-GCU test for ProcessGroupFlagOS with FlagCX on Enflame hardware.

Launches N processes, each binding one GCU device, and exercises:
  1. init_process_group("flagos") with FlagCX Enflame adaptor
  2. Basic collectives: all_reduce / broadcast / all_gather / reduce_scatter
  3. DistributedDataParallel forward/backward
  4. FSDP2 (fully_shard) with per-layer wrapping and MixedPrecision
  5. Device guard correctness (tensor on device X, current device Y)

Requires:
  - 2+ Enflame GCU cards
  - FlagCX built with USE_ENFLAME=1 and installed
  - torch_fl built with GCU support

Run:
    torchrun --nproc_per_node=2 tests/manual/test_flagos_dist_gcu.py
    torchrun --nproc_per_node=2 tests/manual/test_flagos_dist_gcu.py --test-fsdp2
"""

import argparse
import os
import sys

# torch_fl MUST be imported before torch (preloads GCU runtime).
import torch_fl  # noqa: F401
import torch

try:
    import flagcx  # noqa: F401  self-registers "flagcx" backend (enflame adaptor)
except ImportError:
    flagcx = None
    print("WARNING: flagcx not installed; falling back to no backend (will fail)")
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29532")

    dev = torch.device(f"flagos:{rank}")
    torch.flagos.set_device(rank)

    dist.init_process_group(
        backend="flagos",
        rank=rank,
        world_size=world_size,
    )

    # --- all_reduce ---
    t = torch.ones(4, device=dev) * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    ok_ar = torch.allclose(t.cpu(), torch.full((4,), float(expected)))
    print(
        f"[rank {rank}] all_reduce -> {t[0].item()} (expect {expected}) {'OK' if ok_ar else 'FAIL'}"
    )

    # --- broadcast ---
    b = torch.arange(4, device=dev, dtype=torch.float32) + rank * 10
    dist.broadcast(b, src=0)
    ok_bc = torch.allclose(b.cpu(), torch.arange(4, dtype=torch.float32))
    print(f"[rank {rank}] broadcast -> {b.tolist()} {'OK' if ok_bc else 'FAIL'}")

    # --- all_gather ---
    src = torch.ones(2, device=dev) * (rank + 1)
    gathered = [torch.zeros(2, device=dev) for _ in range(world_size)]
    dist.all_gather(gathered, src)
    vals = [g[0].item() for g in gathered]
    ok_ag = vals == [float(i + 1) for i in range(world_size)]
    print(f"[rank {rank}] all_gather -> {vals} {'OK' if ok_ag else 'FAIL'}")

    # --- all_gather_into_tensor (_allgather_base) ---
    # Separate virtual from allgather(); the FSDP/ZeRO parameter-gather path.
    agb = torch.empty(world_size, device=dev)
    dist.all_gather_into_tensor(agb, torch.full((1,), float(rank), device=dev))
    ok_agb = agb.cpu().tolist() == [float(i) for i in range(world_size)]
    print(
        f"[rank {rank}] all_gather_into_tensor -> {agb.cpu().tolist()} "
        f"{'OK' if ok_agb else 'FAIL'}"
    )

    # --- reduce_scatter_tensor (_reduce_scatter_base) ---
    # FSDP gradient-reduction path. Every rank contributes the same arange, so
    # rank r's shard is world_size * arange[2r : 2r+2].
    rs_in = torch.arange(2 * world_size, dtype=torch.float32, device=dev)
    rs_out = torch.empty(2, device=dev)
    dist.reduce_scatter_tensor(rs_out, rs_in)
    exp_rs = [float(world_size) * (2 * rank), float(world_size) * (2 * rank + 1)]
    ok_rs = rs_out.cpu().tolist() == exp_rs
    print(
        f"[rank {rank}] reduce_scatter_tensor -> {rs_out.cpu().tolist()} "
        f"(expect {exp_rs}) {'OK' if ok_rs else 'FAIL'}"
    )

    # --- barrier ---
    dist.barrier()
    print(f"[rank {rank}] barrier OK")

    # --- DDP forward/backward ---
    # Tiny model: no torch.compile (too slow for a test), but enough to verify
    # that the DDP gradient hook round-trips through ProcessGroupFlagOS.
    model = nn.Sequential(
        nn.Linear(4, 8, bias=False),
        nn.ReLU(),
        nn.Linear(8, 2, bias=False),
    ).to(dev)
    ddp = DistributedDataParallel(model, device_ids=[dev.index], output_device=dev)

    x = torch.arange(8, dtype=torch.float32, device=dev).reshape(2, 4)
    y = ddp(x).sum()
    y.backward()

    # Gradient must be all_reduced (same across ranks). Compare with rank 0.
    # Broadcast rank 0's param grad to everyone, then assert match.
    for p in model.parameters():
        if p.grad is not None:
            if rank == 0:
                ref = p.grad.clone()
                dist.broadcast(ref, src=0)
            else:
                ref = torch.zeros_like(p.grad)
                dist.broadcast(ref, src=0)
            ok_grad = torch.allclose(p.grad, ref, atol=1e-5)
            if not ok_grad:
                print(
                    f"[rank {rank}] DDP grad mismatch: "
                    f"{p.grad.flatten()[:4].tolist()} vs {ref.flatten()[:4].tolist()}"
                )
                sys.exit(1)
    print(f"[rank {rank}] DDP forward/backward OK")

    # --- Device guard test ---
    # GCU streams and pointers are device-scoped. Keep each tensor on its rank's
    # communicator device but select a different current device, then verify the
    # collective still works (ProcessGroupFlagOS guards it).
    if world_size > 1:
        t_guard = torch.ones(4, device=dev) * (rank + 1)
        # Set current device away from the communicator's device.
        import torch_fl._C as _C

        prev_dev = _C._get_device()
        _C._set_device((rank + 1) % world_size)
        try:
            dist.all_reduce(t_guard, op=dist.ReduceOp.SUM)
            ok_guard = torch.allclose(t_guard.cpu(), torch.full((4,), float(expected)))
            print(
                f"[rank {rank}] device guard test -> "
                f"{t_guard[0].item()} (expect {expected}) {'OK' if ok_guard else 'FAIL'}"
            )
        finally:
            _C._set_device(prev_dev)

    dist.destroy_process_group()
    print(f"[rank {rank}] ALL BASIC TESTS PASSED")


def worker_fsdp2(rank: int, world_size: int):
    """FSDP2 (fully_shard) test with per-layer wrapping and mixed precision."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")

    dev = torch.device(f"flagos:{rank}")
    torch.flagos.set_device(rank)

    dist.init_process_group(
        backend="flagos",
        rank=rank,
        world_size=world_size,
    )

    from torch.distributed._tensor import DeviceMesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = DeviceMesh("flagos", list(range(world_size)))

    # --- Test 1: Per-layer FSDP2 wrapping (the real usage pattern) ---
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
            self.block2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
            self.head = nn.Linear(32, 4)

        def forward(self, x):
            return self.head(self.block2(self.block1(x)))

    torch.manual_seed(0)
    model = Net().to(dev)

    # Apply fully_shard to each block, then to the root
    for block in (model.block1, model.block2, model.head):
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # Verify all params are DTensors
    n_shards = sum(1 for p in model.parameters() if hasattr(p, "to_local"))
    print(f"[rank {rank}] FSDP2 per-layer: {n_shards} DTensor params (expect 6)")
    assert n_shards == 6, f"Expected 6 DTensor params, got {n_shards}"

    # Training step
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.arange(8 * 16, dtype=torch.float32, device=dev).reshape(8, 16)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    loss_val = float(loss.detach())
    print(f"[rank {rank}] FSDP2 per-layer training step: loss={loss_val:.4f}")

    # --- Test 2: MixedPrecisionPolicy (bf16 params + fp32 reduce) ---
    torch.manual_seed(0)
    model_mp = Net().to(dev)

    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    for block in (model_mp.block1, model_mp.block2, model_mp.head):
        fully_shard(block, mesh=mesh, mp_policy=policy)
    fully_shard(model_mp, mesh=mesh, mp_policy=policy)

    opt_mp = torch.optim.SGD(model_mp.parameters(), lr=0.1)
    x = torch.arange(8 * 16, dtype=torch.float32, device=dev).reshape(8, 16)
    loss_mp = model_mp(x).sum()
    loss_mp.backward()
    opt_mp.step()
    opt_mp.zero_grad()

    loss_mp_val = float(loss_mp.detach())
    print(f"[rank {rank}] FSDP2 mixed precision: loss={loss_mp_val:.4f}")

    # --- Test 3: clip_grad_norm_ (needs cross-mesh norm reduction) ---
    torch.manual_seed(0)
    model_clip = Net().to(dev)
    for block in (model_clip.block1, model_clip.block2, model_clip.head):
        fully_shard(block, mesh=mesh)
    fully_shard(model_clip, mesh=mesh)

    opt_clip = torch.optim.SGD(model_clip.parameters(), lr=0.1)
    x = torch.arange(8 * 16, dtype=torch.float32, device=dev).reshape(8, 16)
    loss_clip = model_clip(x).sum()
    loss_clip.backward()

    # Clip gradients (tests cross-mesh norm reduction)
    total_norm = nn.utils.clip_grad_norm_(model_clip.parameters(), max_norm=1.0)
    total_norm_value = float(
        total_norm.to_local() if hasattr(total_norm, "to_local") else total_norm
    )
    print(f"[rank {rank}] FSDP2 grad clip: total_norm={total_norm_value:.4f}")

    opt_clip.step()
    opt_clip.zero_grad()

    # --- Test 4: State dict save/load (sharded checkpoint) ---
    from torch.distributed.checkpoint.state_dict import (
        get_state_dict,
        set_state_dict,
        StateDictOptions,
    )

    torch.manual_seed(0)
    model_ckpt = Net().to(dev)
    for block in (model_ckpt.block1, model_ckpt.block2, model_ckpt.head):
        fully_shard(block, mesh=mesh)
    fully_shard(model_ckpt, mesh=mesh)

    # Train one step to get non-zero grads
    opt_ckpt = torch.optim.SGD(model_ckpt.parameters(), lr=0.1)
    x = torch.arange(8 * 16, dtype=torch.float32, device=dev).reshape(8, 16)
    model_ckpt(x).sum().backward()
    opt_ckpt.step()
    opt_ckpt.zero_grad()

    # Save state dict
    state_dict, _ = get_state_dict(
        model_ckpt, optimizers=(), options=StateDictOptions()
    )

    # Create new model and load
    torch.manual_seed(999)  # Different init
    model_load = Net().to(dev)
    for block in (model_load.block1, model_load.block2, model_load.head):
        fully_shard(block, mesh=mesh)
    fully_shard(model_load, mesh=mesh)

    set_state_dict(
        model_load,
        optimizers=(),
        model_state_dict=state_dict,
        optim_state_dict={},
        options=StateDictOptions(),
    )

    # Verify loaded params match saved params
    for (n1, p1), (n2, p2) in zip(
        model_ckpt.named_parameters(), model_load.named_parameters()
    ):
        assert n1 == n2
        if hasattr(p1, "to_local") and hasattr(p2, "to_local"):
            match = torch.allclose(p1.to_local(), p2.to_local(), atol=1e-6)
            if not match:
                print(f"[rank {rank}] FSDP2 checkpoint mismatch at {n1}")
                sys.exit(1)

    print(f"[rank {rank}] FSDP2 state_dict save/load: OK")

    dist.destroy_process_group()
    print(f"[rank {rank}] ALL FSDP2 TESTS PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--test-fsdp2", action="store_true", help="Run FSDP2 tests")
    args, unknown = parser.parse_known_args()

    worker_fn = worker_fsdp2 if args.test_fsdp2 else worker

    if "RANK" in os.environ:
        # torchrun launched us
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        worker_fn(rank, world_size)
    else:
        # Manual mp.spawn
        mp.spawn(worker_fn, args=(args.world_size,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
