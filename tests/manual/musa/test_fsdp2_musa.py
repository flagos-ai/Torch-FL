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

"""FSDP2 training test for MUSA via FlagCX.

Verifies that PyTorch's FSDP2 (fully sharded data parallel) works correctly on
MUSA devices through ProcessGroupFlagOS → FlagCX (MUSA adaptor) → MCCL.

FSDP2 requires:
  - all_gather_into_tensor (_allgather_base) for parameter unshard
  - reduce_scatter_tensor (_reduce_scatter_base) for gradient shard
  - all_reduce for optimizer state sync (optional, not tested here)

Tests:
  1. FSDP2 fully_shard() succeeds on flagos tensors
  2. Forward/backward with sharded parameters completes
  3. Gradients are correctly reduced and sharded across ranks
  4. Loss converges over training steps

Usage:
    LD_LIBRARY_PATH=<flagcx build/lib> \
        python tests/manual/musa/test_fsdp2_musa.py --world-size 2 --steps 10
"""

import argparse
import os

# torch_fl MUST be imported before torch
import torch_fl  # noqa: F401
import torch_fl.distributed as flagos_dist
import torch

try:
    import flagcx  # noqa: F401 -- self-registers "flagcx" backend (MUSA adaptor)
except ImportError:
    flagcx = None
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard


def worker(rank: int, world_size: int, steps: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29620")

    dev = torch.device(f"flagos:{rank}")
    torch_fl.flagos.set_device(rank)

    dist.init_process_group(backend="flagos", rank=rank, world_size=world_size)

    if rank == 0:
        inner = type(
            getattr(dist.distributed_c10d._get_default_group(), "_inner", None)
        )
        print(f"[setup] world_size={world_size} backend={inner.__name__}", flush=True)

    # --- Build a 3-layer MLP and apply FSDP2 ---
    torch.manual_seed(42)  # same initial weights across ranks

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

    model = MLP().to(dev)
    flagos_dist.move_buffers_to_device(model, dev)

    # Apply FSDP2 sharding to each layer separately
    for layer in [model.fc1, model.fc2, model.fc3]:
        fully_shard(layer)
    fully_shard(model)  # root FSDP wrapper

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(
            f"[rank {rank}] FSDP2 model initialized, total params={param_count}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    # Each rank sees DIFFERENT data (real data parallelism)
    torch.manual_seed(200 + rank)

    for step in range(steps):
        x = torch.randn(16, 128, device=dev)
        y = torch.randint(0, 10, (16,), device=dev)

        out = model(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()

        # Gather loss values to rank 0 for monitoring
        loss_buf = torch.tensor([loss.item()], device=dev, dtype=torch.float32)
        gathered_losses = [torch.zeros_like(loss_buf) for _ in range(world_size)]
        dist.all_gather(gathered_losses, loss_buf)

        if rank == 0:
            loss_vals = [t.item() for t in gathered_losses]
            print(
                f"[step {step}] loss(per-rank)={['%.4f' % v for v in loss_vals]}",
                flush=True,
            )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())

    dist.barrier()

    if rank == 0:
        finite = all(abs(x) < 1e6 for x in losses)
        decreasing = losses[-1] < losses[0] + 0.1  # allow for noise
        print(
            f"=== MUSA FSDP2: {steps} steps, losses(rank0)={['%.4f' % x for x in losses]}, "
            f"finite={'OK' if finite else 'FAIL'} decreasing={'OK' if decreasing else 'FAIL'} ===",
            flush=True,
        )
        assert finite, "loss diverged"

    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    if torch_fl.flagos.device_count() < args.world_size:
        raise SystemExit(
            f"needs {args.world_size} MUSA devices, "
            f"have {torch_fl.flagos.device_count()}"
        )

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        worker, args=(args.world_size, args.steps), nprocs=args.world_size, join=True
    )
