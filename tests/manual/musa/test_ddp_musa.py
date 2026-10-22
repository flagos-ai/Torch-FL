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

"""DDP forward/backward test for MUSA via FlagCX.

Verifies that DistributedDataParallel correctly synchronizes gradients across
MUSA devices using ProcessGroupFlagOS → FlagCX (MUSA adaptor) → MCCL.

Tests:
  1. DDP construction on flagos tensors succeeds
  2. DDP picks python_reducer path and installs accum-grad hooks
  3. After backward, gradients are identical across all ranks
  4. Loss remains finite and decreases over training steps

Usage:
    LD_LIBRARY_PATH=<flagcx build/lib> \
        python tests/manual/musa/test_ddp_musa.py --world-size 2 --steps 5
"""

import argparse
import os

# torch_fl MUST be imported before torch
import torch_fl  # noqa: F401
import torch

try:
    import flagcx  # noqa: F401 -- self-registers "flagcx" backend (MUSA adaptor)
except ImportError:
    flagcx = None
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def _hash_grads(model):
    """Deterministic scalar summary of all gradients for cross-rank comparison."""
    total = 0.0
    n = 0
    for name, p in sorted(model.named_parameters()):
        if p.grad is not None:
            total += p.grad.double().sum().item()
            n += p.grad.numel()
    return total, n


def worker(rank: int, world_size: int, steps: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29619")

    dev = torch.device(f"flagos:{rank}")
    torch_fl.flagos.set_device(rank)

    dist.init_process_group(backend="flagos", rank=rank, world_size=world_size)

    if rank == 0:
        inner = type(
            getattr(dist.distributed_c10d._get_default_group(), "_inner", None)
        )
        print(f"[setup] world_size={world_size} backend={inner.__name__}", flush=True)

    # --- Build a small 3-layer MLP ---
    torch.manual_seed(42)  # same initial weights across ranks
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(dev)

    ddp = DistributedDataParallel(model)
    print(
        f"[rank {rank}] DDP _use_python_reducer={ddp._use_python_reducer} "
        f"hooks={len(ddp._accum_grad_hooks)}",
        flush=True,
    )
    assert ddp._use_python_reducer, "flagos DDP must use python_reducer"
    assert len(ddp._accum_grad_hooks) > 0, "no accum-grad hooks installed"

    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)

    losses = []
    # Each rank sees DIFFERENT data (real data parallelism)
    torch.manual_seed(100 + rank)

    for step in range(steps):
        x = torch.randn(16, 128, device=dev)
        y = torch.randint(0, 10, (16,), device=dev)

        out = ddp(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()

        # Verify gradients synchronized across ranks
        gsum, gn = _hash_grads(ddp.module)
        buf = torch.tensor([gsum], device=dev, dtype=torch.float64)
        gathered = [torch.zeros_like(buf) for _ in range(world_size)]
        dist.all_gather(gathered, buf)
        vals = [g.item() for g in gathered]
        synced = max(abs(v - vals[0]) for v in vals) < 1e-3

        if rank == 0:
            print(
                f"[step {step}] loss={loss.item():.4f} "
                f"grad_sum={[f'{v:.3f}' for v in vals]} "
                f"synced={'OK' if synced else 'FAIL'}",
                flush=True,
            )

        assert synced, f"grads diverge at step {step}: {vals}"

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())

    dist.barrier()

    if rank == 0:
        finite = all(abs(x) < 1e6 for x in losses)
        decreasing = losses[-1] < losses[0]
        print(
            f"=== MUSA DDP: {steps} steps, losses={['%.3f' % x for x in losses]}, "
            f"finite={'OK' if finite else 'FAIL'} decreasing={'OK' if decreasing else 'FAIL'} ===",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
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
