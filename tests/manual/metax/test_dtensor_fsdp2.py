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

"""FSDP2 (fully_shard) cross-check on flagos with flagcx active (P4).

fully_shard is built on DTensor, so this confirms the G0 fix unblocks the real
FSDP2 path (all-gather in forward, reduce-scatter of gradients) over flagcx.
Checked against a single-GPU reference trajectory.

Run (GPU idle):
    FLAGOS_ACCELERATOR=metax python test_dtensor_fsdp2.py --world-size 2
"""

import argparse
import os

import torch_fl  # noqa: F401
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

IN_DIM, HIDDEN, OUT_DIM = 8, 16, 1
BATCH, STEPS, LR = 4, 3, 0.1


def build_model(device):
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(IN_DIM, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, OUT_DIM),
    ).to(device)


def step_input(step, device):
    torch.manual_seed(1000 + step)
    return torch.randn(BATCH, IN_DIM, device=device)


def reference_losses(device):
    model = build_model(device)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    losses = []
    for step in range(STEPS):
        loss = model(step_input(step, device)).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.detach()))
    return losses


def worker(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29683")
    dev = torch.device(f"flagos:{rank}")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="flagos", rank=rank, world_size=world_size)

    mesh = init_device_mesh("flagos", (world_size,))
    results = []
    ref = reference_losses(dev)

    try:
        model = build_model(dev)
        full_numel = sum(p.numel() for p in model.parameters())
        fully_shard(model, mesh=mesh)
        params = list(model.parameters())
        is_dtensor = all(hasattr(p, "to_local") for p in params)
        local_numel = sum(
            p.to_local().numel() for p in params if hasattr(p, "to_local")
        )
        results.append(("FSDP2 params are DTensor", is_dtensor))
        results.append(("FSDP2 params sharded", 0 < local_numel < full_numel))

        opt = torch.optim.SGD(model.parameters(), lr=LR)
        got = []
        for step in range(STEPS):
            loss = model(step_input(step, dev)).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            got.append(float(loss.detach()))
        match = all(abs(a - b) < 1e-3 * max(1.0, abs(b)) for a, b in zip(got, ref))
        results.append(("FSDP2 losses match single-GPU", match))
        if rank == 0:
            print(
                f"[fsdp2] full={full_numel} local={local_numel} ptype={type(params[0]).__name__}",
                flush=True,
            )
            print(f"[fsdp2] losses={[f'{v:.6f}' for v in got]}", flush=True)
            print(f"[fsdp2] ref   ={[f'{v:.6f}' for v in ref]}", flush=True)
    except Exception as e:  # noqa: BLE001
        results.append(("FSDP2 ran", False))
        if rank == 0:
            print(f"[fsdp2] raised {type(e).__name__}: {e}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        for name, ok in results:
            print(f"[{'OK' if ok else 'FAIL'}] {name}", flush=True)
        status = "ALL PASS" if all(ok for _, ok in results) else "FAILED"
        print(f"=== FSDP2 on flagos (flagcx): {status} ===", flush=True)
    if any(not ok for _, ok in results):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    args = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(args.world_size,), nprocs=args.world_size, join=True)
