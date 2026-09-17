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

"""Live multi-GPU FSDP test on MetaX (run on the 8xC550 box).

Covers both generations of FSDP on the flagos device:

  * FSDP1 -- ``FullyShardedDataParallel``, FlatParameter-based. Needs the comm
    collectives plus ``resizePrivateUse1Bytes`` (csrc/runtime/hooks.h) for
    storage resize during (re)sharding.
  * FSDP2 -- ``fully_shard``, DTensor-based. Additionally needs a DeviceMesh,
    which is where two real bugs surfaced:

    1. ``ProcessGroupFlagOS`` never registered its inner backend, so
       ``pg.group_name`` raised "ProcessGroup name not set" and DeviceMesh could
       not be constructed at all. Fixed by ``_register_inner_backend``.
    2. ``split_with_sizes_copy.out`` -- FSDP2's all-gather copy-out -- left its
       ``self`` input on flagos, so the boxing kernel re-dispatched to
       PrivateUse1 into itself and recursed until SIGSEGV. Fixed in
       scripts/codegen/codegen_ops.py by boxing const Tensor inputs of the
       TensorListBoxingGuard kernels.

Correctness is checked against a *single-GPU* reference rather than just
"it ran": sharding is only right if the sharded model takes the same
optimization trajectory as the unsharded one. Each rank feeds the model the
same input, so per-step losses must match the reference to tolerance and must
agree across ranks.

Run (from repo root):
    ACCELERATOR=metax \
    MACA_PATH=/opt/maca \
    LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/lib64:$LD_LIBRARY_PATH \
    PYTHONPATH=$PWD \
    python tests/manual/metax/test_fsdp_live_metax.py --world-size 2

Force the mccl(NCCL) inner backend (skip FlagCX even if installed):
    ... FLAGOS_DIST_FORCE_NCCL=1 python .../test_fsdp_live_metax.py
"""

import argparse
import os

# torch_fl MUST be imported before torch: in boxing mode it preloads the maca
# libtorch_cuda.so and sets GEMS_VENDOR=metax (which drives the vendor profile).
import torch_fl  # noqa: F401
import torch

if os.environ.get("FLAGOS_DIST_FORCE_NCCL", "0") != "1":
    try:
        import flagcx  # noqa: F401  self-registers "flagcx" (metax adaptor)
    except ImportError:
        flagcx = None
else:
    flagcx = None

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

STEPS = 3
LR = 0.1
IN_DIM, HIDDEN, OUT_DIM = 8, 16, 1
BATCH = 4


def build_model(device):
    """Identical init on every rank (and for the reference) via a fixed seed."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(IN_DIM, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, OUT_DIM),
    ).to(device)


def step_input(step, device):
    """Same input on every rank, so losses are directly comparable."""
    torch.manual_seed(1000 + step)
    return torch.randn(BATCH, IN_DIM, device=device)


def reference_losses(device):
    """Single-GPU (unsharded) trajectory the sharded runs must reproduce."""
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


def train_losses(model, device):
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    losses = []
    for step in range(STEPS):
        loss = model(step_input(step, device)).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.detach()))
    return losses


def _agree_across_ranks(values, device, world_size):
    """True when every rank produced the same list of losses."""
    t = torch.tensor(values, device=device)
    gathered = [torch.zeros(len(values), device=device) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    first = gathered[0].cpu()
    return all(torch.allclose(g.cpu(), first, atol=1e-4) for g in gathered)


def run_fsdp1(rank, world_size, dev, ref, results):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    model = FSDP(build_model(dev), device_id=dev)
    losses = train_losses(model, dev)
    matches = all(abs(a - b) < 1e-3 * max(1.0, abs(b)) for a, b in zip(losses, ref))
    agree = _agree_across_ranks(losses, dev, world_size)
    results.append(("fsdp1 matches single-GPU", matches))
    results.append(("fsdp1 ranks agree", agree))
    if rank == 0:
        print(f"[fsdp1] losses={[f'{v:.6f}' for v in losses]}", flush=True)
        print(f"[fsdp1] ref   ={[f'{v:.6f}' for v in ref]}", flush=True)


def run_fsdp2(rank, world_size, dev, ref, results):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    # Needs pg.group_name -> exercises _register_inner_backend.
    mesh = init_device_mesh("flagos", (world_size,))
    results.append(("fsdp2 device_mesh built", mesh.size() == world_size))

    model = build_model(dev)
    full_numel = sum(p.numel() for p in model.parameters())
    fully_shard(model, mesh=mesh)

    params = list(model.parameters())
    is_dtensor = all(hasattr(p, "to_local") for p in params)
    local_numel = sum(p.to_local().numel() for p in params if hasattr(p, "to_local"))
    results.append(("fsdp2 params are DTensor", is_dtensor))
    # Sharded across `world_size` ranks, so each rank holds strictly less than
    # the whole model (padding keeps it from being exactly full/world_size).
    results.append(("fsdp2 params sharded", 0 < local_numel < full_numel))

    losses = train_losses(model, dev)
    matches = all(abs(a - b) < 1e-3 * max(1.0, abs(b)) for a, b in zip(losses, ref))
    agree = _agree_across_ranks(losses, dev, world_size)
    results.append(("fsdp2 matches single-GPU", matches))
    results.append(("fsdp2 ranks agree", agree))
    if rank == 0:
        print(
            f"[fsdp2] full={full_numel} local={local_numel} "
            f"ptype={type(params[0]).__name__}",
            flush=True,
        )
        print(f"[fsdp2] losses={[f'{v:.6f}' for v in losses]}", flush=True)
        print(f"[fsdp2] ref   ={[f'{v:.6f}' for v in ref]}", flush=True)


def run_split_with_sizes_copy(dev, results):
    """FSDP2's all-gather copy-out primitive; used to recurse into SIGSEGV."""
    src = torch.arange(10, dtype=torch.float32, device=dev)
    out = [torch.empty(4, device=dev), torch.empty(6, device=dev)]
    torch.split_with_sizes_copy(src, [4, 6], dim=0, out=out)
    ok = out[0].cpu().tolist() == [0, 1, 2, 3] and out[1].cpu().tolist() == [
        4,
        5,
        6,
        7,
        8,
        9,
    ]
    results.append(("split_with_sizes_copy.out", ok))


def worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29691")

    dev = torch.device(f"flagos:{rank}")
    torch.cuda.set_device(rank)  # flagos:i shares physical MetaX GPU i

    dist.init_process_group(backend="flagos", rank=rank, world_size=world_size)
    results = []

    if rank == 0:
        print(
            f"[setup] GEMS_VENDOR={os.environ.get('GEMS_VENDOR', '?')} "
            f"flagcx={'yes' if flagcx else 'no'} world_size={world_size}",
            flush=True,
        )

    # group_name is what DeviceMesh needs; assert it directly so a regression
    # here is reported as itself rather than as a confusing DeviceMesh failure.
    pg = dist.distributed_c10d._get_default_group()
    try:
        has_name = bool(pg.group_name)
    except RuntimeError:
        has_name = False
    results.append(("pg.group_name set", has_name))

    run_split_with_sizes_copy(dev, results)

    ref = reference_losses(dev)

    for name, fn in (("fsdp1", run_fsdp1), ("fsdp2", run_fsdp2)):
        try:
            fn(rank, world_size, dev, ref, results)
        except Exception as e:  # noqa: BLE001
            results.append((f"{name} ran", False))
            if rank == 0:
                print(f"[{name}] raised {type(e).__name__}: {e}", flush=True)

    dist.barrier()
    if rank == 0:
        for name, ok in results:
            print(f"[{'OK' if ok else 'FAIL'}] {name}", flush=True)
        n_fail = sum(1 for _, ok in results if not ok)
        status = "ALL PASS" if n_fail == 0 else f"{n_fail} FAILED"
        print(f"=== metax fsdp live: {status} ({len(results)} checks) ===", flush=True)
    dist.destroy_process_group()
    if any(not ok for _, ok in results):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    args = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(args.world_size,), nprocs=args.world_size, join=True)
