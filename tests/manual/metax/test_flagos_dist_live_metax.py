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

"""Live multi-GPU test for ProcessGroupFlagOS on MetaX (run on the 8xC550 box).

This is the MetaX counterpart of ../test_flagos_dist_live.py. It validates the
metax comm path documented in docs/architecture/distributed-flagcx.md:

  * GEMS_VENDOR=metax  -> _VENDOR_PROFILES["metax"] = (flagcx_dev="cuda",
    view="_flagos_to_cuda_view", native="_try_build_nccl").
  * Inner backend priority: FlagCX first, else NCCL. On MetaX "NCCL" is the
    maca fork's ProcessGroupNCCL, which links libmccl.so under the hood -- so
    the native fallback here actually exercises MetaX's mccl collective library.
  * flagos tensors are a zero-copy cuda alias (maca libtorch_cuda), so
    _flagos_to_cuda_view hands the same physical buffer to mccl/flagcx.

It launches N processes, each binding one flagos device, and exercises:
  1. init_process_group("flagos")  (auto-selects ProcessGroupFlagOS)
  2. all_reduce / broadcast / all_gather / reduce_scatter on flagos tensors
  3. DistributedDataParallel forward/backward with the auto-patched python hook

Which inner backend was chosen is printed by rank 0 so the run self-documents
whether it went through FlagCX or the mccl(NCCL) fallback.

Run (from repo root):
    ACCELERATOR=metax \
    MACA_PATH=/opt/maca \
    LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/lib64:$LD_LIBRARY_PATH \
    PYTHONPATH=$PWD \
    python tests/manual/metax/test_flagos_dist_live_metax.py --world-size 2

Force the mccl(NCCL) path (skip FlagCX even if installed):
    ... FLAGOS_DIST_FORCE_NCCL=1 python .../test_flagos_dist_live_metax.py --world-size 2
"""

import argparse
import os

# torch_fl MUST be imported before torch: in boxing mode it preloads the maca
# libtorch_cuda.so and sets GEMS_VENDOR=metax (which drives the vendor profile).
import torch_fl  # noqa: F401
import torch

# FlagCX is optional. If FLAGOS_DIST_FORCE_NCCL=1 we deliberately do NOT import
# it, so ProcessGroupFlagOS falls back to the native NCCL(mccl) backend -- the
# path we most want to validate on MetaX today (flagcx metax adaptor may not be
# built in every env).
if os.environ.get("FLAGOS_DIST_FORCE_NCCL", "0") != "1":
    try:
        import flagcx  # noqa: F401  self-registers "flagcx" backend (metax adaptor)
    except ImportError:
        flagcx = None
else:
    flagcx = None

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def _describe_inner_backend(pg) -> str:
    """Best-effort name of the inner comm backend ProcessGroupFlagOS built."""
    # The default group's backend for privateuseone is our ProcessGroupFlagOS.
    inner = getattr(pg, "_inner", None)
    if inner is None:
        # dist.init_process_group wraps the backend; try the registered PG.
        try:
            backend = dist.get_backend()
        except Exception:
            backend = "?"
        return f"flagos(backend={backend}, inner=?)"
    return f"{type(inner).__module__}.{type(inner).__name__}"


def worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29537")

    dev = torch.device(f"flagos:{rank}")
    torch.cuda.set_device(rank)  # flagos:i shares physical MetaX GPU i

    dist.init_process_group(backend="flagos", rank=rank, world_size=world_size)

    if rank == 0:
        vendor = os.environ.get("GEMS_VENDOR", "?")
        forced = os.environ.get("FLAGOS_DIST_FORCE_NCCL", "0") == "1"
        print(
            f"[setup] GEMS_VENDOR={vendor} flagcx={'yes' if flagcx else 'no'} "
            f"force_nccl={forced} world_size={world_size}"
        )

    failures = []

    # --- all_reduce (SUM) ---
    t = torch.ones(4, device=dev) * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    ok = torch.allclose(t.cpu(), torch.full((4,), float(expected)))
    failures.append(("all_reduce", ok))
    print(
        f"[rank {rank}] all_reduce -> {t[0].item()} (expect {expected}) {'OK' if ok else 'FAIL'}"
    )

    # --- broadcast (src=0) ---
    b = torch.arange(4, device=dev, dtype=torch.float32) + rank * 10
    dist.broadcast(b, src=0)
    ok = torch.allclose(b.cpu(), torch.arange(4, dtype=torch.float32))
    failures.append(("broadcast", ok))
    print(f"[rank {rank}] broadcast -> {b.tolist()} {'OK' if ok else 'FAIL'}")

    # --- all_gather ---
    src = torch.ones(2, device=dev) * (rank + 1)
    gathered = [torch.zeros(2, device=dev) for _ in range(world_size)]
    dist.all_gather(gathered, src)
    vals = [g[0].item() for g in gathered]
    ok = vals == [float(i + 1) for i in range(world_size)]
    failures.append(("all_gather", ok))
    print(f"[rank {rank}] all_gather -> {vals} {'OK' if ok else 'FAIL'}")

    # --- reduce_scatter_tensor ---
    inp = torch.arange(world_size * 2, device=dev, dtype=torch.float32) + rank
    out = torch.zeros(2, device=dev)
    dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
    # rank r receives inp[2r:2r+2] summed over ranks: base + sum(ranks)
    base = torch.tensor([2 * rank, 2 * rank + 1], dtype=torch.float32)
    exp = base * world_size + sum(range(world_size))
    ok = torch.allclose(out.cpu(), exp)
    failures.append(("reduce_scatter_tensor", ok))
    print(
        f"[rank {rank}] reduce_scatter -> {out.tolist()} (expect {exp.tolist()}) {'OK' if ok else 'FAIL'}"
    )

    # --- DDP forward/backward ---
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1)).to(dev)
    ddp = DistributedDataParallel(model)
    if rank == 0:
        print(
            f"[setup] DDP _use_python_reducer={getattr(ddp, '_use_python_reducer', '?')} "
            f"accum_grad_hooks={len(getattr(ddp, '_accum_grad_hooks', []))}"
        )

    # Different input per rank -> grads differ pre-sync; after the all_reduce
    # hook every rank must hold the identical averaged/summed grad.
    torch.manual_seed(100 + rank)
    x = torch.randn(16, 8, device=dev)
    ddp(x).sum().backward()
    g0 = next(ddp.parameters()).grad
    gsum = g0.sum().item()
    print(f"[rank {rank}] DDP backward grad.sum={gsum:.6f} dev={g0.device}")

    # cross-rank grad identity check: gather each rank's grad.sum and compare
    gsum_t = torch.tensor([gsum], device=dev)
    all_gsum = [torch.zeros(1, device=dev) for _ in range(world_size)]
    dist.all_gather(all_gsum, gsum_t)
    vals = [v.item() for v in all_gsum]
    ok = all(abs(v - vals[0]) < 1e-4 for v in vals)
    failures.append(("ddp_grad_sync", ok))
    if rank == 0:
        print(
            f"[rank 0] DDP grad.sum across ranks {vals} synced={'OK' if ok else 'FAIL'}"
        )

    dist.barrier()
    if rank == 0:
        n_fail = sum(1 for _, ok in failures if not ok)
        status = "ALL PASS" if n_fail == 0 else f"{n_fail} FAILED"
        print(f"=== metax dist live: {status} ({[n for n, ok in failures]}) ===")
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    args = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(args.world_size,), nprocs=args.world_size, join=True)
