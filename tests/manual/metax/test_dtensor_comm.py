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

"""Root-based collectives on flagos: broadcast / scatter / reduce / gather (G2).

These are the primitives TP input distribution (mesh_scatter / mesh_broadcast)
depends on, and none of them are exercised by FSDP2. Verified numerically
against the expected values rather than "it did not crash".

Run (GPU idle):
    FLAGOS_ACCELERATOR=metax python test_dtensor_comm.py --world-size 2
"""

import argparse
import os

import torch_fl  # noqa: F401
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29684")
    dev = torch.device(f"flagos:{rank}")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="flagos", rank=rank, world_size=world_size)
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    # --- broadcast: rank0 value replicated everywhere ---
    t = torch.arange(8, dtype=torch.float32, device=dev)
    if rank == 0:
        t = torch.arange(8, dtype=torch.float32, device=dev) * 2
    dist.broadcast(t, src=0)
    check("broadcast", torch.equal(t.cpu(), (torch.arange(8) * 2).float()))

    # --- scatter: rank0 distributes scatter_list, rank i gets all-i ---
    scatter_list = None
    if rank == 0:
        scatter_list = [
            torch.full((4,), i, dtype=torch.float32, device=dev)
            for i in range(world_size)
        ]
    out = torch.zeros(4, dtype=torch.float32, device=dev)
    dist.scatter(out, scatter_list, src=0)
    check("scatter", bool((out.cpu() == rank).all()))

    # --- reduce (sum) to rank0 ---
    t2 = torch.ones(4, dtype=torch.float32, device=dev) * (rank + 1)
    dist.reduce(t2, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        check(
            "reduce sum -> rank0",
            torch.equal(t2.cpu(), torch.tensor([3.0, 3.0, 3.0, 3.0])),
        )

    # --- gather to rank0 ---
    gather_list = None
    if rank == 0:
        gather_list = [
            torch.zeros(4, dtype=torch.float32, device=dev) for _ in range(world_size)
        ]
    local = torch.full((4,), rank, dtype=torch.float32, device=dev)
    dist.gather(local, gather_list, dst=0)
    if rank == 0:
        expect = torch.stack([torch.tensor([float(i)] * 4) for i in range(world_size)])
        check(
            "gather -> rank0",
            all(
                torch.equal(g.cpu(), expect[i].cpu()) for i, g in enumerate(gather_list)
            ),
        )

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        n_fail = sum(1 for _, ok in results if not ok)
        for name, ok in results:
            print(f"[{'OK' if ok else 'FAIL'}] {name}", flush=True)
        print(
            f"=== root collectives (G2): {'ALL PASS' if n_fail == 0 else f'{n_fail} FAILED'} ===",
            flush=True,
        )
    if any(not ok for _, ok in results):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    args = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(args.world_size,), nprocs=args.world_size, join=True)
