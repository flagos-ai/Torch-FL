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

"""2-D DeviceMesh (DP x TP) DTensor verification on flagos (P4).

Builds a (dp, tp) 2-D mesh and checks DTensor placements over it, comparing
full_tensor() against the unsharded reference for each (dp, tp) placement
combination. This validates the mesh-dim routing that TP+FSDP composability
depends on.

Run (GPU idle, 4 ranks -> 2x2):
    FLAGOS_ACCELERATOR=metax python test_dtensor_2d_mesh.py --world-size 4
"""

import argparse
import os

import torch_fl  # noqa: F401
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor


def worker(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29685")
    dev = torch.device(f"flagos:{rank}")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="flagos", rank=rank, world_size=world_size)

    dp = world_size // 2  # 2 x 2 by default
    mesh = init_device_mesh("flagos", (dp, 2), mesh_dim_names=("dp", "tp"))
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    t = torch.arange(64, dtype=torch.float32, device=dev).reshape(8, 8)

    # placement combinations over (dp, tp): Shard/Replicate on each dim
    combos = [
        ("Shard(0),Replicate", [Shard(0), Replicate()]),
        ("Replicate,Shard(0)", [Replicate(), Shard(0)]),
        ("Shard(0),Shard(1)", [Shard(0), Shard(1)]),
        ("Replicate,Replicate", [Replicate(), Replicate()]),
    ]
    for name, placements in combos:
        try:
            dt = distribute_tensor(t, mesh, placements)
            check(
                f"2D mesh {name}: full_tensor == ref", torch.equal(dt.full_tensor(), t)
            )
        except Exception as e:  # noqa: BLE001
            check(f"2D mesh {name}", False)
            if rank == 0:
                print(f"[2dmesh {name}] raised {type(e).__name__}: {e}", flush=True)

    # cross-dim redistribute: dp-Shard -> tp-Shard (exercises mesh-dim routing)
    try:
        dt = distribute_tensor(t, mesh, [Shard(0), Replicate()])
        r = dt.redistribute(placements=[Replicate(), Shard(1)])
        check("2D redistribute dp->tp", torch.equal(r.full_tensor(), t))
    except Exception as e:  # noqa: BLE001
        check("2D redistribute dp->tp", False)
        if rank == 0:
            print(f"[2dmesh redistribute] raised {type(e).__name__}: {e}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        n_fail = sum(1 for _, ok in results if not ok)
        for name, ok in results:
            print(f"[{'OK' if ok else 'FAIL'}] {name}", flush=True)
        print(
            f"=== 2D mesh on flagos: {'ALL PASS' if n_fail == 0 else f'{n_fail} FAILED'} ===",
            flush=True,
        )
    if any(not ok for _, ok in results):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=4)
    args = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(args.world_size,), nprocs=args.world_size, join=True)
