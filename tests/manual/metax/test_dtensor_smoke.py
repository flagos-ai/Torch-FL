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

"""DTensor public-API smoke test on the flagos backend (P1).

Establishes the minimal end-to-end path directly against DTensor, bypassing
FSDP2: ``init_process_group("flagos")`` -> ``init_device_mesh("flagos", ...)``
-> ``distribute_tensor`` with Shard/Replicate/Partial -> ``to_local``/
``full_tensor`` numerical checks -> one ``redistribute`` round-trip.

The redistribute checks deliberately exercise the two collective paths FSDP2
never touches (G1/G2 from the P0 gap list):

  * Shard(0) -> Replicate   : all_gather
  * Replicate -> Shard(1)   : local chunk (no comm)
  * Shard(0) -> Shard(1)    : all_to_all   <-- G1, never covered before
  * Partial  -> Replicate   : all_reduce

Every check compares ``full_tensor()`` against the unsharded reference, not
"it did not crash".

Run (from any dir with torch_fl installed):
    FLAGOS_ACCELERATOR=metax python test_dtensor_smoke.py --world-size 2
"""

import argparse
import os

# torch_fl MUST be imported before torch (preloads vendor libtorch + sets vendor).
import torch_fl  # noqa: F401
import torch

try:
    import flagcx  # noqa: F401  self-registers "flagcx" backend when present
except ImportError:
    flagcx = None

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)


def _report(rank, results):
    if rank == 0:
        for name, ok in results:
            print(f"[{'OK' if ok else 'FAIL'}] {name}", flush=True)
        n_fail = sum(1 for _, ok in results if not ok)
        status = "ALL PASS" if n_fail == 0 else f"{n_fail} FAILED"
        print(f"=== dtensor smoke: {status} ({len(results)} checks) ===", flush=True)


def check_shard(dev, mesh, results):
    t = torch.arange(16, dtype=torch.float32, device=dev).reshape(4, 4)
    dt = distribute_tensor(t, mesh, [Shard(0)])
    results.append(("distribute_tensor -> DTensor", isinstance(dt, DTensor)))
    local = dt.to_local()
    expect_rows = 4 // mesh.size()
    results.append(("Shard(0) local shape", tuple(local.shape) == (expect_rows, 4)))
    results.append(("Shard(0) full_tensor == ref", torch.equal(dt.full_tensor(), t)))


def check_replicate(dev, mesh, results):
    t = torch.arange(16, dtype=torch.float32, device=dev).reshape(4, 4)
    dt = distribute_tensor(t, mesh, [Replicate()])
    results.append(("Replicate full_tensor == ref", torch.equal(dt.full_tensor(), t)))


def check_partial(dev, mesh, results):
    t = torch.arange(8, dtype=torch.float32, device=dev).reshape(2, 4)
    # distribute_tensor(t, [Partial("sum")]) splits t evenly across ranks, so
    # full_tensor() == t (recovered by all_reduce(sum)), NOT t * world_size.
    dt = distribute_tensor(t, mesh, [Partial("sum")])
    results.append(("Partial full_tensor == ref", torch.allclose(dt.full_tensor(), t)))


def check_redistribute(dev, mesh, results):
    t = torch.arange(32, dtype=torch.float32, device=dev).reshape(8, 4)
    dt = distribute_tensor(t, mesh, [Shard(0)])

    rep = dt.redistribute(placements=[Replicate()])  # all_gather
    results.append(
        ("redistribute Shard(0)->Replicate", torch.equal(rep.full_tensor(), t))
    )

    shard1 = rep.redistribute(placements=[Shard(1)])  # local chunk
    results.append(
        ("redistribute Replicate->Shard(1)", torch.equal(shard1.full_tensor(), t))
    )

    # G1: direct Shard(0) -> Shard(1) is the all_to_all path FSDP2 never takes.
    try:
        s01 = dt.redistribute(placements=[Shard(1)])
        results.append(
            (
                "redistribute Shard(0)->Shard(1) [all_to_all]",
                torch.equal(s01.full_tensor(), t),
            )
        )
    except Exception as e:  # noqa: BLE001 - surface the failure, keep other checks
        results.append(("redistribute Shard(0)->Shard(1) [all_to_all]", False))
        if dist.get_rank() == 0:
            print(f"[all_to_all] raised {type(e).__name__}: {e}", flush=True)


def check_partial_to_replicate(dev, mesh, results):
    t = torch.arange(8, dtype=torch.float32, device=dev).reshape(2, 4)
    dt = distribute_tensor(t, mesh, [Partial("sum")])
    rep = dt.redistribute(placements=[Replicate()])  # all_reduce
    # all_reduce(sum) recovers t, not t * world_size (see check_partial).
    results.append(
        ("redistribute Partial->Replicate", torch.allclose(rep.full_tensor(), t))
    )


def worker(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29692")

    dev = torch.device(f"flagos:{rank}")
    torch.cuda.set_device(rank)  # flagos:i shares physical MetaX GPU i

    dist.init_process_group(backend="flagos", rank=rank, world_size=world_size)
    results = []

    if rank == 0:
        print(
            f"[setup] GEMS_VENDOR={os.environ.get('GEMS_VENDOR', '?')} "
            f"flagcx={'yes' if flagcx else 'no'} world={world_size}",
            flush=True,
        )

    mesh = init_device_mesh("flagos", (world_size,))
    results.append(("init_device_mesh('flagos')", mesh.size() == world_size))

    checks = (
        ("shard", check_shard),
        ("replicate", check_replicate),
        ("partial", check_partial),
        ("partial_to_replicate", check_partial_to_replicate),
        ("redistribute", check_redistribute),
    )
    for name, fn in checks:
        try:
            fn(dev, mesh, results)
        except Exception as e:  # noqa: BLE001
            results.append((f"{name} ran", False))
            if rank == 0:
                print(f"[{name}] raised {type(e).__name__}: {e}", flush=True)

    dist.barrier()
    _report(rank, results)
    dist.destroy_process_group()
    if any(not ok for _, ok in results):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    args = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(args.world_size,), nprocs=args.world_size, join=True)
