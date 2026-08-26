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

"""Live multi-device test for ProcessGroupFlagOS.

Launches N processes, each binding one flagos device, and exercises the basic
collective matrix plus DDP. By default it checks numerical results only. Use
``--require-flagcx`` for a strict FlagCX run: native fallback is made fatal and
the selected inner backend is asserted on every rank. Use ``--force-native``
to validate the fallback path independently.

Run:
    LD_LIBRARY_PATH=... python tests/manual/test_flagos_dist_live.py --world-size 2
    ... python tests/manual/test_flagos_dist_live.py --require-flagcx
    ... python tests/manual/test_flagos_dist_live.py --force-native
"""

import argparse
import os
import sys

if "--require-flagcx" in sys.argv:
    os.environ["FLAGOS_REQUIRE_FLAGCX"] = "1"
if "--force-native" in sys.argv:
    os.environ["FLAGOS_FORCE_NATIVE"] = "1"

# torch_fl MUST be imported before torch (preloads libtorch_cuda.so).
import torch_fl  # noqa: F401
import torch

try:
    import flagcx  # noqa: F401  self-registers the "flagcx" backend
except ImportError:
    flagcx = None
import torch.distributed as dist
import torch_fl.comm.process_group as _process_group
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


_FORCE_NATIVE = False
_REQUIRE_FLAGCX = False


def _configure_selection_mode() -> None:
    """Install fail-closed selection guards for the requested live mode."""
    global _FORCE_NATIVE, _REQUIRE_FLAGCX
    _FORCE_NATIVE = os.environ.get("FLAGOS_FORCE_NATIVE") == "1"
    _REQUIRE_FLAGCX = os.environ.get("FLAGOS_REQUIRE_FLAGCX") == "1"

    if _FORCE_NATIVE:
        _process_group.ProcessGroupFlagOS._try_build_flagcx = lambda *args, **kwargs: (
            False
        )
        return

    if not _REQUIRE_FLAGCX:
        return

    original_flagcx_builder = _process_group.ProcessGroupFlagOS._try_build_flagcx

    def require_flagcx(self, store, rank, world_size, timeout):
        selected = original_flagcx_builder(self, store, rank, world_size, timeout)
        if not selected:
            raise AssertionError("FlagCX was unavailable or not selected")
        return True

    _process_group.ProcessGroupFlagOS._try_build_flagcx = require_flagcx
    for native_name in ("_try_build_nccl", "_try_build_hccl", "_try_build_mccl"):
        setattr(
            _process_group.ProcessGroupFlagOS,
            native_name,
            lambda *args, _name=native_name, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"native fallback {_name} was selected")
            ),
        )


_configure_selection_mode()


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual.cpu(), expected.cpu()):
        raise AssertionError(
            f"{name}: got {actual.cpu().tolist()}, expected {expected.cpu().tolist()}"
        )


def worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")

    dev = torch.device(f"flagos:{rank}")
    torch.cuda.set_device(rank)  # flagos:i shares the physical device i

    dist.init_process_group(
        backend="flagos",
        rank=rank,
        world_size=world_size,
    )

    pg = dist.distributed_c10d._get_default_group()
    inner_backend = getattr(pg, "_inner_backend", None)
    if _REQUIRE_FLAGCX and inner_backend != "flagcx":
        raise AssertionError(f"rank {rank}: selected inner backend {inner_backend!r}")
    if _FORCE_NATIVE and not str(inner_backend).startswith("native:"):
        raise AssertionError(
            f"rank {rank}: expected native fallback, got {inner_backend!r}"
        )
    print(f"[rank {rank}] selected inner backend: {inner_backend}")

    # --- all_reduce ---
    t = torch.ones(4, device=dev) * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    _assert_close("all_reduce", t, torch.full((4,), float(expected)))
    print(f"[rank {rank}] all_reduce -> {t[0].item()} (expect {expected}) OK")

    # --- broadcast ---
    b = torch.arange(4, device=dev, dtype=torch.float32) + rank * 10
    dist.broadcast(b, src=0)
    _assert_close("broadcast", b, torch.arange(4, dtype=torch.float32))
    print(f"[rank {rank}] broadcast -> {b.tolist()} OK")

    # --- all_gather ---
    src = torch.ones(2, device=dev) * (rank + 1)
    gathered = [torch.zeros(2, device=dev) for _ in range(world_size)]
    dist.all_gather(gathered, src)
    vals = [g[0].item() for g in gathered]
    expected_vals = [float(i + 1) for i in range(world_size)]
    if vals != expected_vals:
        raise AssertionError(f"all_gather: got {vals}, expected {expected_vals}")
    print(f"[rank {rank}] all_gather -> {vals} OK")

    # --- all_gather_into_tensor (_allgather_base) ---
    agb = torch.empty(world_size, device=dev)
    dist.all_gather_into_tensor(agb, torch.full((1,), float(rank), device=dev))
    _assert_close(
        "all_gather_into_tensor",
        agb,
        torch.arange(world_size, dtype=torch.float32),
    )
    print(f"[rank {rank}] all_gather_into_tensor -> {agb.cpu().tolist()} OK")

    # --- reduce_scatter_tensor (_reduce_scatter_base) ---
    rs_in = torch.arange(2 * world_size, dtype=torch.float32, device=dev)
    rs_out = torch.empty(2, device=dev)
    dist.reduce_scatter_tensor(rs_out, rs_in)
    exp_rs = torch.tensor(
        [float(world_size) * (2 * rank), float(world_size) * (2 * rank + 1)]
    )
    _assert_close("reduce_scatter_tensor", rs_out, exp_rs)
    print(f"[rank {rank}] reduce_scatter_tensor -> {rs_out.cpu().tolist()} OK")

    # --- DDP forward/backward ---
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1)).to(dev)
    ddp = DistributedDataParallel(model)
    x = torch.randn(16, 8, device=dev)
    ddp(x).sum().backward()
    g0 = next(ddp.parameters()).grad
    gsum = g0.sum().item()
    gsum_t = torch.tensor([gsum], device=dev)
    all_gsum = [torch.zeros(1, device=dev) for _ in range(world_size)]
    dist.all_gather(all_gsum, gsum_t)
    grad_values = [value.item() for value in all_gsum]
    if not all(abs(value - grad_values[0]) < 1e-4 for value in grad_values):
        raise AssertionError(f"DDP gradients differ across ranks: {grad_values}")
    print(f"[rank {rank}] DDP backward grad.sum={gsum:.4f} OK")

    dist.barrier()
    if rank == 0:
        print("=== all collectives + DDP completed ===")
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--require-flagcx", action="store_true")
    ap.add_argument("--force-native", action="store_true")
    args = ap.parse_args()
    if args.require_flagcx and args.force_native:
        ap.error("--require-flagcx and --force-native are mutually exclusive")
    if args.world_size < 2:
        ap.error("--world-size must be at least 2")
    if args.require_flagcx:
        os.environ["FLAGOS_REQUIRE_FLAGCX"] = "1"
    if args.force_native:
        os.environ["FLAGOS_FORCE_NATIVE"] = "1"
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(args.world_size,), nprocs=args.world_size, join=True)
