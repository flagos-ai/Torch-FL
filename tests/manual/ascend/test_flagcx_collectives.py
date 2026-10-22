"""Collective smoke test for ProcessGroupFlagOS on Ascend via FlagCX.

Requires a FlagCX built with FLAGCX_NPU_NO_TORCH_NPU=1 so that importing
flagcx does not pull in libtorch_npu.so (which would claim PrivateUse1 and
prevent torch_fl from registering the "flagos" device).

Run on 2 NPUs:
  FLAGCX_NPU_NO_TORCH_NPU=1 HCCL_WHITELIST_DISABLE=1 ASCEND_RT_VISIBLE_DEVICES=2,3 \
    torchrun --nproc_per_node=2 tests/manual/ascend/test_flagcx_collectives.py

Notes:
- HCCL_WHITELIST_DISABLE=1 is required to allow HCCL comm initialization
- Each rank must call torch_fl.flagos.set_device(local_rank) before init_process_group
  to bind the process to its NPU (HCCL's commInitRank runs on the current ACL device)
"""

import os

import torch
import torch.distributed as dist

import torch_fl  # noqa: F401  registers the "flagos" device and backend


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    # Bind this process to its own NPU *before* init_process_group: FlagCX's
    # comm init calls HCCL commInitRank on the current ACL device, so both ranks
    # would otherwise initialize on the same (unbound) device and fail with
    # "flagcxComm is not fully initialized".
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch_fl.flagos.set_device(local_rank)

    dist.init_process_group(backend="flagos")
    rank = dist.get_rank()
    world = dist.get_world_size()
    dev = f"flagos:{local_rank}"

    # allreduce SUM
    x = torch.full((4,), float(rank), device=dev)
    print(
        f"[rank {rank}] x.device = {x.device}, dev = {dev}, local_rank = {local_rank}",
        flush=True,
    )
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = float(sum(range(world)))
    _check(
        torch.allclose(x.cpu(), torch.full((4,), expected)),
        f"allreduce SUM: got {x.cpu().tolist()}, want {expected}",
    )
    print(f"[rank {rank}] allreduce SUM ok")

    # allreduce MAX
    x = torch.full((4,), float(rank), device=dev)
    dist.all_reduce(x, op=dist.ReduceOp.MAX)
    _check(
        torch.allclose(x.cpu(), torch.full((4,), float(world - 1))),
        f"allreduce MAX: got {x.cpu().tolist()}",
    )
    print(f"[rank {rank}] allreduce MAX ok")

    # broadcast from root 0
    x = torch.full((4,), float(rank), device=dev)
    dist.broadcast(x, src=0)
    _check(
        torch.allclose(x.cpu(), torch.zeros(4)),
        f"broadcast: got {x.cpu().tolist()}, want zeros",
    )
    print(f"[rank {rank}] broadcast ok")

    # all_gather_into_tensor
    src = torch.full((2,), float(rank), device=dev)
    dst = torch.empty(2 * world, device=dev)
    dist.all_gather_into_tensor(dst, src)
    want = torch.cat([torch.full((2,), float(r)) for r in range(world)])
    _check(
        torch.allclose(dst.cpu(), want),
        f"all_gather: got {dst.cpu().tolist()}, want {want.tolist()}",
    )
    print(f"[rank {rank}] all_gather_into_tensor ok")

    # reduce_scatter_tensor
    src = torch.arange(2 * world, dtype=torch.float32, device=dev)
    dst = torch.empty(2, device=dev)
    dist.reduce_scatter_tensor(dst, src, op=dist.ReduceOp.SUM)
    base = torch.arange(2 * world, dtype=torch.float32)[2 * rank : 2 * rank + 2]
    _check(
        torch.allclose(dst.cpu(), base * world),
        f"reduce_scatter: got {dst.cpu().tolist()}, want {(base * world).tolist()}",
    )
    print(f"[rank {rank}] reduce_scatter_tensor ok")

    dist.barrier()
    print(f"[rank {rank}] barrier ok")

    dist.destroy_process_group()
    if rank == 0:
        print("all collectives passed")


if __name__ == "__main__":
    main()
