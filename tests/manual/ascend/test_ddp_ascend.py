"""DDP gradient-synchronization test for ProcessGroupFlagOS on Ascend.

Runs standard torch.nn.parallel.DistributedDataParallel over the "flagos"
backend (FlagCX inner backend on Ascend) and checks that gradients are
identical on every rank after backward.

Requires the torch_npu-free FlagCX build (FLAGCX_NPU_NO_TORCH_NPU=1); torch_npu
and torch-fl cannot coexist, since both claim PrivateUse1.

  FLAGCX_NPU_NO_TORCH_NPU=1 HCCL_WHITELIST_DISABLE=1 \
  ASCEND_RT_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
    tests/manual/ascend/test_ddp_ascend.py
"""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import torch_fl  # noqa: F401  (registers the flagos device + "flagos" backend)


def main():
    # Bind the NPU before init_process_group: FlagCX initializes HCCL on the
    # current ACL device, so an unbound device makes comm init fail.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch_fl.flagos.set_device(local_rank)

    dist.init_process_group(backend="flagos")
    rank = dist.get_rank()
    world = dist.get_world_size()
    dev = f"flagos:{local_rank}"

    torch.manual_seed(0)  # identical initial weights on every rank
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    ).to(dev)
    ddp = DistributedDataParallel(model)

    # Rank-dependent input so per-rank local grads differ before the all-reduce.
    torch.manual_seed(100 + rank)
    x = torch.randn(4, 8, device=dev)
    loss = ddp(x).pow(2).mean()
    loss.backward()

    # DDP averages gradients, so every rank must now agree bitwise-closely.
    for name, p in ddp.named_parameters():
        assert p.grad is not None, f"{name}: grad is None"
        local = p.grad.detach().cpu()

        gathered = [torch.zeros_like(p.grad) for _ in range(world)]
        dist.all_gather(gathered, p.grad)
        for peer, g in enumerate(gathered):
            torch.testing.assert_close(
                local,
                g.cpu(),
                rtol=1e-5,
                atol=1e-6,
                msg=lambda m, n=name, p_=peer: (
                    f"{n}: grad differs between rank {rank} and rank {p_}\n{m}"
                ),
            )
        assert torch.isfinite(local).all(), f"{name}: non-finite grad"

    print(f"[rank {rank}] DDP grad sync ok (loss={loss.item():.6f})", flush=True)

    # One optimizer step must keep the ranks in lockstep too.
    # foreach=False: the multi-tensor path needs _foreach_add_.List, which has no
    # Ascend kernel registered yet (only CUDA/GCU), so it raises
    # "_foreach_add_.List: backend not registered". Unrelated to collectives.
    opt = torch.optim.SGD(ddp.parameters(), lr=0.1, foreach=False)
    opt.step()
    opt.zero_grad(set_to_none=True)

    flat = torch.cat([p.detach().reshape(-1) for p in ddp.parameters()])
    peers = [torch.zeros_like(flat) for _ in range(world)]
    dist.all_gather(peers, flat)
    for peer, w in enumerate(peers):
        torch.testing.assert_close(
            flat.cpu(),
            w.cpu(),
            rtol=1e-5,
            atol=1e-6,
            msg=lambda m, p_=peer: f"weights diverged vs rank {p_}\n{m}",
        )

    print(f"[rank {rank}] weights identical after step ok", flush=True)
    dist.barrier()
    if rank == 0:
        print("DDP test passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
