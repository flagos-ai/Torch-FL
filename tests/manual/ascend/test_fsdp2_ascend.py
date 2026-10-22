"""FSDP2 (fully_shard) test for ProcessGroupFlagOS on Ascend.

Exercises the per-parameter-sharding path: DeviceMesh construction, parameter
sharding as DTensor, all-gather in forward, and reduce-scatter of gradients in
backward.

Requires the torch_npu-free FlagCX build (FLAGCX_NPU_NO_TORCH_NPU=1).

  FLAGCX_NPU_NO_TORCH_NPU=1 HCCL_WHITELIST_DISABLE=1 \
  ASCEND_RT_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
    tests/manual/ascend/test_fsdp2_ascend.py
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

import torch_fl  # noqa: F401  (registers the flagos device + "flagos" backend)


class MLP(torch.nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.l1 = torch.nn.Linear(dim, dim * 2)
        self.act = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(dim * 2, dim)

    def forward(self, x):
        return self.l2(self.act(self.l1(x)))


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch_fl.flagos.set_device(local_rank)

    dist.init_process_group(backend="flagos")
    rank = dist.get_rank()
    world = dist.get_world_size()

    mesh = init_device_mesh("flagos", (world,))
    print(f"[rank {rank}] device mesh ok: {mesh}", flush=True)

    torch.manual_seed(0)
    model = MLP().to(f"flagos:{local_rank}")

    # Shard the submodules first, then the root, per FSDP2 convention.
    for layer in (model.l1, model.l2):
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # Parameters are now DTensors sharded on dim 0 across the mesh.
    sharded = 0
    for name, p in model.named_parameters():
        if isinstance(p, torch.distributed.tensor.DTensor):
            sharded += 1
            local_n = p.to_local().shape[0] if p.to_local().dim() > 0 else 0
            full_n = p.shape[0]
            assert local_n <= full_n, f"{name}: local dim0 {local_n} > full {full_n}"
    assert sharded > 0, "no parameter was converted to DTensor by fully_shard"
    print(f"[rank {rank}] params sharded as DTensor: {sharded}", flush=True)

    # Forward all-gathers shards; backward reduce-scatters gradients.
    torch.manual_seed(100 + rank)
    x = torch.randn(4, 16, device=f"flagos:{local_rank}")
    out = model(x)
    assert out.shape == (4, 16), f"unexpected output shape {out.shape}"
    print(f"[rank {rank}] forward (all-gather) ok", flush=True)

    loss = out.pow(2).mean()
    loss.backward()

    ngrad = 0
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name}: grad is None after backward"
        g = (
            p.grad.to_local()
            if isinstance(p.grad, torch.distributed.tensor.DTensor)
            else p.grad
        )
        assert torch.isfinite(g).all(), f"{name}: non-finite grad"
        ngrad += 1
    print(
        f"[rank {rank}] backward (reduce-scatter) ok, {ngrad} grads, "
        f"loss={loss.item():.6f}",
        flush=True,
    )

    opt = torch.optim.SGD(model.parameters(), lr=0.1, foreach=False)
    opt.step()
    opt.zero_grad(set_to_none=True)
    print(f"[rank {rank}] optimizer step ok", flush=True)

    dist.barrier()
    if rank == 0:
        print("FSDP2 test passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
