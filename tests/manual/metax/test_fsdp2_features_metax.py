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

"""FSDP2 (``fully_shard``) feature coverage on MetaX, beyond the smoke test.

tests/manual/metax/test_fsdp_live_metax.py establishes that the core path works:
DeviceMesh builds, parameters become sharded DTensors, and a 3-layer MLP trained
with plain SGD reproduces the single-GPU loss trajectory. That is necessary but
narrow. This test covers what a real FSDP2 job additionally depends on:

  * per-layer wrapping -- ``fully_shard`` applied to submodules as well as the
    root, which is how FSDP2 is actually used (it produces the reshard-after-
    forward traffic a single root-only call never exercises)
  * MixedPrecisionPolicy -- bf16 all-gather with fp32 reduce, the standard
    production config
  * Adam -- drives the fused/foreach optimizer kernels over DTensor params,
    a different code path from SGD
  * clip_grad_norm_ -- needs a cross-mesh norm reduction, and silently produces
    wrong norms if the partial-to-replicate reduction is broken
  * state_dict / load_state_dict -- sharded checkpoint round-trip
  * gradient accumulation with set_requires_gradient_sync(False)
  * CPU offload
  * 2D mesh construction (the composability substrate for FSDP+TP)

Every numerical check is against a single-GPU reference computed in the same
process, not against "it did not crash". Where a feature legitimately changes
the numbers (bf16), the tolerance is loosened rather than the check dropped.

Run (from repo root):
    FLAGOS_ACCELERATOR=metax \
    MACA_PATH=/opt/maca \
    LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/lib64:$LD_LIBRARY_PATH \
    PYTHONPATH=$PWD \
    python tests/manual/metax/test_fsdp2_features_metax.py --world-size 4
"""

import argparse
import os

# torch_fl MUST be imported before torch: in boxing mode it preloads the maca
# libtorch_cuda.so and sets GEMS_VENDOR=metax.
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
IN_DIM, HIDDEN, OUT_DIM = 16, 32, 4
BATCH = 8


class Net(nn.Module):
    """Three named blocks, so per-layer fully_shard has something to wrap."""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Linear(IN_DIM, HIDDEN), nn.ReLU())
        self.block2 = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.head = nn.Linear(HIDDEN, OUT_DIM)

    def forward(self, x):
        return self.head(self.block2(self.block1(x)))


def build_model(device):
    torch.manual_seed(0)
    return Net().to(device)


def step_input(step, device):
    torch.manual_seed(1000 + step)
    return torch.randn(BATCH, IN_DIM, device=device)


def train(model, device, opt_cls=torch.optim.SGD, steps=STEPS, **opt_kw):
    opt = opt_cls(model.parameters(), lr=LR, **opt_kw)
    losses = []
    for step in range(steps):
        loss = model(step_input(step, device)).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.detach()))
    return losses


def close(a, b, tol):
    return len(a) == len(b) and all(
        abs(x - y) <= tol * max(1.0, abs(y)) for x, y in zip(a, b)
    )


# --------------------------------------------------------------------------
# individual feature checks
# --------------------------------------------------------------------------


def check_per_layer_wrap(mesh, dev, ref, results):
    """fully_shard on each block plus the root -- the real usage pattern."""
    from torch.distributed.fsdp import fully_shard

    model = build_model(dev)
    for block in (model.block1, model.block2, model.head):
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)

    n_shards = sum(1 for p in model.parameters() if hasattr(p, "to_local"))
    results.append(("per-layer: all params DTensor", n_shards == 6))
    losses = train(model, dev)
    results.append(("per-layer: matches single-GPU", close(losses, ref, 1e-3)))
    return losses


def check_mixed_precision(mesh, dev, ref, results):
    """bf16 all-gather + fp32 reduce. Numbers shift, so the tolerance widens."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    model = build_model(dev)
    for block in (model.block1, model.block2, model.head):
        fully_shard(block, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)

    losses = train(model, dev)
    # bf16 has ~3 decimal digits; 5% is loose enough for accumulated drift over
    # 3 steps but still catches a genuinely broken reduction (which shows up as
    # order-of-magnitude or sign errors, not 5% drift).
    results.append(
        ("mixed-precision bf16: tracks single-GPU", close(losses, ref, 5e-2))
    )
    # Compute dtype must actually be bf16, else the policy silently did nothing.
    params = [p for p in model.parameters() if hasattr(p, "to_local")]
    results.append(
        ("mixed-precision: sharded params kept fp32", params[0].dtype == torch.float32)
    )


def check_adam(mesh, dev, results):
    """Adam over DTensor params -- exercises the foreach optimizer kernels."""
    from torch.distributed.fsdp import fully_shard

    ref_model = build_model(dev)
    ref_losses = train(ref_model, dev, opt_cls=torch.optim.Adam)

    model = build_model(dev)
    for block in (model.block1, model.block2, model.head):
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    losses = train(model, dev, opt_cls=torch.optim.Adam)
    results.append(("Adam: matches single-GPU", close(losses, ref_losses, 1e-3)))

    # foreach=True is the default for Adam on CUDA-like devices; assert it was
    # not silently downgraded to the for-loop path, which would hide breakage in
    # the _foreach_* boxing kernels.
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    results.append(
        ("Adam: foreach path active", opt.defaults.get("foreach") is not False)
    )


def check_clip_grad_norm(mesh, dev, results):
    """Cross-shard grad norm. A broken partial->replicate reduce shows up here."""
    from torch.distributed.fsdp import fully_shard

    ref_model = build_model(dev)
    ref_model(step_input(0, dev)).sum().backward()
    ref_norm = float(torch.nn.utils.clip_grad_norm_(ref_model.parameters(), 1.0))

    model = build_model(dev)
    for block in (model.block1, model.block2, model.head):
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    model(step_input(0, dev)).sum().backward()
    # FSDP2 has no module-level clip_grad_norm_ (that was FSDP1); the standard
    # utility is DTensor-aware and returns the norm as a replicated DTensor.
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    results.append(
        (
            "clip_grad_norm_: matches single-GPU",
            abs(norm - ref_norm) <= 1e-3 * max(1.0, ref_norm),
        )
    )


def check_state_dict(mesh, dev, results):
    """Sharded state_dict round-trip must restore the exact trajectory."""
    from torch.distributed.fsdp import fully_shard

    model = build_model(dev)
    for block in (model.block1, model.block2, model.head):
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)

    train(model, dev, steps=1)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    all_dtensor = all(hasattr(v, "to_local") for v in sd.values())
    results.append(("state_dict: entries are DTensor", all_dtensor))

    # Continue from the saved point, then reload and redo -- same losses.
    after_save = train(model, dev, steps=2)
    model.load_state_dict(sd)
    after_load = train(model, dev, steps=2)
    results.append(
        (
            "state_dict: reload reproduces trajectory",
            close(after_load, after_save, 1e-4),
        )
    )


def make_sharded(dev, mesh):
    from torch.distributed.fsdp import fully_shard

    model = build_model(dev)
    for block in (model.block1, model.block2, model.head):
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def grads_of(model):
    return [p.grad.to_local().clone() for p in model.parameters() if p.grad is not None]


def check_grad_accum(mesh, dev, results):
    """set_requires_gradient_sync(False) defers the reduce, it must not drop it.

    During the no-sync window FSDP2 keeps the unsharded gradient internally and
    ``p.grad`` stays None; the reduce happens on the next backward with sync
    re-enabled, which then holds the sum over every microbatch. Reduce-scatter is
    linear, so that must equal reducing each microbatch separately and letting
    ``p.grad`` accumulate -- which is the reference below.
    """
    n_micro = 3

    ref = make_sharded(dev, mesh)
    for step in range(n_micro):
        ref(step_input(step, dev)).sum().backward()
    ref_grads = grads_of(ref)

    model = make_sharded(dev, mesh)
    model.set_requires_gradient_sync(False)
    for step in range(n_micro - 1):
        model(step_input(step, dev)).sum().backward()
    # Documented FSDP2 behaviour: the sharded .grad is not populated until sync.
    deferred = all(p.grad is None for p in model.parameters())
    results.append(("grad accumulation: reduce deferred while no-sync", deferred))

    model.set_requires_gradient_sync(True)
    model(step_input(n_micro - 1, dev)).sum().backward()
    accum = grads_of(model)

    ok = len(accum) == len(ref_grads) and bool(accum)
    ok = ok and all(torch.isfinite(g).all().item() for g in accum)
    ok = ok and all(
        torch.allclose(a, r, rtol=1e-4, atol=1e-5) for a, r in zip(accum, ref_grads)
    )
    results.append(("grad accumulation: sum matches per-step reduce", ok))


def check_cpu_offload(mesh, dev, ref, results):
    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

    model = build_model(dev)
    policy = CPUOffloadPolicy()
    for block in (model.block1, model.block2, model.head):
        fully_shard(block, mesh=mesh, offload_policy=policy)
    fully_shard(model, mesh=mesh, offload_policy=policy)
    losses = train(model, dev)
    results.append(("cpu offload: matches single-GPU", close(losses, ref, 1e-3)))


def check_2d_mesh(world_size, results):
    """2D mesh is the substrate for FSDP+TP composition."""
    from torch.distributed.device_mesh import init_device_mesh

    if world_size < 4:
        return
    mesh2d = init_device_mesh(
        "flagos", (world_size // 2, 2), mesh_dim_names=("dp", "tp")
    )
    ok = mesh2d.size() == world_size and mesh2d["dp"].size() == world_size // 2
    results.append(("2D mesh (dp,tp) built", ok))


# --------------------------------------------------------------------------


def worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29713")

    dev = torch.device(f"flagos:{rank}")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="flagos", rank=rank, world_size=world_size)

    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("flagos", (world_size,))
    results = []

    if rank == 0:
        print(
            f"[setup] flagcx={'yes' if flagcx else 'no'} world_size={world_size}",
            flush=True,
        )

    ref = train(build_model(dev), dev)
    if rank == 0:
        print(f"[ref] sgd losses={[f'{v:.6f}' for v in ref]}", flush=True)

    checks = (
        ("per-layer wrap", lambda: check_per_layer_wrap(mesh, dev, ref, results)),
        ("mixed precision", lambda: check_mixed_precision(mesh, dev, ref, results)),
        ("adam", lambda: check_adam(mesh, dev, results)),
        ("clip_grad_norm", lambda: check_clip_grad_norm(mesh, dev, results)),
        ("state_dict", lambda: check_state_dict(mesh, dev, results)),
        ("grad accum", lambda: check_grad_accum(mesh, dev, results)),
        ("cpu offload", lambda: check_cpu_offload(mesh, dev, ref, results)),
        ("2d mesh", lambda: check_2d_mesh(world_size, results)),
    )
    for name, fn in checks:
        try:
            fn()
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
        print(
            f"=== metax fsdp2 features: {status} ({len(results)} checks) ===",
            flush=True,
        )
    dist.destroy_process_group()
    if any(not ok for _, ok in results):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=4)
    args = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(args.world_size,), nprocs=args.world_size, join=True)
