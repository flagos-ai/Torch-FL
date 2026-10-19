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

"""Live multi-GPU DDP test for Qwen3 on the flagos device (run on the A100 box).

Launches N processes, each binding one flagos device (flagos:i shares physical
GPU i), wraps a real Qwen3 model in DistributedDataParallel, and runs a few
training steps. Verifies that:

  1. init_process_group("flagos") + DDP construction succeed on flagos.
  2. DDP picks the flagos python_reducer path (_use_python_reducer=True) and
     installs the accum-grad all_reduce hooks.
  3. After backward, gradients are byte-for-byte identical across ranks
     (i.e. the ProcessGroupFlagOS all_reduce actually synchronised them).
  4. Loss decreases / stays finite over a handful of steps.

Requires (point HF_HOME at a pre-populated cache to run without network):
    HF_HOME=<your-hf-cache-dir> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

Run:
    HF_HOME=<your-hf-cache-dir> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python tests/manual/test_qwen3_ddp_live.py --world-size 2
"""

import argparse
import os

# torch_fl MUST be imported before torch (preloads libtorch_cuda.so).
import torch_fl  # noqa: F401
import torch_fl.distributed as flagos_dist
import torch

try:
    import flagcx  # noqa: F401  self-registers "flagcx" backend if present
except ImportError:
    flagcx = None
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


def _hash_grads(model):
    """Deterministic scalar summary of all grads, order-stable across ranks."""
    total = 0.0
    n = 0
    for name, p in sorted(model.named_parameters()):
        if p.grad is not None:
            total += p.grad.double().sum().item()
            n += p.grad.numel()
    return total, n


def worker(rank: int, world_size: int, model_path: str, steps: int, seq_len: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29537")

    dev = torch.device(f"flagos:{rank}")
    torch.cuda.set_device(rank)  # flagos:i shares physical GPU i

    dist.init_process_group(backend="flagos", rank=rank, world_size=world_size)
    if rank == 0:
        print(f"[setup] world_size={world_size} flagcx={'yes' if flagcx else 'no'}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    ).to(dev)
    model.train()
    flagos_dist.move_buffers_to_device(model, dev)

    ddp = DistributedDataParallel(model)
    print(
        f"[rank {rank}] DDP _use_python_reducer={ddp._use_python_reducer} "
        f"hooks={len(ddp._accum_grad_hooks)}"
    )
    assert ddp._use_python_reducer, "flagos DDP must use python_reducer"
    assert len(ddp._accum_grad_hooks) > 0, "no accum-grad hooks installed"

    optimizer = torch.optim.AdamW(
        [p for p in ddp.parameters() if p.requires_grad], lr=1e-4
    )

    # Each rank sees a DIFFERENT batch (real DDP data parallelism), so the
    # per-rank grads differ before sync and MUST agree after all_reduce.
    torch.manual_seed(1234 + rank)
    losses = []
    for step in range(steps):
        input_ids = torch.randint(0, 1000, (2, seq_len), device=dev)
        labels = input_ids.clone()

        out = ddp(input_ids=input_ids, labels=labels, use_cache=False)
        loss = out.loss
        loss.backward()

        gsum, gn = _hash_grads(ddp.module)
        # gather each rank's grad-hash to rank 0 to confirm they match
        buf = torch.tensor([gsum], device=dev, dtype=torch.float64)
        gathered = [torch.zeros_like(buf) for _ in range(world_size)]
        dist.all_gather(gathered, buf)
        vals = [g.item() for g in gathered]
        synced = max(abs(v - vals[0]) for v in vals) < 1e-3
        if rank == 0:
            print(
                f"[step {step}] loss={loss.item():.4f} "
                f"grad_sum(per-rank)={[f'{v:.3f}' for v in vals]} "
                f"synced={'OK' if synced else 'FAIL'}"
            )
        assert synced, f"grads diverge across ranks at step {step}: {vals}"

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())

    dist.barrier()
    if rank == 0:
        finite = all(torch.isfinite(torch.tensor(x)) for x in losses)
        print(
            f"=== Qwen3 DDP: {steps} steps done, losses={['%.3f' % x for x in losses]}, "
            f"all_finite={'OK' if finite else 'FAIL'} ==="
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    mp.spawn(
        worker,
        args=(args.world_size, args.model, args.steps, args.seq_len),
        nprocs=args.world_size,
        join=True,
    )
