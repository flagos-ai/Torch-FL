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

"""Qwen3 DDP training test for MUSA via FlagCX.

End-to-end test with a real transformer model (Qwen3-0.6B) to verify that DDP
gradient synchronization works correctly on MUSA through FlagCX.

Requires:
    - Qwen3-0.6B model checkpoint (downloaded via HuggingFace)
    - transformers library
    - 2+ MUSA GPUs

Usage:
    HF_HOME=<your-hf-cache-dir> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    LD_LIBRARY_PATH=<flagcx build/lib> \
        python tests/manual/musa/test_qwen3_ddp_musa.py --world-size 2 --steps 5
"""

import argparse
import os

# torch_fl MUST be imported before torch
import torch_fl  # noqa: F401
import torch_fl.distributed as flagos_dist
import torch

try:
    import flagcx  # noqa: F401 -- self-registers "flagcx" backend (MUSA adaptor)
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
    os.environ.setdefault("MASTER_PORT", "29621")

    dev = torch.device(f"flagos:{rank}")
    torch_fl.flagos.set_device(rank)

    dist.init_process_group(backend="flagos", rank=rank, world_size=world_size)
    if rank == 0:
        inner = type(
            getattr(dist.distributed_c10d._get_default_group(), "_inner", None)
        )
        print(f"[setup] world_size={world_size} backend={inner.__name__}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        print(f"[setup] loading Qwen3 model from {model_path}", flush=True)

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
        f"hooks={len(ddp._accum_grad_hooks)}",
        flush=True,
    )
    assert ddp._use_python_reducer, "flagos DDP must use python_reducer"
    assert len(ddp._accum_grad_hooks) > 0, "no accum-grad hooks installed"

    optimizer = torch.optim.AdamW(
        [p for p in ddp.parameters() if p.requires_grad], lr=1e-4
    )

    # Each rank sees a DIFFERENT batch (real DDP data parallelism)
    torch.manual_seed(1234 + rank)
    losses = []

    for step in range(steps):
        input_ids = torch.randint(0, 1000, (2, seq_len), device=dev)
        labels = input_ids.clone()

        out = ddp(input_ids=input_ids, labels=labels, use_cache=False)
        loss = out.loss
        loss.backward()

        # Verify gradients synchronized across ranks
        gsum, gn = _hash_grads(ddp.module)
        buf = torch.tensor([gsum], device=dev, dtype=torch.float64)
        gathered = [torch.zeros_like(buf) for _ in range(world_size)]
        dist.all_gather(gathered, buf)
        vals = [g.item() for g in gathered]
        synced = max(abs(v - vals[0]) for v in vals) < 1e-3

        if rank == 0:
            print(
                f"[step {step}] loss={loss.item():.4f} "
                f"grad_sum(per-rank)={[f'{v:.3f}' for v in vals]} "
                f"synced={'OK' if synced else 'FAIL'}",
                flush=True,
            )
        assert synced, f"grads diverge at step {step}: {vals}"

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())

    dist.barrier()
    if rank == 0:
        finite = all(abs(x) < 1e6 for x in losses)
        print(
            f"=== MUSA Qwen3 DDP: {steps} steps, losses={['%.3f' % x for x in losses]}, "
            f"finite={'OK' if finite else 'FAIL'} ===",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()

    if torch_fl.flagos.device_count() < args.world_size:
        raise SystemExit(
            f"needs {args.world_size} MUSA devices, "
            f"have {torch_fl.flagos.device_count()}"
        )

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        worker,
        args=(args.world_size, args.model, args.steps, args.seq_len),
        nprocs=args.world_size,
        join=True,
    )
