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

"""Qwen3 FSDP2 training on MetaX: flagos vs the vendor's own torch.

The FSDP2 feature tests use a 3-layer MLP, which says nothing about whether a
real transformer converges. This runs the same Qwen3 training job twice -- once
through the flagos device, once through MetaX's native torch on `cuda` -- and
compares the loss trajectories step by step.

The two runs are made comparable rather than merely similar:

  * identical initial weights (loaded from the same checkpoint, no random init)
  * identical batches: the dummy text dataset is tokenized on CPU and sliced
    deterministically per rank, so rank r sees the same tokens in both modes
  * identical optimizer, lr, step count and shard layout (per-decoder-layer
    fully_shard plus the root)

Each mode writes its losses to JSON; `--compare` then diffs them. Because both
runs execute the same fp32 kernels in the same order, the trajectories should
agree to well within fp32 accumulation noise -- a real divergence shows up as a
growing gap, not as jitter in the last digits.

Run (from repo root), sequentially so the two runs do not share GPUs:

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export MACA_PATH=/opt/maca
    export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/lib64:$LD_LIBRARY_PATH
    export PYTHONPATH=$PWD

    ACCELERATOR=metax \
        python tests/manual/metax/test_qwen3_fsdp2_metax.py --mode flagos
    python tests/manual/metax/test_qwen3_fsdp2_metax.py --mode native
    python tests/manual/metax/test_qwen3_fsdp2_metax.py --compare
"""

import argparse
import json
import os
import sys
import time

DEFAULT_MODEL = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    "c1899de289a04d12100db370d81485cdf75e47ca"
)
OUT_DIR = "/tmp/qwen3_fsdp2_metax"


def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("flagos", "native"))
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    # fp32 accumulation over 20 steps of a 0.6B model drifts in the last few
    # digits; a broken shard/reduce shows up far above this.
    ap.add_argument("--tol", type=float, default=2e-2)
    return ap.parse_args(argv)


_ARGS = _parse_args()

# torch_fl MUST be imported before torch in flagos mode: it preloads the maca
# libtorch_cuda.so. In native mode it must not be imported at all, so that the
# run is a true vendor-torch baseline.
if _ARGS.mode == "flagos":
    import torch_fl  # noqa: F401

    try:
        import flagcx  # noqa: F401  self-registers the "flagcx" metax adaptor
    except ImportError:
        flagcx = None
else:
    flagcx = None

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common")
)


def out_path(mode):
    return os.path.join(OUT_DIR, f"losses_{mode}.json")


def build_batches(tokenizer, args, world_size):
    """Tokenize on CPU once, then hand each rank a fixed, disjoint slice.

    Doing this outside the device code keeps the token ids bit-identical between
    the two modes -- no device RNG, no dataloader shuffling.
    """
    from dummy_dataset import DummyTextDataset

    n_needed = args.steps * args.batch_size * world_size
    ds = DummyTextDataset(
        tokenizer, num_samples=max(100, n_needed), max_length=args.seq_len
    )
    ids = torch.stack([ds[i]["input_ids"] for i in range(n_needed)])
    mask = torch.stack([ds[i]["attention_mask"] for i in range(n_needed)])
    # [steps, world_size, batch, seq]
    shape = (args.steps, world_size, args.batch_size, args.seq_len)
    return ids.reshape(shape), mask.reshape(shape)


def worker(rank, world_size, args):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29731")

    flagos = args.mode == "flagos"
    # flagos:i shares physical GPU i, so the vendor device index is set the same
    # way in both modes.
    torch.cuda.set_device(rank)
    dev = torch.device(f"flagos:{rank}" if flagos else f"cuda:{rank}")
    dist.init_process_group(
        backend="flagos" if flagos else "nccl", rank=rank, world_size=world_size
    )
    mesh_device = "flagos" if flagos else "cuda"

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mesh = init_device_mesh(mesh_device, (world_size,))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    all_ids, all_mask = build_batches(tokenizer, args, world_size)

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).to(dev)
    model.train()
    model.config.use_cache = False

    # Per-layer wrapping plus the root: the layout a real FSDP2 job uses, and the
    # one that actually produces reshard-after-forward traffic.
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    n_dtensor = sum(1 for p in model.parameters() if hasattr(p, "to_local"))
    n_param = sum(1 for _ in model.parameters())
    local_numel = sum(
        p.to_local().numel() if hasattr(p, "to_local") else p.numel()
        for p in model.parameters()
    )
    if rank == 0:
        print(
            f"[setup] mode={args.mode} world_size={world_size} "
            f"flagcx={'yes' if flagcx else 'no'} "
            f"params={n_param} dtensor={n_dtensor} local_numel={local_numel / 1e6:.2f}M",
            flush=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    local_losses, global_losses, step_times = [], [], []
    for step in range(args.steps):
        input_ids = all_ids[step, rank].to(dev)
        attention_mask = all_mask[step, rank].to(dev)
        # Mask padding out of the loss. The dataset pads short sentences to
        # seq_len, so leaving pad ids in the labels makes the curve a measure of
        # "learned to emit <pad>" (a 13 -> 0.4 cliff in one step) rather than of
        # language-modelling convergence.
        labels = input_ids.masked_fill(attention_mask == 0, -100)

        torch.cuda.synchronize()
        t0 = time.time()
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_times.append(time.time() - t0)

        # The per-rank loss only covers that rank's batch; the mean over ranks is
        # what a training log reports, and it is what has to match.
        local = float(loss.detach())
        buf = torch.tensor([local], device=dev, dtype=torch.float64)
        dist.all_reduce(buf)
        glob = float(buf.item()) / world_size
        local_losses.append(local)
        global_losses.append(glob)
        if rank == 0:
            print(
                f"[step {step:2d}] loss={glob:.6f} (rank0 {local:.6f}) "
                f"time={step_times[-1]:.2f}s",
                flush=True,
            )

    dist.barrier()
    if rank == 0:
        tokens = args.batch_size * args.seq_len * world_size * args.steps
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(out_path(args.mode), "w") as f:
            json.dump(
                {
                    "mode": args.mode,
                    "world_size": world_size,
                    "steps": args.steps,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                    "local_numel": local_numel,
                    "global_losses": global_losses,
                    "rank0_losses": local_losses,
                    "avg_step_s": sum(step_times) / len(step_times),
                    "throughput_tok_s": tokens / sum(step_times),
                },
                f,
                indent=2,
            )
        drop = global_losses[0] - global_losses[-1]
        print(
            f"=== qwen3 fsdp2 {args.mode}: {args.steps} steps, "
            f"loss {global_losses[0]:.4f} -> {global_losses[-1]:.4f} "
            f"(drop {drop:.4f}), {tokens / sum(step_times):.1f} tok/s ===",
            flush=True,
        )
    dist.destroy_process_group()


def compare(args):
    runs = {}
    for mode in ("flagos", "native"):
        path = out_path(mode)
        if not os.path.exists(path):
            print(f"[FAIL] missing {path} -- run --mode {mode} first")
            return 1
        with open(path) as f:
            runs[mode] = json.load(f)

    a, b = runs["flagos"], runs["native"]
    results = []
    for key in ("world_size", "steps", "lr", "batch_size", "seq_len", "local_numel"):
        results.append((f"same {key}", a[key] == b[key]))

    la, lb = a["global_losses"], b["global_losses"]
    results.append(("same step count", len(la) == len(lb)))
    n = min(len(la), len(lb))

    print(f"\n{'step':>4} {'flagos':>12} {'native':>12} {'abs diff':>10} {'rel':>9}")
    worst, worst_step = 0.0, -1
    for i in range(n):
        d = abs(la[i] - lb[i])
        rel = d / max(1e-9, abs(lb[i]))
        if rel > worst:
            worst, worst_step = rel, i
        print(f"{i:>4} {la[i]:>12.6f} {lb[i]:>12.6f} {d:>10.6f} {rel:>9.2e}")

    results.append(
        (
            f"loss trajectory within {args.tol:.0e} (worst {worst:.2e} @ step {worst_step})",
            worst <= args.tol,
        )
    )
    # Convergence, not just agreement: an untrained 0.6B on repeated text must
    # actually come down, or "matching" would only mean both runs are broken.
    for mode, run in (("flagos", a), ("native", b)):
        drop = run["global_losses"][0] - run["global_losses"][-1]
        results.append((f"{mode}: loss decreased (drop {drop:.4f})", drop > 0.0))
        results.append(
            (
                f"{mode}: all losses finite",
                all(x == x and abs(x) != float("inf") for x in run["global_losses"]),
            )
        )

    print()
    for name, ok in results:
        print(f"[{'OK' if ok else 'FAIL'}] {name}")
    print(
        f"\nthroughput: flagos {a['throughput_tok_s']:.1f} tok/s, "
        f"native {b['throughput_tok_s']:.1f} tok/s "
        f"({a['throughput_tok_s'] / b['throughput_tok_s']:.2f}x)"
    )
    n_fail = sum(1 for _, ok in results if not ok)
    status = "ALL PASS" if n_fail == 0 else f"{n_fail} FAILED"
    print(f"=== qwen3 fsdp2 flagos vs native: {status} ({len(results)} checks) ===")
    return 1 if n_fail else 0


if __name__ == "__main__":
    if _ARGS.compare:
        raise SystemExit(compare(_ARGS))
    if not _ARGS.mode:
        raise SystemExit("need --mode {flagos,native} or --compare")
    mp.set_start_method("spawn", force=True)
    mp.spawn(
        worker,
        args=(_ARGS.world_size, _ARGS),
        nprocs=_ARGS.world_size,
        join=True,
    )
