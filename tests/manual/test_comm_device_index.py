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

"""Collectives must run on their operand's device, not the current one.

The comm-side counterpart to tests/integration/test_compute_device_index.py:
same hazard (work enqueued on the wrong device), different layer.
``ProcessGroupFlagOS`` hands tensors to an inner backend that resolves its
device from whatever is *current*, so the wrapper has to pin the current device
to the operand's before delegating.

Why this needs its own test rather than being covered by
test_flagos_dist_live.py: that test calls ``torch.cuda.set_device(rank)`` AND
happened to leave the flagos current device where the collective needed it, so
it passed even with the bug present. This one deliberately calls ONLY
``torch.cuda.set_device(rank)`` -- the realistic thing for a caller to do -- and
never touches ``torch_fl.flagos.set_device``. Allocating a ``flagos:rank``
tensor then leaves the cuda current device back at 0, so rank 1 enqueued
device-1 buffers onto a device-0 stream.

FlagCX makes this worse than a wrong answer: ``getStreamByIndex`` lazily creates
one stream on the current device and caches it by index alone, so the first
collective binds the comm to that device permanently. On DTK the result was a
GPU fault (VMFault / "invalid resource handle"), which kills the process instead
of failing a test -- so a regression here shows up as a dead rank, not an
assertion. That is still a clear signal.

Manual rather than pytest because it needs a real multi-process world; mirrors
test_flagos_dist_live.py's harness.

Usage:
    LD_LIBRARY_PATH=<flagcx build/lib> \
        python tests/manual/test_comm_device_index.py --world-size 2
"""

import argparse
import os

# torch_fl MUST be imported before torch (preloads libtorch_cuda.so).
import torch_fl  # noqa: F401
import torch

try:
    import flagcx  # noqa: F401 -- self-registers the "flagcx" backend
except ImportError:
    flagcx = None
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29617")

    dev = torch.device(f"flagos:{rank}")
    # Deliberately only the cuda-side bind. Adding
    # torch_fl.flagos.set_device(rank) here would mask the bug this test exists
    # to catch.
    torch.cuda.set_device(rank)

    dist.init_process_group(backend="flagos", rank=rank, world_size=world_size)

    inner = type(getattr(dist.distributed_c10d._get_default_group(), "_inner", None))
    print(f"[rank {rank}] inner backend = {inner.__name__}", flush=True)

    failures = []

    # --- all_reduce: the collective that faulted ---
    t = torch.ones(4, device=dev) * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = float(sum(range(1, world_size + 1)))
    got = t.cpu()[0].item()
    ok = abs(got - expected) < 1e-5
    failures += [] if ok else [f"all_reduce got {got} want {expected}"]
    print(
        f"[rank {rank}] all_reduce -> {got} (expect {expected}) "
        f"{'OK' if ok else 'FAIL'}",
        flush=True,
    )

    # --- barrier: carries no tensor, so the wrapper must reuse the device the
    # comm was already bound to. FlagCX rejects it otherwise with "flagcx
    # communicator was initialized with different device".
    dist.barrier()
    print(f"[rank {rank}] barrier OK", flush=True)

    # --- the tensor's device must survive the round trip ---
    ok_dev = t.device.type == "flagos" and t.device.index == rank
    failures += [] if ok_dev else [f"result on {t.device}, want flagos:{rank}"]
    print(
        f"[rank {rank}] result device {t.device} {'OK' if ok_dev else 'FAIL'}",
        flush=True,
    )

    # --- and the guard must be scoped: it may not leak the switch ---
    cur = torch.cuda.current_device()
    print(f"[rank {rank}] cuda current after collectives = {cur}", flush=True)

    if failures:
        raise AssertionError(f"[rank {rank}] " + "; ".join(failures))

    dist.destroy_process_group()
    if rank == 0:
        print("=== comm device-index checks passed ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    args = ap.parse_args()
    if torch_fl.flagos.device_count() < args.world_size:
        raise SystemExit(
            f"needs {args.world_size} flagos devices, "
            f"have {torch_fl.flagos.device_count()}"
        )
    mp.spawn(worker, args=(args.world_size,), nprocs=args.world_size, join=True)
