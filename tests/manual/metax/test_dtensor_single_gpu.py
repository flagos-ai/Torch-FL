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

"""Single-GPU DTensor coverage on the flagos backend (P2 local subset).

Multi-GPU collectives on this box are currently blocked by an MXKW driver
comm-queue leak (see P1 status), but a 1-rank DeviceMesh needs no collectives,
so the *entire local* surface of DTensor can be validated right now:

  * placement construction -- Shard(0)/Shard(1)/Replicate/Partial(reduce_op)
  * redistribute -- every legal direction is a local op at world_size == 1
  * sharding propagation -- elementwise / reduction / shape / matmul ops

Note: redistribute *into* Partial is rejected by DTensor by design ("Partial is
for internal use only"), so that direction is asserted as a RuntimeError rather
than a value check.

Run:
    FLAGOS_ACCELERATOR=metax python test_dtensor_single_gpu.py
"""

import os

import torch_fl  # noqa: F401
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29694")

dist.init_process_group(backend="flagos", rank=0, world_size=1)
mesh = init_device_mesh("flagos", (1,))
dev = torch.device("flagos:0")

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{'OK' if cond else 'FAIL'}] {name}", flush=True)


t = torch.arange(16, dtype=torch.float32, device=dev).reshape(4, 4)

# --- placement construction ---
for op in ("sum", "avg", "max", "min"):
    p = distribute_tensor(t, mesh, [Partial(op)])
    check(f"Partial('{op}') full_tensor==ref", torch.allclose(p.full_tensor(), t))

# --- redistribute: every legal direction (local at world_size == 1) ---
sh0 = distribute_tensor(t, mesh, [Shard(0)])
sh1 = distribute_tensor(t, mesh, [Shard(1)])
rep = distribute_tensor(t, mesh, [Replicate()])

for src_name, src in (("Shard(0)", sh0), ("Shard(1)", sh1), ("Replicate", rep)):
    for dst in (Shard(0), Shard(1), Replicate()):
        r = src.redistribute(placements=[dst])
        check(f"redistribute {src_name}->{dst}", torch.allclose(r.full_tensor(), t))

# redistribute INTO Partial is disallowed by design.
try:
    sh0.redistribute(placements=[Partial("sum")])
    check("redistribute ->Partial rejected", False)
except RuntimeError as e:
    check("redistribute ->Partial rejected", "internal use only" in str(e))

# --- sharding propagation matrix ---


def check_op(name, ref_fn, *dts):
    out = ref_fn(*dts)
    ref = ref_fn(t) if len(dts) == 1 else ref_fn(t, t)
    check(name, torch.allclose(out.full_tensor(), ref))


check_op("relu", lambda x: torch.relu(x), sh0)
check_op("mul", lambda x: x * 2, sh0)
check_op("add_broadcast", lambda x: x + 3, sh0)
check_op("sigmoid", lambda x: torch.sigmoid(x), sh0)
check_op("sum(dim=0)", lambda x: x.sum(dim=0), sh0)
check_op("mean(dim=1)", lambda x: x.mean(dim=1), sh0)
check_op("transpose", lambda x: x.transpose(0, 1), sh0)
check_op("view", lambda x: x.view(16), sh0)
check_op("cat", lambda x: torch.cat([x, x], dim=0), sh0)
check_op("stack", lambda x: torch.stack([x, x], dim=0), sh0)
check_op("matmul", lambda a, b: a @ b, sh0, sh1)
check_op("addmm", lambda a, b: torch.addmm(a, a, b), sh0, sh1)

n_fail = sum(1 for _, ok in results if not ok)
status = "ALL PASS" if n_fail == 0 else f"{n_fail} FAILED"
print(f"=== single-GPU DTensor: {status} ({len(results)} checks) ===", flush=True)
dist.destroy_process_group()
raise SystemExit(1 if n_fail else 0)
