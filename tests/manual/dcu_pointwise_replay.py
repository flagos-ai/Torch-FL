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

"""Replay the pointwise overloads `backends_dcu.conf` pins to CUDA boxing.

The DCU conf routes eleven `add`/`sub`/`div` overloads to the CUDA boxing kernel
because their FlagGems kernels do not compile on the HCU backend. A pinned entry
is invisible to `flaggems_overload_survey.py` by construction -- that survey
enumerates the ops a conf *file* spells `flaggems`, so pinning an op is exactly
what removes it from the survey's denominator -- and the survey's profile cohort
does not reach either defect anyway. This harness is the evidence that stays
re-runnable afterwards.

Each case runs in its own subprocess with `FLAGOS_OP_<op>=flaggems` (dots
doubled, see `LoadBackendConfig` in `csrc/aten/common.cc`), which puts that one
entry back on FlagGems for that process and leaves the other ten shipped. A
fresh process per case is not optional: the f64 path trips an assertion in the
HCU backend that aborts the interpreter, so anything sharing the process after
it would lose its verdict rather than report one.

Arguments are chosen to reach the entry the way production does. Two details
decide whether a case proves anything:

* ATen boxes a Python float as an f64 wrapped number, so `add.Tensor` with a
  bf16 tensor and a Python float is what puts an f64 operand beside a bf16 one
  in the FlagGems pointwise kernel. The tensor's dtype and the operand's being a
  float are both load-bearing. `QwenImageRMS_norm` reaches it with
  `normalized * self.scale * self.gamma + self.bias`, where `self.bias` is the
  Python float 0.0 whenever the layer is built without a bias.
* The four `*_mode` entries only reach FlagGems' `div_rn` kernels under a named
  rounding mode; `rounding_mode=None` routes to `true_divide`, which is plain
  `x / y` and runs. The survey synthesizes `None`, so it measures these as
  passing.

Run, on a host whose `flagos` backend is the Hygon DCU build:

    python tests/manual/dcu_pointwise_replay.py
    python tests/manual/dcu_pointwise_replay.py --device flagos:0 \
        --ops add.Tensor,div.Scalar_mode

A pinned entry that still fails on the FlagGems route reports the exception and
the innermost flag_gems frame, which is the expected verdict; `ran` on one of
them means the pin is no longer needed for that entry.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The argument shape each entry is reached with in production: a bf16 tensor and
# a Python float. Every entry takes the same pair -- that is what makes the
# table comparable across the family.
CASES = (
    "add.Tensor",
    "add_.Tensor",
    "sub.Tensor",
    "sub_.Tensor",
    "div.Tensor",
    "div_.Tensor",
    "div.out",
    "div.Tensor_mode",
    "div_.Tensor_mode",
    "div.Scalar_mode",
    "div_.Scalar_mode",
)

SHAPE = (1, 384, 1, 116, 208)

CHILD = r"""
import json
import sys
import traceback

import torch_fl  # noqa: F401  - must precede `import torch`
import torch

op = sys.argv[1]
shape = tuple(json.loads(sys.argv[2]))
device = torch.device(sys.argv[3])

t = torch.randn(*shape, dtype=torch.bfloat16, device=device)
o = torch.empty_like(t)

CASES = {
    "add.Tensor": lambda: torch.ops.aten.add.Tensor(t, 2.0),
    "add_.Tensor": lambda: torch.ops.aten.add_.Tensor(t, 2.0),
    "sub.Tensor": lambda: torch.ops.aten.sub.Tensor(t, 2.0),
    "sub_.Tensor": lambda: torch.ops.aten.sub_.Tensor(t, 2.0),
    "div.Tensor": lambda: torch.ops.aten.div.Tensor(t, 2.0),
    "div_.Tensor": lambda: torch.ops.aten.div_.Tensor(t, 2.0),
    "div.out": lambda: torch.ops.aten.div.out(t, 2.0, out=o),
    "div.Tensor_mode": lambda: torch.ops.aten.div.Tensor_mode(
        t, 2.0, rounding_mode="trunc"
    ),
    "div_.Tensor_mode": lambda: torch.ops.aten.div_.Tensor_mode(
        t, 2.0, rounding_mode="trunc"
    ),
    "div.Scalar_mode": lambda: torch.ops.aten.div.Scalar_mode(
        t, 2.0, rounding_mode="trunc"
    ),
    "div_.Scalar_mode": lambda: torch.ops.aten.div_.Scalar_mode(
        t, 2.0, rounding_mode="trunc"
    ),
}

try:
    CASES[op]()
    torch.cuda.synchronize()
except Exception as error:  # noqa: BLE001
    frames = [
        f"{frame.filename.rsplit('/', 2)[-1]}:{frame.lineno}"
        for frame in traceback.extract_tb(error.__traceback__)
        if "/flag_gems/" in frame.filename or "/triton/" in frame.filename
    ]
    cause = error
    chain = []
    while cause is not None and len(chain) < 6:
        chain.append(f"{type(cause).__name__}: {str(cause).strip().splitlines()[-1]}")
        cause = cause.__cause__
    print("RESULT " + json.dumps({
        "verdict": type(error).__name__,
        "frame": frames[-1] if frames else None,
        "causes": chain,
    }))
    sys.exit(1)
print("RESULT " + json.dumps({"verdict": "ran", "frame": None, "causes": []}))
"""


def env_override(op: str) -> str:
    """`FLAGOS_OP_<op>` with dots doubled, matching LoadBackendConfig."""
    return "FLAGOS_OP_" + op.replace(".", "__")


def run_case(op: str, device: str, timeout: int) -> dict:
    env = dict(os.environ)
    env[env_override(op)] = "flaggems"
    child = Path("/tmp") / f"dcu_pointwise_replay_{op.replace('.', '_')}.py"
    child.write_text(CHILD, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(child), op, json.dumps(list(SHAPE)), device],
        capture_output=True,
        text=True,
        env=env,
        cwd="/tmp",
        timeout=timeout,
        check=False,
    )
    out = proc.stdout + proc.stderr
    result = {"op": op, "returncode": proc.returncode, "assert": None}
    for line in out.splitlines():
        if line.startswith("RESULT "):
            result.update(json.loads(line[len("RESULT ") :]))
    if "unsupported conversion" in out:
        result["assert"] = "ElementwiseOpHCUToLLVM TruncFOpConversion"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="flagos:0")
    parser.add_argument("--ops", help="comma-separated subset of the pinned entries")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    ops = tuple(args.ops.split(",")) if args.ops else CASES
    unknown = [op for op in ops if op not in CASES]
    if unknown:
        print(f"unknown op(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    conf = REPO / "torch_fl/configs/backends_dcu.conf"
    routes = {}
    for line in conf.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        op_name, sep, backend = line.partition(" = ")
        if sep:
            routes[op_name.strip()] = backend.split("#")[0].strip()
    pinned = [op for op in CASES if routes.get(op) == "cuda"]
    print(f"pinned to cuda in {conf.name}: {len(pinned)}/{len(CASES)}")
    print(f"{'op':18s} {'route':9s} verdict")
    for op in ops:
        shipped = routes.get(op, "(absent)")
        result = run_case(op, args.device, args.timeout)
        verdict = result.get("verdict", f"no verdict (exit {result['returncode']})")
        cause = (result.get("causes") or ["-"])[-1]
        detail = result["assert"] or cause
        print(f"{op:18s} {'flaggems':9s} {verdict}  {detail}   [shipped: {shipped}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
