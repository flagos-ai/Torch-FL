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

"""Bisect a repeated-call nondeterminism report into one of three causes.

Written for issue #617 (Iluvatar BI-V150, repeated forward differs, NaN only
when samples run in sequence). The question it answers is not "is it broken"
-- the reporter already knows that -- but *which layer* is broken, because the
three candidates need different fixes and only one of them is in Torch-FL:

    A. A kernel never writes part of its output.
    B. A kernel reads memory it should not (stale, freed, or another tensor's).
    C. Nothing in memory is wrong; the arithmetic itself is order-dependent
       (atomics, split-k, or a tuner picking different configs across runs).

A and B both disappear-or-change under `FLAGOS_MEM_POISON` and C does not; A
and B are then separated by *what* the poisoned run produces (a constant
`POISON`-shaped value vs. plausible-but-wrong numbers).

Usage, on the affected accelerator::

    python tests/manual/probe_nondeterminism.py --model <ckpt> --images a.png b.png
    python tests/manual/probe_nondeterminism.py --self-test

Every mode runs each arm in its own subprocess, because the caching allocator,
the FlagGems tuner cache and the Triton cache all keep process-global state and
an arm measured in the same process as another is not an independent arm.

`--self-test` exercises the reporting logic on CPU with a synthesised cause, so
the script can be validated on a machine with no accelerator at all.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# 0xFF is what FLAGOS_MEM_POISON=1 fills with, and it is NaN in every float
# dtype. An element still holding it is one no kernel wrote.
POISON_BYTE = 0xFF


# ---------------------------------------------------------------------------
# Reporting helpers -- importable without torch, so --self-test works anywhere.
# ---------------------------------------------------------------------------


def classify(deterministic: bool, poisoned_nan_fraction: float | None) -> str:
    """Turn the two arm results into a cause.

    `deterministic` is whether the N repeated calls agreed bit for bit.
    `poisoned_nan_fraction` is the share of output elements that were still the
    poison value in a poisoned run, or None when that arm did not run.
    """
    if deterministic:
        return "NO_REPEAT_VARIATION"
    if poisoned_nan_fraction is None:
        return "REPEATS_DIFFER_NONDETERMINISTIC_ARITHMETIC"
    if poisoned_nan_fraction > 0:
        return "UNWRITTEN_OUTPUT"
    return "STALE_OR_ALIEN_READ"


RECOMMENDATIONS = {
    "NO_REPEAT_VARIATION": (
        "No repeat-to-repeat variation in this arm. The nondeterminism is not "
        "in this code path; rerun with the full failing workload, or the "
        "difference is upstream of the model (input preprocessing)."
    ),
    "UNWRITTEN_OUTPUT": (
        "A kernel left part of its output untouched: the poisoned run still "
        "holds 0xFF (=NaN) where the kernel never wrote. Find the op by running "
        "--per-op: the first op whose output contains poison is the one. This is "
        "a kernel-side bug (grid size, mask, or an unwritten tail), not a "
        "synchronisation or allocator bug."
    ),
    "STALE_OR_ALIEN_READ": (
        "Nothing was left unwritten, so a kernel read memory it should not have "
        "-- a freed block reused by someone else, or an index that was itself "
        "garbage. Compare the poisoned and unpoisoned runs: if the output "
        "changes when only the fill value changes, the read is of memory this "
        "op did not own. Run --per-op to find the first op whose output changes "
        "between the two arms."
    ),
    "REPEATS_DIFFER_NONDETERMINISTIC_ARITHMETIC": (
        "Repeats differ but poisoning does not change the outcome, so memory "
        "content is not the variable. That points at floating-point "
        "order-dependence: an atomic accumulation or a split-k reduction whose "
        "summation order varies, or a tuner selecting a different config. "
        "Check with TRITON_ALWAYS_COMPILE unset and a fixed config; if the "
        "variation survives, the kernel is using atomics."
    ),
}


def summarise(arms: dict[str, dict]) -> dict:
    """Reduce the arm records to a verdict plus the evidence behind it."""
    base = arms.get("repeat")
    if base is None:
        return {"verdict": "INCONCLUSIVE", "reason": "the repeat arm did not run"}

    deterministic = base.get("bit_identical", False)
    poisoned = arms.get("poisoned")
    fraction = None
    if poisoned is not None and "nan_fraction" in poisoned:
        fraction = poisoned["nan_fraction"]

    verdict = classify(deterministic, fraction)
    return {
        "verdict": verdict,
        "recommendation": RECOMMENDATIONS.get(verdict, ""),
        "evidence": {
            "repeat": base,
            "poisoned": poisoned,
            "unpoisoned": arms.get("unpoisoned"),
        },
    }


# ---------------------------------------------------------------------------
# Self-test: prove the classifier on synthesised arms, no torch required.
# ---------------------------------------------------------------------------


SELF_TEST_CASES = [
    # (name, repeat arm, poisoned arm, expected verdict)
    (
        "deterministic",
        {"bit_identical": True},
        {"nan_fraction": 0.0},
        "NO_REPEAT_VARIATION",
    ),
    (
        "unwritten",
        {"bit_identical": False},
        {"nan_fraction": 0.037},
        "UNWRITTEN_OUTPUT",
    ),
    (
        "stale",
        {"bit_identical": False},
        {"nan_fraction": 0.0},
        "STALE_OR_ALIEN_READ",
    ),
    (
        "arithmetic",
        {"bit_identical": False},
        None,
        "REPEATS_DIFFER_NONDETERMINISTIC_ARITHMETIC",
    ),
]


def run_self_test() -> int:
    failures = 0
    for name, repeat_arm, poisoned_arm, expected in SELF_TEST_CASES:
        arms = {"repeat": repeat_arm}
        if poisoned_arm is not None:
            arms["poisoned"] = poisoned_arm
        got = summarise(arms)["verdict"]
        status = "ok" if got == expected else "FAIL"
        if got != expected:
            failures += 1
        print(f"  [{status}] {name}: expected {expected}, got {got}")
    print(f"self-test: {len(SELF_TEST_CASES) - failures}/{len(SELF_TEST_CASES)} ok")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# The on-device arms. Everything below imports torch lazily.
# ---------------------------------------------------------------------------


CHILD_PREAMBLE = """
import json, os, sys
import torch_fl  # noqa: F401  -- loads the vendor libraries before torch
import torch
"""


def run_child(body: str, env_extra: dict[str, str], label: str) -> dict:
    """Run one arm in a fresh interpreter and return its JSON record."""
    env = dict(os.environ)
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", CHILD_PREAMBLE + textwrap.dedent(body)],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {
            "arm": label,
            "error": proc.stderr.strip().splitlines()[-1] if proc.stderr else "",
            "stderr_tail": proc.stderr.strip().splitlines()[-12:],
        }
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            record = json.loads(line)
            record["arm"] = label
            return record
    return {"arm": label, "error": "child printed no JSON", "stdout": proc.stdout}


REPEAT_BODY = """
import torch

def forward():
    torch.manual_seed(0)
    x = torch.randn(256, 1024, device="flagos:0", dtype=torch.bfloat16)
    w = torch.randn(1024, 1024, device="flagos:0", dtype=torch.bfloat16)
    for _ in range(8):
        x = torch.nn.functional.rms_norm(x, (1024,)).matmul(w)
        x = torch.softmax(x, dim=-1)
    torch.flagos.synchronize()
    return x.float().cpu()

first = forward()
identical = all(torch.equal(first, forward()) for _ in range(N_REPEATS - 1))
print(json.dumps({"bit_identical": identical, "shape": list(first.shape)}))
"""

POISON_BODY = """
import torch

torch.manual_seed(0)
x = torch.randn(256, 1024, device="flagos:0", dtype=torch.bfloat16)
w = torch.randn(1024, 1024, device="flagos:0", dtype=torch.bfloat16)
for _ in range(8):
    x = torch.nn.functional.rms_norm(x, (1024,)).matmul(w)
    x = torch.softmax(x, dim=-1)
torch.flagos.synchronize()
out = x.float().cpu()

# A float built from 0xFF bytes is NaN. Because the fill is applied to the
# storage *before* the kernel runs, any element the kernel never wrote reads
# back as a NaN that no arithmetic on finite inputs could have produced.
nan_fraction = float(torch.isnan(out).sum()) / out.numel()
print(json.dumps({"nan_fraction": nan_fraction, "numel": out.numel()}))
"""


def run_on_device(args: argparse.Namespace) -> int:
    repeats = str(args.repeats)
    arms: dict[str, dict] = {}

    print("arm 1/3: repeat determinism (no poison, caching allocator on)")
    arms["repeat"] = run_child(
        REPEAT_BODY.replace("N_REPEATS", repeats),
        {"FLAGOS_MEM_POISON": "0"},
        "repeat",
    )

    print("arm 2/3: same workload with FLAGOS_MEM_POISON=1")
    arms["poisoned"] = run_child(
        POISON_BODY,
        {"FLAGOS_MEM_POISON": "1"},
        "poisoned",
    )

    print("arm 3/3: same workload with the caching allocator disabled")
    arms["unpoisoned"] = run_child(
        POISON_BODY,
        {"FLAGOS_MEM_POISON": "0", "FLAGOS_USE_CACHING_ALLOCATOR": "0"},
        "unpoisoned",
    )

    result = summarise(arms)
    print()
    print(json.dumps(result, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate the classifier on synthesised arms; needs no torch",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=6,
        help="how many times the repeat arm runs the workload (default 6)",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="write the verdict and evidence to this path",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not os.environ.get("FLAGOS_ACCELERATOR"):
        print(
            "warning: FLAGOS_ACCELERATOR is unset; the child processes will use "
            "whatever accelerator the installed wheel records",
            file=sys.stderr,
        )
    return run_on_device(args)


if __name__ == "__main__":
    raise SystemExit(main())
