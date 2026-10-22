#!/usr/bin/env python3
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

"""Regenerate ``tests/data/profiler_cuda_baseline.json`` from a real torch+cuda build.

This captures the *structural* shape of torch-cuda's chrome trace so that
``tests/integration/test_profiler_parity.py`` can assert flagos matches it.
Only structure is recorded -- category names and arg key sets. Counts and
durations are deliberately NOT recorded: this is a shared A100 and both drift
run to run, so a test that pinned them would flake.

Run this in the ``torch-cuda-210`` environment (torch 2.10.0+cu128, real CUDA).
It must NOT be run in ``torch-fl-211``: the two have incompatible libc10 ABIs,
and importing torch_fl here would defeat the purpose (the baseline has to come
from stock torch, not from the implementation under test).

    conda activate torch-cuda-210
    python tests/data/gen_profiler_baseline.py

The workload must stay in sync with ``run_traced_ops()`` in the parity test --
see the WORKLOAD note below.
"""

import json
import sys
import tempfile
from pathlib import Path

import torch

# Categories torch-cuda emits that flagos deliberately does not. Recorded (with
# the reason) rather than silently dropped, so the gap stays visible and a future
# task can decide to close it instead of rediscovering it.
KNOWN_GAPS = {
    "overhead": (
        "CUPTI_ACTIVITY_KIND_OVERHEAD is not among the activity kinds flagos "
        "enables (see cupti_device_tracer.cc: KERNEL/MEMCPY/MEMSET/RUNTIME/"
        "EXTERNAL_CORRELATION). These are CUPTI's self-reported profiling "
        "overheads ('Activity Buffer Request', 'Runtime Triggered Module "
        "Loading'), not user work, so their absence does not affect any "
        "measurement of the workload."
    ),
}


def run_traced_ops(device):
    """WORKLOAD -- must stay identical to the parity test's run_traced_ops().

    5x matmul+relu (from Task 1's verification gate: proven to produce sgemm
    kernels, elementwise kernels, cuBLAS workspace memsets and a D2H copy),
    plus a 16-element sort.

    The sort is not decoration. It is the only op in reach that launches a
    kernel whose block size is NOT a multiple of 32 (bitonicSortKVInPlace runs
    at block=[16,1,1]). Task 5 found that 'warps per SM' is plain division, not
    the ceil the spec claimed, and the two forms agree for *every* block size
    that is a multiple of 32 -- so without this kernel the warps-per-SM
    assertion cannot distinguish a correct implementation from a ceil bug.

    The LU factorisation is appended unconditionally here, unlike in the two
    flagos copies of this workload, which route it through
    ``profiler_support.append_boxing_path_probe``: stock torch+cuda always has
    cuSOLVER, while the flagos copies have to ask whether their conf sends
    ``linalg_lu_factor_ex`` to a device backend before paying for the call. It
    contributes cuSOLVER's getrf workspace memset and C++-template kernel names,
    which is what those copies need it for.
    """
    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)
    small = torch.randn(16, device=device)

    activity = (
        torch.profiler.ProfilerActivity.CUDA
        if device.startswith("cuda")
        else torch.profiler.ProfilerActivity.PrivateUse1
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, activity],
        with_stack=False,
    ) as prof:
        for _ in range(5):
            z = (x @ y).relu()
        torch.sort(small)
        torch.linalg.lu_factor(torch.randn(64, 64, device=device))
        z.sum().item()  # force sync so device activity lands inside the window

    return prof


def event_categories(trace):
    """Category names on completed (``ph == "X"``) events.

    Restricted to ``ph == "X"`` on purpose: metadata (``M``) and instant (``i``)
    events carry ``cat: null``, and flow events (``s``/``f``) are checked by the
    flow-pairing assertion rather than by category coverage.
    """
    return sorted(
        {
            e["cat"]
            for e in trace.get("traceEvents", [])
            if e.get("ph") == "X" and e.get("cat") is not None
        }
    )


def arg_keys(trace, category):
    """Union of ``args`` keys over every event in ``category``.

    Union, not the keys of one representative event: Task 5 measured that
    runtime events legitimately come in two shapes on BOTH stacks -- most carry
    ``External id``, a few (the ones kineto could not link) do not. Sampling a
    single event would record whichever shape happened to come first.
    """
    keys = set()
    found = False
    for e in trace.get("traceEvents", []):
        if e.get("cat") == category:
            found = True
            keys.update((e.get("args") or {}).keys())
    return sorted(keys) if found else None


def main():
    if not torch.cuda.is_available():
        print(
            "error: torch.cuda.is_available() is False.\n"
            "  Run this in the torch-cuda-210 env on a CUDA host; the baseline "
            "must come from stock torch+cuda, not from flagos.",
            file=sys.stderr,
        )
        return 2

    device_name = torch.cuda.get_device_name(0)
    print(f"torch {torch.__version__} / {device_name}")

    prof = run_traced_ops("cuda:0")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        trace_path = f.name
    try:
        prof.export_chrome_trace(trace_path)
        with open(trace_path) as f:
            trace = json.load(f)
    finally:
        Path(trace_path).unlink(missing_ok=True)

    observed = event_categories(trace)
    required = [c for c in observed if c not in KNOWN_GAPS]
    print(f"categories observed: {observed}")
    if set(observed) & set(KNOWN_GAPS):
        print(f"  minus known gaps  -> required: {required}")

    baseline = {
        "generated_by": (
            f"torch {torch.__version__} / {device_name} / gen_profiler_baseline.py"
        ),
        "note": (
            "Structure only. Counts and durations are deliberately absent -- "
            "they drift run to run on a shared GPU. Regenerate with "
            "tests/data/gen_profiler_baseline.py under torch-cuda-210."
        ),
        "categories": required,
        "categories_observed_in_torch_cuda": observed,
        "categories_known_gap": {
            c: reason for c, reason in KNOWN_GAPS.items() if c in observed
        },
        "runtime_cat_equivalents": ["cuda_runtime", "privateuse1_runtime"],
        "kernel_arg_keys": arg_keys(trace, "kernel"),
        "gpu_memcpy_arg_keys": arg_keys(trace, "gpu_memcpy"),
        "gpu_memset_arg_keys": arg_keys(trace, "gpu_memset"),
        "runtime_arg_keys": arg_keys(trace, "cuda_runtime"),
    }

    for key in (
        "kernel_arg_keys",
        "gpu_memcpy_arg_keys",
        "gpu_memset_arg_keys",
        "runtime_arg_keys",
    ):
        if baseline[key] is None:
            print(
                f"error: no events found for {key} -- the capture is incomplete, "
                "refusing to write a baseline that would under-constrain the test.",
                file=sys.stderr,
            )
            return 1
        print(f"{key}: {len(baseline[key])} keys {baseline[key]}")

    # Task 5 measured these on this machine. A capture that disagrees means
    # something regressed or the workload changed -- refuse rather than quietly
    # lowering the bar the parity test enforces.
    expected_counts = {
        "kernel_arg_keys": 13,
        "gpu_memcpy_arg_keys": 7,
        "gpu_memset_arg_keys": 7,
        "runtime_arg_keys": 3,
    }
    bad = {
        k: (len(baseline[k]), n)
        for k, n in expected_counts.items()
        if len(baseline[k]) != n
    }
    if bad:
        print(
            "error: captured key counts disagree with Task 5's measured ground "
            f"truth (got vs expected): {bad}\n"
            "  Investigate -- do not lower the baseline to match a bad capture.",
            file=sys.stderr,
        )
        return 1

    out = Path(__file__).parent / "profiler_cuda_baseline.json"
    out.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
