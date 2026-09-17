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

"""
FlagGems dispatch-overhead microbenchmark -- driver.

Runs the worker once per backend in its own subprocess (routing is fixed at
import), then prints a side-by-side table comparing three dispatch paths for
the SAME op on the SAME device (flagos:0, shared GPU memory):

    cuda     : default vendor conf (backends_cuda.conf), pure C++ dispatch
               into the vendor libtorch_cuda kernel.
    gems_py  : FLAGOS_USE_FLAGGEMS=1 (backends_flaggems.conf), Python dispatch
               -- kFlagOsPython crosses into CPython + pybind to launch the
               FlagGems Triton kernel.
    gems_cpp : FLAGOS_USE_FLAGGEMS_CPP=1 (backends_flaggems_cpp.conf), C++
               dispatch -- kFlagOs boxes flagos->cuda metadata and calls the
               FlagGems C++ runtime (liboperators.so, TritonJIT, no GIL).
               Only the 18 Stage-A ops route here; the rest still hit gems_py,
               so gems_cpp == gems_py for non-C++ ops (mul/abs/neg -- flagged).

Two host-side deltas matter:
    py_tax   = gems_py_submit  - cuda_submit  (cost of the Python dispatch path)
    cpp_save = gems_py_submit  - gems_cpp_submit
             = how much of that tax the C++ dispatch actually recovers.

gems_cpp requires a wheel built with FLAGGEMS_CPP=ON and the three runtime
env vars (FLAGGEMS_SOURCE_DIR, LD_LIBRARY_PATH -> liboperators.so). If they are
missing the driver skips that column instead of failing.

Usage:
    python benchmarks/flaggems_dispatch_bench.py
    python benchmarks/flaggems_dispatch_bench.py --submit 500 --e2e 200
    # gems_cpp column additionally needs (single line):
    FLAGGEMS_SOURCE_DIR=/path/to/FlagGems-src/src/flag_gems \
    LD_LIBRARY_PATH=/path/to/FlagGems-src/cpp/build/lib:$LD_LIBRARY_PATH \
    python benchmarks/flaggems_dispatch_bench.py
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_WORKER = Path(__file__).resolve().parent / "flaggems_dispatch_overhead.py"

# Ops that actually route to the C++ kFlagOs backend under backends_flaggems_cpp.conf
# (Stage A). For every other op gems_cpp is identical to gems_py (still Python).
_CPP_OPS = {
    "mm",
    "bmm",
    "bmm.out",
    "addmm",
    "addmm.out",
    "embedding",
    "_softmax",
    "_softmax_backward_data",
    "sum",
    "sum.dim_IntList",
    "max",
    "max.dim",
    "argmax",
    "nonzero",
    "sort",
    "sort.stable",
    "topk",
    "zeros",
}


def _op_is_cpp(label):
    """The bench labels ops like 'mm[512x512]' / 'sum.dim[1024x1024]'; map the
    label stem back to the aten op name used in the conf."""
    stem = label.split("[")[0]
    # bench uses sum.dim / softmax; conf uses sum.dim_IntList / _softmax
    alias = {"sum.dim": "sum.dim_IntList", "softmax": "_softmax"}
    return alias.get(stem, stem) in _CPP_OPS


def _run(backend, env_extra, submit, e2e, warmup):
    env = os.environ.copy()
    env.update(env_extra)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    proc = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "--backend",
            backend,
            "--submit",
            str(submit),
            "--e2e",
            str(e2e),
            "--warmup",
            str(warmup),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"worker for backend={backend} failed")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _cpp_available():
    """gems_cpp needs liboperators.so reachable and FLAGGEMS_SOURCE_DIR set;
    otherwise the worker import (or first C++ op) fails. Detect up front."""
    if not os.environ.get("FLAGGEMS_SOURCE_DIR"):
        return False, "FLAGGEMS_SOURCE_DIR not set"
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if not any((Path(p) / "liboperators.so").exists() for p in ld.split(":") if p):
        return False, "liboperators.so not on LD_LIBRARY_PATH"
    return True, ""


def _index(blob):
    return {r["op"]: r for r in blob["results"]}


def _fmt(v):
    return f"{v:8.1f}" if isinstance(v, (int, float)) else f"{'ERR':>8}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", type=int, default=300)
    ap.add_argument("--e2e", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    cuda = _index(_run("cuda", {}, args.submit, args.e2e, args.warmup))
    gems = _index(
        _run(
            "gems_py", {"FLAGOS_USE_FLAGGEMS": "1"}, args.submit, args.e2e, args.warmup
        )
    )

    cpp_ok, cpp_why = _cpp_available()
    cpp = {}
    if cpp_ok:
        cpp = _index(
            _run(
                "gems_cpp",
                {"FLAGOS_USE_FLAGGEMS_CPP": "1"},
                args.submit,
                args.e2e,
                args.warmup,
            )
        )
    else:
        sys.stderr.write(f"[skip gems_cpp] {cpp_why}\n")

    print()
    print("Per-op host submit time and end-to-end latency (microseconds).")
    print("submit = host dispatch cost (kernel time hidden by the async queue).")
    print("  cuda     : vendor C++ dispatch     gems_py : Python (kFlagOsPython)")
    print("  gems_cpp : FlagGems C++ (kFlagOs)   * = op routes to C++ under cpp conf")
    print("py_tax   = gems_py - cuda    (Python dispatch overhead vs vendor)")
    print("cpp_save = gems_py - gems_cpp (host cost the C++ dispatch recovers)\n")
    hdr = (
        f"{'op':22} {'cat':10} {'cuda_sub':>9} {'gpy_sub':>9} {'gcpp_sub':>9} "
        f"{'py_tax':>8} {'cpp_save':>9} {'gpy_e2e':>9} {'gcpp_e2e':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for op in cuda:
        c, g = cuda[op], gems.get(op, {})
        x = cpp.get(op, {}) if cpp else {}
        cs, gs, xs = c.get("submit_us"), g.get("submit_us"), x.get("submit_us")
        py_tax = (
            (gs - cs)
            if isinstance(cs, (int, float)) and isinstance(gs, (int, float))
            else "ERR"
        )
        cpp_save = (
            (gs - xs)
            if isinstance(gs, (int, float)) and isinstance(xs, (int, float))
            else ("n/a" if not cpp else "ERR")
        )
        mark = " *" if _op_is_cpp(op) else ""
        print(
            f"{op + mark:22} {c.get('category', ''):10} {_fmt(cs)} {_fmt(gs)} "
            f"{_fmt(xs)} {_fmt(py_tax)} {_fmt(cpp_save)} "
            f"{_fmt(g.get('e2e_us'))} {_fmt(x.get('e2e_us'))}"
        )

    if cpp:
        print(
            "\nNote: cpp_save is only meaningful on ops marked * (the 18 Stage-A "
            "C++ ops).\nFor unmarked ops gems_cpp still uses the Python path, so "
            "any delta is measurement noise."
        )


if __name__ == "__main__":
    main()
