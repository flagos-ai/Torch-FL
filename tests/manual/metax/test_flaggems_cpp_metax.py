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

"""Verify the FlagGems C++ dispatch path (kFlagGemsCpp) on MetaX.

This is the MetaX counterpart of tests/integration/ops/test_flaggems_cpp_dispatch.py.

Background: the C++ FlagGems path calls flag_gems' C++ entry points in
liboperators.so, which JIT-compile and launch Triton kernels without touching
Python or the GIL. It was previously CUDA-only on the torch_fl side because
CMakeLists forced FLAGGEMS_CPP=OFF for ACCELERATOR=metax. FlagGems itself
does support MetaX (cpp/ -DFLAGGEMS_BACKEND=MACA), and its kernels reach the
device through the same DeviceBoxingGuard the metax boxing path already uses,
so the C++ path works here once liboperators.so is built for MACA.

Build prerequisites:
  1. FlagGems C++ for MACA:
       cd FlagGems/cpp && cmake -B build-maca \
           -DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_BACKEND=MACA \
           -DMACA_PATH=/opt/maca \
           -DPython_EXECUTABLE=$(which python) \
           -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
       cmake --build build-maca --target operators -j
     Note: a few C++ ops (zeros, fill, copy, ...) load their Triton source at
     runtime from <FlagGems>/cpp/triton_src/, which only exists after
     `cmake --install`. When running straight out of the build dir, link it:
       ln -sfn <FlagGems>/triton_src <FlagGems>/cpp/triton_src
  2. torch_fl against it:
       ACCELERATOR=metax VENDOR_KERNEL=OFF MACA_PATH=/opt/maca \
       FLAGGEMS_CPP=1 FLAGGEMS_DIR=<FlagGems>/cpp/build-maca \
       python setup.py build_ext --inplace

Run (from repo root):
    ACCELERATOR=metax \
    MACA_PATH=/opt/maca METAX_PATH=/opt/maca \
    LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/lib64:$LD_LIBRARY_PATH \
    PYTHONPATH=$PWD \
    python tests/manual/metax/test_flaggems_cpp_metax.py

Each op is checked twice:
  * routing -- with FLAGOS_LOG_DISPATCH=1 the dispatcher logs "-> flagos" for
    the C++ backend (vs "-> flagos_python" / "-> cuda"). Correct numerics alone
    would not prove the C++ path ran, since the boxing fallback is also correct.
  * numerics -- compared against the CPU result.
Routing is checked in a subprocess because the backend table is read once, at
the first dispatch, from FLAGOS_BACKEND_CONFIG.
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def _run_snippet(code):
    """Run code in a fresh interpreter with the C++ FlagGems path enabled."""
    env = os.environ.copy()
    env["FLAGOS_USE_FLAGGEMS_CPP"] = "1"
    env["FLAGOS_LOG_DISPATCH"] = "1"

    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    # The dispatch log is what proves the C++ path ran; keep stderr separate.
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO_ROOT,
    )
    return proc


_PREAMBLE = """
import torch_fl  # noqa: F401  MUST precede torch (boxing preload + GEMS_VENDOR)
import torch

dev = torch.device("flagos:0")
torch.cuda.set_device(0)
"""


def test_conf_selected():
    """CPP + boxing must select the MetaX C++ conf, not the CUDA one."""
    proc = _run_snippet(
        _PREAMBLE
        + """
import os
print("CONF=" + os.environ.get("FLAGOS_BACKEND_CONFIG", "<unset>"))
"""
    )
    conf = ""
    for line in proc.stdout.splitlines():
        if line.startswith("CONF="):
            conf = line[len("CONF=") :]
    ok = conf.endswith("backends_metax.conf")
    check("CPP+boxing selects metax cpp conf", ok, conf or proc.stderr[-300:])


def _routing_case(name, snippet, op_label):
    """Assert the dispatch log shows op_label routed to the C++ 'flagos' backend."""
    proc = _run_snippet(_PREAMBLE + snippet)
    log = proc.stdout + proc.stderr
    # Dispatch log lines look like: "flagos dispatch: mm -> flagos"
    hit = [
        ln
        for ln in log.splitlines()
        if op_label in ln and "->" in ln and ln.rstrip().endswith("flagos")
    ]
    ok = bool(hit)
    detail = hit[0].strip() if hit else f"no '-> flagos' line for {op_label}"
    if not ok:
        # Surface what it actually routed to, to distinguish "fell back to
        # boxing" from "crashed".
        other = [ln for ln in log.splitlines() if op_label in ln and "->" in ln]
        if other:
            detail = f"routed elsewhere: {other[0].strip()}"
        elif proc.returncode != 0:
            detail = f"exit {proc.returncode}: {log.strip()[-300:]}"
    check(f"{name} routes to C++ flagos", ok, detail)


def test_mm_routes_to_boxing():
    """mm must NOT take the C++ path on MetaX (shared-memory limit).

    flag_gems' C++ mm_kernel_general requests 98304 bytes of shared memory;
    C550 provides 65536, so mcModuleLaunchKernel returns mcErrorInvalidValue.
    backends_metax_flaggems_cpp.conf routes mm to the cuda boxing kernel.
    """
    proc = _run_snippet(
        _PREAMBLE
        + """
a = torch.randn(64, 64, device=dev)
c = a @ a
ref = a.cpu() @ a.cpu()
print("MM_OK=" + str(bool(torch.allclose(c.cpu(), ref, atol=1e-2, rtol=1e-2))))
"""
    )
    log = proc.stdout + proc.stderr
    routed = [
        ln
        for ln in log.splitlines()
        if "] mm ->" in ln or ln.strip().startswith("mm ->")
    ]
    to_cuda = bool(routed) and routed[0].rstrip().endswith("cuda")
    check(
        "mm routes to cuda boxing (not C++)",
        to_cuda,
        routed[0].strip() if routed else "no mm dispatch line",
    )
    ok = "MM_OK=True" in log
    check("mm numerics via boxing", ok, "" if ok else log.strip()[-300:])


def test_routing():
    """Each C++-backed op must log '-> flagos' (the kFlagGemsCpp backend)."""
    _routing_case(
        "bmm", "a = torch.randn(4, 32, 32, device=dev)\nc = torch.bmm(a, a)\n", "bmm"
    )
    _routing_case(
        "addmm",
        "a = torch.randn(32, 32, device=dev)\nb = torch.randn(32, device=dev)\n"
        "c = torch.addmm(b, a, a)\n",
        "addmm",
    )
    _routing_case(
        "_softmax",
        "a = torch.randn(32, 32, device=dev)\nc = torch.softmax(a, dim=-1)\n",
        "_softmax",
    )
    _routing_case("sum", "a = torch.randn(64, 64, device=dev)\nc = a.sum()\n", "sum")
    _routing_case(
        "embedding",
        "w = torch.randn(16, 8, device=dev)\n"
        "i = torch.randint(0, 16, (4,), device=dev)\n"
        "c = torch.nn.functional.embedding(i, w)\n",
        "embedding",
    )
    _routing_case("max", "a = torch.randn(64, device=dev)\nc = a.max()\n", "max")
    _routing_case(
        "argmax", "a = torch.randn(64, device=dev)\nc = a.argmax()\n", "argmax"
    )
    _routing_case(
        "sort", "a = torch.randn(64, device=dev)\nv, i = torch.sort(a)\n", "sort"
    )
    _routing_case(
        "topk", "a = torch.randn(64, device=dev)\nv, i = torch.topk(a, 5)\n", "topk"
    )
    _routing_case(
        "nonzero",
        "a = torch.tensor([0.0, 1.0, 0.0, 2.0], device=dev)\nc = torch.nonzero(a)\n",
        "nonzero",
    )


def test_numerics():
    """C++ FlagGems kernels must produce correct results (vs CPU)."""
    proc = _run_snippet(
        _PREAMBLE
        + """
import torch
ok = {}

torch.manual_seed(0)
b = torch.randn(4, 16, 16)
bd = b.to(dev)
ok["bmm"] = torch.allclose(torch.bmm(bd, bd).cpu(), torch.bmm(b, b), atol=1e-3, rtol=1e-3)

s = torch.randn(32, 32)
sd = s.to(dev)
ok["softmax"] = torch.allclose(
    torch.softmax(sd, dim=-1).cpu(), torch.softmax(s, dim=-1), atol=1e-3, rtol=1e-3)

ok["sum"] = torch.allclose(sd.sum().cpu(), s.sum(), atol=1e-2, rtol=1e-3)
ok["sum.dim"] = torch.allclose(
    sd.sum(dim=0).cpu(), s.sum(dim=0), atol=1e-2, rtol=1e-3)

w = torch.randn(16, 8)
idx = torch.randint(0, 16, (5,))
wd, idxd = w.to(dev), idx.to(dev)
ok["embedding"] = torch.allclose(
    torch.nn.functional.embedding(idxd, wd).cpu(),
    torch.nn.functional.embedding(idx, w), atol=1e-4, rtol=1e-4)

m = torch.randn(64)
md = m.to(dev)
ok["max"] = torch.allclose(md.max().cpu(), m.max(), atol=1e-4, rtol=1e-4)
ok["argmax"] = int(md.argmax().cpu()) == int(m.argmax())
ok["sort"] = torch.allclose(torch.sort(md).values.cpu(), torch.sort(m).values, atol=1e-4)
ok["topk"] = torch.allclose(
    torch.topk(md, 5).values.cpu(), torch.topk(m, 5).values, atol=1e-4)

nz = torch.tensor([0.0, 1.0, 0.0, 2.0], device=dev)
ok["nonzero"] = torch.nonzero(nz).cpu().flatten().tolist() == [1, 3]

for k, v in ok.items():
    print(f"NUM {k}={v}")
"""
    )
    seen = {}
    for line in proc.stdout.splitlines():
        if line.startswith("NUM "):
            k, _, v = line[4:].partition("=")
            seen[k] = v == "True"
    if not seen:
        check(
            "numerics ran",
            False,
            f"exit {proc.returncode}: {(proc.stdout + proc.stderr).strip()[-400:]}",
        )
        return
    for k, v in seen.items():
        check(f"{k} numerics", v)


def test_liboperators_linked():
    """libtorch_fl.so must resolve liboperators.so (the C++ FlagGems runtime)."""
    lib = os.path.join(REPO_ROOT, "torch_fl", "lib", "libtorch_fl.so")
    proc = subprocess.run(["ldd", lib], capture_output=True, text=True)
    line = [ln for ln in proc.stdout.splitlines() if "liboperators" in ln]
    ok = bool(line) and "not found" not in line[0]
    check(
        "libtorch_fl.so resolves liboperators.so",
        ok,
        line[0].strip() if line else "liboperators.so not in ldd output",
    )


if __name__ == "__main__":
    for fn in (
        test_liboperators_linked,
        test_conf_selected,
        test_mm_routes_to_boxing,
        test_routing,
        test_numerics,
    ):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            check(fn.__name__, False, f"raised {type(e).__name__}: {e}")

    n_fail = sum(1 for _, ok in results if not ok)
    status = "ALL PASS" if n_fail == 0 else f"{n_fail} FAILED"
    print(f"=== flaggems cpp metax: {status} ({len(results)} checks) ===")
    raise SystemExit(1 if n_fail else 0)
