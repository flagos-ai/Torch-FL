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

"""hbdk4 invocation and compile caching.

A partition is exported to ONNX, compiled to a .hbm by hbdk4, and the artifact
is cached on disk keyed by graph structure so repeated runs skip compilation.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from torch.fx import GraphModule

from torch_fl import _env

log = logging.getLogger("torch_fl.bpu")

# BPU micro-architecture. The BPU is nash-p; nash-e is S100 and nash-m is
# S100P. Confirmed against the vendor docs and the march string embedded in
# /opt/hobot/model/bpu/*.hbm.
DEFAULT_MARCH = _env.value("FLAGOS_BPU_MARCH", "nash-p")

CACHE_DIR = Path(
    _env.path("FLAGOS_BPU_CACHE", str(Path.home() / ".cache" / "torch_fl_bpu"))
)

# Quantization is on by default: without it hbdk4 keeps conv in float and
# lowers it to the CPU, so the BPU never runs the heavy work. Set
# FLAGOS_BPU_QUANTIZE=0 to compile float artifacts (bit-exact, but no speedup).
QUANTIZE = _env.flag("FLAGOS_BPU_QUANTIZE", True)

# Fallback activation scale for tensors with no calibration entry. int8 symmetric
# with this scale covers roughly +-6.35, wide enough for post-BN/ReLU activations.
ACT_SCALE = float(_env.value("FLAGOS_BPU_ACT_SCALE", "0.05"))


class CompileError(RuntimeError):
    """hbdk4 was unavailable or rejected the graph."""


# An x86_64 CPython that has hbdk4 installed, run under an emulator on this
# aarch64 board. hbdk4 ships x86_64-only wheels, so this is the only way to
# compile on the board itself.
X86_PYTHON = _env.value("FLAGOS_BPU_X86_PYTHON", "")

# Explicit emulator override. Useful because the distro box64 is usually too old
# (see x86_emulator) and a self-built one is not on PATH.
X86_EMULATOR = _env.value("FLAGOS_BPU_X86_EMULATOR", "")

# Directory holding import-only stand-ins for numba and torch. hbdk4's ONNX
# entry point imports both unconditionally, but only calls into them for
# custom/numba ops, which a graph exported from ONNX standard operators never
# has. Real numba cannot be used here at all: it imports llvmlite.binding, whose
# x86 LLVM JIT segfaults under box64. Real torch would work but costs several
# hundred MB in the emulated environment for code that never runs.
#
# Defaults to <x86 python prefix>/../stubs so a self-contained setup needs no
# extra configuration; see docs/vendors/bpu/integration.md.
X86_STUBS = _env.value("FLAGOS_BPU_X86_STUBS", "")

_UNSET = object()
_emulator: tuple[str, ...] | None | object = _UNSET


def _stub_dir() -> str | None:
    """Directory of the numba/torch import stubs, or None if absent."""
    if X86_STUBS:
        return X86_STUBS if Path(X86_STUBS).is_dir() else None
    if not X86_PYTHON:
        return None
    # .../<root>/python/bin/python3.11 -> .../<root>/stubs
    guess = Path(X86_PYTHON).resolve().parent.parent.parent / "stubs"
    return str(guess) if guess.is_dir() else None


def _mlir_libs_dir() -> str | None:
    """hbdk4's bundled shared-library directory inside the x86 environment.

    box64 needs this on BOX64_LD_LIBRARY_PATH, and libhbtl.so needs an explicit
    RTLD_GLOBAL preload: _hbdk*.so expects several hbtl symbols
    (hbtl::Storage::createExternal, hbtl::getStrides) to be resolvable, but does
    not list libhbtl.so in its own DT_NEEDED -- it inherits them transitively
    from libHBDKPythonCAPI.so, which box64 does not reproduce. Without the
    preload the dlopen fails with "cannot apply R_X86_64_JUMP_SLOT".
    """
    if not X86_PYTHON:
        return None
    root = Path(X86_PYTHON).resolve().parent.parent  # .../python
    hits = sorted(root.glob("lib/python3.*/site-packages/hbdk4/compiler/_mlir_libs"))
    return str(hits[0]) if hits else None


def x86_env() -> dict[str, str]:
    """Environment for running the x86_64 hbdk4 under an emulator."""
    env = dict(os.environ)
    libs = _mlir_libs_dir()
    if libs:
        # box64 resolves the guest's libraries through its own search path, not
        # the host loader's, so RUNPATH=$ORIGIN inside _hbdk.so is not enough.
        env["BOX64_LD_LIBRARY_PATH"] = (
            f"{libs}:{env['BOX64_LD_LIBRARY_PATH']}"
            if env.get("BOX64_LD_LIBRARY_PATH")
            else libs
        )
    stubs = _stub_dir()
    if stubs:
        # Appended, not prepended: a real numba/torch in the guest's
        # site-packages should win if one is ever installed there.
        env["PYTHONPATH"] = (
            f"{env['PYTHONPATH']}:{stubs}" if env.get("PYTHONPATH") else stubs
        )
    return env


def x86_emulator(refresh: bool = False) -> tuple[str, ...] | None:
    """Command prefix that runs an x86_64 hbdk4 locally, or None.

    Requires FLAGOS_BPU_X86_PYTHON to point at an x86_64 python with
    hbdk4-compiler installed, plus an emulator to run it.

    **box64 must be recent.** Its page-size handling used to be a build-time
    constant, so the widely packaged 0.2.6 (Debian/Ubuntu) aborts on this board
    with "PageSize configuration is wrong: configured with 4096, but got 65536".
    Current box64 reads the host page size at runtime (`box64_pagesize =
    sysconf(_SC_PAGESIZE)`) and maps 4 KB-aligned x86 PT_LOAD segments onto
    64 KB pages itself, so **the stock 64 KB-page kernel is fine** and no kernel
    rebuild is needed. Verified with v0.4.5. Point
    FLAGOS_BPU_X86_EMULATOR at a self-built box64 if the packaged one is old.

    qemu-user is still tried, but it has no equivalent fix: it fails with SIGBUS
    on any .so whose segments are 4 KB-aligned, and pip segfaults under it.
    """
    global _emulator
    if _emulator is not _UNSET and not refresh:
        return _emulator  # type: ignore[return-value]

    _emulator = None
    if not X86_PYTHON:
        return None

    # Preload libhbtl.so before importing, exactly as the compile driver does,
    # so the probe reflects whether a real compile would work.
    libs = _mlir_libs_dir()
    preload = (
        f"import ctypes; ctypes.CDLL({str(Path(libs) / 'libhbtl.so')!r}, "
        "mode=ctypes.RTLD_GLOBAL)\n"
        if libs
        else ""
    )
    probe = preload + "import hbdk4.compiler"

    candidates = [X86_EMULATOR] if X86_EMULATOR else []
    candidates += ["box64", "qemu-x86_64-static", "qemu-x86_64"]

    env = x86_env()
    for emul in candidates:
        exe = shutil.which(emul) or (emul if Path(emul).is_file() else None)
        if not exe:
            continue
        try:
            proc = subprocess.run(
                [exe, X86_PYTHON, "-c", probe],
                capture_output=True,
                text=True,
                timeout=900,
                env=env,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if proc.returncode == 0:
            _emulator = (exe, X86_PYTHON)
            log.info("hbdk4 reachable via %s", Path(exe).name)
            break
        log.debug("%s could not run hbdk4: %s", emul, proc.stderr[-500:])

    return _emulator  # type: ignore[return-value]


def find_hbdk() -> str | None:
    """Locate a usable hbdk4 compiler driver, if any.

    Returns "python-api" for a native import, "x86-emul" when hbdk4 is only
    reachable through an x86_64 emulator (see `x86_emulator`), a path for a CLI
    driver, or None.
    """
    try:
        import hbdk4.compiler  # noqa: F401

        return "python-api"
    except ImportError:
        pass

    for name in ("hbdk-cc", "hbdk4-cc", "hbdk4-compile", "hb_compile"):
        path = shutil.which(name)
        if path:
            return path

    return "x86-emul" if x86_emulator() else None


def graph_key(gm: GraphModule, example_inputs: list[torch.Tensor], march: str) -> str:
    """Stable hash over graph structure, weights, input signature and arch.

    The weights have to be in the key. `backend._frozen_weights` bakes them into
    the artifact as ONNX initializers, so two models with the same architecture
    and different weights produce genuinely different .hbm files -- hashing only
    the structure would make the second model silently reuse the first one's
    artifact and return its answers.

    Large tensors are sampled rather than hashed whole: a strided sample plus
    shape, dtype and the exact byte count separates any two weight sets that
    training would realistically produce, and keeps key computation off the
    critical path for multi-megabyte parameters.
    """
    h = hashlib.sha256()
    h.update(march.encode())
    for node in gm.graph.nodes:
        h.update(f"{node.op}:{node.target}:".encode())
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            h.update(f"{tuple(val.shape)}:{val.dtype}".encode())
    for name, t in sorted(_weight_tensors(gm)):
        h.update(f"{name}:{tuple(t.shape)}:{t.dtype}:".encode())
        h.update(_tensor_digest(t))
    for t in example_inputs:
        h.update(f"{tuple(t.shape)}:{t.dtype}".encode())
    return h.hexdigest()[:16]


# Weight tensors larger than this are hashed from a strided sample instead of
# every byte, bounding key cost for large models.
_DIGEST_FULL_BYTES = 1 << 20
_DIGEST_SAMPLES = 4096


def _weight_tensors(gm: GraphModule) -> list[tuple[str, torch.Tensor]]:
    """Every parameter, buffer and constant attribute reachable from `gm`."""
    out: list[tuple[str, torch.Tensor]] = []
    for name, t in gm.named_parameters():
        out.append((name, t))
    for name, t in gm.named_buffers():
        out.append((name, t))
    # get_attr targets that are plain tensor attributes rather than registered
    # parameters/buffers -- freezing lifts constants this way.
    for node in gm.graph.find_nodes(op="get_attr"):
        target = str(node.target)
        obj = gm
        try:
            for part in target.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, torch.Tensor):
            out.append((target, obj))
    return out


def _tensor_digest(t: torch.Tensor) -> bytes:
    """Content hash of a tensor, sampled if it is large."""
    from torch._subclasses.fake_tensor import FakeTensor, unset_fake_temporarily

    if isinstance(t, FakeTensor) or t.numel() == 0:
        # A fake tensor has no data to hash; shape and dtype are already keyed.
        return b"\x00"
    # This runs inside the FakeTensorMode a Dynamo backend is invoked under, so
    # even detach() on a real tensor asserts unless the mode is suspended.
    with unset_fake_temporarily():
        t = t.detach().reshape(-1)
        if t.dtype.is_floating_point or t.dtype.is_complex:
            t = t.float()
        if t.numel() * t.element_size() > _DIGEST_FULL_BYTES:
            step = max(1, t.numel() // _DIGEST_SAMPLES)
            t = t[::step]
        return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).digest()


def export_onnx(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
    path: Path,
    act_scales: dict[str, float] | None = None,
) -> tuple[list[str], list[str]]:
    """Export a partition to ONNX. Returns (input_names, output_names)."""
    from .decompose import decompose_for_onnx

    decompose_for_onnx(gm)

    n_out = len(gm.graph.find_nodes(op="output")[0].args[0])
    input_names = [f"bpu_in_{i}" for i in range(len(example_inputs))]
    output_names = [f"bpu_out_{i}" for i in range(n_out)]

    gm.eval()
    # A Dynamo backend is invoked inside an active FakeTensorMode. The ONNX
    # exporter traces the module with real tensors, so that mode has to be
    # suspended or every op dispatches to fakes and fails on shape guards.
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    with torch.no_grad(), unset_fake_temporarily():
        torch.onnx.export(
            gm,
            tuple(example_inputs),
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamo=False,
            opset_version=17,
        )

    # hbdk4's ONNX adaptor resolves every node output name through value_info,
    # so intermediates need inferred shapes or it raises "key ... not found".
    try:
        import onnx
        from onnx import shape_inference

        proto = shape_inference.infer_shapes(
            onnx.load(str(path)), strict_mode=False, data_prop=True
        )

        # Without int8 inputs hbdk4 lowers every conv to native::Conv2dNHWC on
        # the CPU, so the BPU sits idle. Q/DQ insertion is what moves the MAC
        # work onto the device — measured 6.1 ms -> 0.64 ms on a 2-conv net.
        if QUANTIZE:
            from .qdq import quantize_onnx

            proto = quantize_onnx(
                proto, act_scales=act_scales, default_act_scale=ACT_SCALE
            )

        onnx.save(proto, str(path))
    except Exception as e:  # noqa: BLE001
        log.warning("onnx post-processing skipped: %s", e)

    return input_names, output_names


def compile_hbm(onnx_path: Path, out_path: Path, march: str = DEFAULT_MARCH) -> Path:
    """Compile an ONNX file to a .hbm via hbdk4.

    Raises CompileError if the toolchain is missing or the compile fails; the
    caller is expected to fall back to CPU execution for that partition.
    """
    driver = find_hbdk()
    if driver is None:
        raise CompileError(
            "hbdk4 compiler not found. It ships x86_64-only wheels, so on this "
            "aarch64 board point FLAGOS_BPU_X86_PYTHON at an x86_64 python "
            "that has hbdk4-compiler installed, and use a recent box64 (0.4.x; "
            "the distro 0.2.6 aborts on this board's "
            f"{os.sysconf('SC_PAGESIZE')}-byte pages) -- see docs/vendors/bpu/integration.md for "
            "the one-time setup."
        )

    if driver == "python-api":
        return _compile_via_python_api(onnx_path, out_path, march)

    if driver == "x86-emul":
        return _compile_via_x86_emulator(onnx_path, out_path, march)

    cmd = [
        driver,
        "--model",
        str(onnx_path),
        "--march",
        march,
        "-o",
        str(out_path),
    ]
    log.info("hbdk4: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CompileError(
            f"hbdk4 failed (exit {proc.returncode}):\n"
            f"{proc.stderr[-2000:] or proc.stdout[-2000:]}"
        )
    if not out_path.exists():
        raise CompileError(f"hbdk4 reported success but {out_path} is missing")
    return out_path


def _compile_via_python_api(
    onnx_path: Path, out_path: Path, march: str, opt: int = 2
) -> Path:
    """Compile through the hbdk4 Python API.

    Signatures per the OE 3.7.0 API reference:
        hbdk4.compiler.onnx.export(proto: onnx.ModelProto) -> Module
        hbdk4.compiler.convert(m, march) -> Module
        hbdk4.compiler.compile(m, path, march, opt=2, ...) -> Hbm

    Note `export` takes a loaded protobuf, not a path, and `compile` writes the
    artifact itself — the output path is its second positional argument.
    """
    try:
        import onnx
        from hbdk4.compiler import compile as hbdk_compile
        from hbdk4.compiler import convert
        from hbdk4.compiler.onnx import export as onnx_export
    except ImportError as e:
        raise CompileError(f"hbdk4 Python API unavailable: {e}") from e

    try:
        proto = onnx.load(str(onnx_path))
        module = onnx_export(proto)
        # Lower hbir to this march's backend IR. Any Q/DQ already present in the
        # graph (inserted by qdq.py) is what lets conv land on the BPU; without
        # it convert() reports "fin is f32, which should be si8, si16 on bpu"
        # and falls back to native::Conv2dNHWC on the CPU.
        quantized = convert(module, march)
        hbdk_compile(quantized, str(out_path), march, opt=opt)
    except Exception as e:
        raise CompileError(f"hbdk4 compile failed: {type(e).__name__}: {e}") from e

    if not out_path.exists():
        raise CompileError(f"hbdk4 returned but {out_path} is missing")
    return out_path


# Driver run by the emulated x86 interpreter. It is the same three hbdk4 calls
# as _compile_via_python_api; only the interpreter differs, so keeping it as a
# string avoids shipping a second copy of the logic as a separate file.
#
# The libhbtl.so preload is not optional -- see _mlir_libs_dir() for why.
_X86_DRIVER = """
import ctypes, os, sys

libs = os.environ.get("FLAGOS_BPU_MLIR_LIBS")
if libs:
    ctypes.CDLL(os.path.join(libs, "libhbtl.so"), mode=ctypes.RTLD_GLOBAL)

import onnx
from hbdk4.compiler import compile as hbdk_compile, convert
from hbdk4.compiler.onnx import export as onnx_export

onnx_path, out_path, march, opt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
proto = onnx.load(onnx_path)
module = onnx_export(proto)

try:
    hbdk_compile(convert(module, march), out_path, march, opt=opt)
except Exception as e:
    # hbdk4's compile() finishes by loading the artifact back through hbrt4 to
    # validate it (apis.link -> Hbm(path)), which claims BPU device memory. Under
    # emulation that step can fail with ResourceExhausted/AllocError *after*
    # link() has already written a complete .hbm. Tolerate exactly that case: the
    # file is the deliverable, and the caller loads it with the board's native
    # aarch64 runtime anyway, which is a stricter check than this one.
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        raise
    print("hbdk4 post-compile validation skipped: %s: %s" % (type(e).__name__, e),
          file=sys.stderr)
"""


def _compile_via_x86_emulator(
    onnx_path: Path, out_path: Path, march: str, opt: int = 2
) -> Path:
    """Compile by running an x86_64 hbdk4 under box64 or qemu-user.

    Everything stays on this machine: the emulator only translates
    instructions, so the ONNX input and .hbm output are ordinary local files.
    Emulated compilation is slow (no JIT cache across runs), which is why
    compile_partition() caches the artifact by graph structure.
    """
    emul = x86_emulator()
    if emul is None:
        raise CompileError("no x86_64 emulator able to run hbdk4")

    cmd = [
        *emul,
        "-c",
        _X86_DRIVER,
        str(onnx_path),
        str(out_path),
        march,
        str(opt),
    ]
    env = x86_env()
    libs = _mlir_libs_dir()
    if libs:
        # Read by _X86_DRIVER to preload libhbtl.so inside the guest.
        env["FLAGOS_BPU_MLIR_LIBS"] = libs
    log.info("hbdk4 via %s: %s", Path(emul[0]).name, onnx_path.name)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600, env=env
        )
    except subprocess.TimeoutExpired as e:
        raise CompileError("hbdk4 timed out under emulation (1h)") from e

    if proc.returncode != 0:
        raise CompileError(
            f"hbdk4 failed under emulation (exit {proc.returncode}):\n"
            f"{(proc.stderr or proc.stdout)[-2000:]}"
        )
    if not out_path.exists():
        raise CompileError(f"hbdk4 returned but {out_path} is missing")
    log.info("compiled: %s (%d bytes)", out_path.name, out_path.stat().st_size)
    return out_path


def compile_partition(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
    march: str = DEFAULT_MARCH,
    cache_dir: Path | None = None,
    act_scales: dict[str, float] | None = None,
) -> tuple[Path, list[str], list[str]]:
    """Export, compile and cache one partition.

    Returns (hbm_path, input_names, output_names).
    """
    cache = cache_dir or CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    key = graph_key(gm, example_inputs, march)
    # Float and int8 artifacts differ, and so do two int8 builds with different
    # activation scales or different versions of the Q/DQ pass, so none of them
    # may share a cache entry.
    if QUANTIZE:
        from .qdq import QDQ_VERSION

        qh = hashlib.sha256(repr(sorted((act_scales or {}).items())).encode())
        qh.update(f"{ACT_SCALE}:v{QDQ_VERSION}".encode())
        key = f"{key}q{qh.hexdigest()[:8]}"
    hbm_path = cache / f"{key}.hbm"
    names_path = cache / f"{key}.names"

    if hbm_path.exists() and names_path.exists():
        ins, outs = names_path.read_text().strip().split("\n")
        log.info("cache hit: %s", hbm_path.name)
        return hbm_path, ins.split(","), outs.split(",")

    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / f"{key}.onnx"
        in_names, out_names = export_onnx(
            gm, example_inputs, onnx_path, act_scales=act_scales
        )
        compile_hbm(onnx_path, hbm_path, march)

    names_path.write_text(f"{','.join(in_names)}\n{','.join(out_names)}\n")
    log.info("compiled: %s", hbm_path.name)
    return hbm_path, in_names, out_names
