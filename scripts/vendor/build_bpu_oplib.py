"""Build a per-operator .hbm library for the BPU eager path.

Each kernel is one hbir op at one fixed shape, compiled by hbdk4 on a build
host and shipped as an artifact the board loads natively. This runs where
hbdk4 lives (x86_64); the artifacts it emits are consumed by the board with no
further compilation. A native x86_64 host builds a kernel in about 0.8 s; the
same build under box64 on the board takes about 10 s.

Graphs are built through hbdk4's Python op API rather than hand-written MLIR.
The text form needs every attribute spelled exactly, and a wrong one surfaces
as a bare "custom op has no custom assembly form"; the Python API checks the
call signature instead and turned three ops from FAIL into OK during the
survey that produced the tables below.

Shapes are bucketed rather than dynamic: HBRT 4.7.5 declares a dynamic axis
with stride 0 while its task validator requires >= 64 and rejects any buffer
that is not exactly the stride extent, so no caller can satisfy both (see
docs/vendors/bpu/dynamic-shape-probe.md). The runtime routes a request up to
the next bucket, so the ladder only has to bound the padding.

Usage:
    python build_bpu_oplib.py --out ./oplib
    python build_bpu_oplib.py --out ./oplib --dtype f16 --shapes 1x64 1x256
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

# Widths, dense where small and sparse where large. Requests route to the
# smallest bucket at or above their width, so what matters is the worst-case
# pad ratio rather than exactness: this ladder keeps it under 8x at 11 rungs.
DEFAULT_BUCKETS = (8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512)

DTYPE_MLIR = {"f32": "f32", "f16": "f16"}
DTYPE_NUMPY = {"f32": "float32", "f16": "float16"}

# op name -> builder taking (hbir module, input). Every entry was verified to
# compile standalone on nash-p.
UNARY = {
    "relu": lambda h, a: h.relu(a),
    "neg": lambda h, a: h.neg(a),
    "abs": lambda h, a: h.abs(a),
    "exp": lambda h, a: h.exp(a),
    "tanh": lambda h, a: h.tanh(a),
    "sigmoid": lambda h, a: h.sigmoid(a),
    "sqrt": lambda h, a: h.sqrt(a),
    "rsqrt": lambda h, a: h.rsqrt(a),
    "log": lambda h, a: h.log(a),
    "ceil": lambda h, a: h.ceil(a),
    "floor": lambda h, a: h.floor(a),
    "round": lambda h, a: h.round(a),
    "sign": lambda h, a: h.sign(a),
    "sin": lambda h, a: h.sin(a),
    "cos": lambda h, a: h.cos(a),
    "tan": lambda h, a: h.tan(a),
    "sinh": lambda h, a: h.sinh(a),
    "cosh": lambda h, a: h.cosh(a),
    "asin": lambda h, a: h.asin(a),
    "acos": lambda h, a: h.acos(a),
    "atan": lambda h, a: h.atan(a),
    "asinh": lambda h, a: h.asinh(a),
    "acosh": lambda h, a: h.acosh(a),
    "atanh": lambda h, a: h.atanh(a),
    "erf": lambda h, a: h.erf(a),
    "reciprocal": lambda h, a: h.reciprocal(a),
    "mish": lambda h, a: h.mish(a),
    "swish": lambda h, a: h.swish(a),
    "gelu": lambda h, a: h.gelu(a),
    "softplus": lambda h, a: h.softplus(a, 1.0, 20.0),
    "elu": lambda h, a: h.elu(a, 1.0),
    "leaky_relu": lambda h, a: h.leaky_relu(a, 0.01),
}

BINARY = {
    "add": lambda h, a, b: h.add(a, b),
    "sub": lambda h, a, b: h.sub(a, b),
    "mul": lambda h, a, b: h.mul(a, b),
    "max": lambda h, a, b: h.max(a, b),
    "min": lambda h, a, b: h.min(a, b),
    "div": lambda h, a, b: h.div(a, b),
    "pow": lambda h, a, b: h.pow(a, b),
    "mod": lambda h, a, b: h.mod(a, b),
}


def _mlir_type(dtype: str, shape: tuple[int, ...]) -> str:
    dims = "x".join(str(d) for d in shape)
    return f"tensor<{dims}x{DTYPE_MLIR[dtype]}>"


def _template_key(op: str, rank: int, dtype: str) -> str:
    return f"{op}/{rank}d/{dtype}"


def _build_module(hbir, mlir, func, tensortype, elemtype, op, shape, nargs):
    """One hbir graph computing `op` over `shape`."""
    tt = tensortype.get(list(shape), elemtype)
    name = f"{op}_{'x'.join(str(d) for d in shape)}"
    module = mlir.Module.parse("module {}")
    with module.context, mlir.InsertionPoint(module.body):
        fn = func.FuncOp(name, ([tt] * nargs, [tt]))
        with mlir.InsertionPoint(fn.add_entry_block()):
            args = list(fn.arguments)
            result = UNARY[op](hbir, args[0]) if nargs == 1 else BINARY[op](hbir, *args)
            func.ReturnOp([result])
    return module


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output directory for artifacts")
    ap.add_argument("--mlir-libs", default=os.environ.get("FLAGOS_BPU_MLIR_LIBS"))
    ap.add_argument("--dtype", default="f32", choices=sorted(DTYPE_MLIR))
    ap.add_argument("--ops", nargs="+", default=None, help="subset of op names")
    ap.add_argument(
        "--shapes",
        nargs="+",
        default=[f"1x{w}" for w in DEFAULT_BUCKETS],
        help="buckets, each written as 1xN (default: the ladder above)",
    )
    ap.add_argument("--march", default="nash_p")
    ap.add_argument("--opt", type=int, default=2)
    args = ap.parse_args(argv)

    if args.mlir_libs:
        # _hbdk.so resolves hbtl symbols only if libhbtl is already global.
        try:
            ctypes.CDLL(
                os.path.join(args.mlir_libs, "libhbtl.so"), mode=ctypes.RTLD_GLOBAL
            )
        except OSError:
            pass

    from hbdk4.compiler import compile as hbdk_compile
    from hbdk4.compiler import ir as mlir
    from hbdk4.compiler import march as march_mod
    from hbdk4.compiler.apis import convert
    from hbdk4.compiler.dialects import func
    from hbdk4.compiler.ir import F16Type, F32Type, RankedTensorType
    from hbdk4.compiler.ops import hbir
    from hbdk4.compiler.overlay import Module

    march = march_mod.March.get(args.march)
    elemtype = F32Type.get() if args.dtype == "f32" else F16Type.get()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = [tuple(int(d) for d in s.lower().split("x")) for s in args.shapes]
    manifest: dict = {
        "march": args.march,
        "dtype": args.dtype,
        "numpy_dtype": DTYPE_NUMPY[args.dtype],
        "buckets": ["x".join(str(d) for d in b) for b in buckets],
        "kernels": [],
        "failed": [],
    }

    plan = [(op, 1) for op in UNARY] + [(op, 2) for op in BINARY]
    if args.ops:
        wanted = set(args.ops)
        plan = [(op, n) for op, n in plan if op in wanted]

    for op, nargs in plan:
        for shape in buckets:
            stem = f"{op}_{'x'.join(str(d) for d in shape)}_{args.dtype}"
            out_path = out_dir / f"{stem}.hbm"
            try:
                module = _build_module(
                    hbir, mlir, func, RankedTensorType, elemtype, op, shape, nargs
                )
                converted = convert(Module(module), march)
                hbdk_compile(converted, str(out_path), march, opt=args.opt)
            except Exception as exc:  # noqa: BLE001 - report and keep going
                # compile() ends by loading its own artifact back through hbrt4,
                # which claims BPU memory; that read-back can fail even though
                # the file is complete. Any complete file is a success.
                if not (out_path.exists() and out_path.stat().st_size > 0):
                    manifest["failed"].append(
                        {"name": stem, "error": f"{type(exc).__name__}: {exc}"[:300]}
                    )
                    print(f"FAILED {stem}: {type(exc).__name__}", file=sys.stderr)
                    continue

            digest = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
            manifest["kernels"].append(
                {
                    "name": stem,
                    "op": op,
                    "nargs": nargs,
                    "shape": list(shape),
                    "file": out_path.name,
                    "bytes": out_path.stat().st_size,
                    "sha256": digest,
                    "template": _template_key(op, len(shape), args.dtype),
                }
            )
            print(f"ok     {stem} ({out_path.stat().st_size} B)", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"\n{len(manifest['kernels'])} kernels, {len(manifest['failed'])} failed "
        f"-> {out_dir / 'manifest.json'}"
    )
    return 1 if manifest["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
