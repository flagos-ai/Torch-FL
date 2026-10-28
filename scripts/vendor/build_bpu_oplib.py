"""Build a per-operator .hbm library for the BPU eager path.

Each kernel is one hbir op at one fixed shape, compiled by hbdk4 on the build
host and shipped as an artifact the board loads natively. This runs on x86
where hbdk4 lives; the artifacts it emits are consumed by the board with no
further compilation.

Shapes are bucketed rather than dynamic: HBRT 4.7.5 declares a dynamic axis
with stride 0 while its task validator requires >= 64 and rejects any buffer
that is not exactly the stride extent, so no caller can satisfy both (see
docs/vendors/bpu/dynamic-shape-probe.md). A bucket table plus a CPU fallback
covers the same ground without depending on that path.

Usage:
    python build_oplib.py --out ./oplib --dtype f32 \
        --shapes 1x64 1x128 1x256 1x512
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

UNARY = ("relu", "neg", "abs", "exp", "tanh", "sigmoid", "sqrt", "rsqrt", "log")
BINARY = ("add", "sub", "mul", "max", "min", "div")

DTYPE_MLIR = {"f32": "f32", "f16": "f16"}
DTYPE_NUMPY = {"f32": "float32", "f16": "float16"}


def _mlir_type(dtype: str, shape: tuple[int, ...]) -> str:
    dims = "x".join(str(d) for d in shape)
    return f"tensor<{dims}x{DTYPE_MLIR[dtype]}>"


def _source(op: str, shape: tuple[int, ...], dtype: str, nargs: int) -> str:
    t = _mlir_type(dtype, shape)
    name = f"{op}_{'x'.join(str(d) for d in shape)}_{dtype}"
    if nargs == 1:
        args = f"%a: {t}"
        call = f'"hbir.{op}"(%a) : ({t}) -> {t}'
    else:
        args = f"%a: {t}, %b: {t}"
        call = f'"hbir.{op}"(%a, %b) : ({t}, {t}) -> {t}'
    return f"""
module {{
  func.func @{name}({args}) -> {t} {{
    %0 = {call}
    return %0 : {t}
  }}
}}
"""


def _template_key(op: str, rank: int, dtype: str) -> str:
    return f"{op}/{rank}d/{dtype}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output directory for artifacts")
    ap.add_argument("--mlir-libs", default=os.environ.get("FLAGOS_BPU_MLIR_LIBS"))
    ap.add_argument("--dtype", default="f32", choices=sorted(DTYPE_MLIR))
    ap.add_argument(
        "--shapes",
        nargs="+",
        default=["1x64", "1x128", "1x256", "1x512"],
        help="buckets, each written as 1xN",
    )
    ap.add_argument("--march", default="nash_p")
    ap.add_argument("--opt", type=int, default=2)
    args = ap.parse_args(argv)

    if not args.mlir_libs:
        ap.error("--mlir-libs (or FLAGOS_BPU_MLIR_LIBS) is required")
    # _hbdk.so resolves hbtl symbols only if libhbtl is already global.
    ctypes.CDLL(os.path.join(args.mlir_libs, "libhbtl.so"), mode=ctypes.RTLD_GLOBAL)

    from hbdk4.compiler import compile as hbdk_compile
    from hbdk4.compiler import ir as mlir
    from hbdk4.compiler import march as march_mod
    from hbdk4.compiler.apis import convert
    from hbdk4.compiler.overlay import Module

    march = march_mod.March.get(args.march)
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
    for op, nargs in plan:
        for shape in buckets:
            name = f"{op}_{'x'.join(str(d) for d in shape)}_{args.dtype}"
            out_path = out_dir / f"{name}.hbm"
            try:
                mod = Module(mlir.Module.parse(_source(op, shape, args.dtype, nargs)))
                converted = convert(mod, march)
                hbdk_compile(converted, str(out_path), march, opt=args.opt)
            except Exception as exc:  # noqa: BLE001 - report and keep going
                # compile() ends by loading its own artifact back through hbrt4,
                # which claims BPU memory; that read-back fails off-board even
                # though the file is complete. Any complete file is a success.
                if not (out_path.exists() and out_path.stat().st_size > 0):
                    manifest["failed"].append(
                        {"name": name, "error": f"{type(exc).__name__}: {exc}"[:300]}
                    )
                    print(f"FAILED {name}: {type(exc).__name__}", file=sys.stderr)
                    continue

            digest = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
            manifest["kernels"].append(
                {
                    "name": name,
                    "op": op,
                    "nargs": nargs,
                    "shape": list(shape),
                    "file": out_path.name,
                    "bytes": out_path.stat().st_size,
                    "sha256": digest,
                    "template": _template_key(op, len(shape), args.dtype),
                }
            )
            print(f"ok     {name} ({out_path.stat().st_size} B)")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"\n{len(manifest['kernels'])} kernels, {len(manifest['failed'])} failed "
        f"-> {out_dir / 'manifest.json'}"
    )
    return 1 if manifest["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
