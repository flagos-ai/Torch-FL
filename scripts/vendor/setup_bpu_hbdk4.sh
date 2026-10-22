#!/usr/bin/env bash
# Set up on-board hbdk4 compilation for the RDK BPU.
#
# hbdk4 (the BPU graph compiler) ships x86_64-only wheels, so it cannot run
# natively on this aarch64 board. This script builds the pieces that let it run
# here anyway, entirely on the board -- no VM, no x86 host, and no kernel
# rebuild:
#
#   1. box64 from source. The distro package (0.2.6) aborts with "PageSize
#      configuration is wrong: configured with 4096, but got 65536" because its
#      page size was a build-time constant. Current box64 reads the host page
#      size at runtime and maps 4 KB-aligned x86 segments onto 64 KB pages
#      itself, so the stock kernel is fine.
#   2. A standalone x86_64 CPython 3.11 (python-build-standalone).
#   3. hbdk4-compiler + hbdk4-march wheels into that interpreter.
#   4. Import-only stubs for numba and torch -- see stubs/ for why.
#
# Usage:
#   scripts/vendor/setup_bpu_hbdk4.sh [--wheels DIR] [--prefix DIR]
#
# --wheels defaults to ~/x86vm/wheels; it must contain
# hbdk4_compiler-*-cp311-*_x86_64.whl and hbdk4_march-*_x86_64.whl from the
# D-Robotics OE package (oe-package-*.tgz on ftp://oeftp@sdk.d-robotics.cc/).
#
# Afterwards, export the two variables the script prints and BPU offload turns
# on automatically:
#   export FLAGOS_BPU_X86_PYTHON=<prefix>/python/bin/python3.11
#   export FLAGOS_BPU_X86_EMULATOR=<prefix>/bin/box64
set -euo pipefail

PREFIX="${HOME}/hbdk4-x86"
WHEELS="${HOME}/x86vm/wheels"
BOX64_SRC="${HOME}/box64-src"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheels) WHEELS="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n=== %s\n' "$*"; }

[[ "$(uname -m)" == "aarch64" ]] || { echo "this script is for the aarch64 board" >&2; exit 1; }

say "page size: $(getconf PAGE_SIZE) (any value is fine; box64 handles 64K)"

# ---------------------------------------------------------------- box64
if [[ -x "${PREFIX}/bin/box64" ]]; then
  say "box64 already built: $("${PREFIX}/bin/box64" --version 2>&1 | head -1)"
else
  say "building box64 from source (needs cmake, gcc; takes a few minutes)"
  [[ -d "${BOX64_SRC}" ]] || git clone --depth 1 https://github.com/ptitSeb/box64.git "${BOX64_SRC}"
  mkdir -p "${BOX64_SRC}/build"
  (
    cd "${BOX64_SRC}/build"
    cmake .. -DARM_DYNAREC=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
    make -j"$(nproc)"
  )
  mkdir -p "${PREFIX}/bin"
  cp "${BOX64_SRC}/build/box64" "${PREFIX}/bin/box64"
  say "box64: $("${PREFIX}/bin/box64" --version 2>&1 | head -1)"
fi

BOX64="${PREFIX}/bin/box64"

# ------------------------------------------------------- x86_64 CPython
PY="${PREFIX}/python/bin/python3.11"
if [[ -x "${PY}" ]]; then
  say "x86_64 python already present"
else
  say "fetching standalone x86_64 CPython 3.11"
  mkdir -p "${PREFIX}"
  python3 - "$PREFIX" <<'PYEOF'
import json, sys, urllib.request
prefix = sys.argv[1]
api = "https://api.github.com/repos/astral-sh/python-build-standalone/releases?per_page=10"
for rel in json.load(urllib.request.urlopen(api, timeout=120)):
    hit = [a for a in rel["assets"]
           if "cpython-3.11" in a["name"]
           and "x86_64-unknown-linux-gnu-install_only." in a["name"]]
    if hit:
        print("downloading", hit[0]["name"])
        urllib.request.urlretrieve(hit[0]["browser_download_url"], f"{prefix}/py311.tar.gz")
        break
else:
    raise SystemExit("no cpython-3.11 x86_64 install_only asset found")
PYEOF
  tar xf "${PREFIX}/py311.tar.gz" -C "${PREFIX}"
  rm -f "${PREFIX}/py311.tar.gz"
fi

# Confirm the emulator actually runs it. This is the step the old box64 fails.
say "checking box64 can run the x86_64 interpreter"
"${BOX64}" "${PY}" -c 'import platform; print("guest machine:", platform.machine())'

# --------------------------------------------------------------- hbdk4
say "installing hbdk4 into the x86_64 interpreter"
shopt -s nullglob
COMPILER_WHL=("${WHEELS}"/hbdk4_compiler-*cp311*x86_64.whl)
MARCH_WHL=("${WHEELS}"/hbdk4_march-*x86_64.whl)
shopt -u nullglob
if [[ ${#COMPILER_WHL[@]} -eq 0 ]]; then
  echo "no hbdk4_compiler cp311 x86_64 wheel in ${WHEELS}" >&2
  echo "get it from the D-Robotics OE package; see docs/vendors/bpu/integration.md" >&2
  exit 1
fi

"${BOX64}" "${PY}" -m pip install --no-cache-dir "${COMPILER_WHL[0]}" "${MARCH_WHL[@]}"
# onnx is the input format; sympy is imported by hbdk4's opset13 module.
# numba is deliberately NOT installed: it imports llvmlite.binding, whose x86
# LLVM JIT segfaults under box64. The stubs below stand in for it.
"${BOX64}" "${PY}" -m pip install --no-cache-dir onnx sympy

# ---------------------------------------------------------------- stubs
say "installing import-only stubs for numba and torch"
STUBS="${PREFIX}/stubs"
mkdir -p "${STUBS}/numba/core" "${STUBS}/torch"

cat > "${STUBS}/numba/__init__.py" <<'EOF'
"""Import-safe stand-in for numba, used only by hbdk4 under box64.

hbdk4's ONNX entry point imports numba unconditionally
(hbdk4/compiler/onnx/__init__.py -> hbdk4.compiler.numba.tools), but only
*calls* it when the graph contains a numba op: compile_numba() returns the
module untouched when has_numba_op() is false, which is always the case for a
graph exported from ONNX standard operators.

Real numba cannot be used here. It imports llvmlite.binding, which loads
libllvmlite.so and initializes an x86 LLVM JIT; that segfaults under box64. The
crash is in llvmlite's JIT setup, not in numba or hbdk4.

Anything that would actually run raises, so a graph that genuinely needs numba
fails loudly instead of silently miscompiling.
"""

__version__ = "0.0.0+hbdk4-stub"


def _unavailable(name):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"numba.{name} is unavailable: llvmlite's x86 JIT segfaults under "
            "box64, so numba is stubbed out. A graph containing a custom/numba "
            "op must be compiled on an x86_64 host instead."
        )

    return _raise


njit = _unavailable("njit")
typeof = _unavailable("typeof")
EOF

cat > "${STUBS}/numba/core/__init__.py" <<'EOF'
"""Stub for numba.core. See ../__init__.py."""

from . import extending, types  # noqa: F401
EOF

cat > "${STUBS}/numba/core/extending.py" <<'EOF'
"""Stub for numba.core.extending.

hbdk4 uses one name from here, is_jitted, and only to assert that a custom op's
entry function carries @numba.njit. Nothing ever does under this stub, so
returning False makes hbdk4's own assertion fire with its own message rather
than an AttributeError.
"""


def is_jitted(_func) -> bool:
    return False


def register_jitable(*_args, **_kwargs):
    def _identity(fn):
        return fn

    return _identity
EOF

cat > "${STUBS}/numba/core/types.py" <<'EOF'
"""Stub for numba.core.types: any attribute resolves to a placeholder."""


class _Placeholder:
    def __init__(self, name: str):
        self._name = name

    def __call__(self, *_args, **_kwargs):
        return self

    def __getitem__(self, _key):
        return self

    def __repr__(self) -> str:
        return f"<numba-stub type {self._name}>"


def __getattr__(name: str) -> _Placeholder:
    return _Placeholder(name)
EOF

cat > "${STUBS}/torch/__init__.py" <<'EOF'
"""Import-safe stand-in for torch, used only by hbdk4 under box64.

hbdk4's ONNX path reaches hbdk4/compiler/numba/trace.py, which imports torch at
module scope. Everything it uses torch for lives in the custom-op tracing path
(isinstance checks and torch.jit.trace), which an ONNX-exported graph never
enters. Installing a real x86_64 torch would add several hundred MB to the
emulated environment for code that never runs.

This is the *emulated* interpreter's torch, entirely separate from the board's
own aarch64 torch that torch_fl runs on.

Names used in isinstance() or typing.Union[...] must be real classes, which is
why the fallbacks below produce types rather than raising placeholders --
hbdk4's modules evaluate their annotations eagerly at import time.
"""

__version__ = "0.0.0+hbdk4-stub"


class Tensor:
    """For isinstance() only; no object in the ONNX path is an instance."""


class Graph:
    """Named in hbdk4's torch-jit adaptor annotations."""


def _unavailable(name):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"torch.{name} is unavailable: this is a stub torch that exists "
            "only so hbdk4's tracing module can be imported under box64. A "
            "graph needing it must be compiled on an x86_64 host."
        )

    return _raise


zeros = _unavailable("zeros")
from_numpy = _unavailable("from_numpy")


class _AttrIsAType:
    """Base whose unknown attributes resolve to fresh classes."""

    def __getattr__(self, name: str):
        cls = type(name, (), {})
        setattr(self, name, cls)
        return cls


class _Nn(_AttrIsAType):
    class Module:
        pass

    class functional:  # noqa: N801 - mirrors torch.nn.functional
        pass


class _Jit(_AttrIsAType):
    trace = staticmethod(_unavailable("jit.trace"))


nn = _Nn()
jit = _Jit()


def __getattr__(name: str):
    return type(name, (), {})
EOF

# ---------------------------------------------------------------- verify
say "verifying hbdk4 imports on the board"
LIBS="$(echo "${PREFIX}"/python/lib/python3.*/site-packages/hbdk4/compiler/_mlir_libs)"
PYTHONPATH="${STUBS}" BOX64_LD_LIBRARY_PATH="${LIBS}" "${BOX64}" "${PY}" -c "
import ctypes, os
ctypes.CDLL(os.path.join('${LIBS}', 'libhbtl.so'), mode=ctypes.RTLD_GLOBAL)
from hbdk4.compiler import compile, convert
from hbdk4.compiler.onnx import export
import onnx
print('hbdk4 OK (onnx', onnx.__version__ + ')')
"

cat <<EOF

=== Done. Add this to your shell profile:

  export FLAGOS_BPU_X86_PYTHON=${PY}
  export FLAGOS_BPU_X86_EMULATOR=${BOX64}

Then torch.compile(model, backend="bpu") offloads to the BPU automatically.
EOF
