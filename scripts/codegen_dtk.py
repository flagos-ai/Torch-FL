#!/usr/bin/env python3
# Copyright (c) 2026, BAAI. All rights reserved.
"""Code generator for the Hygon DCU SDK-native operator backend.

Why this exists
---------------
DCU currently gets its operator coverage from DTK's *forked* PyTorch: every
route spelled ``= cuda`` in ``backends_dcu_flaggems.conf`` boxes PrivateUse1 to
the CUDA dispatch key and lands in DTK's ``libtorch_hip.so``. That pins the
install to a vendor torch build. The goal is to serve those operators from the
DTK **SDK** instead -- rocBLAS/hipBLASLt, MIOpen/hipDNN, rocSOLVER, rocFFT,
rocRAND, rocSPARSE, RCCL, plus ``hipcc``-compiled kernels for the long tail --
none of which contain a single torch symbol. Same trick ``codegen_mudnn.py``
plays for Moore Threads: kernels link the vendor SDK only, so they are
independent of the torch version.

Handwriting that is not viable: 1554 operators reach the vendor fallback today.
This generator is the scalable path -- adding an operator is a row in ``OPS``,
not a new C++ function.

Why 1554 operators do not need 1554 kernels
-------------------------------------------
PyTorch's own composite layers absorb most of them. Classifying the 1554 by
their ``native_functions.yaml`` metadata (see ``--classify``)::

     876  CompositeExplicitAutograd  -- generic impl, reusable as-is
     133  structured_delegate        -- rides its .out sibling
       2  CompositeImplicitAutograd  -- decomposes into primitives
       4  view op                    -- metadata only, no compute
     539  needs a real kernel

So ~1015 come free once the primitives underneath them exist, and the real
target is 539 kernels grouped into a few dozen reusable categories.

What this emits
---------------
``--emit slots``   ``csrc/aten/generated/dtk_slot_table.cc``
    The core-side name -> typed-dispatcher table for **every** operator in
    ``generated/ops.h``. This is what makes ABI v2's untyped ``{op_name, fn}``
    records safe: the table owns the one cast from ``void*`` back to the real
    signature, guarded by the fn-type tag the plugin reports. Generated for the
    full operator set, independent of how many kernels any plugin implements.

``--emit kernels`` ``csrc/aten/backends/dcu_sdk/generated/dtk_kernels.cc``
                   ``csrc/aten/backends/dcu_sdk/generated/dtk_kernels_decl.h``
    The plugin-side kernels for the operators listed in ``OPS``, one per
    category template, plus the ``FlagosDcuSdkKernel`` array literal.

``--emit registry`` ``torch_fl/configs/dcu_route_registry.json``
    Full route metadata for every DCU dispatcher row, including FlagGems,
    SDK-native, composite, structured-delegate, view, and unsupported states.

``--emit manifest`` ``torch_fl/configs/dcu_sdk_manifest.json``
    Runtime validation metadata derived from the route registry.

``--emit conf``    the generated full SDK hybrid config on stdout.

``--classify``     the coverage arithmetic above, from live torchgen metadata.
``--coverage``     covered / remaining, per category and overall.

Idempotency: re-running with no ``OPS`` change must produce a byte-identical
tree. CI depends on that (``--check``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent

# Reuse the authoritative symbol naming from the CUDA codegen so the emitted
# table entries match the DECLARE_DISPATCHER names in generated/ops.h exactly.
# Deriving them independently here would drift and fail to link.
sys.path.insert(0, str(REPO / "scripts"))
from codegen_ops import schema_to_cpp_name  # noqa: E402

OUT_SLOTS = REPO / "csrc/aten/generated/dtk_slot_table.cc"
GEN_DIR = REPO / "csrc/aten/backends/dcu_sdk/generated"
OUT_KERNELS = GEN_DIR / "dtk_kernels.cc"
OUT_REGISTER = GEN_DIR / "dtk_kernels_decl.h"
OUT_MANIFEST = REPO / "torch_fl/configs/dcu_sdk_manifest.json"
OUT_ROUTE_REGISTRY = REPO / "torch_fl/configs/dcu_route_registry.json"
OPS_H = REPO / "csrc/aten/generated/ops.h"
DCU_CONF = REPO / "torch_fl/configs/backends_dcu_flaggems.conf"
DCU_SDK_CONF = REPO / "torch_fl/configs/backends_dcu_sdk.conf"
DCU_SDK_ONLY_CONF = REPO / "torch_fl/configs/backends_dcu_sdk_only.conf"
DCU_SDK_ABI_VERSION = 2
PLUGIN_NAME = "libdcu_aten_ops.so"
SUPPORTED_ROUTE_STATES = (
    "flaggems",
    "dcu_sdk",
    "composite",
    "structured_delegate",
    "view",
    "unsupported",
)
SUPPORTED_SDK_CATEGORIES = {
    "hip_unary_out",
    "hip_binary_out",
}
DEFAULT_DTYPES = ["float32", "float64", "int32", "int64", "bool"]
DEFAULT_LAYOUTS = ["contiguous", "transposed", "storage_offset"]
DEFAULT_STREAM_BEHAVIOR = "caller-current-stream-asynchronous"
DEFAULT_PROVENANCE = {
    "source": "scripts/codegen_dtk.py",
    "hardware": "not revalidated",
    "validation": "static route classification only",
    "evidence_gap": "SDK-native rows require per-category hardware validation before support is claimed.",
}

# The native functions file is the only source of truth for route metadata. Keep
# its import lazy: --emit slots/kernels can still be used in a source tree where
# torchgen is unavailable, while full-route artifacts require torchgen explicitly.
try:
    import torchgen
    from torchgen.gen import parse_native_yaml
except ImportError:  # pragma: no cover - exercised by source-only tooling
    torchgen = None
    parse_native_yaml = None


@lru_cache(maxsize=1)
def native_functions():
    if torchgen is None or parse_native_yaml is None:
        raise RuntimeError(
            "torchgen is required for DCU route/manifest generation; install "
            "the matching official PyTorch package"
        )
    base = Path(torchgen.__file__).parent / "packaged/ATen/native"
    return parse_native_yaml(
        str(base / "native_functions.yaml"), str(base / "tags.yaml")
    )


def torch_version_parts() -> Tuple[int, int, int]:
    """Read the exact version from torchgen's sibling torch/version.py."""
    if torchgen is None:
        return (0, 0, 0)
    version_file = Path(torchgen.__file__).parent.parent / "torch" / "version.py"
    version = "0.0.0"
    if version_file.exists():
        for line in version_file.read_text().splitlines():
            if line.startswith("__version__") and "=" in line:
                version = line.split("=", 1)[1].strip().strip("'\"")
                break
    version = version.split("+", 1)[0]
    parts = version.split(".")
    try:
        values = [int(part) for part in parts[:3]]
    except ValueError:
        values = [0, 0, 0]
    return tuple((values + [0, 0, 0])[:3])


def sdk_version() -> str:
    """Return the DTK version without importing the runtime."""
    candidates = []
    for root in (
        os.environ.get("FLAGOS_DCU_SDK_ROOT"),
        os.environ.get("DTK_ROOT"),
        os.environ.get("ROCM_PATH"),
        "/opt/dtk",
    ):
        if root:
            candidates.extend(
                [Path(root) / ".dtk_version", Path(root) / ".info" / "version-libs"]
            )
    for path in candidates:
        try:
            text = path.read_text().splitlines()[0].strip()
        except (OSError, IndexError):
            continue
        if text:
            return text
    return "unknown"


def _cpp_abi() -> str:
    return "cxx11" if os.environ.get("_GLIBCXX_USE_CXX11_ABI", "1") != "0" else "cxx03"


def _read_route_file(path: Path) -> Dict[str, str]:
    routes: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        text = line.split("#", 1)[0].strip()
        if not text or "=" not in text:
            continue
        op, backend = (part.strip() for part in text.split("=", 1))
        if op and backend:
            routes[op] = backend
    return routes


def _flaggems_routes() -> Dict[str, str]:
    """Return the current measured FlagGems route set from the DCU config."""
    return {
        op: "flaggems"
        for op, backend in _read_route_file(DCU_CONF).items()
        if backend in ("flagos", "flagos_python", "flaggems", "flaggems_python")
    }


def _classification() -> Dict[str, List[str]]:
    """Classify every non-FlagGems DCU route using torchgen metadata."""
    funcs = {str(f.func.name): f for f in native_functions().native_functions}
    routes = _read_route_file(DCU_CONF)
    buckets: Dict[str, List[str]] = {
        "composite": [],
        "structured_delegate": [],
        "view": [],
        "needs_kernel": [],
        "not_in_yaml": [],
    }
    for op, backend in routes.items():
        if backend in ("flagos", "flagos_python", "flaggems", "flaggems_python"):
            continue
        func = funcs.get(op)
        if func is None:
            buckets["not_in_yaml"].append(op)
        elif func.has_composite_implicit_autograd_kernel or (
            func.has_composite_explicit_autograd_kernel
            or func.has_composite_explicit_autograd_non_functional_kernel
        ):
            buckets["composite"].append(op)
        elif func.structured_delegate:
            buckets["structured_delegate"].append(op)
        elif func.is_view_op:
            buckets["view"].append(op)
        else:
            buckets["needs_kernel"].append(op)
    return {key: sorted(values) for key, values in buckets.items()}


def _route_registry() -> List[Dict[str, object]]:
    """Build the complete, auditable route registry for all 2034 DCU rows."""
    routes = _read_route_file(DCU_CONF)
    classification = _classification()
    class_by_op = {
        op: state for state, operators in classification.items() for op in operators
    }
    flaggems = _flaggems_routes()
    sdk_rows = set(OPS)
    registry = []
    for op in sorted(routes):
        original = routes[op]
        if op in flaggems:
            state = "flaggems"
            category = None
            fallback = "dcu_sdk" if op in sdk_rows else "unsupported"
        elif op in sdk_rows:
            state = "dcu_sdk"
            category = OPS[op][0]
            fallback = "unsupported"
        else:
            state = class_by_op.get(op, "unsupported")
            if state == "needs_kernel":
                # Keep the internal classification distinct from the public route
                # vocabulary: this is an uncovered real-kernel target, not a
                # backend that can be selected by a generated config.
                state = "unsupported"
            category = None
            fallback = "unsupported"
        registry.append(
            {
                "operator": op,
                "original_backend": original,
                "state": state,
                "category": category,
                "fallback": fallback,
                "sdk_implemented": op in sdk_rows,
                "provenance": dict(DEFAULT_PROVENANCE),
            }
        )
    return registry


def _registry_by_state(registry: List[Dict[str, object]], state: str) -> List[str]:
    return [str(row["operator"]) for row in registry if row["state"] == state]


def gen_route_registry() -> str:
    registry = _route_registry()
    states = Counter(str(row["state"]) for row in registry)
    payload = {
        "schema_version": 1,
        "generated_by": "scripts/codegen_dtk.py",
        "dispatch_key": "PrivateUse1",
        "route_count": len(registry),
        "states": dict(sorted(states.items())),
        "routes": registry,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def gen_manifest() -> str:
    registry = _route_registry()
    counts = Counter(str(row["state"]) for row in registry)
    sdk_ops = [str(row["operator"]) for row in registry if row["state"] == "dcu_sdk"]
    real_kernel_targets = len(_classification()["needs_kernel"])
    version = torch_version_parts()
    payload = {
        "schema_version": 2,
        "torch_base": ".".join(str(value) for value in version),
        "torch_version": {
            "major": version[0],
            "minor": version[1],
            "patch": version[2],
        },
        "torch_abi": _cpp_abi(),
        "sdk": "dtk",
        "sdk_version": sdk_version(),
        "registration_abi": DCU_SDK_ABI_VERSION,
        "dispatch_key": "PrivateUse1",
        "library": PLUGIN_NAME,
        "route_count": len(registry),
        "operators": sdk_ops,
        "operator_count": len(sdk_ops),
        "route_states": dict(sorted(counts.items())),
        "flaggems_count": counts["flaggems"],
        "sdk_category_count": counts["dcu_sdk"],
        "composite_count": counts["composite"],
        "structured_delegate_count": counts["structured_delegate"],
        "view_count": counts["view"],
        "unsupported_count": counts["unsupported"],
        "real_kernel_target_count": real_kernel_targets,
        "generated_kernel_count": len(OPS),
        "offered_kernel_count": len(OPS),
        "known_slot_count": len(registry),
        "dtypes": DEFAULT_DTYPES,
        "layouts": DEFAULT_LAYOUTS,
        "stream_behavior": DEFAULT_STREAM_BEHAVIOR,
        "fallback": "FlagGems-first, SDK-native category fallback, explicit unsupported",
        "sdk_only": {
            "enabled": True,
            "forbidden_routes": ["cuda"],
            "vendor_torch_libraries": False,
        },
        "hardware_validation": {
            "status": "not revalidated",
            "architecture": os.environ.get("FLAGOS_DCU_GPU_ARCH", "gfx936"),
        },
        "provenance": dict(DEFAULT_PROVENANCE),
        "evidence_gap": (
            "Route classification is static. Generated SDK rows and every other "
            "category require DCU hardware validation before being advertised as supported."
        ),
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def gen_route_conf(*, sdk_only: bool) -> str:
    """Generate full configs from the registry, with no implicit CUDA fallback."""
    registry = _route_registry()
    lines = [
        LICENSE.replace("//", "#"),
        "\n",
        "# @generated by scripts/codegen_dtk.py -- DO NOT EDIT.\n",
        "# Full DCU route registry: FlagGems first, then SDK-native categories.\n",
        "# Composite/delegate/view routes are explicit metadata states; routes with\n",
        "# no validated implementation are explicit unsupported diagnostics.\n",
        "# Format: op_name = backend\n",
        "\n",
    ]
    for row in registry:
        state = str(row["state"])
        if state == "flaggems":
            backend = "flagos_python"
        elif state == "dcu_sdk":
            backend = "dcu_sdk"
        elif sdk_only:
            # Composite/delegate/view routes are intentionally not claimed by the
            # SDK dispatcher. They execute through ATen's own composite machinery
            # when possible; unsupported rows are omitted and fail explicitly.
            continue
        elif state in ("composite", "structured_delegate", "view"):
            backend = "flagos"
        else:
            continue
        lines.append(f"{row['operator']} = {backend}\n")
    return "".join(lines)


def gen_conf_artifacts() -> Dict[Path, str]:
    return {
        DCU_SDK_CONF: gen_route_conf(sdk_only=False),
        DCU_SDK_ONLY_CONF: gen_route_conf(sdk_only=True),
    }


LICENSE = "// Copyright (c) 2026, BAAI. All rights reserved.\n"
BANNER = "// @generated by scripts/codegen_dtk.py -- DO NOT EDIT.\n"


# ---------------------------------------------------------------------------
# Operator registry: schema name -> (category, category-specific payload).
#
# Categories are implemented by the templates in KERNEL_TEMPLATES below. A row
# here is a claim that the operator's aten semantics are exactly expressible by
# that template -- operand order, alpha/beta placement, dtype promotion and
# empty-tensor behaviour included. Every row must be confirmed on device before
# it lands; an unverified row is worse than an absent one, because an absent
# operator falls through to a working path while a wrong one silently computes
# garbage.
#
# `hip_elementwise` payload is the HIP expression computing one output element
# from the inputs, written against `a` (and `b` for binary). hipcc compiles it;
# no vendor math library is involved, which is what lets the long tail scale.
# ---------------------------------------------------------------------------
# NOTE ON WHAT BELONGS HERE.
# Rows must come from the *required* set -- the 539 operators that genuinely need
# a kernel (``--classify``). It is easy to pick plausible-looking operators that
# are already served by FlagGems or by a composite path; those rows cost review
# effort and advance coverage by zero. ``--coverage`` flags them, and CI treats
# an off-target row as an error. Concretely: ``abs`` is NOT a valid row (FlagGems
# serves it) while ``abs.out`` IS -- the ``.out`` variants are the structured
# kernels the composite layers ultimately bottom out in, which is why 599 of them
# sit in the vendor-fallback set.
OPS: Dict[str, Tuple[str, object]] = {
    # ---- unary elementwise .out, hipcc-generated ----
    # All verified present in the required set. Expressions are written against
    # `a` with `T` the promoted compute type.
    "abs.out": ("hip_unary_out", "a < T(0) ? -a : a"),
    "acos.out": ("hip_unary_out", "::acos(a)"),
    "acosh.out": ("hip_unary_out", "::acosh(a)"),
    "asin.out": ("hip_unary_out", "::asin(a)"),
    "atan.out": ("hip_unary_out", "::atan(a)"),
    "atanh.out": ("hip_unary_out", "::atanh(a)"),
    "cos.out": ("hip_unary_out", "::cos(a)"),
    "erf.out": ("hip_unary_out", "::erf(a)"),
    "erfc.out": ("hip_unary_out", "::erfc(a)"),
    "exp2.out": ("hip_unary_out", "::exp2(a)"),
    "log.out": ("hip_unary_out", "::log(a)"),
    "log2.out": ("hip_unary_out", "::log2(a)"),
    "frac.out": ("hip_unary_out", "a - ::trunc(a)"),
    # ---- binary elementwise .out, hipcc-generated ----
    "bitwise_and.Tensor_out": ("hip_binary_out", "a & b"),
    "bitwise_or.Tensor_out": ("hip_binary_out", "a | b"),
    "bitwise_xor.Tensor_out": ("hip_binary_out", "a ^ b"),
    "gcd.out": ("hip_binary_out", "dtk_ops::Gcd(a, b)"),
    "hypot.out": ("hip_binary_out", "::hypot(a, b)"),
    "nextafter.out": ("hip_binary_out", "::nextafter(a, b)"),
}


# ---------------------------------------------------------------------------
# Category templates.
#
# Each returns the C++ body of one kernel. They receive the parsed schema so the
# signature is derived from native_functions.yaml rather than assumed -- that is
# what keeps a template honest when an operator has an unusual argument order.
# ---------------------------------------------------------------------------
KERNEL_TEMPLATES = ("hip_unary_out", "hip_binary_out")


# Enum tag for the device-side op selector. Derived from the schema name so it is
# stable and collision-free (mangling '.' which is not a C identifier char).
def op_tag(op: str) -> str:
    return op.replace(".", "_").replace("_out", "").upper().strip("_")


def emit_hip_unary_out(op: str, expr: str) -> str:
    """out = f(self), writing into a caller-provided output tensor.

    The .out contract is stricter than the functional one: `out` is supplied by
    the caller, may be empty, may be non-contiguous, and may alias `self`. Each
    of those is handled explicitly below -- resize to the input shape, stage
    through a contiguous buffer when the caller's `out` is not contiguous, and
    read the input through a contiguous copy so self-aliasing cannot have the
    kernel reading memory it has already overwritten.
    """
    fn_type, _ = schema_to_cpp_name(op)
    name = fn_type[:-2]  # strip the "Fn" suffix
    return f"""
// {op}: elementwise unary into a caller-provided out. HIP kernel compiled by
// hipcc -- no vendor math library and no torch device symbols involved.
at::Tensor& {name}KernelDtk(const at::Tensor& self, at::Tensor& out) {{
  // aten allows an undersized/empty out; the structured contract is to resize.
  if (!out.sizes().equals(self.sizes())) {{
    out.resize_(self.sizes());
  }}
  if (out.numel() == 0) return out;
  // Contiguous copy of the input: also breaks any aliasing with `out`, so an
  // in-place style call (out.is_same(self)) cannot read already-written data.
  auto in = self.contiguous();
  // Compute into `out` directly only when its layout lets us index it linearly.
  const bool direct = out.is_contiguous() && out.scalar_type() == self.scalar_type();
  auto dst = direct ? out : at::empty(self.sizes(), out.options().dtype(self.scalar_type()));
  dtk_ops::LaunchUnary(
      in.data_ptr(), dst.data_ptr(), dst.numel(), self.scalar_type(),
      dtk_ops::UnaryOp::{op_tag(op)},
      ::GetCurrentStreamForDevice(self.device().index()));
  if (!direct) out.copy_(dst);
  return out;
}}
"""


def emit_hip_binary_out(op: str, expr: str) -> str:
    """out = f(self, other) after broadcasting and type promotion.

    Promotion and broadcast shape come from aten's own utilities rather than
    being re-derived here, so the result matches CPU exactly; only the inner
    loop is ours.
    """
    fn_type, _ = schema_to_cpp_name(op)
    name = fn_type[:-2]
    return f"""
// {op}: elementwise binary into a caller-provided out. Broadcasting and dtype
// promotion are resolved with aten's utilities; only the inner loop is ours.
at::Tensor& {name}KernelDtk(
    const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {{
  auto promoted = at::result_type(self, other);
  auto shape = at::infer_size(self.sizes(), other.sizes());
  if (!out.sizes().equals(shape)) {{
    out.resize_(shape);
  }}
  if (out.numel() == 0) return out;
  // expand().contiguous() materializes the broadcast operands and, as with the
  // unary case, breaks aliasing between the inputs and `out`.
  auto a = self.to(promoted).expand(shape).contiguous();
  auto b = other.to(promoted).expand(shape).contiguous();
  const bool direct = out.is_contiguous() && out.scalar_type() == promoted;
  auto dst = direct ? out : at::empty(shape, out.options().dtype(promoted));
  dtk_ops::LaunchBinary(
      a.data_ptr(), b.data_ptr(), dst.data_ptr(), dst.numel(), promoted,
      dtk_ops::BinaryOp::{op_tag(op)},
      ::GetCurrentStreamForDevice(self.device().index()));
  if (!direct) out.copy_(dst);
  return out;
}}
"""


EMITTERS = {
    "hip_unary_out": emit_hip_unary_out,
    "hip_binary_out": emit_hip_binary_out,
}


# ---------------------------------------------------------------------------
# Operator set discovery
# ---------------------------------------------------------------------------
def all_dispatcher_ops() -> List[Tuple[str, str, str]]:
    """Every operator with a dispatcher slot: (op_name, fn_type, dispatcher).

    Read back out of the conf file rather than ops.h, because the conf is the
    authoritative routed set and ops.h holds only the derived C++ names.
    """
    ops: List[Tuple[str, str, str]] = []
    for line in DCU_CONF.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or " = " not in line:
            continue
        op = line.split(" = ")[0].strip()
        fn_type, disp = schema_to_cpp_name(op)
        ops.append((op, fn_type, disp))
    ops.sort(key=lambda t: t[0])
    return ops


def vendor_fallback_ops() -> List[str]:
    """Operators still served by DTK's libtorch_hip.so, i.e. the target set."""
    out = []
    for line in DCU_CONF.read_text().splitlines():
        s = line.strip()
        if s.startswith("#") or " = " not in s:
            continue
        if s.endswith("= cuda"):
            out.append(s.split(" = ")[0].strip())
    return sorted(set(out))


# ---------------------------------------------------------------------------
# --emit slots
# ---------------------------------------------------------------------------
def gen_slot_table() -> str:
    ops = all_dispatcher_ops()
    parts = [
        LICENSE,
        "//\n",
        BANNER,
        "//\n",
        "// Core-side resolution table for the DCU SDK plugin ABI (v2).\n",
        "//\n",
        "// ABI v2 hands over untyped {op_name, void* fn, fn_type} records so that\n",
        "// growing operator coverage never changes the struct layout. The cast back to\n",
        "// each dispatcher's real signature has to happen exactly once, in one audited\n",
        "// place -- this file. Every entry checks the plugin's reported fn-type tag\n",
        "// against the type this core compiled, so a signature disagreement is a clean\n",
        "// rejection instead of a wild call through a mistyped pointer.\n",
        "//\n",
        f"// {len(ops)} slots, one per routed operator.\n",
        "\n",
        '#include "../backends/dcu_sdk/dcu_sdk_abi.h"\n',
        '#include "ops.h"\n',
        '#include "../common.h"\n',
        "\n",
        "#include <cstring>\n",
        "#include <unordered_map>\n",
        "\n",
        "namespace at::native::flagos::dtk {\n",
        "\n",
        "namespace {\n",
        "\n",
        "// Installer for one operator: validates the plugin's type tag and, when\n",
        "// `commit` is set, performs the single reinterpret_cast into the typed\n",
        "// dispatcher slot. `commit=false` is the dry run the bridge uses to validate\n",
        "// an entire plugin table before mutating any dispatcher, so one bad row\n",
        "// cannot leave a partially-routed backend behind.\n",
        "using Installer = bool (*)(void* fn, const char* type_tag, bool commit);\n",
        "\n",
        "template <typename FnType, Dispatcher<FnType>* kDispatcher,\n",
        "          const char* kTypeName>\n",
        "bool InstallSlot(void* fn, const char* type_tag, bool commit) {\n",
        "  // Tag mismatch means plugin and core disagree on the signature; refuse.\n",
        "  // Comparing the generated fn-type name is what makes the cast below safe:\n",
        "  // both sides derive the tag from the same schema, so equal tags imply\n",
        "  // identical signatures.\n",
        "  if (type_tag == nullptr || std::strcmp(type_tag, kTypeName) != 0) {\n",
        "    return false;\n",
        "  }\n",
        "  if (commit) {\n",
        "    kDispatcher->RegisterKernel(\n",
        "        Backend::kDcuSdk, reinterpret_cast<FnType>(fn));\n",
        "  }\n",
        "  return true;\n",
        "}\n",
        "\n",
    ]

    # Each fn-type name needs external linkage to be a template argument.
    for op, fn_type, _ in ops:
        parts.append(f'constexpr char kName_{fn_type}[] = "{fn_type}";\n')
    parts.append("\n")

    parts.append("// op_name -> installer. Built once, read-only thereafter.\n")
    parts.append("const std::unordered_map<std::string, Installer>& SlotTable() {\n")
    parts.append(
        "  static const std::unordered_map<std::string, Installer> table = {\n"
    )
    for op, fn_type, disp in ops:
        parts.append(
            f'      {{"{op}", &InstallSlot<{fn_type}, &{disp}, kName_{fn_type}>}},\n'
        )
    parts.append("  };\n  return table;\n}\n\n")
    parts.append("} // namespace\n\n")
    parts.append(
        """// Shared lookup for both entry points below. Returns the installer, or nullptr
// with `status` set to the reason.
namespace {
Installer Resolve(const char* op_name, void* fn, FlagosDcuSdkStatus* status) {
  if (op_name == nullptr || fn == nullptr) {
    *status = kFlagosDcuSdkMissingKernel;
    return nullptr;
  }
  const auto& table = SlotTable();
  auto it = table.find(std::string(op_name));
  if (it == table.end()) {
    *status = kFlagosDcuSdkUnknownOperator;
    return nullptr;
  }
  *status = kFlagosDcuSdkOk;
  return it->second;
}
} // namespace

FlagosDcuSdkStatus CheckKernel(
    const char* op_name, void* fn, const char* type_tag) {
  FlagosDcuSdkStatus status = kFlagosDcuSdkOk;
  Installer inst = Resolve(op_name, fn, &status);
  if (inst == nullptr) return status;
  // Dry run: same type check as the install path, but nothing is written.
  return inst(fn, type_tag, /*commit=*/false)
      ? kFlagosDcuSdkOk
      : kFlagosDcuSdkSignatureMismatch;
}

FlagosDcuSdkStatus InstallKernel(
    const char* op_name, void* fn, const char* type_tag) {
  FlagosDcuSdkStatus status = kFlagosDcuSdkOk;
  Installer inst = Resolve(op_name, fn, &status);
  if (inst == nullptr) return status;
  return inst(fn, type_tag, /*commit=*/true)
      ? kFlagosDcuSdkOk
      : kFlagosDcuSdkSignatureMismatch;
}

size_t KnownSlotCount() { return SlotTable().size(); }

"""
    )
    parts.append("} // namespace at::native::flagos::dtk\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# --classify / --coverage
# ---------------------------------------------------------------------------
def classify() -> Dict[str, object]:
    """Split the vendor-fallback set by what it would actually take to serve it."""
    try:
        from torchgen.gen import parse_native_yaml
    except ImportError:
        print("error: torchgen not importable; need torch installed", file=sys.stderr)
        raise SystemExit(1)

    import torchgen

    base = Path(torchgen.__file__).parent / "packaged/ATen/native"
    nf = parse_native_yaml(str(base / "native_functions.yaml"), str(base / "tags.yaml"))
    funcs = {str(f.func.name): f for f in nf.native_functions}

    buckets: Dict[str, List[str]] = {
        "composite_implicit": [],
        "composite_explicit": [],
        "structured_delegate": [],
        "view": [],
        "needs_kernel": [],
        "not_in_yaml": [],
    }
    for op in vendor_fallback_ops():
        f = funcs.get(op)
        if f is None:
            buckets["not_in_yaml"].append(op)
        elif f.has_composite_implicit_autograd_kernel:
            buckets["composite_implicit"].append(op)
        elif (
            f.has_composite_explicit_autograd_kernel
            or f.has_composite_explicit_autograd_non_functional_kernel
        ):
            buckets["composite_explicit"].append(op)
        elif f.structured_delegate:
            buckets["structured_delegate"].append(op)
        elif f.is_view_op:
            buckets["view"].append(op)
        else:
            buckets["needs_kernel"].append(op)
    return buckets


def cmd_classify(as_json: bool) -> int:
    b = classify()
    if as_json:
        print(json.dumps({k: sorted(v) for k, v in b.items()}, indent=2))
        return 0
    total = sum(len(v) for v in b.values())
    labels = [
        ("composite_explicit", "CompositeExplicitAutograd -- generic impl reusable"),
        ("structured_delegate", "structured_delegate -- rides its .out sibling"),
        ("composite_implicit", "CompositeImplicitAutograd -- decomposes"),
        ("view", "view op -- metadata only, no compute"),
        ("needs_kernel", "NEEDS A REAL KERNEL"),
        ("not_in_yaml", "not in native_functions.yaml"),
    ]
    print(f"vendor-fallback operators (= cuda on DCU): {total}\n")
    for key, label in labels:
        print(f"  {len(b[key]):5d}  {label}")
    free = total - len(b["needs_kernel"]) - len(b["not_in_yaml"])
    print(f"\n  => kernels to design: {len(b['needs_kernel'])}")
    print(f"  => derived for free once those exist: {free}")
    return 0


def cmd_coverage() -> int:
    b = classify()
    need = set(b["needs_kernel"])
    covered = set(OPS)
    print(f"kernels required : {len(need)}")
    print(f"kernels in OPS   : {len(covered)}")
    print(f"remaining        : {len(need - covered)}")
    pct = 100.0 * len(covered & need) / max(len(need), 1)
    print(f"coverage of the required set: {pct:.1f}%")
    by_cat: Dict[str, int] = {}
    for _op, (cat, _p) in OPS.items():
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print("\nby category:")
    for cat in sorted(by_cat):
        print(f"  {by_cat[cat]:5d}  {cat}")
    extra = covered - need
    if extra:
        print(f"\nnote: {len(extra)} row(s) in OPS are not in the required set")
        print("      (already served by FlagGems or by a composite path):")
        for op in sorted(extra):
            print(f"        {op}")
    return 0


# ---------------------------------------------------------------------------
# --emit kernels
# ---------------------------------------------------------------------------
def gen_kernels() -> Tuple[str, str]:
    body = [
        LICENSE,
        "//\n",
        BANNER,
        "//\n",
        "// DCU SDK-native kernels. These link the DTK SDK and hipcc-compiled device\n",
        "// code only -- never DTK's libtorch_hip.so/libc10_hip.so -- so the plugin is\n",
        "// independent of the vendor's PyTorch build. That independence is the whole\n",
        "// point of this backend; keep it when adding categories.\n",
        "\n",
        '#include "../../../generated/ops.h"\n',
        '#include "../dtk_launch.h"\n',
        "#include <flagos.h>\n",
        "\n",
        "#include <ATen/core/Tensor.h>\n",
        "#include <ATen/ExpandUtils.h>\n",
        "#include <ATen/ops/empty.h>\n",
        "#include <ATen/ops/result_type.h>\n",
        "\n",
        "namespace at::native::flagos::dcu_sdk {\n",
        "\n",
        "namespace dtk_ops = at::native::flagos::dtk_ops;\n",
    ]
    rows: List[Tuple[str, str, str]] = []
    for op in sorted(OPS):
        cat, payload = OPS[op]
        emitter = EMITTERS.get(cat)
        if emitter is None:
            raise SystemExit(f"error: no emitter for category {cat!r} (op {op})")
        body.append(emitter(op, payload))
        fn_type, _ = schema_to_cpp_name(op)
        rows.append((op, fn_type[:-2] + "KernelDtk", fn_type))
    body.append("\n} // namespace at::native::flagos::dcu_sdk\n")

    # Declaration header rather than an .inc fragment spliced into the middle of
    # a function: the plugin then has an ordinary translation unit boundary, and
    # the compiler checks each kernel's real signature against its declaration
    # instead of the mismatch surfacing only as a link error.
    reg = [
        LICENSE,
        "//\n",
        BANNER,
        "//\n",
        "// The FlagosDcuSdkKernel table for the operators this plugin implements.\n",
        "// Included by dcu_sdk_plugin.cc. Adding an operator changes this list and\n",
        "// nothing about the ABI -- that is the point of the v2 table layout.\n",
        "\n",
        "#pragma once\n",
        "\n",
        '#include "../dcu_sdk_abi.h"\n',
        "\n",
        "#include <ATen/core/Tensor.h>\n",
        "#include <c10/core/Scalar.h>\n",
        "\n",
        "#include <cstddef>\n",
        "\n",
        "namespace at::native::flagos::dcu_sdk {\n",
        "\n",
        "// Defined in dtk_kernels.cc.\n",
    ]
    for op, sym, fn_type in rows:
        reg.append(f"// {op}\n")
        reg.append(f"extern at::Tensor& {sym}(\n")
        # Signature is derived from the category, matching the emitter above.
        cat = OPS[op][0]
        if cat == "hip_unary_out":
            reg.append("    const at::Tensor& self, at::Tensor& out);\n")
        else:
            reg.append(
                "    const at::Tensor& self, const at::Tensor& other,"
                " at::Tensor& out);\n"
            )
    reg.append("\n")
    reg.append(f"inline constexpr size_t kDtkKernelCount = {len(rows)};\n")
    reg.append("\n")
    reg.append("inline const FlagosDcuSdkKernel kDtkKernels[] = {\n")
    for op, sym, fn_type in rows:
        reg.append(f'    {{"{op}", reinterpret_cast<void*>(&{sym}), "{fn_type}"}},\n')
    reg.append("};\n")
    reg.append("\n")
    reg.append(
        "static_assert(\n"
        "    sizeof(kDtkKernels) / sizeof(kDtkKernels[0]) == kDtkKernelCount,\n"
        '    "kDtkKernelCount disagrees with the generated table");\n'
    )
    reg.append("\n} // namespace at::native::flagos::dcu_sdk\n")
    return "".join(body), "".join(reg)


# ---------------------------------------------------------------------------
# --emit conf
# ---------------------------------------------------------------------------
def gen_conf_lines() -> str:
    return "".join(f"{op} = dcu_sdk\n" for op in sorted(OPS))


# ---------------------------------------------------------------------------
def write_if_changed(path: Path, text: str, check: bool) -> bool:
    old = path.read_text() if path.exists() else None
    if old == text:
        return False
    if check:
        print(f"error: {path} is stale; re-run scripts/codegen_dtk.py", file=sys.stderr)
        raise SystemExit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--emit",
        choices=["slots", "kernels", "registry", "manifest", "conf", "all"],
        help="generate artifacts",
    )
    ap.add_argument("--classify", action="store_true", help="coverage arithmetic")
    ap.add_argument("--coverage", action="store_true", help="covered vs remaining")
    ap.add_argument("--json", action="store_true", help="machine-readable --classify")
    ap.add_argument(
        "--check",
        action="store_true",
        help="fail if any generated file would change (for CI idempotency)",
    )
    args = ap.parse_args()

    if args.classify:
        return cmd_classify(args.json)
    if args.coverage:
        return cmd_coverage()
    if not args.emit:
        ap.print_help()
        return 1

    changed: List[Path] = []
    if args.emit in ("slots", "all"):
        if write_if_changed(OUT_SLOTS, gen_slot_table(), args.check):
            changed.append(OUT_SLOTS)
    if args.emit in ("kernels", "all"):
        kern, reg = gen_kernels()
        if write_if_changed(OUT_KERNELS, kern, args.check):
            changed.append(OUT_KERNELS)
        if write_if_changed(OUT_REGISTER, reg, args.check):
            changed.append(OUT_REGISTER)
    if args.emit in ("registry", "all"):
        if write_if_changed(OUT_ROUTE_REGISTRY, gen_route_registry(), args.check):
            changed.append(OUT_ROUTE_REGISTRY)
    if args.emit in ("manifest", "all"):
        if write_if_changed(OUT_MANIFEST, gen_manifest(), args.check):
            changed.append(OUT_MANIFEST)
    if args.emit in ("conf", "all"):
        for path, content in gen_conf_artifacts().items():
            if write_if_changed(path, content, args.check):
                changed.append(path)
    if args.emit == "conf" and not args.check:
        sys.stdout.write(gen_route_conf(sdk_only=False))
        return 0

    for p in changed:
        print(f"wrote {p.relative_to(REPO)}")
    if not changed:
        print("up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
