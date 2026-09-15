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

"""
Codegen for AutogradPrivateUse1 kernels.

Why this exists
---------------
`aten::matmul` (and friends) are CompositeImplicitAutograd: PyTorch has no
backend kernel for them, it decomposes them into mm/bmm/view and lets autograd
record the *sub-ops*. A backend that wants to own the fused op (one aclnnMatmul
instead of mm + bmm + view churn) hits a wall: registering a concrete kernel on
plain PrivateUse1 stops the decomposition, so autograd stops seeing sub-ops and
instead binds the op's real derivative -- `aten::matmul_backward` -- which the
backend must then implement. Without it, training decays to CPU and crashes.

That is why csrc/aten/register.cc used to take the fused path only when
`!requires_grad`: inference got one aclnnMatmul, training fell back to the
decomposition (+816 forward and +1464 backward dispatches/step vs torch_npu on
Qwen3-0.6B).

The fix is the one torch_npu uses (its codegen/autograd/, which in turn reuses
PyTorch's own torchgen): generate a `VariableType::<op>` kernel that builds the
proper autograd node, and register it on **AutogradPrivateUse1**. That kernel
creates e.g. MatmulBackward0, calls `set_history`, and redispatches below
autograd to the backend's fused kernel. Backward then goes through
`aten::matmul_backward`, which we implement once with aclnn.

What we generate vs what torch_npu generates
--------------------------------------------
torch_npu regenerates the whole autograd stack (Functions.h/cpp,
ADInplaceOrViewType, python bindings) because it also adds *custom* ops with
*custom* derivative formulas. We only ever re-own ops that already exist in
PyTorch's derivatives.yaml, so the backward node classes (MatmulBackward0, ...)
are already compiled into libtorch and declared in the shipped
torch/csrc/autograd/generated/Functions.h. We therefore generate only the thin
VariableType layer and link against libtorch's node classes.

The function bodies come verbatim from torchgen's own `emit_body()` -- the same
code that produces PyTorch's in-tree VariableType_N.cpp -- so the autograd
bookkeeping (saved variables, version counters, fw-grad, view/inplace handling)
is exactly what upstream does, not a hand-rolled imitation.

Reads:
  - AUTOGRAD_OPS below (the ops to re-own)
  - PyTorch derivatives.yaml + native_functions.yaml (via torchgen)

Generates (into csrc/aten/generated/):
  - variable_type.cc   VariableType::<op> definitions + TORCH_LIBRARY_IMPL(aten,
                       AutogradPrivateUse1) registrations

Usage:
    python scripts/codegen/codegen_autograd.py
"""

import os
import re
import sys
from pathlib import Path

try:
    import torchgen
    from torchgen.api import cpp
    from torchgen.api.autograd import match_differentiability_info
    from torchgen.context import native_function_manager
    from torchgen.gen import parse_native_yaml
    from torchgen.packaged.autograd.gen_inplace_or_view_type import (
        METHOD_DEFINITION,
        gen_formals,
        use_derived,
    )
    from torchgen.packaged.autograd.gen_trace_type import type_wrapper_name
    from torchgen.packaged.autograd.gen_variable_type import (
        emit_body,
        gen_wrapper_registration,
    )
    from torchgen.packaged.autograd.load_derivatives import load_derivatives
except ImportError as e:  # pragma: no cover
    print(
        f"Error: torchgen not found ({e}). Install torch>=2.0 and pyyaml.",
        file=sys.stderr,
    )
    raise SystemExit(1)


# Ops to re-own on AutogradPrivateUse1.
#
# Add an op here only when the backend registers a *fused* kernel for it on
# PrivateUse1 AND the op is CompositeImplicitAutograd. The op's derivative
# (e.g. matmul -> matmul_backward) must then have a PrivateUse1 kernel, or
# backward will fall back to CPU. Ops whose backward is itself composite over
# already-registered ops need no extra work.
AUTOGRAD_OPS = [
    "matmul",
]

# Preprocessor guard wrapped around the registrations. The ops above are re-owned
# only because the Ascend backend has a fused kernel for them; on other backends
# no such kernel exists, PyTorch's composite decomposition is still what runs, and
# claiming the autograd key would bind a derivative (matmul_backward) that has no
# kernel there. Compiling the registrations out keeps those backends untouched.
REGISTRATION_GUARD = "USE_ASCEND"


# torchgen's emitted body carries two things we must strip.
#
# 1) The JVP/forward-AD branch calls `run_jit_decomposition_with_args_for_jvp`,
#    which re-enters the *composite* decomposition through the JIT. That defeats
#    the whole point (we registered a fused kernel to avoid the decomposition)
#    and drags in JIT decomposition machinery. We keep only the else-branch
#    (the plain redispatch). Consequence: forward-mode AD is not supported for
#    these ops on this backend; reverse-mode (what training uses) is unaffected.
#    torch_npu strips the same pattern for the same reason.
_JIT_DECOMP_RE = re.compile(
    r"if \(\(.*?\)\) \{.*?static c10::OperatorName full_name\("
    r"\"aten::.*?\", .*?\);\n.*?"
    r"return impl::run_jit_decomposition_with_args_for_jvp<.*?>"
    r"\(\".*?\", \*opt_op, ks, .*?\);\n\s*\} else \{\n\s*(.*?)\n\s*\}",
    re.DOTALL,
)

# 2) A debug-only assert that the result storage is uniquely owned. Our fused
#    kernels may return a tensor that shares storage with a cache entry (the
#    executor cache in op_api_common.h owns its tensors), so this NDEBUG-only
#    check can trip spuriously. Dropped, as torch_npu does.
_USE_COUNT_RE = re.compile(
    r"if \(\S+\.has_storage\(\) && !at::impl::dispatch_mode_enabled\(\) && "
    r"!at::impl::tensor_has_dispatch\(\S+\)\) \{\s+TORCH_INTERNAL_ASSERT\("
    r"\S+\.storage\(\)\.use_count\(\) == 1, \"function: \S+\"\);\s+\}",
    re.DOTALL,
)

FILE_HEADER = """\
// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// @generated by scripts/codegen/codegen_autograd.py -- DO NOT EDIT.
//
// AutogradPrivateUse1 kernels for CompositeImplicitAutograd ops that a backend
// re-owns with a fused kernel. Each function body is produced by torchgen's own
// emit_body(), the same generator behind PyTorch's in-tree VariableType_N.cpp,
// so the autograd bookkeeping matches upstream exactly. The backward node
// classes (MatmulBackward0, ...) come from libtorch; we only add the thin
// VariableType layer that builds them.
//
// Registering here (AutogradPrivateUse1) rather than PrivateUse1 is what lets
// the fused kernel run while autograd still records a proper graph: this kernel
// creates the grad_fn, then redispatches below autograd to the backend kernel.
// See scripts/codegen/codegen_autograd.py for the full rationale.

#include <ATen/RedispatchFunctions.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/generated/Functions.h>
#include <torch/library.h>

#if defined({guard})

using namespace at;
// torchgen's emitted bodies call the autograd helpers (unpack,
// compute_requires_grad, collect_next_edges, SavedVariable, set_history, ...)
// unqualified, because upstream's VariableType_N.cpp is itself compiled inside
// namespace torch::autograd. We generate into our own namespace, so pull them in.
using namespace torch::autograd;
using namespace torch::autograd::generated;

namespace at::flagos::autograd {{

namespace VariableType {{

// torchgen's bodies open with `unpack(arg, "arg", i)`. Upstream declares it in
// torch/csrc/autograd/generated/VariableType.h but defines it in
// VariableTypeManual.cpp, which is not part of the installed library -- the
// symbol is not exported, so we cannot link against it. It is only a
// defined-ness check that returns the tensor unchanged (upstream's
// checked_cast_variable), so define it here, as torch_npu likewise does for its
// own generated VariableType.
namespace {{

inline at::Tensor& unpack(at::Tensor& t, const char* name, int pos) {{
  TORCH_CHECK(t.defined(),
      "Expected a proper Tensor but got None (or an undefined Tensor in C++) "
      "for argument #", pos, " '", name, "'");
  return t;
}}

inline const at::Tensor& unpack(const at::Tensor& t, const char* name, int pos) {{
  TORCH_CHECK(t.defined(),
      "Expected a proper Tensor but got None (or an undefined Tensor in C++) "
      "for argument #", pos, " '", name, "'");
  return t;
}}

}}  // namespace
"""

FILE_FOOTER = """\
}}  // namespace VariableType

namespace {{

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {{
{registrations}}}

}}  // namespace

}}  // namespace at::flagos::autograd
"""


def main() -> int:
    repo_root = Path(__file__).parent.parent.parent
    out_dir = repo_root / "csrc/aten/generated"
    out_dir.mkdir(exist_ok=True)

    root = Path(torchgen.__file__).parent
    native_yaml = str(root / "packaged/ATen/native/native_functions.yaml")
    tags_yaml = str(root / "packaged/ATen/native/tags.yaml")
    derivatives_yaml = str(root / "packaged/autograd/derivatives.yaml")

    print("Loading derivatives and native functions via torchgen...")
    infos, _ = load_derivatives(derivatives_yaml, native_yaml, tags_yaml)
    native_funcs = parse_native_yaml(native_yaml, tags_yaml).native_functions
    fns = match_differentiability_info(native_funcs, infos)

    wanted = set(AUTOGRAD_OPS)
    definitions: list[str] = []
    registrations: list[str] = []
    seen: set[str] = set()

    for fn in fns:
        name = str(fn.func.func.name)
        if name not in wanted:
            continue
        if fn.info is None:
            print(
                f"  WARNING: {name} has no derivative info; skipping", file=sys.stderr
            )
            continue
        if not use_derived(fn):
            print(
                f"  WARNING: {name} is not a derived-type function; skipping",
                file=sys.stderr,
            )
            continue

        with native_function_manager(fn.func):
            body = emit_body(fn, "Default")
            definition = METHOD_DEFINITION.substitute(
                return_type=cpp.returns_type(fn.func.func.returns).cpp_type(),
                type_wrapper_name=type_wrapper_name(fn.func),
                type_definition_body=body,
                formals=gen_formals(fn.func),
            )
            definition = _JIT_DECOMP_RE.sub(r"\1", definition)
            definition = _USE_COUNT_RE.sub("", definition)
            definitions.append(definition)
            registrations.append(gen_wrapper_registration(fn.func, "Default"))
        seen.add(name)
        node = ", ".join(sorted({i.op for i in fn.info.values() if i.op}))
        print(f"  {name}: autograd node {node}")

    missing = wanted - seen
    if missing:
        print(f"Error: no derivative found for {sorted(missing)}", file=sys.stderr)
        return 1

    out = out_dir / "variable_type.cc"
    body = "\n".join(definitions)
    regs = "".join(f"  {r}\n" for r in registrations)
    text = (
        FILE_HEADER.format(guard=REGISTRATION_GUARD)
        + "\n"
        + body
        + "\n"
        + FILE_FOOTER.format(registrations=regs)
        + f"\n#endif  // {REGISTRATION_GUARD}\n"
    )
    out.write_text(text)
    print(
        f"Generated {out.relative_to(repo_root)} "
        f"({len(definitions)} kernel(s), {os.path.getsize(out)} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
