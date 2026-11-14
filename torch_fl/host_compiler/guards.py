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

import sympy
import torch
from torch._inductor.codegen.cpp_utils import cexpr


def expression(value):
    return (
        value.node.expr
        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool))
        else sympy.sympify(value)
    )


def native_guard(gm):
    tensors = [n.meta["val"] for n in gm.graph.nodes if n.op == "placeholder"]
    env = gm.shape_env
    bindings = {}
    checks = []
    decls = []
    for i, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            raise RuntimeError("probe expects tensor placeholders")
        checks.append(f"args[{i}].dim() == {t.ndim}")
        names = {
            "float32": "Float",
            "float64": "Double",
            "float16": "Half",
            "bfloat16": "BFloat16",
            "bool": "Bool",
            "int64": "Long",
            "int32": "Int",
            "int16": "Short",
            "int8": "Char",
            "uint8": "Byte",
        }
        dtype = names[str(t.dtype).split(".")[-1]]
        checks.append(f"args[{i}].scalar_type() == at::ScalarType::{dtype}")
        checks.append(f"args[{i}].device().type() == c10::kPrivateUse1")
        checks.append(f"args[{i}].device().index() == {t.device.index}")
        checks.append(f"!args[{i}].is_conj() && !args[{i}].is_neg()")
        checks.append(
            f"(reinterpret_cast<uintptr_t>(args[{i}].const_data_ptr()) % 16) == 0"
        )
        metadata = [(v, f"args[{i}].size({j})") for j, v in enumerate(t.shape)]
        metadata += [(v, f"args[{i}].stride({j})") for j, v in enumerate(t.stride())]
        metadata.append((t.storage_offset(), f"args[{i}].storage_offset()"))
        for v, runtime in metadata:
            e = env.simplify(expression(v))
            if isinstance(e, sympy.Symbol) and e not in bindings:
                bindings[e] = runtime
                decls.append(f"int64_t {e} = {bindings[e]};")
    expr_checks = []

    def render(expr):
        expr = env.simplify(expression(expr))
        unknown = expr.free_symbols - set(bindings)
        if unknown:
            raise RuntimeError(f"unbound guard symbols: {unknown}")
        return str(cexpr(expr))

    for i, t in enumerate(tensors):
        for j, v in enumerate(t.shape):
            expr_checks.append(f"args[{i}].size({j}) == ({render(v)})")
        for j, v in enumerate(t.stride()):
            expr_checks.append(f"args[{i}].stride({j}) == ({render(v)})")
        expr_checks.append(
            f"args[{i}].storage_offset() == ({render(t.storage_offset())})"
        )
    guard_exprs = [env.simplify(g.expr) for g in env.guards]
    replacements, reduced = sympy.cse(
        guard_exprs, symbols=sympy.numbered_symbols("guard_tmp_")
    )
    guard_decls = []
    for symbol, expr in replacements:
        guard_decls.append(f"auto {symbol} = {render(expr)};")
        bindings[symbol] = "temporary"
    expr_checks.extend(render(expr) for expr in reduced)
    range_checks = []
    for symbol in list(bindings):
        limits = env.var_to_range.get(symbol)
        if limits:
            for bound, op in [(limits.lower, ">="), (limits.upper, "<=")]:
                if isinstance(bound, sympy.Integer):
                    range_checks.append(f"{symbol} {op} {bound}")
    checks = list(dict.fromkeys(checks))
    expr_checks = list(dict.fromkeys(expr_checks))
    lines = [
        "#include <ATen/ATen.h>",
        'extern "C" int flagos_host_guard_version() { return 2; }',
        'extern "C" bool flagos_host_guard(const at::Tensor* args, size_t count) {',
        f"  if (count != {len(tensors)}) return false;",
    ]
    lines += [f"  if (!({c})) return false;" for c in checks]
    lines += ["  " + d for d in decls]
    # Validate ranges before CSE temporaries can divide by a derived tile count.
    lines += [f"  if (!({c})) return false;" for c in range_checks]
    lines += ["  " + d for d in guard_decls]
    lines += [f"  if (!({c})) return false;" for c in expr_checks]
    lines += ["  return true;", "}"]
    return "\n".join(lines) + "\n"
