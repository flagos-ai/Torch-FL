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

"""The generated matrix wrappers and the software low-precision calls they make.

Two rules, both learned from failures the hardware lane reported:

1. A wrapper's low-precision gate comes before any soft call the same schema
   can make for an ordinary pair. ``bmm.out`` has a soft call for FP8/FP4 and,
   if the wrapper took it on every dtype, every bf16 ``bmm(..., out=)`` would
   be handed to the software path as well.

2. A schema the generator does not emit a wrapper for is not named in any of
   its tables. A soft call for an op the build never registers is unreachable
   -- the FP8/FP4 gate went missing on MetaX that way, a wrapper existed and
   nothing claimed it -- and an entry that can never be emitted reads as
   coverage that does not exist.

3. The routed call and the conf's route agree on the signature. The forms whose
   soft call writes the caller's tensor spread the operands into their own
   helpers, so an operand in the wrong position binds to the next parameter
   rather than failing to compile (``addmm.out`` passed ``self`` where the bias
   goes), and a route check made against the const-``out`` dispatcher says yes
   for a key only the in-place signature answers (``addmm_``).

The generator is parsed rather than imported: it pulls in torchgen and the
platform confs at import time, and the tables are plain literals.
"""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_CODEGEN = REPO_ROOT / "scripts/codegen/codegen_ops.py"
_REGISTER_INC = REPO_ROOT / "csrc/aten/generated/register.inc"
_OPS_CC = REPO_ROOT / "csrc/aten/backends/soft_lowp/ops.cc"

_TABLES = (
    "_SOFT_LOWP_DIRECT_CALLS",
    "_SOFT_LOWP_GATE_OPS",
)


def _generator_tables() -> dict[str, set[str]]:
    """The soft-lowp name tables, as sets of schema names."""
    tree = ast.parse(_CODEGEN.read_text(encoding="utf-8"))
    tables: dict[str, set[str]] = {}
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in _TABLES
        ):
            continue
        value = node.value
        names: set[str] = set()
        if isinstance(value, ast.Set):
            names = {e.value for e in value.elts if isinstance(e, ast.Constant)}
        elif isinstance(value, ast.Dict):
            names = {k.value for k in value.keys if isinstance(k, ast.Constant)}
        tables[node.targets[0].id] = names
    return tables


def _emitted_ops() -> set[str]:
    """The schemas register.inc supplies an m.impl for."""
    return set(
        re.findall(r'm\.impl\("([^"]+)",', _REGISTER_INC.read_text(encoding="utf-8"))
    )


def test_every_soft_lowp_table_entry_is_a_schema_or_a_conf_key():
    """Each name is a schema the generator knows, and its value is not a schema.

    A name the generator has never seen would make gen_wrapper's `op in table`
    test a no-op, so the table would read as coverage while gating nothing --
    which is how the FP8/FP4 gate went missing on MetaX (the wrapper was never
    emitted and nothing claimed it).
    """
    tables = _generator_tables()
    assert set(tables) == set(_TABLES), (
        f"codegen_ops.py no longer defines {sorted(set(_TABLES) - set(tables))}; "
        "this test reads those tables by name"
    )
    known = _generator_schema_names()
    unknown = sorted(set().union(*tables.values()) - known)
    assert not unknown, (
        f"codegen_ops.py names {unknown} in its software low-precision tables, "
        "but the generator has no rule for them: the entry gates nothing."
    )


def _generator_schema_names() -> set[str]:
    """Every schema name a string literal in the generator spells.

    The matrices are spread over the soft-lowp tables, the dispatcher table, the
    flaggems skip lists and the call maps; a name in any of them is one the
    generator's own gates can match.
    """
    tree = ast.parse(_CODEGEN.read_text(encoding="utf-8"))
    names = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    tables = _generator_tables()
    for values in tables.values():
        names |= values
    return names


_OPENING_BRACE = re.compile(r"\{\s*\n")
_CLOSING_BRACE = re.compile(r"\n\}")


def _wrapper_body(register_inc: str, wrapper: str) -> str | None:
    """The body of one generated wrapper, or None when there is no such wrapper.

    The generator sorts its schemas case-insensitively, so the wrappers are not
    in case-sensitive name order (`WrapperBmmDtypeOut` follows `WrapperBmm`,
    but `WrapperBlockDiag` does not). Search for the definition itself rather
    than for a position in the file.
    """
    # Functional rather than one regex: the parameter list repeats the name, so
    # a non-greedy signature match runs into that copy on a file with several
    # wrappers and then backtracks through the rest of the file.
    start = re.search(rf"\n[^\n]*?\b{wrapper}\(", register_inc)
    if start is None:
        return None
    opening = _OPENING_BRACE.search(register_inc, start.end())
    if opening is None:
        return None
    closing = _CLOSING_BRACE.search(register_inc, opening.end())
    if closing is None:
        return None
    return register_inc[opening.end() : closing.start()]


_FRESH_REGISTER_INC = """

at::Tensor & WrapperBmmOut(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
#if defined(FLAGOS_SOFT_LOWP)
  if (!at::native::flagos::soft_lowp::IsDispatchSuppressed() && (at::native::flagos::soft_lowp::IsLowPrecision(self.scalar_type()))) {
    return at::native::flagos::soft_lowp::BmmOut(self, mat2, out);
  }
#endif
  return at::native::flagos::bmm_out_dispatcher(self, mat2, out);
}
"""

_STALE_REGISTER_INC = """

at::Tensor & WrapperBmmOut(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
    return at::native::flagos::soft_lowp::BmmOut(self, mat2, out);
}
"""


def _assert_bmm_out_gated(register_inc: str) -> None:
    body = _wrapper_body(register_inc, "WrapperBmmOut")
    assert body is not None, "register.inc no longer defines WrapperBmmOut"
    lowp_gate = body.find("IsLowPrecision")
    soft_call = body.find("soft_lowp::BmmOut")
    assert lowp_gate != -1 and soft_call != -1, (
        "WrapperBmmOut must gate on IsLowPrecision and call "
        "soft_lowp::BmmOut; one of the two is gone:\n" + body
    )
    assert lowp_gate < soft_call, (
        "WrapperBmmOut calls soft_lowp::BmmOut before checking "
        "IsLowPrecision, so an ordinary bf16 `bmm(..., out=)` would take the "
        "software path:\n" + body
    )


def test_bmm_out_gate_check_accepts_a_gated_wrapper():
    _assert_bmm_out_gated(_FRESH_REGISTER_INC)


def test_bmm_out_gate_check_rejects_an_ungated_wrapper():
    with pytest.raises(AssertionError):
        _assert_bmm_out_gated(_STALE_REGISTER_INC)


def test_generated_bmm_out_is_gated():
    """The shipped register.inc, when it carries a WrapperBmmOut at all.

    No generation on this tree emits one (bmm.out is in no CUDA conf as a route
    of its own), so the file may legitimately have none. Where a platform's
    generation does emit it, the shipped copy is what CI gated, and a stale one
    -- the FP8/FP4 gate missing -- fails on that platform rather than here.
    """
    register_inc = _REGISTER_INC.read_text(encoding="utf-8")
    if _wrapper_body(register_inc, "WrapperBmmOut") is None:
        return
    if "soft_lowp::BmmOut(" not in register_inc:
        pytest.skip(
            "register.inc predates the bmm.out soft call; "
            "regenerate on the platform that routes bmm.out"
        )
    _assert_bmm_out_gated(register_inc)


_ADDMM_OUT_CALL = re.compile(r"RunMatrixOut\(\"addmm\.out\",\s*(.*?)\);\n\}", re.S)
_ADDMM_OUT_ROUTE = re.compile(r"case DecodedShape::kAddmmOut: \{\n(.*?)\n    \}", re.S)


def _argument_list(source: str) -> list[str]:
    """Split a C++ argument list at its top-level commas."""
    args, depth, start = [], 0, 0
    for index, char in enumerate(source):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(source[start:index].strip())
            start = index + 1
    args.append(source[start:].strip())
    return args


def test_addmm_out_passes_its_bias_where_the_bias_goes():
    """``addmm.out`` is ``RunMatrixOut(variant, self, mat2, dtype, rank, bias, ...)``.

    The op's one input tensor is its *bias*; the soft call was handed ``self``
    as it, which decodes that operand (it is dense, so nothing changes for it)
    and gives the bias the const ``self`` -- while ``mat1`` and ``mat2`` took
    the bias's place. It compiled on every target whose ``ScalarType`` is an
    enum class, and stopped at the signature on DCU, where the enum converts to
    int.
    """
    match = _ADDMM_OUT_CALL.search(_OPS_CC.read_text(encoding="utf-8"))
    assert match is not None, "soft_lowp::AddmmOut no longer calls RunMatrixOut"
    args = _argument_list(match.group(1))
    assert len(args) == 8, args
    self_, mat2, out_dtype, rank, bias = args[:5]
    assert (self_, mat2, rank) == ("mat1", "mat2", "2"), (
        f"soft_lowp::AddmmOut no longer hands addmm.out its own operands: {args}"
    )
    assert out_dtype == "out.scalar_type()", args
    assert bias == "self", (
        "soft_lowp::AddmmOut must pass self as the bias (addmm's `self`), "
        f"got {bias!r}: {args}"
    )
    assert args[7] == "out", args


def test_addmm_out_routes_through_a_dispatcher_that_writes_out():
    """The decoded pair may only be routed where ``out`` is what the kernel writes.

    ``DispatchDecodedOut``'s addmm.out arm asks ``addmm_out_dispatcher``, whose
    kernel takes the bias, both operands, both scalars and ``out``; every arm
    before it hands ``out`` to a kernel that writes it. Passing anything else
    (a re-bound ``self``, the general route's const operand) either writes the
    wrong tensor or binds a const tensor to ``Tensor&``.

    ``Select`` names the key the dispatcher answers for (``addmm``), not the
    caller's: ``addmm_`` reaches this arm too and has no key of its own, and a
    key it names no kernel for runs Select once for nothing.
    """
    source = _OPS_CC.read_text(encoding="utf-8")
    match = _ADDMM_OUT_ROUTE.search(source)
    assert match is not None, "DispatchDecodedOut no longer routes addmm.out"
    arm = match.group(1)
    assert "addmm_out_dispatcher" in arm, arm
    call = re.search(r"AddmmOutCall call\{(\w+)\}", arm)
    assert call is not None and call.group(1) == "addmm_out_dispatcher", arm
    called = re.search(
        r"call\.Call\(call\.Select\(addmm_dispatcher\.OpKey\(\)\), (.*?)\);",
        arm,
        re.S,
    )
    assert called is not None, arm
    args = _argument_list(called.group(1))
    assert args == [
        "pair.bias",
        "pair.lhs",
        "pair.rhs",
        "beta",
        "alpha",
        "out",
    ], args


def test_addmm_inplace_computes_through_the_out_writer():
    """``addmm_`` has only the in-place signature, and keeps no route of its own.

    The const-``out`` general addmm dispatcher can resolve for a conf key whose
    in-place slot holds nothing -- on a build without FlagGems C++ kernels the
    fallback wrappers fill the former and not the latter -- and a yes there
    routes to a kernel that is not there. So ``addmm_`` has no variant entry
    and no route check, and reaches the conf's route through
    ``RunMatrixOut``'s addmm.out arm, whose kernel writes the caller's tensor.
    """
    source = _OPS_CC.read_text(encoding="utf-8")
    handlers = source[source.find("bool DecodedPairHasRoute(") :]
    handlers = handlers[: handlers.find("\n}\n")]
    assert "addmm_inplace_dispatcher.RouteAcceptsDecoded" not in handlers, (
        "DecodedPairHasRoute must not ask addmm_inplace_dispatcher about "
        "addmm_: the general addmm dispatcher answers for a different "
        "signature, and its slot is the one that can be filled"
    )
    body = source[source.find("at::Tensor& AddmmInplace(") :]
    body = body[: body.find("\n}\n")]
    assert 'RunMatrixOut("addmm_"' in body, body


_PYTHON_CALLER = REPO_ROOT / "csrc/aten/backends/flagos/python_op_caller.cc"


def test_a_decoded_fp8_out_dtype_reaches_the_python_route():
    """An FP8 ``out_dtype`` gets past the Python route's dtype conversion.

    The FlagGems route is reached through ``python_op_caller.cc``, which
    converts a dtype to a Python object before gems sees the call. It used to
    name only the dtypes a FlagGems call is made with, so an FP8 ``out_dtype``
    raised there (``OptionalDtypeToPython has no Python dtype for ScalarType
    24``) on every DCU ``addmm`` the conf routes to FlagGems -- through the
    keyword path and through ``CallPythonOp_TD`` alike. Each of those call
    sites would have needed the same opt-in, so the conversion itself now names
    the four FP8 types (soft_lowp/format.h) and every caller gets them.
    """
    source = _PYTHON_CALLER.read_text(encoding="utf-8")
    start = source.find("py::object OptionalDtypeToPython(")
    assert start != -1, "python_op_caller.cc no longer defines OptionalDtypeToPython"
    body = source[start:]
    body = body[: body.find("\n}\n")]
    assert "soft_lowp::Fp8PythonDtype(*dtype)" in body, (
        "OptionalDtypeToPython must name the FP8 dtypes; otherwise an FP8 "
        "out_dtype raises here before the route runs:\n" + body
    )
