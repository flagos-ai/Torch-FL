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

"""Unit coverage for the ``tensorlist_dim`` FlagGems codegen category.

``cat(tensors, dim)`` is ``(Tensor[], int) -> Tensor``. ``detect_category()``
sees the TensorList and files it under ``foreach_tensorlist`` -- the
``_foreach_*`` family, whose kernels are rank-identical elementwise unaries --
which is not a py category at all, so the schema alone does not route cat to the
FlagGems Python path. ``_FLAGGEMS_TENSORLIST_DIM`` is the explicit membership
list that does, and the emitter's ``tensorlist_dim`` branch turns it into a call
through the dedicated list caller instead of the generic positional one.

Nothing here needs a parsed native YAML, a flag_gems install, or a device: the
generator is loaded by path and its emitter driven directly, and the rest is
text read back out of the committed generated sources. ``codegen_ops`` does need
``torchgen`` importable, so on a host without it the whole file skips rather than
erroring. What is asserted is the whole bridge, because each link can break on
its own:

    _FLAGGEMS_TENSORLIST_DIM  ("cat" is a member)
      -> discover gate        (the set is consulted, matches are filed here)
      -> emitter              (category -> the exact committed CatKernelPython)
      -> ops.h typedef        (the emitted signature satisfies CatFn)
      -> python_op_caller.h   (the caller accepts those argument types)
      -> register.inc         (m.impl("cat", ...) reaches cat_dispatcher)

Only the first two run the generator; the rest pin the generated artifact that
was reviewed and committed alongside the generator change, which is what the
``codegen-checks`` lanes cannot check on their own -- ``codegen_ops.py`` needs
torchgen and an importable built ``torch_fl``, so it is a manual regen.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest


# tests/unit/<this file> -> repo root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CODEGEN = _REPO_ROOT / "scripts" / "codegen" / "codegen_ops.py"
_COVERAGE = _REPO_ROOT / "scripts" / "codegen" / "backend_coverage.py"
_OPS_H = _REPO_ROOT / "csrc" / "aten" / "generated" / "ops.h"
_REGISTER_INC = _REPO_ROOT / "csrc" / "aten" / "generated" / "register.inc"
_KERNELS_CC = _REPO_ROOT / "csrc" / "aten" / "generated" / "flaggems_python_kernels.cc"
_CALLER_H = _REPO_ROOT / "csrc" / "aten" / "backends" / "flagos" / "python_op_caller.h"

# The CppSignature torchgen produces for `cat(Tensor[] tensors, int dim) -> Tensor`
# on the PrivateUse1 path: the schema's `Tensor[]` is spelled `const
# at::ITensorListRef &` because cat is part of a structured group. These are the
# pairs the generator sees in `op_info[op]["args"]`.
_CAT_CPP_ARGS = [
    ("const at::ITensorListRef &", "tensors"),
    ("int64_t", "dim"),
]


def _load():
    """Import codegen_ops.py by path -- scripts/ is not an importable package.

    This needs ``torchgen`` on the path: ``codegen_ops`` imports it at module
    scope and exits the process when it is missing, so the whole file (including
    the class below, which fires at import time) is skipped wherever torchgen is
    absent. That is the same dependency ``codegen_ops.py`` itself has, so the
    test still runs wherever the generator can run at all.
    """
    spec = importlib.util.spec_from_file_location("_codegen_ops_under_test", _CODEGEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# `codegen_ops` exits the process when torchgen is missing, so gate on it here:
# a host that cannot run the generator should skip this file, not error on it.
pytest.importorskip("torchgen", reason="codegen_ops.py needs torchgen to import")

codegen_ops = _load()


class _FakeArg:
    """One aten argument, standing in for torchgen's ``Argument``.

    The emitter walks ``flat_all`` to build the generic caller's positional list
    before it dispatches on the category, so these need the two attributes it
    reads there. The tensorlist_dim branch then takes its parameter names from
    the CppSignature pairs passed as ``args`` instead, so nothing else about
    these values has to be torchgen-shaped.
    """

    def __init__(self, type_, name):
        self.type = type_
        self.name = name


class _FakeArguments:
    def __init__(self, pairs):
        self.flat_all = [_FakeArg(t, n) for t, n in pairs]
        self.out = []


class _FakeSignature:
    def __init__(self, pairs):
        self.arguments = _FakeArguments(pairs)


class _FakeNativeFunction:
    """``func.func`` is the schema; only ``.arguments`` is reachable here."""

    def __init__(self, pairs):
        self.func = _FakeSignature(pairs)


def _read(path: Path) -> str:
    assert path.is_file(), f"expected generated/source file is missing: {path}"
    return path.read_text()


def _emit(**overrides) -> str:
    """Generate the kernel body for cat through the real emitter."""
    call = dict(
        op="cat",
        fn_type="CatFn",
        ret_type="at::Tensor",
        args=list(_CAT_CPP_ARGS),
        gems_func="flag_gems.cat",
        category="tensorlist_dim",
        func=_FakeNativeFunction([("Tensor[]", "tensors"), ("int", "dim")]),
        kwargs=None,
    )
    call.update(overrides)
    return codegen_ops.gen_flaggems_python_kernel(**call)


def _committed_cat_kernel() -> str:
    """The ``CatKernelPython`` definition as it was committed."""
    match = re.search(r"at::Tensor CatKernelPython\(.*?\n\}", _read(_KERNELS_CC), re.S)
    assert match, f"no CatKernelPython definition in {_KERNELS_CC.name}"
    return match.group(0)


def _module_op_set(module_path: Path, name: str) -> set:
    """Read a ``name = frozenset({...})`` literal out of a script."""
    for node in ast.walk(ast.parse(_read(module_path))):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            continue
        value = node.value
        if isinstance(value, ast.Call):
            value = value.args[0]
        return set(ast.literal_eval(value))
    raise AssertionError(f"{name} is missing from {module_path.name}")


def _discover_body() -> ast.FunctionDef:
    module = ast.parse(_read(_CODEGEN))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "discover_flaggems_ops":
            return node
    raise AssertionError("discover_flaggems_ops is missing from codegen_ops.py")


class TestCategoryMembership:
    """The op is admitted by an explicit list, not by the schema."""

    def test_cat_is_a_member(self):
        assert "cat" in codegen_ops._FLAGGEMS_TENSORLIST_DIM

    def test_the_category_is_not_a_py_category(self):
        """Membership is the only way in, so the name must not leak into the set.

        ``_FLAGGEMS_TENSORLIST_DIM`` ops are rejected by the schema-shaped gate
        (``detect_category``) and admitted by the list; if ``tensorlist_dim``
        were also a py category the emitter branch would become reachable
        without the arity gate that guards it.
        """
        assert "tensorlist_dim" not in codegen_ops._FLAGGEMS_PY_CATEGORIES

    def test_discovery_consults_the_set(self):
        """``discover_flaggems_ops`` must reference the list, or cat disappears.

        A source-level assertion because driving the real function needs
        torchgen, torch_fl and flag_gems. It is written against the AST rather
        than the text so that reformatting the gate does not fail it -- only
        dropping the reference does.
        """
        names = {
            node.id for node in ast.walk(_discover_body()) if isinstance(node, ast.Name)
        }
        assert "_FLAGGEMS_TENSORLIST_DIM" in names, (
            "discover_flaggems_ops no longer reads _FLAGGEMS_TENSORLIST_DIM, so "
            "its members are never discovered"
        )

    def test_matches_are_filed_under_this_category(self):
        literals = {
            node.value
            for node in ast.walk(_discover_body())
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert "tensorlist_dim" in literals, (
            "discover_flaggems_ops does not store the tensorlist_dim category"
        )

    def test_cat_is_in_the_coverage_ceiling(self):
        """Coverage gates the conf route: without this entry cat routes to cuda.

        ``gen_vendor_confs.py`` intersects each platform conf with
        ``FLAGGEMS_PYTHON_OPS``, so a conf line of ``cat = flaggems`` only holds
        while the op is in the set.
        """
        assert "cat" in _module_op_set(_COVERAGE, "FLAGGEMS_PYTHON_OPS")


class TestEmitter:
    """The category produces the list-call body and nothing else."""

    def test_body_calls_the_list_caller(self):
        body = _emit()
        assert 'CallPythonOp_ListI("flag_gems.cat", tensors, dim);' in body

    def test_body_unboxes_and_returns_the_result(self):
        body = _emit()
        assert "UnboxToFlagos(result);" in body
        assert "return result;" in body

    def test_parameter_names_come_from_the_cpp_signature(self):
        """The decl has to match ``CatFn``, so it cannot be rebuilt from the schema.

        torchgen spells the schema's ``Tensor[]`` as ``const at::ITensorListRef
        &``; taking the names from the CppSignature pairs is what makes the
        emitted declaration satisfy the typedef.
        """
        assert _emit().startswith(
            "at::Tensor CatKernelPython(const at::ITensorListRef & tensors, "
            "int64_t dim) {"
        )

    def test_missing_tensorlist_raises(self):
        """The guard, not a silently wrong call, when the shape is unexpected."""
        with pytest.raises(ValueError, match="tensorlist_dim needs"):
            _emit(args=[("int64_t", "dim")])

    def test_missing_dim_raises(self):
        with pytest.raises(ValueError, match="tensorlist_dim needs"):
            _emit(args=[("const at::ITensorListRef &", "tensors")])

    def test_committed_kernel_is_what_the_emitter_produces(self):
        """The reviewed artifact and the generator must not have drifted.

        ``codegen_ops.py`` is a manual regen, so nothing in CI regenerates this
        file; comparing the two is the only automatic check that the committed
        definition is still the one the emitter writes.
        """
        assert _committed_cat_kernel() == _emit()


class TestGeneratedArtifact:
    """The committed sources the emitter's output has to fit into."""

    def test_signature_satisfies_the_fn_typedef(self):
        expected = "at::Tensor (*)(" + ", ".join(t for t, _ in _CAT_CPP_ARGS) + ")"
        assert f"using CatFn = {expected};" in _read(_OPS_H), (
            "CatFn no longer matches the signature the generator emits for cat"
        )

    def test_the_caller_accepts_these_argument_types(self):
        header = re.sub(r"\s+", " ", _read(_CALLER_H))
        assert "CallPythonOp_ListI(const char* func_name," in header
        assert "const at::ITensorListRef& tensors" in header
        assert "int64_t dim)" in header

    def test_the_kernel_is_registered_on_the_flaggems_backend(self):
        assert (
            "REGISTER_IMPL_TO_DISPATCHER(CatFn, cat_dispatcher, "
            "Backend::kFlagGems, CatKernelPython)" in _read(_KERNELS_CC)
        )

    def test_the_op_is_implemented_and_reaches_the_dispatcher(self):
        register = _read(_REGISTER_INC)
        assert 'm.impl("cat", WrapperCat);' in register
        wrapper = re.search(r"\bWrapperCat\([^;{]*\)\s*\{(.*?)\n\}", register, re.S)
        assert wrapper, "register.inc has no WrapperCat definition"
        assert "cat_dispatcher(" in wrapper.group(1)
