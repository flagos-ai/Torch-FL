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

"""Which callable the C++ side hands a FlagGems route to.

codegen freezes the *generic* ``flag_gems.ops.*`` path into every generated
table so the artifact stays hardware-portable. That path must not be what the
process calls: FlagGems' ``SpecOpRegistrar`` rebinds the package-level name to
the vendor implementation for the active backend, and discovery measured the
arity and tune-config schema against that vendor callable. Calling the generic
kernel with a vendor-shaped config is how ``mm.out`` died on Ascend --
``_ascend.ops.mm`` declares ``SPLIT_K``, ``flag_gems/ops/mm.py:mm_kernel_general``
does not, and the mismatch surfaces inside the tuner's own benchmark run as
``KeyError: 'Keyword argument SPLIT_K was specified but unrecognised'``.

Two resolution paths have to agree on that: ``GetFunc`` (dotted names, the
plain calls) and ``GetFuncBare`` (the keyword-argument calls ``mm_out`` and the
``.out`` forms use). They did not, and ``mm`` worked while ``mm.out`` did not.

These assertions read the C++ rather than linking it: the file needs pybind11
and a Python interpreter to compile, and the property being guarded is
structural -- which lookup comes first, and which names are allowed to take it.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_CALLER = REPO_ROOT / "csrc/aten/backends/flagos/python_op_caller.cc"
_FLAGGEMS_KERNELS = REPO_ROOT / "csrc/aten/generated/flaggems_python_kernels.cc"
_TILEOPS_KERNELS = REPO_ROOT / "csrc/aten/generated/tileops_python_kernels.cc"

_PYTHON_OP_CALLER = (
    _FLAGGEMS_KERNELS,
    _TILEOPS_KERNELS,
)


def _read(path: Path) -> str:
    """``path``'s text, or skip when this job has no checkout to read it from.

    These pins read the arm out of the tree rather than measuring it, so they
    answer on a host with no device -- but not on a job that installed the wheel
    and runs from a temporary directory, where ``REPO_ROOT`` holds no ``csrc/``.
    Reading it there is a ``FileNotFoundError``, which reads as the regression
    the pin looks for and is not one, so it is a skip (same reasoning as
    ``tests/integration/ops/test_dtype_cast_copy.py``, #675).
    """
    if not path.is_file():
        pytest.skip(f"no source checkout to read {path} from")
    return path.read_text(encoding="utf-8")


def _source() -> str:
    return _read(_CALLER)


def _function_body(source: str, signature: str) -> str:
    """A function from its signature to the matching close brace.

    Braces are counted rather than matched by indentation: the two keyword-call
    entry points close at different columns, and a nested block would end an
    indentation match early. Line comments and string literals are skipped so
    their braces do not count.
    """
    start = source.index(signature)
    open_at = source.index("{", start)
    depth = 0
    i = open_at
    while i < len(source):
        ch = source[i]
        if ch == "/" and source.startswith("//", i):
            i = source.index("\n", i)
            continue
        if ch == '"':
            i += 1
            while source[i] != '"':
                i += 2 if source[i] == "\\" else 1
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    raise AssertionError(f"unterminated body for {signature!r}")


def test_keyword_calls_ask_the_active_backend_first():
    """``GetFuncBare`` resolves through VendorOpByName before flag_gems.ops.

    This is the Ascend ``mm.out`` fix (CI 40656617119 at 0a2a74b3).
    """
    body = _function_body(_source(), "py::object GetFuncBare(const char* name) {")

    vendor_at = body.index("VendorOpByName(")
    generic_at = body.index("OpsModule().attr(")
    assert vendor_at < generic_at, (
        "GetFuncBare must try the vendor-resolved callable before falling back "
        "to the generic flag_gems.ops kernel"
    )

    # The generic lookup must remain the fallback, not be replaced: a few
    # private helpers exist only under flag_gems.ops.
    assert "if (!func)" in body[vendor_at:generic_at], (
        "the flag_gems.ops lookup must be guarded by the vendor lookup failing"
    )


def test_keyword_calls_only_consult_the_backend_for_flag_gems_names():
    """``torch_fl`` TileOPs shims must not import flag_gems.

    The cache is shared, and a host with TileOPs but no FlagGems fails every
    TileOPs call the moment flag_gems is imported for one of its shims.
    """
    body = _function_body(_source(), "py::object GetFuncBare(const char* name) {")

    assert 'qual.rfind("flag_gems", 0) == 0' in body, (
        "the vendor lookup in GetFuncBare must be gated on the flag_gems "
        "qualifier, or a TileOPs-only host will try to import flag_gems"
    )


def test_both_keyword_call_forms_use_the_shared_lookup():
    """``CallPythonOp_GenericKw`` and the tuple form share GetFuncBare.

    The tuple form serves ``mm.out``-style ops where part of the signature is
    positional; if it resolved independently it would keep the generic callable.
    """
    source = _source()
    for signature in (
        "at::Tensor CallPythonOp_GenericKw(",
        "CallPythonOp_GenericKwTuple(",
    ):
        body = _function_body(source, signature)
        assert "cache.GetFuncBare(func_name)" in body


def test_generated_targets_are_qualified_so_the_cache_key_is_unambiguous():
    """Every generated Python-op target names its module.

    ``GetFuncBare`` keys its cache on the bare name. A qualified target starting
    with ``flag_gems`` resolves vendor-first and any other qualified target is
    left alone, so the shared entry is correct; a bare target would carry no
    qualifier at all and could not be classified.
    """
    pattern = re.compile(r'CallPythonOp_GenericKw(?:Tuple)?\(\s*"([^"]+)"')
    targets = set()
    for path in _PYTHON_OP_CALLER:
        targets.update(pattern.findall(_read(path)))

    assert targets, "no Python-op targets found in the generated kernel tables"
    unqualified = sorted(t for t in targets if "." not in t)
    assert not unqualified, (
        "generated Python-op targets must be dotted qualnames; these are bare: "
        f"{unqualified}"
    )
