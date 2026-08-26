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
FlagGems routing consistency (full coverage)

Every op that ``backends_flaggems.conf`` routes to ``flagos_python`` must have a
real ``Backend::kFlagOsPython`` kernel generated in the C++ layer. Generated
kernels intentionally retained for explicit per-op overrides are also allowed
when the code generator lists them in ``flaggems_recursive_fallback``. This
guards the whole FlagGems Python surface against drift between the runtime
config and the codegen output.

This is a pure text/parse check: it reads the shipped config and generated
sources, so it needs no GPU, no ``flag_gems`` install, and runs in
milliseconds on any platform.

The op-name -> kernel bridge is:

    conf ``op = flagos_python``
      -> register.inc  ``m.impl("op", WrapperFoo);``
      -> WrapperFoo body ``... foo_dispatcher(...)``
      -> flaggems_python_kernels.cc
         ``REGISTER_IMPL_TO_DISPATCHER(_, foo_dispatcher, Backend::kFlagOsPython, _)``

so we compare the set of ``*_dispatcher`` names on each side.

Usage:
    pytest tests/integration/ops/test_flaggems_conf_consistency.py -v
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


# ``special_i1.out`` is deliberately routed to CUDA in the generated config;
# torchgen still emits the shared dispatcher wrapper, but no Python kernel.
_SKIP_ROUTE_SET = {"special_i1_out_dispatcher"}


# tests/integration/ops/<this file> -> repo root is three levels up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONF = _REPO_ROOT / "torch_fl" / "configs" / "backends_flaggems.conf"
_REGISTER_INC = _REPO_ROOT / "csrc" / "aten" / "generated" / "register.inc"
_KERNELS_CC = _REPO_ROOT / "csrc" / "aten" / "generated" / "flaggems_python_kernels.cc"
_CODEGEN = _REPO_ROOT / "scripts" / "codegen_ops.py"

# The generated C++ sources this check parses exist only in a repo checkout; they
# are never shipped in a wheel. Pipelines that stage a wheel-only workspace (the
# Enflame GCU pipeline copies just `tests/` and `pyproject.toml`) therefore have
# nothing to compare, so the module skips rather than reporting drift it cannot
# see. `scripts/` and `csrc/` are the markers: if the checkout is present, every
# individual file below must be too, and a missing one still fails.
if not (_REPO_ROOT / "scripts").is_dir() or not (_REPO_ROOT / "csrc").is_dir():
    pytest.skip(
        "FlagGems conf/codegen consistency needs a repo checkout; this workspace "
        f"has no scripts/ or csrc/ under {_REPO_ROOT}",
        allow_module_level=True,
    )


def _read(path: Path) -> str:
    assert path.is_file(), f"expected generated/config file is missing: {path}"
    return path.read_text()


def _conf_flagos_python_ops() -> set[str]:
    """Op names that backends_flaggems.conf routes to the flagos_python slot."""
    ops: set[str] = set()
    for raw in _read(_CONF).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        op, backend = (part.strip() for part in line.split("=", 1))
        if backend == "flagos_python":
            ops.add(op)
    return ops


def _codegen_op_set(name: str) -> set[str]:
    """Read a ``name = {"op", ...}`` set literal out of the codegen source."""
    module = ast.parse(_read(_CODEGEN))
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        assert isinstance(value, set) and all(isinstance(op, str) for op in value)
        return value
    raise AssertionError(f"{name} is missing from codegen_ops.py")


def _override_only_ops() -> set[str]:
    """Ops with generated Python kernels retained only for explicit overrides.

    Two sets in codegen_ops.py force a route back to the cuda boxing kernel:

    ``flaggems_recursive_fallback`` -- gems re-enters ``torch.<op>`` for non-cuda
    device types, which on a PrivateUse1 (flagos) tensor lands back on the same
    flagos_python kernel and recurses forever.

    ``flaggems_runtime_broken`` -- the gems call raises (required keyword-only
    ``out``, a hard ``device.type == "cuda"`` assert, a DTK triton compile
    failure) or returns wrong numerics on the flagos device.

    In both cases the kernel stays generated and reachable via
    ``FLAGOS_OP_<op>=flagos_python``, so it is a legitimate orphan on the
    kernels side.
    """
    return _codegen_op_set("flaggems_recursive_fallback") | _codegen_op_set(
        "flaggems_runtime_broken"
    )


def _op_to_wrapper() -> dict[str, str]:
    """``m.impl("op", WrapperFoo);`` -> {op: WrapperFoo} from register.inc."""
    return dict(re.findall(r'm\.impl\("([^"]+)",\s*(\w+)\);', _read(_REGISTER_INC)))


def _wrapper_to_dispatcher() -> dict[str, str]:
    """WrapperFoo(...) { ... foo_dispatcher(...) } -> {WrapperFoo: foo_dispatcher}.

    The wrapper name is anchored to ``Wrapper`` because the file's own header
    comment (``// ... m.impl() lines.``) otherwise matches ``(\\w+)\\([^;{]*\\)``
    and consumes the first real wrapper definition along with it.
    """
    return dict(
        re.findall(
            r"\b(Wrapper\w*)\([^;{]*\)\s*\{\s*(?:return\s+)?"
            r"(?:at::native::flagos::)?(\w+_dispatcher)\(",
            _read(_REGISTER_INC),
        )
    )


def _cc_flagos_python_dispatchers() -> set[str]:
    """Dispatcher names registered with Backend::kFlagOsPython in the kernels cc."""
    return set(
        re.findall(
            r"REGISTER_IMPL_TO_DISPATCHER\(\s*\w+\s*,\s*(\w+)\s*,"
            r"\s*Backend::kFlagOsPython",
            _read(_KERNELS_CC),
        )
    )


def _ops_to_dispatchers(ops: set[str]) -> tuple[set[str], list[str]]:
    """Map op names to their dispatcher names via register.inc.

    Returns (dispatcher_names, unmapped_ops).
    """
    op2wrap = _op_to_wrapper()
    wrap2disp = _wrapper_to_dispatcher()
    dispatchers: set[str] = set()
    unmapped: list[str] = []
    for op in ops:
        wrapper = op2wrap.get(op)
        dispatcher = wrap2disp.get(wrapper) if wrapper else None
        if dispatcher:
            dispatchers.add(dispatcher)
        else:
            unmapped.append(op)
    return dispatchers, sorted(unmapped)


def _conf_dispatchers() -> tuple[set[str], list[str]]:
    """Map configured flagos_python ops to dispatcher names."""
    return _ops_to_dispatchers(_conf_flagos_python_ops())


def _override_only_dispatchers() -> tuple[set[str], list[str]]:
    """Map explicit-override-only ops to dispatcher names."""
    return _ops_to_dispatchers(_override_only_ops())


class TestFlagGemsConfConsistency:
    """backends_flaggems.conf <-> generated kFlagOsPython kernels must agree."""

    @pytest.mark.anyplatform
    def test_conf_has_flagos_python_ops(self):
        """Sanity: the conf actually routes a meaningful number of ops here."""
        ops = _conf_flagos_python_ops()
        assert len(ops) > 100, (
            f"expected many flagos_python ops in {_CONF.name}, got {len(ops)}"
        )

    @pytest.mark.anyplatform
    def test_every_conf_op_maps_to_a_dispatcher(self):
        """Every flagos_python op resolves through register.inc to a dispatcher."""
        _, unmapped = _conf_dispatchers()
        assert not unmapped, (
            "conf routes these ops to flagos_python but register.inc has no "
            f"m.impl/dispatcher for them: {unmapped}"
        )

    @pytest.mark.anyplatform
    def test_conf_ops_have_flagos_python_kernels(self):
        """Each flagos_python op must have a real kFlagOsPython C++ kernel."""
        conf_disp, _ = _conf_dispatchers()
        cc_disp = _cc_flagos_python_dispatchers()
        missing = sorted(conf_disp - cc_disp)
        assert not missing, (
            "these ops are routed to flagos_python in "
            f"{_CONF.name} but have NO kFlagOsPython kernel in "
            f"{_KERNELS_CC.name} (conf/codegen drift): {missing}"
        )

    @pytest.mark.anyplatform
    def test_override_only_ops_are_not_default_routes(self):
        """Recursive fallbacks keep kernels but must not use them by default."""
        overlap = sorted(_override_only_ops() & _conf_flagos_python_ops())
        assert not overlap, (
            f"override-only ops are still routed to flagos_python by default: {overlap}"
        )

    @pytest.mark.anyplatform
    def test_no_orphan_flagos_python_kernels(self):
        """Every kernel is configured by default or retained for an override."""
        conf_disp, _ = _conf_dispatchers()
        override_disp, unmapped = _override_only_dispatchers()
        assert not unmapped, (
            f"override-only ops have no generated dispatcher: {unmapped}"
        )
        cc_disp = _cc_flagos_python_dispatchers()
        orphans = sorted(cc_disp - conf_disp - override_disp)
        assert not orphans, (
            f"these kFlagOsPython kernels in {_KERNELS_CC.name} are not routed "
            f"by {_CONF.name} and are not listed as override-only kernels: {orphans}"
        )

    @pytest.mark.anyplatform
    def test_counts_match(self):
        """Configured plus override-only routes match generated kernels."""
        conf_disp, _ = _conf_dispatchers()
        override_disp, _ = _override_only_dispatchers()
        cc_disp = _cc_flagos_python_dispatchers()
        expected = conf_disp | override_disp
        expected -= _SKIP_ROUTE_SET
        assert len(expected) == len(cc_disp), (
            f"flagos_python route count mismatch: conf={len(conf_disp)} "
            f"override_only={len(override_disp)} kernels={len(cc_disp)}"
        )

    def test_python_kernels_use_canonical_modules(self):
        """Generated calls must not freeze a vendor architecture alias."""
        paths = re.findall(
            r'CallPythonOp_\w+\("([^"]+)"',
            _read(_KERNELS_CC),
        )
        assert paths, f"no Python operation calls found in {_KERNELS_CC.name}"
        noncanonical = sorted(
            path for path in paths if not path.startswith("flag_gems.ops.")
        )
        assert not noncanonical, (
            "generated FlagGems calls must use flag_gems.ops.* rather than a "
            f"vendor-specific module path: {noncanonical}"
        )
