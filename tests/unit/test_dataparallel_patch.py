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

"""Device-independent coverage for the DataParallel/flagos patches.

The patches themselves need a flagos device (see
``tests/manual/test_dataparallel_live.py`` for the on-device run), so this file
pins the two parts that need no hardware:

  * the device-type helpers, which are what decide whether a module is routed to
    the flagos construction path at all -- a wrong answer here is silent, and
    sends a flagos module down the cuda-first path the patch exists to avoid;
  * the wiring: that the ecosystem phase installs all three patches, that the
    comm patch rebinds every function DataParallel's comm layer calls, and that
    a module which is *not* on a flagos device still reaches the original
    ``__init__``.

Parsed and exec'd rather than imported: ``torch_fl/__init__.py`` imports torch
and loads ``torch_fl._C``, which needs a built wheel and hardware. Same approach
as ``tests/unit/test_import_phase_order.py``, for the same reason.
"""

import ast
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INIT = REPO_ROOT / "torch_fl" / "__init__.py"

_DEVICE_TYPES = "_FLAGOS_DEVICE_TYPES"
_HELPERS = ("_flagos_device_type_of_tensors", "_flagos_module_device_type")

#: Functions the comm patch must replace: the three CUDA-only C++ ops that
#: DataParallel's forward path reaches (scatter, then replicate ->
#: broadcast_coalesced, then gather). ``reduce_add_coalesced`` is deliberately
#: absent -- it is the backward path, and it already works for flagos tensors
#: because ``comm.reduce_add`` falls back to its device-agnostic Python
#: implementation whenever ``nccl.is_available()`` is False, which it is for
#: privateuseone tensors.
_COMM_FUNCTIONS = {"scatter", "gather", "broadcast_coalesced"}


class _Tensor:
    """Minimal stand-in: the helpers read only ``.device.type``."""

    def __init__(self, device_type, index=0):
        self.device = types.SimpleNamespace(type=device_type, index=index)


class _Module:
    def __init__(self, params=(), buffers=()):
        self._params = list(params)
        self._buffers = list(buffers)

    def parameters(self):
        return iter(self._params)

    def buffers(self):
        return iter(self._buffers)


def _tree():
    return ast.parse(INIT.read_text(encoding="utf-8"))


def _function(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(
        f"{name} is not defined at module level in torch_fl/__init__.py"
    )


def _helpers():
    """The device-type helpers, exec'd straight out of the module source."""
    tree = _tree()
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in _HELPERS)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == _DEVICE_TYPES
                for target in node.targets
            )
        )
    ]
    assert len(wanted) == len(_HELPERS) + 1, [ast.dump(n)[:40] for n in wanted]
    namespace = {}
    exec(
        compile(ast.Module(body=wanted, type_ignores=[]), str(INIT), "exec"), namespace
    )
    return namespace


def test_device_type_of_tensors_reads_the_first_flagos_tensor():
    helpers = _helpers()
    classify = helpers["_flagos_device_type_of_tensors"]
    assert (
        classify([_Tensor("cpu"), _Tensor("flagos", 3), _Tensor("flagos", 1)])
        == "flagos"
    )
    # The pre-rename spelling names the same device and must be accepted too.
    assert classify([_Tensor("privateuseone", 0)]) == "privateuseone"
    assert classify([_Tensor("cuda", 0), _Tensor("cpu")]) is None
    assert classify([]) is None


def test_module_device_type_reads_parameters_and_buffers():
    helpers = _helpers()
    classify = helpers["_flagos_module_device_type"]
    assert classify(_Module(params=[_Tensor("flagos", 0)])) == "flagos"
    # A module can carry its state in buffers only; DataParallel's own guard
    # walks parameters and buffers together.
    assert classify(_Module(buffers=[_Tensor("flagos", 2)])) == "flagos"
    assert classify(_Module(params=[_Tensor("cuda", 0)])) is None
    assert classify(_Module()) is None


def test_ecosystem_installs_the_dataparallel_patches():
    calls = [
        ast.unparse(node.value.func)
        for node in _function(_tree(), "_phase_ecosystem").body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    for name in (
        "_patch_comm_for_flagos",
        "_patch_dataparallel_for_flagos",
        "_patch_data_parallel_for_flagos",
    ):
        assert name in calls, f"{name} is not called by _phase_ecosystem: {calls}"
    # The comm branches have to exist before anything can construct a flagos
    # DataParallel, and both follow the DDP patch they mirror.
    assert calls.index("_patch_ddp_for_flagos") < calls.index("_patch_comm_for_flagos")
    assert calls.index("_patch_comm_for_flagos") < calls.index(
        "_patch_dataparallel_for_flagos"
    )


def test_comm_patch_rebinds_every_function_the_comm_layer_calls():
    rebound = {
        target.attr
        for node in ast.walk(_function(_tree(), "_patch_comm_for_flagos"))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "_comm"
    }
    assert rebound == _COMM_FUNCTIONS


def _nested(tree, outer, inner):
    for node in ast.walk(_function(tree, outer)):
        if isinstance(node, ast.FunctionDef) and node.name == inner:
            return node
    raise AssertionError(f"{outer} does not define {inner}")


def _called_names(function):
    return {
        ast.unparse(node.func).rsplit(".", 1)[-1]
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    }


def test_dataparallel_init_keeps_the_stock_path_for_non_flagos_modules():
    init = _nested(_tree(), "_patch_dataparallel_for_flagos", "_patched_init")
    calls = _called_names(init)
    # The fallback: a module whose tensors are not on a flagos device must be
    # handed to the original __init__ untouched. (Commented in the source too --
    # this pins the call, not the comment.)
    assert "_orig_init" in calls, sorted(calls)
    # And the flagos path must not run the CUDA-only balance probe, which is both
    # meaningless there and, on MetaX, a side effect on the current device.
    assert "_check_balance" not in calls, sorted(calls)
