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
  * the wiring: that optional activation installs all three patches, that the
    comm patch hands torch's comm primitives to the extension, and that the
    extension publishes exactly the primitives torch's comm layer calls -- a
    torch upgrade that reaches for a new one would otherwise escape the patch
    silently;
  * that the scatter scope a flagos ``DataParallel`` sets is restored on the way
    out, including when the scatter raises;
  * that a module which is *not* on a flagos device still reaches the original
    ``__init__``.

Parsed and exec'd rather than imported: ``torch_fl/__init__.py`` imports torch
and loads ``torch_fl._C``, which needs a built wheel and hardware. Same approach
as ``tests/unit/test_import_phase_order.py``, for the same reason.
"""

import ast
import re
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INIT = REPO_ROOT / "torch_fl" / "__init__.py"
COMM_CC = REPO_ROOT / "torch_fl" / "csrc" / "dataparallel_comm.cc"

_DEVICE_TYPES = "_FLAGOS_DEVICE_TYPES"
_HELPERS = ("_flagos_device_type_of_tensors", "_flagos_module_device_type")

#: The CUDA-only C++ ops on ``torch._C`` that DataParallel's comm layer reaches:
#: scatter, then replicate -> broadcast_coalesced, then gather, plus the ``_out``
#: forms the ``out=`` argument selects. They are replaced by the extension, not
#: by Python, and each replacement falls back to the original it replaced as
#: soon as no flagos tensor is involved.
#: ``reduce_add_coalesced`` is deliberately absent -- it is the backward path,
#: and it already works for flagos tensors because ``comm.reduce_add`` falls back
#: to its device-agnostic Python implementation whenever ``nccl.is_available()``
#: is False, which it is for privateuseone tensors.
_COMM_PRIMITIVES = {
    "_broadcast",
    "_broadcast_coalesced",
    "_broadcast_out",
    "_gather",
    "_gather_out",
    "_scatter",
    "_scatter_out",
}


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


def test_optional_activation_installs_the_dataparallel_patches():
    activation = _function(_tree(), "activate_optional_integrations")
    installers = next(
        node
        for node in ast.walk(activation)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "installers"
            for target in node.targets
        )
    )
    mapped = {
        key.value: ast.unparse(value)
        for key, value in zip(installers.value.keys, installers.value.values)
    }
    assert mapped["parallel_comm"] == "_patch_comm_for_flagos"
    assert mapped["dataparallel"] == "_patch_dataparallel_for_flagos"
    assert mapped["data_parallel"] == "_patch_data_parallel_for_flagos"

    prerequisites = next(
        node
        for node in ast.walk(activation)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "prerequisites"
            for target in node.targets
        )
    )
    dependencies = ast.literal_eval(prerequisites.value)
    assert dependencies["dataparallel"] == ("parallel_comm",)
    assert dependencies["data_parallel"] == ("parallel_comm", "dataparallel")


def test_comm_patch_hands_the_primitives_to_the_extension():
    patch = _function(_tree(), "_patch_comm_for_flagos")
    calls = _called_names(patch)
    # The implementations live in the extension -- because they have to read and
    # write flagos tensors with C++ tensor ops, and because rebinding
    # ``torch._C``'s attributes is a step Python cannot take without shadowing
    # the module for everyone, including torch's own lazily-imported callers.
    assert "_init_dataparallel_comm" in calls, sorted(calls)
    # And the Python wrappers that used to stand in for it are gone: a leftover
    # ``_comm.scatter = ...`` would be a second, drifting copy of the same rules.
    rebound = {
        target.attr
        for node in ast.walk(patch)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "_comm"
    }
    assert rebound == set(), sorted(rebound)


def test_comm_patch_scatter_restores_the_scope_it_sets():
    scatter = _nested(_tree(), "_patch_dataparallel_for_flagos", "_patched_scatter")
    calls = [
        node
        for node in ast.walk(scatter)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).endswith("_set_scatter_scope")
    ]
    # Set once, restored once.
    assert len(calls) == 2, ast.unparse(scatter)
    # And restored in a `finally`: the scope is a thread-local living in the
    # extension, and a scatter that raises must not leave it set for whatever
    # this thread scatters next.
    finals = [
        node
        for node in ast.walk(scatter)
        if isinstance(node, ast.Try) and node.finalbody
    ]
    assert finals, ast.unparse(scatter)
    assert any(
        isinstance(inner, ast.Call)
        and ast.unparse(inner.func).endswith("_set_scatter_scope")
        for node in finals
        for stmt in node.finalbody
        for inner in ast.walk(stmt)
    ), ast.unparse(scatter)


def _published_primitives():
    """The ``torch._C`` names the extension's rebinding publishes."""
    source = COMM_CC.read_text(encoding="utf-8")
    return set(re.findall(r'py::setattr\(\s*m\s*,\s*"(_[A-Za-z0-9_]+)"', source))


def _torch_comm_primitives():
    """The ``torch._C`` names torch's own comm layer reaches for.

    Scraped from the installed source rather than pinned to a list here: the
    point of the check is that our rebinding tracks whatever torch ships, so a
    primitive added upstream fails this test instead of escaping the patch.
    """
    import torch

    comm_py = Path(torch.__file__).resolve().parent / "nn" / "parallel" / "comm.py"
    assert comm_py.is_file(), comm_py
    return set(
        re.findall(r"torch\._C\.(_[A-Za-z0-9_]+)", comm_py.read_text(encoding="utf-8"))
    )


def test_extension_publishes_every_primitive_torch_comm_calls():
    called = _torch_comm_primitives()
    # If this half fails, torch grew (or lost) a primitive and
    # _COMM_PRIMITIVES -- and the rebinding -- have to follow.
    assert called == _COMM_PRIMITIVES, sorted(called)
    published = _published_primitives()
    assert published == called, sorted(published ^ called)


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
