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

"""Exercise the startup contract without importing the native extension."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "torch_fl" / "__init__.py"
HOOKS = (
    "apex",
    "ddp",
    "parallel_comm",
    "dataparallel",
    "data_parallel",
    "compile",
    "flex_attention",
    "bpu",
)
INSTALLERS = (
    "_install_apex_compat",
    "_patch_ddp_for_flagos",
    "_patch_comm_for_flagos",
    "_patch_dataparallel_for_flagos",
    "_patch_data_parallel_for_flagos",
    "_register_compile_backend",
    "_install_flex_attention_compat",
    "_register_bpu_compile_backend",
)


def _tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _lifecycle(installers):
    """Load only the public lifecycle functions with controllable hook stubs."""
    selected = [
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in ("optional_integration_status", "activate_optional_integrations")
    ]
    namespace = dict(zip(INSTALLERS, installers))
    messages = []
    namespace.update(
        _OPTIONAL_INTEGRATIONS=HOOKS,
        _optional_integration_state={},
        _env=SimpleNamespace(warn=messages.append),
        _build_accelerator=lambda: "cuda",
        messages=messages,
    )
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace


def test_explicit_activation_is_ordered_and_idempotent():
    calls = []
    lifecycle = _lifecycle([lambda name=name: calls.append(name) for name in HOOKS])
    activate = lifecycle["activate_optional_integrations"]

    assert set(activate("data_parallel", "flex_attention")) == set(HOOKS)
    assert calls == [
        "parallel_comm",
        "dataparallel",
        "data_parallel",
        "compile",
        "flex_attention",
    ]
    activate("data_parallel", "compile", "flex_attention")
    assert len(calls) == 5
    assert lifecycle["optional_integration_status"]()["apex"] == "inactive"
    assert lifecycle["optional_integration_status"]()["compile"] == "active"


def test_optional_failure_does_not_install_dependent_or_block_unrelated_hooks():
    calls = []

    def fail_comm():
        calls.append("parallel_comm")
        raise RuntimeError("bad vendor comm")

    installers = [lambda name=name: calls.append(name) for name in HOOKS]
    installers[2] = fail_comm
    lifecycle = _lifecycle(installers)
    status = lifecycle["activate_optional_integrations"](
        "data_parallel", "compile", strict=False
    )

    assert calls == ["parallel_comm", "compile"]
    assert status["parallel_comm"].startswith("failed: RuntimeError:")
    assert status["dataparallel"] == "inactive"
    assert status["data_parallel"].startswith("failed: prerequisite")
    assert status["compile"] == "active"
    assert len(lifecycle["messages"]) == 2
    with pytest.raises(RuntimeError, match="previously failed"):
        lifecycle["activate_optional_integrations"]("parallel_comm")
    assert calls == ["parallel_comm", "compile"]


def test_unknown_hook_is_rejected_without_changes():
    lifecycle = _lifecycle([lambda: None] * len(HOOKS))
    with pytest.raises(ValueError, match="Unknown torch_fl integration"):
        lifecycle["activate_optional_integrations"]("compiel")
    assert set(lifecycle["optional_integration_status"]().values()) == {"inactive"}


def test_missing_optional_package_is_not_reported_as_active():
    installers = [lambda: None] * len(HOOKS)
    installers[0] = lambda: False  # Apex is not installed.
    lifecycle = _lifecycle(installers)
    status = lifecycle["activate_optional_integrations"]("apex", strict=False)
    assert status["apex"] == "inactive"
    with pytest.raises(RuntimeError, match="unavailable"):
        lifecycle["activate_optional_integrations"]("apex")


def test_profiles_keep_mandatory_registration_outside_optional_gate():
    optional = next(
        ast.literal_eval(node.value)
        for node in _tree().body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_OPTIONAL_INTEGRATIONS"
            for target in node.targets
        )
    )
    assert not {"flagtree", "flaggems", "flag_gems", "flagcx", "distributed"} & set(
        optional
    )

    funcs = {
        node.name: node for node in _tree().body if isinstance(node, ast.FunctionDef)
    }
    ecosystem = funcs["_phase_ecosystem"]
    calls = [
        (node.lineno, ast.unparse(node.func))
        for node in ast.walk(ecosystem)
        if isinstance(node, ast.Call)
    ]
    mandatory = {
        "_register_flaggems_operators",
        "_register_distributed_backend",
    }
    for name in mandatory:
        assert sum(callee == name for _, callee in calls) == 1
    optional_line = next(
        line for line, name in calls if name == "activate_optional_integrations"
    )
    assert all(line < optional_line for line, name in calls if name in mandatory)
    gate = next(
        node
        for node in ecosystem.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "_STARTUP_PROFILE == 'full'"
    )
    assert gate.lineno < optional_line < gate.end_lineno


def test_profile_selection_is_part_of_the_first_phase():
    conf = next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "_phase_conf"
    )
    assert any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "_env.choice"
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "FLAGOS_STARTUP_PROFILE"
        for node in ast.walk(conf)
    )
