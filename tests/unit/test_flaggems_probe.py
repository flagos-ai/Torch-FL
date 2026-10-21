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

"""Run the generated FlagGems routing child without an accelerator."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


INTEGRATION = Path(__file__).resolve().parents[1] / "integration"
FLAGGEMS = INTEGRATION / "flaggems"
for directory in (FLAGGEMS, INTEGRATION):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import flaggems_probe  # noqa: E402
from flaggems_caps import (  # noqa: E402
    FLAGGEMS_EXECUTION_CALLS,
    FLAGGEMS_ROUTING_CALLS,
)


def _execution_operand(tensor):
    """The public module's helper, without importing torch_fl in this process."""
    source = (FLAGGEMS / "test_flaggems_execution.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "requires_grad_for_backward"
    )
    namespace = {}
    module = ast.Module(body=[helper], type_ignores=[])
    exec(compile(module, "<helper>", "exec"), namespace)
    return namespace["requires_grad_for_backward"](tensor)


def test_generated_child_runs_every_shared_call(monkeypatch, capsys):
    """The child table and its order must reach the module-level loop."""
    calls = {
        "neg": "torch.neg(a)",
        "silu": "torch.nn.functional.silu(a)",
    }
    prelude = "a = torch.tensor([-1.0, 2.0])\n"
    monkeypatch.setattr(flaggems_probe, "FLAGGEMS_ROUTING_CALLS", calls)
    monkeypatch.setattr(flaggems_probe, "FLAGGEMS_EXECUTION_CALLS", {})
    monkeypatch.setattr(flaggems_probe, "PRELUDE", prelude)
    monkeypatch.setattr(flaggems_probe, "DEVICE", "cpu")
    source = flaggems_probe._child_source()
    source = source.replace(
        "import torch_fl  # noqa: F401  -- claiming the flagos device is the point\n",
        "",
    )
    namespace = {"torch": torch}
    exec(compile(source, "<generated probe>", "exec"), namespace)

    entries = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [entry["op"] for entry in entries] == ["neg", "silu"]
    assert all(entry["error"] is None for entry in entries)


def test_the_real_child_table_has_every_declared_case(monkeypatch):
    # Run the committed table under an isolated interpreter. Shared inputs such
    # as ``x`` must exist before the first call, and a call must leave the
    # variables every later call relies on untouched.
    prelude = "from flaggems_caps import PRELUDE\n" + flaggems_probe.PRELUDE
    source = flaggems_probe._child_source().replace(
        "import torch_fl  # noqa: F401  -- claiming the flagos device is the point\n",
        "",
    )
    source = source.replace(
        "DEVICE = 'flagos:0'\n",
        "DEVICE = 'cpu'\n",
    )
    source = source.replace(flaggems_probe.PRELUDE, prelude)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(FLAGGEMS), str(INTEGRATION), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    entries = [json.loads(line) for line in result.stdout.splitlines()]

    table = flaggems_probe._call_table()
    calls = {**FLAGGEMS_ROUTING_CALLS, **FLAGGEMS_EXECUTION_CALLS}
    assert table == {"calls": calls, "order": sorted(calls)}
    assert [entry["op"] for entry in entries] == sorted(calls)
    assert [entry["error"] for entry in entries] == [None] * len(calls)


def test_execution_operands_require_gradients_only_for_float_tensors(monkeypatch):
    indices = _execution_operand(torch.tensor([0, 2]))
    weights = _execution_operand(torch.randn(3, 4))
    assert indices.dtype == torch.int64
    assert not indices.requires_grad
    assert weights.requires_grad


def test_extracting_the_call_table_rejects_a_missing_prelude():
    """A source change must not make the child-table test pass vacuously."""
    with pytest.raises(ValueError):
        flaggems_probe._generated_table("CALLS = {}")


def test_the_call_table_is_what_the_child_reads(monkeypatch):
    monkeypatch.setattr(
        flaggems_probe, "FLAGGEMS_ROUTING_CALLS", {"neg": "torch.neg(a)"}
    )
    monkeypatch.setattr(flaggems_probe, "FLAGGEMS_EXECUTION_CALLS", {})
    source = flaggems_probe._child_source()
    assert 'CALLS["order"]' in source
    assert 'CALLS["calls"]' in source


def test_the_generated_child_reports_a_call_error_without_stopping(monkeypatch, capsys):
    calls = {"a": "raise ValueError('expected')", "b": "pass"}
    prelude = ""
    monkeypatch.setattr(flaggems_probe, "PRELUDE", prelude)
    monkeypatch.setattr(flaggems_probe, "DEVICE", "cpu")
    monkeypatch.setattr(
        flaggems_probe,
        "_call_table",
        lambda: {"calls": calls, "order": sorted(calls)},
    )
    source = flaggems_probe._child_source()
    source = source.replace(
        "import torch_fl  # noqa: F401  -- claiming the flagos device is the point\n",
        "",
    )
    namespace = {"torch": torch}
    exec(compile(source, "<generated probe>", "exec"), namespace)
    assert namespace["CALLS"] == {"calls": calls, "order": ["a", "b"]}
    entries = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert entries[0]["error"] == "ValueError: expected"
    assert entries[1]["error"] is None
