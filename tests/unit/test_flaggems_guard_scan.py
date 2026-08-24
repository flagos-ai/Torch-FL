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

"""Device-guard scan tests for the flag_gems routing codegen."""

import types

import pytest

from scripts.codegen_ops import _scan_flaggems_cuda_guards


@pytest.fixture
def fake_gems(tmp_path):
    """A fake flag_gems package with one guarded and one clean op."""
    ops = tmp_path / "ops"
    ops.mkdir()
    (ops / "guarded_op.py").write_text(
        "def guarded_op(x):\n"
        '    assert x.is_cuda, "must be on cuda device"\n'
        "    return x\n"
    )
    (ops / "dynamic_guard.py").write_text(
        "def dynamic_guard(x):\n"
        "    if x.device.type != flag_gems.device:\n"
        '        raise ValueError("wrong device")\n'
        "    return x\n"
    )
    (ops / "clean_op.py").write_text("def clean_op(x):\n    return x\n")
    pkg = types.SimpleNamespace(__file__=str(tmp_path / "__init__.py"))
    return pkg, ops


def _routes(*names):
    return {
        n: (f"flag_gems.ops.{n}.{n.split('.')[0]}", "functional_pure", [])
        for n in names
    }


def test_guarded_op_flagged_when_not_exempt(fake_gems, capsys):
    pkg, _ = fake_gems
    flagged = _scan_flaggems_cuda_guards(_routes("guarded_op"), set(), pkg)
    assert flagged == {"guarded_op": 'assert x.is_cuda, "must be on cuda device"'}
    assert "WARNING" in capsys.readouterr().err


def test_guarded_op_not_flagged_when_exempt(fake_gems, capsys):
    pkg, _ = fake_gems
    flagged = _scan_flaggems_cuda_guards(_routes("guarded_op"), {"guarded_op"}, pkg)
    assert flagged == {"guarded_op": 'assert x.is_cuda, "must be on cuda device"'}
    assert "WARNING" not in capsys.readouterr().err


def test_dynamic_device_guard_not_flagged(fake_gems, capsys):
    pkg, _ = fake_gems
    flagged = _scan_flaggems_cuda_guards(_routes("dynamic_guard"), set(), pkg)
    assert flagged == {}
    assert "WARNING" not in capsys.readouterr().err


def test_clean_op_not_flagged(fake_gems, capsys):
    pkg, _ = fake_gems
    flagged = _scan_flaggems_cuda_guards(_routes("clean_op"), set(), pkg)
    assert flagged == {}
    assert "WARNING" not in capsys.readouterr().err


def test_comment_guard_not_flagged(fake_gems, capsys):
    pkg, ops = fake_gems
    (ops / "commented.py").write_text(
        "# assert x.is_cuda  # historical note\ndef commented(x):\n    return x\n"
    )
    flagged = _scan_flaggems_cuda_guards(_routes("commented"), set(), pkg)
    assert flagged == {}


def test_assert_not_is_cuda_not_flagged(fake_gems, capsys):
    pkg, ops = fake_gems
    (ops / "neg_guard.py").write_text(
        "def neg_guard(x):\n"
        "    assert not x.is_cuda, 'expected non-cuda'\n"
        "    return x\n"
    )
    flagged = _scan_flaggems_cuda_guards(_routes("neg_guard"), set(), pkg)
    assert flagged == {}
    assert "WARNING" not in capsys.readouterr().err
