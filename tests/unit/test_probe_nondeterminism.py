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

"""Unit coverage for tests/manual/probe_nondeterminism.py's classifier.

The probe itself needs an accelerator and a failing workload, so it cannot run
here. Its *verdict* logic can, and it is the part a wrong answer would send
someone down the wrong fix for: "unwritten output" and "stale read" want a
kernel fix, "arithmetic" wants an atomics/config fix, and telling them apart
is the whole point of the tool.

Loaded by path: the file lives under tests/manual/, which is not a package, and
importing it must not drag in torch.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "manual" / "probe_nondeterminism.py"


def _load():
    spec = importlib.util.spec_from_file_location("probe_nondeterminism", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load()


@pytest.mark.parametrize(
    "deterministic, nan_fraction, expected",
    [
        # Repeats agree: nothing to explain, whatever the poison arm said.
        (True, 0.0, "NO_REPEAT_VARIATION"),
        (True, 0.05, "NO_REPEAT_VARIATION"),
        # Poison survives to the output: a kernel did not write those elements.
        (False, 0.037, "UNWRITTEN_OUTPUT"),
        (False, 1.0, "UNWRITTEN_OUTPUT"),
        # Nothing unwritten, yet the result moves: memory that was read, not
        # written, is the variable.
        (False, 0.0, "STALE_OR_ALIEN_READ"),
        # No poisoned arm: memory content cannot be ruled in or out.
        (False, None, "REPEATS_DIFFER_NONDETERMINISTIC_ARITHMETIC"),
    ],
)
def test_classify(deterministic, nan_fraction, expected):
    assert probe.classify(deterministic, nan_fraction) == expected


def test_every_verdict_has_a_recommendation():
    verdicts = {
        "NO_REPEAT_VARIATION",
        "UNWRITTEN_OUTPUT",
        "STALE_OR_ALIEN_READ",
        "REPEATS_DIFFER_NONDETERMINISTIC_ARITHMETIC",
    }
    assert verdicts <= set(probe.RECOMMENDATIONS)
    assert all(probe.RECOMMENDATIONS[v] for v in verdicts)


def test_summarise_reports_the_evidence_it_used():
    arms = {
        "repeat": {"bit_identical": False},
        "poisoned": {"nan_fraction": 0.25},
        "unpoisoned": {"nan_fraction": 0.0},
    }
    out = probe.summarise(arms)
    assert out["verdict"] == "UNWRITTEN_OUTPUT"
    # The numbers behind the verdict travel with it, so a report can be checked.
    assert out["evidence"]["poisoned"]["nan_fraction"] == 0.25


def test_summarise_without_a_repeat_arm_is_inconclusive():
    assert probe.summarise({})["verdict"] == "INCONCLUSIVE"


def test_self_test_passes():
    assert probe.run_self_test() == 0
