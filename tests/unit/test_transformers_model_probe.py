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

"""Unit coverage for the transformers model probe's reporting logic.

Pure logic over recorded layer events: no accelerator, no transformers, no
subprocess. What is covered here is the reasoning that decides what a hardware
run *means* — which failures count as coverage results, which are environment
noise, and which layers a fault may be blamed on. Those judgements are the part
that silently corrupts a support claim when wrong, and the part a hardware run
cannot check for itself.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tests" / "manual" / "transformers_model_probe.py"

_spec = importlib.util.spec_from_file_location("transformers_model_probe", SCRIPT)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

ALL_LAYERS = list(probe.LAYERS)


def _result(**layers):
    return {"layers": {name: {"layer": name, **rec} for name, rec in layers.items()}}


# --- verdicts -----------------------------------------------------------------


def test_all_layers_passing_is_a_pass():
    result = _result(
        config={"status": "PASS"},
        **{name: {"status": "PASS"} for name in ALL_LAYERS},
    )
    assert probe.verdict(result, ALL_LAYERS) == "PASS"


def test_unsupported_layer_does_not_defeat_a_pass():
    """A model without a differentiable output or generate() is not a failure.

    Counting a genuinely inapplicable layer as a defect would understate
    support for every architecture that simply has no such path.
    """
    result = _result(
        config={"status": "PASS"},
        transfer={"status": "PASS"},
        forward={"status": "PASS"},
        backward={"status": "UNSUPPORTED"},
        generate={"status": "UNSUPPORTED"},
    )
    assert probe.verdict(result, ALL_LAYERS) == "PASS"


def test_wrong_result_outranks_passing_layers():
    result = _result(
        config={"status": "PASS"},
        transfer={"status": "PASS"},
        forward={"status": "WRONG"},
        backward={"status": "NOT_REACHED"},
        generate={"status": "NOT_REACHED"},
    )
    assert probe.verdict(result, ALL_LAYERS) == "WRONG"


def test_crash_outranks_error():
    result = _result(
        config={"status": "PASS"},
        transfer={"status": "PASS"},
        forward={"status": "ERROR"},
        backward={"status": "CRASH"},
        generate={"status": "NOT_REACHED"},
    )
    assert probe.verdict(result, ALL_LAYERS) == "CRASH"


def test_environment_error_is_not_a_coverage_failure():
    """An unusable environment must not be recorded as a device defect."""
    result = _result(config={"status": "ENVIRONMENT_ERROR"})
    assert probe.verdict(result, ALL_LAYERS) == "ENVIRONMENT_ERROR"


def test_config_too_large_is_reported_verbatim():
    """The parameter cap is a safety boundary, not a measurement."""
    result = _result(config={"status": "CONFIG_TOO_LARGE", "params": 2_723_312_896})
    assert probe.verdict(result, ALL_LAYERS) == "CONFIG_TOO_LARGE"


def test_unknown_model_is_distinct_from_a_failure():
    result = _result(config={"status": "UNKNOWN_MODEL"})
    assert probe.verdict(result, ALL_LAYERS) == "UNKNOWN_MODEL"


# --- model classes ------------------------------------------------------------


def test_a_causal_lm_is_tried_first():
    """Generation is a layer, and only a causal-LM head has it."""
    assert probe.MODEL_CLASSES[0] == "AutoModelForCausalLM"


def test_an_encoder_is_reachable_without_a_causal_lm_mapping():
    """Issue #254: a causal-LM-only list reported every encoder UNSUPPORTED_TASK.

    The model has to be reachable through a task head or the bare body, or the
    probe never gets past the config layer for mpnet, bert and roberta.
    """
    assert {"AutoModelForMaskedLM", "AutoModel"} <= set(probe.MODEL_CLASSES)


def test_the_body_is_the_last_fallback():
    """A head should produce the measured model; the body only rescues it."""
    assert probe.MODEL_CLASSES[-1] == "AutoModel"


def test_every_class_the_child_names_exists_in_transformers():
    """The child resolves each name with getattr and silently skips a missing one.

    A typo would therefore remove a fallback without failing anything, which is
    how the causal-LM-only list stayed invisible. Checked against the installed
    transformers when there is one, and skipped where transformers is absent.
    """
    try:
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("transformers is not installed on this host")
    for name in probe.MODEL_CLASSES:
        assert hasattr(transformers, name), name


def test_the_child_receives_the_class_list():
    """The order is the parent's; the child must not carry a second copy."""
    assert '"model_classes": list(MODEL_CLASSES)' in _source()
    assert 'MODEL_CLASSES = request["model_classes"]' in probe.CHILD


def test_the_built_class_is_reported_with_the_config():
    """Which head measured the model changes what a PASS means, so it is recorded."""
    assert (
        'emit("config", "PASS", params=params, model_class=MODEL_CLASS)' in probe.CHILD
    )


def test_summary_names_the_class_that_was_measured():
    result = _result(
        config={
            "status": "PASS",
            "params": 24_672,
            "model_class": "AutoModelForMaskedLM",
        }
    )
    result.update(model="mpnet", device="flagos:0", dtype="float32")
    result.update(verdict="PASS", duration_s=1.0)
    assert "class      AutoModelForMaskedLM" in probe.summarize(result, [])


def _source() -> str:
    return SCRIPT.read_text()


def test_verdict_only_considers_selected_layers():
    """A layer that was not requested cannot decide the verdict."""
    result = _result(
        config={"status": "PASS"},
        transfer={"status": "PASS"},
        forward={"status": "PASS"},
    )
    assert probe.verdict(result, ["transfer", "forward"]) == "PASS"


# --- tolerances ---------------------------------------------------------------


def test_every_supported_dtype_has_a_tolerance():
    for dtype, tol in probe.TOLERANCES.items():
        assert tol["rtol"] > 0 and tol["atol"] > 0, dtype


def test_reduced_precision_is_looser_than_fp32():
    """Reduction order differs between CPU and accelerator kernels."""
    assert probe.TOLERANCES["float16"]["atol"] > probe.TOLERANCES["float32"]["atol"]
    assert probe.TOLERANCES["bfloat16"]["rtol"] > probe.TOLERANCES["float16"]["rtol"]


# --- poison detection --------------------------------------------------------


def test_poison_patterns_are_recognized():
    for text in (
        "RuntimeError: MUSA error: illegal memory access was encountered",
        "device-side assert triggered",
        "unspecified launch failure",
        "misaligned address",
    ):
        assert probe.POISON_RE.search(text), text


def test_ordinary_operator_gap_is_not_poison():
    """An unsupported operator leaves the context usable; it must not be
    treated as a fault that voids the layers behind it."""
    assert not probe.POISON_RE.search("RuntimeError: zero_ failed: NOT_SUPPORTED")


# --- atomic write ------------------------------------------------------------


def test_explicit_version_mismatch_is_fatal(monkeypatch):
    """Measuring a version other than the requested one is not comparable."""
    monkeypatch.setattr(probe, "installed_transformers", lambda: "5.16.1")
    with pytest.raises(SystemExit) as excinfo:
        probe.check_transformers_version("4.50.2", offline=True)
    assert "4.50.2" in str(excinfo.value)


def test_matching_explicit_version_is_recorded(monkeypatch):
    monkeypatch.setattr(probe, "installed_transformers", lambda: "4.50.2")
    record = probe.check_transformers_version("4.50.2", offline=True)
    assert record["installed"] == "4.50.2"


def test_missing_transformers_explains_the_no_deps_install(monkeypatch):
    monkeypatch.setattr(probe, "installed_transformers", lambda: None)
    with pytest.raises(SystemExit) as excinfo:
        probe.check_transformers_version("latest", offline=True)
    assert "--no-deps" in str(excinfo.value)


def test_latest_offline_does_not_query_pypi(monkeypatch):
    """A vendor box is often offline; that must not block a run."""
    monkeypatch.setattr(probe, "installed_transformers", lambda: "5.16.1")

    def fail():  # pragma: no cover - must not be called
        raise AssertionError("PyPI queried in offline mode")

    monkeypatch.setattr(probe, "latest_transformers", fail)
    record = probe.check_transformers_version("latest", offline=True)
    assert record["latest"] is None


def test_latest_records_the_resolved_release(monkeypatch):
    monkeypatch.setattr(probe, "installed_transformers", lambda: "5.16.1")
    monkeypatch.setattr(probe, "latest_transformers", lambda: "5.17.0")
    record = probe.check_transformers_version("latest", offline=False)
    assert record == {
        "requested": "latest",
        "installed": "5.16.1",
        "latest": "5.17.0",
    }


def test_unreachable_pypi_falls_back_to_the_installed_version(monkeypatch):
    monkeypatch.setattr(probe, "installed_transformers", lambda: "5.16.1")
    monkeypatch.setattr(probe, "latest_transformers", lambda: None)
    record = probe.check_transformers_version("latest", offline=False)
    assert record["installed"] == "5.16.1" and record["latest"] is None


def test_atomic_write_leaves_no_temporary_files(tmp_path):
    target = tmp_path / "nested" / "result.json"
    probe.atomic_write(target, {"verdict": "PASS"})
    assert target.exists()
    assert sorted(p.name for p in target.parent.iterdir()) == ["result.json"]
