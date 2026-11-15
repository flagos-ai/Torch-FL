# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""A low aggregate error must not hide a failing logit or an incomplete reference."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "benchmarks/host_compiler"
SPEC = importlib.util.spec_from_file_location(
    "logits_validation_test", DIRECTORY / "logits_validation.py"
)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


def test_absolute_and_relative_tolerances_apply_per_element():
    expected = torch.tensor([0.0, 1000.0])
    actual = torch.tensor([0.0009, 1001.0])
    result = validation.tensor_accuracy(actual, expected)
    assert result["passed"] and result["violations"] == 0
    assert result["max_scaled_error"] <= 1


def test_one_bad_logit_cannot_hide_in_a_small_relative_l2():
    expected = torch.ones(10000)
    actual = expected.clone()
    actual[42] += 0.004
    assert (actual - expected).norm() / expected.norm() < 1e-3
    result = validation.tensor_accuracy(actual, expected)
    assert not result["passed"] and result["violations"] == 1
    summary = validation.summarize_accuracy([result], 1, atol=1e-3, rtol=1e-3)
    assert not summary["passed"] and summary["failed_steps"] == [0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_fail_even_when_both_sides_match(value):
    result = validation.tensor_accuracy(torch.tensor([value]), torch.tensor([value]))
    assert not result["passed"] and result["nonfinite"] == 1
    json.dumps(result, allow_nan=False)


def test_bf16_checker_does_not_round_away_a_difference():
    result = validation.tensor_accuracy(
        torch.tensor([1.0078125], dtype=torch.bfloat16),
        torch.tensor([1.0], dtype=torch.bfloat16),
    )
    assert result["violations"] == 1


def test_identical_values_do_not_hide_a_precision_change():
    with pytest.raises(ValueError, match="dtype"):
        validation.tensor_accuracy(torch.ones(2), torch.ones(2, dtype=torch.bfloat16))


@pytest.mark.parametrize("shape", [(3,), (1, 1, 3)])
def test_broadcastable_shapes_are_not_a_matching_reference(shape):
    with pytest.raises(ValueError, match="shapes"):
        validation.tensor_accuracy(torch.zeros(1, 3), torch.zeros(shape))


def test_empty_logits_cannot_pass():
    with pytest.raises(ValueError, match="nonempty"):
        validation.tensor_accuracy(torch.empty(0), torch.empty(0))


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("name", ["atol", "rtol"])
def test_invalid_tolerances_are_rejected(name, value):
    with pytest.raises(ValueError, match="tolerances"):
        validation.tensor_accuracy(torch.zeros(1), torch.zeros(1), **{name: value})


def test_zero_tolerance_remains_exact_and_json_serializable():
    same = validation.tensor_accuracy(torch.zeros(1), torch.zeros(1), atol=0, rtol=0)
    different = validation.tensor_accuracy(
        torch.ones(1), torch.zeros(1), atol=0, rtol=0
    )
    assert same["passed"] and same["max_scaled_error"] == 0
    assert not different["passed"] and different["violations"] == 1
    json.dumps(different, allow_nan=False)


def reference():
    return {
        "inputs": [torch.tensor([[5]]), torch.tensor([[7]])],
        "logits": [torch.zeros(1, 8) for _ in range(3)],
        "prefill_logits": torch.zeros(1, 3, 8),
        "prompt_ids": torch.tensor([[1, 2, 3]]),
        "contract": {
            "checkpoint_sha256": {"model.safetensors": "digest"},
            "dtype": "float32",
            "math_settings": {"float32_matmul_precision": "high"},
        },
        "producer": {
            "backend": "vendor",
            "torch": "2.10.0+cu130",
            "math_settings_after": {"float32_matmul_precision": "high"},
        },
    }


@pytest.mark.parametrize("field", ["inputs", "logits"])
def test_truncated_references_are_rejected_before_execution(field):
    data = reference()
    data[field].pop()
    with pytest.raises(ValueError, match="reference"):
        validation.validate_reference(data, 2)


@pytest.mark.parametrize(
    "change",
    ["missing", "dtype", "checkpoint", "prompt", "backend", "math", "prefill"],
)
def test_acceptance_rejects_an_incompatible_reference(change):
    data = reference()
    expected = reference()
    if change == "missing":
        del data["contract"]
    elif change == "dtype":
        data["contract"]["dtype"] = "bfloat16"
    elif change == "checkpoint":
        data["contract"]["checkpoint_sha256"]["model.safetensors"] = "different"
    elif change == "backend":
        data["producer"]["backend"] = "flaggems"
    elif change == "math":
        data["producer"]["math_settings_after"]["float32_matmul_precision"] = "highest"
    elif change == "prefill":
        del data["prefill_logits"]
    else:
        data["prompt_ids"][0, 0] = 9
    with pytest.raises(ValueError, match="reference"):
        validation.validate_reference(
            data, 2, contract=expected["contract"], prompt_ids=expected["prompt_ids"]
        )


def test_matching_reference_and_explicit_shorter_traversal_are_valid():
    data = reference()
    validation.validate_reference(
        data, 1, contract=data["contract"], prompt_ids=data["prompt_ids"]
    )


def test_summary_requires_every_requested_step():
    row = validation.tensor_accuracy(torch.ones(1), torch.ones(1))
    with pytest.raises(ValueError, match="comparison count"):
        validation.summarize_accuracy([row], 2, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("accuracy_failure", [False, True])
def test_strict_wrapper_keeps_failure_and_only_audits_a_fresh_report(
    tmp_path, accuracy_failure
):
    for name in ("qwen_strict_eager.py", "logits_validation.py"):
        shutil.copy(DIRECTORY / name, tmp_path / name)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"stale": True}))
    if accuracy_failure:
        code = f"""import json
from pathlib import Path
from logits_validation import LogitsValidationFailed
Path({str(report)!r}).write_text(json.dumps({{"execution": "eager", "host_program_stats": [0, 0, 0]}}))
raise LogitsValidationFailed("deliberately failing logits")
"""
    else:
        code = "raise SystemExit(1)\n"
    (tmp_path / "qwen_eager.py").write_text(code)
    result = subprocess.run(
        [
            sys.executable,
            str(tmp_path / "qwen_strict_eager.py"),
            "--output",
            str(report),
        ],
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stderr
    payload = json.loads(report.read_text())
    if accuracy_failure:
        assert payload["model_capture_audit"]["calls"] == 0
        assert "deliberately failing logits" in result.stderr
    else:
        assert payload == {"stale": True}
