# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate typed vendor adapters against the published ATen schema cohort."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def generator():
    pytest.importorskip("torchgen")
    spec = importlib.util.spec_from_file_location(
        "native_vendor_codegen_test", ROOT / "scripts/codegen/codegen_native_vendor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_entries_match_published_schemas(generator):
    first = generator.render(generator.load_policy())
    second = generator.render(generator.load_policy())
    assert first == second
    for path, content in first.items():
        assert path.read_text() == content, path


@pytest.mark.parametrize("field,value", [("vendor", "other"), ("schema_version", 2)])
def test_unknown_policy_is_rejected(generator, tmp_path, field, value):
    policy = generator.load_policy()
    policy[field] = value
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="unsupported vendor precision policy"):
        generator.load_policy(path)


@pytest.mark.parametrize("mutation", ["duplicate", "dtype", "family", "evidence"])
def test_unmeasured_policy_is_rejected(generator, tmp_path, mutation):
    policy = generator.load_policy()
    row = policy["operators"][0]
    if mutation == "duplicate":
        policy["operators"].append(copy.deepcopy(row))
    elif mutation == "dtype":
        row["dtypes"] = ["Half"]
    elif mutation == "family":
        row["family"] = "handwritten"
    else:
        row["reason"] = ""
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        generator.load_policy(path)


def test_unknown_schema_fails_before_emission(generator):
    policy = generator.load_policy()
    policy["operators"][0]["op"] = "missing_operator"
    with pytest.raises(ValueError, match="no structured functional schema"):
        generator.render(policy)


def test_policy_change_invalidates_all_entry_abis(generator):
    policy = generator.load_policy()
    original = json.loads(generator.render(policy)[generator.CATALOG_PATH])
    policy["operators"][0]["dtypes"] = ["BFloat16"]
    changed = json.loads(generator.render(policy)[generator.CATALOG_PATH])
    assert {row["abi"] for row in original}.isdisjoint(row["abi"] for row in changed)
