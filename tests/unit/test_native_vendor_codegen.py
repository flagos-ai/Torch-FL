# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate typed vendor adapters against the published ATen schema cohort."""

import copy
import importlib.util
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
    for path, content in first[0].items():
        assert path.read_text() == content, path


def test_conf_dtype_routes_match_generated_vendor_capabilities(generator):
    conf = (ROOT / "torch_fl/configs/backends_nvidia.conf").read_text()
    clauses = {}
    for line in conf.splitlines():
        name, assignment, value = line.split("#", 1)[0].partition("=")
        if assignment and ";" in value:
            base, clause = value.split(";", 1)
            dtypes, backend = clause.strip().split(":")
            assert base.strip() == "flaggems"
            assert backend == "nvidia"
            clauses[name.strip()] = sorted(dtypes.split(","))
    catalog = generator.render(generator.load_policy())[1]
    assert clauses == {row["op"]: row["dtypes"] for row in catalog}


@pytest.mark.parametrize("field,value", [("vendor", "other"), ("schema_version", 2)])
def test_unknown_policy_is_rejected(generator, field, value):
    policy = generator.load_policy()
    policy[field] = value
    with pytest.raises(ValueError, match="unsupported vendor precision policy"):
        generator.validate_policy(policy)


@pytest.mark.parametrize("mutation", ["duplicate", "dtype", "family", "evidence"])
def test_unmeasured_policy_is_rejected(generator, mutation):
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
    with pytest.raises(ValueError):
        generator.validate_policy(policy)


def test_unknown_schema_fails_before_emission(generator):
    policy = generator.load_policy()
    policy["operators"][0]["op"] = "missing_operator"
    with pytest.raises(ValueError, match="no structured functional schema"):
        generator.render(policy)


def test_policy_change_invalidates_all_entry_abis(generator):
    policy = generator.load_policy()
    original = generator.render(policy)[1]
    policy["operators"][0]["dtypes"] = ["BFloat16"]
    changed = generator.render(policy)[1]
    assert {row["abi"] for row in original}.isdisjoint(row["abi"] for row in changed)


def test_wheel_builder_generates_catalog_without_source_json(generator, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "torch_fl/native_host"))
    spec = importlib.util.spec_from_file_location(
        "vendor_builder_test", ROOT / "torch_fl/native_host/vendor_build.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert (
        builder.generate_catalog(ROOT) == generator.render(generator.load_policy())[1]
    )
    for name in (
        "source_variants.json",
        "vendor_precision.json",
        "vendor_catalog.json",
    ):
        assert not (ROOT / "torch_fl/native_host" / name).exists()


def test_catalog_emission_rejects_stale_core_sources(
    generator, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(generator, "SOURCE_PATH", tmp_path / "stale.cc")
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    monkeypatch.setattr(
        generator.sys, "argv", ["codegen", "--check", "--print-catalog"]
    )
    with pytest.raises(SystemExit, match="stale native vendor entries"):
        generator.main()
    assert not capsys.readouterr().out
