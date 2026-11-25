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


@pytest.mark.parametrize("vendor", ["nvidia", "hygon", "metax"])
def test_wheel_builder_generates_catalog_without_source_json(
    generator, monkeypatch, vendor
):
    monkeypatch.syspath_prepend(str(ROOT / "torch_fl/native_host"))
    spec = importlib.util.spec_from_file_location(
        "vendor_builder_test", ROOT / "torch_fl/native_host/vendor_build.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    if vendor == "metax":
        spec = importlib.util.spec_from_file_location(
            "metax_vendor_codegen_test",
            ROOT / "scripts/codegen/codegen_metax_native_vendor.py",
        )
        metax = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metax)
        expected = metax.render()[1]
    else:
        expected = generator.render(generator.load_policy(vendor))[1]
    assert builder.generate_catalog(ROOT, vendor) == expected
    for name in (
        "source_variants.json",
        "vendor_precision.json",
        "vendor_catalog.json",
    ):
        assert not (ROOT / "torch_fl/native_host" / name).exists()


def test_catalog_emission_rejects_stale_core_sources(
    generator, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(generator, "SOURCE_DIR", tmp_path)
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    monkeypatch.setattr(
        generator.sys, "argv", ["codegen", "--check", "--print-catalog"]
    )
    with pytest.raises(SystemExit, match="stale native vendor entries"):
        generator.main()
    assert not capsys.readouterr().out


def test_hcu_entries_use_published_structured_implementations(generator):
    outputs, catalog = generator.render(generator.load_policy("hygon"))
    assert len(catalog) == 12
    for path, content in outputs.items():
        assert path.read_text() == content
    source = next(content for path, content in outputs.items() if path.suffix == ".cc")
    assert "#define FLAGOS_NATIVE_HYGON 1" in source
    assert "BindStreamScope(module)" in source
    assert "StructuredCall<at::native::structured_mm_out_cuda>" in source
    assert "StructuredCall<at::native::structured_bmm_out_cuda>" in source
    assert "Matrix<" not in source
    assert "DeviceBoxingGuard" not in source and "redispatch" not in source
    routes = (ROOT / "torch_fl/configs/backends_hygon.conf").read_text()
    availability = {
        "any",
        "any.dim",
        "any.dims",
        "nonzero",
        "sort",
        "sort.stable",
        "topk",
    }
    for row in catalog:
        assert row["vendor"] == "hygon"
        if row["op"] in availability:
            assert f"\n{row['op']} = hygon" in routes, row["op"]
        else:
            expected = ",".join(row["dtypes"]) + ":hygon"
            assert f"{row['op']} = flaggems; {expected}" in routes


def test_hcu_availability_entries_route_to_the_vendor_kernel(generator):
    outputs, catalog = generator.render(generator.load_policy("hygon"))
    source = next(content for path, content in outputs.items() if path.suffix == ".cc")
    routes = (ROOT / "torch_fl/configs/backends_hygon.conf").read_text()

    # The flag reductions call the named ATen stub on the vendor device: the
    # shared structured impl dispatches on the operand's own device type,
    # which no vendor kernel is registered under.
    assert "FlagReduce<at::meta::structured_any_dim>(" in source
    assert "FlagReduce<at::meta::structured_any_dims>(" in source
    assert "FlagReduce<at::meta::structured_any>(" in source
    assert "at::native::or_stub" in source
    # Index pairs return both tensors the stub fills.
    assert "PairOutput<at::meta::structured_sort_stable> kernel;" in source
    assert (
        "at::native::sort_stub(at::kCUDA, self, kernel.values, kernel.indices, dim, descending, false);"
        in source
    )
    assert (
        "topk_stub(at::kCUDA, self, kernel.values, kernel.indices, k, dim, largest, sorted);"
        in source
    )
    # Nonzero hands the vendor kernel an out it resizes itself.
    assert "NonzeroOut<at::native::nonzero_out_cuda>(self)" in source

    availability = {
        "any",
        "any.dim",
        "any.dims",
        "nonzero",
        "sort",
        "sort.stable",
        "topk",
    }
    for row in catalog:
        if row["op"] not in availability:
            continue
        assert row["family"] in {
            "flag_reduce_dim",
            "flag_reduce_dims",
            "flag_reduce_all",
            "index_pair",
            "nonzero_out",
        }
        assert f"\n{row['op']} = hygon" in routes, row["op"]
