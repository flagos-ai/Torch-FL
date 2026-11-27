# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate explicit MetaX SDK admission and immutable numerical provenance."""

import copy
import hashlib
import importlib.util
import runpy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    package = ModuleType("sdk_precision_fixture")
    package.__path__ = []
    cohort = ModuleType(package.__name__ + ".vendor_cohort")
    cohort.digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, cohort.__name__, cohort)
    spec = importlib.util.spec_from_file_location(
        package.__name__ + ".vendor", ROOT / "torch_fl/native_host/vendor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    library = tmp_path / "sdk.so"
    library.write_bytes(b"published SDK numerical implementation")
    catalog = {"policy": {"vendor": "metax"}, "policy_sha256": "validated"}
    libraries = {"mcblas": str(library)}
    versions = {"flag_gems": "5.4.0", "flagtree": "0.7.0.post2+metax3.6"}
    manifest = {
        **copy.deepcopy(catalog),
        **versions,
        "numerical_libraries": {
            "mcblas": {"path": str(library), "sha256": cohort.digest(library)}
        },
    }
    torch = ModuleType("torch")
    torch.ops = SimpleNamespace(
        flagos=SimpleNamespace(
            native_metax_sdk_catalog=lambda: json.dumps(catalog),
            native_metax_sdk_libraries=lambda: libraries,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(module.importlib.metadata, "version", versions.__getitem__)
    return module, manifest, catalog, libraries, versions, library


@pytest.mark.parametrize("mutation", ["policy", "binary", "library_set", "cohort"])
def test_sdk_activation_rejects_changed_numerical_contract(sdk, mutation):
    module, manifest, catalog, libraries, versions, library = sdk
    module._verify_sdk(manifest)
    if mutation == "policy":
        catalog["policy"]["vendor"] = "other"
    elif mutation == "binary":
        library.write_bytes(b"different implementation")
    elif mutation == "library_set":
        libraries["unknown"] = str(library)
    else:
        versions["flag_gems"] = "unmeasured"
    with pytest.raises(RuntimeError, match="native MetaX SDK"):
        module._verify_sdk(manifest)


def test_sdk_binary_identity_allows_installation_at_another_path(sdk, tmp_path):
    module, manifest, _, libraries, _, library = sdk
    relocated = tmp_path / "relocated.so"
    relocated.write_bytes(library.read_bytes())
    libraries["mcblas"] = str(relocated)
    module._verify_sdk(manifest)


@pytest.mark.parametrize("vendor", ["metax", "thead"])
def test_sdk_precision_is_explicit_and_native_only(vendor):
    generator = runpy.run_path(str(ROOT / "scripts/codegen/gen_vendor_confs.py"))
    config = ROOT / f"torch_fl/configs/backends_{vendor}.conf"
    native = generator["render_native_host_config"](
        vendor, generator["parse_conf"](config)
    )
    ordinary = config.read_text()
    for operation in ("mm", "bmm"):
        assert f"{operation} = flaggems; BFloat16:{vendor}\n" in native
        assert f"{operation} = flaggems; BFloat16:{vendor}\n" not in ordinary


@pytest.mark.parametrize(("vendor", "name"), [("metax", "MetaX"), ("thead", "T-Head")])
def test_generated_sdk_entries_match_the_validated_policy(vendor, name):
    pytest.importorskip("torchgen")
    spec = importlib.util.spec_from_file_location(
        "metax_sdk_generator",
        ROOT / "scripts/codegen/codegen_metax_native_precision.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = module.OUTPUT.with_name(f"{vendor}_native_precision_kernels.cc")
    assert (
        module.render(vendor=vendor)
        == module.render(vendor=vendor)
        == output.read_text()
    )
    bad = copy.deepcopy(module.NATIVE_SDK_PRECISION[vendor])
    bad["operators"][0]["dtypes"] = ["Float"]
    with pytest.raises(ValueError, match=f"unmeasured {name} SDK dtype"):
        module.render(bad, vendor=vendor)
