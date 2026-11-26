# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Published vendor assets must be identified without importing torch_fl."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "vendor_cohort_test", ROOT / "torch_fl/native_host/vendor_cohort.py"
)
cohort = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cohort)


@pytest.fixture
def published(tmp_path):
    policy = cohort.load_build_policy(ROOT).precision_policy("nvidia")
    library = tmp_path / "torch/lib"
    library.mkdir(parents=True)
    (library.parent / "version.py").write_text(
        f"__version__: str = {policy['vendor_torch']!r}\n"
        f"git_version = {policy['torch_git_version']!r}\ncuda = '13.0'\n"
    )
    for name in cohort.LIBRARIES:
        (library / name).write_bytes(name.encode())
    info = tmp_path / "torch-2.10.0+cu130.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Name: torch\nVersion: {policy['vendor_torch']}\n"
        'Requires-Dist: nvidia-cublas==13.1.0.3; platform_system == "Linux"\n'
        "Requires-Dist: packaging>=23\n"
    )
    (info / "licenses").mkdir()
    (info / "licenses/LICENSE").write_text("published license")
    return library, policy, info


def test_build_policy_views_share_cohort_and_do_not_mutate_registry():
    policy = cohort.load_build_policy(ROOT)
    precision = policy.precision_policy("nvidia")
    sources = policy.source_policy("nvidia")
    for key in ("schema_version", "vendor", "flag_gems"):
        assert precision[key] == sources[key]
    precision["operators"].clear()
    sources["entries"].clear()
    assert policy.precision_policy("nvidia")["operators"]
    assert policy.source_policy("nvidia")["entries"]
    assert not policy.precision_policy("mthreads")["operators"]
    assert policy.source_policy("mthreads")["entries"]
    with pytest.raises(ValueError, match="unmeasured native-host vendor"):
        policy.precision_policy("apple")


def test_bundle_uses_content_identity_and_preserves_license(published, tmp_path):
    library, policy, _ = published
    record = cohort.describe(library, policy)
    assert record["requirements"] == [
        'nvidia-cublas==13.1.0.3; platform_system == "Linux"'
    ]
    destination = tmp_path / "package"
    hashes = cohort.bundle(record, destination)
    # Same length is insufficient to accept a stale binary.
    target = destination / "lib/libc10_cuda.so"
    target.write_bytes(b"x" * target.stat().st_size)
    assert cohort.bundle(record, destination) == hashes
    assert target.read_bytes() == (library / target.name).read_bytes()
    assert (
        destination / "native_host/_generated/licenses/vendor_torch_LICENSE"
    ).read_text() == "published license"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("version", "validated published Torch cohort"),
        ("missing_library", "missing published vendor library"),
        ("wrong_metadata", "missing or ambiguous"),
        ("floating_requirement", "verified exact pins"),
        ("wildcard_requirement", "verified exact pins"),
        ("missing_license", "license files are missing"),
    ],
)
def test_cohort_rejects_unverified_assets(published, mutation, match):
    library, policy, info = published
    metadata = info / "METADATA"
    if mutation == "version":
        (library.parent / "version.py").write_text("__version__ = '2.10.1+cu130'\n")
    elif mutation == "missing_library":
        (library / cohort.LIBRARIES[0]).unlink()
    elif mutation == "wrong_metadata":
        metadata.write_text(metadata.read_text().replace("Name: torch", "Name: other"))
    elif mutation.endswith("requirement"):
        replacement = ">=13" if mutation == "floating_requirement" else "==13.*"
        metadata.write_text(metadata.read_text().replace("==13.1.0.3", replacement))
    else:
        (info / "licenses/LICENSE").unlink()
    with pytest.raises(ValueError, match=match):
        cohort.describe(library, policy)


def test_version_reader_does_not_execute_the_package(tmp_path):
    marker = tmp_path / "executed"
    source = tmp_path / "version.py"
    source.write_text(
        f"open({str(marker)!r}, 'w').write('bad')\n__version__ = '2.10.0+cu130'\n"
    )
    assert cohort.version_fields(source) == {"__version__": "2.10.0+cu130"}
    assert not marker.exists()


def test_hcu_cohort_checks_vendor_git_and_metadata_separately(tmp_path):
    policy = cohort.load_build_policy(ROOT).precision_policy("hygon")
    library = tmp_path / "torch/lib"
    library.mkdir(parents=True)
    version = library.parent / "version.py"
    version.write_text(
        f"__version__ = {policy['vendor_source_version']!r}\n"
        f"git_version = {policy['torch_git_version']!r}\n"
        f"hip = {policy['hip']!r}\ncuda = None\n"
    )
    for name in ("libtorch_hip.so", "libc10_hip.so"):
        (library / name).write_bytes(name.encode())
    info = tmp_path / "torch-2.10.0.dist-info"
    (info / "licenses").mkdir(parents=True)
    (info / "licenses/LICENSE").write_text("published HCU license")
    (info / "METADATA").write_text(f"Name: torch\nVersion: {policy['vendor_torch']}\n")
    record = cohort.describe(library, policy)
    assert record["vendor"] == "hygon" and record["cuda"] is None
    assert record["version"] == policy["vendor_torch"]
    assert record["git_version"] != policy["cpu_torch_git_version"]
    version.write_text(version.read_text().replace(policy["hip"], "unknown"))
    with pytest.raises(ValueError, match="validated published Torch cohort"):
        cohort.describe(library, policy)


def test_hcu_bundle_must_use_the_existing_dtk_builder(tmp_path):
    with pytest.raises(ValueError, match="existing DTK bundle builder"):
        cohort.bundle(
            {"vendor": "hygon", "library_dir": str(tmp_path)}, tmp_path / "package"
        )


@pytest.mark.parametrize("vendor", ["metax", "thead"])
def test_device_only_cohort_bundles_gpu_libraries_and_flat_licenses(tmp_path, vendor):
    policy = cohort.load_build_policy(ROOT).precision_policy(vendor)
    library = tmp_path / "torch/lib"
    library.mkdir(parents=True)
    (library.parent / "version.py").write_text(
        f"__version__ = {policy['vendor_torch']!r}\n"
        f"git_version = {policy['torch_git_version']!r}\n"
        f"cuda = {policy.get('cuda', '11.6')!r}\n"
    )
    expected_hashes = {}
    for name in cohort.CUDA_DEVICE_LIBRARIES:
        asset = library / name
        asset.write_text(name)
        expected_hashes[name] = cohort.digest(asset)
    if "libraries" in policy:
        assert policy["libraries"].keys() == expected_hashes.keys()
        policy["libraries"] = expected_hashes
    for name in policy["host_headers"]:
        header = library.parent / "include" / name
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text("published CUDA macros")
        policy["host_headers"][name] = cohort.digest(header)
    (library / "libtorch_cpu.so").write_text("vendor CPU core must remain outside")
    info = tmp_path / "torch-2.10.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Name: torch\nVersion: {policy['vendor_torch']}\n"
        "License-File: LICENSE\nLicense-File: NOTICE\n"
    )
    for name in ("LICENSE", "NOTICE"):
        (info / name).write_text(name)

    record = cohort.describe(library, policy)
    assert record["requirements"] == []
    assert record["git_version"] == policy["torch_git_version"]
    destination = tmp_path / "package"
    assert cohort.bundle(record, destination) == expected_hashes
    assert not (destination / f"lib_{vendor}").exists()
    root = destination / "native_host/_generated"
    assert {
        asset.name for asset in (root / f"{vendor}_vendor").iterdir()
    } == expected_hashes.keys()
    assert {asset.name for asset in (root / "licenses").iterdir()} == {
        "vendor_torch_LICENSE",
        "vendor_torch_NOTICE",
    }
    asset = library / "libtorch_cuda.so"
    if "libraries" in policy:
        asset.write_text("different published bits")
    else:
        asset.unlink()
    with pytest.raises(ValueError, match="published"):
        cohort.describe(library, policy)


def test_hcu_published_source_selection_is_explicit_and_dtype_scoped():
    from types import SimpleNamespace

    policy = cohort.source_policy_for_build(
        SimpleNamespace(NATIVE_VENDOR=True, ACCELERATOR="hygon"), ROOT
    )
    assert policy["vendor"] == "hygon" and policy["requires_native_vendor"]
    assert len(policy["entries"]) == 1
    row = policy["entries"][0]
    assert row["op"] == "aten::mul.Tensor"
    assert row["dtypes"] == ["Float", "BFloat16"]
    assert row["module"] == "flag_gems.runtime.backend._sunrise.ops.mul"
