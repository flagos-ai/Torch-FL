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
    with pytest.raises(ValueError, match="unmeasured native-host vendor"):
        policy.precision_policy("mthreads")


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
