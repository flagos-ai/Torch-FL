# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Published source selection must preserve the pinned package and public API."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "source_variant_test", ROOT / "torch_fl/native_host/source_variants.py"
)
variants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variants)
spec = importlib.util.spec_from_file_location(
    "source_variant_cohort_test", ROOT / "torch_fl/native_host/vendor_cohort.py"
)
cohort = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohort)


@pytest.fixture(params=["nvidia", "metax"])
def policy(request):
    return cohort.load_build_policy(ROOT).source_policy(request.param)


@pytest.fixture
def published(tmp_path, monkeypatch, policy):
    package = tmp_path / "flag_gems"
    package.mkdir()
    root = ModuleType("flag_gems")
    root.__file__ = str(package / "__init__.py")
    root.mul = object()
    monkeypatch.setitem(sys.modules, "flag_gems", root)
    row = policy["entries"][0]
    policy["entries"] = [row]
    path = package.joinpath(*row["module"].split(".")[1:]).with_suffix(".py")
    path.parent.mkdir(parents=True)
    path.write_text("def mul(a, b):\n    return a * b\n")
    path.parent.joinpath("__init__.py").write_text(
        "raise AssertionError('unrelated backend import')\n"
    )
    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.delitem(sys.modules, row["module"], raising=False)
    monkeypatch.setattr(variants.importlib.metadata, "version", lambda _: "5.4.0")
    yield root, path
    sys.modules.pop(row["module"], None)


def test_policy_requires_the_selected_native_build(policy, tmp_path):
    assert (
        cohort.source_policy_for_build(SimpleNamespace(NATIVE_VENDOR=False), tmp_path)
        is None
    )
    assert cohort.source_policy_for_build(
        SimpleNamespace(
            NATIVE_HOSTS=True, NATIVE_VENDOR=False, ACCELERATOR=policy["vendor"]
        ),
        ROOT,
    ) == (None if policy["requires_native_vendor"] else policy)
    assert (
        cohort.source_policy_for_build(
            SimpleNamespace(NATIVE_VENDOR=True, ACCELERATOR=policy["vendor"]), ROOT
        )
        == policy
    )
    with pytest.raises(ValueError, match="unmeasured native-host vendor"):
        cohort.source_policy_for_build(
            SimpleNamespace(NATIVE_VENDOR=True, ACCELERATOR="apple"), ROOT
        )


@pytest.mark.parametrize(
    "vendor",
    ["ascend", "iluvatar", "metax", "mthreads", "thead", "tsingmicro"],
)
def test_sources_require_native_hosts_without_bundling_vendor_torch(vendor):
    policy = cohort.load_build_policy(ROOT).source_policy(vendor)
    assert policy["requires_native_vendor"] is False
    assert variants.validate(policy) == policy
    assert (
        cohort.source_policy_for_build(
            SimpleNamespace(NATIVE_HOSTS=True, NATIVE_VENDOR=False, ACCELERATOR=vendor),
            ROOT,
        )
        == policy
    )
    assert (
        cohort.source_policy_for_build(
            SimpleNamespace(
                NATIVE_HOSTS=False, NATIVE_VENDOR=False, ACCELERATOR=vendor
            ),
            ROOT,
        )
        is None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("same_tensor_metadata", 1),
        ("unit_stride_axis", 1),
        ("min_tensors", True),
        ("min_tensors", 0),
    ],
)
def test_source_metadata_constraints_are_typed(policy, field, value):
    policy["entries"][0][field] = value
    with pytest.raises(ValueError, match="invalid published source"):
        variants.validate(policy)


def test_load_keeps_public_operator_and_published_source_unchanged(policy, published):
    root, path = published
    public, source = root.mul, path.read_bytes()
    selected = variants.load(policy)
    assert selected["published_mul"](3, 7) == 21
    assert root.mul is public
    assert path.read_bytes() == source
    assert variants.load(policy)["published_mul"] is selected["published_mul"]


@pytest.mark.parametrize("vendor", ["nvidia", "mthreads", "thead"])
def test_shared_lerp_loads_scalar_functions_without_the_xpu_host(
    tmp_path, monkeypatch, vendor
):
    policy = cohort.load_build_policy(ROOT).source_policy(vendor)
    row = next(
        item for item in policy["entries"] if item["id"] == "published_bf16_lerp"
    )
    policy["entries"] = [row]
    package = tmp_path / "flag_gems"
    path = package.joinpath(*row["module"].split(".")[1:]).with_suffix(".py")
    path.parent.mkdir(parents=True)
    published = (
        "raise AssertionError('vendor host imported')\n"
        "@pointwise_dynamic\n"
        "def lerp_scalar_kernel_head(start, end, weight):\n"
        "    return start + weight * (end - start)\n"
        "def lerp_scalar_(start, end, weight):\n"
        "    return lerp_scalar_kernel_head(start, end, weight)\n"
    )
    path.write_text(published)
    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    root = ModuleType("flag_gems")
    root.__file__ = str(package / "__init__.py")
    triton = ModuleType("triton")
    triton.__path__ = []
    triton.language = ModuleType("triton.language")
    pointwise = ModuleType("flag_gems.utils.pointwise_dynamic")
    calls = []

    def bind(function):
        calls.append(function.__name__)
        return function

    pointwise.pointwise_dynamic = bind
    for module in (root, triton, triton.language, pointwise):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.delitem(sys.modules, row["module"], raising=False)
    monkeypatch.setattr(variants.importlib.metadata, "version", lambda _: "5.4.0")
    generated = tmp_path / "_generated"
    generated.mkdir()
    bytecode = variants.build_bytecode(policy, generated)

    def reject_runtime_compilation(*args, **kwargs):
        pytest.fail("The published host source must be precompiled")

    monkeypatch.setattr(variants, "shared_lerp_source", reject_runtime_compilation)
    try:
        with pytest.raises(RuntimeError, match="Missing prebuilt"):
            variants.load(policy, package=generated, bytecode={})
        loaded = variants.load(policy, package=generated, bytecode=bytecode)
        assert loaded[row["id"]](1.0, 3.0, 0.25) == 1.5
        assert loaded[row["id"]].__code__.co_filename == str(path)
        assert calls == ["lerp_scalar_kernel_head"]
        assert path.read_text() == published
        sys.modules.pop(row["module"])
        artifact = generated / variants.BYTECODE_DIRECTORY / f"{row['id']}.pyc"
        artifact.write_bytes(b"changed")
        with pytest.raises(RuntimeError, match="bytecode changed"):
            variants.load(policy, package=generated, bytecode=bytecode)
    finally:
        sys.modules.pop(row["module"], None)


@pytest.mark.parametrize("change", ["version", "source", "manifest", "module_path"])
def test_cohort_changes_fail_before_admission(policy, published, monkeypatch, change):
    _, path = published
    row = policy["entries"][0]
    sources = {row["module"]: {"sha256": row["sha256"]}}
    if change == "version":
        monkeypatch.setattr(variants.importlib.metadata, "version", lambda _: "5.5.0")
    elif change == "source":
        path.write_text(path.read_text() + "# changed\n")
    elif change == "manifest":
        sources[row["module"]]["sha256"] = "0" * 64
    else:
        module = ModuleType(row["module"])
        module.__file__ = str(path.with_name("another.py"))
        monkeypatch.setitem(sys.modules, row["module"], module)
    with pytest.raises(RuntimeError):
        variants.load(policy, sources)


@pytest.mark.parametrize("vendor", ["kunlunxin", "metax", "tsingmicro"])
def test_published_vendor_source_accepts_only_a_verified_registration_alias(
    tmp_path, vendor
):
    alias = f"_{vendor}.ops.argmax"
    module = f"flag_gems.runtime.backend.{alias}"
    source = tmp_path.joinpath(*module.split(".")[1:]).with_suffix(".py")
    source.parent.mkdir(parents=True)
    source.write_text("def argmax(): pass\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    row = {"module": module, "sha256": digest}
    assert variants._published_source(row, tmp_path, {alias: {"sha256": digest}}) == (
        source,
        source.read_bytes(),
    )
    with pytest.raises(RuntimeError, match="published source changed"):
        variants._published_source(row, tmp_path, {alias: {"sha256": "0" * 64}})
    with pytest.raises(RuntimeError, match="published source changed"):
        variants._published_source(
            row, tmp_path, {alias: {"sha256": digest}, module: {"sha256": "0" * 64}}
        )


@pytest.mark.parametrize(
    "change",
    ["duplicate", "dtype", "module", "hash", "reason", "wrapped_numbers", "adaptation"],
)
def test_unvalidated_selections_are_rejected(policy, change):
    row = policy["entries"][0]
    if change == "duplicate":
        policy["entries"].append(copy.deepcopy(row))
    elif change == "dtype":
        row["dtypes"] = ["Half"]
    elif change == "module":
        row["module"] = "flag_gems...outside"
    elif change == "wrapped_numbers":
        row["allow_wrapped_numbers"] = "yes"
    elif change == "hash":
        row["sha256"] = "unknown"
    elif change == "adaptation":
        row["adaptation"] = row["id"]
    else:
        row["reason"] = ""
    with pytest.raises(ValueError):
        variants.validate(policy)
