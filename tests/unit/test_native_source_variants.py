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


@pytest.fixture
def policy():
    return cohort.load_build_policy(ROOT).source_policy("nvidia")


@pytest.fixture
def published(tmp_path, monkeypatch, policy):
    package = tmp_path / "flag_gems"
    package.mkdir()
    root = ModuleType("flag_gems")
    root.__file__ = str(package / "__init__.py")
    root.mul = object()
    monkeypatch.setitem(sys.modules, "flag_gems", root)
    row = policy["entries"][0]
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


def test_policy_is_opt_in_to_the_validated_vendor_build(policy, tmp_path):
    assert (
        cohort.source_policy_for_build(SimpleNamespace(NATIVE_VENDOR=False), tmp_path)
        is None
    )
    assert (
        cohort.source_policy_for_build(
            SimpleNamespace(NATIVE_VENDOR=True, ACCELERATOR="nvidia"), ROOT
        )
        == policy
    )
    with pytest.raises(ValueError, match="unmeasured native-host vendor"):
        cohort.source_policy_for_build(
            SimpleNamespace(NATIVE_VENDOR=True, ACCELERATOR="mthreads"), ROOT
        )


def test_load_keeps_public_operator_and_published_source_unchanged(policy, published):
    root, path = published
    public, source = root.mul, path.read_bytes()
    selected = variants.load(policy)
    assert selected["published_mul"](3, 7) == 21
    assert root.mul is public
    assert path.read_bytes() == source
    assert variants.load(policy)["published_mul"] is selected["published_mul"]


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


@pytest.mark.parametrize("change", ["duplicate", "dtype", "module", "hash", "reason"])
def test_unvalidated_selections_are_rejected(policy, change):
    row = policy["entries"][0]
    if change == "duplicate":
        policy["entries"].append(copy.deepcopy(row))
    elif change == "dtype":
        row["dtypes"] = ["Half"]
    elif change == "module":
        row["module"] = "flag_gems...outside"
    elif change == "hash":
        row["sha256"] = "unknown"
    else:
        row["reason"] = ""
    with pytest.raises(ValueError):
        variants.validate(policy)
