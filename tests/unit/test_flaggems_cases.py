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

"""Reject source-table shapes that would make a shared inventory drift."""

import importlib.util
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "flaggems_cases", ROOT / ".github/scripts/flaggems_cases.py"
)
assert SPEC is not None and SPEC.loader is not None
CASES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CASES)


def _source(tmp_path, source):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "cases.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return CASES.Source((path,)), path


@pytest.mark.parametrize(
    ("expression", "message"),
    (
        ("tuple(" + "PLATFORMS.upper())", "unsupported table expression"),
        ("tuple(sorted(PLATFORMS.parents))", "unsupported table expression"),
        (
            'tuple("x" + platform for platform in PLATFORMS)',
            "unsupported table expression",
        ),
    ),
)
def test_case_generation_accepts_only_literal_tables(tmp_path, expression, message):
    source, path = _source(
        tmp_path,
        f"""
        PLATFORMS = ("cuda", "metax")
        CASES = {expression}
        """,
    )
    with pytest.raises(CASES.InventoryError, match=message):
        source.list(path, source.lookup(path, "CASES")[1])


def test_duplicate_case_tags_are_rejected(tmp_path):
    source, path = _source(
        tmp_path,
        """
        CASES = _case(("same", "same"), None)

        def _case(tag, value):
            return list(tag)
        """,
    )
    with pytest.raises(CASES.InventoryError, match="single tag"):
        source.list(path, source.lookup(path, "CASES")[1])

    source, path = _source(
        tmp_path / "helper",
        """
        CASES = _case("same", None)

        def _case(tag, value):
            return [tag, tag]
        """,
    )
    assert source.list(path, source.lookup(path, "CASES")[1]) == ["same"]

    with pytest.raises(CASES.InventoryError, match="repeats"):
        CASES._unique(["same", "same"], "test")


def test_explicit_ids_name_boolean_parametrization(tmp_path):
    source, path = _source(
        tmp_path,
        """
        class TestMode:
            @pytest.mark.parametrize(
                "training", [True, False], ids=["train", "eval"]
            )
            def test_mode_matches_cpu(self, training):
                pass
        """,
    )
    original_root = CASES.ROOT
    CASES.ROOT = tmp_path
    try:
        assert CASES._file_nodeids(source, path) == [
            "cases.py::TestMode::test_mode_matches_cpu[train]",
            "cases.py::TestMode::test_mode_matches_cpu[eval]",
        ]
    finally:
        CASES.ROOT = original_root


@pytest.fixture(autouse=True)
def _select_checkout_files(monkeypatch):
    monkeypatch.setattr(CASES, "ROOT", ROOT)


def test_both_flaggems_passes_share_one_ordered_inventory():
    cases = CASES.flaggems_cases()
    assert cases
    assert CASES.flaggems_kernel_cases() == cases
    assert any("device_operands_cases.py" in case for case in cases)
    assert any("pinned_cases.py" in case for case in cases)
    assert len(cases) == len(set(cases))


def test_vendor_sdpa_inventory_expands_runtime_row_cases():
    cases = CASES.vendor_sdpa_cases()
    assert cases
    assert len(cases) == len(set(cases))
    assert cases == sorted(cases)
    assert any("test_the_vendor_lane_holds[no_mask]" in case for case in cases)
    assert any("test_the_platform_names_the_lane_it_carries" in case for case in cases)
