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

"""The platform capability matrix must match the tree it documents.

docs/reference/platform-capability-matrix.md is the authoritative answer to
"what does each platform actually support?" -- the three per-platform trees have
asymmetric membership (PPU Python-only, no TsingMicro Python compat, soft_lowp
aten-only) that is otherwise only discoverable by listing directories. This
fails when the documented directory inventory, conf route counts or CI coverage
drift from reality, so the page cannot go stale silently.

Pure text + os.listdir: no torch needed.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "reference" / "platform-capability-matrix.md"


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _dirs(relative: str) -> list[str]:
    base = REPO_ROOT / relative
    return sorted(
        p.name
        for p in base.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"
    )


def _inventory(relative: str) -> list[str]:
    """The directory list the doc gives for one tree's inventory line."""
    pattern = re.compile(rf"^- `{re.escape(relative)}`: (.+)$", re.M)
    match = pattern.search(_doc())
    assert match, f"{relative} has no inventory line"
    return sorted(name.strip() for name in match.group(1).split(","))


def test_document_exists():
    assert DOC.is_file()


def test_runtime_inventory_matches_the_tree():
    assert _inventory("csrc/runtime/accelerator/") == _dirs("csrc/runtime/accelerator")


def test_aten_inventory_matches_the_tree():
    assert _inventory("csrc/aten/backends/") == _dirs("csrc/aten/backends")


def test_python_compat_inventory_matches_the_tree():
    assert _inventory("torch_fl/accelerator/") == _dirs("torch_fl/accelerator")


def _matrix_rows() -> dict[str, list[str]]:
    """platform -> table cells, from the markdown matrix table (header skipped)."""
    rows: dict[str, list[str]] = {}
    for line in _doc().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        platform = cells[0].strip("`")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", platform):
            continue  # header row (FLAGOS_ACCELERATOR)
        rows[platform] = cells
    return rows


def _conf_route_count(platform: str) -> int:
    conf = REPO_ROOT / "torch_fl" / "configs" / f"backends_{platform}.conf"
    count = 0
    for line in conf.read_text(encoding="utf-8").splitlines():
        body = line.split("#", 1)[0]
        if "=" in body and body.split("=", 1)[0].strip():
            count += 1
    return count


def test_matrix_covers_every_accelerator():
    """One row per conf, and one conf per row."""
    confs = {
        p.stem.removeprefix("backends_")
        for p in (REPO_ROOT / "torch_fl" / "configs").glob("backends_*.conf")
    }
    assert set(_matrix_rows()) == confs


def test_documented_conf_route_counts_match_the_confs():
    for platform, cells in _matrix_rows().items():
        documented = int(cells[5])
        assert documented == _conf_route_count(platform), platform


def test_documented_ci_coverage_matches_the_configs():
    for platform, cells in _matrix_rows().items():
        documented = cells[6].lower() == "yes"
        actual = (REPO_ROOT / ".github" / "configs" / f"{platform}.yml").is_file()
        assert documented == actual, platform
