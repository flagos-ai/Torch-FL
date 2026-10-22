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

"""Unit coverage for scripts/codegen/gen_cupti_runtime_cbid.py (#186).

Every ``CUPTI_ACTIVITY_KIND_RUNTIME`` record is named through
``cuptiRuntimeCbidToName`` in ``csrc/profiler/cupti_shim.h``. That function used
to hold a hand-written switch of ~20 callback ids with
``default: return "cudaRuntime"``, so on CUDA 180 of 203 runtime events in a
plain matmul workload, and on PPU 72 of 117, were exported under one
indistinguishable label while the raw ``cbid`` sat unused in the event metadata.

The table is now generated from CUPTI's own ``cupti_runtime_cbid.h``, and these
tests pin the properties that make that safe to trust:

* the two committed artifacts agree (the same contract the Codegen checks job
  runs through ``--check``, restated here so it fails in seconds on a machine
  with no CUDA toolkit);
* a stale or hand-edited ``.inc`` is rejected rather than silently used;
* name normalisation collapses ``_ptsz``/``_vNNNNN`` variants without eating the
  ``_v2`` in an API whose name really contains it;
* the ten callback ids the issue reported as unresolved resolve to the names the
  issue lists, which is the regression itself rather than a proxy for it;
* the shim has no hand-written id table left, and MetaX's separate lookup -- a
  different id namespace, resolved through ``mcptiActivityGetApiName`` on any
  current MCPTI build -- is still intact.

The parsing helpers are exercised against synthetic headers written to tmp_path
rather than the local toolkit, so the vendor header remains optional for both
this suite and CI.

Run: pytest tests/unit/test_cupti_runtime_cbid_table.py
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "codegen" / "gen_cupti_runtime_cbid.py"
GENERATED = REPO_ROOT / "csrc" / "profiler" / "generated"
TABLE_TXT = GENERATED / "cupti_runtime_cbid.txt"
TABLE_INC = GENERATED / "cupti_runtime_cbid_names.inc"
SHIM = REPO_ROOT / "csrc" / "profiler" / "cupti_shim.h"


def _load():
    """Import the generator by path -- scripts/ is not an importable package."""
    spec = importlib.util.spec_from_file_location("gen_cupti_runtime_cbid", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g = _load()


def _committed():
    """Return the committed (entries, provenance) pair."""
    return g.parse_txt(TABLE_TXT.read_text())


# ---------------------------------------------------------------------------
# The committed artifacts
# ---------------------------------------------------------------------------


def test_committed_inc_matches_committed_txt():
    """The C++ the shim includes is exactly the render of the committed table."""
    entries, provenance = _committed()
    rendered = g.render_inc(entries, provenance)
    assert rendered == TABLE_INC.read_text(), (
        "csrc/profiler/generated/cupti_runtime_cbid_names.inc does not match "
        "cupti_runtime_cbid.txt; run: python3 scripts/codegen/gen_cupti_runtime_cbid.py"
    )


def test_check_mode_passes():
    """--check is the gate the Codegen checks job runs, and it must be green."""
    sys.argv = ["gen_cupti_runtime_cbid.py", "--check"]
    g.main()  # raises SystemExit on a stale .inc


def test_check_mode_rejects_a_stale_inc(tmp_path, monkeypatch):
    """A hand-edited .inc must fail --check instead of being used."""
    table = tmp_path / TABLE_TXT.name
    table.write_text(TABLE_TXT.read_text())
    stale = tmp_path / TABLE_INC.name
    stale.write_text("inline constexpr const char* const kNames[] = {};\n")

    monkeypatch.setattr(g, "REPO", tmp_path)
    monkeypatch.setattr(g, "OUT_TXT", table)
    monkeypatch.setattr(g, "OUT_INC", stale)
    monkeypatch.setattr(sys, "argv", ["gen_cupti_runtime_cbid.py", "--check"])
    with pytest.raises(SystemExit, match="does not match"):
        g.main()


def test_only_id_zero_is_unnamed():
    """Id 0 is CUPTI's INVALID sentinel; every other id names a real API."""
    entries, _ = _committed()
    assert 0 not in entries, "cbid 0 is the INVALID sentinel, not an API"
    assert entries, "the committed table is empty"

    inc = TABLE_INC.read_text()
    assert f"inline constexpr size_t kNameCount = {max(entries) + 1};" in inc
    assert inc.count("nullptr") == 1, "only the INVALID sentinel may be null"


def test_table_has_no_gaps():
    """Ids are contiguous, which is why the table can be indexed by cbid.

    A gap would still index correctly -- it renders as an unnamed id -- but it
    would mean the parse dropped an entry, and pinning contiguity is what makes
    "id is unnamed" mean "CUPTI does not declare an API here" instead of "the
    parser missed a line".
    """
    entries, _ = _committed()
    assert sorted(entries) == list(range(1, max(entries) + 1))


# ---------------------------------------------------------------------------
# Parsing rules
# ---------------------------------------------------------------------------

SYNTHETIC_HEADER = """\
  CUPTI_RUNTIME_TRACE_CBID_INVALID                                       = 0,
  CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020                         = 1,
  CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_ptsz_v7000                    = 2,
  CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v2_v11030            = 3,
  CUPTI_RUNTIME_TRACE_CBID_SIZE                                          = 4,
  CUPTI_RUNTIME_TRACE_CBID_FORCE_INT                                     = 0x7fffffff
"""


def test_parse_normalises_only_the_shape_suffixes(tmp_path):
    """_ptsz and _vNNNNN are dropped; a real _v2 in the name is kept."""
    header = tmp_path / "cupti_runtime_cbid.h"
    header.write_text(SYNTHETIC_HEADER)
    entries, _ = g.parse_header(header)

    # The per-thread-default-stream variant is the same entry point.
    assert entries[1] == entries[2] == "cudaMemcpyAsync"
    # cudaStreamGetCaptureInfo_v2 is the API's real name: _v2 is not a version
    # suffix here, and stripping it would name an API that does not exist.
    assert entries[3] == "cudaStreamGetCaptureInfo_v2"


def test_parse_skips_the_sentinels(tmp_path):
    """INVALID and SIZE are not API names, and FORCE_INT is not decimal."""
    header = tmp_path / "cupti_runtime_cbid.h"
    header.write_text(SYNTHETIC_HEADER)
    entries, _ = g.parse_header(header)
    assert sorted(entries) == [1, 2, 3]


def test_parse_rejects_a_duplicate_cbid(tmp_path):
    """Two names for one id would silently drop one of them."""
    header = tmp_path / "cupti_runtime_cbid.h"
    header.write_text(
        "  CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020     = 7,\n"
        "  CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020   = 7,\n"
    )
    with pytest.raises(SystemExit, match="twice"):
        g.parse_header(header)


def test_parse_rejects_a_header_with_no_entries(tmp_path):
    """An empty parse must fail loudly rather than emit an empty table."""
    header = tmp_path / "cupti_runtime_cbid.h"
    header.write_text("// renamed or moved in a newer toolkit\n")
    with pytest.raises(SystemExit, match="no CUPTI_RUNTIME_TRACE_CBID_ entries"):
        g.parse_header(header)


# ---------------------------------------------------------------------------
# The regression from #186
# ---------------------------------------------------------------------------

# The ten callback ids the issue measured as unresolved (17 x94, 200 x35,
# 134 x18, 10 x16, 251 x8, 210 x4, 406 x2, 53, 317, 15) with the CUPTI symbol
# each one denotes. Before the table was generated these all rendered as
# "cudaRuntime".
ISSUE_186_IDS = {
    17: "cudaGetDevice",
    200: "cudaDeviceGetAttribute",
    134: "cudaEventCreateWithFlags",
    10: "cudaGetLastError",
    251: "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    210: "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
    406: "cudaGetDriverEntryPoint",
    53: "cudaGetSymbolAddress",
    317: "cudaStreamIsCapturing",
    15: "cudaFuncGetAttributes",
}


def test_issue_186_ids_resolve_to_real_api_names():
    """Every id from the issue names its API instead of the fallback."""
    entries, _ = _committed()
    resolved = {cbid: entries.get(cbid) for cbid in ISSUE_186_IDS}
    assert resolved == ISSUE_186_IDS


def test_table_covers_the_ppu_workload_ids():
    """The ids PPU puts behind the fallback also resolve."""
    entries, _ = _committed()
    # Measured on PPU with the issue's reproducer: cudaGetDeviceProperties (x33),
    # cudaGetDeviceCount, cudaSetDevice, and the mempool calls that replaced
    # cudaMalloc/cudaFree on that runtime.
    for cbid, name in {
        440: "cudaGetDeviceProperties",
        4: "cudaGetDeviceProperties",
        3: "cudaGetDeviceCount",
        16: "cudaSetDevice",
        383: "cudaMemPoolCreate",
        378: "cudaMemPoolSetAttribute",
        391: "cudaMallocFromPoolAsync",
        375: "cudaFreeAsync",
    }.items():
        assert entries.get(cbid) == name, f"cbid {cbid} is not named"


# ---------------------------------------------------------------------------
# The shim
# ---------------------------------------------------------------------------


def _nvidia_arm() -> str:
    """The part of cupti_shim.h compiled for non-MetaX builds."""
    text = SHIM.read_text()
    start = text.index("#if !defined(FLAGOS_METAX_MCPTI)")
    end = text.index("#endif", start)
    return text[start:end]


def test_shim_has_no_hand_written_id_table():
    """The lookup must stay a table lookup, not drift back into a switch."""
    arm = _nvidia_arm()
    assert 'include "generated/cupti_runtime_cbid_names.inc"' in arm
    assert "flagos_cupti_runtime_cbid::kNames" in arm
    assert "case " not in arm, "the hand-written cbid switch is back in cupti_shim.h"


def test_metax_lookup_is_untouched():
    """MetaX ids are a different namespace and keep their own resolver.

    #186 scopes the generated table to NVIDIA: MCPTI resolves runtime names
    through mcptiActivityGetApiName, and mcptiRuntimeCbidToName is only the
    fallback for MCPTI builds that do not export it. Applying the NVIDIA table
    there would label MetaX records with unrelated NVIDIA entry points.
    """
    text = SHIM.read_text()
    anchor = "inline const char* mcptiRuntimeCbidToName(uint32_t cbid)"
    name_at = text.index(anchor)
    start = text.rindex("#if defined(FLAGOS_METAX_MCPTI)", 0, name_at)
    arm = text[start : text.index("#endif", name_at)]
    assert anchor in arm
    assert 'default: return "mcRuntime";' in arm
    # A representative case, so the arm is a real lookup and not a stub.
    assert 'case 107: return "mcMalloc";' in arm
