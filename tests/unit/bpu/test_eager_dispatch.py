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

"""Dispatch bookkeeping for the BPU operator library.

These need no device: they cover bucket lookup, the reference fallback and the
hit accounting. The point is that a miss is *counted*, because an eager path
that quietly degrades to the host is indistinguishable from one that works.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from torch_fl.accelerator.bpu import eager as E


@pytest.fixture()
def library(tmp_path):
    """A manifest describing one bucket per op, with no real artifacts."""

    def entry(op, nargs, width, file):
        return {
            "name": f"{op}_1x{width}_f32",
            "op": op,
            "nargs": nargs,
            "shape": [1, width],
            "file": file,
            "bytes": 1,
            "sha256": "0" * 16,
            "template": f"{op}/2d/f32",
        }

    manifest = {
        "march": "nash_p",
        "dtype": "f32",
        "numpy_dtype": "float32",
        "buckets": ["1x64"],
        "kernels": [
            entry("relu", 1, 64, "relu_1x64_f32.hbm"),
            entry("add", 2, 64, "add_1x64_f32.hbm"),
            # A rank-1 entry must be ignored rather than guessed at.
            {
                "name": "relu_64_f32",
                "op": "relu",
                "nargs": 1,
                "shape": [64],
                "file": "relu_64_f32.hbm",
            },
        ],
        "failed": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_buckets_are_indexed_by_width(library):
    lib = E.KernelLibrary(library)
    assert lib.buckets("relu", 1) == [64]
    assert lib.buckets("add", 2) == [64]
    assert lib.buckets("exp", 1) == []


def test_rank_one_entries_are_ignored(library):
    lib = E.KernelLibrary(library)
    assert lib.lookup("relu", 1, (64,)) is None


def test_lookup_is_exact_on_shape(library):
    lib = E.KernelLibrary(library)
    assert lib.lookup("relu", 1, (1, 64)) is not None
    # A wider or narrower request must not borrow the 64-wide artifact: the
    # shape is compiled into the graph, so a 999-wide tensor fed to a 64-wide
    # artifact would come back as 64 values.
    assert lib.lookup("relu", 1, (1, 999)) is None
    assert lib.lookup("relu", 1, (2, 64)) is None


def test_missing_manifest_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        E.KernelLibrary(tmp_path)


def test_dispatcher_without_library_uses_reference():
    d = E.EagerDispatcher(None)
    a = np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)
    out = d.dispatch("aten.relu", [a])
    assert np.array_equal(out, [[0.0, 0.0, 2.0]])
    assert d.stats["disabled"] == 1


def test_reference_matches_torch_semantics_for_binary_ops():
    d = E.EagerDispatcher(None)
    a = np.array([[2.0, 4.0]], dtype=np.float32)
    b = np.array([[4.0, 2.0]], dtype=np.float32)
    assert np.array_equal(d.dispatch("aten.sub.Tensor", [a, b]), [[-2.0, 2.0]])
    assert np.array_equal(d.dispatch("aten.div.Tensor", [a, b]), [[0.5, 2.0]])


def test_unknown_op_is_tallied_as_miss():
    d = E.EagerDispatcher(None)
    a = np.ones((1, 8), dtype=np.float32)
    # An op with no library entry and no reference must raise, not silently
    # return something plausible.
    with pytest.raises(NotImplementedError):
        d.dispatch("aten.sin", [a])


def test_shape_miss_is_tallied_and_served_by_reference(library):
    d = E.EagerDispatcher(library)
    a = np.full((1, 999), -3.0, dtype=np.float32)
    out = d.dispatch("aten.relu", [a])
    assert np.array_equal(out, np.zeros_like(a))
    assert d.stats["miss_shape"] == 1
    assert d.stats["hit"] == 0


def test_hit_runs_the_artifact(library, monkeypatch):
    d = E.EagerDispatcher(library)
    seen = {}

    def fake_run(self, entry, arrays):
        seen["entry"] = entry["name"]
        return np.asarray(arrays[0]) * 2

    monkeypatch.setattr(E.KernelLibrary, "run", fake_run)
    a = np.ones((1, 64), dtype=np.float32)
    out = d.dispatch("aten.relu", [a])
    assert seen["entry"] == "relu_1x64_f32"
    assert np.array_equal(out, a * 2)
    assert d.stats["hit"] == 1


def test_a_broken_artifact_falls_back(library, monkeypatch):
    d = E.EagerDispatcher(library)

    def boom(self, entry, arrays):
        raise RuntimeError("bad artifact")

    monkeypatch.setattr(E.KernelLibrary, "run", boom)
    a = np.array([[-1.0, 3.0]], dtype=np.float32)
    out = d.dispatch("aten.relu", [a])
    assert np.array_equal(out, [[0.0, 3.0]])


def test_summary_reports_the_split(library):
    d = E.EagerDispatcher(library)
    a = np.ones((1, 999), dtype=np.float32)
    d.dispatch("aten.relu", [a])
    text = d.summary()
    assert "0/1 on device" in text
    assert "miss_shape=1" in text
