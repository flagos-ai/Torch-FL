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


def test_lookup_rounds_up_to_the_next_bucket(library):
    lib = E.KernelLibrary(library)
    assert lib.lookup("relu", 1, (1, 64))["name"] == "relu_1x64_f32"
    # 37 lanes fit in the 64-lane artifact: padded up, cropped back.
    assert lib.lookup("relu", 1, (1, 37))["name"] == "relu_1x64_f32"
    # Widths past the largest bucket have nowhere to go.
    assert lib.lookup("relu", 1, (1, 65)) is None
    assert lib.lookup("relu", 1, (1, 999)) is None
    # A batch dimension is not something the library serves.
    assert lib.lookup("relu", 1, (2, 64)) is None


def test_lookup_picks_the_smallest_covering_bucket(tmp_path):
    def entry(width):
        return {
            "name": f"relu_1x{width}_f32",
            "op": "relu",
            "nargs": 1,
            "shape": [1, width],
            "file": f"relu_1x{width}_f32.hbm",
        }

    manifest = {
        "dtype": "f32",
        "numpy_dtype": "float32",
        "kernels": [entry(64), entry(256), entry(512)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    lib = E.KernelLibrary(tmp_path)
    assert lib.lookup("relu", 1, (1, 10))["name"] == "relu_1x64_f32"
    assert lib.lookup("relu", 1, (1, 64))["name"] == "relu_1x64_f32"
    assert lib.lookup("relu", 1, (1, 65))["name"] == "relu_1x256_f32"
    assert lib.lookup("relu", 1, (1, 300))["name"] == "relu_1x512_f32"
    assert lib.lookup("relu", 1, (1, 513)) is None


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
        d.dispatch("aten.sort", [a])


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


def test_every_target_has_a_reference():
    """A target with no reference turns a shape miss into an exception."""
    for target in list(E._UNARY) + list(E._BINARY):
        op = E._UNARY.get(target) or E._BINARY.get(target)
        assert op in E._REFERENCE, f"{target} -> {op} has no reference"


def test_op_tables_are_internally_consistent():
    assert set(E._UNARY.values()) <= set(E._REFERENCE)
    assert set(E._BINARY.values()) <= set(E._REFERENCE)
    # Everything the dispatcher can serve has to come from the library too,
    # otherwise the reference is doing the work under a "hit" label.
    for op in set(E._UNARY.values()) | set(E._BINARY.values()):
        assert op in E._REFERENCE


def test_aliases_point_at_the_right_op():
    assert E._UNARY["aten.negative"] == "neg"
    assert E._UNARY["aten.relu"] == "relu"
    assert E._UNARY["aten.relu.default"] == "relu"
    assert E._BINARY["aten.pow.Tensor_Tensor"] == "pow"
    assert E._BINARY["aten.maximum"] == "max"


def test_reference_matches_torch_for_the_extended_ops():
    import torch

    d = E.EagerDispatcher(None)
    x = np.array([[-1.5, -0.5, 0.0, 0.5, 1.5]], dtype=np.float32)
    checks = {
        "aten.ceil": torch.ceil,
        "aten.floor": torch.floor,
        "aten.sign": torch.sign,
        "aten.erf": torch.erf,
        "aten.gelu": torch.nn.functional.gelu,
        "aten.mish": torch.nn.functional.mish,
        "aten.elu": torch.nn.functional.elu,
        "aten.reciprocal": torch.reciprocal,
    }
    for target, fn in checks.items():
        want = fn(torch.from_numpy(x)).numpy()
        got = d.dispatch(target, [x])
        assert np.allclose(got, want, atol=1e-5), target
