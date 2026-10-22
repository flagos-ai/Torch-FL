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

"""The sliding-window KV layout an LLM .hbm expects.

These are pure arithmetic checks on `KVWindow.mask_range`, so they need no
device. They exist because the layout is not documented anywhere and getting it
wrong does not raise -- the model produces fluent, wrong text. The expected
values below are transcribed from an `LD_PRELOAD` trace of the vendor's own
`llm` demo on Qwen3-0.6B, so they are what the runtime actually passed to
`hbDNNInferV2`, not a reading of the graph.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from torch_fl.accelerator.bpu.infer import KVWindow  # noqa: E402

WINDOW = 4096
CHUNK = 512


class _FakeWindow:
    """`KVWindow`'s geometry without its allocations."""

    def __init__(self, window=WINDOW, pos=0):
        self.window = window
        self.pos = pos

    mask_range = KVWindow.mask_range


def test_prefill_first_rows_match_the_vendor_trace():
    """Traced: row r of a fresh 512-token prefill opens [3584, 3584+r+1)."""
    w = _FakeWindow()
    assert w.mask_range(CHUNK, 0) == (3584, 3585)
    assert w.mask_range(CHUNK, 1) == (3584, 3586)
    assert w.mask_range(CHUNK, 2) == (3584, 3587)


def test_decode_at_position_24_matches_the_vendor_trace():
    """Traced: with 24 tokens of history, decode opens [4071, 4096).

    Note the width is 25, not 24 -- the current token's own key is the last
    column, because the graph attends over concat(window[span:], new_keys).
    """
    w = _FakeWindow(pos=24)
    lo, hi = w.mask_range(1)
    assert (lo, hi) == (4071, 4096)
    assert hi - lo == 25


@pytest.mark.parametrize("pos", [0, 1, 25, 100, 3583])
def test_decode_window_is_history_plus_self(pos):
    w = _FakeWindow(pos=pos)
    lo, hi = w.mask_range(1)
    assert hi == WINDOW
    assert hi - lo == pos + 1


def test_window_saturates_rather_than_going_negative():
    """Past a full window the range clamps at 0 instead of wrapping."""
    w = _FakeWindow(pos=WINDOW * 2)
    lo, hi = w.mask_range(1)
    assert lo == 0
    assert hi == WINDOW


def test_the_current_step_owns_the_last_span_columns():
    """The final `span` columns are the step's own tokens, never history."""
    for span in (1, 8, CHUNK):
        w = _FakeWindow(pos=1000)
        lo, _ = w.mask_range(span, 0)
        assert lo == WINDOW - span - 1000
        # Row r sees exactly r+1 of its own columns.
        for r in (0, 3):
            _, hi = w.mask_range(span, r)
            assert hi - (WINDOW - span) == r + 1
