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

"""Allocator contract: a documented stats key set and a working free/cache path.

The documented key set is the interface other tooling reads; this file asserts
it is a superset of the nine documented keys and that the byte counts obey the
invariants (peaks never trail the current counters). Whether a chip caches
memory or reports zero at start is a vendor detail and is not asserted.
"""

import pytest
import torch

import torch_fl

DEVICE = "flagos:0"

#: The nine keys torch_fl.flagos.memory_stats() documents.
DOCUMENTED_KEYS = (
    "allocated_bytes",
    "reserved_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "num_alloc_calls",
    "num_free_calls",
    "num_device_malloc",
    "num_device_free",
    "num_alloc_retries",
)


@pytest.mark.anyplatform
def test_memory_stats_covers_the_documented_keys():
    stats = torch_fl.flagos.memory_stats()
    assert set(DOCUMENTED_KEYS) <= set(stats)
    for key in DOCUMENTED_KEYS:
        assert isinstance(stats[key], int), f"{key} is not an int"


@pytest.mark.anyplatform
def test_allocated_never_exceeds_reserved_peaks():
    stats = torch_fl.flagos.memory_stats()
    assert stats["peak_allocated_bytes"] >= stats["allocated_bytes"]
    assert stats["peak_reserved_bytes"] >= stats["reserved_bytes"]


@pytest.mark.anyplatform
def test_alloc_free_round_trip():
    torch_fl.flagos.empty_cache()
    base = torch_fl.flagos.memory_stats()["allocated_bytes"]
    buffer = torch.empty(4 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    allocated = torch_fl.flagos.memory_stats()["allocated_bytes"]
    assert allocated >= base
    assert torch_fl.flagos.memory_allocated() == allocated
    del buffer
    # empty_cache is idempotent and must be safe to call on an idle allocator.
    torch_fl.flagos.empty_cache()
    torch_fl.flagos.empty_cache()


@pytest.mark.anyplatform
def test_reset_peak_memory_stats_flattens_the_peaks_to_current():
    torch_fl.flagos.reset_peak_memory_stats()
    stats = torch_fl.flagos.memory_stats()
    assert stats["peak_allocated_bytes"] == stats["allocated_bytes"]
    assert stats["peak_reserved_bytes"] == stats["reserved_bytes"]


@pytest.mark.anyplatform
def test_memory_reserved_matches_stats_after_a_live_allocation():
    buffer = torch.empty(1024 * 1024, dtype=torch.uint8, device=DEVICE)
    stats = torch_fl.flagos.memory_stats()
    assert torch_fl.flagos.memory_reserved() == stats["reserved_bytes"]
    assert stats["reserved_bytes"] >= stats["allocated_bytes"]
    del buffer
