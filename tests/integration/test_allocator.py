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

"""Tests for the caching device allocator."""

import pytest
import torch
import torch_fl  # noqa: F401

# Selected by the manifests' shared `Multi-device contracts` step (issue #391).
pytestmark = pytest.mark.multi_device


@pytest.fixture(autouse=True)
def _setup():
    """Ensure flagos device is available."""
    if not torch_fl.flagos.is_available():
        pytest.skip("No flagos device available")
    torch_fl.flagos.init()


class TestCachingAllocatorBasic:
    """Basic allocation/deallocation and cache hit tests."""

    def test_allocate_and_free(self):
        """Tensor allocation should succeed."""
        device = torch.device("flagos", 0)
        x = torch.randn(1024, device=device)
        assert x.device.type in ("privateuseone", "flagos")
        assert x.numel() == 1024
        del x

    def test_cache_reuse(self):
        """After freeing, the same size allocation should reuse cached memory."""
        device = torch.device("flagos", 0)

        # Clear cache to start fresh.
        torch_fl.flagos.empty_cache()
        stats_before = torch_fl.flagos.memory_stats(0)
        malloc_before = stats_before.get("num_device_malloc", 0)

        # First allocation triggers a device malloc.
        x = torch.randn(1024, device=device)
        del x

        # Second allocation of same size should reuse cached block.
        y = torch.randn(1024, device=device)
        stats_after = torch_fl.flagos.memory_stats(0)
        malloc_after = stats_after.get("num_device_malloc", 0)

        # Should have at most one new device malloc (the first one).
        # The second should hit cache.
        assert malloc_after - malloc_before <= 1, (
            f"Expected cache reuse, but got {malloc_after - malloc_before} "
            f"device mallocs for two same-size allocations"
        )
        del y

    def test_empty_cache(self):
        """empty_cache should release reserved memory."""
        device = torch.device("flagos", 0)
        x = torch.randn(1024 * 1024, device=device)  # 4MB
        del x

        reserved_before = torch_fl.flagos.memory_reserved(0)
        assert reserved_before > 0

        torch_fl.flagos.empty_cache()
        reserved_after = torch_fl.flagos.memory_reserved(0)
        # After empty_cache, reserved should decrease.
        assert reserved_after < reserved_before

    def test_memory_stats_basic(self):
        """memory_stats should return a dict with expected keys."""
        stats = torch_fl.flagos.memory_stats(0)
        assert isinstance(stats, dict)
        expected_keys = [
            "allocated_bytes",
            "reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "num_alloc_calls",
            "num_free_calls",
            "num_device_malloc",
            "num_device_free",
            "num_alloc_retries",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_memory_allocated_tracking(self):
        """memory_allocated should reflect current tensor allocation."""
        device = torch.device("flagos", 0)
        torch_fl.flagos.empty_cache()

        alloc_before = torch_fl.flagos.memory_allocated(0)
        x = torch.randn(1024 * 256, device=device)  # 1MB float32
        alloc_during = torch_fl.flagos.memory_allocated(0)
        del x
        alloc_after = torch_fl.flagos.memory_allocated(0)

        # During: should be higher than before.
        assert alloc_during > alloc_before
        # After: should drop back.
        assert alloc_after < alloc_during


class TestCachingAllocatorMultipleSizes:
    """Test allocation across different size classes."""

    def test_small_pool_allocation(self):
        """Small allocations (<=1MB) should use the small pool."""
        device = torch.device("flagos", 0)
        # 512 bytes - definitely small pool
        x = torch.zeros(128, dtype=torch.float32, device=device)
        assert x.numel() == 128
        del x

    def test_large_pool_allocation(self):
        """Large allocations (>1MB) should use the large pool."""
        device = torch.device("flagos", 0)
        # ~4MB - large pool
        x = torch.zeros(1024 * 1024, dtype=torch.float32, device=device)
        assert x.numel() == 1024 * 1024
        del x

    def test_multiple_allocations(self):
        """Multiple tensors can coexist."""
        device = torch.device("flagos", 0)
        tensors = []
        for i in range(10):
            tensors.append(torch.randn(1024, device=device))
        # All should be valid.
        for t in tensors:
            assert t.numel() == 1024
        del tensors

    def test_varied_sizes(self):
        """Allocations of different sizes should all work."""
        device = torch.device("flagos", 0)
        sizes = [1, 64, 512, 1024, 4096, 1024 * 1024]
        for size in sizes:
            x = torch.randn(size, device=device)
            assert x.numel() == size
            del x


class TestCachingAllocatorOps:
    """Test that normal operations work with the caching allocator."""

    def test_matmul(self):
        """Matrix multiplication should work with cached allocations."""
        device = torch.device("flagos", 0)
        a = torch.randn(32, 64, device=device)
        b = torch.randn(64, 16, device=device)
        c = torch.mm(a, b)
        assert c.shape == (32, 16)

    def test_copy_between_devices(self):
        """Host-to-device and device-to-host copies should work."""
        device = torch.device("flagos", 0)
        x_cpu = torch.randn(1024)
        x_dev = x_cpu.to(device)
        x_back = x_dev.to("cpu")
        assert torch.allclose(x_cpu, x_back, atol=1e-6)


class TestCachingAllocatorPeakStats:
    """reset_peak_memory_stats must drop the watermarks and nothing else.

    Regression cover for the self-managed (non-delegating) allocator path, which
    used to zero its whole stats struct on a reset. The live totals went with it,
    so the peak read afterwards counted only what the *next* call allocated: a
    resident model's weights disappeared from the number and the reported peak
    was off by exactly the live footprint (6.87 GiB instead of 37.09 GiB on a
    Qwen-Image-2.1 benchmark). The delegating CUDA/DCU path never had the bug --
    it forwards to the platform allocator's resetPeakStats, which re-bases the
    peak on the live total -- and these tests pin the self-managed path to the
    same semantics.
    """

    def test_reset_keeps_live_allocation(self):
        """A held tensor survives the reset and becomes the new peak floor."""
        device = torch.device("flagos", 0)
        torch_fl.flagos.empty_cache()

        x = torch.randn(1024 * 1024, device=device)  # 4MB float32, kept alive
        live = torch_fl.flagos.memory_allocated(0)
        assert live > 0

        torch_fl.flagos.reset_peak_memory_stats(0)

        assert torch_fl.flagos.memory_allocated(0) == live
        stats = torch_fl.flagos.memory_stats(0)
        assert stats["allocated_bytes"] == live
        # Re-based on the live total rather than zeroed, so the next peak read is
        # a maximum over what is live from here on.
        assert stats["peak_allocated_bytes"] == live
        assert stats["peak_reserved_bytes"] == stats["reserved_bytes"]
        del x

    def test_reset_drops_the_watermark(self):
        """A peak above the live total must come back down to the live total."""
        device = torch.device("flagos", 0)
        torch_fl.flagos.empty_cache()

        x = torch.randn(1024 * 1024, device=device)
        assert torch_fl.flagos.memory_stats(0)["peak_allocated_bytes"] > 0
        del x
        torch_fl.flagos.empty_cache()

        torch_fl.flagos.reset_peak_memory_stats(0)

        stats = torch_fl.flagos.memory_stats(0)
        assert stats["peak_allocated_bytes"] == stats["allocated_bytes"]
        assert stats["peak_reserved_bytes"] == stats["reserved_bytes"]

    def test_reset_keeps_cumulative_counters(self):
        """The call counters are cumulative and must not restart at the reset."""
        device = torch.device("flagos", 0)
        torch_fl.flagos.empty_cache()

        torch.randn(1024, device=device)  # warm the pool, then drop it
        before = torch_fl.flagos.memory_stats(0)

        torch_fl.flagos.reset_peak_memory_stats(0)

        x = torch.randn(1024, device=device)
        after = torch_fl.flagos.memory_stats(0)
        assert after["num_alloc_calls"] > before["num_alloc_calls"]
        assert after["num_device_malloc"] >= before["num_device_malloc"]
        del x
