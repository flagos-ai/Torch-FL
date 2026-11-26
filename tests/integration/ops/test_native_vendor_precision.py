# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Device and thread isolation of the published vendor precision adapters."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import torch_fl

pytestmark = [
    pytest.mark.nvidia,
    pytest.mark.native_source_host,
    pytest.mark.native_vendor_precision,
    pytest.mark.flaggems,
    pytest.mark.main_ops,
]


@pytest.fixture(autouse=True)
def require_native_vendor():
    assert torch_fl._build_config.NATIVE_VENDOR


def exercise(device):
    value = torch.full((3, 8), 4.0, device=device)
    matrix = torch.ones((3, 8), dtype=torch.bfloat16, device=device)
    other = torch.ones((8, 5), dtype=torch.bfloat16, device=device)
    results = (
        (value.pow(2), torch.full((3, 8), 16.0)),
        (value.mean(-1), torch.full((3,), 4.0)),
        (value.rsqrt(), torch.full((3, 8), 0.5)),
        (value.softmax(-1), torch.full((3, 8), 0.125)),
        (matrix @ other, torch.full((3, 5), 8.0, dtype=torch.bfloat16)),
        (
            torch.bmm(matrix.expand(2, 3, 8), other.expand(2, 8, 5)),
            torch.full((2, 3, 5), 8.0, dtype=torch.bfloat16),
        ),
    )
    for actual, expected in results:
        assert actual.device == torch.device(device)
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.min_devices(2)
def test_vendor_precision_preserves_ambient_device_and_stream():
    initial = torch_fl.flagos.current_device()
    other = (initial + 1) % torch_fl.flagos.device_count()
    stream = torch.Stream(device=f"flagos:{other}")
    with stream, torch.no_grad():
        torch_fl.flagos.set_device(initial)
        exercise(f"flagos:{other}")
        assert torch_fl.flagos.current_device() == initial
        assert torch.accelerator.current_stream(other).stream_id == stream.stream_id
    assert torch_fl.flagos.current_device() == initial


def test_vendor_precision_handles_are_thread_and_stream_local():
    def worker():
        stream = torch.Stream(device="flagos:0")
        with stream, torch.no_grad():
            for _ in range(3):
                exercise("flagos:0")
            return stream.stream_id

    with ThreadPoolExecutor(max_workers=3) as pool:
        streams = list(pool.map(lambda _: worker(), range(3)))
    assert len(set(streams)) == 3
