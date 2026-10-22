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

"""Stream and event contracts, expressed through their methods only.

The stream object carries vendor-shaped attributes (``cuda_stream``,
``gcu_stream``, ``musa_stream``, ...) that differ per backend, so nothing here
touches an attribute: the contract is the protocol -- ``record``, ``wait``,
``synchronize``, ``query``, ``wait_stream``, ``record_event``, ``wait_event``,
``elapsed_time``. A computation on a non-default stream must, after
synchronize, equal the same computation on the default stream, and an event's
elapsed time must be non-negative. Both are hardware-agnostic invariants.
"""

import pytest
import torch

import torch_fl

DEVICE = "flagos:0"

_STREAM_METHODS = (
    "synchronize",
    "query",
    "wait_stream",
    "wait_event",
    "record_event",
)


def _operands(size: int = 64):
    torch.manual_seed(0)
    return torch.randn(size, size), torch.randn(size, size)


@pytest.mark.anyplatform
def test_current_stream_exposes_the_protocol():
    stream = torch_fl.flagos.current_stream()
    for name in _STREAM_METHODS:
        assert callable(getattr(stream, name)), f"stream lacks {name}()"
    assert stream.query() in (True, False)
    stream.synchronize()


@pytest.mark.anyplatform
def test_matmul_on_non_default_stream_matches_default():
    a, b = _operands()
    reference = torch.mm(a, b)
    device_a = a.to(DEVICE)
    device_b = b.to(DEVICE)
    on_default = torch.mm(device_a, device_b)
    on_other = None
    stream = torch_fl.flagos.Stream()
    with torch_fl.flagos.stream(stream):
        on_other = torch.mm(device_a, device_b)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(on_other.cpu(), on_default.cpu(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(on_other.cpu(), reference, rtol=1e-3, atol=1e-3)


@pytest.mark.anyplatform
def test_event_elapsed_time_is_non_negative():
    torch_fl.flagos.synchronize()
    start = torch_fl.flagos.Event(enable_timing=True)
    end = torch_fl.flagos.Event(enable_timing=True)
    start.record()
    x = torch.randn(64, 64, device=DEVICE)
    torch.mm(x, x)
    end.record()
    end.synchronize()
    assert start.elapsed_time(end) >= 0.0


@pytest.mark.anyplatform
def test_event_wait_orders_work_across_streams():
    a, b = _operands()
    reference = torch.mm(a, b)
    device_a = a.to(DEVICE)
    device_b = b.to(DEVICE)
    stream = torch_fl.flagos.Stream()
    with torch_fl.flagos.stream(stream):
        out = torch.mm(device_a, device_b)
        event = torch_fl.flagos.Event()
        event.record()
    torch_fl.flagos.current_stream().wait_event(event)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(out.cpu(), reference, rtol=1e-3, atol=1e-3)
