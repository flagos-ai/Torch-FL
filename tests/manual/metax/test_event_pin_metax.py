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

"""Verify flagos Event (real device semantics) and pin_memory on MetaX.

Run (from repo root):
    FLAGOS_ACCELERATOR=metax \
    MACA_PATH=/opt/maca \
    LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/lib64:$LD_LIBRARY_PATH \
    PYTHONPATH=$PWD \
    python tests/manual/metax/test_event_pin_metax.py

Two areas:
  1. Event -- flagos.Event now wraps torch.cuda.Event (real maca event), so
     record/wait enforce cross-stream ordering and elapsed_time is on-device.
     The previous host-timestamp stand-in had a no-op wait() -- this test would
     have raced.
  2. pin_memory -- x.pin_memory() must produce is_pinned()==True host memory,
     and non_blocking H2D from a pinned tensor must land correct data. Also
     checks tensor.to(device, pin_memory-side) round trips.
"""

import torch_fl  # noqa: F401  MUST precede torch (boxing preload + GEMS_VENDOR)
import torch

results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def test_event_timing():
    """elapsed_time must be a real on-device measure (> 0 for real work)."""
    dev = torch.device("flagos:0")
    torch.cuda.set_device(0)
    start = torch_fl.flagos.Event(enable_timing=True)
    end = torch_fl.flagos.Event(enable_timing=True)

    a = torch.randn(2048, 2048, device=dev)
    start.record()
    for _ in range(20):
        a = a @ a
        a = a / a.norm()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end)
    check("event.elapsed_time>0", ms > 0.0, f"{ms:.3f} ms")
    check("event.query after sync", end.query() is True)


def test_event_cross_stream_ordering():
    """event.wait must make a second stream wait for work on the first.

    Under the old host-timestamp Event, wait() was a no-op, so the consumer
    stream could read x before the producer finished -> wrong result / race.
    With a real event, the ordering holds.
    """
    dev = torch.device("flagos:0")
    torch.cuda.set_device(0)

    producer = torch_fl.flagos.Stream(device=0)
    consumer = torch_fl.flagos.Stream(device=0)
    ev = torch_fl.flagos.Event()

    n = 4096
    with torch_fl.flagos.stream(producer):
        x = torch.ones(n, n, device=dev)
        for _ in range(30):
            x = x + 1.0  # heavy-ish producer work
        ev.record(producer)

    # consumer must not proceed until producer's event fires
    consumer.wait_event(ev)
    with torch_fl.flagos.stream(consumer):
        y = x * 2.0
    torch_fl.flagos.synchronize()

    expected = (1.0 + 30) * 2.0
    ok = torch.allclose(y.cpu(), torch.full((n, n), expected))
    check("event cross-stream ordering", ok, f"got {y[0, 0].item()} expect {expected}")


def test_pin_memory():
    """x.pin_memory() yields real pinned host memory usable for async H2D."""
    x = torch.randn(1024, 1024)
    check("cpu tensor not pinned initially", x.is_pinned() is False)

    xp = x.pin_memory()
    check("pin_memory() -> is_pinned", xp.is_pinned() is True)
    check("pin_memory preserves data", torch.allclose(xp, x))

    # async H2D from pinned staging buffer
    dev = torch.device("flagos:0")
    torch.cuda.set_device(0)
    d = xp.to(dev, non_blocking=True)
    torch_fl.flagos.synchronize()
    check("non_blocking H2D from pinned correct", torch.allclose(d.cpu(), x))


def test_to_pin_memory_flag():
    """tensor.to(..., pin_memory=True) path. Historically raised on flagos.

    A CPU-destination copy with pin_memory=True should yield pinned host mem;
    the previous _to_copy hard-rejected any pin_memory=True. A non-CPU dest
    with pin_memory=True must still raise (you cannot pin device memory).
    """
    dev = torch.device("flagos:0")
    torch.cuda.set_device(0)
    d = torch.randn(512, 512, device=dev)

    # baseline device->host still works
    c = d.to("cpu", copy=True)
    check("device->cpu copy", torch.allclose(c.cpu(), d.cpu()))

    # device -> CPU with pin_memory=True now yields pinned host memory
    cpu_pinned = torch.ops.aten._to_copy(d, device=torch.device("cpu"), pin_memory=True)
    check("_to_copy(cpu, pin=True) is_pinned", cpu_pinned.is_pinned() is True)
    check("_to_copy(cpu, pin=True) data", torch.allclose(cpu_pinned, d.cpu()))

    # pin_memory=True with a non-CPU (flagos) destination must be rejected
    try:
        torch.ops.aten._to_copy(c, device=dev, pin_memory=True)
        check("_to_copy(flagos, pin=True) rejected", False, "no error raised")
    except RuntimeError as e:
        check("_to_copy(flagos, pin=True) rejected", "pin_memory" in str(e))


if __name__ == "__main__":
    torch_fl.flagos._lazy_init()

    for fn in (
        test_event_timing,
        test_event_cross_stream_ordering,
        test_pin_memory,
        test_to_pin_memory_flag,
    ):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            check(fn.__name__, False, f"raised {type(e).__name__}: {e}")

    n_fail = sum(1 for _, ok in results if not ok)
    status = "ALL PASS" if n_fail == 0 else f"{n_fail} FAILED"
    print(f"=== event/pin metax: {status} ({[n for n, ok in results]}) ===")
    raise SystemExit(1 if n_fail else 0)
