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

"""Live single-process test for DataParallel on flagos devices (issue #384).

Single process, several devices -- DataParallel is intra-process, so unlike the
DDP tests under this directory there is nothing to spawn. Covers the three
device decisions that are CUDA-only in stock torch, the comm layer under them,
and the forward/backward paths that exercise both.

Run:
    PYTHONPATH=<repo root> python tests/manual/test_dataparallel_live.py
    PYTHONPATH=<repo root> python tests/manual/test_dataparallel_live.py --devices 4
"""

import argparse

# torch_fl MUST be imported before torch (preloads libtorch_cuda.so).
import torch_fl  # noqa: F401
import torch
import torch.nn as nn
from torch.nn.parallel import comm
from torch.nn.parallel.data_parallel import data_parallel

FAILED = []
SKIPPED = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def skip(name, reason):
    print(f"SKIP  {name} -- {reason}")
    SKIPPED.append(name)


class Net(nn.Module):
    """Linear + a buffer, so the parameters-and-buffers guard has both to check."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
        self.register_buffer("scale", torch.ones(4))

    def forward(self, x):
        return self.fc(x) * self.scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        type=int,
        default=2,
        help="number of flagos devices to parallelise over (default: 2)",
    )
    args = parser.parse_args()

    available = torch.flagos.device_count()
    device_count = min(args.devices, available)
    if device_count < 2:
        print(f"need at least 2 flagos devices, found {available}")
        return 1
    device_ids = list(range(device_count))
    print(f"flagos devices: {available}, parallelising over {device_ids}\n")

    torch.manual_seed(0)

    # --- root cause 1: the device type DataParallel resolves ------------------
    from torch._utils import _get_available_device_type

    device_type = _get_available_device_type()
    check(
        "rc1: device resolution is cuda-first, so the patch must not rely on it",
        device_type == "cuda" and str(torch.device(0)) == "flagos:0",
        f"_get_available_device_type()={device_type!r} torch.device(0)={torch.device(0)}",
    )

    # --- comm.scatter / gather / broadcast_coalesced --------------------------
    rows = 4 * device_count  # divisible, so chunk() yields one chunk per device
    x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4).to(0)
    chunks = comm.scatter(x, device_ids, None, 0)
    parts = x.chunk(device_count, 0)
    check(
        "comm.scatter(flagos tensor) -> one chunk per flagos device",
        all(c.device.type == "flagos" for c in chunks)
        and [c.device.index for c in chunks] == device_ids
        and all(torch.equal(c.cpu(), p.cpu()) for c, p in zip(chunks, parts)),
        str([str(c.device) for c in chunks]),
    )

    # chunk_sizes=None goes through at::chunk, which returns *fewer* chunks than
    # requested when the size does not divide -- the row count here is chosen to
    # divide. Pin the un-divided case against the C++ behaviour by running the
    # same call on cuda-labelled tensors, where the stock op still works (on PPU
    # and MetaX cuda:N is the same hardware): the chunk counts and sizes have to
    # agree, or DataParallel's forward would pick a different set of replicas.
    cuda_x = x.to(torch.device("cuda", 0))
    for probe_rows in (rows - 1, 1):
        got = comm.scatter(x[:probe_rows], device_ids, None, 0)
        stock = comm.scatter(cuda_x[:probe_rows], device_ids, None, 0)
        check(
            f"comm.scatter matches the C++ chunking for {probe_rows} rows "
            f"over {device_count} devices",
            [c.shape[0] for c in got] == [c.shape[0] for c in stock]
            and all(c.device.type == "flagos" for c in got),
            f"{[c.shape[0] for c in got]} chunks, same as "
            f"{[str(c.device) for c in stock]}",
        )

    gathered = comm.gather(list(chunks), 0, 0)
    check(
        "comm.gather(flagos tensors, destination=0) -> flagos:0",
        str(gathered.device) == "flagos:0" and torch.equal(gathered.cpu(), x.cpu()),
        str(gathered.device),
    )

    to_cpu = comm.gather(list(chunks), 0, -1)
    check(
        "comm.gather(..., destination=-1) -> cpu",
        to_cpu.device.type == "cpu" and torch.equal(to_cpu, x.cpu()),
        str(to_cpu.device),
    )

    # Explicit chunk_sizes take the split path instead of the chunk path. Three
    # devices where the host has them, so the split is uneven either way.
    if device_count >= 3:
        split_devices, split_sizes = [0, 1, 2], [rows // 2, rows // 4, rows // 4]
    else:
        split_devices, split_sizes = [0, 1], [rows // 2, rows - rows // 2]
    uneven = comm.scatter(x, split_devices, split_sizes, 0)
    check(
        "comm.scatter with explicit chunk_sizes -> uneven flagos chunks",
        [c.shape[0] for c in uneven] == split_sizes
        and [c.device.index for c in uneven] == split_devices
        and all(c.device.type == "flagos" for c in uneven),
        str([(c.shape[0], str(c.device)) for c in uneven]),
    )

    t1, t2 = torch.ones(2).to(0), torch.full((2,), 2.0).to(0)
    bcast = comm.broadcast_coalesced([t1, t2], device_ids)
    check(
        "comm.broadcast_coalesced -> list per device, all flagos",
        all(
            [str(t.device) for t in bcast[i]] == [f"flagos:{i}"] * 2 for i in device_ids
        )
        and torch.equal(bcast[-1][1].cpu(), t2.cpu()),
        str([[str(t.device) for t in per_device] for per_device in bcast]),
    )

    # --- DataParallel construction and forward --------------------------------
    model = Net().to(0)
    batch = 2 * device_count
    base_input = torch.randn(batch, 8)
    reference = model(base_input.to(0)).detach()

    for label, move in (
        ("cpu input", lambda t: t),
        ("flagos input", lambda t: t.to(0)),
    ):
        dp = nn.DataParallel(model, device_ids=device_ids)
        check(
            f"DataParallel(device_ids={device_ids}) state -- {label}",
            dp.device_ids == device_ids and str(dp.src_device_obj) == "flagos:0",
            f"device_ids={dp.device_ids} src={dp.src_device_obj}",
        )
        out = dp(move(base_input)).detach()
        check(
            f"DataParallel forward matches a single device -- {label}",
            str(out.device) == "flagos:0"
            and torch.allclose(out.cpu(), reference.cpu(), atol=1e-5),
            f"{out.device} max_delta={(out.cpu() - reference.cpu()).abs().max().item():.3e}",
        )

    # Default device_ids come from the flagos device count, not torch.cuda's.
    dp_all = nn.DataParallel(model)
    check(
        "DataParallel() default device_ids are the flagos devices",
        dp_all.device_ids == list(range(available)),
        f"{dp_all.device_ids} over {available} devices",
    )

    if device_count >= 3:
        # Uneven split through the public entry point: an odd batch over an odd
        # number of devices leaves the last replica short (5 rows over 3 devices
        # is 2/2/1), so this goes through the same at::chunk path as the
        # scatter check above rather than a hand-built device list.
        odd_input = base_input[: 2 * device_count - 1]
        dp3 = nn.DataParallel(model, device_ids=[0, 1, 2])
        out3 = dp3(odd_input).detach()
        check(
            "DataParallel forward with an uneven split over 3 devices",
            str(out3.device) == "flagos:0"
            and torch.allclose(
                out3.cpu(), model(odd_input.to(0)).detach().cpu(), atol=1e-5
            ),
            f"{odd_input.shape[0]} rows: {out3.device} "
            f"max_delta={(out3.cpu() - model(odd_input.to(0)).detach().cpu()).abs().max().item():.3e}",
        )

    # Fewer chunks than devices: the tail replicas are simply unused.
    short = base_input[:2]
    dp_short = nn.DataParallel(model, device_ids=device_ids)
    out_short = dp_short(short).detach()
    check(
        "DataParallel forward with fewer inputs than devices",
        str(out_short.device) == "flagos:0"
        and torch.allclose(
            out_short.cpu(), model(short.to(0)).detach().cpu(), atol=1e-5
        ),
        f"batch {short.shape[0]} over {device_count} devices -> {out_short.shape}",
    )

    # output_device=-1 gathers back to the host.
    dp_cpu_out = nn.DataParallel(model, device_ids=device_ids, output_device=-1)
    out_cpu = dp_cpu_out(base_input).detach()
    check(
        "DataParallel(output_device=-1) gathers to cpu",
        out_cpu.device.type == "cpu"
        and torch.allclose(out_cpu, reference.cpu(), atol=1e-5),
        str(out_cpu.device),
    )

    # A single-device DataParallel moves the module and takes the no-scatter path.
    dp_one = nn.DataParallel(model, device_ids=[0])
    out_one = dp_one(base_input.to(0)).detach()
    check(
        "DataParallel(device_ids=[0]) forward",
        str(out_one.device) == "flagos:0"
        and torch.allclose(out_one.cpu(), reference.cpu(), atol=1e-5),
        str(out_one.device),
    )

    # The functional entry point reaches the same patched class.
    out_fn = data_parallel(model, base_input, device_ids=device_ids).detach()
    check(
        "torch.nn.parallel.data_parallel(functional) matches a single device",
        str(out_fn.device) == "flagos:0"
        and torch.allclose(out_fn.cpu(), reference.cpu(), atol=1e-5),
        f"{out_fn.device} max_delta={(out_fn.cpu() - reference.cpu()).abs().max().item():.3e}",
    )

    # --- training: backward + optimizer step ----------------------------------
    train_model = Net().to(0)
    train_dp = nn.DataParallel(train_model, device_ids=device_ids)
    optimizer = torch.optim.SGD(train_dp.parameters(), lr=0.1)
    loss = train_dp(torch.randn(batch, 8)).sum()
    loss.backward()
    grads = [
        (p.grad.device.type, p.grad.device.index)
        for p in train_model.parameters()
        if p.grad is not None
    ]
    check(
        "backward reaches the source module with flagos grads",
        grads and all(t == "flagos" for t, _ in grads),
        str(grads),
    )
    optimizer.step()
    check("optimizer.step() after a DataParallel backward", True)

    # --- nothing else changes: a cuda-placed module keeps the stock path ------
    try:
        cuda_model = Net().to(torch.device("cuda", 0))
        cuda_dp = nn.DataParallel(cuda_model, device_ids=[0, 1])
        check(
            "a cuda-placed module still takes the original __init__ path",
            getattr(cuda_dp, "_flagos_device_type", None) is None
            and str(cuda_dp.src_device_obj) == "cuda:0",
            f"src={cuda_dp.src_device_obj}",
        )
    except Exception as exc:  # noqa: BLE001 - reported as a failure, not raised
        check(
            "a cuda-placed module still takes the original __init__ path",
            False,
            repr(exc),
        )

    # --- root cause 3: device identity survives a properties sweep ------------
    # The MetaX-only leak is not observable here: PPU does not route
    # _query_metax_device_properties, so this only records that the sweep is
    # harmless on this backend. See the PR for the evidence gap.
    if torch.flagos.device_count() > 0:
        before = str(torch.tensor([1.0]).to(0).device)
        for index in range(torch.flagos.device_count()):
            torch.cuda.get_device_properties(index)
        after = str(torch.tensor([1.0]).to(0).device)
        check(
            "rc3 (PPU): .to(0) still lands on flagos:0 after a properties sweep",
            after == "flagos:0",
            f"{before} -> {after}",
        )
    else:
        skip("rc3 properties sweep", "no flagos devices")

    print()
    print(f"FAILED: {FAILED if FAILED else 'none'}")
    print(f"SKIPPED: {SKIPPED if SKIPPED else 'none'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
