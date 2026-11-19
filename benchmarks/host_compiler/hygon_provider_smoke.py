# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Check the pinned Hygon provider independently of native-host admission.

Run with the manufacturer's PyTorch and the published FlagGems/FlagTree wheels.
This does not load torch_fl, compile native hosts, or measure warm performance.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import time
import traceback
import zipfile

import torch
import triton
import flag_gems


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flagtree-wheel", type=Path, required=True)
    parser.add_argument("--flaggems-wheel", type=Path, required=True)
    args = parser.parse_args()
    if "torch_fl" in sys.modules or torch.version.hip is None:
        raise RuntimeError("Use a separate vendor PyTorch process without torch_fl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261117)
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    assert flag_gems.vendor_name == "hygon", flag_gems.vendor_name
    from triton.backends.hcu import driver

    record = {
        "scope": "Original FlagGems Python hosts on vendor PyTorch; not native-host execution",
        "torch": torch.__version__,
        "vendor_torch_distribution": importlib.metadata.version("torch"),
        "torch_file": torch.__file__,
        "torch_hip": torch.version.hip,
        "flaggems": importlib.metadata.version("flag_gems"),
        "flagtree": importlib.metadata.version("flagtree"),
        "triton": triton.__version__,
        "flaggems_vendor": flag_gems.vendor_name,
        "device": torch.cuda.get_device_name(0),
        "visible_devices": torch.cuda.device_count(),
        "torch_fl_imported": "torch_fl" in sys.modules,
        "sources": {},
        "cases": [],
    }
    driver_root = Path(driver.__file__).parent
    with zipfile.ZipFile(args.flagtree_wheel) as archive:
        for name in ("driver.py", "driver.c", "compiler.py", "compiler_hcu.py"):
            checksum = sha256(driver_root / name)
            published = hashlib.sha256(
                archive.read("triton/backends/hcu/" + name)
            ).hexdigest()
            record["sources"][name] = {
                "sha256": checksum,
                "matches_cached_wheel": checksum == published,
            }
    if not all(source["matches_cached_wheel"] for source in record["sources"].values()):
        raise RuntimeError(
            "Installed HCU driver/compiler differs from the supplied wheel"
        )
    record["device_properties"] = (
        triton.runtime.driver.active.utils.get_device_properties(0)
    )
    record["compiler_target"] = str(triton.runtime.driver.active.get_current_target())

    def save():
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    def check(name, dtype, inputs, gems, vendor, *, stream=None):
        case = {
            "op": name,
            "dtype": str(dtype),
            "inputs": [
                {"shape": list(x.shape), "stride": list(x.stride())} for x in inputs
            ],
        }
        begin = time.monotonic()
        try:
            with torch.no_grad(), torch.cuda.stream(stream):
                expected = vendor(*inputs)
                actual = gems(*inputs)
            torch.cuda.synchronize()
            case["output_device"] = str(actual.device)
            if actual.device != device or expected.device != device:
                raise AssertionError(
                    "Provider smoke must return tensors on the Hygon device"
                )
            expected, actual = expected.cpu(), actual.cpu()
            tolerance = {
                torch.float32: (1e-4, 1e-4),
                torch.float16: (3e-3, 3e-3),
                torch.bfloat16: (3e-2, 5e-2),
            }[dtype]
            torch.testing.assert_close(
                actual, expected, rtol=tolerance[0], atol=tolerance[1]
            )
            case.update(
                status="passed",
                max_abs_error=float((actual.float() - expected.float()).abs().max())
                if actual.numel()
                else 0.0,
                rtol=tolerance[0],
                atol=tolerance[1],
            )
        except Exception as exc:
            case.update(
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        case["elapsed_seconds"] = round(time.monotonic() - begin, 3)
        case["stream"] = "nondefault" if stream is not None else "default"
        record["cases"].append(case)
        save()
        print(json.dumps(case), flush=True)

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(3, 5, 7, dtype=dtype, device=device)
        y = torch.randn(7, dtype=dtype, device=device)
        strided = torch.randn(11, 19, dtype=dtype, device=device).t()
        empty = torch.empty(0, dtype=dtype, device=device)
        a = torch.randn(17, 128, dtype=dtype, device=device)
        b = torch.randn(64, 128, dtype=dtype, device=device).t()
        for name, inputs, gems, vendor in (
            ("add_broadcast", (x, y), flag_gems.add, torch.add),
            ("mul_broadcast", (x, y), flag_gems.mul, torch.mul),
            ("mul_strided", (strided, strided), flag_gems.mul, torch.mul),
            ("cos", (x,), flag_gems.cos, torch.cos),
            ("cos_empty", (empty,), flag_gems.cos, torch.cos),
            ("silu", (x,), flag_gems.silu, torch.nn.functional.silu),
            (
                "softmax",
                (x,),
                lambda z: flag_gems.softmax(z, -1, False),
                lambda z: torch.softmax(z, -1),
            ),
            ("mm_transposed", (a, b), flag_gems.mm, torch.mm),
        ):
            check(name, dtype, inputs, gems, vendor)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        check(
            "cos_nondefault_stream",
            dtype,
            (x,),
            flag_gems.cos,
            torch.cos,
            stream=stream,
        )
    record["host_identities"] = {
        name: {
            "module": getattr(flag_gems, name).__module__,
            "source": inspect.getsourcefile(getattr(flag_gems, name)),
        }
        for name in ("add", "mul", "mm", "softmax", "silu", "cos")
    }
    provider_root = Path(flag_gems.__file__).resolve().parent.parent
    with zipfile.ZipFile(args.flaggems_wheel) as archive:
        for host in record["host_identities"].values():
            path = Path(host["source"]).resolve()
            host["sha256"] = sha256(path)
            original = archive.read(str(path.relative_to(provider_root)))
            host["matches_cached_wheel"] = (
                host["sha256"] == hashlib.sha256(original).hexdigest()
            )
    record["summary"] = {
        status: sum(c["status"] == status for c in record["cases"])
        for status in ("passed", "failed")
    }
    save()
    if not all(
        host["matches_cached_wheel"] for host in record["host_identities"].values()
    ):
        raise RuntimeError("Installed FlagGems host differs from the supplied wheel")
    print(json.dumps(record["summary"]), flush=True)
    return int(record["summary"]["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
