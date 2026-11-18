# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run on original torch_gcu: python enflame_strided_add_probe.py OUTPUT.json."""

import json
from pathlib import Path
import sys

import torch
import torch_gcu  # noqa: F401
import flag_gems

assert "torch_fl" not in sys.modules
results = []
with torch.no_grad():
    for n in (2, 3, 9):
        for dtype in (torch.int32, torch.float32):
            for strided in (False, True):
                cpu_a = torch.arange(n * 2, dtype=dtype, device="cpu").reshape(n, 2)
                cpu_b = (
                    torch.arange(n * 2, dtype=dtype, device="cpu").reshape(n, 1, 2) * 20
                    + 100
                )
                a, b = cpu_a.to("gcu")[:, 0], cpu_b.to("gcu")[:, :, 0]
                if not strided:
                    a, b = a.contiguous(), b.contiguous()
                torch.gcu.synchronize()
                expected = torch.add(a, b)
                torch.gcu.synchronize()
                actual = flag_gems.add(a, b)
                torch.gcu.synchronize()
                host = actual.cpu()
                contiguous_host = actual.contiguous().cpu()
                reference = cpu_a[:, 0] + cpu_b[:, :, 0]
                results.append(
                    {
                        "n": n,
                        "dtype": str(dtype),
                        "strided": strided,
                        "a_stride": list(a.stride()),
                        "b_stride": list(b.stride()),
                        "output_stride": list(actual.stride()),
                        "different_cpu_values": int((host != reference).sum()),
                        "different_contiguous_cpu_values": int(
                            (contiguous_host != reference).sum()
                        ),
                        "different_gpu_values_against_vendor": int(
                            (actual != expected).sum().item()
                        ),
                        "actual": host.tolist(),
                        "expected": reference.tolist(),
                    }
                )
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in results[-1].items()
                            if k not in ("actual", "expected")
                        }
                    ),
                    flush=True,
                )
Path(sys.argv[1]).write_text(json.dumps(results, indent=2) + "\n")
