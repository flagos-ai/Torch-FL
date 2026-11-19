# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare unmodified published FlagGems against original torch_gcu, without Torch-FL."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import traceback

import torch
import torch_gcu  # noqa: F401
import flag_gems


def compare(actual, expected):
    actual, expected = actual.cpu(), expected.cpu()
    error = (actual.double() - expected.double()).abs()
    mismatches = (actual != expected).nonzero()
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "bitwise_equal": torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        ),
        "different_values": len(mismatches),
        "max_abs_error": error.max().item() if error.numel() else 0,
        "violations_atol_rtol_1e_3": int(
            (error > 1e-3 + 1e-3 * expected.double().abs()).sum()
        ),
        "examples": [
            {
                "index": list(index),
                "actual": actual[tuple(index)].item(),
                "expected": expected[tuple(index)].item(),
            }
            for index in mismatches[:6].tolist()
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert "torch_fl" not in sys.modules
    report = {
        "scope": __doc__,
        "torch_fl_imported": False,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch-gcu", "flag_gems", "flagtree")
        },
        "sources": {},
        "cases": {},
    }
    for name in ("add", "cat", "mean_dim", "rsqrt", "pow_tensor_scalar", "mm"):
        function = getattr(flag_gems, name)
        path = Path(inspect.getsourcefile(function))
        report["sources"][name] = {
            "module": function.__module__,
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run(name, function, vendor, inputs, kwargs=None, reference=None):
        kwargs = kwargs or {}
        print(json.dumps({"starting": name}), flush=True)
        try:
            torch.gcu.synchronize()
            expected = vendor(*inputs, **kwargs)
            torch.gcu.synchronize()
            actual = function(*inputs, **kwargs)
            torch.gcu.synchronize()
            result = {"against_vendor": compare(actual, expected)}
            if reference is not None:
                result["flag_gems_against_cpu_double_rounded"] = compare(
                    actual, reference
                )
                result["vendor_against_cpu_double_rounded"] = compare(
                    expected, reference
                )
        except Exception:
            result = {"error": traceback.format_exc()}
        report["cases"][name] = result
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"finished": name, **result}), flush=True)

    longs = torch.tensor(
        [
            -(2**63),
            -(2**35) - 1,
            -(2**31),
            -1,
            0,
            2**31 - 1,
            2**32 - 1,
            2**35,
            2**63 - 1,
        ],
        dtype=torch.int64,
        device="cpu",
    )
    words = longs.view(torch.int32).reshape(9, 2)
    other_words = longs.flip(0).view(torch.int32).reshape(9, 1, 2)
    a, b = words.to("gcu"), other_words.to("gcu")
    run(
        "int32_add_low_words_broadcast",
        flag_gems.add,
        torch.add,
        (a[:, 0], b[:, :, 0]),
        reference=words[:, 0] + other_words[:, :, 0],
    )
    run(
        "int32_add_high_words_broadcast",
        flag_gems.add,
        torch.add,
        (a[:, 1], b[:, :, 1]),
        reference=words[:, 1] + other_words[:, :, 1],
    )
    run(
        "int32_add_low_words_scalar",
        flag_gems.add,
        torch.add,
        (a[:, 0], 1),
        reference=words[:, 0] + 1,
    )

    generator = torch.Generator(device="cpu").manual_seed(20261119)
    hidden = torch.randn((1, 23, 1024), generator=generator, device="cpu")
    squares = hidden.square()
    variance = squares.mean(-1, keepdim=True) + 1e-6
    run(
        "float32_pow_square",
        flag_gems.pow_tensor_scalar,
        torch.pow,
        (hidden.to("gcu"), 2),
        reference=hidden.double().square().float(),
    )
    run(
        "float32_mean_dim",
        flag_gems.mean_dim,
        torch.mean,
        (squares.to("gcu"),),
        {"dim": [-1], "keepdim": True},
        reference=squares.double().mean(-1, keepdim=True).float(),
    )
    run(
        "float32_rsqrt",
        flag_gems.rsqrt,
        torch.rsqrt,
        (variance.to("gcu"),),
        reference=variance.double().rsqrt().float(),
    )

    lhs = (torch.randn((23, 1024), generator=generator, device="cpu") * 0.1).bfloat16()
    rhs = (
        torch.randn((1024, 2048), generator=generator, device="cpu") * 0.1
    ).bfloat16()
    reference = (lhs.double() @ rhs.double()).bfloat16()
    run(
        "bfloat16_mm_prefill",
        flag_gems.mm,
        torch.mm,
        (lhs.to("gcu"), rhs.to("gcu")),
        reference=reference,
    )
    run(
        "bfloat16_mm_decode",
        flag_gems.mm,
        torch.mm,
        (lhs[:1].to("gcu"), rhs.to("gcu")),
        reference=reference[:1],
    )

    key = (
        torch.arange(23 * 8 * 128, dtype=torch.float32, device="cpu")
        .reshape(1, 23, 8, 128)
        .bfloat16()
        .to("gcu")
        .transpose(1, 2)
    )
    new_key = (
        torch.arange(8 * 128, dtype=torch.float32, device="cpu")
        .reshape(1, 1, 8, 128)
        .bfloat16()
        .to("gcu")
        .transpose(1, 2)
    )
    run(
        "bfloat16_cat_kv_23_plus_1",
        flag_gems.cat,
        torch.cat,
        ([key, new_key],),
        {"dim": 2},
    )
    assert "torch_fl" not in sys.modules


if __name__ == "__main__":
    with torch.no_grad():
        main()
