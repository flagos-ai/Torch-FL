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

"""TEMPORARY runner diagnostic for the MPS native kernel suite.

The native kernels are only executed by the `backend_ops` group, and on the
hosted ``macos-14`` runner that process aborts before pytest prints a summary --
so the numbers behind the failures never reach the log. This probe runs the same
shapes as ``test_mps_layer_norm.py`` through explicit reporting and always exits
0, so the measurements arrive without turning the pipeline red. It is scaffolding
rather than a test: it carries no assertions and comes out with the manifest
entry that invokes it once the runner behaviour is understood.
"""

import json
import os
import platform
import subprocess

import torch
import torch_fl  # noqa: F401

DEVICE = "flagos:0"


def _shell(command):
    try:
        done = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        return f"<{exc}>"
    return (done.stdout + done.stderr).strip()


def _diff_report(label, got, expected, rtol=1e-5, atol=1e-5):
    for index, name in enumerate(("out", "mean", "rstd")):
        got_t = got[index]
        exp_t = expected[index]
        line = {
            "case": label,
            "tensor": name,
            "device": got_t.device.type,
            "dtype": str(got_t.dtype),
            "expected_dtype": str(exp_t.dtype),
            "shape": list(got_t.shape),
            "expected_shape": list(exp_t.shape),
        }
        if tuple(got_t.shape) != tuple(exp_t.shape):
            line["verdict"] = "SHAPE-DIFFERS"
            print(json.dumps(line), flush=True)
            continue
        got_c = got_t.detach().cpu().float()
        exp_c = exp_t.detach().cpu().float()
        line["maxdiff"] = (got_c - exp_c).abs().max().item()
        try:
            torch.testing.assert_close(got_c, exp_c, rtol=rtol, atol=atol)
            line["verdict"] = "close"
        except AssertionError as exc:
            line["verdict"] = "NOT-CLOSE"
            line["detail"] = str(exc).strip().splitlines()[0]
        print(json.dumps(line), flush=True)


def _layer_norm(host, normalized_shape, weight=None, bias=None, eps=1e-5):
    return torch.ops.aten.native_layer_norm(
        host.clone().to(DEVICE), list(normalized_shape), weight, bias, eps
    )


def _reference(host, normalized_shape, weight=None, bias=None, eps=1e-5):
    return torch.ops.aten.native_layer_norm(
        host, list(normalized_shape), weight, bias, eps
    )


TRAILING_CASES = [
    ((2, 8, 512), (512,)),
    ((1, 128, 1024), (1024,)),
    ((2, 512, 1024), (1024,)),
    ((2, 3, 8), (8,)),
    ((2, 3, 8), (3, 8)),
    ((4, 6, 7, 8), (6, 7, 8)),
    ((2, 3, 8), (2, 3, 8)),
]


def _report_environment():
    print(
        json.dumps(
            {
                "kind": "environment",
                "machine": platform.machine(),
                "macos": _shell(["sw_vers", "-productVersion"]),
                "build": _shell(["sw_vers", "-buildVersion"]),
                "sdk": _shell(["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
                "xcode": _shell(["xcodebuild", "-version"]),
                "torch": torch.__version__,
                "python": platform.python_version(),
                "pointer_size": torch.tensor([1]).element_size(),
                "cwd": os.getcwd(),
            }
        ),
        flush=True,
    )


def _report_layer_norm():
    torch.manual_seed(0)
    for shape, normalized_shape in TRAILING_CASES:
        host = torch.randn(*shape)
        _diff_report(
            f"trailing {shape} ns={normalized_shape}",
            _layer_norm(host, normalized_shape),
            _reference(host, normalized_shape),
        )

    host = torch.randn(2, 3, 8)
    weight = torch.randn(8) * 2.0 + 1.0
    bias = torch.randn(8)
    _diff_report(
        "weight+bias",
        _layer_norm(host, (8,), weight, bias),
        _reference(host, (8,), weight, bias),
    )

    for eps in (1e-5, 1e-3):
        host = torch.randn(2, 8, 512)
        _diff_report(
            f"eps={eps}",
            _layer_norm(host, [512], eps=eps),
            _reference(host, [512], eps=eps),
        )

    host = torch.randn(2, 3, 8)
    weight = torch.randn(8) + 1.0
    _diff_report(
        "weight-only host affine",
        _layer_norm(host, (8,), weight),
        _reference(host, (8,), weight),
    )

    host = torch.randn(2, 8, 512)
    weight = torch.randn(512) * 2.0 + 1.0
    bias = torch.randn(512)
    _diff_report(
        "host affine pair",
        _layer_norm(host, [512], weight, bias),
        _reference(host, [512], weight, bias),
    )

    host = torch.randn(2, 3, 8)
    bias = torch.randn(8)
    _diff_report(
        "bias-only",
        _layer_norm(host, (8,), None, bias),
        _reference(host, (8,), None, bias),
    )

    host = torch.randn(4, 6, 7, 8)
    weight = torch.randn(4, 6, 7, 8)
    bias = torch.randn(4, 6, 7, 8)
    _diff_report(
        "full normalized shape",
        _layer_norm(host, (4, 6, 7, 8), weight, bias),
        _reference(host, (4, 6, 7, 8), weight, bias),
    )

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        host = (torch.randn(2, 3, 8) * 3.0).to(dtype)
        tolerance = {} if dtype == torch.float32 else {"rtol": 1e-2, "atol": 1e-2}
        _diff_report(
            f"dtype {dtype}",
            _layer_norm(host, [8]),
            _reference(host, [8]),
            **tolerance,
        )

    host = torch.randn(8, 4, 6).transpose(0, 1)
    _diff_report(
        "non-contiguous input",
        _layer_norm(host, [6]),
        _reference(host, [6]),
    )

    host = torch.randn(2, 3, 8)
    weight = torch.randn(16)[::2]
    _diff_report(
        "non-contiguous weight",
        _layer_norm(host, (8,), weight),
        _reference(host, (8,), weight),
    )

    host = torch.randn(16)
    _diff_report("rank-one", _layer_norm(host, [16]), _reference(host, [16]))


def _report_div():
    tiny = torch.tensor([1e-38])
    divisor = torch.tensor([2.0])
    cases = {
        "div.Tensor subnormal": torch.ops.aten.div.Tensor(
            tiny.clone().to(DEVICE), divisor.clone().to(DEVICE)
        ),
        "div.Scalar subnormal": torch.ops.aten.div.Scalar(tiny.clone().to(DEVICE), 2.0),
        "mul.Scalar subnormal": torch.ops.aten.mul.Scalar(tiny.clone().to(DEVICE), 0.5),
        "mul.Tensor subnormal": tiny.clone().to(DEVICE) * divisor.clone().to(DEVICE),
    }
    expected = {
        "div.Tensor subnormal": (tiny / divisor).item(),
        "div.Scalar subnormal": (tiny / 2.0).item(),
        "mul.Scalar subnormal": (tiny * 0.5).item(),
        "mul.Tensor subnormal": (tiny * divisor).item(),
    }
    for name, tensor in cases.items():
        print(
            json.dumps(
                {
                    "kind": "subnormal",
                    "case": name,
                    "got": tensor.detach().cpu().item(),
                    "expected": expected[name],
                    "flushed": tensor.detach().cpu().item() == 0.0,
                }
            ),
            flush=True,
        )


def main():
    _report_environment()
    _report_div()
    _report_layer_norm()
    # The aborting case, last and alone, so its neighbours' reports survive it.
    print(json.dumps({"kind": "abort-case", "stage": "entering"}), flush=True)
    host = torch.randn(2, 8, 512)
    weight = torch.randn(512)
    bias = torch.randn(512)
    out = torch.nn.functional.layer_norm(
        host.clone().to(DEVICE), (512,), weight=weight, bias=bias, eps=1e-5
    )
    expected = torch.nn.functional.layer_norm(
        host, (512,), weight=weight, bias=bias, eps=1e-5
    )
    _diff_report("F.layer_norm", (out,), (expected,))
    print(json.dumps({"kind": "abort-case", "stage": "survived"}), flush=True)


if __name__ == "__main__":
    main()
