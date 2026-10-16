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

The probe is delivered as ``python _mps_env_probe.py`` with no argument; that
entry point is the *driver*, and it re-invokes this same file once per section
with the section name as the argument. The split is the point: the unit under
test aborts the whole process (``Fatal Python error: Aborted`` with no traceback
of its own), so a single process that ran every section in sequence would lose
the report of every section after the first abort. Run one per child, and the
driver prints the child's signal next to the sections that did survive.
"""

import json
import os
import platform
import subprocess
import sys

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
    # ``native_layer_norm`` answers with a triple and ``F.layer_norm`` with the
    # single ``out``, so the names walk whichever length the caller handed over
    # rather than a fixed three.
    names = ("out", "mean", "rstd")
    for index in range(min(len(got), len(expected))):
        name = names[index] if index < len(names) else f"result{index}"
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
        "python div operator": tiny.clone().to(DEVICE) / 2.0,
    }
    expected = {
        "div.Tensor subnormal": (tiny / divisor).item(),
        "div.Scalar subnormal": (tiny / 2.0).item(),
        "mul.Scalar subnormal": (tiny * 0.5).item(),
        "mul.Tensor subnormal": (tiny * divisor).item(),
        "python div operator": (tiny / 2.0).item(),
    }
    for name, tensor in cases.items():
        got = tensor.detach().cpu().item()
        # The two suites disagree about this value and both are "pinned", so the
        # probe reports the direction explicitly rather than a bare number: the
        # dev host flushes a graph subnormal to zero, and a runner that *keeps*
        # it fails `TestSubnormalBoundaryIsDocumented` on the same assertion
        # that passes here.
        print(
            json.dumps(
                {
                    "kind": "subnormal",
                    "case": name,
                    "got": got,
                    "expected": expected[name],
                    "flushed": got == 0.0,
                    "direction": "FLUSHED" if got == 0.0 else "KEPT-THE-SUBNORMAL",
                }
            ),
            flush=True,
        )


def _report_upstream_mps():
    """What upstream PyTorch's own MPS layer_norm makes of the same shapes.

    The torch-fl body feeds MPSGraph the same nodes upstream's MPS
    ``native_layer_norm`` does, so if the *hosted runner* also disagrees with
    the CPU reference through ``torch.mps`` directly, the divergence is the
    runner's Metal rather than this backend's graph. If upstream agrees and
    torch-fl does not, the graph here is the difference. Either way the answer
    separates the two possibilities the failure list cannot: the probe reports
    both sides on one machine.
    """
    if not torch.backends.mps.is_available():
        print(json.dumps({"kind": "upstream", "available": False}), flush=True)
        return
    print(json.dumps({"kind": "upstream", "available": True}), flush=True)
    torch.manual_seed(0)
    for shape, normalized_shape in TRAILING_CASES:
        host = torch.randn(*shape)
        moved = host.clone().to("mps")
        got = torch.ops.aten.native_layer_norm(
            moved, list(normalized_shape), None, None, 1e-5
        )
        reference = torch.ops.aten.native_layer_norm(
            host, list(normalized_shape), None, None, 1e-5
        )
        for index, name in enumerate(("out", "mean", "rstd")):
            got_c = got[index].detach().cpu().float()
            exp_c = reference[index].detach().cpu().float()
            same_shape = tuple(got_c.shape) == tuple(exp_c.shape)
            print(
                json.dumps(
                    {
                        "kind": "upstream",
                        "case": f"trailing {shape} ns={normalized_shape}",
                        "tensor": name,
                        "shape": list(got_c.shape),
                        "maxdiff": (got_c - exp_c).abs().max().item()
                        if same_shape
                        else None,
                    }
                ),
                flush=True,
            )


def _report_timing():
    """The per-call times the three cost classes bound, with their numbers."""
    import time

    import torch.nn.functional as F

    # `torch_fl.synchronize` does not exist (the tests guard it with `hasattr`
    # and so measure launch time); report both with and without a real sync so a
    # runner whose launch cost is higher than the device cost is visible.
    def measure(fn, n):
        for _ in range(3):
            fn()
        if hasattr(torch_fl, "_C") and hasattr(torch_fl._C, "_synchronize"):
            torch_fl._C._synchronize()
        start = time.perf_counter()
        for _ in range(n):
            fn()
        if hasattr(torch_fl, "_C") and hasattr(torch_fl._C, "_synchronize"):
            torch_fl._C._synchronize()
        return (time.perf_counter() - start) / n * 1e3

    vocab, hidden, seq = 151936, 1024, 512
    weight = torch.randn(vocab, hidden, device=DEVICE)
    indices = torch.randint(0, vocab, (1, seq), device=DEVICE)
    print(
        json.dumps(
            {
                "kind": "timing",
                "case": "embedding",
                "ms": measure(lambda: F.embedding(indices, weight), 10),
                "bound": 10.0,
            }
        ),
        flush=True,
    )

    for rows, cols, index_len in ((4096, 1024, 512), (1024, 1024, 512)):
        operand = torch.randn(rows, cols, device=DEVICE)
        index = torch.randint(0, rows, (index_len,), device=DEVICE)
        print(
            json.dumps(
                {
                    "kind": "timing",
                    "case": f"index_select {rows}x{cols}",
                    "ms": measure(
                        lambda: torch.ops.aten.index_select(operand, 0, index), 10
                    ),
                    "bound": 10.0,
                }
            ),
            flush=True,
        )


def _report_frontend():
    """The aborting case, alone in its own process.

    ``F.layer_norm`` with a host affine pair is the call the runner dies inside
    (``test_the_layer_norm_front_end_reaches_the_same_kernel``), and the death
    is a ``Fatal Python error: Aborted`` with no assertion in front of it. It
    shares its section with nothing on purpose: the driver runs this section
    last and reads its exit signal, so a crash is *the measurement* here rather
    than the loss of the other sections' numbers.
    """
    print(json.dumps({"kind": "frontend", "stage": "entering"}), flush=True)
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
    print(json.dumps({"kind": "frontend", "stage": "survived"}), flush=True)


SECTIONS = {
    "environment": _report_environment,
    "div": _report_div,
    "layer_norm": _report_layer_norm,
    "upstream": _report_upstream_mps,
    "timing": _report_timing,
    "frontend": _report_frontend,
}

# The order the driver runs them in. `frontend` is last because it is the one
# expected to abort: everything before it has already been printed and flushed
# by the time its child dies.
SECTION_ORDER = ["environment", "div", "layer_norm", "timing", "frontend"]


def _run_section(name):
    print(json.dumps({"kind": "section-begin", "section": name}), flush=True)
    SECTIONS[name]()
    print(json.dumps({"kind": "section-end", "section": name}), flush=True)


def _drive():
    """Run each section in its own child, then summarize the exit signals."""
    interpreter = sys.executable
    for name in SECTION_ORDER:
        # A child that aborts must not take the driver with it; the driver
        # reports the child's returncode so a SIGABRT is visible as the value it
        # is, and the sections after it still run.
        done = subprocess.run(
            [interpreter, os.path.abspath(__file__), name],
            text=True,
        )
        print(
            json.dumps(
                {
                    "kind": "section-result",
                    "section": name,
                    "returncode": done.returncode,
                    "signal": -done.returncode if done.returncode < 0 else None,
                }
            ),
            flush=True,
        )


def main():
    if len(sys.argv) > 1 and sys.argv[1] in SECTIONS:
        _run_section(sys.argv[1])
    else:
        _drive()


if __name__ == "__main__":
    main()
