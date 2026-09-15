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

"""Moore Threads MUSA routing tests.

The per-op files in this directory assert a ``-> cuda`` routing that MUSA builds
cannot produce: no CUDA boxing kernels are compiled in (the platform ships no
cudart), so those tests are skipped by conftest's platform gate. This file is the
MUSA equivalent -- it checks that ops land on accelerated backends (FlagGems or
mudnn native), that the per-op env override works, and that the results match a
CPU reference.

Current MUSA strategy prioritizes FlagGems (Triton) implementations where available,
falling back to mudnn native kernels. Ops covered by FlagGems route to
``flagos_python``; ops with only mudnn implementations route to ``musa``, and so
does an op FlagGems cannot compile for this target -- ``add.Tensor`` and friends
on bf16 with a Python-float operand, listed in ``NATIVE_TRITON_GAPS['musa']``.

Usage:
    pytest tests/integration/ops/test_musa_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"
# Second device for the cross-device copy tests; the MTT S5000 exposes 8.
DEVICE_B = "flagos:1"

# op name as it appears in the dispatch log -> (snippet, expected backend)
# Expected backend can be "musa" (mudnn native) or "flagos_python" (FlagGems)
_OPS = {
    "mm": ("a @ b", "flagos_python"),  # FlagGems coverage
    # FlagGems' pointwise promotion cannot serve a bf16 tensor against the
    # float64 0-dim tensor ATen boxes a Python-float operand into: the
    # mthreads LLVM lowering has no double overload for
    # llvm.musa.float2bfloat16, so the kernel fails to compile. mudnn takes
    # the scalar as its Unary alpha instead. Listed in
    # NATIVE_TRITON_GAPS['musa'] so the conf and the registration agree.
    "add.Tensor": ("a + b", "musa"),
    "mul.Tensor": ("a * b", "musa"),  # mudnn native only
    "_softmax": ("torch.softmax(a, -1)", "flagos_python"),  # FlagGems coverage
    "relu": ("torch.relu(a)", "flagos_python"),  # FlagGems coverage
}

# Ops mudnn has no mode for, so they are served by FlagGems rather than the
# vendor backend. They used to be left unregistered to reach the cpu_fallback;
# under the FlagGems-first routing they have a kFlagGems slot and reach it, so
# what this checks is that the route resolves *and* the answer is right.
_FLAGGEMS_ONLY_OPS = {
    "sinh": ("a.sinh()", lambda x: x.sinh()),
    "cosh": ("a.cosh()", lambda x: x.cosh()),
    "asin": ("a.clamp(-1, 1).asin()", lambda x: x.clamp(-1, 1).asin()),
}


def _run_dispatch_subprocess(expr: str, extra_env: dict) -> subprocess.CompletedProcess:
    """Evaluate `expr` over two flagos tensors in a fresh interpreter."""
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        f"a = torch.randn(8, 8, device='{DEVICE}'); "
        f"b = torch.randn(8, 8, device='{DEVICE}'); "
        f"r = {expr}; "
        "torch.flagos.synchronize()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )


class TestMusaDispatch:
    """Ops route to accelerated backends (FlagGems or mudnn) and produce correct results."""

    @pytest.mark.musa
    @pytest.mark.parametrize("op,expr_backend", sorted(_OPS.items()))
    def test_dispatch_log_musa(self, op, expr_backend):
        """Every covered op routes to its configured backend (FlagGems or mudnn)."""
        expr, expected_backend = expr_backend
        result = _run_dispatch_subprocess(expr, {"FLAGOS_LOG_DISPATCH": "1"})
        assert f"[flagos dispatch] {op} -> {expected_backend}" in result.stderr, (
            f"Expected {expected_backend} dispatch for {op}, got:\n{result.stderr}"
        )

    @pytest.mark.musa
    def test_dispatch_log_musa_override(self):
        """FLAGOS_OP_mm=musa pins mm to the musa backend explicitly."""
        result = _run_dispatch_subprocess(
            "a @ b", {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_mm": "musa"}
        )
        assert "[flagos dispatch] mm -> musa" in result.stderr, (
            f"Expected musa dispatch log, got:\n{result.stderr}"
        )

    @pytest.mark.musa
    def test_dispatch_log_mm_out_musa(self):
        """mm.out routes to FlagGems (mudnn MatMul fallback available but FlagGems preferred)."""
        env = os.environ.copy()
        env["FLAGOS_LOG_DISPATCH"] = "1"
        code = (
            "import torch_fl, torch; "
            f"a = torch.randn(8, 8, device='{DEVICE}'); "
            f"b = torch.randn(8, 8, device='{DEVICE}'); "
            f"out = torch.empty(8, 8, device='{DEVICE}'); "
            "torch.mm(a, b, out=out); "
            "torch.flagos.synchronize()"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        assert "[flagos dispatch] mm.out -> flagos_python" in result.stderr, (
            f"Expected flagos_python dispatch log, got:\n{result.stderr}"
        )


class TestMusaEmptyInplace:
    """mudnn Fill must no-op safely for zero-element tensors."""

    @pytest.mark.musa
    @pytest.mark.parametrize("value", [0.0, 3.5])
    def test_empty_fill_stays_on_device(self, value):
        x = torch.empty(4, 0, device=DEVICE)
        out = x.fill_(value)
        assert out is x
        assert out.device.type == "flagos"
        assert out.shape == (4, 0)

    @pytest.mark.musa
    def test_empty_zero_stays_on_device(self):
        x = torch.empty(4, 0, device=DEVICE)
        out = x.zero_()
        assert out is x
        assert out.device.type == "flagos"
        assert out.shape == (4, 0)

    @pytest.mark.musa
    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
    def test_nonempty_fill_regression(self, dtype):
        x = torch.empty(4, device=DEVICE, dtype=dtype)
        x.fill_(3)
        torch.testing.assert_close(x.cpu(), torch.full((4,), 3, dtype=dtype))


class TestMusaCorrectness:
    """mudnn kernels agree with a CPU reference."""

    @pytest.mark.musa
    @pytest.mark.parametrize(
        "fn",
        [
            # matmul
            lambda x, y: x @ y,
            lambda x, y: torch.bmm(x.unsqueeze(0), y.unsqueeze(0)),
            # binary tensor, incl. the alpha and comparison variants
            lambda x, y: x + y,
            lambda x, y: torch.add(x, y, alpha=2.5),
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: x / y,
            lambda x, y: torch.maximum(x, y),
            lambda x, y: torch.minimum(x, y),
            lambda x, y: x > y,
            lambda x, y: x == y,
            # scalar overloads (mudnn takes the scalar as Unary alpha)
            lambda x, y: x * 3.0,
            lambda x, y: x / 3.0,
            lambda x, y: x + 3.0,
            lambda x, y: torch.add(x, 3.0, alpha=2.0),
            lambda x, y: x - 3.0,
            lambda x, y: x > 0.5,
            lambda x, y: x.abs() ** 2.0,
            # unary, including the three composed ones
            lambda x, y: torch.relu(x),
            lambda x, y: torch.sigmoid(x),
            lambda x, y: torch.tanh(x),
            lambda x, y: x.abs(),
            lambda x, y: x.abs().sqrt(),
            lambda x, y: x.abs().log(),
            lambda x, y: -x,
            lambda x, y: x.trunc(),
            lambda x, y: x.expm1(),
            lambda x, y: x.sign(),
            # activations / normalization
            lambda x, y: torch.softmax(x, -1),
            lambda x, y: torch.nn.functional.gelu(x),
            lambda x, y: torch.nn.functional.gelu(x, approximate="tanh"),
            # reductions
            lambda x, y: x.sum(),
            lambda x, y: x.mean(),
            lambda x, y: x.sum(dim=1),
            lambda x, y: x.sum(dim=1, keepdim=True),
            lambda x, y: x.mean(0),
            lambda x, y: x.sum(dim=(0, 1)),
        ],
    )
    def test_matches_cpu(self, fn):
        torch.manual_seed(42)
        a_cpu = torch.randn(64, 64)
        b_cpu = torch.randn(64, 64).abs() + 0.5  # keep div well-conditioned
        a, b = a_cpu.to(DEVICE), b_cpu.to(DEVICE)
        torch.testing.assert_close(
            fn(a, b).cpu(), fn(a_cpu, b_cpu), rtol=1e-4, atol=1e-4
        )

    @pytest.mark.musa
    @pytest.mark.parametrize(
        "fn",
        [
            lambda x, y: x.sum(),
            lambda x, y: x.sum(dim=0),
            lambda x, y: x * y,
            lambda x, y: x + y,
            lambda x, y: -x,
            lambda x, y: x > y,
            lambda x, y: x * 3,
            lambda x, y: x % 3,
        ],
    )
    def test_matches_cpu_int64(self, fn):
        """int64 runs on device here.

        Unlike topsaten (which has no int64 kernels at all, so the GCU backend
        falls back to CPU for them), mudnn handles int64 across
        Unary/Binary/Reduce -- so these must be exact, not approximate.
        """
        a_cpu = torch.arange(-32, 32, dtype=torch.int64).reshape(8, 8)
        b_cpu = torch.arange(1, 65, dtype=torch.int64).reshape(8, 8)
        a, b = a_cpu.to(DEVICE), b_cpu.to(DEVICE)
        assert torch.equal(fn(a, b).cpu(), fn(a_cpu, b_cpu))

    @pytest.mark.musa
    @pytest.mark.parametrize(
        "fn",
        [
            lambda x: x.expand(2, 4, 10, 10).sum([0, 2, 3]),
            lambda x: x.expand(2, 4).sum(),
            lambda x: x.expand(3, 4).sum(0),
            lambda x: x.expand(4, 5).sum(0),
            lambda x: x.expand(4, 5).sum([0], keepdim=True),
            lambda x: x.expand(64, 128).sum(0),
            lambda x: x.expand(2, 4, 10, 10).mean([0, 2, 3]),
        ],
    )
    def test_fully_broadcast_reduce(self, fn):
        """A reduce over a fully 0-strided input must be correct, not just alive.

        mudnn v3300 mishandles a Reduce whose input is a broadcast of one
        element in two ways, so the kernels materialize that case first:
        a multi-dim reduce raises SIGFPE (an uncatchable crash, not a status),
        and a single-dim reduce intermittently writes only out[0], leaving the
        rest of the output as whatever the caching allocator last left there.
        Both reach real code through bias gradients, where autograd feeds
        `ones.expand(...)` straight into the reduction.
        """
        a_cpu = torch.ones(1)
        torch.testing.assert_close(fn(a_cpu.to(DEVICE)).cpu(), fn(a_cpu))

    @pytest.mark.musa
    def test_bias_gradient_matches_cpu(self):
        """`linear(...).sum().backward()` reduces an `ones.expand()` grad.

        This is the path that exposed the single-dim partial write: the bias
        gradient came back as `[N, <stale>, <stale>, ...]` -- element 0 correct
        and the rest left over from an earlier op in the same allocator block.
        A pure-tensor reduce cannot stand in for it, because the bug only shows
        when the output buffer holds recycled non-zero data.
        """
        torch.manual_seed(42)
        x_cpu = torch.randn(4, 3, requires_grad=True)
        w_cpu = torch.randn(5, 3, requires_grad=True)
        b_cpu = torch.randn(5, requires_grad=True)
        x, w, b = (
            t.detach().to(DEVICE).requires_grad_(True) for t in (x_cpu, w_cpu, b_cpu)
        )

        torch.nn.functional.linear(x_cpu, w_cpu, b_cpu).sum().backward()
        torch.nn.functional.linear(x, w, b).sum().backward()

        torch.testing.assert_close(b.grad.cpu(), b_cpu.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(w.grad.cpu(), w_cpu.grad, rtol=1e-4, atol=1e-4)

    @pytest.mark.musa
    @pytest.mark.parametrize("op,expr_fn", sorted(_FLAGGEMS_ONLY_OPS.items()))
    def test_flaggems_only_ops_route_and_stay_correct(self, op, expr_fn):
        """The ops with no mudnn mode land on FlagGems and stay numerically right."""
        expr, fn = expr_fn
        result = _run_dispatch_subprocess(expr, {"FLAGOS_LOG_DISPATCH": "1"})
        assert f"[flagos dispatch] {op} -> flagos_python" in result.stderr, (
            f"Expected flagos_python dispatch for {op}, got:\n{result.stderr}"
        )

        torch.manual_seed(42)
        a_cpu = torch.randn(16, 8)
        torch.testing.assert_close(
            fn(a_cpu.to(DEVICE)).cpu(), fn(a_cpu), rtol=1e-4, atol=1e-4
        )

    @pytest.mark.musa
    @pytest.mark.parametrize(
        "fn",
        [
            lambda x, y: x.expand(3, *x.shape).sum(0),
            lambda x, y: x + y[0],  # row broadcast
            lambda x, y: x + y[:, :1],  # column broadcast
        ],
    )
    def test_broadcast_matches_cpu(self, fn):
        """Broadcasting is expressed with 0-strides, which mudnn reads directly.

        No `.contiguous()` materialization happens on this path, so a wrong
        stride would show up as wrong values rather than a slow-but-right answer.
        """
        torch.manual_seed(42)
        a_cpu = torch.randn(8, 8)
        b_cpu = torch.randn(8, 8)
        torch.testing.assert_close(
            fn(a_cpu.to(DEVICE), b_cpu.to(DEVICE)).cpu(),
            fn(a_cpu, b_cpu),
            rtol=1e-4,
            atol=1e-4,
        )

    @pytest.mark.musa
    @pytest.mark.parametrize(
        "fn",
        [
            lambda x: x.t().contiguous(),
            lambda x: x.t().clone(),
            lambda x: x[::2].contiguous(),
            lambda x: x[:, 1:5].contiguous(),
            lambda x: x.to(torch.float16).float(),
            lambda x: x.to(torch.int32).to(torch.int64),
        ],
    )
    def test_strided_and_cast_copies(self, fn):
        """mudnn's Unary IDENTITY/CAST handle strides and dtype casts on device.

        These go through MudnnCopy rather than a generated kernel, and are the
        paths that would otherwise reach the CUDA DispatchStub and fail.
        """
        torch.manual_seed(42)
        a_cpu = torch.randn(16, 8)
        torch.testing.assert_close(
            fn(a_cpu.to(DEVICE)).cpu(), fn(a_cpu), rtol=1e-3, atol=1e-3
        )


# Each entry is (name, body, expected device, expected dtype) for
# _run_cross_device_probe. `body` must leave the copy in `r` and a CPU
# expectation for it in `ref`. The two devices hold the same values, so a copy
# that silently reads the wrong device's memory still lands in range and has to
# be caught by the value comparison rather than by the device check.
_CROSS_DEVICE_CASES = [
    (
        "to_second_device_with_cast",
        "r = a.to('flagos:1', torch.int32)\nref = a_cpu.to(torch.int32)",
        "flagos:1",
        torch.int32,
    ),
    (
        "to_first_device_with_cast",
        "r = b.to('flagos:0', torch.int32)\nref = b_cpu.to(torch.int32)",
        "flagos:0",
        torch.int32,
    ),
    (
        "to_second_device_strided",
        "r = a.t().to('flagos:1')\nref = a_cpu.t()",
        "flagos:1",
        torch.float32,
    ),
    # The shape transformers hits: a boolean mask cast to int32 on the device
    # that owns the logits (modeling_layers.py, `(input_ids != pad_token_id)
    # .to(logits.device, torch.int32)`).
    (
        "bool_mask_to_second_device",
        "r = (a > 7).to('flagos:1', torch.int32)\nref = (a_cpu > 7).to(torch.int32)",
        "flagos:1",
        torch.int32,
    ),
    (
        "copy_into_second_device_with_cast",
        "r = torch.empty(4, 4, dtype=torch.int32, device='flagos:1')\n"
        "r.copy_(a)\nref = a_cpu.to(torch.int32)",
        "flagos:1",
        torch.int32,
    ),
    (
        "copy_into_second_device_strided",
        "r = torch.empty(4, 4, device='flagos:1')\nr.copy_(a.t())\nref = a_cpu.t()",
        "flagos:1",
        torch.float32,
    ),
]


def _require_second_device() -> None:
    """Skip when only one MUSA device is visible.

    The marker file is authoritative for the *platform*, not for how many
    devices this host exposes; a single-device MUSA machine has nothing to copy
    between.
    """
    if torch_fl.flagos.device_count() < 2:
        pytest.skip("cross-device copies need a second MUSA device")


def _run_cross_device_probe(body: str) -> subprocess.CompletedProcess:
    """Run `body` over two devices in a fresh interpreter, then a canary.

    The canary re-runs an ordinary kernel on each device afterwards: a
    cross-device copy that faults poisons the whole device context, so the copy
    itself may raise while the damage shows up as every *later* op failing.
    """
    code = (
        "import torch, torch_fl\n"
        f"a = torch.arange(16, dtype=torch.float32, device='{DEVICE}').reshape(4, 4)\n"
        f"b = torch.arange(16, dtype=torch.float32, device='{DEVICE_B}').reshape(4, 4)\n"
        "a_cpu, b_cpu = a.cpu(), b.cpu()\n"
        f"{body}\n"
        "torch.testing.assert_close(r.cpu(), ref)\n"
        "print('COPY', r.device, r.dtype)\n"
        f"c0 = (torch.ones(4, device='{DEVICE}') + 1).sum().item()\n"
        f"c1 = (torch.ones(4, device='{DEVICE_B}') + 1).sum().item()\n"
        "print('CANARY', c0, c1)\n"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


# The expression transformers actually reaches (modeling_layers.py:167):
# `input_ids` and `logits` land on different devices under device_map="auto",
# and the bool pad mask is cast to int32 across them. Crossing devices *and*
# changing dtype is the mudnn path in StageSourceOnDstDevice.
_ORDERING_STEP = f"((x > 0).to('{DEVICE_B}', torch.int32))"
_ORDERING_REF = "(x_cpu > 0).to(torch.int32)"

# Iterations for the ordering probe, and the tensor size they run at.
#
# The hazard is a race, not a deterministic wrong answer, so a single execution
# of the copy -- what test_cross_device_copy does -- mostly passes on an unfixed
# build. Measured on MTT S5000 at 4 Mi elements, against a control build with
# only the two peer-device barriers removed and everything else identical: 7
# mismatches in 800 calls over four runs (3/2/1/1), every run failing; 0 in 600
# calls over three runs with the barriers in place. That is ~1% per call, so the
# 200 below fail an unfixed build most of the time but not always: a green run
# is weak evidence the barrier is present, and that rate is what the guard is
# worth.
#
# The same-dtype cross-device copy (copy_ops.cc, CrossDeviceMemcpy) is
# measurably rarer -- 2 mismatches in 20000 calls -- so sampling it here would
# need a run an order of magnitude longer for the same power. It is covered for
# correctness by test_cross_device_copy above; its race is not gated here.
_ORDERING_ITERS = 200
_ORDERING_NUMEL = 1 << 22


def _run_ordering_probe(step: str, ref: str, iters: int) -> subprocess.CompletedProcess:
    """Run `step` `iters` times across devices and report the mismatch count.

    The loop alternates between two sources with different contents. That
    matters for detection: the caching allocator hands the next copy the block
    the previous one just filled, so re-running one source would leave
    stale-but-*correct* results in the recycled block, and a copy that never
    landed would read them and pass. Alternating means the stale block holds
    the other source's values, which the comparison catches.

    A mismatch is reported as (iteration, first differing index, got, want);
    the first six elements are the same in the observed failures, so the index
    is what identifies one.
    """
    code = f"""
import torch, torch_fl

srcs = [
    torch.arange({_ORDERING_NUMEL}, dtype=torch.int64, device='{DEVICE}'),
    torch.arange({_ORDERING_NUMEL}, dtype=torch.int64, device='{DEVICE}') + 1,
]
refs = []
for x in srcs:
    x_cpu = x.cpu()
    refs.append({ref})

bad = 0
first = None
for i in range({iters}):
    x = srcs[i % 2]
    r = {step}
    want = refs[i % 2]
    got = r.cpu()
    try:
        torch.testing.assert_close(got, want)
    except AssertionError:
        bad += 1
        if first is None:
            j = int((got != want).nonzero()[0])
            first = (i, j, got.flatten()[j].item(), want.flatten()[j].item())
print('ORDERING', bad, {iters}, first)
"""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


class TestMusaCrossDeviceCopy:
    """Copying between two MUSA devices must not poison the device context.

    mudnn runs a whole op on one device -- handle, stream and every operand
    address resolve against whatever device is current -- so a source and a
    destination on different devices cannot be one Unary::Run. Handed both
    anyway, mudnn dereferences the foreign pointer, the device raises an illegal
    memory access, and the context stays poisoned for the rest of the process
    (issue #250). Issue #265 is the transformers test that reaches it:
    `device_map="auto"` splits a model over devices 0 and 1, so `input_ids` and
    `logits` disagree on device and the dtype cast between them crosses devices.

    These run in subprocesses because the failure mode is a dead device, not a
    raised assertion: run in-process on an unfixed build, the first case would
    poison the context and the *following* tests in this file would be the ones
    reporting failures.

    The cases above assert the answer once, which is the right shape for a
    deterministic wrong answer. A cross-device copy has a second, separate
    hazard that is not deterministic -- it is ordered only against the queue of
    the device it is issued on, so the destination can be read before the
    transfer lands (issue #281) -- and that one is covered by the repeating
    probe below.
    """

    @pytest.mark.musa
    @pytest.mark.parametrize(
        "name,body,want_device,want_dtype",
        _CROSS_DEVICE_CASES,
        ids=[case[0] for case in _CROSS_DEVICE_CASES],
    )
    def test_cross_device_copy(self, name, body, want_device, want_dtype):
        _require_second_device()
        result = _run_cross_device_probe(body)
        assert result.returncode == 0, f"copy failed:\n{result.stdout}{result.stderr}"
        assert f"COPY {want_device} {want_dtype}" in result.stdout, result.stdout
        assert "CANARY 8.0 8.0" in result.stdout, (
            f"device context was poisoned by the copy:\n{result.stdout}{result.stderr}"
        )
        assert "illegal memory access" not in result.stderr, result.stderr

    @pytest.mark.musa
    def test_cross_device_copy_does_not_wedge_the_allocator(self):
        """A cross-device copy must still free its staging buffer cleanly.

        The reported banner is `musaFree(...) failed` at teardown rather than
        the copy itself, so assert on the allocator's own diagnostics: the
        failing run leaves them on stderr even when every op appears to succeed.
        """
        _require_second_device()
        result = _run_cross_device_probe(
            "r = a.to('flagos:1', torch.int32)\nref = a_cpu.to(torch.int32)"
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos-musa]" not in result.stderr, result.stderr

    @pytest.mark.musa
    def test_cross_device_copy_lands_before_it_is_read(self):
        """A copy across devices must be visible to the receiving device.

        A blocking musaMemcpy is ordered only against the device that is
        current when it is issued; the receiving device's queue is not ordered
        against it, so a consumer there can read the destination before the
        transfer lands. Draining either device's *default stream* does not
        close it -- only a device-wide sync of the issuing device does.

        Probabilistic by construction: this is a sampled guard, not a proof.
        See _ORDERING_ITERS for what it is worth on an unfixed build.
        """
        _require_second_device()
        result = _run_ordering_probe(_ORDERING_STEP, _ORDERING_REF, _ORDERING_ITERS)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"ORDERING 0 {_ORDERING_ITERS} " in result.stdout, (
            "a copy that crossed devices was read before it landed:\n"
            f"{result.stdout}{result.stderr}"
        )


class TestMusaConvolution:
    """mudnn's Convolution class, via the handwritten mudnn_conv.cc kernels.

    ``convolution_overrideable`` is the one op that cannot be left unregistered:
    ATen's default for it is a raising ``TORCH_CHECK``, not something the
    cpu_fallback can box. mudnn covers 2 spatial dims only, so conv1d runs as a
    2D conv with a unit H dim and conv3d takes the CPU fallback.
    """

    @pytest.mark.musa
    @pytest.mark.parametrize(
        "shape,wshape,kwargs",
        [
            # conv1d -- run as 2D with a synthetic H=1 dim
            ((1, 8, 32), (16, 8, 4), {"padding": 3}),
            ((2, 4, 16), (4, 1, 3), {"padding": 1, "groups": 4}),  # depthwise
            ((1, 4, 20), (8, 4, 5), {"stride": 2, "dilation": 2}),
            # conv2d -- mudnn's native case
            ((2, 3, 8, 8), (4, 3, 3, 3), {"padding": 1}),
            ((1, 4, 12, 12), (6, 2, 3, 3), {"padding": 1, "groups": 2}),
            ((1, 4, 12, 12), (6, 4, 3, 3), {"stride": 2, "dilation": 2}),
        ],
    )
    @pytest.mark.parametrize("bias", [True, False])
    def test_conv_matches_cpu(self, shape, wshape, kwargs, bias):
        conv = (
            torch.nn.functional.conv1d
            if len(shape) == 3
            else torch.nn.functional.conv2d
        )
        torch.manual_seed(0)
        x = torch.randn(*shape)
        w = torch.randn(*wshape)
        b = torch.randn(wshape[0]) if bias else None
        ref = conv(x, w, b, **kwargs)
        got = conv(x.to(DEVICE), w.to(DEVICE), b.to(DEVICE) if bias else None, **kwargs)
        torch.testing.assert_close(got.cpu(), ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.musa
    def test_conv_backward_matches_cpu(self):
        """grad_input / grad_weight come from RunBwdData / RunBwdFilter."""
        torch.manual_seed(0)
        x_cpu = torch.randn(2, 3, 10, 10, requires_grad=True)
        w_cpu = torch.randn(4, 3, 3, 3, requires_grad=True)
        b_cpu = torch.randn(4, requires_grad=True)
        x = x_cpu.detach().to(DEVICE).requires_grad_(True)
        w = w_cpu.detach().to(DEVICE).requires_grad_(True)
        b = b_cpu.detach().to(DEVICE).requires_grad_(True)

        torch.nn.functional.conv2d(x, w, b, padding=1).sum().backward()
        torch.nn.functional.conv2d(x_cpu, w_cpu, b_cpu, padding=1).sum().backward()

        for got, ref in (
            (x.grad, x_cpu.grad),
            (w.grad, w_cpu.grad),
            (b.grad, b_cpu.grad),
        ):
            torch.testing.assert_close(got.cpu(), ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.musa
    def test_grouped_conv_backward_matches_cpu(self):
        """Grouped conv is where the recommended algorithm is unusable.

        GetRecommendForwardAlgorithm names DIRECT for this config, and DIRECT
        then rejects the Run, so the kernel tries the other algorithms in turn.
        """
        torch.manual_seed(3)
        x_cpu = torch.randn(4, 8, 12, 12, requires_grad=True)
        w_cpu = torch.randn(4, 2, 3, 3, requires_grad=True)
        x = x_cpu.detach().to(DEVICE).requires_grad_(True)
        w = w_cpu.detach().to(DEVICE).requires_grad_(True)

        torch.nn.functional.conv2d(x, w, padding=1, groups=4).sum().backward()
        torch.nn.functional.conv2d(x_cpu, w_cpu, padding=1, groups=4).sum().backward()

        torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(w.grad.cpu(), w_cpu.grad, rtol=1e-3, atol=1e-3)

    @pytest.mark.musa
    def test_conv3d_falls_back_to_cpu(self):
        """3 spatial dims: mudnn reports "Unexpected tensor format NCHW"."""
        torch.manual_seed(0)
        x = torch.randn(1, 2, 4, 4, 4)
        w = torch.randn(3, 2, 2, 2, 2)
        torch.testing.assert_close(
            torch.nn.functional.conv3d(x.to(DEVICE), w.to(DEVICE)).cpu(),
            torch.nn.functional.conv3d(x, w),
            rtol=1e-3,
            atol=1e-3,
        )


class TestMusaMixedDeviceOperandOrder:
    """A wrapped-scalar CPU operand may land in either the `self` or `other`
    slot of a binary op's Tensor overload (issue #238).

    `rsub.Scalar(self, other, alpha)` decomposes to
    `sub.Tensor(wrapped_scalar_tensor(other), self, alpha)`, so the CPU
    scalar ends up as `self` and the device tensor as `other` -- the reverse
    of the ordinary `tensor - scalar` call, where the device tensor is
    `self`. The binary mudnn kernels must produce a device result and the
    correct value in both orderings, not just the ordinary one.
    """

    @pytest.mark.musa
    def test_rsub_scalar(self):
        u = torch.ones(4, dtype=torch.long, device=DEVICE)
        out = torch.rsub(u, 1)
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), torch.zeros(4, dtype=torch.long))

    @pytest.mark.musa
    def test_sub_tensor_cpu_self_device_other(self):
        u = torch.ones(4, device=DEVICE)
        out = torch.sub(torch.tensor(1.0), u)
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), torch.zeros(4))

    @pytest.mark.musa
    @pytest.mark.parametrize(
        "fn",
        [
            lambda scalar, t: scalar - t,
            lambda scalar, t: scalar + t,
            lambda scalar, t: scalar * t,
            lambda scalar, t: torch.maximum(scalar, t),
            lambda scalar, t: scalar > t,
            lambda scalar, t: scalar == t,
        ],
    )
    def test_binary_ops_with_cpu_scalar_self(self, fn):
        """Sweep the binary op family with a CPU-origin `self` operand."""
        t_cpu = torch.arange(1, 5, dtype=torch.float32)
        t = t_cpu.to(DEVICE)
        scalar = torch.tensor(2.0)
        out = fn(scalar, t)
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), fn(scalar, t_cpu))


class TestMusaDegenerateStride:
    """A transpose of a size-1 dimension is still `is_contiguous() == True`
    (issue #240): PyTorch's contiguity check ignores the stride of any
    size-1 dim, since no read ever depends on it, so `.contiguous()` is a
    no-op and the degenerate stride reaches mudnn unchanged. mudnn's own
    validation is stricter and rejects it (`MatMul` raised `INVALID_PARAMETER,
    lda 1`); `MudnnTensorWrapper` must rewrite it before handing tensors to
    any mudnn op, not just `mm`.
    """

    @pytest.mark.musa
    def test_mm_transposed_size_one_dim(self):
        # The degenerate stride only survives a transpose done *after* the
        # tensor is already on-device -- moving a CPU tensor there via
        # `.to()` renormalizes the stride, silently missing the bug.
        a = torch.randn(2, 1, device=DEVICE)
        b = torch.randn(2, 32, device=DEVICE)
        at = a.t()
        assert at.is_contiguous()
        out = torch.mm(at, b)
        torch.testing.assert_close(out.cpu(), torch.mm(a.cpu().t(), b.cpu()))

    @pytest.mark.musa
    def test_bmm_transposed_size_one_dim(self):
        a = torch.randn(3, 2, 1, device=DEVICE)
        b = torch.randn(3, 2, 32, device=DEVICE)
        at = a.transpose(1, 2)
        assert at.is_contiguous()
        out = torch.bmm(at, b)
        torch.testing.assert_close(
            out.cpu(), torch.bmm(a.cpu().transpose(1, 2), b.cpu())
        )

    @pytest.mark.musa
    def test_bert_regression_head_backward(self):
        """BertForSequenceClassification with num_labels=1 (regression via
        MSE loss) produces exactly this shape in the final linear layer's
        weight gradient during backward."""
        transformers = pytest.importorskip("transformers")
        torch.manual_seed(0)
        config = transformers.BertConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=128,
            num_labels=1,
        )
        model = transformers.BertForSequenceClassification(config).to(DEVICE)
        input_ids = torch.randint(0, config.vocab_size, (2, 8), device=DEVICE)
        labels = torch.randn(2, 1, device=DEVICE)
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        assert loss.device.type == "flagos"


class TestMusaAutograd:
    """Autograd works through the mudnn kernels.

    The AutogradPrivateUse1 fallthrough used to be skipped on MUSA, because
    libmusa_python.so registered its own and the dispatcher hard-errors on a
    second one. With the vendor library gone, flagos registers it like every
    other platform -- these tests cover that.
    """

    @pytest.mark.musa
    def test_backward_matches_cpu(self):
        torch.manual_seed(42)
        a_cpu = torch.randn(16, 16, requires_grad=True)
        b_cpu = torch.randn(16, 16, requires_grad=True)
        a = a_cpu.detach().to(DEVICE).requires_grad_(True)
        b = b_cpu.detach().to(DEVICE).requires_grad_(True)

        ((a @ b) * 2.0 + a).sum().backward()
        ((a_cpu @ b_cpu) * 2.0 + a_cpu).sum().backward()

        torch.testing.assert_close(a.grad.cpu(), a_cpu.grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(b.grad.cpu(), b_cpu.grad, rtol=1e-4, atol=1e-4)

    @pytest.mark.musa
    def test_module_to_device_preserves_grad(self):
        """nn.Module.to() goes through set_data / swap_tensors."""
        torch.manual_seed(42)
        model = torch.nn.Linear(16, 8)
        model.to(DEVICE)
        assert model.weight.device.type == torch._C._get_privateuse1_backend_name()
        assert model.weight.requires_grad

        x = torch.randn(4, 16, device=DEVICE)
        model(x).sum().backward()
        assert model.weight.grad is not None
        assert model.weight.grad.device.type == torch._C._get_privateuse1_backend_name()
