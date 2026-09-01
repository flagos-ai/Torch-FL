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
MUSA equivalent -- it checks that ops land on the ``musa`` backend, that the
per-op env override works, and that the results match a CPU reference.

The kernels here call mudnn (the vendor kernel library) directly, so "routes to
musa" and "runs the vendor kernel" are the same statement.

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

# op name as it appears in the dispatch log -> snippet exercising it
_OPS = {
    "mm": "a @ b",
    "add.Tensor": "a + b",
    "mul.Tensor": "a * b",
    "_softmax": "torch.softmax(a, -1)",
    "relu": "torch.relu(a)",
}

# Ops in the coverage set that mudnn has no mode for. They are deliberately left
# unregistered so they reach the cpu_fallback -- registering an op with no kernel
# behind it would instead trip the dispatcher's "backend not registered" check.
_CPU_FALLBACK_OPS = {
    "sinh": lambda x: x.sinh(),
    "cosh": lambda x: x.cosh(),
    "asin": lambda x: x.clamp(-1, 1).asin(),
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
    """Ops route to the musa backend and produce correct results."""

    @pytest.mark.musa
    @pytest.mark.parametrize("op,expr", sorted(_OPS.items()))
    def test_dispatch_log_musa(self, op, expr):
        """Every covered op reports `-> musa` in the dispatch log by default."""
        result = _run_dispatch_subprocess(expr, {"FLAGOS_LOG_DISPATCH": "1"})
        assert f"[flagos dispatch] {op} -> musa" in result.stderr, (
            f"Expected musa dispatch log for {op}, got:\n{result.stderr}"
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
        """mm.out routes to musa too (mudnn MatMul into a caller-provided out)."""
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
        assert "[flagos dispatch] mm.out -> musa" in result.stderr, (
            f"Expected musa dispatch log, got:\n{result.stderr}"
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
    @pytest.mark.parametrize("op,fn", sorted(_CPU_FALLBACK_OPS.items()))
    def test_cpu_fallback_ops_still_correct(self, op, fn):
        """The ops with no mudnn mode stay correct via the cpu_fallback."""
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
