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

"""
mul.Tensor dispatch tests

Verifies that torch.mul (Tensor variant):
  - produces correct results on flagos device
  - produces correct results for a Python-number operand (`a * 2.0`), which
    arrives as a wrapped scalar tensor
  - C++ wrapper routes to the backend the platform conf lists for the .Tensor
    overload (the vendor kernel on every current platform)
  - explicit overrides can still route to flaggems_python
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_mul_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import routed_backend


DEVICE = "flagos:0"


def _run_mul_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,4,device='flagos:0'); "
        "b = torch.randn(4,4,device='flagos:0'); "
        "torch.mul(a, b)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestMulTensorCorrectness:
    """torch.mul correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    @pytest.mark.anyplatform
    def test_mul_shape(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE)
        b = torch.randn(*shape, device=DEVICE)
        out = torch.mul(a, b)
        assert out.shape == shape
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_mul_broadcast(self):
        torch.manual_seed(1)
        a = torch.randn(4, 8, device=DEVICE)
        b = torch.randn(8, device=DEVICE)
        out = torch.mul(a, b)
        ref = a.cpu() * b.cpu()
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.cuda
    def test_mul_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(2)
        a_cuda = torch.randn(64, 64, device="cuda:0")
        b_cuda = torch.randn(64, 64, device="cuda:0")
        ref = torch.mul(a_cuda, b_cuda)
        a = a_cuda.to(DEVICE)
        b = b_cuda.to(DEVICE)
        out = torch.mul(a, b)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)


class TestMulPythonNumberOperand:
    """The `a * 2.0` spelling, whose operand is a wrapped scalar tensor.

    ``torch.mul(a, 2.0)`` dispatches to ``mul.Tensor`` with a 0-dim tensor typed
    from the number -- float64 for ``2.0``, int64 for ``2`` -- and flagged
    ``is_wrapped_number``. Promotion deliberately stops that from widening ``a``,
    so a kernel that tests the operand's own dtype instead of ``at::result_type``
    sends the whole call to the host and back for the most ordinary spelling
    there is. The dtype is asserted as well as the values, because the wrong
    result type is the failure mode that reads as correct.
    """

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("operand", [2.0, 2, 1.5, -3])
    def test_python_number_matches_cpu(self, operand):
        torch.manual_seed(0)
        a = torch.randn(64, 64, device=DEVICE)
        out = a * operand
        ref = a.cpu() * operand
        assert out.dtype == ref.dtype, f"device {out.dtype} vs cpu {ref.dtype}"
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.anyplatform
    def test_integral_operand_keeps_the_integral_dtype(self):
        """`int32 * 2` stays int32; the python int must not widen it."""
        a = torch.arange(16, dtype=torch.int32, device=DEVICE)
        out = a * 2
        assert out.dtype == torch.int32
        torch.testing.assert_close(out.cpu(), a.cpu() * 2)

    @pytest.mark.anyplatform
    def test_zero_dim_device_operand_matches_python_number(self):
        """A 0-dim tensor on the card takes the same path as the number does."""
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE)
        two = torch.tensor(2.0, device=DEVICE)
        torch.testing.assert_close((a * two).cpu(), (a * 2.0).cpu(), rtol=0, atol=0)


class TestMulTensorDispatch:
    """Verify dispatch routing for mul.Tensor op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_mul_subprocess(
            {
                "FLAGOS_LOG": "dispatch",
                "FLAGOS_OP_mul__Tensor": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] mul.Tensor -> flagos_python" in result.stderr

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    def test_dispatch_log_flaggems_runtime(self):
        """mul.Tensor dispatches to whatever backend this platform's conf lists.

        ``flagos_python`` on CUDA/DCU/PPU and ``musa`` on MUSA: FlagGems has no
        Triton kernel for the .Tensor overload, so each platform's generated conf
        keeps it on its own vendor kernel. Reading the conf keeps the assertion
        true on all of them.
        """
        result = _run_mul_subprocess({"FLAGOS_LOG": "dispatch"})
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        expected = routed_backend("mul.Tensor")
        assert f"[flagos dispatch] mul.Tensor -> {expected}" in result.stderr

    @pytest.mark.cuda
    @pytest.mark.main_ops
    def test_dispatch_log_cuda_override(self):
        result = _run_mul_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_mul__Tensor": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] mul.Tensor -> cuda" in result.stderr


class TestMulTensorAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify mul.Tensor on ascend backend matches CPU reference."""
        result = _run_mul_subprocess({"FLAGOS_OP_mul__Tensor": "ascend"})
        assert result.returncode == 0


def _decompose(a, b):
    """The real-view product, spelled out on the host."""
    ar, ai = a.real.contiguous(), a.imag.contiguous()
    br, bi = b.real.contiguous(), b.imag.contiguous()
    return torch.complex(ar * br - ai * bi, ar * bi + ai * br)


def _run_complex64_subprocess(transposed: bool) -> subprocess.CompletedProcess:
    """A complex64 multiply in a fresh interpreter, with the dispatch log on.

    The log is the oracle because a Python ``TorchDispatchMode`` cannot be one
    here: a routed GCU kernel re-enters the dispatcher below the mode, so the
    nested ``at::view_as_real`` the generated kernel takes never reaches
    ``__torch_dispatch__`` -- only the outermost ``mul.Tensor`` does.

    The script prints nothing, so the only ``view_as_real`` lines in the log are
    the ones the kernel took: reading ``out.real`` would add one of its own.
    """
    env = os.environ.copy()
    env["FLAGOS_LOG"] = "dispatch"
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(64, 64, dtype=torch.complex64, device='flagos:0'); "
        "b = torch.randn(64, 64, dtype=torch.complex64, device='flagos:0'); "
        + ("a = a.t(); " if transposed else "")
        + "out = torch.mul(a, b); "
        "assert out.shape == (64, 64) and out.dtype == torch.complex64; "
        "ref = a.cpu() * b.cpu(); "
        "assert torch.allclose(out.cpu().float(), ref.float(), rtol=1e-5, atol=1e-5), "
        "'the device product differs from the host product'"
    )
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )


def _log_lines(result: subprocess.CompletedProcess, op: str) -> int:
    return result.stderr.count(f"[flagos dispatch] {op} -> ")


class TestMulTensorComplex64:
    """A complex64 multiply is decomposed into real multiplies before the vendor
    op sees it.

    ``topsatenMul`` accepts a complex64 pair and computes it on the device, but at
    a cost that is not the kernel's: the promotion and the argument marshalling
    behind ``topsaten_common.h`` run on the host, so a complex64 multiply of two
    ``(1, 4096, 24, 64)`` operands was measured at 50.166 ms against 2.27 ms for
    the two real multiplies that carry the same arithmetic. The generated kernel
    therefore takes ``view_as_real`` of both operands and rebuilds the product
    from four real multiplies, which is bit-identical to what it replaced -- and
    only when the promoted result is ``complex64`` and both operands are
    contiguous after the cast, since the decomposition is a view and cannot be
    taken of a strided operand.

    The values are asserted because the decomposition is where a sign error would
    show up, and the *route* is asserted because the values alone cannot tell the
    two apart: the path this replaced also computes a correct complex64 product.
    The route is read from the dispatch log in a subprocess, because a Python
    ``TorchDispatchMode`` only ever sees the outermost ``mul.Tensor``.
    """

    @pytest.fixture(autouse=True)
    def _gcu_only(self):
        from torch_fl._build_config import ACCELERATOR

        if ACCELERATOR != "gcu":
            pytest.skip("GCU build required")

    @pytest.mark.parametrize(
        "shape",
        [(1, 4096, 24, 64), (64, 64), (1, 8, 1, 32), (3, 5, 7)],
    )
    def test_complex64_matches_cpu(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, dtype=torch.complex64, device=DEVICE)
        b = torch.randn(*shape, dtype=torch.complex64, device=DEVICE)
        out = torch.mul(a, b)
        assert out.dtype == torch.complex64
        torch.testing.assert_close(out.cpu(), a.cpu() * b.cpu(), rtol=1e-5, atol=1e-5)

    def test_complex64_broadcast_matches_cpu(self):
        torch.manual_seed(1)
        a = torch.randn(4, 8, dtype=torch.complex64, device=DEVICE)
        b = torch.randn(8, dtype=torch.complex64, device=DEVICE)
        out = torch.mul(a, b)
        assert out.shape == (4, 8)
        torch.testing.assert_close(out.cpu(), a.cpu() * b.cpu(), rtol=1e-5, atol=1e-5)

    def test_real_times_complex64_promotes_and_matches_cpu(self):
        """A mixed pair promotes to complex64 and takes the same branch."""
        torch.manual_seed(2)
        a = torch.randn(64, 64, device=DEVICE)
        b = torch.randn(64, 64, dtype=torch.complex64, device=DEVICE)
        out = torch.mul(a, b)
        assert out.dtype == torch.complex64
        torch.testing.assert_close(out.cpu(), a.cpu() * b.cpu(), rtol=1e-5, atol=1e-5)

    def test_complex64_is_bit_identical_to_the_real_view_product(self):
        """The branch's own arithmetic, spelled out on the host."""
        torch.manual_seed(3)
        a = torch.randn(1, 4096, 24, 64, dtype=torch.complex64, device=DEVICE)
        b = torch.randn(1, 4096, 24, 64, dtype=torch.complex64, device=DEVICE)
        out = torch.mul(a, b).cpu()
        ref = _decompose(a.cpu(), b.cpu())
        assert torch.equal(out, ref), (
            "the real-view decomposition is not the product that was computed: "
            f"max|d| {(out - ref).abs().max().item():.3e}"
        )

    def test_complex64_takes_the_real_views(self):
        """Route: the decomposition is visible in the dispatch log.

        Two ``view_as_real`` for the operands and one ``view_as_complex`` to
        rebuild the product; the native multiply this replaced issues none of
        them, so their presence is what pins the branch.
        """
        result = _run_complex64_subprocess(transposed=False)
        assert result.returncode == 0, f"the multiply failed:\n{result.stderr}"
        assert _log_lines(result, "mul.Tensor") >= 5, (
            f"the multiply did not run on the card:\n{result.stderr}"
        )
        assert _log_lines(result, "view_as_real") == 2, (
            "the decomposition did not view both operands as real: "
            f"{_log_lines(result, 'view_as_real')} views\n{result.stderr}"
        )
        assert _log_lines(result, "view_as_complex") == 1, (
            f"the product was not rebuilt as a complex value:\n{result.stderr}"
        )

    def test_a_transposed_complex64_operand_declines_the_decomposition(self):
        """A strided operand cannot be viewed as real, so the branch stands down.

        The values must still be right -- this only pins that the branch's
        contiguity test is what it says it is. The ``mul.Tensor`` line is
        asserted too, so that the absence of the views means the branch
        declined rather than that nothing ran.
        """
        result = _run_complex64_subprocess(transposed=True)
        assert result.returncode == 0, f"the multiply failed:\n{result.stderr}"
        assert _log_lines(result, "mul.Tensor") >= 1, (
            f"the multiply did not run on the card:\n{result.stderr}"
        )
        assert _log_lines(result, "view_as_real") == 0, (
            f"a strided operand was viewed as real:\n{result.stderr}"
        )
        assert _log_lines(result, "view_as_complex") == 0, (
            f"a strided operand was rebuilt as a complex value:\n{result.stderr}"
        )
