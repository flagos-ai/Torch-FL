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
add.Tensor dispatch tests

Verifies that torch.add (Tensor variant):
  - produces correct results on flagos device
  - C++ wrapper routes to the backend the platform conf lists (FlagGems-
    first, i.e. flaggems_python, wherever FlagGems covers the op)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_add_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import (
    _OVERRIDE_UNSERVED,
    override_is_logged,
    route_is_served,
    routed_backend,
    routed_backend_or_none,
)


DEVICE = "flagos:0"


def _run_add_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,4,device='flagos:0'); "
        "b = torch.randn(4,4,device='flagos:0'); "
        "torch.add(a, b)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        # The child binds the same flagos device this process already holds. On
        # a runtime that allows one process per device (TsingMicro's TX) it
        # never gets it, so without a timeout the suite hangs instead of
        # reporting a failure. test_cat_dispatch.py already carried this.
        timeout=60,
    )


class TestAddTensorCorrectness:
    """torch.add correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    @pytest.mark.anyplatform
    def test_add_shape(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE)
        b = torch.randn(*shape, device=DEVICE)
        out = torch.add(a, b)
        assert out.shape == shape
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_add_alpha(self):
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE)
        b = torch.randn(32, 32, device=DEVICE)
        out = torch.add(a, b, alpha=2.0)
        ref = a.cpu() + 2.0 * b.cpu()
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.anyplatform
    def test_add_broadcast(self):
        torch.manual_seed(2)
        a = torch.randn(4, 8, device=DEVICE)
        b = torch.randn(8, device=DEVICE)
        out = torch.add(a, b)
        ref = a.cpu() + b.cpu()
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.anyplatform
    def test_add_out_empty_destination(self):
        """`torch.add(a, b, out=empty)` writes its destination, never reads it.

        add.out is a structured op whose destination may arrive empty. A kernel
        that adds into a destination it assumes already holds one addend answers
        whatever the allocator left there: measured on u180, garbage plus self.
        """
        torch.manual_seed(4)
        a = torch.randn(16, 32, device=DEVICE)
        b = torch.randn(32, device=DEVICE)
        out = torch.empty(16, 32, device=DEVICE)
        result = torch.add(a, b, out=out)
        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(out.cpu(), a.cpu() + b.cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.anyplatform
    def test_add_broadcast_out_winning_size(self):
        """An out= destination of the broadcast shape is written in place."""
        torch.manual_seed(5)
        a = torch.randn(8, 1, device=DEVICE)
        b = torch.randn(1, 16, device=DEVICE)
        out = torch.empty(8, 16, device=DEVICE)
        result = torch.add(a, b, out=out)
        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(out.cpu(), a.cpu() + b.cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.anyplatform
    def test_add_inplace_broadcast_to_64x64(self):
        """`a += b` with a (64, 1) destination and a (1, 64) addend.

        This is the frame-plus-bias in-place add. The destination repeats its
        addend, which one linear axpy over the destination cannot express: on
        u180 the vendor path segfaulted on exactly this pair, while the
        out-of-place `torch.add(a, b)` on the same operands answered.
        """
        torch.manual_seed(6)
        a = torch.randn(64, 1, device=DEVICE)
        b = torch.randn(1, 64, device=DEVICE)
        ref = a.cpu() + b.cpu()
        a += b
        torch.testing.assert_close(a.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.anyplatform
    def test_add_inplace_broadcast_to_64x1(self):
        """`a.add_(b)` on a (64, 1) destination with a (1, 64) addend must not raise.

        The destination keeps its (64, 1) shape, so nothing is broadcast onto
        its size; the addend is read as a broadcast. A kernel that assumes the
        destination's last dimension is the whole repeated operand fails here
        with "output with shape [1, 64] doesn't match the broadcast shape
        [64, 64]" (measured on u180).
        """
        torch.manual_seed(8)
        a = torch.randn(64, 1, device=DEVICE).round(decimals=2)
        b = torch.randn(1, 64, device=DEVICE)
        ref = a.cpu() + b.cpu()
        a.add_(b)
        torch.testing.assert_close(a.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.anyplatform
    def test_add_broadcast_then_out_of_place(self):
        """The out-of-place `a + b` covers the same shapes as the in-place add.

        ATen's composite lowers `a + b` to add_.Tensor, whose structured
        delegate is add.out; registering a kernel under add.Tensor instead turns
        add.out/add_.Tensor/add_.Scalar into backend-fallback calls, and the
        composite then runs only against the CPU dispatcher's tensors. Measured
        on u180: an 8x8 pair failed with "The size of tensor a (64) must match
        the size of tensor b (8)", and (64, 1) plus (1, 64) with "output with
        shape [1, 64] doesn't match the broadcast shape [64, 64]".
        """
        torch.manual_seed(7)
        for shape_a, shape_b in (
            ((8, 8), (8, 8)),
            ((128, 256), (128, 256)),
            ((64, 1), (64, 1)),
            ((64, 1), (1, 64)),
            ((8192,), (8192,)),
        ):
            a = torch.randn(shape_a, device=DEVICE)
            b = torch.randn(shape_b, device=DEVICE)
            ref = torch.add(a.cpu(), b.cpu())
            got = a + b
            assert got.shape == ref.shape
            torch.testing.assert_close(got.cpu(), ref, rtol=1e-4, atol=1e-4)
            got = a.add(b, alpha=2.0)
            torch.testing.assert_close(
                got.cpu(),
                torch.add(a.cpu(), b.cpu(), alpha=2.0),
                rtol=1e-4,
                atol=1e-4,
            )

    @pytest.mark.cuda
    def test_add_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(3)
        a_cuda = torch.randn(64, 64, device="cuda:0")
        b_cuda = torch.randn(64, 64, device="cuda:0")
        ref = torch.add(a_cuda, b_cuda)
        a = a_cuda.to(DEVICE)
        b = b_cuda.to(DEVICE)
        out = torch.add(a, b)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)


class TestAddTensorDispatch:
    """Verify dispatch routing for add.Tensor op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        if not override_is_logged("add.Tensor", "flaggems_python"):
            pytest.skip(_OVERRIDE_UNSERVED)
        result = _run_add_subprocess(
            {
                "FLAGOS_LOG": "dispatch",
                "FLAGOS_OP_add__Tensor": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] add.Tensor -> flagos_python" in result.stderr

    @pytest.mark.flaggems
    @pytest.mark.main_ops
    def test_dispatch_log_flaggems_runtime(self):
        """add.Tensor dispatches to whatever backend this platform's conf lists.

        FlagGems-first is the generated default, but it is not unconditional:
        MUSA routes add.Tensor to its native ``musa`` kernel because FlagGems
        cannot compile the bf16 wrapped-number promotion on the mthreads Triton
        stack (see NATIVE_TRITON_GAPS["musa"]), and Iluvatar routes it to
        ``none`` because FlagGems returns zeros for complex128 -- that conf
        leaves the op to cpu_fallback and registers no ``add.Tensor`` wrapper,
        so the call logs no dispatch line and there is no route to assert.
        Asserting the conf's own value keeps this test meaningful on every
        platform rather than pinning it to the one it was written on; where the
        conf deliberately leaves the op unaccelerated the case skips, the same
        containment ``test_bmm_dispatch.py`` uses.
        """
        expected = routed_backend_or_none("add.Tensor")
        if expected is None:
            pytest.skip("add.Tensor does not take an accelerated route here")
        if not route_is_served("add.Tensor"):
            pytest.skip(
                f"this wheel did not compile the `{expected}` route the conf "
                "names, so the op is unregistered and logs no dispatch"
            )
        result = _run_add_subprocess({"FLAGOS_LOG": "dispatch"})
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        expected = routed_backend("add.Tensor")
        assert f"[flagos dispatch] add.Tensor -> {expected}" in result.stderr

    @pytest.mark.cuda
    @pytest.mark.main_ops
    def test_dispatch_log_cuda_override(self):
        result = _run_add_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_add__Tensor": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] add.Tensor -> boxing" in result.stderr

    @pytest.mark.ascend
    def test_dispatch_log_ascend(self):
        result = _run_add_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_add__Tensor": "ascend"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] add.Tensor -> ascend" in result.stderr


class TestAddTensorAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify add.Tensor on ascend backend matches CPU reference."""
        result = _run_add_subprocess({"FLAGOS_OP_add__Tensor": "ascend"})
        assert result.returncode == 0
