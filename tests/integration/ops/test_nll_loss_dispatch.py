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
nll_loss_forward and nll_loss_backward dispatch tests

Verifies that torch.nn.functional.nll_loss:
  - produces correct results on flagos device
  - backward produces correct gradients
  - C++ wrapper selects the installed route or a supported FlagGems override
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_nll_loss_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401

from backend_conf import override_is_logged, routed_backend_or_none


DEVICE = "flagos:0"


def _run_nll_subprocess(
    extra_env: dict, *, backward: bool = False
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    code = """
import sys

# Load the vendor runtime before Torch initializes its C++ device hooks.
import torch_fl
import torch

torch_fl.flagos.init()
op_forward = torch.ops.aten.nll_loss_forward.default
op_backward = torch.ops.aten.nll_loss_backward.default
device = "flagos:0"
cpu = torch.linspace(-2, 1, 40).reshape(4, 10).log_softmax(1)
labels = torch.tensor([0, -100, 5, 9], dtype=torch.int64)
options = (None, 1, -100)
values, targets = cpu.to(device), labels.to(device)
reference = op_forward(cpu, labels, *options)

if sys.argv[1] == "backward":
    seed = torch.tensor(0.5)
    expected = op_backward(seed, cpu, labels, *options, reference[1])
    actual = op_backward(seed.to(device), values, targets, *options, reference[1].to(device))
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
else:
    actual = op_forward(values, targets, *options)
    for result, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
"""
    return subprocess.run(
        [sys.executable, "-c", code, "backward" if backward else "forward"],
        env=env,
        capture_output=True,
        text=True,
    )


def _nll_route_probe(op, route, *, backward=False):
    env = {"FLAGOS_LOG": "dispatch"}
    if override_is_logged(op, "flaggems_python"):
        env[f"FLAGOS_OP_{op}"] = "flaggems_python"
        route = "flagos_python"
    result = _run_nll_subprocess(env, backward=backward)
    assert result.returncode == 0, result.stderr
    assert f"[flagos dispatch] {op} -> {route}" in result.stderr, result.stderr


class TestNllLossForwardCorrectness:
    """nll_loss_forward correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_nll_loss_basic(self):
        torch.manual_seed(0)
        inp = torch.randn(8, 10, device=DEVICE).log_softmax(dim=1)
        target = torch.randint(0, 10, (8,), device=DEVICE)
        loss = F.nll_loss(inp, target)
        assert loss.shape == ()
        assert loss.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_nll_loss_matches_cpu(self):
        torch.manual_seed(1)
        inp_cpu = torch.randn(16, 5).log_softmax(dim=1)
        target_cpu = torch.randint(0, 5, (16,))
        ref = F.nll_loss(inp_cpu, target_cpu)

        inp_fl = inp_cpu.to(DEVICE)
        target_fl = target_cpu.to(DEVICE)
        out = F.nll_loss(inp_fl, target_fl)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.anyplatform
    def test_nll_loss_reduction(self, reduction):
        torch.manual_seed(2)
        inp_cpu = torch.randn(8, 10).log_softmax(dim=1)
        target_cpu = torch.randint(0, 10, (8,))
        ref = F.nll_loss(inp_cpu, target_cpu, reduction=reduction)

        inp_fl = inp_cpu.to(DEVICE)
        target_fl = target_cpu.to(DEVICE)
        out = F.nll_loss(inp_fl, target_fl, reduction=reduction)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.anyplatform
    def test_nll_loss_ignore_index(self):
        torch.manual_seed(3)
        inp_cpu = torch.randn(8, 10).log_softmax(dim=1)
        target_cpu = torch.randint(0, 10, (8,))
        target_cpu[0] = -100
        ref = F.nll_loss(inp_cpu, target_cpu, ignore_index=-100)

        inp_fl = inp_cpu.to(DEVICE)
        target_fl = target_cpu.to(DEVICE)
        out = F.nll_loss(inp_fl, target_fl, ignore_index=-100)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)


class TestNllLossBackwardCorrectness:
    """nll_loss_backward correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_nll_loss_backward_basic(self):
        torch.manual_seed(0)
        inp = torch.randn(8, 10, device=DEVICE, requires_grad=True)
        log_inp = inp.log_softmax(dim=1)
        target = torch.randint(0, 10, (8,), device=DEVICE)
        loss = F.nll_loss(log_inp, target)
        loss.backward()
        assert inp.grad is not None
        assert inp.grad.shape == (8, 10)

    @pytest.mark.anyplatform
    def test_nll_loss_backward_matches_cpu(self):
        torch.manual_seed(1)
        inp_cpu = torch.randn(16, 5, requires_grad=True)
        log_inp_cpu = inp_cpu.log_softmax(dim=1)
        target_cpu = torch.randint(0, 5, (16,))
        loss_cpu = F.nll_loss(log_inp_cpu, target_cpu)
        loss_cpu.backward()

        # Use the SAME input values on flagos (copy from CPU) rather than
        # re-seeding randn: flagos randn now uses the CUDA RNG, which does not
        # match the CPU RNG for the same seed, so re-seeding would compare
        # gradients of different inputs.
        inp_fl = inp_cpu.detach().to(DEVICE).requires_grad_(True)
        log_inp_fl = inp_fl.log_softmax(dim=1)
        target_fl = target_cpu.to(DEVICE)
        loss_fl = F.nll_loss(log_inp_fl, target_fl)
        loss_fl.backward()

        torch.testing.assert_close(
            inp_fl.grad.cpu(), inp_cpu.grad, rtol=1e-4, atol=1e-4
        )


class TestNllLossDispatch:
    """Verify dispatch routing for nll_loss ops."""

    @pytest.mark.flaggems_python
    def test_nll_forward_uses_available_route(self):
        op = "nll_loss_forward"
        route = routed_backend_or_none(op)
        if route is None:
            pytest.skip("nll_loss_forward is routed to 'none' on this platform")
        _nll_route_probe(op, route)

    @pytest.mark.flaggems_python
    def test_nll_backward_uses_available_route(self):
        op = "nll_loss_backward"
        route = routed_backend_or_none(op)
        if route is None:
            pytest.skip("nll_loss_backward is routed to 'none' on this platform")
        _nll_route_probe(op, route, backward=True)


class TestNllLossForwardAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify nll_loss_forward on ascend backend matches CPU reference."""
        result = _run_nll_subprocess({"FLAGOS_OP_nll_loss_forward": "ascend"})
        assert result.returncode == 0, result.stderr
