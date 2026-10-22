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
narrow / diff dispatch tests

Regression coverage for the flagos ``narrow`` view kernel. The hand-written
kernel used to call ``self.narrow(...)`` (the Tensor member method), which
re-dispatches through PrivateUse1 back into the same kernel -> infinite
recursion -> stack overflow (SIGSEGV). It now calls ``at::native::narrow_symint``
directly (pure metadata, redispatches only to the registered ``slice`` kernel).

``torch.diff`` is a CompositeImplicitAutograd op that decomposes into ``narrow``,
so it segfaulted for the same reason; it is exercised here too because the
transformers causal-mask path (find_packed_sequence_indices) relies on it.

Usage:
    pytest tests/integration/ops/test_narrow_dispatch.py -v
"""

import pytest

import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


class TestNarrowCorrectness:
    """narrow must not recurse and must match CPU semantics."""

    @pytest.mark.anyplatform
    def test_narrow_basic(self):
        x = torch.arange(8, device=DEVICE).unsqueeze(0)
        y = torch.narrow(x, 1, 1, 7)
        assert tuple(y.shape) == (1, 7)
        torch.testing.assert_close(y.cpu(), torch.arange(1, 8).unsqueeze(0))

    @pytest.mark.anyplatform
    def test_narrow_negative_dim(self):
        x = torch.arange(8, device=DEVICE).unsqueeze(0)
        y = torch.narrow(x, -1, 2, 4)
        torch.testing.assert_close(y.cpu(), torch.arange(2, 6).unsqueeze(0))

    @pytest.mark.anyplatform
    def test_narrow_full_length(self):
        # length == full size: exercises the no-op narrow that still re-entered
        # the kernel in the recursion bug.
        x = torch.arange(8, device=DEVICE).unsqueeze(0)
        y = torch.ops.aten.narrow.default(x, 1, 0, 8)
        torch.testing.assert_close(y.cpu(), x.cpu())

    @pytest.mark.anyplatform
    def test_narrow_matches_cpu_2d(self):
        torch.manual_seed(0)
        x_cpu = torch.randn(4, 16)
        x_fl = x_cpu.to(DEVICE)
        torch.testing.assert_close(
            torch.narrow(x_fl, 1, 3, 10).cpu(), torch.narrow(x_cpu, 1, 3, 10)
        )


class TestNarrowAutograd:
    """narrow must stay differentiable (flagos-ai/Torch-FL#205).

    Claiming ``narrow`` on PrivateUse1 fills the backend slot that the
    CompositeImplicitAutograd fallback checks, so ``AutogradPrivateUse1`` stopped
    decomposing it and hit torchgen's "derivative for aten::narrow is not
    implemented" stub. Forward still worked, so only backward regressed. The
    AutogradPrivateUse1 registration re-runs the composite above autograd, which
    records SliceBackward0 through the dispatched ``at::slice`` inside.
    """

    @pytest.mark.anyplatform
    def test_narrow_backward(self):
        x_cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        x_fl = x_cpu.to(DEVICE).detach().requires_grad_(True)
        x_ref = x_cpu.clone().detach().requires_grad_(True)

        torch.narrow(x_fl, 1, 1, 3).square().sum().backward()
        torch.narrow(x_ref, 1, 1, 3).square().sum().backward()

        assert x_fl.grad is not None
        torch.testing.assert_close(x_fl.grad.cpu(), x_ref.grad)

    @pytest.mark.anyplatform
    def test_narrow_backward_records_grad_fn(self):
        x = torch.randn(4, 8, device=DEVICE, requires_grad=True)
        y = torch.narrow(x, 1, 1, 5)
        # The composite decomposes to slice, so the recorded node is SliceBackward0
        # -- the same one x[:, 1:6] produces.
        assert y.grad_fn is not None
        assert y.requires_grad

    @pytest.mark.anyplatform
    @pytest.mark.parametrize(
        "dim,start,length",
        [
            (-1, 1, 5),  # negative dim
            (1, -3, 2),  # negative start counts from the end
            (1, 2, 0),  # zero length
            (0, 0, 4),  # full length along dim 0
        ],
    )
    def test_narrow_backward_edge_cases(self, dim, start, length):
        torch.manual_seed(0)
        base = torch.randn(4, 8)
        x_fl = base.to(DEVICE).detach().requires_grad_(True)
        x_ref = base.clone().detach().requires_grad_(True)

        torch.narrow(x_fl, dim, start, length).square().sum().backward()
        torch.narrow(x_ref, dim, start, length).square().sum().backward()

        torch.testing.assert_close(x_fl.grad.cpu(), x_ref.grad)

    @pytest.mark.anyplatform
    def test_narrow_backward_non_contiguous(self):
        torch.manual_seed(0)
        base = torch.randn(4, 8)
        x_fl = base.to(DEVICE).detach().requires_grad_(True)
        x_ref = base.clone().detach().requires_grad_(True)

        torch.narrow(x_fl.t(), 0, 1, 5).square().sum().backward()
        torch.narrow(x_ref.t(), 0, 1, 5).square().sum().backward()

        torch.testing.assert_close(x_fl.grad.cpu(), x_ref.grad)

    @pytest.mark.anyplatform
    def test_narrow_second_order_grad(self):
        # slice_backward is itself differentiable, so double backward must work.
        torch.manual_seed(0)
        base = torch.randn(4, 8)
        x_fl = base.to(DEVICE).detach().requires_grad_(True)
        x_ref = base.clone().detach().requires_grad_(True)

        def double_grad(x):
            (g,) = torch.autograd.grad(
                torch.narrow(x, 1, 1, 5).pow(3).sum(), x, create_graph=True
            )
            return torch.autograd.grad(g.sum(), x)[0]

        torch.testing.assert_close(double_grad(x_fl).cpu(), double_grad(x_ref))

    @pytest.mark.anyplatform
    def test_narrow_matches_slice_backward(self):
        # The `slice` control from the issue: both must produce the same grad.
        torch.manual_seed(0)
        base = torch.randn(4, 6)
        x_narrow = base.to(DEVICE).detach().requires_grad_(True)
        x_slice = base.to(DEVICE).detach().requires_grad_(True)

        torch.narrow(x_narrow, 1, 1, 3).square().sum().backward()
        x_slice[:, 1:4].square().sum().backward()

        torch.testing.assert_close(x_narrow.grad.cpu(), x_slice.grad.cpu())

    @pytest.mark.anyplatform
    def test_narrow_no_grad_still_forward_only(self):
        # The ESM-2 backward trace hits narrow with requires_grad=False; that path
        # must stay a plain view with no grad_fn.
        x = torch.arange(24, dtype=torch.float32, device=DEVICE).reshape(4, 6)
        y = torch.narrow(x, 1, 1, 3)
        assert y.grad_fn is None
        assert not y.requires_grad
        torch.testing.assert_close(y.cpu(), torch.narrow(x.cpu(), 1, 1, 3))


class TestDiffCorrectness:
    """torch.diff decomposes into narrow; must not segfault."""

    @pytest.mark.anyplatform
    def test_diff_basic(self):
        pos = torch.arange(32, device=DEVICE).unsqueeze(0)
        d = torch.diff(pos, dim=-1)
        assert tuple(d.shape) == (1, 31)
        torch.testing.assert_close(d.cpu(), torch.ones(1, 31, dtype=torch.long))

    @pytest.mark.anyplatform
    def test_diff_with_prepend(self):
        # The transformers find_packed_sequence_indices pattern.
        pos = torch.arange(32, device=DEVICE).unsqueeze(0)
        first = pos[:, :1] - 1
        d = torch.diff(pos, prepend=first, dim=-1)
        d_cpu = torch.diff(pos.cpu(), prepend=first.cpu(), dim=-1)
        assert tuple(d.shape) == (1, 32)
        torch.testing.assert_close(d.cpu(), d_cpu)

    @pytest.mark.anyplatform
    def test_diff_then_cumsum(self):
        pos = torch.arange(32, device=DEVICE).unsqueeze(0)
        first = pos[:, :1] - 1
        mask = (torch.diff(pos, prepend=first, dim=-1) != 1).cumsum(-1)
        # single contiguous sequence -> all zeros
        assert (mask[:, -1] == 0).all().cpu().item()
