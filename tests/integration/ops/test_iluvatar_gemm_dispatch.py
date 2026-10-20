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

"""Iluvatar ``linear`` / ``addmm``: the shapes ``nn.Linear`` actually uses.

The Iluvatar lane's "vendor backend, main ops" group does not sweep
``test_factory_ops.py``, so nothing there exercises the matrix shapes a model
layer reaches -- and the first one to do so, issue #570's MPS-side
``test_module_to_device_migrates_parameters_in_place``, failed with
``allclose`` on values off by orders of magnitude: the FlagGems ``_iluvatar``
``linear`` kernel wrote a result the CPU reference does not share, rather than
something off by a rounding step. That makes this a correctness case in its own
right, independent of which test file happened to find it: a shape-parametrized
CPU comparison pins it, and the shape list is short on purpose (it is a sweep of
one kernel, run on hardware).

Every case stays enabled: the comparison is against the CPU for the same
operands, so it asserts what ATen defines rather than what this build happens to
produce, and no case is skipped on a platform that can run the op.
"""

import pytest
import torch

DEVICE = "flagos:0"

#: ``(M, K, N)`` -- the shapes ``nn.Linear(in, out)`` reaches for a 4-wide
#: input, then the identity and a larger tiled case.
SHAPES = [
    (1, 4, 2),
    (4, 4, 8),
    (4, 8, 4),
    (8, 8, 8),
    (32, 64, 16),
    (64, 128, 32),
]


def _assert_matches_cpu(op, actual, reference):
    torch.testing.assert_close(
        actual.float().cpu(),
        reference.float(),
        rtol=1e-2,
        atol=1e-2,
        msg=f"{op} on {DEVICE} differs from the CPU result",
    )


class TestIluvatarLinear:
    @pytest.mark.parametrize("M,K,N", SHAPES)
    @pytest.mark.anyplatform
    def test_functional_linear_matches_cpu(self, M, K, N):
        torch.manual_seed(0)
        x = torch.randn(M, K)
        w = torch.randn(N, K)
        b = torch.randn(N)

        reference = torch.nn.functional.linear(x, w, b)
        actual = torch.nn.functional.linear(x.to(DEVICE), w.to(DEVICE), b.to(DEVICE))
        _assert_matches_cpu(f"functional.linear [{M},{K}]x[{N},{K}]", actual, reference)

    @pytest.mark.parametrize("M,K,N", SHAPES)
    @pytest.mark.anyplatform
    def test_module_linear_after_to_device_matches_cpu(self, M, K, N):
        """The sequence a layer goes through before its first forward.

        Built on CPU, moved with ``.to(device)``, run under ``no_grad`` -- the
        path ``test_module_to_device_migrates_parameters_in_place`` takes -- and
        compared against the same module's CPU forward.
        """
        torch.manual_seed(1)
        layer = torch.nn.Linear(K, N)
        x = torch.randn(M, K)
        reference = layer(x).detach()

        layer.to(DEVICE)
        with torch.no_grad():
            actual = layer(x.to(DEVICE)).cpu()
        _assert_matches_cpu(f"nn.Linear({K}, {N}) on [{M},{K}]", actual, reference)


class TestIluvatarAddmm:
    @pytest.mark.parametrize("M,K,N", SHAPES)
    @pytest.mark.anyplatform
    def test_addmm_matches_cpu(self, M, K, N):
        torch.manual_seed(2)
        bias = torch.randn(M, N)
        a = torch.randn(M, K)
        b = torch.randn(K, N)

        reference = torch.addmm(bias, a, b)
        actual = torch.addmm(bias.to(DEVICE), a.to(DEVICE), b.to(DEVICE))
        _assert_matches_cpu(f"addmm [{M},{K}]x[{K},{N}]", actual, reference)
