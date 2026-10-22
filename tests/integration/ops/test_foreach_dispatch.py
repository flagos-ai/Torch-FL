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
_foreach_* TensorList dispatch tests (the AdamW foreach=True path).

The list lengths here are the point of the test. CANN's aclnnForeach* kernels
only process the first 50 entries of an aclTensorList: past that they either
error or -- worse -- return success and leave the remaining tensors UNTOUCHED.
The Ascend kernels therefore slice their lists into sub-50 chunks, so every case
below runs a length that straddles a chunk boundary and asserts on EVERY entry,
not just the first few. A regression that drops the chunking is silent unless
the tail entries are checked.

Usage:
    pytest tests/integration/ops/test_foreach_dispatch.py -v
"""

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"

# 51 and 60 cross aclnn's raw 50-entry cap; 310 is what AdamW passes for
# Qwen3-0.6B and crosses many chunk boundaries.
LENGTHS = [1, 31, 32, 33, 51, 60, 128, 310]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOL = {
    torch.float32: dict(rtol=1e-4, atol=1e-5),
    torch.float16: dict(rtol=1e-2, atol=1e-2),
    torch.bfloat16: dict(rtol=5e-2, atol=5e-2),
}


def _lists(n, dtype, seed, numel=8):
    """n CPU tensors of varying shape + their flagos copies."""
    g = torch.Generator().manual_seed(seed)
    cpu = [
        ((torch.rand(1 + (i % 3), numel, generator=g) + 0.5).to(dtype))
        for i in range(n)
    ]
    return cpu, [t.to(DEVICE) for t in cpu]


def _assert_all_close(got, ref, dtype):
    """Compare every entry -- a truncating kernel only differs in the tail."""
    assert len(got) == len(ref)
    for i, (g, r) in enumerate(zip(got, ref)):
        torch.testing.assert_close(
            g.cpu().float(),
            r.float(),
            msg=lambda m, i=i: f"entry {i}: {m}",
            **TOL[dtype],
        )


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.anyplatform
class TestForeachAgainstCpu:
    """Each _foreach_* Ascend kernel vs the CPU implementation, entry by entry."""

    def test_foreach_mul_scalar(self, n, dtype):
        cpu, dev = _lists(n, dtype, 1)
        torch._foreach_mul_(cpu, 0.9)
        torch._foreach_mul_(dev, 0.9)
        _assert_all_close(dev, cpu, dtype)

    def test_foreach_add_scalar(self, n, dtype):
        cpu, dev = _lists(n, dtype, 2)
        torch._foreach_add_(cpu, 1.0)
        torch._foreach_add_(dev, 1.0)
        _assert_all_close(dev, cpu, dtype)

    def test_foreach_sqrt(self, n, dtype):
        cpu, dev = _lists(n, dtype, 3)
        _assert_all_close(torch._foreach_sqrt(dev), torch._foreach_sqrt(cpu), dtype)

    def test_foreach_lerp_scalar(self, n, dtype):
        cpu, dev = _lists(n, dtype, 4)
        # distinct end points: lerp_ toward a copy of self is a no-op and would
        # hide a truncating kernel entirely.
        end_cpu, end_dev = _lists(n, dtype, 5)
        torch._foreach_lerp_(cpu, end_cpu, 0.3)
        torch._foreach_lerp_(dev, end_dev, 0.3)
        _assert_all_close(dev, cpu, dtype)

    def test_foreach_addcmul_scalar(self, n, dtype):
        cpu, dev = _lists(n, dtype, 6)
        t1_cpu, t1_dev = _lists(n, dtype, 7)
        t2_cpu, t2_dev = _lists(n, dtype, 8)
        torch._foreach_addcmul_(cpu, t1_cpu, t2_cpu, 0.1)
        torch._foreach_addcmul_(dev, t1_dev, t2_dev, 0.1)
        _assert_all_close(dev, cpu, dtype)

    def test_foreach_div_scalarlist(self, n, dtype):
        cpu, dev = _lists(n, dtype, 9)
        scalars = [0.5 + 0.01 * i for i in range(n)]
        torch._foreach_div_(cpu, scalars)
        torch._foreach_div_(dev, scalars)
        _assert_all_close(dev, cpu, dtype)

    def test_foreach_addcdiv_scalarlist(self, n, dtype):
        cpu, dev = _lists(n, dtype, 10)
        t1_cpu, t1_dev = _lists(n, dtype, 11)
        t2_cpu, t2_dev = _lists(n, dtype, 12)
        scalars = [0.5 + 0.01 * i for i in range(n)]
        torch._foreach_addcdiv_(cpu, t1_cpu, t2_cpu, scalars)
        torch._foreach_addcdiv_(dev, t1_dev, t2_dev, scalars)
        _assert_all_close(dev, cpu, dtype)


class TestForeachAdamW:
    """AdamW(foreach=True) must match AdamW(foreach=False) on the same model."""

    @pytest.mark.anyplatform
    def test_adamw_foreach_matches_single_tensor(self):
        # >50 params so the optimizer's TensorLists cross a chunk boundary.
        torch.manual_seed(0)
        shapes = [(8, 8)] * 40 + [(16,)] * 40

        def build():
            torch.manual_seed(0)
            return [torch.nn.Parameter(torch.randn(*s, device=DEVICE)) for s in shapes]

        def run(foreach):
            params = build()
            opt = torch.optim.AdamW(params, lr=1e-2, foreach=foreach)
            for _ in range(3):
                for p in params:
                    p.grad = torch.ones_like(p) * 0.1
                opt.step()
                opt.zero_grad()
            return [p.detach().cpu().clone() for p in params]

        for i, (a, b) in enumerate(zip(run(True), run(False))):
            torch.testing.assert_close(
                a, b, rtol=1e-4, atol=1e-5, msg=lambda m, i=i: f"param {i}: {m}"
            )
