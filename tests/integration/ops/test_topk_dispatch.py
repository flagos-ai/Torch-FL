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
topk dispatch tests

`topk` on Ascend used to route to FlagGems, whose single-stage kernel returns
all zeros -- both the values *and* the indices -- and does it silently. A caller
got `[0.0] * k` and `[0] * k` back from a call that raised nothing, so the only
check that catches it is one against a reference *value*; "the op ran" is
exactly the property the zeros satisfy. The route is now `ascend` (aclnnTopk),
and these tests pin both halves: the result matches the CPU, and the route that
produced it is still the aclnn one.

Usage:
    pytest tests/integration/ops/test_topk_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch

import torch_fl  # noqa: F401

DEVICE = "flagos:0"

# (shape, dim, largest, dtype). N is kept below the FlagGems two-stage
# threshold and above it in one case so the whole op is covered rather than
# one branch of it.
_CASES = [
    ((128,), -1, True, torch.float32),
    ((128,), -1, False, torch.float32),
    ((7,), -1, True, torch.float32),
    ((1,), -1, True, torch.float32),
    ((8, 16), -1, True, torch.float32),
    ((8, 16), -1, False, torch.float32),
    ((8, 16), 0, True, torch.float32),
    ((3, 5, 17), -1, True, torch.float32),
    ((128,), -1, True, torch.float16),
    ((128,), -1, True, torch.bfloat16),
]


def _ids(case):
    shape, dim, largest, dtype = case
    return f"{'x'.join(map(str, shape))}-dim{dim}-{'max' if largest else 'min'}-{dtype}"


def _reference(shape, dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype)
    return torch.randint(-1000, 1000, shape, dtype=dtype)


@pytest.mark.ascend
@pytest.mark.parametrize("case", _CASES, ids=_ids)
def test_matches_cpu(case):
    """Values and indices, against the CPU, at the caller's dim and order."""
    shape, dim, largest, dtype = case
    cpu = _reference(shape, dtype)
    k = min(5, shape[dim])

    expected_values, expected_indices = torch.topk(cpu, k, dim=dim, largest=largest)
    values, indices = torch.topk(cpu.to(DEVICE), k, dim=dim, largest=largest)

    assert values.device.type == "flagos"
    assert indices.device.type == "flagos"
    assert values.dtype == expected_values.dtype
    assert indices.dtype == expected_indices.dtype
    torch.testing.assert_close(values.cpu(), expected_values)
    # Indices are exact: a near-miss here is a different element, not a
    # rounding difference. Asserting them is what separates "sorted something"
    # from "sorted this".
    torch.testing.assert_close(indices.cpu(), expected_indices, rtol=0, atol=0)


@pytest.mark.ascend
def test_never_returns_all_zeros():
    """The defect's own signature, stated directly.

    FlagGems' topk answered `[0.0] * k` and `[0] * k` for every input. Those
    two answers are not reachable from a working topk of a random tensor: the
    indices would have to be `0` k times, where a real result holds k distinct
    positions out of N. A tolerance-based check on the values alone would still
    catch it here, but the indices make the failure unambiguous -- and this
    test names it, so a future re-route points at the right thing.
    """
    torch.manual_seed(0)
    cpu = torch.randn(128)
    values, indices = torch.topk(cpu.to(DEVICE), 5)

    assert int((values.cpu() == 0).sum()) == 0, values.cpu().tolist()
    assert len(set(indices.cpu().tolist())) == 5, indices.cpu().tolist()
    assert max(indices.cpu().tolist()) > 0, indices.cpu().tolist()


@pytest.mark.ascend
def test_route_is_the_aclnn_kernel():
    """The route the fix depends on, read from the dispatch log.

    `flaggems` is the route that was wrong; asserting the resolved backend
    keeps a regeneration of backends_ascend.conf from putting `topk` back on
    FlagGems' kernel without anything noticing. Uses no FLAGOS_OP_* override,
    so this is the conf's own answer.
    """
    code = (
        "import torch, torch_fl; "
        "x = torch.randn(128, device='flagos:0'); "
        "torch.topk(x, 5)"
    )
    env = os.environ.copy()
    env["FLAGOS_LOG"] = "dispatch"
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "[flagos dispatch] topk -> ascend" in result.stderr, result.stderr
