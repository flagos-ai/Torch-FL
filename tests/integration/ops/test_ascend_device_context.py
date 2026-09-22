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
Ascend device-context regression (issue #326).

Every layer below an Ascend kernel reads the *ambient* current device rather
than the device of its operands: the aclnn workspace is allocated with an
index-less ``at::kPrivateUse1`` TensorOptions, ``EXEC_ASCEND_CMD`` takes its
launch stream from ``GetCurrentAclStream()``, and ``ExecAscendCached`` keys the
executor cache off ``::GetDevice``. So an op called on flagos:0 tensors while
flagos:1 was current returned its result on flagos:1 and enqueued the kernel on
that device's stream.

Each Ascend kernel now opens with an ``OpDeviceGuard`` built from its primary
tensor argument -- or, for the factories that take no tensor at all, from their
explicit ``device`` -- which makes the operand's device current for the call and
restores the caller's on exit. These tests are the aclnn-side counterpart of
``tests/integration/test_compute_device_index.py`` and
``tests/integration/test_factory_device_index.py``, which cover the same hazard
for the FlagGems/python-op-caller path.

Only meaningful with 2+ devices: with the operands already on the current device
the missing guard is invisible, which is why every test here runs with a device
that does *not* own the operands. The default conf for the Ascend wheel routes
these ops to aclnn; nothing here depends on which backend serves them, only on
the device the result lands on.

Usage:
    pytest tests/integration/ops/test_ascend_device_context.py -v
"""

import pytest
import torch
import torch_fl


pytestmark = [
    pytest.mark.ascend,
    pytest.mark.skipif(
        torch_fl.flagos.device_count() < 2,
        reason="needs at least 2 flagos devices",
    ),
]

# Operands live on 1 while 0 is current: matching indices would pass with or
# without the guard.
DEVICE = "flagos:1"
INDEX = 1
OTHER = 0


@pytest.fixture(autouse=True)
def current_device_zero():
    """Pin the current device to 0 for every test in this file.

    This is the point of the file: an op on a flagos:1 tensor has to switch to
    device 1 itself. With device 1 already current the hazard disappears.
    """
    prev = torch_fl.flagos.current_device()
    torch_fl.flagos.set_device(OTHER)
    try:
        yield
    finally:
        torch_fl.flagos.set_device(prev)


def _check(
    out: torch.Tensor,
    expected: torch.Tensor,
    what: str,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    assert out.device.type == "flagos", f"{what}: got device type {out.device.type}"
    assert out.device.index == INDEX, (
        f"{what}: result on flagos:{out.device.index}, expected flagos:{INDEX}"
    )
    torch.testing.assert_close(out.cpu(), expected, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# One op per guard-argument shape
# ---------------------------------------------------------------------------


def test_tensor_operand():
    """``self`` is the primary argument for a plain unary op."""
    cpu = torch.rand(64)
    _check(torch.sqrt(cpu.to(DEVICE)), torch.sqrt(cpu), "sqrt")


def test_two_tensor_operands():
    a, b = torch.randn(32, 16), torch.randn(32, 16)
    _check(a.to(DEVICE) * b.to(DEVICE), a * b, "mul")


def test_matrix_multiply():
    """The op in the #326 report: aclnn workspace + stream both follow it."""
    torch.manual_seed(0)
    a, b = torch.randn(64, 64), torch.randn(64, 64)
    # The cube's input path rounds to ~1e-4 relative (measured 1.5e-4 at
    # max|ref| 31 on 910), well outside fp32 -- that is the kernel's own
    # accuracy and is unchanged by this fix, so the reference comparison is
    # loose here while test_mm_matches_same_device_result pins the exactness
    # that #326 is actually about.
    _check(torch.mm(a.to(DEVICE), b.to(DEVICE)), a @ b, "mm", rtol=1e-3, atol=1e-2)


def test_mm_matches_same_device_result():
    """Cross-device and same-device calls must agree bit for bit.

    The sharpest statement of #326: with the guard the op reads and writes the
    same memory either way, so the two results are identical rather than merely
    close. Without it the operands of the flagos:1 call are addressed on
    device 0's stream.
    """
    torch.manual_seed(0)
    a, b = torch.randn(64, 64), torch.randn(64, 64)
    x, w = a.to(DEVICE), b.to(DEVICE)

    torch_fl.flagos.set_device(INDEX)  # ambient == operand device
    same = torch.mm(x, w)
    torch_fl.flagos.set_device(OTHER)  # ambient != operand device
    cross = torch.mm(x, w)

    assert cross.device.index == INDEX, f"mm on flagos:{cross.device.index}"
    assert torch.equal(cross.cpu(), same.cpu()), (
        "the cross-device mm result differs from the same-device one; the "
        f"max difference is {(cross.cpu() - same.cpu()).abs().max().item():.3e}"
    )


def test_scalar_first_signature():
    """``pow.Scalar`` takes the Scalar first, so the guard must use the tensor.

    The generator picks the primary argument by name, not by position; a
    positional choice would guard the unborn exponent and set no device.
    """
    base = torch.rand(64) + 0.5
    _check(torch.pow(2.0, base.to(DEVICE)), torch.pow(2.0, base), "pow.Scalar")


def test_tensor_list_operand():
    """``cat`` takes a TensorList; its first element names the device."""
    xs = [torch.randn(4, 6) for _ in range(3)]
    _check(torch.cat([x.to(DEVICE) for x in xs], dim=0), torch.cat(xs, dim=0), "cat")


def test_out_variant():
    """The result is written into a caller-supplied buffer on the operand's
    device while a different device is current."""
    torch.manual_seed(0)
    a, b = torch.randn(16, 32), torch.randn(32, 8)
    out = torch.empty(16, 8, device=DEVICE)
    torch.mm(a.to(DEVICE), b.to(DEVICE), out=out)
    _check(out, a @ b, "mm.out", rtol=1e-3, atol=1e-2)


def test_inplace_op():
    cpu = torch.randn(64)
    t = cpu.to(DEVICE)
    t.relu_()
    _check(t, torch.relu(cpu), "relu_")


def test_tuple_returning_op():
    """``sort`` returns (values, indices); both must land on the operand device."""
    cpu = torch.randn(128)
    values, indices = torch.sort(cpu.to(DEVICE))
    exp_values, exp_indices = torch.sort(cpu)
    _check(values, exp_values, "sort.values")
    assert indices.device.index == INDEX, (
        f"sort.indices on flagos:{indices.device.index}"
    )
    assert indices.cpu().tolist() == exp_indices.tolist()


def test_factory_device_argument():
    """A factory has no tensor operand, so the guard reads its ``device``."""
    out = torch.zeros(8, device=DEVICE)
    assert out.device.index == INDEX, f"zeros on flagos:{out.device.index}"
    assert out.cpu().tolist() == [0.0] * 8

    out = torch.ones(4, device=DEVICE)
    assert out.device.index == INDEX, f"ones on flagos:{out.device.index}"
    assert out.cpu().tolist() == [1.0] * 4


# ---------------------------------------------------------------------------
# Entries reached through the copy machinery rather than through a generated
# kernel: `_to_copy`'s dtype cast and `contiguous()`.
#
# These are the ones the per-kernel guard does not cover by itself, because
# their callers (`copy_ops.cc`, `contiguous_ops.cc`) are shared with the other
# backends and carry no device of their own. Both allocate their result from the
# operand's TensorOptions, so they land on the right device even unguarded --
# what the cast got wrong was the *value*, because it also goes through the
# device-keyed `ExecAscendCached` and read its input before the producer had
# written it. The strided copy was measured correct either way; it is checked
# here so the path stays covered.
# ---------------------------------------------------------------------------


def test_dtype_cast_matches_same_device_result():
    """``.float()`` on an int64 operand must not depend on the current device.

    The guard is what makes this reachable at all: once `randperm` lands its
    result on flagos:1 while flagos:0 stays current, the cast below is the first
    device-less entry to consume it. Unguarded, its cached aclnn executor is
    built on the wrong device and the cast reads the freshly written permutation
    back as the *previous* draw -- the allocator has recycled the output block,
    so the stale contents look like a plausible answer rather than like garbage,
    and a seed-based reproducibility check cannot tell the two apart.
    """
    torch.manual_seed(1234)
    x = torch.randperm(50, device=DEVICE)
    expected = x.cpu()

    torch_fl.flagos.set_device(INDEX)  # ambient == operand device
    same = x.float()
    torch_fl.flagos.set_device(OTHER)  # ambient != operand device
    cross = x.float()

    assert cross.device.index == INDEX, f"cast on flagos:{cross.device.index}"
    assert torch.equal(cross.cpu(), same.cpu()), (
        "the cross-device cast differs from the same-device one: "
        f"{cross.cpu()[:8].tolist()} vs {same.cpu()[:8].tolist()}"
    )
    # And the cast has to read the *current* contents of x, not a recycled
    # buffer: comparing against x itself is what the seed-based check above
    # cannot see when both draws return the same stale block.
    assert cross.cpu().tolist() == expected.tolist()


def test_contiguous_matches_same_device_result():
    """``contiguous()`` on a transposed flagos:1 tensor, different device current."""
    torch.manual_seed(0)
    x = torch.randn(16, 32, device=DEVICE).t()

    torch_fl.flagos.set_device(INDEX)
    same = x.contiguous()
    torch_fl.flagos.set_device(OTHER)
    cross = x.contiguous()

    assert cross.device.index == INDEX, f"contiguous on flagos:{cross.device.index}"
    assert torch.equal(cross.cpu(), same.cpu()), (
        "the cross-device contiguous() differs from the same-device one"
    )
    assert torch.equal(cross.cpu(), x.cpu())


# ---------------------------------------------------------------------------
# The current device must be left as it was found
# ---------------------------------------------------------------------------


def test_current_device_is_restored():
    """The guard is scoped: an op on flagos:1 must not leave 1 current."""
    assert torch_fl.flagos.current_device() == OTHER
    torch.abs(torch.randn(16, device=DEVICE))
    assert torch_fl.flagos.current_device() == OTHER, (
        "an op on flagos:1 leaked the device switch to the caller"
    )


def test_repeated_calls_alternate_correctly():
    """Guarding is per call, not sticky: interleaving devices stays correct."""
    cpu = torch.randn(32)
    on_one = cpu.to(DEVICE)
    on_zero = cpu.to("flagos:0")
    for _ in range(3):
        _check(torch.abs(on_one), torch.abs(cpu), "abs on flagos:1")
        zero_out = torch.abs(on_zero)
        assert zero_out.device.index == OTHER, (
            f"abs on flagos:0 landed on flagos:{zero_out.device.index}"
        )
        assert torch_fl.flagos.current_device() == OTHER
