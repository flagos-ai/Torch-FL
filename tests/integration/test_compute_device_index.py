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
Multi-device compute regression: a non-factory op must run on its operand's device.

Companion to test_factory_device_index.py, which covers the factory callers.
This covers the other 16 ``CallPythonOp_*`` callers in
csrc/aten/backends/flagos/python_op_caller.cc -- the ones that take tensors in
rather than materializing them.

Same hazard, reached from the other side: gems allocates its intermediates with
``device=input.device`` but launches Triton on the *current* device. Call an op
on a tensor living on device 1 while device 0 is current and the kernel reads
and writes across devices. That does not raise -- it faults the GPU (segfault /
KERNEL VMFault / "Invalid address access"), which is how
``torch.multinomial(x_on_flagos_1, ...)`` died before the callers took a
DeviceGuard.

The fault is a *process kill*, not an exception, so a regression here takes the
whole pytest process down rather than reporting a failure. That is intentional
and still a clear signal; ``-x`` is not required to notice it.

Only meaningful with 2+ devices, so it skips otherwise. Worth running both ways:
the default conf routing for the path that broke, and ALL_USE_VENDOR=1 so the
boxing path stays covered too.

Usage:
    pytest tests/integration/test_compute_device_index.py -v
    ALL_USE_VENDOR=1 pytest tests/integration/test_compute_device_index.py -v
"""

import pytest
import torch
import torch_fl


pytestmark = pytest.mark.skipif(
    torch_fl.flagos.device_count() < 2,
    reason="needs at least 2 flagos devices",
)

# Device 1 with device 0 current is the combination that faults: matching indices
# would pass with or without the guard.
DEVICE = "flagos:1"
INDEX = 1


@pytest.fixture(autouse=True)
def current_device_zero():
    """Pin the current device to 0 for every test in this file.

    This is the whole point: an op on a flagos:1 tensor has to switch to device 1
    itself. If the test ran with device 1 already current, the missing guard
    would be invisible.
    """
    prev = torch_fl.flagos.current_device()
    torch_fl.flagos.set_device(0)
    try:
        yield
    finally:
        torch_fl.flagos.set_device(prev)


def _check(out: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    assert out.device.type == "flagos", f"{what}: got device type {out.device.type}"
    assert out.device.index == INDEX, (
        f"{what}: result on flagos:{out.device.index}, expected flagos:{INDEX}"
    )
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# The op that actually crashed
# ---------------------------------------------------------------------------


def test_multinomial_on_nonzero_device():
    """The original repro: segfaulted under FLAGOS_USE_FLAGGEMS=1 on flagos:1.

    Routes through CallPythonOp_Generic. Uses a one-hot weight vector so the
    result is deterministic -- a cross-device read returns whatever garbage the
    other device's memory holds, which random sampling alone would not catch.
    """
    weights = torch.zeros(10, device=DEVICE)
    weights[7] = 1.0
    out = torch.multinomial(weights, 5, replacement=True)
    assert out.device.index == INDEX, f"result on flagos:{out.device.index}"
    assert out.cpu().tolist() == [7] * 5, out.cpu().tolist()


# ---------------------------------------------------------------------------
# One op per caller shape
# ---------------------------------------------------------------------------


def test_unary_op():
    """CallPythonOp_T."""
    cpu = torch.randn(64)
    _check(torch.abs(cpu.to(DEVICE)), torch.abs(cpu), "abs")


def test_unary_inplace_op():
    """CallPythonOp_T_inplace."""
    cpu = torch.randn(64)
    t = cpu.to(DEVICE)
    t.relu_()
    _check(t, torch.relu(cpu), "relu_")


def test_binary_tensor_op():
    """CallPythonOp_TT."""
    a, b = torch.randn(32, 16), torch.randn(32, 16)
    _check(a.to(DEVICE) * b.to(DEVICE), a * b, "mul")


def test_binary_tensor_scalar_op():
    """CallPythonOp_TTS (add with alpha)."""
    a, b = torch.randn(32, 16), torch.randn(32, 16)
    out = torch.add(a.to(DEVICE), b.to(DEVICE), alpha=2.0)
    _check(out, torch.add(a, b, alpha=2.0), "add.alpha")


def test_tensor_scalar_op():
    """CallPythonOp_TS."""
    cpu = torch.randn(64)
    _check(cpu.to(DEVICE) + 3.0, cpu + 3.0, "add.Scalar")


def test_dim_reduction_op():
    """CallPythonOp_TIB (dim + keepdim)."""
    cpu = torch.randn(8, 16)
    _check(cpu.to(DEVICE).sum(dim=1, keepdim=True), cpu.sum(dim=1, keepdim=True), "sum")


def test_ternary_tensor_op():
    """CallPythonOp_TTT (addmm: 3 tensor operands)."""
    a, b, c = torch.randn(4, 4), torch.randn(4, 8), torch.randn(8, 4)
    out = torch.addmm(a.to(DEVICE), b.to(DEVICE), c.to(DEVICE))
    _check(out, torch.addmm(a, b, c), "addmm")


def test_embedding_op():
    """CallPythonOp_Embedding (weight + indices)."""
    weight = torch.randn(20, 8)
    idx = torch.tensor([3, 1, 4, 1, 5])
    out = torch.nn.functional.embedding(idx.to(DEVICE), weight.to(DEVICE))
    _check(out, torch.nn.functional.embedding(idx, weight), "embedding")


def test_tensor_list_op():
    """CallPythonOp_ListI (cat over a TensorList)."""
    xs = [torch.randn(4, 6) for _ in range(3)]
    out = torch.cat([x.to(DEVICE) for x in xs], dim=0)
    _check(out, torch.cat(xs, dim=0), "cat")


def test_tuple_returning_op():
    """CallPythonOp_GenericTuple (topk returns values + indices)."""
    cpu = torch.randn(128)
    vals, idx = torch.topk(cpu.to(DEVICE), 5)
    exp_vals, exp_idx = torch.topk(cpu, 5)
    _check(vals, exp_vals, "topk.values")
    assert idx.device.index == INDEX, f"topk.indices on flagos:{idx.device.index}"
    assert idx.cpu().tolist() == exp_idx.tolist()


def test_kwarg_op():
    """CallPythonOp_GenericKw (softmax takes dim as a kwarg)."""
    cpu = torch.randn(8, 32)
    out = torch.softmax(cpu.to(DEVICE), dim=-1)
    _check(out, torch.softmax(cpu, dim=-1), "softmax")


def test_random_inplace_op():
    """CallPythonOp_RandomInplace: writes on the operand's device, plausibly."""
    t = torch.empty(4096, device=DEVICE)
    t.normal_(0.0, 1.0)
    assert t.device.index == INDEX, f"normal_ on flagos:{t.device.index}"
    c = t.cpu()
    assert abs(c.mean().item()) < 0.15, c.mean().item()
    assert 0.8 < c.std().item() < 1.2, c.std().item()


# ---------------------------------------------------------------------------
# CPU-scalar promotion
# ---------------------------------------------------------------------------


def test_cpu_scalar_operand_lands_on_operand_device():
    """A CPU scalar operand is hopped to the device by TensorToPython.

    That hop resolves to the *current* device, so without the guard it landed on
    device 0 while the other operand sat on device 1 -- a mixed-device call into
    gems. Under the guard the current device is the operand's, so both agree.
    """
    cpu = torch.randn(32)
    out = cpu.to(DEVICE) * torch.tensor(2.0)
    _check(out, cpu * 2.0, "mul with cpu scalar")


# ---------------------------------------------------------------------------
# The current device must be left as it was found
# ---------------------------------------------------------------------------


def test_current_device_is_restored():
    """The guard is scoped: an op on flagos:1 must not leave 1 current."""
    assert torch_fl.flagos.current_device() == 0
    torch.abs(torch.randn(16, device=DEVICE))
    assert torch_fl.flagos.current_device() == 0, (
        "an op on flagos:1 leaked the device switch to the caller"
    )
