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

"""Native MPSGraph operations on Qwen3.5's GatedDeltaNet path.

The reference Qwen3.5 model performs a grouped 1-D convolution, indexes a
recurrent state with dense integer tensors, solves a batched triangular system
and builds a bool causal mask. These tests pin those operations in the shapes
and dtypes the model presents and require each accepted call to dispatch to MPS.
"""

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import dispatched_to_mps
from platform_support import detect_platform
from run_in_subprocess import run_python_source


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]


def _to_device(value):
    return value.clone().to(DEVICE)


def _assert_close(metal, cpu, **tolerance):
    assert metal.device.type == "flagos"
    assert metal.dtype == cpu.dtype
    assert metal.shape == cpu.shape
    torch.testing.assert_close(metal.cpu(), cpu, **tolerance)


@pytest.mark.anyplatform
def test_dense_conv1d_uses_the_grouped_graph():
    """The Qwen3.5 mixer performs a padded, grouped, dense conv1d."""
    torch.manual_seed(0)
    hidden_size = 32
    host_x = torch.randn(1, hidden_size, 5).to(torch.bfloat16)
    host_w = torch.randn(hidden_size, 1, 4).to(torch.bfloat16)
    out = torch.nn.functional.conv1d(
        _to_device(host_x),
        _to_device(host_w),
        padding=3,
        groups=hidden_size,
    )
    expected = torch.nn.functional.conv1d(host_x, host_w, padding=3, groups=hidden_size)
    _assert_close(out, expected, rtol=5e-3, atol=1e-3)
    source = """
import contextlib, io, torch, torch_fl
x = torch.randn(1, 32, 5, device='flagos:0', dtype=torch.bfloat16)
w = torch.randn(32, 1, 4, device='flagos:0', dtype=torch.bfloat16)
with contextlib.redirect_stderr(io.StringIO()) as trace:
    torch.nn.functional.conv1d(x, w, padding=3, groups=32)
    torch_fl._C._synchronize()
print(trace.getvalue(), end='')
"""
    assert dispatched_to_mps(source) >= 1


@pytest.mark.anyplatform
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("left", [False, True])
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [
        (torch.float32, 2e-5),
        (torch.float16, 5e-3),
        (torch.bfloat16, 2e-2),
    ],
)
@pytest.mark.parametrize("unitriangular", [False, True])
def test_batched_triangular_solve_uses_the_graph(
    upper, left, dtype, tolerance, unitriangular
):
    """Batched, opposite-side and unit-diagonal solves keep aten semantics."""
    matrix_shape = (3, 5, 5)
    a = torch.randn(matrix_shape, dtype=dtype)
    a = (
        a.tril() + 5.0 * torch.eye(5, dtype=dtype)
        if not upper
        else a.triu() + 5.0 * torch.eye(5, dtype=dtype)
    )
    b_shape = (3, 5, 2) if left else (3, 2, 5)
    b = torch.randn(b_shape, dtype=torch.float32).to(dtype)
    host_a = _to_device(a)
    device_b = _to_device(b)
    out = torch.linalg.solve_triangular(
        host_a, device_b, upper=upper, left=left, unitriangular=unitriangular
    )
    source = f"""
import torch, torch_fl
a = {"torch.triu" if upper else "torch.tril"}(
    torch.randn({matrix_shape}, dtype={dtype!r}, device='flagos:0'))
b = torch.randn({b_shape}, dtype={dtype!r}, device='flagos:0')
torch.linalg.solve_triangular(
    a, b, upper={upper!r}, left={left!r}, unitriangular={unitriangular!r}
)
torch_fl._C._synchronize()
"""
    result = run_python_source(source, {"FLAGOS_LOG": "dispatch,fallback"})
    assert "[flagos dispatch] linalg_solve_triangular -> mps" in result.stderr
    assert "[flagos cpu_fallback] aten::linalg_solve_triangular" not in result.stderr
    expected = torch.linalg.solve_triangular(
        a.float(),
        b.float(),
        upper=upper,
        left=left,
        unitriangular=unitriangular,
    ).to(dtype)
    _assert_close(out, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.anyplatform
def test_checkpoint_shaped_bfloat16_triangular_solve():
    """The (batch, head, one, n) right-hand side is explicit in the plan key."""
    a = (torch.randn(1, 16, 64, 64).tril() + 2.0 * torch.eye(64)).bfloat16()
    b = torch.randn(1, 16, 1, 64, dtype=torch.bfloat16)
    out = torch.linalg.solve_triangular(
        _to_device(a), _to_device(b), upper=False, left=False, unitriangular=True
    )
    expected = torch.linalg.solve_triangular(
        a.float(), b.float(), upper=False, left=False, unitriangular=True
    ).bfloat16()
    _assert_close(out, expected, rtol=5e-3, atol=1e-3)


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    "source",
    [
        "import torch, torch_fl\n"
        "a = torch.eye(64, device='flagos:0').expand(1, 2, 64, 64)\n"
        "b = torch.randn(1, 2, 1, 64, device='flagos:0', dtype=torch.bfloat16)\n"
        "torch.linalg.solve_triangular(a, b, upper=False, left=False)\n"
        "torch_fl._C._synchronize()\n",
        "import torch, torch_fl\n"
        "a = torch.tensor([[2.0, 0.0], [1.0, 3.0]], device='flagos:0')\n"
        "b = torch.tensor([[2.0], [7.0]], device='flagos:0')\n"
        "torch.linalg.solve_triangular(a, b, upper=False)\n"
        "torch_fl._C._synchronize()\n",
    ],
)
def test_triangular_solve_dispatches_to_mps(source):
    result = run_python_source(source, {"FLAGOS_LOG": "dispatch,fallback"})
    assert "[flagos dispatch] linalg_solve_triangular -> mps" in result.stderr
    assert "[flagos cpu_fallback] aten::linalg_solve_triangular" not in result.stderr
