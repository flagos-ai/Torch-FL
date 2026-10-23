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

"""DCU complex-dtype routing and device-residency regression tests (issue #540).

Two separate defects met on the shipped DCU wheel:

* the wheel's dtype converter had no complex arm, so a factory that asked for
  ``complex64``/``complex128`` reached FlagGems with ``dtype=None`` and came
  back as ``float32`` -- a silent wrong answer, not an error;
* FlagGems' HCU Triton backend has no complex pointer type, so even with the
  dtype preserved the call fails inside the compiler.

The fix names complex in the converter and adds DCU to the runtime dtype
escape, which reroutes the complex call to the CUDA boxing kernels. The other
half of the contract is that the escape must be *dtype-scoped*: the float route
stays exactly where the conf put it.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl


pytestmark = pytest.mark.main_ops


def _require_dcu() -> None:
    if not torch_fl.backend_config_path().endswith("backends_hygon.conf"):
        pytest.skip("requires the DCU boxing runtime")
    if torch_fl.flagos.device_count() < 1:
        pytest.skip("DCU device is unavailable")


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_zeros_keeps_complex_dtype(dtype):
    """The reported defect: an explicit complex dtype used to become float32."""
    _require_dcu()
    tensor = torch.zeros(4, dtype=dtype, device="flagos:0")
    assert tensor.dtype == dtype
    assert tensor.device.type == "flagos"
    torch.testing.assert_close(tensor.cpu(), torch.zeros(4, dtype=dtype))


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_randn_keeps_complex_dtype(dtype):
    """Not named in the issue, but the same converter path."""
    _require_dcu()
    tensor = torch.randn(4, dtype=dtype, device="flagos:0")
    assert tensor.dtype == dtype
    assert tensor.device.type == "flagos"


@pytest.mark.parametrize(
    "dtype,real_dtype",
    ((torch.complex64, torch.float32), (torch.complex128, torch.float64)),
)
def test_abs_of_complex_is_real_and_correct(dtype, real_dtype):
    """abs() of a complex tensor is the matching-precision real dtype."""
    _require_dcu()
    host = torch.tensor([3 + 4j, -1 - 1j, 0 + 2j], dtype=dtype)
    result = torch.abs(host.to("flagos:0"))
    assert result.dtype == real_dtype
    assert result.device.type == "flagos"
    torch.testing.assert_close(result.cpu(), torch.abs(host))


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_ones_and_fill_keep_complex_dtype(dtype):
    """`ones` composes through `fill_`; both divert through the same escape."""
    _require_dcu()
    tensor = torch.ones(4, dtype=dtype, device="flagos:0")
    assert tensor.dtype == dtype
    assert tensor.device.type == "flagos"
    torch.testing.assert_close(tensor.cpu(), torch.ones(4, dtype=dtype))

    filled = torch.empty(3, dtype=dtype, device="flagos:0").fill_(1 + 2j)
    assert filled.dtype == dtype
    torch.testing.assert_close(filled.cpu(), torch.full((3,), 1 + 2j, dtype=dtype))


def test_complex_escape_leaves_the_float_route_alone():
    """The runtime dtype escape must be dtype-scoped, not route-wide."""
    _require_dcu()
    code = """
import torch_fl
import torch

for dtype in (torch.float32, torch.complex64):
    value = torch.ones(4, dtype=dtype, device="flagos:0")
    torch.abs(value)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "FLAGOS_LOG": "dispatch"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    routes = [
        line
        for line in result.stderr.splitlines()
        if line.startswith("[flagos dispatch] abs -> ")
    ]
    assert routes == [
        "[flagos dispatch] abs -> flagos_python",
        "[flagos dispatch] abs -> boxing",
    ], result.stderr
