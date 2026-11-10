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
all dispatch tests

Verifies that torch.all:
  - produces correct results on flagos device
  - configured routes are reported by the dispatch log
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_all_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import (
    dispatched_to_mps,
    override_is_logged,
    routed_backend_or_none,
    route_is_served,
)

from tests.integration.case_gap_issues import tracked_skip


DEVICE = "flagos:0"

# This file is part of the shared case cohort. Every operator lane selects
# `main_ops`, including Apple's backend lane and Ascend's vendor lane.
pytestmark = pytest.mark.main_ops


def _flag_gems_route_serves(op: str) -> bool:
    """Whether this wheel serves ``op`` through the FlagGems Python path."""
    return routed_backend_or_none(op) == "flagos_python" and route_is_served(op)


def _run_all_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.ones(4,4,device='flagos:0', dtype=torch.bool); "
        "torch.all(a)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestAllCorrectness:
    """torch.all correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_all_true(self):
        a = torch.ones(32, 32, device=DEVICE, dtype=torch.bool)
        out = torch.all(a)
        assert out.item() is True
        assert out.dtype == torch.bool

    @pytest.mark.anyplatform
    def test_all_false(self):
        a = torch.zeros(32, 32, device=DEVICE, dtype=torch.bool)
        out = torch.all(a)
        assert out.item() is False

    @pytest.mark.anyplatform
    def test_all_mixed(self):
        a = torch.ones(32, 32, device=DEVICE, dtype=torch.bool)
        a[0, 0] = False
        out = torch.all(a)
        assert out.item() is False

    @tracked_skip(
        condition=_flag_gems_route_serves("all"),
        issue="#710",
    )
    @pytest.mark.anyplatform
    def test_all_empty_tensor(self):
        """Empty tensor: all() is vacuously true.

        This global overload has its own conf route, so the tracked gap is
        attached only when that route is FlagGems. Other backends run it.
        """
        a = torch.tensor([], device=DEVICE, dtype=torch.bool)
        out = torch.all(a)
        assert out.item() is True

    @pytest.mark.anyplatform
    @pytest.mark.parametrize(
        "shape",
        [
            (0, 3),
            (2, 0, 3),
            (0, 2),
        ],
    )
    @pytest.mark.parametrize("dim", [0, 1, -1])
    @tracked_skip(
        condition=_flag_gems_route_serves("all.dim"),
        issue="#710",
    )
    def test_all_empty_dimensions_dim(self, shape, dim):
        """Reductions along empty dimensions keep the aten result and shape."""
        host = torch.empty(shape, dtype=torch.float32)
        expected = torch.all(host, dim=dim)
        actual = torch.all(host.to(DEVICE), dim=dim)
        assert actual.shape == expected.shape
        assert actual.dtype == torch.bool
        assert torch.equal(actual.cpu(), expected)

    @tracked_skip(
        condition=_flag_gems_route_serves("all.dims"),
        issue="#710",
    )
    @pytest.mark.anyplatform
    @pytest.mark.parametrize(
        "shape, dims",
        [
            ((0, 3), (0, 1)),
            ((2, 0, 3), (0, 2)),
        ],
    )
    def test_all_empty_dimensions_dims(self, shape, dims):
        """The tuple-of-dimensions overload preserves aten's empty result."""
        host = torch.empty(shape, dtype=torch.float32)
        expected = torch.all(host, dim=dims)
        actual = torch.all(host.to(DEVICE), dim=dims)
        assert actual.shape == expected.shape
        assert actual.dtype == torch.bool
        assert torch.equal(actual.cpu(), expected)

    @tracked_skip(
        condition=_flag_gems_route_serves("all.dim"),
        issue="#710",
    )
    @pytest.mark.anyplatform
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_all_dim_empty_reduction_reaches_its_route(self, keepdim):
        """The dimension overload is tested when the global reduction has no route."""
        host = torch.empty((3, 0), dtype=torch.float32)
        expected = torch.all(host, dim=-1, keepdim=keepdim)
        actual = torch.all(host.to(DEVICE), dim=-1, keepdim=keepdim)
        assert actual.shape == expected.shape
        assert actual.dtype == torch.bool
        assert torch.equal(actual.cpu(), expected)

    @tracked_skip(
        condition=_flag_gems_route_serves("all"),
        issue="#710",
    )
    @pytest.mark.anyplatform
    @pytest.mark.parametrize("shape", [(0, 0), (3, 0), (0, 3)])
    def test_all_empty_full_reduction(self, shape):
        """Full reduction over an empty N-D tensor is vacuously true.

        The MUSA FlagGems route raises ``ZeroDivisionError: integer division or
        modulo by zero`` here: its ``all`` kernel computes
        ``triton.cdiv(n_elements, block_size)`` and derives ``block_size`` from
        the element count, which is 0. Owned by the same issue as the other
        empty-input cases above.
        """
        for dtype in (torch.bool, torch.float32):
            a = torch.empty(shape, device=DEVICE, dtype=dtype)
            assert torch.all(a).item() is True

    @tracked_skip(
        condition=_flag_gems_route_serves("all.dim"),
        issue="#710",
    )
    @pytest.mark.anyplatform
    def test_all_empty_dim_reduction(self):
        """Reducing an empty axis returns True for every output element."""
        host = torch.empty((0, 3), dtype=torch.bool)
        expected = torch.all(host, dim=0)
        out = torch.all(host.to(DEVICE), dim=0)
        assert out.shape == (3,)
        assert out.dtype == torch.bool
        assert torch.equal(out.cpu(), expected)

    @pytest.mark.anyplatform
    def test_all_numeric_types(self):
        """all() should work on numeric types (non-zero = True)."""
        torch.manual_seed(0)
        a = torch.randn(16, 16, device=DEVICE)
        out = torch.all(a)
        ref = torch.all(a.cpu())
        assert out.item() == ref.item()

    @pytest.mark.anyplatform
    def test_all_matches_cpu_reference(self):
        torch.manual_seed(1)
        host = torch.randint(0, 2, (64, 64), dtype=torch.bool)
        a = host.to(DEVICE)
        out = torch.all(a)
        assert out.item() == torch.all(host).item()

    @tracked_skip(
        condition=_flag_gems_route_serves("all"),
        issue="#710",
    )
    @pytest.mark.anyplatform
    def test_all_empty_results_keep_their_shape_global(self):
        """The global reduction on a tensor with an empty dimension is True."""
        host = torch.ones((2, 0, 3), dtype=torch.float32)
        actual = torch.all(host.to(DEVICE))
        expected = torch.all(host)
        assert actual.shape == expected.shape
        assert actual.dtype == torch.bool
        assert torch.equal(actual.cpu(), expected)

    @tracked_skip(
        condition=_flag_gems_route_serves("all.dim"),
        issue="#710",
    )
    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_all_empty_results_keep_their_shape_dim(self, dim):
        """The integer-dimension overload retains an empty result's shape."""
        host = torch.ones((2, 0, 3), dtype=torch.float32)
        actual = torch.all(host.to(DEVICE), dim=dim)
        expected = torch.all(host, dim=dim)
        assert actual.shape == expected.shape
        assert actual.dtype == torch.bool
        assert torch.equal(actual.cpu(), expected)

    @tracked_skip(
        condition=_flag_gems_route_serves("all.dims"),
        issue="#710",
    )
    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dims", [(0, 1), (1, 2)])
    def test_all_empty_results_keep_their_shape_dims(self, dims):
        """The tuple-dimension overload retains an empty result's shape."""
        host = torch.ones((2, 0, 3), dtype=torch.float32)
        actual = torch.all(host.to(DEVICE), dim=dims)
        expected = torch.all(host, dim=dims)
        assert actual.shape == expected.shape
        assert actual.dtype == torch.bool
        assert torch.equal(actual.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.bool, torch.float32, torch.int32])
    @pytest.mark.anyplatform
    def test_all_dtypes(self, dtype):
        if dtype == torch.bool:
            a = torch.ones(16, 16, device=DEVICE, dtype=dtype)
        elif dtype.is_floating_point:
            a = torch.randn(16, 16, device=DEVICE, dtype=dtype) + 1.0
        else:
            a = torch.randint(1, 10, (16, 16), device=DEVICE, dtype=dtype)
        out = torch.all(a)
        assert out.dtype == torch.bool

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_float_widths_follow_aten(self, dtype):
        host = torch.tensor([[1, 3], [4, 5]], dtype=dtype)
        actual = torch.all(host.to(DEVICE))
        assert actual.item() is torch.all(host).item()
        assert torch.all((host > 0).to(DEVICE)).item() is True

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dim", [0, -1, (0, 1)])
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_dim_and_keepdim_follow_aten(self, dim, keepdim):
        host = torch.tensor([[1, 3], [4, 5]], dtype=torch.float32)
        actual = torch.all(host.to(DEVICE), dim=dim, keepdim=keepdim)
        expected = torch.all(host, dim=dim, keepdim=keepdim)
        assert actual.shape == expected.shape
        assert actual.dtype == torch.bool
        assert torch.equal(actual.cpu(), expected)


class TestAllDispatch:
    """Verify dispatch routing for all op."""

    @pytest.mark.main_ops
    @pytest.mark.flaggems_python
    @tracked_skip(
        condition=not override_is_logged("all", "flaggems_python"),
        issue="#710",
    )
    def test_dispatch_log_flaggems_python(self):
        """The override selects the FlagGems Python path on a serving wheel."""
        assert route_is_served("all")
        result = _run_all_subprocess(
            {
                "FLAGOS_LOG": "dispatch",
                "FLAGOS_OP_all": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] all -> flagos_python" in result.stderr

    @pytest.mark.cuda
    @pytest.mark.main_ops
    def test_dispatch_log_cuda_override(self):
        result = _run_all_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_all": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] all -> boxing" in result.stderr

    @pytest.mark.mps
    def test_each_spelling_uses_mps(self):
        """Every public ``all`` spelling reaches Apple's MPS kernels."""
        source = """
import torch, torch_fl
x = torch.tensor([[1, 0], [3, 4]], dtype=torch.int32, device='flagos:0')
x = x.to(torch.float32)
torch.all(x)
torch.all(x, dim=0)
torch.all(x, dim=(0, 1), keepdim=True)
torch_fl._C._synchronize()
"""
        assert dispatched_to_mps(source) == 3


class TestAllAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify all on ascend backend matches CPU reference."""
        result = _run_all_subprocess({"FLAGOS_OP_all": "ascend"})
        assert result.returncode == 0
