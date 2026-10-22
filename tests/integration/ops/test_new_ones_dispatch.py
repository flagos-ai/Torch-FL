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
new_ones dispatch tests

Verifies that tensor.new_ones:
  - produces correct results on flagos device
  - C++ wrapper routes to cuda/metax backend
  - on GCU, reaches the op's own route rather than FlagGems' `ones` kernel, and
    returns ones for every dtype the vendor entry point declines

Usage:
    pytest tests/integration/ops/test_new_ones_dispatch.py -v
"""

import os
import pytest
import subprocess
import sys

import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"

# The package this session imported, which is not necessarily the one under the
# working directory. `python -c` puts the process's cwd first on `sys.path`, so a
# child launched from a checkout whose prebuilt `torch_fl/_C*.so` is stale imports
# that one instead -- and a route assertion is a claim about the extension that is
# loaded, so the two have to be the same build. A tree whose build is current gets
# the same answer either way; the pin only removes the ambient dependency.
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(torch_fl.__file__)))


def _run_child(code: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Run `code` in a child pinned to the `torch_fl` this session imported."""
    env = os.environ.copy()
    env.update(extra_env or {})
    pinned = f"import sys; sys.path.insert(0, {_PACKAGE_ROOT!r})\n{code}"
    return subprocess.run(
        [sys.executable, "-c", pinned],
        env=env,
        capture_output=True,
        text=True,
    )


def _run_subprocess(extra_env: dict, check: bool = True) -> subprocess.CompletedProcess:
    code = (
        "import torch_fl, torch; "
        "x = torch.randn(4,4,device='flagos:0'); "
        "x.new_ones(3, 3)"
    )
    return _run_child(code, extra_env)


class TestNewOnesCorrectness:
    """tensor.new_ones correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_basic(self):
        x = torch.randn(4, 4, device=DEVICE)
        out = x.new_ones(3, 3)
        assert out.shape == (3, 3)
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), torch.ones(3, 3))

    @pytest.mark.anyplatform
    def test_preserves_dtype(self):
        x = torch.randn(4, 4, device=DEVICE, dtype=torch.float16)
        out = x.new_ones(2, 2)
        assert out.dtype == torch.float16
        torch.testing.assert_close(out.cpu().float(), torch.ones(2, 2))

    @pytest.mark.anyplatform
    def test_override_dtype(self):
        x = torch.randn(4, 4, device=DEVICE)
        out = x.new_ones(2, 2, dtype=torch.int32)
        assert out.dtype == torch.int32
        torch.testing.assert_close(out.cpu(), torch.ones(2, 2, dtype=torch.int32))

    @pytest.mark.anyplatform
    def test_all_ones(self):
        x = torch.randn(8, device=DEVICE)
        out = x.new_ones(100)
        assert (out.cpu() == 1).all()


class TestNewOnesDispatch:
    """Verify dispatch routing."""

    @pytest.mark.cuda
    def test_dispatch_log_cuda(self):
        result = _run_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_new_ones": "cuda"}
        )
        if result.returncode != 0 and "backend not registered" in result.stderr:
            pytest.skip("cuda backend not available in this build")
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] new_ones -> cuda" in result.stderr

    @pytest.mark.metax
    def test_dispatch_log_metax(self):
        result = _run_subprocess(
            {"FLAGOS_LOG": "dispatch", "FLAGOS_OP_new_ones": "metax"}
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] new_ones -> metax" in result.stderr


class TestNewOnesAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify new_ones on ascend backend matches CPU reference."""
        result = _run_subprocess({"FLAGOS_OP_new_ones": "ascend"})
        assert result.returncode == 0


class TestNewOnesGcuRoute:
    """The route `new_ones` takes on GCU, and the limits of the kernel behind it.

    The op is registered natively on this platform, so the route is asserted and
    not inferred from the result. The case that matters is an int64 result: it
    used to leave for FlagGems, whose `new_ones` is a wrapper over its `ones`
    kernel, and FlagTree cannot lower that kernel's int64 instantiation for
    GCU300, so the call aborted in the compiler pipeline instead of returning a
    value. A test that only compared values would keep passing on a build where
    the native registration had gone away and the op was back on FlagGems.

    `topsatenNewOnes` takes its result dtype as an argument and declines
    everything except fp32/fp16/bf16, so the dtypes below split into two groups
    with different evidence: the accepted three are written by the vendor op,
    and the declined ones fall back inside the kernel to
    `at::compositeexplicitautograd::new_ones` -- `empty` + `fill_`, the same
    decomposition the `none` route reached. The dispatch log can tell them
    apart, because only the fallback runs a `fill_`.
    """

    @pytest.mark.gcu
    def test_int64_route_is_gcu_and_not_flaggems(self):
        result = _run_child(
            "import torch, torch_fl\n"
            "x = torch.zeros(2, 4, device='flagos:0', dtype=torch.int64)\n"
            "out = x.new_ones((2, 4))\n"
            "assert out.dtype == torch.int64, out.dtype\n"
            "assert out.device.type == 'flagos', out.device\n"
            "assert out.cpu().eq(1).all().item()\n"
            "print('ok')\n",
            {"FLAGOS_LOG": "dispatch"},
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] new_ones -> gcu" in result.stderr
        assert "flagos_python" not in result.stderr

    @pytest.mark.gcu
    def test_accepted_dtypes_skip_the_composite(self):
        result = _run_child(
            "import torch, torch_fl\n"
            "x = torch.zeros(2, 4, device='flagos:0', dtype=torch.float32)\n"
            "for dtype in (torch.float32, torch.float16, torch.bfloat16):\n"
            "    out = x.new_ones((2, 4), dtype=dtype)\n"
            "    assert out.dtype == dtype, (dtype, out.dtype)\n"
            "    assert out.device.type == 'flagos', (dtype, out.device)\n"
            "print('ok')\n",
            {"FLAGOS_LOG": "dispatch"},
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] new_ones -> gcu" in result.stderr
        # The vendor entry point writes the plane itself; nothing fills.
        assert "fill_.Scalar" not in result.stderr

    @pytest.mark.gcu
    def test_declined_dtypes_take_the_composite_on_the_device(self):
        result = _run_child(
            "import torch, torch_fl\n"
            "x = torch.zeros(2, 4, device='flagos:0', dtype=torch.int64)\n"
            "for dtype in (torch.int64, torch.int32, torch.int8, torch.bool,\n"
            "              torch.float64):\n"
            "    out = x.new_ones((2, 4), dtype=dtype)\n"
            "    assert out.dtype == dtype, (dtype, out.dtype)\n"
            "    assert out.device.type == 'flagos', (dtype, out.device)\n"
            "    assert out.cpu().eq(1).all().item(), dtype\n"
            "print('ok')\n",
            {"FLAGOS_LOG": "dispatch"},
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        # `fill_`, not `empty`-then-copy: the fallback must not round-trip the host.
        assert "[flagos dispatch] fill_.Scalar -> gcu" in result.stderr
        assert "cpu_fallback" not in result.stderr

    @pytest.mark.gcu
    def test_rank_zero_and_empty_size_do_not_abort(self):
        # topsaten rejects an empty dims/strides vector by throwing, which aborts
        # the process rather than propagating, so both shapes have to be kept
        # away from the vendor entry point.
        result = _run_child(
            "import torch, torch_fl\n"
            "x = torch.zeros(2, 4, device='flagos:0', dtype=torch.int64)\n"
            "zero_rank = x.new_ones(())\n"
            "assert zero_rank.shape == (), zero_rank.shape\n"
            "assert zero_rank.cpu().item() == 1\n"
            "empty = x.new_ones((0, 4))\n"
            "assert empty.shape == (0, 4), empty.shape\n"
            "assert empty.numel() == 0\n"
            "print('ok')\n"
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"

    @pytest.mark.gcu
    def test_pin_memory_is_declined_rather_than_silently_ignored(self):
        x = torch.zeros(2, 4, device=DEVICE, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="Pin memory can only be on CPU"):
            x.new_ones(3, pin_memory=True)

    @pytest.mark.gcu
    @pytest.mark.parametrize("self_dtype", [torch.int64, torch.float32, torch.bool])
    def test_result_dtype_follows_the_argument_not_the_operand(self, self_dtype):
        x = torch.zeros(2, 4, device=DEVICE, dtype=self_dtype)
        native = x.new_ones((2, 4), dtype=torch.float32)
        assert native.dtype == torch.float32
        assert native.device.type == "flagos"
        torch.testing.assert_close(native.cpu(), torch.ones(2, 4))
        composite = x.new_ones((2, 4))
        assert composite.dtype == self_dtype
        assert composite.device.type == "flagos"
        assert composite.cpu().eq(1).all().item()
