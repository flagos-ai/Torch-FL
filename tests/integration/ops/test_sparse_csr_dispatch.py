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

"""Compressed sparse (CSR/CSC/BSR/BSC) on the flagos device (issue #293).

Construction and ``torch._C._dispatch_keys`` already worked for
``torch.sparse_csr_tensor(..., device="flagos:0")`` -- the tensor carries
``SparseCsrPrivateUse1`` -- but nothing served that key, so the accessors fell
through to the ``CompositeExplicitAutograd`` defaults and raised the *layout*
error (``crow_indices expected sparse row compressed tensor layout``) while
``empty.memory_format`` had no default at all.  csrc/aten/sparse_csr_ops.cc
registers the structure surface and layout entry points for that key.

Only that surface is asserted here.  The compressed sparse *value* kernels
(``_sparse_csr_sum``/``_sparse_csr_prod``, the sparse arithmetic family,
``torch.sparse.mm(..., reduce="sum")``) stay outside the change's scope and
are deliberately not asserted, so this file reads as a statement of what is
supported rather than of what is not.

Every flagos result is compared against the same construction on CPU.
"""

import pytest
import torch

import torch_fl  # noqa: F401
from backend_conf import routed_backend_or_none

DEVICE = "flagos:0"

# Densifying a compressed tensor lays its stored values into a zero-filled
# dense tensor with `index_add_`, and every generated conf routes `index_add_`
# to FlagGems.  The MThreads build divides `src.numel()` by `index.numel()`
# without guarding the empty index
# (flag_gems/runtime/backend/_mthreads/ops/index_add.py:157), so a structure
# with nothing stored divides by zero there.  The Hygon build is not exposed:
# `_hygon/ops/index_add.py` declines its contiguous-suffix fast path when
# `src.numel() == 0` (line 248), so the same densification passes on the DCU
# host.  That is FlagGems' defect rather than this surface's, and not a sparse
# one at all -- any empty-index `index_add_` on MUSA reaches the same line --
# so the two empty-structure densifications below are xfailed rather than
# routed around.  Non-strict, and narrowed to the one exception type, so a
# fixed kernel reports an xpass and anything else stays a failure.  The
# predicate reads the build's accelerator rather than the conf, so a conf that
# cannot be found cannot quietly drop the mark and turn the job red again.
#
# The guard reads the routing table rather than the accelerator name. MUSA is
# the platform the divide was found on, but what reaches the defect is the
# `index_add_` route: on the MUSA lane the sparse surface was reached through
# `TorchFunctionMode` before `torch_fl` had activated, and the parameterized
# `index_add` slot answered while `index_add_` -- routed `flaggems`, with the
# divide in it -- was not. A router that skips `index_add_` cannot reach the
# line at all, so the cases run for real there instead of reporting an xfail
# for a defect they never touch.


def _routes_to_flaggems(op):
    return routed_backend_or_none(op) == "flagos_python"


FLAGGEMS_EMPTY_INDEX_ADD = _routes_to_flaggems("index_add_")

empty_densification_xfail = pytest.mark.xfail(
    FLAGGEMS_EMPTY_INDEX_ADD,
    reason="FlagGems' MThreads index_add_ divides by a zero-length index "
    "(flag_gems/runtime/backend/_mthreads/ops/index_add.py:157)",
    strict=False,
    raises=ZeroDivisionError,
)

# The 2x2 identity every plain CSR/CSC construction below uses.
CROW = [0, 1, 2]
CCOL = [0, 1, 2]
COL = [0, 1]
ROW = [0, 1]
VALUES = [1.0, 1.0]
IDENTITY = torch.tensor([[1.0, 0.0], [0.0, 1.0]])


def _csr(device=DEVICE, dtype=torch.float32):
    return torch.sparse_csr_tensor(
        CROW, COL, VALUES, (2, 2), dtype=dtype, device=device
    )


def _csc(device=DEVICE, dtype=torch.float32):
    return torch.sparse_csc_tensor(
        CCOL, ROW, VALUES, (2, 2), dtype=dtype, device=device
    )


def _keys(x):
    return str(torch._C._dispatch_keys(x))


def _dense(x):
    """Densify for comparison. `_to_dense` lays values into a dense zeros
    tensor with `index_add_`, all of which act on the dense operand."""
    return x.to_dense()


class TestConstruction:
    @pytest.mark.anyplatform
    def test_issue_293_reproducer(self):
        """The accessor and allocation checks from the report must work."""
        csr = _csr()
        csc = _csc()
        assert (csc.layout, csr.layout) == (torch.sparse_csc, torch.sparse_csr)
        torch.testing.assert_close(csc.ccol_indices().cpu(), torch.tensor(CCOL))
        torch.testing.assert_close(csr.crow_indices().cpu(), torch.tensor(CROW))
        allocated = torch.empty((2, 2), layout=torch.sparse_csr, device=DEVICE)
        assert allocated.layout == torch.sparse_csr

    @pytest.mark.anyplatform
    def test_dispatch_key_is_sparse_csr_privateuse1(self):
        """The tensor must really be a SparseCsrPrivateUse1 tensor.

        A tensor that merely reported the right layout while sitting on the
        dense key would pass a shape assertion and fail everywhere else.
        """
        assert "SparseCsrPrivateUse1" in _keys(_csr())
        assert "SparseCsrPrivateUse1" in _keys(_csc())

    @pytest.mark.anyplatform
    def test_index_and_value_tensors_stay_on_device(self):
        """Construction must not silently stage the buffers through the host."""
        csr = _csr()
        for member in (csr.crow_indices(), csr.col_indices(), csr.values()):
            assert member.device == csr.device

    @pytest.mark.anyplatform
    def test_csr_accessors_match_cpu(self):
        gpu, cpu = _csr(), _csr("cpu")
        torch.testing.assert_close(gpu.crow_indices().cpu(), cpu.crow_indices())
        torch.testing.assert_close(gpu.col_indices().cpu(), cpu.col_indices())
        torch.testing.assert_close(gpu.values().cpu(), cpu.values())

    @pytest.mark.anyplatform
    def test_csc_accessors_match_cpu(self):
        """CSC reads the column-compressed pair; getting the layout wrong here
        is what the CompositeExplicitAutograd default used to raise for."""
        gpu, cpu = _csc(), _csc("cpu")
        torch.testing.assert_close(gpu.ccol_indices().cpu(), cpu.ccol_indices())
        torch.testing.assert_close(gpu.row_indices().cpu(), cpu.row_indices())
        torch.testing.assert_close(gpu.values().cpu(), cpu.values())

    @pytest.mark.anyplatform
    def test_empty_allocation(self):
        """`empty.memory_format` has no CompositeExplicitAutograd default, so
        this is the op the report's allocation check actually failed on."""
        for layout in (torch.sparse_csr, torch.sparse_csc):
            allocated = torch.empty((2, 2), layout=layout, device=DEVICE)
            assert allocated.layout == layout
            assert str(allocated.device) == DEVICE

    @pytest.mark.anyplatform
    def test_empty_rejects_block_layouts(self):
        """ATen's sparse `empty` is non-block compressed only, so the block
        layouts must be refused here exactly as they are on the host rather
        than quietly allocating something else."""
        for layout in (torch.sparse_bsr, torch.sparse_bsc):
            with pytest.raises(RuntimeError, match="block"):
                torch.empty((4, 4), layout=layout, device=DEVICE)
            with pytest.raises(RuntimeError, match="block"):
                torch.empty((4, 4), layout=layout)

    @pytest.mark.anyplatform
    def test_zeros_allocation(self):
        """zeros must build the structure, and it must hold nothing: a `zeros`
        that allocated values would be a dense tensor wearing a sparse layout."""
        zero = torch.zeros((2, 2), layout=torch.sparse_csr, device=DEVICE)
        assert zero.layout == torch.sparse_csr
        assert zero._nnz() == 0
        assert zero.values().numel() == 0

    @pytest.mark.anyplatform
    @empty_densification_xfail
    def test_zeros_densify_to_zeros(self):
        zero = torch.zeros((2, 2), layout=torch.sparse_csr, device=DEVICE)
        torch.testing.assert_close(_dense(zero).cpu(), torch.zeros(2, 2))

    @pytest.mark.anyplatform
    def test_empty_like_keeps_layout_and_device(self):
        copy = torch.empty_like(_csr())
        assert copy.layout == torch.sparse_csr
        assert copy.device == _csr().device

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
    def test_supported_value_dtypes(self, dtype):
        csr = _csr(dtype=dtype)
        assert csr.dtype == dtype
        assert csr.values().dtype == dtype


class TestStructureQueries:
    @pytest.mark.anyplatform
    def test_shape_and_dims(self):
        csr = _csr()
        assert tuple(csr.shape) == (2, 2)
        assert csr.dim() == 2
        assert csr.numel() == 4
        assert csr.sparse_dim() == 2
        assert csr.dense_dim() == 0
        assert csr._nnz() == 2

    @pytest.mark.anyplatform
    def test_layout_mismatched_accessor_raises(self):
        """The layout error must still be a layout error, not a crash."""
        for tensor, name in ((_csr(), "ccol_indices"), (_csc(), "crow_indices")):
            with pytest.raises(RuntimeError, match="layout"):
                getattr(tensor, name)()

    @pytest.mark.anyplatform
    def test_repr_reports_layout_and_device(self):
        text = repr(_csr())
        assert "sparse_csr" in text
        assert "flagos" in text


class TestLayoutConversion:
    @pytest.mark.anyplatform
    def test_to_dense_matches_cpu(self):
        """Densification goes through `_convert_indices_from_csr_to_coo` and
        `index_add_` on the dense operand, so it needs no vendor sparsity."""
        for tensor in (_csr(), _csc()):
            torch.testing.assert_close(_dense(tensor).cpu(), IDENTITY)


class TestBlockLayouts:
    """BSR/BSC, the layouts that carry a block size.

    Construction, densification and the two conversions that run on this
    tree are asserted.  `Tensor.to_sparse_bsr` on a compressed input is not:
    ATen routes it to `_compressed_to_block_compressed_cpu`, which segfaults
    on this wheel, and it does so with plain `python` and no flagos device at
    all, so it is a property of the wheel rather than of this change.
    """

    @pytest.mark.anyplatform
    def test_bsr_construction_and_accessors(self):
        """The block size is not an argument -- it is the shape of the
        trailing dimensions of the values tensor, which is
        `(num_blocks, block_rows, block_cols, *dense_shape)`."""
        values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        crow, col = [0, 1, 2], [0, 1]
        gpu = torch.sparse_bsr_tensor(crow, col, values, (4, 4), device=DEVICE)
        cpu = torch.sparse_bsr_tensor(crow, col, values, (4, 4))
        assert gpu.layout == torch.sparse_bsr
        assert str(gpu.device) == DEVICE
        assert gpu._nnz() == cpu._nnz() == 2
        assert tuple(gpu.values().shape) == tuple(values.shape)
        torch.testing.assert_close(gpu.crow_indices().cpu(), cpu.crow_indices())
        torch.testing.assert_close(gpu.col_indices().cpu(), cpu.col_indices())
        torch.testing.assert_close(gpu.values().cpu(), cpu.values())
        torch.testing.assert_close(_dense(gpu).cpu(), _dense(cpu))

    @pytest.mark.anyplatform
    def test_strided_to_sparse_bsr_matches_cpu(self):
        """A dense input takes the dense->sparse route rather than the
        compressed one, so nothing here goes near the stub or cuSPARSE and
        the test runs on every platform."""
        gpu = torch.eye(4, device=DEVICE).to_sparse_bsr((2, 2))
        cpu = torch.eye(4).to_sparse_bsr((2, 2))
        assert gpu.layout == torch.sparse_bsr
        assert str(gpu.device) == DEVICE
        torch.testing.assert_close(_dense(gpu).cpu(), _dense(cpu))


class TestBufferMovement:
    @pytest.mark.anyplatform
    def test_clone_matches_cpu_reference(self):
        gpu, cpu = _csr(), _csr("cpu")
        cloned = gpu.clone()
        assert cloned.layout == torch.sparse_csr
        assert cloned.device == gpu.device
        assert cloned._nnz() == gpu._nnz()
        torch.testing.assert_close(_dense(cloned).cpu(), _dense(cpu))

    @pytest.mark.anyplatform
    def test_clone_does_not_alias_the_values(self):
        """A clone that shared storage would make later writes visible twice."""
        csr = _csr()
        csr.clone().values().fill_(0.0)
        torch.testing.assert_close(csr.values().cpu(), torch.tensor(VALUES))

    @pytest.mark.anyplatform
    def test_copy_between_flagos_tensors(self):
        source = _csr()
        target = torch.sparse_csr_tensor(CROW, COL, [0.0, 0.0], (2, 2), device=DEVICE)
        target.copy_(source)
        assert target.device == source.device
        torch.testing.assert_close(_dense(target).cpu(), IDENTITY)

    @pytest.mark.anyplatform
    def test_copy_rejects_a_different_structure(self):
        """ATen's compressed `copy_` moves values between two tensors of the
        same shape, so a target with a different `_nnz` is refused -- here
        exactly as on the host, and not by silently reallocating it."""
        source = _csr()
        empty = torch.zeros((2, 2), layout=torch.sparse_csr)
        for target in (
            torch.zeros((2, 2), layout=torch.sparse_csr, device=DEVICE),
            empty,
        ):
            with pytest.raises(RuntimeError, match="specified elements"):
                target.copy_(source.to(target.device))

    @pytest.mark.anyplatform
    def test_cpu_roundtrip_matches_cpu_reference(self):
        gpu = _csr()
        moved = gpu.to("cpu")
        assert moved.layout == torch.sparse_csr
        assert moved.device.type == "cpu"
        back = moved.to(DEVICE)
        assert back.layout == torch.sparse_csr
        assert back.device == gpu.device
        torch.testing.assert_close(_dense(back).cpu(), _dense(moved))

    @pytest.mark.anyplatform
    def test_zero_empties_the_structure(self):
        """`zero_` on a compressed tensor discards the stored values rather
        than writing zeros into them -- the host does the same -- so the
        result is a structure with no non-zeros, not a dense zero.  The index
        tensors shrink with it: the host leaves all-zero row pointers, not the
        original pattern with zeroed values behind it."""
        csr = _csr()
        csr.zero_()
        assert csr._nnz() == 0
        assert csr.values().numel() == 0
        torch.testing.assert_close(
            csr.crow_indices().cpu(), torch.zeros(len(CROW), dtype=torch.int64)
        )

    @pytest.mark.anyplatform
    @empty_densification_xfail
    def test_zeroed_structure_densifies_to_zeros(self):
        """`zero_`'s result is a compressed tensor with nothing to lay in, so
        it densifies through the same empty `index_add_` as `zeros` does."""
        csr = _csr()
        csr.zero_()
        torch.testing.assert_close(_dense(csr).cpu(), torch.zeros(2, 2))
