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
registers the structure surface, the layout conversions and the matrix
multiply for that key.

Only that surface is asserted here.  The compressed sparse *value* kernels
(``_sparse_csr_sum``/``_sparse_csr_prod``, the sparse arithmetic family,
``torch.sparse.mm(..., reduce="sum")``) stay outside the change's scope and
are deliberately not asserted, so this file reads as a statement of what is
supported rather than of what is not.

Two groups need the CUDA-compatible libtorch and are compiled only where the
guard in csrc/aten/sparse_csr_ops.cc admits it: the CSC<->CSR and CSR->BSC
conversions, which need a ``PrivateUse1`` slot on ATen's ``flatten_indices``
stub, and the sparse matrix multiply, which computes its product in cuSPARSE.
They are skipped where the plugin was built without it, the same way the ops
themselves are.  The rest -- accessors, allocation, buffer movement,
densification, and the dense->BSR conversion -- reaches no vendor library and
is asserted everywhere.

Every flagos result is compared against the same construction on CPU.
"""

import pytest
import torch

import torch_fl  # noqa: F401

DEVICE = "flagos:0"

# The accelerators whose libtorch does not carry the cuSPARSE symbols or the
# DispatchStub CUDA slot the guarded registrations need.  This mirrors the
# `#if !defined(USE_ASCEND) && ...` guard in csrc/aten/sparse_csr_ops.cc; the
# C++ side is the source of truth and this only keeps the two in step.
NOT_CUDA_BOXING = frozenset({"ascend", "gcu", "musa", "bpu"})

requires_cuda_boxing = pytest.mark.skipif(
    torch_fl._build_accelerator() in NOT_CUDA_BOXING,
    reason="csrc/aten/sparse_csr_ops.cc compiles this for CUDA-boxing only",
)

# The 2x2 identity every plain CSR/CSC construction below uses.
CROW = [0, 1, 2]
CCOL = [0, 1, 2]
COL = [0, 1]
ROW = [0, 1]
VALUES = [1.0, 1.0]
IDENTITY = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

# A 3x3 with one entry per row, off the diagonal.  The identity's index
# tensors are already sorted in both orientations, so a conversion that
# dropped the sort order would still pass against it; this one's are not.
PERM_CROW = [0, 1, 2, 3]
PERM_COL = [1, 2, 0]
PERM_VALUES = [1.0, 2.0, 3.0]

# A 4x4 with six non-zeros, two of them sharing a row, so the block form has
# a block to fill and rows to skip.  `to_sparse_bsr`/`to_sparse_bsc` need the
# sparse size to be divisible by the block size.
BLOCK_CROW = [0, 2, 3, 4, 6]
BLOCK_COL = [1, 3, 0, 2, 1, 3]
BLOCK_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def _csr(device=DEVICE, dtype=torch.float32):
    return torch.sparse_csr_tensor(
        CROW, COL, VALUES, (2, 2), dtype=dtype, device=device
    )


def _csc(device=DEVICE, dtype=torch.float32):
    return torch.sparse_csc_tensor(
        CCOL, ROW, VALUES, (2, 2), dtype=dtype, device=device
    )


def _perm_csr(device=DEVICE):
    return torch.sparse_csr_tensor(
        PERM_CROW, PERM_COL, PERM_VALUES, (3, 3), device=device
    )


def _block_csr(device=DEVICE):
    return torch.sparse_csr_tensor(
        BLOCK_CROW, BLOCK_COL, BLOCK_VALUES, (4, 4), device=device
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
        """The accessor and allocation checks from the report must not raise.

        The report's two remaining checks -- CSC->CSR and SpMM -- are the ones
        that needed the CUDA-compatible libtorch and are covered below.
        """
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
        """zeros must produce an all-zero dense view, not merely a tensor."""
        zero = torch.zeros((2, 2), layout=torch.sparse_csr, device=DEVICE)
        assert zero.layout == torch.sparse_csr
        assert zero._nnz() == 0
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
    @requires_cuda_boxing
    def test_csc_to_sparse_csr_matches_cpu(self):
        """The report's fourth check. It hashes the rebuilt indices through
        ATen's flatten_indices stub, which has no PrivateUse1 kernel."""
        gpu, cpu = _csc().to_sparse_csr(), _csc("cpu").to_sparse_csr()
        assert gpu.layout == torch.sparse_csr
        assert gpu.device == _csc().device
        torch.testing.assert_close(gpu.crow_indices().cpu(), cpu.crow_indices())
        torch.testing.assert_close(gpu.col_indices().cpu(), cpu.col_indices())
        torch.testing.assert_close(gpu.values().cpu(), cpu.values())

    @pytest.mark.anyplatform
    @requires_cuda_boxing
    def test_csr_to_sparse_csc_matches_cpu(self):
        gpu, cpu = _csr().to_sparse_csc(), _csr("cpu").to_sparse_csc()
        assert gpu.layout == torch.sparse_csc
        torch.testing.assert_close(gpu.ccol_indices().cpu(), cpu.ccol_indices())
        torch.testing.assert_close(gpu.row_indices().cpu(), cpu.row_indices())
        torch.testing.assert_close(gpu.values().cpu(), cpu.values())

    @pytest.mark.anyplatform
    @requires_cuda_boxing
    def test_roundtrip_returns_the_original_structure(self):
        """A flip that lost the sort order would pass a dense comparison and
        still break every consumer that reads the index tensors."""
        csr = _csr()
        back = csr.to_sparse_csc().to_sparse_csr()
        torch.testing.assert_close(back.crow_indices().cpu(), csr.crow_indices().cpu())
        torch.testing.assert_close(back.col_indices().cpu(), csr.col_indices().cpu())
        torch.testing.assert_close(back.values().cpu(), csr.values().cpu())

    @pytest.mark.anyplatform
    @requires_cuda_boxing
    def test_csr_to_sparse_csc_off_diagonal(self):
        """The identity above is already sorted in both orientations, so the
        conversions would still pass a hash that dropped the order.  This
        pattern, which shares a row between two entries, does not."""
        gpu, cpu = _block_csr(), _block_csr("cpu")
        flipped = gpu.to_sparse_csc()
        expected = cpu.to_sparse_csc()
        assert flipped.device == gpu.device
        torch.testing.assert_close(
            flipped.ccol_indices().cpu(), expected.ccol_indices()
        )
        torch.testing.assert_close(flipped.row_indices().cpu(), expected.row_indices())
        torch.testing.assert_close(_dense(flipped).cpu(), _dense(cpu))

    @pytest.mark.anyplatform
    @requires_cuda_boxing
    def test_csr_to_sparse_bsc_matches_cpu(self):
        """CSR->BSC is the block conversion this tree can run: ATen sends it
        through `to_sparse_csc` first, i.e. through the flipped-layout helper
        and the stub.  CSR->BSR reaches a different kernel and is not asserted
        here -- it faults inside the torch wheel with no plugin loaded at all,
        on the host as well as on flagos; see TestBlockLayouts."""
        gpu, cpu = _block_csr(), _block_csr("cpu")
        converted = gpu.to_sparse_bsc((2, 2))
        assert converted.layout == torch.sparse_bsc
        assert converted.device == gpu.device
        torch.testing.assert_close(
            _dense(converted).cpu(), _dense(cpu.to_sparse_bsc((2, 2)))
        )

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


class TestMatmul:
    @pytest.mark.anyplatform
    @requires_cuda_boxing
    def test_spmm_matches_cpu(self):
        """The report's fifth check: `torch.sparse.mm` bottoms out in
        `addmm.out`, which computes the product with cuSPARSE."""
        ones = torch.ones(2, 1)
        gpu = torch.sparse.mm(_csr(), ones.to(DEVICE))
        cpu = torch.sparse.mm(_csr("cpu"), ones)
        assert gpu.device == _csr().device
        torch.testing.assert_close(gpu.cpu(), cpu)

    @pytest.mark.anyplatform
    @requires_cuda_boxing
    def test_spmm_off_diagonal(self):
        """A permutation's product is not symmetric, so this one would catch
        a product computed from the transpose."""
        dense = torch.arange(1.0, 10.0).reshape(3, 3)
        gpu = torch.sparse.mm(_perm_csr(), dense.to(DEVICE))
        cpu = torch.sparse.mm(_perm_csr("cpu"), dense)
        assert gpu.device == _perm_csr().device
        torch.testing.assert_close(gpu.cpu(), cpu)

    @pytest.mark.anyplatform
    @requires_cuda_boxing
    def test_addmm_with_dense_self(self):
        """The dense operand is the one boxed by storage; the sparse operand
        is boxed on its own impl.  Both have to read as CUDA for the kernel's
        `_check_is_cuda`."""
        dense = torch.ones(3, 3)
        gpu = torch.addmm(dense.to(DEVICE), _perm_csr(), dense.to(DEVICE))
        cpu = torch.addmm(dense, _perm_csr("cpu"), dense)
        assert gpu.device == _perm_csr().device
        torch.testing.assert_close(gpu.cpu(), cpu)

    @pytest.mark.anyplatform
    @requires_cuda_boxing
    def test_mm_out_writes_through_the_out_tensor(self):
        """The `.out` form must fill the tensor it was handed rather than
        return a fresh one -- that is the registration under test."""
        out = torch.empty(2, 1, device=DEVICE)
        torch.ops.aten.mm.out(_csr(), torch.ones(2, 1, device=DEVICE), out=out)
        assert out.device == _csr().device
        torch.testing.assert_close(out.cpu(), torch.tensor([[1.0], [1.0]]))


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
        result is a structure with no non-zeros, not a dense zero."""
        csr = _csr()
        csr.zero_()
        assert csr._nnz() == 0
        torch.testing.assert_close(_dense(csr).cpu(), torch.zeros(2, 2))
