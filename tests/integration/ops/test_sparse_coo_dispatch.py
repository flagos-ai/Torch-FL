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

"""COO construction and structure surface on the flagos device (issue #294).

``torch.sparse_coo_tensor(..., device="flagos:0")`` used to raise
``NotImplementedError`` for ``aten::_sparse_coo_tensor_with_dims_and_tensors``
on the ``Sparseflagos`` backend, because nothing served
``c10::DispatchKey::SparsePrivateUse1``.  csrc/aten/sparse_ops.cc registers the
constructor and the COO structure surface for that key.

Only the structure surface is asserted here: sizes, the coalesced flag,
indices/values access and buffer movement.  The sparse *value* kernels (sparse
add, ``sparse_mask``, ``_coalesce`` for genuinely uncoalesced input, ...) are
outside that change's scope and are deliberately not asserted, so this file
stays a statement of what is supported rather than of what is not.

Every flagos result is compared against the same construction on CPU.  The
comparison goes through ``.to("cpu")`` rather than ``.to_dense()``: the dense
bridge is upstream's ``zeros(...).add_(self)``, i.e. it belongs to the sparse
add family and is not part of the structure surface.
"""

import pytest
import torch

import torch_fl  # noqa: F401

DEVICE = "flagos:0"

# `bmm` is the one op in this file that needs a vendor value kernel rather than
# an ATen symbol shared across backends, so it is registered only on a
# CUDA-boxing build (see the `#if ... && defined(FLAGOS_BOXING_BUILD)` guard in
# csrc/aten/sparse_ops.cc).  That is expressed as the `cuda` mark rather than a
# local `skipif`: the marker is what ops/conftest.py reads, and its build-record
# gate skips a `cuda`-marked case on any wheel built without the boxing kernels
# -- the same rule tests/integration/ops/test_sparse_csr_dispatch.py encodes by
# hand, and one this file cannot restate locally because a new legacy skip call
# in tests/integration/ops/ is refused by skip_policy_baseline.json.


def _invariants_for(x, layout=torch.sparse_coo):
    """The TensorOptions a COO construction was asked for must survive."""
    assert x.layout == layout
    assert str(x.device) == DEVICE


def _keys(x):
    return str(torch._C._dispatch_keys(x))


class TestConstruction:
    @pytest.mark.anyplatform
    def test_issue_294_reproducer(self):
        """The exact snippet from the report: it must construct, not raise."""
        x = torch.sparse_coo_tensor([[0, 1], [0, 1]], [1.0, 1.0], (2, 2), device=DEVICE)
        _invariants_for(x)
        assert tuple(x.shape) == (2, 2)
        assert x.dtype == torch.float32

    @pytest.mark.anyplatform
    def test_dispatch_key_is_sparse_privateuse1(self):
        """The tensor must really be a SparsePrivateUse1 tensor.

        A tensor that merely reported the right layout while sitting on the
        dense key would pass a shape assertion and fail everywhere else.
        """
        x = torch.sparse_coo_tensor([[0, 1], [0, 1]], [1.0, 1.0], (2, 2), device=DEVICE)
        assert "SparsePrivateUse1" in _keys(x)

    @pytest.mark.anyplatform
    def test_indices_and_values_stay_on_device(self):
        """Construction must not silently stage the buffers through the host."""
        indices = torch.tensor([[0, 1], [0, 1]], device=DEVICE)
        values = torch.tensor([1.0, 2.0], device=DEVICE)
        x = torch.sparse_coo_tensor(indices, values, (2, 2), device=DEVICE)
        assert x._indices().device == indices.device
        assert x._values().device == values.device
        torch.testing.assert_close(x._indices().cpu(), torch.tensor([[0, 1], [0, 1]]))
        torch.testing.assert_close(x._values().cpu(), torch.tensor([1.0, 2.0]))

    @pytest.mark.anyplatform
    def test_size_only_is_empty_and_coalesced(self):
        """A size-only construction has no entries and is coalesced by fiat."""
        x = torch.sparse_coo_tensor((2, 2), dtype=torch.float64, device=DEVICE)
        _invariants_for(x)
        assert x.dtype == torch.float64
        assert x._nnz() == 0
        assert x.is_coalesced()

    @pytest.mark.anyplatform
    def test_explicit_zero_nnz(self):
        indices = torch.empty((2, 0), dtype=torch.int64, device=DEVICE)
        values = torch.empty((0,), dtype=torch.float32, device=DEVICE)
        x = torch.sparse_coo_tensor(indices, values, (2, 2), device=DEVICE)
        _invariants_for(x)
        assert x._nnz() == 0
        assert x.is_coalesced()

    @pytest.mark.anyplatform
    def test_declared_coalesced(self):
        """`is_coalesced=True` is forwarded, not dropped."""
        indices = torch.tensor([[0, 1], [0, 1]], device=DEVICE)
        values = torch.tensor([1.0, 2.0], device=DEVICE)
        x = torch.sparse_coo_tensor(
            indices, values, (2, 2), device=DEVICE, is_coalesced=True
        )
        assert x.is_coalesced()

    @pytest.mark.anyplatform
    def test_duplicate_indices_are_not_coalesced(self):
        """Duplicate coordinates must not be reported as coalesced.

        Deciding that they are would make `indices()`/`values()` return a
        tensor whose entries do not match the sum the same data densifies to.
        """
        indices = torch.tensor([[0, 0, 1], [0, 0, 1]], device=DEVICE)
        values = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        x = torch.sparse_coo_tensor(indices, values, (2, 2), device=DEVICE)
        assert not x.is_coalesced()
        assert x._nnz() == 3
        torch.testing.assert_close(
            x.to("cpu").to_dense(), torch.tensor([[3.0, 0.0], [0.0, 3.0]])
        )

    @pytest.mark.anyplatform
    def test_dense_dim(self):
        """sparse_dim=1, dense_dim=1: indices (1, nnz), values (nnz, 2)."""
        indices = torch.tensor([[0, 1]], device=DEVICE)
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=DEVICE)
        x = torch.sparse_coo_tensor(indices, values, (2, 2), device=DEVICE)
        _invariants_for(x)
        assert x.sparse_dim() == 1
        assert x.dense_dim() == 1
        torch.testing.assert_close(
            x.to("cpu").to_dense(), torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        )

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.int64))
    def test_supported_value_dtypes(self, dtype):
        indices = torch.tensor([[0, 3], [1, 4]], device=DEVICE)
        values = torch.tensor([1, 2], dtype=dtype, device=DEVICE)
        x = torch.sparse_coo_tensor(indices, values, (5, 5), device=DEVICE)
        _invariants_for(x)
        assert x.dtype == dtype
        assert tuple(x.shape) == (5, 5)
        assert x._values().dtype == dtype

    @pytest.mark.anyplatform
    def test_out_variant_with_sparse_out(self):
        """The `.out` form with a sparse `out` lands on the same key."""
        out = torch.sparse_coo_tensor((2, 2), dtype=torch.float32, device=DEVICE)
        x = torch.ops.aten._sparse_coo_tensor_with_dims.out(2, 0, [2, 2], out=out)
        _invariants_for(x)
        assert tuple(x.shape) == (2, 2)


class TestStructureQueries:
    @pytest.mark.anyplatform
    def test_shape_and_dims(self):
        x = torch.sparse_coo_tensor([[0, 1], [0, 1]], [1.0, 2.0], (2, 3), device=DEVICE)
        assert tuple(x.size()) == (2, 3)
        assert x.dim() == 2
        assert x.numel() == 6
        assert x.sparse_dim() == 2
        assert x.dense_dim() == 0
        assert x._dimI() == 2
        assert x._dimV() == 0

    @pytest.mark.anyplatform
    def test_checked_accessors_require_coalesced(self):
        """`indices()`/`values()` must fail the same way they do on CPU."""
        indices = torch.tensor([[0, 1], [0, 1]])
        values = torch.tensor([1.0, 2.0])
        cpu = torch.sparse_coo_tensor(indices, values, (2, 2))
        gpu = torch.sparse_coo_tensor(indices.to(DEVICE), values.to(DEVICE), (2, 2))

        for tensor in (cpu, gpu):
            assert not tensor.is_coalesced()
            with pytest.raises(RuntimeError, match="uncoalesced"):
                tensor.indices()
            with pytest.raises(RuntimeError, match="uncoalesced"):
                tensor.values()

    @pytest.mark.anyplatform
    def test_checked_accessors_match_cpu_when_coalesced(self):
        indices = torch.tensor([[0, 1], [0, 1]])
        values = torch.tensor([1.0, 2.0])
        cpu = torch.sparse_coo_tensor(indices, values, (2, 2), is_coalesced=True)
        gpu = torch.sparse_coo_tensor(
            indices.to(DEVICE),
            values.to(DEVICE),
            (2, 2),
            device=DEVICE,
            is_coalesced=True,
        )
        assert gpu.is_coalesced()
        torch.testing.assert_close(gpu.indices().cpu(), cpu.indices())
        torch.testing.assert_close(gpu.values().cpu(), cpu.values())

    @pytest.mark.anyplatform
    def test_coalesced_flag_setter(self):
        """`_coalesced_` must flip the flag on this backend too."""
        x = torch.sparse_coo_tensor([[0, 1], [0, 1]], [1.0, 2.0], (2, 2), device=DEVICE)
        assert not x.is_coalesced()
        assert x._coalesced_(True).is_coalesced()

    @pytest.mark.anyplatform
    def test_repr_does_not_need_the_dense_bridge(self):
        """Printing a COO tensor reads indices/values, not the dense bridge."""
        x = torch.sparse_coo_tensor([[0, 1], [0, 1]], [1.0, 2.0], (2, 2), device=DEVICE)
        text = repr(x)
        assert "sparse_coo" in text
        assert "flagos" in text


class TestBufferMovement:
    @pytest.mark.anyplatform
    def test_cpu_roundtrip_matches_cpu_reference(self):
        indices = torch.tensor([[0, 1], [0, 1]])
        values = torch.tensor([1.0, 2.0])
        cpu = torch.sparse_coo_tensor(indices, values, (2, 2))
        gpu = torch.sparse_coo_tensor(
            indices.to(DEVICE), values.to(DEVICE), (2, 2), device=DEVICE
        )
        torch.testing.assert_close(gpu.to("cpu").to_dense(), cpu.to_dense())

    @pytest.mark.anyplatform
    def test_transfer_preserves_layout_and_device(self):
        x = torch.sparse_coo_tensor([[0, 1], [0, 1]], [1.0, 2.0], (2, 2), device=DEVICE)
        moved = x.to("cpu")
        assert moved.layout == torch.sparse_coo
        assert moved.device.type == "cpu"
        back = moved.to(DEVICE)
        assert back.layout == torch.sparse_coo
        assert back.device == x.device
        torch.testing.assert_close(back.to("cpu").to_dense(), moved.to_dense())

    @pytest.mark.anyplatform
    def test_clone_matches_cpu_reference(self):
        indices = torch.tensor([[0, 1], [0, 1]])
        values = torch.tensor([1.0, 2.0])
        cpu = torch.sparse_coo_tensor(indices, values, (2, 2))
        gpu = torch.sparse_coo_tensor(
            indices.to(DEVICE), values.to(DEVICE), (2, 2), device=DEVICE
        )
        cloned = gpu.clone()
        assert cloned.layout == torch.sparse_coo
        assert cloned.device == gpu.device
        assert cloned._nnz() == gpu._nnz()
        torch.testing.assert_close(cloned.to("cpu").to_dense(), cpu.to_dense())

    @pytest.mark.anyplatform
    def test_clone_does_not_alias_the_buffers(self):
        """A clone that shared storage would make later writes visible twice."""
        x = torch.sparse_coo_tensor([[0, 1], [0, 1]], [1.0, 2.0], (2, 2), device=DEVICE)
        cloned = x.clone()
        cloned._values().fill_(0.0)
        torch.testing.assert_close(x._values().cpu(), torch.tensor([1.0, 2.0]))


@pytest.mark.anyplatform
@pytest.mark.dcu
@pytest.mark.cuda
class TestSparseBmm:
    """Sparse COO left operand x dense right operand (issue #793).

    The COO constructor and the structure surface above need no vendor kernel,
    so they are registered on ATen's own device-agnostic symbols. ``bmm``
    multiplies values and has no such symbol, so a boxing build serves it by
    boxing the operands onto the vendor's CUDA kernels. These cases state that
    contract: the sparse-left form matches CPU, the caller's operands come back
    unchanged (the boxed call must not leak its retag), and the forms upstream
    itself rejects are still rejected rather than answered differently here.

    Two marks carry the applicability. ``cuda`` is the build-record gate: the
    boxed call needs kernels behind the CUDA dispatch key, and ops/conftest.py
    skips a ``cuda``-marked case on a wheel built without them (the default,
    FlagGems-primary wheel every CUDA-ABI vendor lane builds). ``dcu`` is what
    gets these cases collected at all on the lane that can run them -- the Hygon
    entry in .github/integration-manifest.yml selects
    ``-m "(dcu or main_ops) ..."``, which does not select a case carrying only
    ``anyplatform``. The rest of this file is in that position (every case above
    is ``anyplatform``-only), so the COO structure surface is not exercised on
    DCU hardware by CI either; this class does not repeat that.
    """

    @pytest.mark.anyplatform
    def test_issue_793_reproducer(self):
        """The exact snippet from the report: it must match the CPU result."""
        i = torch.tensor([[0, 0, 0], [0, 1, 1], [0, 1, 2]])
        s = torch.sparse_coo_tensor(i, torch.tensor([1.0, 2.0, 3.0]), (1, 2, 3))
        d = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        s, d = s.coalesce(), d
        expected = torch.bmm(s, d)
        actual = torch.bmm(s.to(DEVICE), d.to(DEVICE))
        assert actual.device.type == "flagos"
        assert tuple(actual.shape) == (1, 2, 2)
        torch.testing.assert_close(actual.cpu(), expected)

    @pytest.mark.anyplatform
    def test_matches_cpu_over_multiple_batches(self):
        """Larger nnz and a batch dim > 1, compared against CPU."""
        torch.manual_seed(0)
        n_batches, rows, cols, inner = 3, 4, 5, 6
        dense = torch.randn(n_batches, inner, cols)
        sparse_dense = torch.randn(n_batches, rows, inner).masked_fill(
            torch.rand(n_batches, rows, inner) < 0.5, 0.0
        )
        cpu_sparse = sparse_dense.to_sparse().coalesce()
        expected = torch.bmm(cpu_sparse, dense)
        actual = torch.bmm(cpu_sparse.to(DEVICE), dense.to(DEVICE))
        torch.testing.assert_close(actual.cpu(), expected)

    @pytest.mark.anyplatform
    def test_operands_are_restored_after_the_call(self):
        """The boxed call must not leave the caller's tensors retagged or moved."""
        s = torch.sparse_coo_tensor(
            [[0, 0, 0], [0, 1, 1], [0, 1, 2]],
            [1.0, 2.0, 3.0],
            (1, 2, 3),
            device=DEVICE,
        ).coalesce()
        d = torch.randn(1, 3, 2, device=DEVICE)
        before_s, before_d = s.device, d.device
        torch.bmm(s, d)
        assert s.device == before_s and s.layout == torch.sparse_coo
        assert d.device == before_d
        # Still usable exactly as it was: a second call gives the same answer.
        again = torch.bmm(s, d)
        torch.testing.assert_close(again.cpu(), torch.bmm(s.cpu(), d.cpu()))

    @pytest.mark.anyplatform
    def test_empty_sparse_operand(self):
        """Zero nnz is a valid COO tensor and must produce a zeros result."""
        empty = torch.sparse_coo_tensor(
            torch.zeros(3, 0, dtype=torch.long),
            torch.zeros(0),
            (1, 2, 3),
            device=DEVICE,
        ).coalesce()
        dense = torch.randn(1, 3, 2, device=DEVICE)
        actual = torch.bmm(empty, dense)
        assert tuple(actual.shape) == (1, 2, 2)
        torch.testing.assert_close(actual.cpu(), torch.zeros(1, 2, 2), atol=0, rtol=0)

    @pytest.mark.anyplatform
    def test_dense_left_sparse_right_is_rejected_like_upstream(self):
        """`bmm(dense, sparse)` is not served: upstream rejects it too.

        Answering it here would invent behaviour the vendor's own CUDA kernels
        do not have, so it has to keep raising rather than quietly swapping
        which operand is sparse.
        """
        s = torch.sparse_coo_tensor(
            [[0, 0, 0], [0, 1, 1], [0, 1, 2]],
            [1.0, 2.0, 3.0],
            (1, 3, 2),
            device=DEVICE,
        ).coalesce()
        d = torch.randn(1, 2, 3, device=DEVICE)
        with pytest.raises(RuntimeError):
            torch.bmm(d, s)

    @pytest.mark.anyplatform
    def test_grad_requiring_operand_is_declined(self):
        """A call that would record a graph must not go through the boxing.

        The boxed CUDA call would build its backward from the CUDA-keyed
        operands the guard restores, so a caller-visible ``grad_fn`` would point
        at tensors nobody holds. Declining keeps the pre-fix boundary for that
        case instead of returning a wrong graph.
        """
        s = torch.sparse_coo_tensor(
            [[0, 0, 0], [0, 1, 1], [0, 1, 2]], [1.0, 2.0, 3.0], (1, 2, 3)
        ).coalesce()
        d = torch.randn(1, 3, 2)
        s, d = s.to(DEVICE), d.to(DEVICE).requires_grad_(True)
        with pytest.raises(RuntimeError):
            torch.bmm(s, d)
