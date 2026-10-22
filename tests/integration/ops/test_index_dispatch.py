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

"""Regression tests for indexing on the flagos device.

Two things are covered. The first is CPU indices through the CUDA boxing
kernels. The second is the base-type side effect that a Python workaround for
advanced indexing used to have: assigning ``torch.Tensor.__getitem__`` also
installs ``sq_item`` on the base type, which makes every Tensor, on every
device, look like a sequence to CPython.
"""

import ctypes

import pytest
import torch

import torch_fl  # noqa: F401

DEVICE = "flagos:0"


class TestIndexTensor:
    @pytest.mark.ascend
    @pytest.mark.parametrize(
        "mask",
        (
            torch.tensor([True, False, True, False]),
            torch.tensor([False, False, False, False]),
        ),
    )
    def test_bool_mask(self, mask):
        q_cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        q = q_cpu.to(DEVICE)

        actual = q[mask.to(DEVICE)].cpu()
        expected = q_cpu[mask]

        torch.testing.assert_close(actual, expected)

    @pytest.mark.ascend
    def test_multidimensional_bool_mask(self):
        q_cpu = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
        mask_cpu = torch.tensor([[True, False, True, False, True]])
        q = q_cpu.to(DEVICE)

        actual = q[mask_cpu.to(DEVICE)].cpu()
        expected = q_cpu[mask_cpu]

        torch.testing.assert_close(actual, expected)

    @pytest.mark.ascend
    def test_nonleading_bool_mask(self):
        q_cpu = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        mask_cpu = torch.tensor([True, False, True, False])
        q = q_cpu.to(DEVICE)

        actual = q[:, mask_cpu.to(DEVICE)].cpu()
        expected = q_cpu[:, mask_cpu]

        torch.testing.assert_close(actual, expected)

    @pytest.mark.anyplatform
    def test_cpu_index(self):
        q_cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        q = q_cpu.to(DEVICE)
        index = torch.tensor([0, 2])

        torch.testing.assert_close(q[index].cpu(), q_cpu[index])

    @pytest.mark.anyplatform
    def test_flagos_index(self):
        q_cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        q = q_cpu.to(DEVICE)
        index_cpu = torch.tensor([0, 2])
        index = index_cpu.to(DEVICE)

        torch.testing.assert_close(q[index].cpu(), q_cpu[index_cpu])

    @pytest.mark.anyplatform
    def test_multiple_cpu_indices(self):
        q_cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        q = q_cpu.to(DEVICE)
        index = torch.tensor([0, 2])

        torch.testing.assert_close(q[index, index].cpu(), q_cpu[index, index])

    @pytest.mark.anyplatform
    def test_whisper_mixed_indices(self):
        torch.manual_seed(0)
        logits_cpu = torch.randn(1, 5, 99)
        logits = logits_cpu.to(DEVICE)
        positions = torch.arange(5)
        token_ids_cpu = torch.randint(0, 99, (5,))
        token_ids = token_ids_cpu.to(DEVICE)

        actual = logits[:, positions, token_ids].cpu()
        expected = logits_cpu[:, positions, token_ids_cpu]
        torch.testing.assert_close(actual, expected)

    @pytest.mark.anyplatform
    @pytest.mark.ascend
    def test_nonleading_tensor_index(self):
        # The spelling a Python-level `__getitem__` workaround used to
        # intercept: a tuple whose only Tensor index is not in the leading
        # dimension. It has to reach the same kernel through the dispatcher.
        # Marked `ascend` as well: the C++ dispatcher pads the skipped leading
        # dimension with an engaged-but-undefined tensor, which the Ascend
        # kernel has to ignore exactly like a `None` placeholder.
        q_cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        q = q_cpu.to(DEVICE)
        index_cpu = torch.tensor([1, 3, 5])
        index = index_cpu.to(DEVICE)

        torch.testing.assert_close(q[:, index].cpu(), q_cpu[:, index_cpu])

    @pytest.mark.anyplatform
    def test_unsafe_index_with_cpu_index(self):
        q_cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        q = q_cpu.to(DEVICE)
        index = torch.tensor([0, 2])

        actual = torch.ops.aten._unsafe_index.Tensor(q, [None, index])
        expected = torch.ops.aten._unsafe_index.Tensor(q_cpu, [None, index])
        torch.testing.assert_close(actual.cpu(), expected)

    @pytest.mark.anyplatform
    def test_invalid_index_restores_devices(self):
        q_cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        q = q_cpu.to(DEVICE)
        bad_index = torch.ones(2, device=DEVICE)

        with pytest.raises((RuntimeError, IndexError)):
            torch.ops.aten.index.Tensor(q, [bad_index])

        assert q.device.type == "flagos"
        assert bad_index.device.type == "flagos"
        torch.testing.assert_close((q + 1).cpu(), q_cpu + 1)
        torch.testing.assert_close(
            (bad_index + 1).cpu(), torch.ones(2, dtype=bad_index.dtype) + 1
        )


def _x(device):
    return torch.arange(24, dtype=torch.float32, device=device).reshape(4, 6)


def _j(values, device, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype, device=device)


def _mask(*shape, device):
    return torch.ones(shape, dtype=torch.bool, device=device)


# Every way an out-of-range index *value* reaches the index family, as
# ``(id, run)`` where ``run(device)`` performs the call. ``run`` is evaluated on
# the CPU as well, so the same expression supplies the expected message.
_OOB_CASES = (
    ("index-tensor-index-equals-size", lambda d: _x(d)[_j([0, 4], d)]),
    ("index-tensor-index-past-size", lambda d: _x(d)[_j([1, 9], d)]),
    ("index-tensor-negative-below-size", lambda d: _x(d)[_j([0, -5], d)]),
    ("index-tensor-nonleading-dim", lambda d: _x(d)[:, _j([1, 9], d)]),
    ("index-tensor-cpu-index", lambda d: _x(d)[_j([0, 4], "cpu")]),
    ("index-tensor-int32", lambda d: _x(d)[_j([0, 4], d, torch.int32)]),
    (
        "index-tensor-dispatcher",
        lambda d: torch.ops.aten.index.Tensor(_x(d), [_j([0, 4], d)]),
    ),
    ("index-tensor-zero-size-dim", lambda d: torch.zeros(0, 6, device=d)[_j([0], d)]),
    (
        "index-tensor-mask-then-index",
        lambda d: torch.zeros(2, 3, 4, device=d)[_mask(2, 3, device=d), _j([9], d)],
    ),
    ("index-select", lambda d: _x(d).index_select(0, _j([0, 4], d))),
    ("index-select-nonleading-dim", lambda d: _x(d).index_select(1, _j([6], d))),
    (
        "index-fill-nonleading-dim",
        lambda d: _x(d).clone().index_fill_(1, _j([6], d), 0.0),
    ),
    (
        "index-add-negative",
        lambda d: (
            _x(d).clone().index_add_(0, _j([0, -5], d), torch.ones(2, 6, device=d))
        ),
    ),
    (
        "index-copy",
        lambda d: (
            _x(d).clone().index_copy_(0, _j([0, 4], d), torch.ones(2, 6, device=d))
        ),
    ),
    (
        "index-reduce",
        lambda d: (
            _x(d)
            .clone()
            .index_reduce_(0, _j([0, 4], d), torch.ones(2, 6, device=d), "prod")
        ),
    ),
)

# The subset whose CPU kernel words the failure the way ATen's shared index
# check does (``index <value> is out of bounds for dimension <dim> with size
# <size>``, ``Indexer::get``). The rest of the family raises from its own CPU
# kernel with its own text -- ``index_select`` and ``index_reduce`` say only
# ``index out of range in self`` (and index_select says
# ``INDICES element is out of DATA bounds`` when the index is past a
# non-leading dimension), ``index_copy_`` prefixes the op name, and
# ``index_add_`` reports it through ``scatter_add_`` -- so only these can be
# compared message-for-message. The exception *type* is a separate axis of the
# same question; see _OOB_TYPE_DIVERGENT below.
#
# index.Tensor is in the set even where the number is surprising. Its kernel
# counts only the advanced entries, so ``x[:, j]`` reports dimension 0 (with
# dimension 1's size) and ``x[mask2d, j]`` reports dimension 2 -- see
# CheckIndexListInRange, which reproduces the counting.
_OOB_MESSAGE_PARITY = frozenset(
    {
        "index-tensor-index-equals-size",
        "index-tensor-index-past-size",
        "index-tensor-negative-below-size",
        "index-tensor-nonleading-dim",
        "index-tensor-cpu-index",
        "index-tensor-int32",
        "index-tensor-dispatcher",
        "index-tensor-zero-size-dim",
        "index-tensor-mask-then-index",
        "index-fill-nonleading-dim",
        # index_add_ reaches the shared wording through scatter_add_'s own
        # range check, so its message matches even though its type does not.
        "index-add-negative",
    }
)

# The two cases whose CPU kernel reports the out-of-range value as RuntimeError
# while the check reports IndexError. Both are fast paths in the CPU kernel
# rather than the op's contract:
#
#   index-select-nonleading-dim  ``index_select_out_cpu_`` hands dimension 1 of
#     a contiguous result to ``index_select_out_cpu_dim1_``, whose
#     ``check_indexarray_range`` uses plain ``TORCH_CHECK``.
#   index-add-negative           ``index_add_cpu_out`` forwards to
#     ``scatter_add_`` when the dimension is 0 or the last one, the index is
#     int64 and ``alpha == 1.0``; otherwise it checks indices itself with
#     ``TORCH_CHECK_INDEX``. ``index_add_(0, j, src, alpha=2)`` and an int32
#     ``j`` therefore raise IndexError for the same out-of-range value.
#
# Reproducing the first would mean a dimension test the check does not otherwise
# need, and the second is not decidable from the op alone. The check keeps the
# IndexError spelling that ATen's shared index check and the rest of the family
# use, and these two are the documented cost of that.
_OOB_TYPE_DIVERGENT = frozenset(
    {
        "index-select-nonleading-dim",
        "index-add-negative",
    }
)


class TestIndexValueBounds:
    """An out-of-range index *value* has to raise, not address memory.

    ATen defines an index outside ``[-size, size)`` as an error, and the CPU
    kernels enforce it (``Indexer::get``,
    ``ATen/native/cpu/IndexKernelUtils.h``). The HIP index kernels in this DCU
    torch build do not -- ``libtorch_hip.so`` carries none of the
    "out of bounds for dimension" message family -- so the value reaches the
    offset arithmetic and the kernel dereferences outside the tensor: unrelated
    memory (silently wrong data, the worse outcome), or a DCU ``KERNEL VMFault``
    that aborts the interpreter. Shape errors were always caught; the gap is the
    value range.

    The check therefore runs in the generated ``PrivateUse1`` wrappers
    (``csrc/aten/index_bounds.h``), which is the one layer every route passes
    through -- CUDA boxing, FlagGems, vendor-native -- and it runs before
    ``DeviceBoxingGuard``, so the index is still an ordinary flagos tensor. It is
    compiled in on DCU only (``FLAGOS_INDEX_BOUNDS_CHECK`` in
    ``csrc/CMakeLists.txt``), which is why these tests are marked ``dcu``: where
    the macro is absent the wrappers do not check, and the failure mode is a
    dead interpreter rather than a failed assertion.

    The cases below all abort or return data on an unchecked build. Each runs in
    the same process, so a regression is reported as the abort it is.
    """

    @pytest.mark.dcu
    @pytest.mark.parametrize("run", [pytest.param(c[1], id=c[0]) for c in _OOB_CASES])
    def test_out_of_bounds_raises(self, run):
        with pytest.raises((IndexError, RuntimeError)) as caught:
            run(DEVICE)

        assert "out of bounds" in str(caught.value) or "out of range" in str(
            caught.value
        )

    @pytest.mark.dcu
    @pytest.mark.parametrize(
        "run",
        [
            pytest.param(c[1], id=c[0])
            for c in _OOB_CASES
            if c[0] in _OOB_MESSAGE_PARITY
        ],
    )
    def test_out_of_bounds_message_matches_cpu(self, run):
        with pytest.raises((IndexError, RuntimeError)) as on_cpu:
            run("cpu")
        with pytest.raises((IndexError, RuntimeError)) as on_device:
            run(DEVICE)

        assert str(on_device.value) == str(on_cpu.value)

    @pytest.mark.dcu
    @pytest.mark.parametrize(
        "run",
        [
            pytest.param(c[1], id=c[0])
            for c in _OOB_CASES
            if c[0] not in _OOB_TYPE_DIVERGENT
        ],
    )
    def test_out_of_bounds_exception_type_matches_cpu(self, run):
        # The exception *type* is API surface, unlike the message: IndexError and
        # RuntimeError are siblings in Python, so a caller that wraps the call in
        # one of the two would stop catching an out-of-range index if the check
        # raised the other. ATen's shared index check (``Indexer::get``,
        # ``TORCH_CHECK_INDEX``) raises IndexError, and so does the CPU kernel of
        # every op here but two fast paths -- see _OOB_TYPE_DIVERGENT, which is
        # excluded for the reason recorded there.
        with pytest.raises((IndexError, RuntimeError)) as on_cpu:
            run("cpu")
        with pytest.raises((IndexError, RuntimeError)) as on_device:
            run(DEVICE)

        assert type(on_device.value) is type(on_cpu.value)

    @pytest.mark.dcu
    @pytest.mark.parametrize(
        "run",
        [
            pytest.param(c[1], id=c[0])
            for c in _OOB_CASES
            if c[0] in _OOB_TYPE_DIVERGENT
        ],
    )
    def test_divergent_exception_type_is_index_error(self, run):
        # The two cases in _OOB_TYPE_DIVERGENT report RuntimeError on the CPU and
        # IndexError here. This pins the choice so that changing it is deliberate,
        # and so that the divergence cannot widen unnoticed: every other case is
        # held to the CPU's type by the test above.
        with pytest.raises((IndexError, RuntimeError)) as on_cpu:
            run("cpu")
        with pytest.raises((IndexError, RuntimeError)) as on_device:
            run(DEVICE)

        assert type(on_cpu.value) is RuntimeError
        assert type(on_device.value) is IndexError

    @pytest.mark.dcu
    def test_device_survives_out_of_bounds(self):
        # The point of the fix is that the call raises while the process and the
        # device stay usable. If the check regresses the interpreter dies in the
        # kernel instead, which pytest reports as a crash rather than as this
        # assertion -- either way the failure is loud, and this catches the
        # quieter case of an error that leaves the device in a bad state.
        x = _x(DEVICE)

        with pytest.raises((IndexError, RuntimeError)):
            x[_j([0, 4], DEVICE)]

        torch.testing.assert_close((x + 1).cpu(), _x("cpu") + 1)

    @pytest.mark.dcu
    def test_in_range_is_unaffected(self):
        # The check has to be invisible to every valid call: negatives resolved
        # from the end, an empty index, a CPU-resident index, repeats, and a
        # boolean mask. These are the shapes a model actually issues.
        x_cpu = _x("cpu")
        x = _x(DEVICE)

        # index.Tensor is the spelling that resolves a negative value from the
        # end, so it is the one that can be compared with a negative in range.
        for values in ([0, 2], [-1, -4], [1, 1], []):
            j_cpu = _j(values, "cpu")
            torch.testing.assert_close(
                x[_j(values, DEVICE)].cpu(), x_cpu[j_cpu], msg=str(values)
            )
            torch.testing.assert_close(x[j_cpu].cpu(), x_cpu[j_cpu], msg=str(values))

        # index_select's and index_fill_'s CPU kernels reject a negative index
        # outright, so only the non-negative forms are comparable. The check
        # must not turn any of them into an error.
        for values in ([0, 2], [1, 1], []):
            j_cpu = _j(values, "cpu")
            j = _j(values, DEVICE)
            torch.testing.assert_close(
                x.index_select(0, j).cpu(),
                x_cpu.index_select(0, j_cpu),
                msg=str(values),
            )
            torch.testing.assert_close(
                x.clone().index_fill_(1, j, 0.0).cpu(),
                x_cpu.clone().index_fill_(1, j_cpu, 0.0),
                msg=str(values),
            )

        mask_cpu = torch.tensor([True, False, True, False])
        torch.testing.assert_close(x[mask_cpu.to(DEVICE)].cpu(), x_cpu[mask_cpu])


class TestBaseTensorSequenceProtocol:
    """The device must not change how the base Tensor type behaves.

    Initializing flagos used to assign ``torch.Tensor.__getitem__``. CPython
    special-method assignment fills in the matching type slot, so
    ``PySequence_Check()`` started returning 1 for every Tensor -- including
    CPU ones, and including a Tensor the function's own device guard would
    have declined to touch. ``torch.tensor([zero_dim, ...])`` then takes the
    sequence branch in ``tensor_new.cpp::compute_sizes`` instead of the scalar
    branch and fails in ``Tensor.__len__``.

    Restoring the attribute after the fact does not undo the slot, so these
    tests have to observe the effect of initialization itself.
    """

    @staticmethod
    def _init_device():
        # Every case here is about what the first device op leaves behind, so
        # it has to run before anything is asserted.
        torch.zeros(1, device=DEVICE)

    def test_base_tensor_is_not_a_sequence(self):
        self._init_device()
        sequence_check = ctypes.pythonapi.PySequence_Check
        sequence_check.argtypes = [ctypes.py_object]
        sequence_check.restype = ctypes.c_int

        assert sequence_check(torch.tensor(1.0)) == 0
        assert sequence_check(torch.arange(4.0, device=DEVICE)) == 0

    def test_cpu_scalar_list_after_device_init(self):
        self._init_device()
        values = [torch.tensor(1.0), torch.tensor(2.0)]

        torch.testing.assert_close(torch.tensor(values), torch.tensor([1.0, 2.0]))

    def test_device_scalar_list_after_device_init(self):
        self._init_device()
        values = [
            torch.tensor(1.0, device=DEVICE),
            torch.tensor(2.0, device=DEVICE),
        ]

        torch.testing.assert_close(
            torch.tensor(values, device=DEVICE).cpu(), torch.tensor([1.0, 2.0])
        )
