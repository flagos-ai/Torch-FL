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
