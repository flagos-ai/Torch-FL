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

"""Regression tests for CPU indices through the CUDA boxing kernels."""

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
