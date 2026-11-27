# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""The stable sort result must have the layout advertised by ATen."""

import pytest
import torch

import torch_fl  # noqa: F401 -- registers the accelerator


@pytest.mark.anyplatform
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.bfloat16])
def test_stable_sort_preserves_transposed_dense_layout(dtype):
    source = torch.tensor([[[[1, 0], [1, 1]]]], dtype=dtype)
    reference = source.transpose(-2, -1)
    tensor = source.to("flagos").transpose(-2, -1)
    assert not tensor.is_contiguous()
    expected = torch.sort(reference, dim=-1, descending=True, stable=True)
    actual = torch.sort(tensor, dim=-1, descending=True, stable=True)
    for result, baseline in zip(actual, expected):
        assert result.device == tensor.device
        assert result.stride() == baseline.stride()
        torch.testing.assert_close(result.cpu(), baseline, rtol=0, atol=0)
