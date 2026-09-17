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

import pytest
import torch
import torch.nn.functional as F

import torch_fl  # noqa: F401

DEVICE = "flagos:0"


def _inputs(query_length=5, key_length=5):
    torch.manual_seed(7)
    query = torch.randn(1, 2, query_length, 16, dtype=torch.bfloat16)
    key = torch.randn(1, 2, key_length, 16, dtype=torch.bfloat16)
    value = torch.randn(1, 2, key_length, 16, dtype=torch.bfloat16)
    return query, key, value


def _assert_sdpa_matches_cpu(query, key, value, mask=None, is_causal=False):
    expected = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, is_causal=is_causal
    )
    actual = F.scaled_dot_product_attention(
        query.to(DEVICE),
        key.to(DEVICE),
        value.to(DEVICE),
        attn_mask=None if mask is None else mask.to(DEVICE),
        is_causal=is_causal,
    )
    assert actual.device.type == "flagos"
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.ascend
def test_sdpa_bool_allow_mask():
    query, key, value = _inputs()
    mask = torch.tensor(
        [[[[True, True, False, True, False]]]], dtype=torch.bool
    ).expand(1, 1, 5, 5)
    _assert_sdpa_matches_cpu(query, key, value, mask=mask)


@pytest.mark.ascend
def test_sdpa_rejects_finite_additive_bias():
    query, key, value = _inputs()
    bias = torch.linspace(0.0, -1.0, 5, dtype=torch.float32).reshape(1, 1, 1, 5)

    with pytest.raises(RuntimeError, match="finite additive attention bias"):
        F.scaled_dot_product_attention(
            query.to(DEVICE),
            key.to(DEVICE),
            value.to(DEVICE),
            attn_mask=bias.to(DEVICE),
        )


@pytest.mark.ascend
def test_sdpa_causal_non_square():
    query, key, value = _inputs(query_length=3, key_length=5)
    _assert_sdpa_matches_cpu(query, key, value, is_causal=True)
