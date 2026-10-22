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


def _inputs(query_length=5, key_length=5, heads=2, kv_heads=None, head_dim=16):
    torch.manual_seed(7)
    kv_heads = heads if kv_heads is None else kv_heads
    query = torch.randn(1, heads, query_length, head_dim, dtype=torch.bfloat16)
    key = torch.randn(1, kv_heads, key_length, head_dim, dtype=torch.bfloat16)
    value = torch.randn(1, kv_heads, key_length, head_dim, dtype=torch.bfloat16)
    return query, key, value


def _repeat_kv(tensor, heads):
    """Expand [B, N_kv, S, D] to [B, heads, S, D] for the CPU reference."""
    if tensor.size(1) == heads:
        return tensor
    return tensor.repeat_interleave(heads // tensor.size(1), dim=1)


def _assert_sdpa_matches_cpu(query, key, value, mask=None, is_causal=False, scale=None):
    # The Ascend kernel replicates kv heads internally for GQA; the CPU math path
    # does not, so expand them first and compare against the same head layout.
    query_length = query.size(1)
    expected = F.scaled_dot_product_attention(
        query,
        _repeat_kv(key, query_length),
        _repeat_kv(value, query_length),
        attn_mask=mask,
        is_causal=is_causal,
        scale=scale,
    )
    actual = F.scaled_dot_product_attention(
        query.to(DEVICE),
        key.to(DEVICE),
        value.to(DEVICE),
        attn_mask=None if mask is None else mask.to(DEVICE),
        is_causal=is_causal,
        scale=scale,
    )
    assert actual.device.type == "flagos"
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)


def _allow_mask(length=5):
    return torch.tril(torch.ones(length, length, dtype=torch.bool))


def _additive_mask(length=5):
    """The 0/-inf additive spelling of the boolean allow-mask above."""
    return torch.zeros(length, length).masked_fill(~_allow_mask(length), float("-inf"))


def _finite_bias(*shape):
    torch.manual_seed(11)
    return torch.randn(*shape, dtype=torch.float32) * 0.5


@pytest.mark.ascend
def test_sdpa_no_mask():
    query, key, value = _inputs()
    _assert_sdpa_matches_cpu(query, key, value)


@pytest.mark.ascend
def test_sdpa_bool_allow_mask():
    query, key, value = _inputs()
    mask = torch.tensor(
        [[[[True, True, False, True, False]]]], dtype=torch.bool
    ).expand(1, 1, 5, 5)
    _assert_sdpa_matches_cpu(query, key, value, mask=mask)


@pytest.mark.ascend
def test_sdpa_additive_zero_neg_inf_mask():
    query, key, value = _inputs()
    _assert_sdpa_matches_cpu(query, key, value, mask=_additive_mask())


@pytest.mark.ascend
@pytest.mark.parametrize(
    "bias_shape",
    [(5, 5), (2, 5, 5), (1, 1, 5, 5), (1, 2, 5, 5), (1, 2, 1, 5)],
    ids=["2d", "3d", "bcast_b", "bcast_none", "bcast_s"],
)
def test_sdpa_finite_additive_bias_broadcast(bias_shape):
    query, key, value = _inputs()
    _assert_sdpa_matches_cpu(query, key, value, mask=_finite_bias(*bias_shape))


@pytest.mark.ascend
@pytest.mark.parametrize("scale", [None, 1.0, 0.5, 2.0], ids=str)
def test_sdpa_finite_additive_bias_scales(scale):
    # ACLNN applies its realShift input to the *unscaled* scores, so the bias is
    # pre-divided by the scale. This pins that against the post-scale addition
    # PyTorch documents, at scales either side of the default 1/sqrt(D).
    query, key, value = _inputs()
    _assert_sdpa_matches_cpu(query, key, value, mask=_finite_bias(5, 5), scale=scale)


@pytest.mark.ascend
@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
def test_sdpa_finite_additive_bias_non_square(is_causal):
    # A broadcast bias against a non-square causal mask is where a shape bug in
    # the two-input split would show up: the bias carries [S_q, S_kv] while the
    # causal mask carries [1, 1, S_q, S_kv].
    query, key, value = _inputs(query_length=3, key_length=5)
    _assert_sdpa_matches_cpu(
        query, key, value, mask=_finite_bias(3, 5), is_causal=is_causal
    )


@pytest.mark.ascend
def test_sdpa_finite_additive_bias_gqa():
    query, key, value = _inputs(heads=4, kv_heads=2)
    _assert_sdpa_matches_cpu(query, key, value, mask=_finite_bias(1, 1, 5, 5))


@pytest.mark.ascend
def test_sdpa_finite_additive_bias_with_neg_inf():
    # A mixed tensor: finite shifts on the diagonal-and-below, exclusion above.
    # The -inf entries must reach the attenMask input rather than the realShift
    # one, where an infinity can turn the online softmax into NaN.
    query, key, value = _inputs()
    bias = _finite_bias(5, 5).masked_fill(~_allow_mask(), float("-inf"))
    _assert_sdpa_matches_cpu(query, key, value, mask=bias)


@pytest.mark.ascend
def test_sdpa_causal_non_square():
    query, key, value = _inputs(query_length=3, key_length=5)
    _assert_sdpa_matches_cpu(query, key, value, is_causal=True)


@pytest.mark.ascend
def test_sdpa_causal_with_explicit_mask():
    query, key, value = _inputs()
    _assert_sdpa_matches_cpu(query, key, value, mask=_allow_mask(), is_causal=True)


@pytest.mark.ascend
def test_sdpa_causal_with_finite_additive_bias():
    query, key, value = _inputs()
    _assert_sdpa_matches_cpu(query, key, value, mask=_finite_bias(5, 5), is_causal=True)


@pytest.mark.ascend
def test_sdpa_runs_dropout():
    # dropout_p > 0 must not raise: fused_sdp_choice_ascend declines the fused
    # aclnn backend (it has no dropout plumbing) and returns SDPBackend::math, so
    # the composite takes its math decomposition and applies dropout. The value
    # is not compared to a reference -- the point is that the training forward
    # completes instead of hard-failing, which is what broke the BERT suite.
    query, key, value = _inputs()
    out = F.scaled_dot_product_attention(
        query.to(DEVICE), key.to(DEVICE), value.to(DEVICE), dropout_p=0.5
    )
    assert out.device.type == "flagos"
    assert out.shape == query.shape
