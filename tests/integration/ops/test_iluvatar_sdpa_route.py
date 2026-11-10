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

"""The Iluvatar ``scaled_dot_product_attention`` vendor route.

``backends_iluvatar.conf`` routes the composite to ``iluvatar``, and
``csrc/aten/sdp_choice_stub.cc`` answers that route by calling CoreX's fused
``libixattnbkd`` kernel (``csrc/aten/backends/iluvatar/fused_attention.h``).
The conf key cannot say which arm a call took -- one PrivateUse1 kernel serves
the composite and reads the conf itself -- so the cases below assert the two
things that can break instead: the answer for the calls inside the measured
envelope, and the answer for the calls just outside it, which must keep the
composite and stay correct.

The envelope is deliberately narrow (bf16, head_dim 64, 128 or 256, 4-D
self-attention with an integer query/key head ratio, any strides, no mask, no
dropout). Every case outside it is a *user-visible* path, which is why they are
asserted here rather than left to the conf test: a gate that widens by accident
would change what a model computes, not just how fast.
"""

import pytest
import torch
import torch_fl

DEVICE = "flagos:0"

pytestmark = [pytest.mark.anyplatform]

#: ``(B, H, S, D)`` -- a decode-sized call, a non-power-of-two length (the
#: vendor gemm has tile-shape defects elsewhere on this chip, so an odd length
#: is worth pinning), the Qwen prefill length, and the two wider head dims.
SHAPES = [
    (1, 2, 128, 64),
    (1, 3, 257, 64),
    (1, 4, 1000, 64),
    (1, 2, 512, 128),
    (1, 2, 267, 256),
]


def _conf_route(op: str):
    """The raw conf value for ``op``, parsed the way the C++ reader does."""
    route = None
    with open(torch_fl.backend_config_path(), encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#", 1)[0]
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == op:
                route = value.strip()
    return route


def _route():
    route = _conf_route("scaled_dot_product_attention")
    if route != "iluvatar":
        pytest.skip(f"this conf routes the composite to {route!r}, not 'iluvatar'")


def _reference(q, k, v, causal=False, mask=None, gqa=False):
    return torch.nn.functional.scaled_dot_product_attention(
        q.float().cpu(),
        k.float().cpu(),
        v.float().cpu(),
        attn_mask=None if mask is None else mask.cpu(),
        is_causal=causal,
        enable_gqa=gqa,
    )


def _check(tag, q, k, v, causal=False, mask=None, gqa=False):
    actual = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=causal, enable_gqa=gqa
    )
    reference = _reference(q, k, v, causal, mask, gqa)
    torch.testing.assert_close(
        actual.float().cpu(),
        reference,
        rtol=2e-2,
        atol=2e-2,
        msg=f"{tag} on {DEVICE}",
    )


@pytest.mark.parametrize("B,H,S,D", SHAPES)
def test_fused_route_matches_cpu(B, H, S, D):
    _route()
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16) for _ in range(3)
    )
    _check(f"fused bf16 [{B},{H},{S},{D}]", q, k, v)


def test_fused_route_causal_matches_cpu():
    _route()
    torch.manual_seed(1)
    q, k, v = (
        torch.randn(1, 2, 512, 64, device=DEVICE, dtype=torch.bfloat16)
        for _ in range(3)
    )
    _check("fused causal bf16 [1,2,512,64]", q, k, v, causal=True)


def test_non_contiguous_call_matches_cpu():
    """HF's vision tower hands SDPA a transposed view; the route copies first."""
    _route()
    torch.manual_seed(5)
    q, k, v = (
        torch.randn(1, 336, 16, 64, device=DEVICE, dtype=torch.bfloat16).transpose(1, 2)
        for _ in range(3)
    )
    assert not q.is_contiguous()
    _check("fused non-contiguous bf16 [1,16,336,64]", q, k, v)


def test_grouped_query_call_matches_cpu():
    """The Qwen3.5 text decoder: 8 query heads over 2 key/value heads."""
    _route()
    torch.manual_seed(6)
    q = torch.randn(1, 8, 267, 256, device=DEVICE, dtype=torch.bfloat16)
    k, v = (
        torch.randn(1, 2, 267, 256, device=DEVICE, dtype=torch.bfloat16)
        for _ in range(2)
    )
    _check("fused GQA bf16 [1,8,267,256] causal", q, k, v, causal=True, gqa=True)


def test_masked_call_keeps_a_correct_path():
    """A mask is outside the fused envelope; it must stay correct, not fast."""
    _route()
    torch.manual_seed(2)
    q, k, v = (
        torch.randn(1, 2, 128, 64, device=DEVICE, dtype=torch.bfloat16)
        for _ in range(3)
    )
    # An all-true bool mask admits every key, so the answer is the maskless
    # one; a mask of zeros would make the reference itself undefined (softmax
    # over an empty key set).
    mask = torch.ones(1, 2, 1, 128, device=DEVICE, dtype=torch.bool)
    _check("masked bf16 [1,2,128,64]", q, k, v, mask=mask)


def test_float16_call_keeps_a_correct_path():
    _route()
    torch.manual_seed(3)
    q, k, v = (
        torch.randn(1, 2, 128, 64, device=DEVICE, dtype=torch.float16) for _ in range(3)
    )
    _check("fp16 [1,2,128,64]", q, k, v)


def test_head_dim_outside_the_envelope_keeps_a_correct_path():
    _route()
    torch.manual_seed(4)
    q, k, v = (
        torch.randn(1, 2, 128, 96, device=DEVICE, dtype=torch.bfloat16)
        for _ in range(3)
    )
    _check("bf16 head_dim 96 [1,2,128,96]", q, k, v)
