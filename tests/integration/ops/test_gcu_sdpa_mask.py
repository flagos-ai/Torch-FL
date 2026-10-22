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

"""The GCU fused SDPA lane, with a mask.

`_scaled_dot_product_efficient_attention` is the leaf
`csrc/aten/backends/gcu/scaled_dot_product_attention.cc` registers, and unlike
the vendor flash op it takes a mask. The predicate that decides whether a given
mask may go to it lives in that file; these cases are the spellings Qwen-Image-2.1
asks for, plus the ones the predicate declines.

The route is asserted, not inferred from the result. Every decline here is a
*correct* answer on the math path, so a test that only compared outputs would
keep passing on a build where the fused lane had silently gone away -- and at the
target-image shape that lane is 13.81 ms per call against math's 147.14 ms.

The masks are built on the device rather than copied there: `.to()` does not
preserve an expanded or transposed view, and two of the cases below are about
exactly those strides.

A second group covers what the lane does with a mask that is the additive
identity. An all-allow caller mask reaches the leaf as an expanded bf16 buffer of
zeros, and the vendor op charges 2.5x for a mask it reads and multiplies by zero,
so the kernel narrows the broadcast axes away and asks whether anything is left
before handing the mask on. The drop is soundness-critical and is pinned as such:
an all-zero additive mask must return *bitwise* what no mask returns, and a mask
with a single non-zero entry must still block its key. What makes the spelling
safe is that it cannot be dropped by mistake -- zero is only the identity for an
additive mask, and the predicate reads bf16 additive masks alone.

One case records a deviation rather than asserting a desirable property: the
vendor op applies a *finite* additive mask at reduced precision, so a mask that
adds the same constant to every key -- a no-op in exact arithmetic -- moves the
answer, by more than 1e-1 once the constant reaches -1000. `-inf`, which is what
the composite builds and the only large value the model ever produces, is exact.
The test fails if that changes, because the risk recorded in
docs/vendors/gcu/scaled-dot-product-attention.md would then be stale.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

import torch_fl  # noqa: F401  -- registers the flagos backends

DEVICE = torch.device("flagos:0")
HEADS, HEAD_DIM = 2, 128


@pytest.fixture(autouse=True)
def _gcu_only():
    from torch_fl._build_config import ACCELERATOR

    if ACCELERATOR != "gcu":
        pytest.skip("GCU build required")


class _Route(TorchDispatchMode):
    """Which attention implementation the composite ended up calling.

    The math path is a composite, so what the mode sees is its decomposition --
    `bmm` under `_safe_softmax` -- rather than a single math leaf.
    """

    def __init__(self):
        super().__init__()
        self.counts = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] = self.counts.get(str(func), 0) + 1
        return func(*args, **(kwargs or {}))

    @property
    def route(self):
        if any(
            name.endswith("_scaled_dot_product_efficient_attention.default")
            for name in self.counts
        ):
            return "fused"
        if any(
            name.endswith("_safe_softmax.default") or name.endswith("bmm.default")
            for name in self.counts
        ):
            return "math"
        return "other"


def _inputs(query_length, key_length):
    """q/k/v in the layout the model hands over: a dense (B, S, H*D) buffer read
    as (B, H, S, D), which is the strided view the vendor op was measured on."""
    generator = torch.Generator().manual_seed(20260920)

    def rand(rows):
        flat = torch.randn(1, rows, HEADS * HEAD_DIM, generator=generator)
        return flat.bfloat16()

    query = rand(query_length).view(1, query_length, HEADS, HEAD_DIM).transpose(1, 2)
    key = rand(key_length).view(1, key_length, HEADS, HEAD_DIM).transpose(1, 2)
    value = rand(key_length).view(1, key_length, HEADS, HEAD_DIM).transpose(1, 2)
    return query, key, value


def _key_valid(kv, device):
    """The model's own spelling: a rank-4 bool allow-mask on the key axis."""
    mask = torch.ones(1, 1, 1, kv, dtype=torch.bool, device=device)
    mask[..., kv // 2 :] = False
    return mask


def _additive(kv, device):
    """What the composite builds from that bool mask, in the query's dtype."""
    mask = torch.zeros(1, 1, 1, kv, dtype=torch.bfloat16, device=device)
    mask[..., kv // 2 :] = float("-inf")
    return mask


def _all_allow(kv, device):
    """Every key allowed: the caller spelling the target-image call actually uses."""
    return torch.ones(1, 1, 1, kv, dtype=torch.bool, device=device)


def _check(query, key, value, mask, mask_cpu, expect, is_causal=False):
    # Under grad the stub deliberately answers `math`: the backward leaf is a
    # FlagGems kernel and pairing it with a vendor forward is unvalidated. The
    # diffusion loop this lane exists for runs under `torch.no_grad()`.
    with torch.no_grad():
        route = _Route()
        with route:
            actual = F.scaled_dot_product_attention(
                query.to(DEVICE),
                key.to(DEVICE),
                value.to(DEVICE),
                attn_mask=mask,
                is_causal=is_causal,
            )
    expected = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask_cpu, is_causal=is_causal
    )
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2
    )
    assert route.route == expect, (
        f"expected the {expect} route, got {route.route}: "
        f"{dict(sorted(route.counts.items()))}"
    )
    return actual, route


@pytest.mark.gcu
def test_sdpa_no_mask_uses_the_fused_lane():
    query, key, value = _inputs(64, 80)
    _check(query, key, value, None, None, expect="fused")


@pytest.mark.gcu
def test_sdpa_key_valid_mask_non_square():
    """The target-image call: 4096 queries attending 4122 keys, `q_len != kv_len`."""
    query, key, value = _inputs(64, 80)
    _check(
        query,
        key,
        value,
        _key_valid(80, DEVICE),
        _key_valid(80, "cpu"),
        expect="fused",
    )


@pytest.mark.gcu
def test_sdpa_prefix_segment_square_mask():
    """The prefix-segment call: `(1, 1, 26, 26)`, `q_len == kv_len`.

    This spelling used to raise: the stub admitted the caller's bool mask and the
    leaf refused the additive one the composite had built from it.
    """
    query, key, value = _inputs(26, 26)
    _check(
        query,
        key,
        value,
        _key_valid(26, DEVICE),
        _key_valid(26, "cpu"),
        expect="fused",
    )


@pytest.mark.gcu
def test_sdpa_additive_mask_in_the_query_dtype():
    """The converted spelling: `-inf` on the invalid keys, bf16 like the query."""
    query, key, value = _inputs(64, 80)
    _check(
        query,
        key,
        value,
        _additive(80, DEVICE),
        _additive(80, "cpu"),
        expect="fused",
    )


@pytest.mark.gcu
def test_sdpa_row_padded_to_a_multiple_of_eight():
    """A 78-wide key axis, whose rows the conversion pads to 80 and slices back."""
    query, key, value = _inputs(64, 78)
    mask = _key_valid(80, DEVICE)[..., :78]
    mask_cpu = _key_valid(80, "cpu")[..., :78]
    _check(query, key, value, mask, mask_cpu, expect="fused")


@pytest.mark.gcu
def test_sdpa_declines_a_foreign_mask_dtype():
    """A float32 bias is not the query's dtype, so it stays on math.

    The vendor op answers SUCCESS for this mask and reads the row as bf16 bytes,
    so it has to be kept off the fused lane even though it computes fine there.
    """
    query, key, value = _inputs(64, 80)
    mask = torch.zeros(1, 1, 1, 80, dtype=torch.float32, device=DEVICE)
    mask[..., 40:] = float("-inf")
    mask_cpu = torch.zeros(1, 1, 1, 80, dtype=torch.float32)
    mask_cpu[..., 40:] = float("-inf")
    _check(query, key, value, mask, mask_cpu, expect="math")


@pytest.mark.gcu
def test_sdpa_declines_a_strided_innermost_axis():
    """A caller-level transpose, whose innermost stride is the key count.

    The composite asks the stub about the caller's mask, before it converts a bool
    mask, so this reaches the predicate with the un-normalised stride and falls to
    math -- correct, and slower, which is the bound the predicate states.
    """
    query, key, value = _inputs(64, 80)
    mask = torch.zeros(80, 64, dtype=torch.bfloat16, device=DEVICE).t()
    mask_cpu = torch.zeros(80, 64, dtype=torch.bfloat16).t()
    assert mask.stride(-1) == 64 and mask.size(-1) == 80
    _check(query, key, value, mask, mask_cpu, expect="math")


@pytest.mark.gcu
def test_sdpa_under_grad_stays_on_math():
    """Training keeps today's path: the stub answers math while grad is enabled."""
    query, key, value = _inputs(64, 80)
    mask = _key_valid(80, DEVICE)
    qd = query.to(DEVICE).requires_grad_(True)
    kd = key.to(DEVICE).requires_grad_(True)
    vd = value.to(DEVICE).requires_grad_(True)
    route = _Route()
    with torch.set_grad_enabled(True), route:
        actual = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=mask)
    expected = F.scaled_dot_product_attention(
        query, key, value, attn_mask=_key_valid(80, "cpu")
    )
    torch.testing.assert_close(
        actual.detach().cpu().float(), expected.float(), rtol=2e-2, atol=2e-2
    )
    assert route.route == "math", (
        f"expected the math route under grad, got {route.route}: "
        f"{dict(sorted(route.counts.items()))}"
    )


@pytest.mark.gcu
def test_sdpa_all_allow_caller_mask_is_the_same_call_as_no_mask():
    """The target-image spelling: a bool allow-mask with nothing masked out.

    The composite turns it into an expanded bf16 buffer of zeros before the leaf
    sees it. Zero is the additive identity, so the answer has to be the one the
    maskless call gives -- bitwise, not approximately, because the lane is
    entitled to drop the mask on exactly this argument.
    """
    query, key, value = _inputs(64, 80)
    actual, _ = _check(
        query,
        key,
        value,
        _all_allow(80, DEVICE),
        _all_allow(80, "cpu"),
        expect="fused",
    )
    baseline, _ = _check(query, key, value, None, None, expect="fused")
    assert torch.equal(actual.cpu(), baseline.cpu()), (
        "an all-allow mask changed the answer: "
        f"max|d| {(actual.cpu().float() - baseline.cpu().float()).abs().max().item():.3e}"
    )


@pytest.mark.gcu
def test_sdpa_all_zero_additive_mask_is_the_same_call_as_no_mask():
    """The same statement in the converted spelling the leaf is handed."""
    query, key, value = _inputs(64, 80)
    zeros = torch.zeros(1, 1, 1, 80, dtype=torch.bfloat16, device=DEVICE)
    actual, _ = _check(query, key, value, zeros, zeros.cpu(), expect="fused")
    baseline, _ = _check(query, key, value, None, None, expect="fused")
    assert torch.equal(actual.cpu(), baseline.cpu()), (
        "an all-zero additive mask changed the answer: "
        f"max|d| {(actual.cpu().float() - baseline.cpu().float()).abs().max().item():.3e}"
    )


@pytest.mark.gcu
def test_sdpa_one_non_zero_entry_is_not_dropped():
    """A mask that differs from all-zero in one entry must still be read.

    This is the case that bounds the drop: the kernel decides by narrowing the
    broadcast axes away and asking whether a single stored value is non-zero, so
    one blocked key is enough to keep the mask. Blocking a key at `-1000.0`
    removes its contribution to all 64 queries, which is far above bf16
    resolution at this output magnitude.
    """
    query, key, value = _inputs(64, 80)
    mask = torch.zeros(1, 1, 1, 80, dtype=torch.bfloat16, device=DEVICE)
    mask[..., 80 // 3] = -1000.0
    assert int(torch.count_nonzero(mask).item()) == 1
    actual, _ = _check(query, key, value, mask, mask.cpu(), expect="fused")
    baseline, _ = _check(query, key, value, None, None, expect="fused")
    assert not torch.equal(actual.cpu(), baseline.cpu()), (
        "a mask with one blocked key was dropped"
    )


@pytest.mark.gcu
def test_sdpa_a_finite_mask_offset_loses_precision_where_inf_does_not():
    """A recorded deviation, not a desirable behaviour.

    Adding the same constant to every key is a no-op in exact arithmetic --
    softmax is shift-invariant -- so the host answers the maskless question for
    any constant. The vendor op applies the additive mask at reduced precision,
    so a *finite* constant moves the answer, and by an amount that grows with the
    constant's magnitude: on this geometry, max|d| against the host is 1.6e-2 at
    -8, 3.5e-2 at -30, 1.3e-1 at -100 and 1.3 at -1000, while `-inf` is exact to
    the bit. Qwen-Image-2.1 only ever spells the mask with 0 and `-inf`, so the
    lane is exact everywhere the model uses it; this case is recorded rather than
    declined because the predicate would need an arbitrary threshold to decline
    it. If a future vendor op stops deviating here, this test fails on purpose:
    the risk written up in docs/vendors/gcu/scaled-dot-product-attention.md is
    then stale and has to be re-measured.
    """
    query, key, value = _inputs(64, 80)
    device = (query.to(DEVICE), key.to(DEVICE), value.to(DEVICE))

    def run(mask, mask_cpu):
        route = _Route()
        with torch.no_grad(), route:
            out = F.scaled_dot_product_attention(
                *device, attn_mask=None if mask is None else mask
            )
        expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask_cpu)
        assert route.route == "fused", f"expected the fused route, got {route.route}"
        return (out.cpu().float() - expected.float()).abs().max().item()

    finite = torch.full((1, 1, 1, 80), -1000.0, dtype=torch.bfloat16, device=DEVICE)
    # `-inf` on every key is the same degenerate row in the other spelling, and it
    # is the one the composite can build from a bool mask, so it is pinned exact.
    blocked = torch.full(
        (1, 1, 1, 80), float("-inf"), dtype=torch.bfloat16, device=DEVICE
    )
    assert run(blocked, blocked.cpu()) == 0.0
    deviation = run(finite, finite.cpu())
    assert deviation > 1e-1, (
        "a large finite constant in the mask no longer moves the answer, so the "
        "deviation recorded in docs/vendors/gcu/scaled-dot-product-attention.md "
        f"is stale: max|d| {deviation:.3e}"
    )


@pytest.mark.gcu
def test_sdpa_a_per_head_mask_is_read_on_every_head():
    """A mask that is real along the head axis cannot be treated as a broadcast.

    The model's mask has stride 0 on the head and query axes, which is what makes
    narrowing them a view of the same values. A per-head mask has a real stride
    there, so the predicate has to leave it alone and the op has to read all of
    it, not the first head's row. Head 1 blocks the back half of the keys and head
    0 blocks nothing, so a lane that read only head 0 would answer the maskless
    call.
    """
    query, key, value = _inputs(64, 80)
    mask = torch.zeros(1, HEADS, 1, 80, dtype=torch.bfloat16, device=DEVICE)
    mask[:, 1, :, 80 // 2 :] = float("-inf")
    assert mask.stride(1) != 0
    assert mask[0, 0].count_nonzero() == 0
    actual, _ = _check(query, key, value, mask, mask.cpu(), expect="fused")
    baseline, _ = _check(query, key, value, None, None, expect="fused")
    assert not torch.equal(actual.cpu(), baseline.cpu()), "a per-head mask was dropped"
