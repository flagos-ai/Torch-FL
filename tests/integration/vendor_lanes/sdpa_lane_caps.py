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

"""The vendor SDPA lanes, on one case set per lane.

``scaled_dot_product_attention`` is the operator this package exists for. Three
different things serve it, and until this module each of them had its own file:
``tests/integration/ops/test_ascend_sdpa_mask_bias.py`` pinned what ACLNN does
with a mask, ``tests/integration/ops/test_gcu_sdpa_mask.py`` pinned which of
GCU's two lanes a call takes and what the fused one does with the additive it is
handed, and ``tests/integration/flaggems/test_flaggems_sdpa_route.py`` pins the
FlagGems arm DCU and MetaX route to. The first two are here now; the third is
where it belongs, beside the conf data it reads.

What is shared, and what is not, is the point of the split. The *contract* is
shared: a call whose mask the lane accepts must agree with the host reference
within the bound that lane was measured to hold, a mask that is the additive
identity must leave the answer where the maskless call left it, a mask that
blocks a key must move it, and a call a lane declines must take the other one.
The *spelling* is not: Ascend's ACLNN kernel takes the mask split into a
``realShift`` bias and an ``attenMask``, and applies the bias to the unscaled
scores, so a caller's bias is pre-divided by the scale; GCU's fused lane takes
the whole additive in the query's own dtype, narrows a broadcast mask before
handing it on, and reads a finite offset at reduced precision. Those are rows.
Adding a chip means adding a row, not a file, and the chip inherits every case
its peers run -- which is the whole claim this package makes.

Nothing at import time touches the device: every tensor the cases hand to the
runtime is built inside ``build``, so importing this module never allocates on
an accelerator.

Nothing here imports ``torch_fl``: the test module imports the package, and the
builders below only need ``torch`` and the device string ``[flagos dispatch]``
prints for the flagos device.
"""

from __future__ import annotations

import sys
from typing import Callable

from device_init import torch
import torch.nn.functional as F
from shared_sdpa_caps import COMMON_TAGS, SHARED_ROUTES

from vendor_caps import DEVICE, Case, LaneRow, register_domain

# --------------------------------------------------------------------------- #
# Ascend: the ACLNN lane, asserted by the answer it produces.
# --------------------------------------------------------------------------- #

#: ACLNN's bias input carries the query length in its second-to-last axis and the
#: key length in its last, so the operands are laid out (1, H, S, D) rather than
#: the (1, S, H*D) view GCU's model call passes.
_ASCEND_HEAD_DIM = 16


def _ascend_inputs(query_length=5, key_length=5, heads=2, kv_heads=None):
    torch.manual_seed(7)
    kv_heads = heads if kv_heads is None else kv_heads
    query = torch.randn(1, heads, query_length, _ASCEND_HEAD_DIM, dtype=torch.bfloat16)
    key = torch.randn(1, kv_heads, key_length, _ASCEND_HEAD_DIM, dtype=torch.bfloat16)
    value = torch.randn(1, kv_heads, key_length, _ASCEND_HEAD_DIM, dtype=torch.bfloat16)
    return query, key, value


def _ascend_repeat_kv(tensor, heads):
    """Expand [B, N_kv, S, D] to [B, heads, S, D] for the host reference."""
    if tensor.size(1) == heads:
        return tensor
    return tensor.repeat_interleave(heads // tensor.size(1), dim=1)


def _ascend_allow_mask(length=5):
    return torch.tril(torch.ones(length, length, dtype=torch.bool))


def _ascend_additive_mask(length=5):
    """The 0/-inf additive spelling of the boolean allow-mask above."""
    return torch.zeros(length, length).masked_fill(
        ~_ascend_allow_mask(length), float("-inf")
    )


def _ascend_finite_bias(*shape):
    torch.manual_seed(11)
    return torch.randn(*shape, dtype=torch.float32) * 0.5


def _ascend_reference(query, key, value, mask, **kwargs):
    """Host answer, against the same head layout the kernel replicates to.

    The ACLNN kernel replicates kv heads internally for GQA; the host math path
    does not, so they are expanded here before the comparison.
    """
    heads = query.size(1)
    return F.scaled_dot_product_attention(
        query,
        _ascend_repeat_kv(key, heads),
        _ascend_repeat_kv(value, heads),
        attn_mask=mask,
        **kwargs,
    )


def _ascend_build(query, key, value, mask=None, **kwargs) -> dict:
    device = (
        query.to(DEVICE),
        key.to(DEVICE),
        value.to(DEVICE),
    )
    mask_device = None if mask is None else mask.to(DEVICE)
    return {
        "run": lambda: F.scaled_dot_product_attention(
            *device, attn_mask=mask_device, **kwargs
        ),
        "reference": _ascend_reference(query, key, value, mask, **kwargs),
    }


def _ascend_compare(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.device.type == "flagos"
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)


def _ascend_case(tag: str, clause: str, **build) -> Case:
    query, key, value = _ascend_inputs(**build.pop("inputs", {}))
    return Case(tag, clause, lambda: _ascend_build(query, key, value, **build))


def _ascend_dropout_case() -> Case:
    """The dropout call: the fused lane declines it, the math path serves it.

    ``dropout_p > 0`` must not raise. ``fused_sdp_choice_ascend`` has no dropout
    plumbing in the ACLNN kernel, so it declines the fused backend and returns
    ``SDPBackend::math``; the composite then takes its math decomposition, which
    applies dropout correctly. The value is deliberately not compared against a
    host reference -- dropout is stochastic and the point of the case is that the
    training forward completes instead of hard-failing, which is what broke the
    BERT suite before the decline existed. ``_ascend_compare`` checks the device
    and the shape and is given the output as its own reference so the tolerance
    comparison is vacuous rather than wrong.
    """
    query, key, value = _ascend_inputs()

    def build() -> dict:
        device = (query.to(DEVICE), key.to(DEVICE), value.to(DEVICE))
        return {
            "run": lambda: F.scaled_dot_product_attention(*device, dropout_p=0.5),
            "reference": None,
        }

    return Case(
        "runs_dropout",
        "a nonzero dropout_p, which the math decomposition serves",
        build,
    )


def _ascend_cases() -> tuple[Case, ...]:
    cases = [
        _ascend_case(
            "ascend_no_mask",
            "the maskless ACLNN call, which is the lane's simplest agreement",
        ),
        _ascend_case(
            "bool_allow_mask",
            "a (1,1,5,5) bool allow-mask broadcast from one row",
            mask=torch.tensor(
                [[[[True, True, False, True, False]]]], dtype=torch.bool
            ).expand(1, 1, 5, 5),
        ),
        _ascend_case(
            "ascend_bool_allow_mask_as_additive",
            "the same allow-mask in its 0/-inf additive spelling",
            mask=_ascend_additive_mask(),
        ),
    ]
    for shape, name in (
        ((5, 5), "2d"),
        ((2, 5, 5), "3d"),
        ((1, 1, 5, 5), "bcast_b"),
        ((1, 2, 5, 5), "bcast_none"),
        ((1, 2, 1, 5), "bcast_s"),
    ):
        cases.append(
            _ascend_case(
                f"finite_additive_bias_{name}",
                f"a finite additive bias at shape {shape}",
                mask=_ascend_finite_bias(*shape),
            )
        )
    for scale in (None, 1.0, 0.5, 2.0):
        # ACLNN applies its realShift input to the *unscaled* scores, so the
        # caller's bias is pre-divided by the scale. This pins that against the
        # post-scale addition PyTorch documents, at scales either side of the
        # default 1/sqrt(D).
        cases.append(
            _ascend_case(
                f"finite_additive_bias_scale_{scale}",
                f"a finite additive bias at scale={scale}",
                mask=_ascend_finite_bias(5, 5),
                scale=scale,
            )
        )
    for is_causal, name in ((False, "full"), (True, "causal")):
        # A broadcast bias against a non-square causal mask is where a shape bug
        # in the two-input split would show up: the bias carries [S_q, S_kv]
        # while the causal mask carries [1, 1, S_q, S_kv].
        cases.append(
            _ascend_case(
                f"finite_additive_bias_non_square_{name}",
                f"a non-square bias against the {name} causal mask",
                inputs={"query_length": 3, "key_length": 5},
                mask=_ascend_finite_bias(3, 5),
                is_causal=is_causal,
            )
        )
    cases.append(
        _ascend_case(
            "finite_additive_bias_gqa",
            "a broadcast bias with four query heads over two kv heads",
            inputs={"heads": 4, "kv_heads": 2},
            mask=_ascend_finite_bias(1, 1, 5, 5),
        )
    )
    cases.append(
        _ascend_case(
            "finite_additive_bias_with_neg_inf",
            "a mixed tensor: finite shifts below the diagonal, -inf above",
            mask=_ascend_finite_bias(5, 5).masked_fill(
                ~_ascend_allow_mask(), float("-inf")
            ),
        )
    )
    cases.append(
        _ascend_case(
            "causal_non_square",
            "is_causal with a non-square query/key pair",
            inputs={"query_length": 3, "key_length": 5},
            is_causal=True,
        )
    )
    cases.append(
        _ascend_case(
            "causal_with_explicit_mask",
            "is_causal against an explicit triangular allow-mask",
            mask=_ascend_allow_mask(),
            is_causal=True,
        )
    )
    cases.append(
        _ascend_case(
            "causal_with_finite_additive_bias",
            "is_causal against a finite additive bias",
            mask=_ascend_finite_bias(5, 5),
            is_causal=True,
        )
    )
    cases.append(_ascend_dropout_case())
    return tuple(cases)


# --------------------------------------------------------------------------- #
# GCU: the fused lane, asserted by the lane it takes and the answer it gives.
# --------------------------------------------------------------------------- #

#: The layout the model hands over: a dense (B, S, H*D) buffer read as
#: (B, H, S, D), which is the strided view the vendor op was measured on.
_GCU_HEADS, _GCU_HEAD_DIM = 2, 128
_GCU_QUERY_LENGTH, _GCU_KEY_LENGTH = 64, 80

#: The masks are built on the device rather than copied there: ``.to()`` does not
#: preserve an expanded or transposed view, and two of the cases below are about
#: exactly those strides. That is also why each mask is a factory the case calls
#: when it runs, rather than a tensor held at import.
MaskPair = Callable[[], tuple]


def _gcu_key_valid(kv, device):
    """The model's own spelling: a rank-4 bool allow-mask on the key axis."""
    mask = torch.ones(1, 1, 1, kv, dtype=torch.bool, device=device)
    mask[..., kv // 2 :] = False
    return mask


def _gcu_additive(kv, device):
    """What the composite builds from that bool mask, in the query's dtype."""
    mask = torch.zeros(1, 1, 1, kv, dtype=torch.bfloat16, device=device)
    mask[..., kv // 2 :] = float("-inf")
    return mask


def _gcu_all_allow(kv, device):
    """Every key allowed: the caller spelling the target-image call uses."""
    return torch.ones(1, 1, 1, kv, dtype=torch.bool, device=device)


def _gcu_foreign_dtype_mask():
    """A float32 bias, which is not the query's dtype and so stays on math."""
    mask = torch.zeros(1, 1, 1, _GCU_KEY_LENGTH, dtype=torch.float32, device=DEVICE)
    mask[..., _GCU_KEY_LENGTH // 2 :] = float("-inf")
    mask_cpu = torch.zeros(1, 1, 1, _GCU_KEY_LENGTH, dtype=torch.float32)
    mask_cpu[..., _GCU_KEY_LENGTH // 2 :] = float("-inf")
    return mask, mask_cpu


def _gcu_strided_innermost_mask():
    """A caller-level transpose, whose innermost stride is the key count."""

    def transposed(device):
        return torch.zeros(
            _GCU_KEY_LENGTH, _GCU_QUERY_LENGTH, dtype=torch.bfloat16, device=device
        ).t()

    mask, mask_cpu = transposed(DEVICE), transposed("cpu")
    assert mask.stride(-1) == _GCU_QUERY_LENGTH and mask.size(-1) == _GCU_KEY_LENGTH
    return mask, mask_cpu


def _gcu_row_padded_mask():
    """A 78-wide key axis, whose rows the conversion pads to 80 and slices back."""
    return _gcu_key_valid(80, DEVICE)[..., :78], _gcu_key_valid(80, "cpu")[..., :78]


def _gcu_all_zero_mask():
    return (
        torch.zeros(1, 1, 1, _GCU_KEY_LENGTH, dtype=torch.bfloat16, device=DEVICE),
        torch.zeros(1, 1, 1, _GCU_KEY_LENGTH, dtype=torch.bfloat16),
    )


def _gcu_one_blocked_key_mask():
    """The bound on the drop: one key blocked at -1000.0 keeps the mask.

    Blocking a key at ``-1000.0`` removes its contribution to all 64 queries,
    far above bf16 resolution at this output magnitude; the kernel decides by
    narrowing the broadcast axes away and asking whether a single stored value
    is non-zero.
    """
    index = _GCU_KEY_LENGTH // 3
    mask = torch.zeros(1, 1, 1, _GCU_KEY_LENGTH, dtype=torch.bfloat16, device=DEVICE)
    mask[..., index] = -1000.0
    mask_cpu = torch.zeros(1, 1, 1, _GCU_KEY_LENGTH, dtype=torch.bfloat16)
    mask_cpu[..., index] = -1000.0
    assert int(torch.count_nonzero(mask).item()) == 1
    return mask, mask_cpu


def _gcu_finite_offset_mask():
    """A constant added to every key: a no-op in exact arithmetic."""
    return (
        torch.full(
            (1, 1, 1, _GCU_KEY_LENGTH), -1000.0, dtype=torch.bfloat16, device=DEVICE
        ),
        torch.full((1, 1, 1, _GCU_KEY_LENGTH), -1000.0, dtype=torch.bfloat16),
    )


def _gcu_all_inf_mask():
    """The same degenerate row in the spelling the composite can build."""
    return (
        torch.full(
            (1, 1, 1, _GCU_KEY_LENGTH),
            float("-inf"),
            dtype=torch.bfloat16,
            device=DEVICE,
        ),
        torch.full((1, 1, 1, _GCU_KEY_LENGTH), float("-inf"), dtype=torch.bfloat16),
    )


def _gcu_per_head_mask():
    """A mask with a real head axis: head 1 blocks the back half of the keys.

    The model's mask has stride 0 on the head and query axes, which is what makes
    narrowing them a view of the same values. This one has a real stride there,
    so head 0 blocks nothing and a lane that read only head 0 would answer the
    maskless call.
    """

    def per_head(device):
        mask = torch.zeros(
            1, _GCU_HEADS, 1, _GCU_KEY_LENGTH, dtype=torch.bfloat16, device=device
        )
        mask[:, 1, :, _GCU_KEY_LENGTH // 2 :] = float("-inf")
        return mask

    mask, mask_cpu = per_head(DEVICE), per_head("cpu")
    assert mask.stride(1) != 0
    assert mask[0, 0].count_nonzero() == 0
    return mask, mask_cpu


def _gcu_inputs(query_length, key_length):
    generator = torch.Generator().manual_seed(20260920)

    def rand(rows):
        flat = torch.randn(1, rows, _GCU_HEADS * _GCU_HEAD_DIM, generator=generator)
        return flat.bfloat16()

    query = (
        rand(query_length)
        .view(1, query_length, _GCU_HEADS, _GCU_HEAD_DIM)
        .transpose(1, 2)
    )
    key = (
        rand(key_length).view(1, key_length, _GCU_HEADS, _GCU_HEAD_DIM).transpose(1, 2)
    )
    value = (
        rand(key_length).view(1, key_length, _GCU_HEADS, _GCU_HEAD_DIM).transpose(1, 2)
    )
    return query, key, value


def _gcu_build(mask_pair: MaskPair, query_length, key_length, grad=False) -> dict:
    query, key, value = _gcu_inputs(query_length, key_length)
    mask, mask_cpu = mask_pair()
    device = [t.to(DEVICE) for t in (query, key, value)]
    if grad:
        device = [t.requires_grad_(True) for t in device]
    reference = F.scaled_dot_product_attention(query, key, value, attn_mask=mask_cpu)

    def call(attn_mask):
        return F.scaled_dot_product_attention(
            device[0], device[1], device[2], attn_mask=attn_mask
        )

    return {
        "run": lambda: call(mask),
        "baseline": lambda: call(None),
        "reference": reference,
    }


def _gcu_compare(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2
    )


def _gcu_case(
    tag: str,
    clause: str,
    mask_pair: MaskPair,
    query_length: int = _GCU_QUERY_LENGTH,
    key_length: int = _GCU_KEY_LENGTH,
    **claim,
) -> Case:
    return Case(
        tag,
        clause,
        lambda: _gcu_build(mask_pair, query_length, key_length),
        **claim,
    )


def _gcu_cases() -> tuple[Case, ...]:
    return (
        _gcu_case(
            "no_mask_uses_the_fused_lane",
            "the maskless call, which the fused lane serves",
            lambda: (None, None),
        ),
        _gcu_case(
            "key_valid_mask_non_square",
            "the target-image call: 4096 queries attending 4122 keys, q_len != kv_len",
            lambda: (_gcu_key_valid(80, DEVICE), _gcu_key_valid(80, "cpu")),
        ),
        _gcu_case(
            "prefix_segment_square_mask",
            "the prefix-segment call at (1,1,26,26), which used to raise: the stub "
            "admitted the caller's bool mask and the leaf refused the additive the "
            "composite had built from it",
            lambda: (_gcu_key_valid(26, DEVICE), _gcu_key_valid(26, "cpu")),
            query_length=26,
            key_length=26,
        ),
        _gcu_case(
            "additive_mask_in_the_query_dtype",
            "the converted spelling: -inf on the invalid keys, bf16 like the query",
            lambda: (_gcu_additive(80, DEVICE), _gcu_additive(80, "cpu")),
        ),
        _gcu_case(
            "row_padded_to_a_multiple_of_eight",
            "a 78-wide key axis, whose rows the conversion pads to 80 and slices back",
            _gcu_row_padded_mask,
            key_length=78,
        ),
        _gcu_case(
            "declines_a_foreign_mask_dtype",
            "a float32 bias is not the query's dtype, so it stays on math: the "
            "vendor op answers SUCCESS for this mask and reads the row as bf16 "
            "bytes, so it has to be kept off the fused lane even though it computes "
            "fine there",
            _gcu_foreign_dtype_mask,
            lane="math",
        ),
        _gcu_case(
            "declines_a_strided_innermost_axis",
            "a caller-level transpose, whose innermost stride is the key count: the "
            "composite asks the stub about the caller's mask before it converts a "
            "bool one, so the un-normalised stride reaches the predicate and falls "
            "to math -- correct, and slower, which is the bound the predicate states",
            _gcu_strided_innermost_mask,
            lane="math",
        ),
        Case(
            "under_grad_stays_on_math",
            "training keeps the math path: the stub answers math while grad is "
            "enabled, because the backward leaf is a FlagGems kernel and pairing it "
            "with a vendor forward is unvalidated",
            lambda: _gcu_build(
                lambda: (_gcu_key_valid(80, DEVICE), _gcu_key_valid(80, "cpu")),
                _GCU_QUERY_LENGTH,
                _GCU_KEY_LENGTH,
                grad=True,
            ),
            lane="math",
            grad=True,
        ),
        _gcu_case(
            "all_allow_caller_mask_is_the_same_call_as_no_mask",
            "the target-image spelling: a bool allow-mask with nothing masked out "
            "converts to an expanded buffer of additive zeros, which the lane is "
            "entitled to drop, so the answer has to be the maskless one bitwise",
            lambda: (_gcu_all_allow(80, DEVICE), _gcu_all_allow(80, "cpu")),
            equals_baseline=True,
        ),
        _gcu_case(
            "all_zero_additive_mask_is_the_same_call_as_no_mask",
            "the same statement in the converted spelling the leaf is handed",
            _gcu_all_zero_mask,
            equals_baseline=True,
        ),
        _gcu_case(
            "one_non_zero_entry_is_not_dropped",
            "the case that bounds the drop: the kernel decides by narrowing the "
            "broadcast axes away and asking whether a single stored value is "
            "non-zero, so one key blocked at -1000.0 keeps the mask",
            _gcu_one_blocked_key_mask,
            differs_from_baseline=True,
        ),
        _gcu_case(
            "a_finite_mask_offset_loses_precision",
            "a recorded deviation, not a desirable behaviour: adding the same finite "
            "constant to every key is a no-op in exact arithmetic, and the vendor op "
            "applies it at reduced precision instead -- max|d| against the host grows "
            "with the constant, over 1e-1 by -1000. Qwen-Image-2.1 only ever spells "
            "the mask with 0 and -inf, so this is recorded rather than declined; if "
            "a future vendor op stops deviating here the risk written up in "
            "docs/vendors/gcu/scaled-dot-product-attention.md is stale",
            _gcu_finite_offset_mask,
            deviation_floor=1e-1,
        ),
        _gcu_case(
            "an_all_inf_mask_is_exact",
            "the degenerate row in the spelling the composite can build, which the "
            "vendor op answers exactly",
            _gcu_all_inf_mask,
            deviation_ceiling=0.0,
        ),
        _gcu_case(
            "a_per_head_mask_is_read_on_every_head",
            "a mask that is real along the head axis cannot be treated as a "
            "broadcast: head 1 blocks the back half of the keys and head 0 blocks "
            "nothing, so a lane that read only head 0 would answer the maskless call",
            _gcu_per_head_mask,
            differs_from_baseline=True,
        ),
    )


# --------------------------------------------------------------------------- #
# The table.
# --------------------------------------------------------------------------- #

SDPA_LANE_ROWS: dict[str, LaneRow] = {
    "ascend": LaneRow(
        lane="aclnn",
        # The ACLNN path is reached through the composite, so the leaves the mode
        # sees are the composite's decomposition rather than the vendor kernel --
        # what this row asserts is the answer, and the mask spelling that answer
        # encodes.
        observe=False,
        rules=(),
        compare=_ascend_compare,
        no_grad=False,
        diff_bound=2e-2,
        shared_cases=COMMON_TAGS,
        common_lanes={tag: "aclnn" for tag in COMMON_TAGS},
        additive_cases=_ascend_cases(),
        measured_on="Ascend910 / CANN 9.0.0",
    ),
    "gcu": LaneRow(
        lane="fused",
        observe=True,
        rules=(
            ("_scaled_dot_product_efficient_attention.default", "fused"),
            ("_safe_softmax.default", "math"),
            ("bmm.default", "math"),
        ),
        compare=_gcu_compare,
        # The diffusion loop this lane exists for runs under torch.no_grad(); the
        # grad case asks for the other mode itself.
        no_grad=True,
        diff_bound=2e-2,
        shared_cases=COMMON_TAGS,
        # The shared cases reached math in the GCU CI run. Indexing this mapping
        # also keeps a new common case from silently inheriting the fused lane;
        # the vendor spellings below still pin the fused lane.
        common_lanes={tag: "math" for tag in COMMON_TAGS},
        additive_cases=_gcu_cases(),
        measured_on="Enflame S60 / topsaten",
        revalidated=False,
    ),
    "cuda": LaneRow(
        lane="cuda",
        observe=False,
        rules=(),
        compare=lambda actual, expected: torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2
        ),
        no_grad=False,
        diff_bound=2e-2,
        shared_cases=COMMON_TAGS,
        common_lanes={tag: "cuda" for tag in COMMON_TAGS},
        additive_cases=(),
        measured_on="CUDA, shared bounds not revalidated",
        revalidated=False,
    ),
    "dcu": LaneRow(
        lane="flaggems",
        observe=False,
        rules=(),
        compare=lambda actual, expected: torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=5e-2, atol=5e-2
        ),
        no_grad=False,
        diff_bound=5e-2,
        shared_cases=COMMON_TAGS,
        common_lanes={tag: "flaggems" for tag in COMMON_TAGS},
        additive_cases=(),
        measured_on="Hygon DCU, shared FlagGems bounds inherited",
        revalidated=False,
    ),
    "iluvatar": LaneRow(
        lane="iluvatar",
        observe=False,
        rules=(),
        compare=lambda actual, expected: torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2
        ),
        no_grad=False,
        diff_bound=2e-2,
        shared_cases=COMMON_TAGS,
        common_lanes={tag: "iluvatar" for tag in COMMON_TAGS},
        additive_cases=(),
        measured_on="Iluvatar BI-V150, shared bounds not revalidated",
        revalidated=False,
    ),
    "kunlunxin": LaneRow(
        lane="kunlunxin",
        observe=False,
        rules=(),
        compare=lambda actual, expected: torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2
        ),
        no_grad=False,
        diff_bound=2e-2,
        shared_cases=COMMON_TAGS,
        common_lanes={tag: "kunlunxin" for tag in COMMON_TAGS},
        additive_cases=(),
        measured_on="Kunlunxin P800 / XDNN, shared bounds not revalidated",
        revalidated=False,
    ),
    "metax": LaneRow(
        lane="flaggems",
        observe=False,
        rules=(),
        compare=lambda actual, expected: torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=5e-2, atol=5e-2
        ),
        no_grad=False,
        diff_bound=5e-2,
        shared_cases=COMMON_TAGS,
        common_lanes={tag: "flaggems" for tag in COMMON_TAGS},
        additive_cases=(),
        measured_on="MetaX C550, bounds inherited from DCU",
        revalidated=False,
    ),
    "mps": LaneRow(
        lane="mps",
        observe=False,
        rules=(),
        compare=lambda actual, expected: torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2
        ),
        no_grad=False,
        diff_bound=2e-2,
        shared_cases=COMMON_TAGS,
        common_lanes={tag: "mps" for tag in COMMON_TAGS},
        additive_cases=(),
        measured_on="Apple Silicon MPS, shared bounds not revalidated",
        revalidated=False,
    ),
    "musa": LaneRow(
        lane="musa",
        observe=False,
        rules=(),
        compare=lambda actual, expected: torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2
        ),
        no_grad=False,
        diff_bound=2e-2,
        shared_cases=COMMON_TAGS,
        common_lanes={tag: "musa" for tag in COMMON_TAGS},
        additive_cases=(),
        measured_on="Moore Threads S5000 / MUSA, shared bounds not revalidated",
        revalidated=False,
    ),
    "ppu": LaneRow(
        lane="cuda",
        observe=False,
        rules=(),
        compare=lambda actual, expected: torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2
        ),
        no_grad=False,
        diff_bound=2e-2,
        shared_cases=COMMON_TAGS,
        common_lanes={tag: "cuda" for tag in COMMON_TAGS},
        additive_cases=(),
        measured_on="PPU / CUDA-compatible SDK, shared bounds not revalidated",
        revalidated=False,
    ),
}


def check_rows() -> None:
    """Reject a platform or shared case that did not get a row."""
    assert set(SDPA_LANE_ROWS) == set(SHARED_ROUTES), (
        f"SDPA rows {sorted(SDPA_LANE_ROWS)} must cover {sorted(SHARED_ROUTES)}"
    )
    for accel, row in SDPA_LANE_ROWS.items():
        assert row.shared_cases == COMMON_TAGS, (
            f"{accel} does not carry every shared SDPA case"
        )
        assert set(row.common_lanes) == set(COMMON_TAGS), (
            f"{accel} must name a lane for every shared SDPA case"
        )
        shared_tags = set(row.shared_cases)
        additive_tags = [case.tag for case in row.additive_cases]
        duplicates = set(additive_tags) & shared_tags
        assert not duplicates, (
            f"{accel} has additive cases that repeat shared cases: {sorted(duplicates)}"
        )
        assert len(additive_tags) == len(set(additive_tags)), (
            f"{accel} has duplicate additive cases: {additive_tags}"
        )
        assert SHARED_ROUTES[accel].flaggems == (row.lane == "flaggems"), (
            f"{accel} has an inconsistent configured SDPA arm"
        )


check_rows()


def declared_lines(accel: str) -> str:
    """This platform's SDPA row, or the configuration failure."""
    row = SDPA_LANE_ROWS.get(accel)
    if row is None:
        return f"{accel or '<no conf>'}: no SDPA lane row"
    return (
        f"{accel}: lane={row.lane} cases={len(row.cases)} "
        f"measured_on={row.measured_on}"
        + ("" if row.revalidated else " (bounds not revalidated)")
    )


register_domain("sdpa_lanes", sys.modules[__name__])
