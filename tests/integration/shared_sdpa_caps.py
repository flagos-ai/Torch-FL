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

"""The scaled-dot-product-attention calls every configured platform runs.

The inputs are float32 tensors built on the host and moved to ``flagos:0``; the
reference is the float32 host result. Each row of the platform table names the
route this suite drives and whether it expects the FlagGems callable to see the
call. Every row must compose every case here. A missing route row or a missing
case is a failure in the configured hardware job rather than a skip.

The no-mask, all-allow, key-dropping bool mask, additive mask, finite bias,
non-square and causal calls live here. The vendor rows add the spellings that
are only meaningful on their backend, and the FlagGems row adds the wider shape
envelope the shared C++ eligibility clause admits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

from device_init import torch
import torch.nn.functional as F

#: The device string ``[flagos dispatch]`` prints for the flagos device.
DEVICE = "flagos:0"

#: The tolerance for every common call at the shared argument shapes. Numeric
#: bounds remain per platform; a platform that cannot hold this contract fails
#: its own row.
COMMON_RTOL = 2e-2
COMMON_ATOL = 2e-2

Operands = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class CommonCase:
    """One parameterised call in the table."""

    tag: str
    clause: str
    setup: Callable[[], Operands]
    run: Callable[[Operands], torch.Tensor]
    reference: Callable[[Operands], torch.Tensor]
    lane: str | None = None
    math: bool = False
    causal: bool = False
    mask: Callable[[], torch.Tensor] | None = None
    triangular: bool = False
    maskless: Callable[..., torch.Tensor] | None = None
    changes_answer: bool | None = None
    identity: Callable[[Operands], torch.Tensor] | None = None


def _small() -> Operands:
    generator = torch.Generator().manual_seed(1729)
    return (
        torch.randn(1, 1, 2, 8, generator=generator),
        torch.randn(1, 1, 8, 8, generator=generator),
        torch.randn(1, 1, 8, 8, generator=generator),
    )


def _heads() -> Operands:
    generator = torch.Generator().manual_seed(7)
    return (
        torch.randn(1, 2, 5, 16, generator=generator),
        torch.randn(1, 2, 5, 16, generator=generator),
        torch.randn(1, 2, 5, 16, generator=generator),
    )


def _nonsquare() -> Operands:
    generator = torch.Generator().manual_seed(23)
    return (
        torch.randn(1, 2, 2, 8, generator=generator),
        torch.randn(1, 2, 3, 8, generator=generator),
        torch.randn(1, 2, 3, 8, generator=generator),
    )


def _causal() -> Operands:
    generator = torch.Generator().manual_seed(29)
    return (
        torch.randn(1, 1, 4, 8, generator=generator),
        torch.randn(1, 1, 4, 8, generator=generator),
        torch.randn(1, 1, 4, 8, generator=generator),
    )


def _allow(query_length: int, key_length: int) -> torch.Tensor:
    return torch.ones(1, 1, query_length, key_length, dtype=torch.bool)


def _partly_valid(key_length: int) -> torch.Tensor:
    """One key-invalid row broadcast over every query."""
    mask = torch.ones(1, 1, 1, key_length, dtype=torch.bool)
    mask[..., 0] = False
    return mask


def _causal_allow(query_length: int, key_length: int) -> torch.Tensor:
    return torch.tril(torch.ones(query_length, key_length, dtype=torch.bool))


def _additive(query_length: int, key_length: int) -> torch.Tensor:
    mask = torch.zeros(1, 1, query_length, key_length)
    mask[..., 0] = float("-inf")
    return mask


def _bias(shape: tuple[int, ...]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(19)
    return torch.randn(shape, generator=generator) * 0.25


def _causal_additive(query_length: int, key_length: int) -> torch.Tensor:
    return torch.zeros(query_length, key_length).masked_fill(
        ~_causal_allow(query_length, key_length),
        float("-inf"),
    )


def _host(operands: Operands, **kwargs: object) -> torch.Tensor:
    query, key, value = (tensor.float() for tensor in operands)
    return F.scaled_dot_product_attention(query, key, value, **kwargs)


def _runtime_sdpa(
    operands: Operands,
    mask: torch.Tensor | None = None,
    *,
    causal: bool = False,
) -> Callable[[], torch.Tensor]:
    """The device call, made while the test's dispatch recorder is active."""
    query, key, value = (tensor.to(DEVICE) for tensor in operands)
    device_mask = None if mask is None else mask.to(DEVICE)
    return lambda: F.scaled_dot_product_attention(
        query, key, value, attn_mask=device_mask, is_causal=causal
    )


def _plain_case(
    tag: str,
    clause: str,
    setup: Callable[[], Operands],
    mask: Callable[[], torch.Tensor] | None = None,
    *,
    causal: bool = False,
    triangular: bool = False,
    changes_answer: bool | None = None,
) -> CommonCase:
    def reference(operands: Operands) -> torch.Tensor:
        return _host(
            operands,
            attn_mask=None if mask is None or causal else mask(),
            is_causal=causal,
        )

    def run(operands: Operands) -> torch.Tensor:
        return _runtime_sdpa(
            operands,
            None if causal or mask is None else mask(),
            causal=causal,
        )()

    def maskless() -> torch.Tensor:
        return _runtime_sdpa(setup(), causal=False)()

    identity = _host if changes_answer is not None else None

    return CommonCase(
        tag,
        clause,
        setup,
        run,
        reference,
        causal=causal,
        mask=mask,
        triangular=triangular,
        maskless=maskless if changes_answer is not None else None,
        changes_answer=changes_answer,
        identity=identity,
    )


def _math_case() -> CommonCase:
    operands = _small()

    def run(operands: Operands) -> torch.Tensor:
        query, key, value = (tensor.to(DEVICE) for tensor in operands)
        return _runtime_math(query, key, value)

    query, key, value = operands
    scores = query.float() @ key.float().transpose(-2, -1) * key.shape[-1] ** -0.5
    expected = torch.cat(
        (_host(operands).flatten(), torch.softmax(scores, -1).flatten())
    )

    def reference(operands: Operands) -> torch.Tensor:
        del operands
        return expected

    def setup() -> Operands:
        return operands

    return CommonCase(
        "math",
        "the math SDPA operator, whose output and attention weights are checked",
        setup,
        run,
        reference,
        lane="math",
        math=True,
    )


def _runtime_math(query, key, value) -> torch.Tensor:
    output, weights = torch.ops.aten._scaled_dot_product_attention_math(
        query, key, value, None, 0.0, False, None
    )
    # Both the output and the weights are checked. The score scale is the one
    # the SDPA operator documents, computed on the host from the same operands
    # as the device call.
    return torch.cat((output.flatten(), weights.flatten()))


COMMON_CASES: tuple[CommonCase, ...] = (
    _plain_case("no_mask", "a float32 [1,1,2,8] call without a mask", _small),
    _plain_case(
        "all_allow_mask",
        "an all-true rank-4 bool mask for a float32 call, where every key attends",
        _small,
        lambda: _allow(1, 8),
        changes_answer=False,
    ),
    _plain_case(
        "key_valid_mask",
        "a float32 call with a (1,1,1,8) bool row whose invalid key changes the answer",
        _small,
        lambda: _partly_valid(8),
        changes_answer=True,
    ),
    _plain_case(
        "additive_zero_neg_inf",
        "a float32 call with the partly-valid mask in 0/-inf spelling",
        _small,
        lambda: _additive(1, 8),
        changes_answer=True,
    ),
    _plain_case(
        "finite_additive_bias",
        "a float32 call with a finite rank-4 additive bias",
        _heads,
        lambda: _bias((1, 1, 5, 5)),
    ),
    _plain_case(
        "non_square",
        "a float32 two-query, three-key call",
        _causal,
    ),
    _plain_case(
        "causal",
        "a square float32 causal call",
        _causal,
        causal=True,
        changes_answer=True,
    ),
    _plain_case(
        "causal_with_allow_mask",
        "a float32 call with an explicit triangular allow mask and causal flag",
        _causal,
        mask=lambda: _causal_allow(4, 4),
        triangular=True,
        changes_answer=False,
    ),
    _plain_case(
        "causal_with_neg_inf",
        "the same float32 causal mask in 0/-inf form",
        _causal,
        mask=lambda: _causal_additive(4, 4),
        triangular=True,
        changes_answer=False,
    ),
    _plain_case(
        "additive_batch",
        "a float32 non-square call with a finite additive bias",
        _nonsquare,
        lambda: _bias((1, 2, 2, 3)),
    ),
    _math_case(),
)

COMMON_TAGS: tuple[str, ...] = tuple(case.tag for case in COMMON_CASES)
COMMON_BY_TAG: dict[str, CommonCase] = {case.tag: case for case in COMMON_CASES}
COMMON_ONLY_TAGS = tuple(tag for tag in COMMON_TAGS if tag != "math")


def assert_every_mask_is_read() -> None:
    """Check on the host that each mask has the effect its case claims.

    A wrongly shaped bool mask or a zero additive passes the comparison below
    while exercising no mask path, so both sides of the mask claim are checked.
    """
    for case in COMMON_CASES:
        if case.changes_answer is None or case.maskless is None:
            continue
        operands = case.setup()
        if case.triangular:
            causal = _host(operands, is_causal=True)
            difference = (case.reference(operands) - causal).abs().max().item()
            assert difference <= COMMON_RTOL, (
                f"{case.tag} ({case.clause}): the triangular mask does not "
                f"match the causal call, moving it by {difference:.3e}"
            )
            assert (causal - _host(operands)).abs().max().item() > COMMON_ATOL
            continue
        identity = case.identity
        assert identity is not None
        difference = (case.reference(operands) - identity(operands)).abs().max().item()
        if case.changes_answer:
            assert difference > COMMON_ATOL, (
                f"{case.tag} ({case.clause}): the mask moved the host answer by "
                f"{difference:.3e}, within the comparison tolerance"
            )
        else:
            assert difference <= COMMON_RTOL, (
                f"{case.tag} ({case.clause}): the identity mask moved the host "
                f"answer by {difference:.3e}"
            )


def assert_mask_claim(case: CommonCase, actual: torch.Tensor) -> None:
    """Check the device result against the same call without a mask.

    This checks that the mask reached the route under test and had the semantic
    effect the case names. The host check below validates the other half.
    """
    if case.changes_answer is None or case.maskless is None:
        return
    actual = actual.float()
    maskless = case.maskless()
    if case.triangular:
        causal = _runtime_sdpa(case.setup(), causal=True)().float()
        torch.testing.assert_close(causal, actual, rtol=COMMON_RTOL, atol=COMMON_ATOL)
        return
    difference = (actual.cpu().float() - maskless.cpu().float()).abs().max().item()
    if case.changes_answer:
        assert difference > COMMON_ATOL, (
            f"{case.tag} ({case.clause}): the answer moved by only "
            f"{difference:.3e}; the mask was not applied"
        )
    else:
        assert difference <= COMMON_RTOL, (
            f"{case.tag} ({case.clause}): the identity mask moved the answer "
            f"by {difference:.3e}"
        )


def assert_case(case: CommonCase, actual: torch.Tensor) -> None:
    """Run the one shared result contract for every configured platform."""
    expected = case.reference(case.setup())
    assert actual.device == torch.device(DEVICE), (
        f"{case.tag} ({case.clause}): the result is on {actual.device}"
    )
    torch.testing.assert_close(
        actual.cpu().float(),
        expected.float(),
        rtol=COMMON_RTOL,
        atol=COMMON_ATOL,
        msg=f"{case.tag} ({case.clause})",
    )
    assert_mask_claim(case, actual)


def route_claimed(accel: str) -> bool:
    """Whether this platform's configured wrapper calls the FlagGems callable."""
    return SHARED_ROUTES[accel].flaggems


class SharedRoute(NamedTuple):
    """The configured SDPA arm every platform takes for the common calls."""

    lane: str | None
    flaggems: bool
    describe: str


SHARED_ROUTES: dict[str, SharedRoute] = {
    "ascend": SharedRoute(
        "aclnn",
        False,
        "ACLNN fused attention through the boxed composite and the math path",
    ),
    "cuda": SharedRoute(
        "cuda",
        False,
        "CUDA composite with CUDA-backed leaves",
    ),
    "dcu": SharedRoute(
        "flaggems",
        True,
        "FlagGems Triton attention through the DCU hcu backend",
    ),
    "gcu": SharedRoute(
        "fused",
        False,
        "the GCU fused leaf through the boxed composite",
    ),
    "iluvatar": SharedRoute(
        None,
        False,
        "the composite with native and FlagGems leaves for the unlisted arm",
    ),
    "kunlunxin": SharedRoute(
        None,
        False,
        "the boxed composite and XDNN leaves, with the arm left unobserved",
    ),
    "metax": SharedRoute(
        "flaggems",
        True,
        "FlagGems Triton attention through the MetaX backend",
    ),
    "musa": SharedRoute(
        None,
        False,
        "the boxed composite and MUSA leaves, with the arm left unobserved",
    ),
    "ppu": SharedRoute(
        "cuda",
        False,
        "the PPU CUDA-compatible composite and leaves",
    ),
    "mps": SharedRoute(
        None,
        False,
        "the native Metal composite and leaves",
    ),
}
