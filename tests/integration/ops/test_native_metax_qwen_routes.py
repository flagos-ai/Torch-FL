# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Installed native routes needed by the unmodified eager Qwen3 workload."""

from pathlib import Path

import pytest
import torch
import torch_fl

pytestmark = [pytest.mark.metax, pytest.mark.native_source_host, pytest.mark.flaggems]


@pytest.fixture(autouse=True)
def native_contract():
    assert torch_fl._build_config.NATIVE_HOSTS
    assert "boxing" not in torch_fl._build_config.KERNELS
    assert (
        Path(torch_fl.backend_config_path()).name == "backends_metax_native_host.conf"
    )
    import flag_gems

    assert flag_gems.runtime.device.name == "flagos"


@pytest.mark.parametrize("operation", ["add", "mul", "mean", "cat", "slice", "mul_"])
@torch.no_grad()
def test_qwen_native_route_matches_cpu(operation):
    generator = torch.Generator().manual_seed(845)
    reference = torch.randn(1, 60, 1024, generator=generator)
    if operation in ("add", "mul", "cat"):
        reference = reference.bfloat16()
    value = reference.to("flagos:0")
    if operation == "add":
        actual, expected = value + value, reference + reference
    elif operation == "mul":
        actual, expected = value * value, reference * reference
    elif operation == "mean":
        actual, expected = (
            value.mean(-1, keepdim=True),
            reference.mean(-1, keepdim=True),
        )
    elif operation == "cat":
        actual, expected = (
            torch.cat([value, value], dim=1),
            torch.cat([reference, reference], dim=1),
        )
    elif operation == "slice":
        actual, expected = value[:, 1::2, :], reference[:, 1::2, :]
        assert actual.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        assert actual.stride() == expected.stride()
    else:
        address = value.data_ptr()
        actual, expected = value.mul_(value), reference.mul_(reference)
        assert actual.data_ptr() == address
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("tokens", [1, 60])
@pytest.mark.parametrize("out_variant", [False, True])
@torch.no_grad()
def test_qwen_rope_small_k_bmm(tokens, out_variant):
    generator = torch.Generator().manual_seed(832)
    left = torch.randn(1, 64, 1, generator=generator) * 0.25
    right = torch.randn(1, 1, tokens, generator=generator) * 0.25
    a, b = left.to("flagos:0"), right.to("flagos:0")
    if out_variant:
        storage = torch.empty(1, 64, tokens * 2, device="flagos:0")
        output = storage[..., ::2]
        actual = torch.bmm(a, b, out=output)
        assert actual.data_ptr() == output.data_ptr()
        assert actual.stride() == output.stride()
    else:
        actual = torch.bmm(a, b)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(
        actual.cpu(), torch.bmm(left, right), atol=1e-3, rtol=1e-3
    )


@torch.no_grad()
def test_missing_numerical_route_still_rejects_cpu_fallback():
    value = torch.ones(4, device="flagos:0")
    with pytest.raises(
        RuntimeError, match="native FlagGems hosts do not permit CPU fallback"
    ):
        torch.ops.aten.igamma.default(value, value)


@torch.no_grad()
def test_float64_matmul_rejects_the_explicit_cpu_escape():
    value = torch.ones(2, 2, dtype=torch.float64).to("flagos:0")
    with pytest.raises(
        RuntimeError, match="CPU fallback: aten::mm .*host matmul escape"
    ):
        torch.mm(value, value)
