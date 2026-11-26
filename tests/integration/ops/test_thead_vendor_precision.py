# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from torch_fl.native_host import vendor

pytestmark = [
    pytest.mark.ppu,
    pytest.mark.native_source_host,
    pytest.mark.native_vendor_precision,
    pytest.mark.flaggems,
]


def test_thead_mul_preserves_broadcast_layout_and_wrapped_scalar():
    lhs = torch.arange(128, dtype=torch.bfloat16).reshape(1, 1, 1, 128)
    rhs = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8, 1, 1)
    lhs_device = lhs.to("flagos:0")
    rhs_device = rhs.to("flagos:0")
    before = vendor.audit()["hits"]["mul.Tensor"]
    with torch.inference_mode():
        result = lhs_device * rhs_device
    target = lhs * rhs
    assert result.stride() == target.stride()
    torch.testing.assert_close(result.cpu(), target, atol=0, rtol=0)
    with torch.no_grad():
        result = result.transpose(1, 3) * 0.25
    torch.testing.assert_close(
        result.cpu(), target.transpose(1, 3) * 0.25, atol=0, rtol=0
    )
    assert vendor.audit()["hits"]["mul.Tensor"] == before + 2


def test_thead_cat_preserves_narrow_views_and_falls_back_outside_guard():
    value = torch.arange(768, dtype=torch.bfloat16).reshape(1, 2, 3, 128)
    value_device = value.to("flagos:0")
    parts = value.transpose(1, 2).split(64, dim=-1)
    parts_device = value_device.transpose(1, 2).split(64, dim=-1)
    before = vendor.audit()["hits"]["cat"]
    with torch.inference_mode():
        result = torch.cat(parts_device, dim=-1)
        fallback = torch.cat((value_device, value_device, value_device), dim=-1)
    target = torch.cat(parts, dim=-1)
    assert result.stride() == target.stride()
    torch.testing.assert_close(result.cpu(), target, atol=0, rtol=0)
    torch.testing.assert_close(
        fallback.cpu(), torch.cat((value, value, value), dim=-1), atol=0, rtol=0
    )
    assert vendor.audit()["hits"]["cat"] == before + 1
