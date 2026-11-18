# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Installed optional vendor precision; original CPU core and device storage."""

import pytest
import torch
import torch_fl

from torch_fl.native_host import vendor

pytestmark = [pytest.mark.metax, pytest.mark.native_source_host, pytest.mark.flaggems]


@pytest.fixture(autouse=True)
def measured_vendor_cohort():
    assert torch_fl._build_config.NATIVE_HOSTS
    assert torch_fl._build_config.NATIVE_VENDOR
    assert "boxing" not in torch_fl._build_config.KERNELS
    vendor._verify_cpu_core()


def _values():
    # Frozen manufacturer results include the observed midpoint and signed
    # underflow. The independent diagnostic covers all BF16 bit patterns.
    source = torch.tensor([5.9375, -87.5, -88.0, -88.5, 0.0, -0.0]).bfloat16()
    expected = torch.tensor([5.9375, -0.0, -0.0, -0.0, 0.0, -0.0]).bfloat16()
    return source, expected


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided", "broadcast"])
@torch.no_grad()
def test_measured_bf16_silu_uses_original_vendor_numerics(layout):
    source, expected = _values()
    source = source.repeat(60, 1).to("flagos")
    expected = expected.repeat(60, 1)
    if layout == "transpose":
        source, expected = source.t(), expected.t()
    elif layout == "strided":
        source, expected = source[:, ::2], expected[:, ::2]
    elif layout == "broadcast":
        source, expected = source[:1].expand(60, -1), expected[:1].expand(60, -1)
    before = vendor.audit()["hits"]["silu"]
    output = torch.nn.functional.silu(source)
    torch_fl.flagos.synchronize()
    assert output.device == source.device and output.dtype == source.dtype
    assert torch.equal(
        output.cpu().contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
    )
    assert vendor.audit()["hits"]["silu"] == before + 1


@torch.no_grad()
def test_vendor_silu_empty_input_retains_device_storage():
    source = torch.empty(0, 6, dtype=torch.bfloat16, device="flagos")
    output = torch.nn.functional.silu(source)
    assert output.shape == source.shape and output.device == source.device


@torch.no_grad()
def test_unmeasured_silu_dtype_keeps_the_original_flaggems_route():
    before = vendor.audit()["hits"]["silu"]
    source = torch.linspace(-2, 2, 31)
    output = torch.nn.functional.silu(source.to("flagos"))
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(
        output.cpu(), torch.nn.functional.silu(source), atol=1e-6, rtol=1e-6
    )
    assert vendor.audit()["hits"]["silu"] == before


@torch.no_grad()
def test_vendor_silu_uses_the_current_flagos_stream():
    source, expected = _values()
    stream = torch.Stream(device="flagos:0")
    with stream:
        output = torch.nn.functional.silu(source.to("flagos"))
    stream.synchronize()
    assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16))


def test_vendor_precision_audit_includes_sdk_and_typed_stub():
    audit = vendor.audit()
    assert audit["vendor"] == "metax"
    assert {row["op"] for row in audit["catalog"]} == {
        "mm",
        "bmm",
        "silu",
        "_softmax",
        "mean.dim",
    }
    assert set(audit["numerical_libraries"]) == {"torch_cuda", "c10_cuda"}
    assert set(audit["sdk_precision"]["numerical_libraries"]) == {
        "mcblas",
        "mcblas_lt",
        "mcruntime",
    }
    assert "libtorch_cpu.so" not in " ".join(audit["artifacts"])


@pytest.mark.parametrize(
    "shape", [(0, 6), (2, 0), (1, 16, 1, 66), (1, 16, 60, 60), (3, 129), (2, 572)]
)
@torch.no_grad()
def test_vendor_fp32_softmax_shapes_and_empty_inputs(shape):
    source = torch.randn(shape, generator=torch.Generator().manual_seed(832))
    before = vendor.audit()["hits"]["_softmax"]
    output = torch.softmax(source.to("flagos"), dim=-1)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(
        output.cpu(), torch.softmax(source, dim=-1), atol=1e-6, rtol=1e-6
    )
    assert vendor.audit()["hits"]["_softmax"] == before + 1


@torch.no_grad()
def test_vendor_softmax_strides_and_current_stream():
    source = torch.randn(7, 13, generator=torch.Generator().manual_seed(833))
    stream = torch.Stream(device="flagos:0")
    with stream:
        value = source.to("flagos").t()
        output = torch.softmax(value, dim=-1)
    stream.synchronize()
    torch.testing.assert_close(
        output.cpu(), torch.softmax(source.t(), dim=-1), atol=1e-6, rtol=1e-6
    )


@torch.no_grad()
def test_vendor_fp32_softmax_rejects_unmeasured_half_promotion():
    value = torch.ones(2, 3, device="flagos")
    with pytest.raises(RuntimeError, match="half promotion"):
        torch.ops.aten._softmax.default(value, -1, True)


@torch.no_grad()
def test_bf16_softmax_retains_its_flaggems_route():
    before = vendor.audit()["hits"]["_softmax"]
    source = torch.randn(3, 31, generator=torch.Generator().manual_seed(832)).bfloat16()
    output = torch.softmax(source.to("flagos"), dim=-1)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(
        output.cpu(), torch.softmax(source, dim=-1), atol=1e-3, rtol=1e-3
    )
    assert vendor.audit()["hits"]["_softmax"] == before


@pytest.mark.parametrize("dims", [None, (-1,), (0,), (0, 2)])
@pytest.mark.parametrize("keepdim", [False, True])
@torch.no_grad()
def test_vendor_fp32_mean_preserves_reduction_metadata(dims, keepdim):
    source = torch.randn(2, 3, 5, generator=torch.Generator().manual_seed(832))
    before = vendor.audit()["hits"]["mean.dim"]
    actual = source.to("flagos").mean(dim=dims, keepdim=keepdim)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(
        actual.cpu(), source.mean(dim=dims, keepdim=keepdim), atol=1e-6, rtol=1e-6
    )
    assert vendor.audit()["hits"]["mean.dim"] == before + 1


@torch.no_grad()
def test_vendor_mean_empty_input_preserves_nan_identity():
    source = torch.empty(2, 0, 5, device="flagos")
    actual = source.mean(dim=1)
    torch_fl.flagos.synchronize()
    assert actual.shape == (2, 5) and torch.isnan(actual.cpu()).all()


@torch.no_grad()
def test_vendor_mean_rejects_unmeasured_output_dtype():
    value = torch.ones(2, 3, device="flagos")
    with pytest.raises(RuntimeError, match="FP32 output"):
        value.mean(dim=-1, dtype=torch.float64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 60, 3072), (1, 16, 60, 128), (7, 129)])
@torch.no_grad()
def test_published_mul_variant_preserves_large_broadcasts(dtype, shape):
    generator = torch.Generator().manual_seed(832)
    left = torch.randn(shape, generator=generator).to(dtype)
    right = torch.randn(shape[-1], generator=generator).to(dtype)
    a, b = left.to("flagos"), right.to("flagos")
    before = torch_fl.native_host.audit()["source_variant_hits"]["published_mul"]
    output = a * b
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(output.cpu(), left * right, atol=0, rtol=0)
    assert (
        torch_fl.native_host.audit()["source_variant_hits"]["published_mul"]
        == before + 1
    )
