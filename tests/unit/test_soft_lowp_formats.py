# Copyright 2026 FlagOS Contributors

import pytest

from torch_fl.quantization.formats import (
    LowPrecisionFormat,
    get_format_spec,
    normalize_format,
    supported_formats,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("float8_e4m3fn", LowPrecisionFormat.FLOAT8_E4M3FN),
        ("torch.float8_e5m2", LowPrecisionFormat.FLOAT8_E5M2),
        (LowPrecisionFormat.FLOAT4_E2M1FN_X2, LowPrecisionFormat.FLOAT4_E2M1FN_X2),
        ("mxfp4", LowPrecisionFormat.MXFP4),
    ],
)
def test_normalize_format(value, expected):
    assert normalize_format(value) is expected


def test_format_specs_distinguish_packed_and_block_formats():
    fp8 = get_format_spec("float8_e4m3fn")
    fp4 = get_format_spec("float4_e2m1fn_x2")
    mxfp4 = get_format_spec("mxfp4")

    assert not fp8.is_packed
    assert fp8.bits_per_value == 8
    assert fp4.is_packed
    assert fp4.values_per_byte == 2
    assert not fp4.is_block_scaled
    assert mxfp4.is_packed
    assert mxfp4.is_block_scaled
    assert mxfp4.block_shape == (1, 32)


def test_all_required_formats_are_public():
    assert set(supported_formats()) == {item.value for item in LowPrecisionFormat}


def test_unknown_format_has_actionable_error():
    with pytest.raises(ValueError, match="Unsupported low-precision format"):
        normalize_format("float3")
