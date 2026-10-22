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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch


class LowPrecisionFormat(str, Enum):
    FLOAT8_E4M3FN = "float8_e4m3fn"
    FLOAT8_E5M2 = "float8_e5m2"
    FLOAT8_E4M3FNUZ = "float8_e4m3fnuz"
    FLOAT8_E5M2FNUZ = "float8_e5m2fnuz"
    FLOAT8_E8M0FNU = "float8_e8m0fnu"
    FLOAT4_E2M1FN_X2 = "float4_e2m1fn_x2"
    MXFP4 = "mxfp4"
    NVFP4 = "nvfp4"
    BLOCK_FP4 = "block_fp4"


@dataclass(frozen=True)
class FormatSpec:
    format: LowPrecisionFormat
    bits_per_value: int
    values_per_byte: int
    torch_dtype_name: str | None
    block_shape: tuple[int, int] | None
    scale_dtype_name: str | None
    max_finite: float
    signed: bool = True
    has_inf: bool = False
    has_nan: bool = False
    unsigned_zero: bool = False

    @property
    def is_block_scaled(self) -> bool:
        return self.block_shape is not None

    @property
    def is_packed(self) -> bool:
        return self.values_per_byte > 1

    def torch_dtype(self) -> "torch.dtype | None":
        if self.torch_dtype_name is None:
            return None
        import torch

        dtype = getattr(torch, self.torch_dtype_name, None)
        if dtype is None:
            raise RuntimeError(
                f"The active PyTorch build does not define torch.{self.torch_dtype_name}"
            )
        return dtype

    def scale_dtype(self) -> "torch.dtype | None":
        if self.scale_dtype_name is None:
            return None
        import torch

        dtype = getattr(torch, self.scale_dtype_name, None)
        if dtype is None:
            raise RuntimeError(
                f"The active PyTorch build does not define torch.{self.scale_dtype_name}"
            )
        return dtype


_FORMAT_SPECS = {
    LowPrecisionFormat.FLOAT8_E4M3FN: FormatSpec(
        format=LowPrecisionFormat.FLOAT8_E4M3FN,
        bits_per_value=8,
        values_per_byte=1,
        torch_dtype_name="float8_e4m3fn",
        block_shape=None,
        scale_dtype_name=None,
        max_finite=448.0,
        has_nan=True,
    ),
    LowPrecisionFormat.FLOAT8_E5M2: FormatSpec(
        format=LowPrecisionFormat.FLOAT8_E5M2,
        bits_per_value=8,
        values_per_byte=1,
        torch_dtype_name="float8_e5m2",
        block_shape=None,
        scale_dtype_name=None,
        max_finite=57344.0,
        has_inf=True,
        has_nan=True,
    ),
    LowPrecisionFormat.FLOAT8_E4M3FNUZ: FormatSpec(
        format=LowPrecisionFormat.FLOAT8_E4M3FNUZ,
        bits_per_value=8,
        values_per_byte=1,
        torch_dtype_name="float8_e4m3fnuz",
        block_shape=None,
        scale_dtype_name=None,
        max_finite=240.0,
        has_nan=True,
        unsigned_zero=True,
    ),
    LowPrecisionFormat.FLOAT8_E5M2FNUZ: FormatSpec(
        format=LowPrecisionFormat.FLOAT8_E5M2FNUZ,
        bits_per_value=8,
        values_per_byte=1,
        torch_dtype_name="float8_e5m2fnuz",
        block_shape=None,
        scale_dtype_name=None,
        max_finite=57344.0,
        has_nan=True,
        unsigned_zero=True,
    ),
    LowPrecisionFormat.FLOAT8_E8M0FNU: FormatSpec(
        format=LowPrecisionFormat.FLOAT8_E8M0FNU,
        bits_per_value=8,
        values_per_byte=1,
        torch_dtype_name="float8_e8m0fnu",
        block_shape=None,
        scale_dtype_name=None,
        max_finite=2.0**127,
        signed=False,
        has_nan=True,
        unsigned_zero=True,
    ),
    LowPrecisionFormat.FLOAT4_E2M1FN_X2: FormatSpec(
        format=LowPrecisionFormat.FLOAT4_E2M1FN_X2,
        bits_per_value=4,
        values_per_byte=2,
        torch_dtype_name="float4_e2m1fn_x2",
        block_shape=None,
        scale_dtype_name=None,
        max_finite=6.0,
    ),
    LowPrecisionFormat.MXFP4: FormatSpec(
        format=LowPrecisionFormat.MXFP4,
        bits_per_value=4,
        values_per_byte=2,
        torch_dtype_name=None,
        block_shape=(1, 32),
        scale_dtype_name="float8_e8m0fnu",
        max_finite=6.0,
    ),
    LowPrecisionFormat.NVFP4: FormatSpec(
        format=LowPrecisionFormat.NVFP4,
        bits_per_value=4,
        values_per_byte=2,
        torch_dtype_name=None,
        block_shape=(1, 16),
        scale_dtype_name="float8_e4m3fn",
        max_finite=6.0,
    ),
    LowPrecisionFormat.BLOCK_FP4: FormatSpec(
        format=LowPrecisionFormat.BLOCK_FP4,
        bits_per_value=4,
        values_per_byte=2,
        torch_dtype_name=None,
        block_shape=(1, 32),
        scale_dtype_name="float32",
        max_finite=6.0,
    ),
}


def normalize_format(value: str | LowPrecisionFormat) -> LowPrecisionFormat:
    if isinstance(value, LowPrecisionFormat):
        return value
    normalized = value.lower().removeprefix("torch.")
    try:
        return LowPrecisionFormat(normalized)
    except ValueError as exc:
        choices = ", ".join(item.value for item in LowPrecisionFormat)
        raise ValueError(
            f"Unsupported low-precision format {value!r}; expected one of: {choices}"
        ) from exc


def get_format_spec(value: str | LowPrecisionFormat) -> FormatSpec:
    return _FORMAT_SPECS[normalize_format(value)]


def supported_formats() -> tuple[str, ...]:
    return tuple(item.value for item in LowPrecisionFormat)
