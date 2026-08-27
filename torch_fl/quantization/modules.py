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

import torch
from torch import nn
from torch.nn import functional as F

from .formats import LowPrecisionFormat, get_format_spec, normalize_format


class _QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        ctx.input_dtype = value.dtype
        return value.to(dtype).to(value.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class SoftLowpLinear(nn.Module):
    """Transparent linear wrapper for a tensorwise PyTorch FP8 payload."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        format_id: LowPrecisionFormat,
        bias: bool,
        orig_dtype: torch.dtype,
        store_master_weights: bool,
    ) -> None:
        super().__init__()
        format_id = normalize_format(format_id)
        self.in_features = in_features
        self.out_features = out_features
        self.format = format_id.value
        self.orig_dtype = orig_dtype
        self.store_master_weights = store_master_weights
        self.register_buffer("weight_payload", None, persistent=True)
        if store_master_weights:
            self.master_weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=orig_dtype)
            )
        else:
            self.register_parameter("master_weight", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=orig_dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        format_id: LowPrecisionFormat,
        store_master_weights: bool,
    ) -> "SoftLowpLinear":
        spec = get_format_spec(format_id)
        dtype = spec.torch_dtype()
        if dtype is None:
            raise NotImplementedError(
                f"No PyTorch storage dtype is available for {format_id.value}"
            )
        result = cls(
            linear.in_features,
            linear.out_features,
            format_id=format_id,
            bias=linear.bias is not None,
            orig_dtype=linear.weight.dtype,
            store_master_weights=store_master_weights,
        ).to(device=linear.weight.device)
        with torch.no_grad():
            result.weight_payload = linear.weight.detach().to(dtype)
            if result.master_weight is not None:
                result.master_weight.copy_(linear.weight.detach())
            if result.bias is not None and linear.bias is not None:
                result.bias.copy_(linear.bias.detach())
        return result

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_payload

    def _weight_for_compute(self) -> torch.Tensor:
        if self.master_weight is not None and self.training:
            return _QuantizeSTE.apply(self.master_weight, self.weight_payload.dtype)
        return self.weight_payload

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device != self.weight_payload.device:
            raise RuntimeError(
                f"soft-lowp linear requires input and weight on the same device, "
                f"got {input.device} and {self.weight_payload.device}"
            )
        # The native low-precision operator route is supplied by the C++ backend.
        # Keeping the call in F.linear preserves standard Transformers behavior and
        # lets the backend own dtype-aware dispatch without a Python CPU fallback.
        weight = self._weight_for_compute()
        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"format={self.format}, bias={self.bias is not None}"
        )
