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

from collections.abc import Iterable

from torch import nn

from .formats import LowPrecisionFormat, get_format_spec, normalize_format
from .modules import SoftLowpLinear


def _matches_excluded(name: str, excluded: Iterable[str]) -> bool:
    return any(name == item or name.startswith(f"{item}.") for item in excluded)


def convert(
    model: nn.Module,
    *,
    format: str | LowPrecisionFormat = LowPrecisionFormat.FLOAT8_E4M3FN,
    modules_to_not_convert: Iterable[str] | None = None,
    block_size: tuple[int, int] | None = None,
    store_master_weights: bool = False,
) -> nn.Module:
    """Convert supported ``nn.Linear`` modules to soft-lowp inference modules.

    The first implementation quantizes from the current parameter values. It
    intentionally requires the model to be on an accelerator: quantized payload
    creation must not silently move weights through CPU memory.
    """
    format_id = normalize_format(format)
    spec = get_format_spec(format_id)
    if spec.torch_dtype_name is None:
        raise NotImplementedError(
            f"Runtime conversion for {format_id.value} is not implemented yet; "
            "use one of the ordinary PyTorch FP8 dtypes"
        )
    if block_size is not None:
        raise NotImplementedError(
            "block_size is reserved for block-scaled formats and is not supported "
            "by the initial tensorwise conversion path"
        )
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")

    excluded = tuple(modules_to_not_convert or ())
    converted = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, SoftLowpLinear):
            continue
        if _matches_excluded(name, excluded):
            continue
        if module.weight.device.type in ("cpu", "meta"):
            raise RuntimeError(
                f"soft-lowp conversion requires accelerator-resident weights; "
                f"{name or '<root>'}.weight is on {module.weight.device}"
            )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacement = SoftLowpLinear.from_linear(
            module,
            format_id=format_id,
            store_master_weights=store_master_weights,
        )
        setattr(parent, child_name, replacement)
        converted += 1

    if converted == 0:
        raise ValueError(
            "soft-lowp conversion did not find any eligible nn.Linear modules"
        )
    return model
