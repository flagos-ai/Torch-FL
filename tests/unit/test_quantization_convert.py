# Copyright 2026 FlagOS Contributors

import pytest
import torch
from torch import nn

from torch_fl.quantization import SoftLowpLinear, convert


def test_convert_requires_accelerator_resident_weights():
    model = nn.Sequential(nn.Linear(4, 3))
    with pytest.raises(RuntimeError, match="accelerator-resident weights"):
        convert(model, format="float8_e4m3fn")


def test_convert_replaces_nested_linear_on_meta_device():
    model = nn.Sequential(nn.Linear(4, 3, device="meta"))
    with pytest.raises(RuntimeError, match="accelerator-resident weights"):
        convert(model, format="float8_e4m3fn")


def test_convert_rejects_block_size_until_block_path_exists():
    model = nn.Sequential(nn.Linear(4, 3, device="meta"))
    with pytest.raises(NotImplementedError, match="block-scaled"):
        convert(model, format="float8_e4m3fn", block_size=(1, 32))


def test_soft_lowp_linear_state_is_explicit():
    module = SoftLowpLinear(
        4,
        3,
        format_id="float8_e4m3fn",
        bias=True,
        orig_dtype=torch.bfloat16,
        store_master_weights=False,
    )
    assert module.format == "float8_e4m3fn"
    assert module.in_features == 4
    assert module.out_features == 3
