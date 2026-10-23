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

"""Both ``masked_fill`` tensor spellings on Apple Silicon's MPSGraph layer.

Qwen3.5's GatedDeltaNet uses a scalar fill to mask the causal pairwise decay.
The fallback copied the temporary to the host; a queued MPSGraph read of the
same temporary could then observe a later allocation after the temporary was
released back to the allocator. This suite holds the complete value, the
broadcasting and dtype contracts, and the lifetime that made the bug visible.

The output is compared exactly when both paths select the same stored values.
NaNs are compared by shape because CPU and GPU need not preserve a particular
NaN payload. The arithmetic dtypes are compared by value and dtype.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import dispatched_to_mps, routed_backend_or_none
from platform_support import detect_platform


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]

VALUES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
]


def _device(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone().to(DEVICE)


def _reference(self_, mask, value):
    return self_.masked_fill(mask, value)


def _assert_matches(out, expected, *, exact: bool = True) -> None:
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    actual = out.cpu()
    assert not (torch.isnan(actual) ^ torch.isnan(expected)).any().item()
    if exact:
        finite = ~torch.isnan(expected)
        assert torch.equal(actual[finite], expected[finite])
    else:
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.anyplatform
@pytest.mark.parametrize("dtype", VALUES)
def test_unmasked_values_survive_a_fill(dtype):
    """A stored false mask leaves its input element in place.

    Every unmasked element has a distinct nonzero value. A graph that returned
    an empty result, the fill value, or the mask itself cannot pass.
    """
    self_ = torch.tensor([[42.0, 43.0, 44.0, 45.0, 46.0]]).to(dtype)
    mask = torch.tensor([[True, False, True, False, True]])
    out = _device(self_).masked_fill(_device(mask), 7)
    _assert_matches(out, _reference(self_, mask, 7))


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    ("self_shape", "mask_shape"),
    [
        ((3, 1, 4, 5), (1, 4, 1)),  # only self broadcasts
        ((4, 1, 4), (1, 4, 1)),  # only mask broadcasts
        ((2, 1, 3), (2, 4, 3)),  # only self broadcasts
        ((2, 3, 5), (1, 3, 1)),  # only mask broadcasts
    ],
)
def test_broadcasting_writes_the_complete_result(self_shape, mask_shape):
    self_ = torch.randn(self_shape, dtype=torch.float32) + 10.0
    mask = torch.rand(mask_shape) > 0.5
    out = _device(self_).masked_fill(_device(mask), -3.25)
    _assert_matches(out, _reference(self_, mask, -3.25))


@pytest.mark.anyplatform
def test_float_special_values_including_nan():
    self_ = torch.tensor([1.0, 2.0, 3.0, 4.0])
    for value in (-float("inf"), float("inf"), float("nan")):
        mask = torch.tensor([True, False, True, False])
        out = _device(self_).masked_fill(_device(mask), value)
        _assert_matches(out, _reference(self_, mask, value))


@pytest.mark.anyplatform
@pytest.mark.parametrize("dtype", VALUES)
def test_a_bool_mask_is_required(dtype):
    self_ = torch.arange(4).to(dtype)
    for mask_dtype in (torch.int32, torch.float32):
        mask = torch.tensor([1, 0, 1, 0], dtype=mask_dtype)
        with pytest.raises(RuntimeError, match="only supports boolean masks"):
            _device(self_).masked_fill(_device(mask), 9)


@pytest.mark.anyplatform
def test_unbroadcastable_inputs_raise_the_reference_error():
    self_ = torch.arange(3)
    mask = torch.tensor([True, False])
    with pytest.raises(RuntimeError, match="size of tensor"):
        _device(self_).masked_fill(_device(mask), 5)


@pytest.mark.anyplatform
def test_the_input_is_not_modified():
    self_ = torch.tensor([42.0, 43.0, 44.0, 45.0])
    out = _device(self_).masked_fill(
        _device(torch.tensor([1, 0, 1, 0], dtype=torch.bool)), 7.0
    )
    assert self_.tolist() == [42.0, 43.0, 44.0, 45.0]
    _assert_matches(out, torch.tensor([7.0, 43.0, 7.0, 45.0]))


@pytest.mark.anyplatform
def test_interleaved_shapes_and_scalars_reuse_the_right_plan():
    """Warm plans do not bake in the first call's operand or scalar."""
    for i in range(8):
        for self_shape, mask_shape in (
            ((3, 5), (3, 5)),
            ((1, 5), (3, 1)),
            ((3, 1), (1, 4)),
        ):
            base = torch.arange(self_shape[0] * self_shape[1], dtype=torch.float32)
            base = base.reshape(self_shape) + 100 * i
            mcount = mask_shape[0] * mask_shape[1]
            mbase = torch.arange(mcount).reshape(mask_shape) % (i + 2) == 0
            out = _device(base).masked_fill(_device(mbase), i - 2.0)
            expected = base.masked_fill(mbase, i - 2.0)
            _assert_matches(out, expected)


@pytest.mark.anyplatform
def test_queued_temporary_inputs_are_not_reused():
    """The exact allocator lifetime that the fallback violated.

    Inputs are deliberately dropped before the second fill. Their requests of
    the same shape can reuse the released blocks, while the first MPSGraph call
    may still be queued. Every result is retained until the end to remove
    Python-side liveness as an accidental source of protection.
    """
    size = 1 << 18
    masks = [(torch.arange(size) % 3 == i) for i in range(8)]
    selfs = [torch.arange(size, dtype=torch.float32) + 1000.0 * i for i in range(8)]
    outputs = []
    for self_, mask in zip(selfs, masks):
        x = _device(self_)
        m = _device(mask)
        outputs.append(x.masked_fill(m, -1.0 * len(outputs)))
    torch_fl._C._synchronize()
    for i, (out, self_, mask) in enumerate(zip(outputs, selfs, masks)):
        _assert_matches(out, self_.masked_fill(mask, -1.0 * i))


@pytest.mark.anyplatform
def test_empty_cache_does_not_free_a_block_with_a_pending_graph_event():
    """``empty_cache`` keeps a queued read alive until its event completes."""
    self_ = torch.arange(1 << 18, dtype=torch.float32) + 1000.0
    mask = torch.arange(1 << 18) % 2 == 0
    out = _device(self_).masked_fill(_device(mask), 7.0)
    torch_fl.flagos.empty_cache()
    _assert_matches(out, self_.masked_fill(mask, 7.0))


@pytest.mark.anyplatform
def test_integer_values_keep_cpu_conversion():
    for dtype in (torch.int32, torch.int64):
        self_ = torch.tensor([100, 200, 300, 400], dtype=dtype)
        mask = torch.tensor([True, False, True, False])
        out = _device(self_).masked_fill(_device(mask), 2.9)
        _assert_matches(out, self_.masked_fill(mask, 2.9))


@pytest.mark.anyplatform
def test_bool_input_uses_value_truthiness():
    self_ = torch.tensor([True, False, True, False])
    mask = torch.tensor([True, True, False, False])
    for value in (0.0, -3.0, 2.0):
        out = _device(self_).masked_fill(_device(mask), value)
        _assert_matches(out, self_.masked_fill(mask, value))


@pytest.mark.anyplatform
def test_inplace_spelling_still_matches_the_reference():
    self_ = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, False, True])
    inplace = _device(self_)
    inplace.masked_fill_(_device(mask), 6.0)
    _assert_matches(inplace, self_.masked_fill(mask, 6.0))


@pytest.mark.anyplatform
@pytest.mark.parametrize("value_dtype", VALUES)
@pytest.mark.parametrize("dtype", VALUES)
def test_tensor_value_keeps_unmasked_elements(dtype, value_dtype):
    """The tensor spelling keeps every unmasked value and dtype.

    The path is checked against the complete host result, including false-mask
    positions and the result dtype.
    """
    self_ = torch.tensor([42, 43, 44, 45], dtype=dtype)
    mask = torch.tensor([True, False, True, False])
    value = torch.tensor(8, dtype=value_dtype)
    expected = _reference(self_, mask, value)
    out = _device(self_).masked_fill(_device(mask), _device(value))
    _assert_matches(out, expected)


@pytest.mark.anyplatform
def test_tensor_value_follows_the_scalar_tensor_contract():
    self_ = torch.arange(6, dtype=torch.float32)
    mask = torch.tensor([True, False, True, False, True, False])
    for value in (
        torch.tensor(2.75),
        torch.tensor(2.0, dtype=torch.float64),
        torch.tensor(2, dtype=torch.int64),
    ):
        out = _device(self_).masked_fill(_device(mask), _device(value))
        _assert_matches(out, self_.masked_fill(mask, value))
    with pytest.raises(RuntimeError, match="0-dimensional value tensor"):
        _device(self_).masked_fill(_device(mask), _device(torch.tensor([2.0])))


@pytest.mark.anyplatform
def test_mask_and_tensor_value_broadcast_to_the_operand():
    shape = (2, 3, 4)
    self_ = torch.arange(24, dtype=torch.float32).reshape(shape) + 10
    mask = torch.tensor([True, False, True, False])
    value = torch.tensor(7.25)
    out = _device(self_).masked_fill(_device(mask), _device(value))
    _assert_matches(out, self_.masked_fill(mask, value))


@pytest.mark.anyplatform
def test_tensor_value_queued_temporary_inputs_are_not_reused():
    size = 1 << 18
    masks = [torch.arange(size) % 3 == i for i in range(8)]
    outputs = []
    for i, mask in enumerate(masks):
        values = torch.arange(size, dtype=torch.float32) + 1000.0 * i
        outputs.append(
            _device(values).masked_fill(_device(mask), _device(torch.tensor(i - 2.0)))
        )
    torch_fl._C._synchronize()
    for i, (out, mask) in enumerate(zip(outputs, masks)):
        self_ = torch.arange(size, dtype=torch.float32) + 1000.0 * i
        expected = self_.masked_fill(mask, torch.tensor(i - 2.0))
        _assert_matches(out, expected)


@pytest.mark.main_ops
def test_both_out_of_place_spellings_are_native():
    assert routed_backend_or_none("masked_fill.Scalar") == "mps"
    assert routed_backend_or_none("masked_fill.Scalar_out") is None
    assert routed_backend_or_none("masked_fill.Tensor") == "mps"
    assert routed_backend_or_none("masked_fill.Tensor_out") is None
    assert routed_backend_or_none("masked_fill_.Scalar") is None
    assert routed_backend_or_none("masked_fill_.Tensor") is None


@pytest.mark.main_ops
def test_dispatch_log_names_mps():
    env = os.environ.copy()
    env["FLAGOS_LOG"] = "dispatch"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, torch_fl; "
            "x = torch.ones(1, 1, 4, device='flagos:0'); "
            "m = torch.tensor([True, False, True, False], device='flagos:0'); "
            "x.masked_fill(m, 7.0)",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "[flagos dispatch] masked_fill.Scalar -> apple" in result.stderr


@pytest.mark.anyplatform
def test_a_large_fill_is_served_for_every_call():
    assert (
        dispatched_to_mps(
            """
import torch, torch_fl
x = torch.randn(256, 1024, device='flagos:0')
m = torch.rand(256, 1024, device='flagos:0') > 0.5
x.masked_fill(m, 1.0)
torch_fl._C._synchronize()
for _ in range(4):
    x.masked_fill(m, 1.0)
torch_fl._C._synchronize()
"""
        )
        == 5
    )
