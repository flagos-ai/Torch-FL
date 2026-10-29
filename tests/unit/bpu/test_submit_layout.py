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

"""The buffer layout the submit-ahead path hands to hbDNNInferV2.

These are arithmetic checks, so they need no device. They exist because the
failure they guard against does not raise: an artifact whose device stride is
wider than its row gets a flat memmove, every row shifts by the padding, and
the model answers a plausible wrong image. On the ResNet-18 artifact the Y
plane is `[1, 224, 224, 1]` at `stride[1] == 256`, and `[1, 112, 112, 2]` at the
same 256, both transcribed from `hbDNNGetInputTensorProperties` on the board.
"""

from __future__ import annotations

import ctypes

import numpy as np

from torch_fl.accelerator.bpu import submit as S


def _props(dims, strides, aligned=-1, tensor_type=3):
    """A DNNTensorProperties carrying exactly the layout given."""
    p = S.DNNTensorProperties()
    p.validShape.numDimensions = len(dims)
    for i, d in enumerate(dims):
        p.validShape.dimensionSize[i] = d
    for i, s in enumerate(strides):
        p.stride[i] = s
    p.alignedByteSize = aligned
    p.tensorType = tensor_type
    return p


def _buffer(nbytes):
    mem = S.UCPSysMem()
    mem.memSize = nbytes
    mem.virAddr = ctypes.addressof(ctypes.create_string_buffer(nbytes))
    return mem


def test_dynamic_strides_resolve_innermost_out():
    # Y plane of the ResNet-18 artifact: [-1, -1, 1, 1] over [1, 224, 224, 1].
    p = _props([1, 224, 224, 1], [-1, -1, 1, 1])
    S._prepare_stride(p)
    # 224 * 1 = 224, padded to 256; then dim[1] * 256 = 57344.
    assert [p.stride[i] for i in range(4)] == [57344, 256, 1, 1]


def test_dynamic_strides_on_a_two_channel_plane():
    # UV plane: [1, 112, 112, 2] over [-1, -1, 2, 1].
    p = _props([1, 112, 112, 2], [-1, -1, 2, 1])
    S._prepare_stride(p)
    assert [p.stride[i] for i in range(4)] == [28672, 256, 2, 1]


def test_dynamic_last_dimension_is_rejected():
    p = _props([1, 8, -1], [-1, -1, -1])
    try:
        S._prepare_stride(p)
    except RuntimeError:
        return
    raise AssertionError("a dynamic innermost stride must be rejected")


def test_buffer_size_prefers_aligned_byte_size():
    p = _props([1, 1000], [4000, 4], aligned=4096)
    assert S._buffer_size(p) == 4096
    # Without an aligned size, the outermost stride times its dimension is the
    # extent, which is what HB_HBMRuntime::PrepareInputTensorData does.
    q = _props([1, 224, 224, 1], [57344, 256, 1, 1], aligned=-1)
    assert S._buffer_size(q) == 57344


def test_fill_places_rows_at_their_pitch():
    """A padded artifact must not get a flat memmove of the numpy array."""
    dims, strides = [1, 4, 4, 1], [1024, 64, 1, 1]
    p = _props(dims, strides, aligned=1024)
    arr = np.arange(16, dtype=np.uint8).reshape(dims)
    mem = _buffer(1024)
    S.BPUSubmitter._fill(arr, p, mem)

    # Rows are the dim-1 axis, each at the stride[1] pitch of 64 bytes. A flat
    # memmove would put row r at 4*r instead, so this pins the pitch.
    # The padding itself is not asserted on: hbUCPMallocCached does not zero a
    # buffer, so what sits past a row is whatever was there before.
    raw = np.frombuffer(
        (ctypes.c_char * 1024).from_address(mem.virAddr), dtype=np.uint8
    )
    for row in range(4):
        start = row * 64
        assert np.array_equal(raw[start : start + 4], arr[0, row, :, 0]), row


def test_fill_is_a_plain_memmove_when_pitch_matches():
    dims, strides = [1, 2, 8], [16, 8, 1]
    p = _props(dims, strides, aligned=16)
    arr = np.arange(16, dtype=np.uint8).reshape(dims)
    mem = _buffer(16)
    S.BPUSubmitter._fill(arr, p, mem)
    raw = np.frombuffer(
        (ctypes.c_char * 16).from_address(mem.virAddr), dtype=np.uint8
    ).reshape(dims)
    assert np.array_equal(raw, arr)


def test_fill_rejects_a_shape_that_does_not_match_the_artifact():
    p = _props([1, 4, 4, 1], [1024, 64, 1, 1], aligned=1024)
    mem = _buffer(1024)
    try:
        S.BPUSubmitter._fill(np.zeros((1, 8, 8, 3), np.uint8), p, mem)
    except RuntimeError:
        return
    raise AssertionError("a mismatched input shape must be rejected")


def test_sched_param_matches_the_board_layout():
    """hbUCPSchedParam is {i32, i64, u64, u32} padded to 32 bytes."""
    p = S.UCPSchedParam(priority=0, customId=0, backend=S.BPU_CORE_ANY, deviceId=0)
    assert ctypes.sizeof(S.UCPSchedParam) == 32
    assert p.backend == 0x80
    assert S.UCPSchedParam.backend.offset == 16


def test_dnn_tensor_layout_matches_the_board():
    """Offsets confirmed against live arguments captured from libdnn."""
    assert S.DNNTensor.properties.offset == 24
    assert S.DNNTensorProperties.tensorType.offset == 44
    assert S.DNNTensorProperties.alignedByteSize.offset == 88
