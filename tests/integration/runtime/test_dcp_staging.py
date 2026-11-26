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

"""End-to-end DCP overlapping save/load verification on FlagOS devices."""

import pytest
import torch_fl  # noqa: F401

# isort: split
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemWriter


@pytest.mark.skipif(
    not torch.flagos.is_available(), reason="requires a FlagOS accelerator"
)
def test_dcp_save_and_load_overlapping_default_and_side_stream(tmp_path):
    torch.manual_seed(20261125)
    cpu_ref = {
        "linear_w": torch.randn(
            512, 512, dtype=torch.float32
        ),  # 1 MiB -> drain + refill
        "linear_b": torch.randn(512, dtype=torch.float32),
        "bf16_buf": torch.randn(256, 256, dtype=torch.bfloat16),
        "step": torch.tensor([42], dtype=torch.int64),
    }
    dev_state = {k: v.to("flagos:0") for k, v in cpu_ref.items()}

    # 1. Default stream with default overlapping FileSystemWriter.
    default_before = torch.flagos.current_stream(0).stream_id
    ckpt_default = tmp_path / "ckpt_default"
    dcp.save(dev_state, checkpoint_id=str(ckpt_default))
    assert torch.flagos.current_stream(0).stream_id == default_before

    loaded_default = {k: torch.zeros_like(v) for k, v in dev_state.items()}
    dcp.load(loaded_default, checkpoint_id=str(ckpt_default))
    for k, ref in cpu_ref.items():
        got = loaded_default[k].cpu()
        assert torch.equal(got, ref), f"mismatch on {k} after default-stream DCP load"

    # 2. Non-default FlagOS stream: verify overlapping staging preserves active stream.
    side_stream = torch.flagos.Stream()
    ckpt_side = tmp_path / "ckpt_side"
    with torch.flagos.stream(side_stream):
        side_id = torch.flagos.current_stream(0).stream_id
        computed = {
            k: (v + 1 if v.is_floating_point() else v + 1) for k, v in dev_state.items()
        }
        writer = FileSystemWriter(str(ckpt_side), thread_count=1)
        dcp.save(computed, storage_writer=writer)
        assert torch.flagos.current_stream(0).stream_id == side_id
    assert torch.flagos.current_stream(0).stream_id == default_before

    loaded_side = {k: torch.zeros_like(v) for k, v in dev_state.items()}
    dcp.load(loaded_side, checkpoint_id=str(ckpt_side))
    for k, ref in cpu_ref.items():
        expected = ref + 1
        assert torch.equal(loaded_side[k].cpu(), expected), (
            f"mismatch on {k} on side stream"
        )
