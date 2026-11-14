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

"""Native eager dispatch must preserve values, guards, streams and storage."""

import json
import sys
from pathlib import Path

import pytest
import torch
import torch_fl
from torch_fl import host_compiler

pytestmark = [
    pytest.mark.anyplatform,
    pytest.mark.nvidia,
    pytest.mark.host_compiler,
]


@pytest.fixture(scope="module")
def compiled_mul(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-host")
    host_compiler.enable(directory)
    with torch.no_grad():
        x = torch.randn(7, 128).to("flagos:0")
        torch.mul(x, x)
    manifests = list(Path(directory).glob("*/artifact.json"))
    assert manifests, [p.read_text() for p in Path(directory).glob("*/failure.txt")]
    host_compiler._mode = "load"
    yield json.loads(manifests[0].read_text())
    host_compiler.disable()


def test_parameterized_eager_dispatch_and_live_outputs(compiled_mul):
    before = torch.ops.flagos.host_program_stats()[2]
    live = []
    with torch.no_grad():
        for shape in [(7, 128), (19, 128), (11, 256), (31, 96)]:
            cpu = torch.randn(shape)
            x = cpu.to("flagos:0")
            result = torch.mul(x, x)
            live.append((result, cpu * cpu))
        for result, expected in live:
            torch.testing.assert_close(result.cpu(), expected)
    assert torch.ops.flagos.host_program_stats()[2] - before == 4


def test_native_hit_does_not_enter_flaggems_or_triton(compiled_mul):
    calls = []
    with torch.no_grad():
        x = torch.randn(7, 128).to("flagos:0")

        def profile(frame, event, arg):
            if event == "call" and any(
                p in frame.f_code.co_filename for p in ("/flag_gems/", "/triton/")
            ):
                calls.append(frame.f_code.co_name)

        sys.setprofile(profile)
        try:
            result = torch.mul(x, x)
        finally:
            sys.setprofile(None)
        torch.testing.assert_close(result.cpu(), x.cpu().square())
    assert calls == []


def test_guard_rejects_dtype_layout_and_arity_before_entry(compiled_mul):
    key = compiled_mul["key"]
    with torch.no_grad():
        x = torch.randn(7, 128).to("flagos:0")
        for inputs in ([x], [x.half(), x.half()], [x.t(), x.t()], [x[:0], x[:0]]):
            with pytest.raises(RuntimeError, match="outside compiled guard domain"):
                torch.ops.flagos.host_program_run(key, inputs)


def test_current_stream_is_resolved_per_call(compiled_mul):
    with torch.no_grad():
        x = torch.randn(7, 128).to("flagos:0")
        stream = torch_fl.flagos.Stream()
        stream.wait_stream(torch_fl.flagos.current_stream())
        before = torch.ops.flagos.host_program_stats()[2]
        with torch_fl.flagos.stream(stream):
            first = torch.mul(x, x)
            second = torch.mul(first, first)
        stream.synchronize()
        torch.testing.assert_close(second.cpu(), x.cpu().square().square())
    assert torch.ops.flagos.host_program_stats()[2] - before == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sparse", [False, True])
def test_generated_embedding_forward_contract(dtype, sparse):
    weight = torch.randn(127, 64, dtype=dtype).t().contiguous().t()
    indices = torch.randint(0, 127, (3, 7))
    with torch.no_grad():
        result = torch.nn.functional.embedding(
            indices.to("flagos:0"),
            weight.to("flagos:0"),
            padding_idx=0,
            scale_grad_by_freq=True,
            sparse=sparse,
        )
        expected = torch.nn.functional.embedding(
            indices, weight, padding_idx=0, scale_grad_by_freq=True, sparse=sparse
        )
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


def test_device_copy_preserves_offsets_and_unwritten_storage():
    with torch.no_grad():
        cpu = torch.randn(6, 8).bfloat16()
        source = cpu.to("flagos:0").t()
        dst = torch.full((10, 10), -3.0, device="flagos:0")
        view = dst[1:9, 2:8]
        view.copy_(source)
        expected = torch.full((10, 10), -3.0)
        expected[1:9, 2:8] = cpu.t().float()
        torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)


def test_device_copy_rejects_overlapping_destination():
    with torch.no_grad():
        backing = torch.zeros(1, 3, device="flagos:0")
        source = torch.ones(5, 3, device="flagos:0")
        with pytest.raises(RuntimeError, match="more than one element"):
            backing.expand(5, 3).copy_(source)
        torch.testing.assert_close(backing.cpu(), torch.zeros(1, 3))


def test_native_slice_returns_a_fresh_view(compiled_mul):
    host_compiler._mode = "compile"
    try:
        with torch.no_grad():
            x = torch.randn(7, 128).to("flagos:0")
            x[:]
            before = torch.ops.flagos.host_program_stats()[2]
            view = x[:]
            assert torch.ops.flagos.host_program_stats()[2] == before + 1
            assert view._cdata != x._cdata
            assert torch._C._is_alias_of(view, x)
            assert view.size() == x.size()
            assert view.stride() == x.stride()
            assert view.storage_offset() == x.storage_offset()
            host_compiler._mode = "load"
            x.add_(1)
            torch.testing.assert_close(view.cpu(), x.cpu())
            assert view._version == x._version
    finally:
        host_compiler._mode = "load"
