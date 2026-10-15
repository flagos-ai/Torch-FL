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

"""Stage-1 Apple Silicon (Metal / MPSGraph) unary kernels.

The exit criterion for stage 1 of the MPSGraph operator layer (see
docs/superpowers/specs/2026-10-14-apple-mps-native-backend-design.md section 7)
is that a unary op matches the CPU reference within tolerance. ``abs`` is the
first op, so this file is where that criterion is checked end to end: the op
reaches the PrivateUse1 ``mps`` kernel (a dispatch-log assertion, not just a
numerical one, so a silent cpu_fallback cannot pass), produces CPU-equal
results on every dtype the kernel spells out, and round-trips the cases the
kernel deliberately declines (bool, which MPSGraph has an element type for but
aten's ``abs`` does not accept on any backend).

Every case self-skips off an MPS build: the marker keeps CI manifests explicit
and the skipif keeps a broad ``pytest tests/integration`` collection from
touching Metal on a machine that has none.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import routed_backend_or_none
from platform_support import detect_platform


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


class TestAbsCorrectness:
    """``abs`` on the flagos device equals ``abs`` on the CPU reference."""

    @pytest.mark.parametrize("shape", [(1,), (128, 256), (4, 8, 16)])
    @pytest.mark.anyplatform
    def test_matches_cpu_for_float32(self, shape):
        a = _to_device(torch.randn(*shape) * 3.0)
        out = torch.abs(a)
        torch.testing.assert_close(out.cpu(), torch.abs(a.cpu()), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 5),  # 20-byte rows
            (4, 6),  # 24-byte rows
            (3, 7),
            (8, 3),
            (2, 3, 5),
            (4, 3, 5),
            (128, 3),
            (5, 4),  # controls: rows already a multiple of 16 bytes
            (4, 8, 16),
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_for_rows_narrower_than_a_16_byte_multiple(self, shape):
        """The shapes MPSNDArray's row pitch would silently corrupt.

        MPSNDArray rounds the stride between the rows of an N-D array up to 16
        bytes. Describing a torch-contiguous buffer with its aten shape therefore
        reads every row after the first from the wrong offset whenever the
        innermost row is not already 16-byte aligned: a (4, 5) float32 operand
        has 20-byte rows that MPSNDArray reads as 32, so its rows 1-3 came back
        shifted. The kernel describes the buffers as 1-D and reshapes inside the
        graph, and this is what holds it to that; the aligned shapes in the list
        are what show the padding was the variable rather than the shape.
        """
        host = torch.randn(*shape) * 3.0
        out = torch.abs(_to_device(host))
        torch.testing.assert_close(out.cpu(), torch.abs(host), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "dtype",
        [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_for_unaligned_integer_rows(self, dtype):
        # Rows of 5 elements: 5 bytes for the 8-bit types, 10 for int16, 20 for
        # int32 and 40 for int64. None is a multiple of 16, so each one is a
        # different element width against the same row pitch.
        host = torch.tensor([[-7, -1, 0, 3, 11]] * 3, dtype=dtype)
        out = torch.abs(_to_device(host))
        torch.testing.assert_close(out.cpu(), torch.abs(host))

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_for_every_supported_dtype(self, dtype):
        if dtype == torch.uint8:
            host = torch.tensor([0, 1, 128, 200, 255], dtype=dtype)
        else:
            host = torch.tensor([-7, -1, 0, 3, 11], dtype=dtype)
        a = _to_device(host)
        out = torch.abs(a)
        torch.testing.assert_close(out.cpu(), torch.abs(host))

    @pytest.mark.anyplatform
    def test_signed_min_is_identity_like_cpu(self):
        """``abs`` of the most-negative integer has no positive counterpart.

        MPSGraph's ``absoluteWithTensor:`` is the hardware identity here, which
        is what CPU torch computes too: ``abs(-128) == -128`` for int8. Pinning
        it keeps the MPS kernel from being "fixed" into a wrapping negate that
        would silently disagree with the reference.
        """
        host = torch.tensor([-128, -127, 0, 127], dtype=torch.int8)
        out = torch.abs(_to_device(host))
        torch.testing.assert_close(out.cpu(), torch.abs(host))
        assert out.cpu()[0].item() == -128

    @pytest.mark.parametrize("shape", [(8, 8), (8, 10)])
    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self, shape):
        """A non-contiguous operand takes the same answer as the CPU.

        MPSGraph reads dense memory, so the kernel materialises the operand
        first; this asserts that path, not just the contiguous one. The second
        shape leaves 5 columns after the stride, so the materialised operand has
        the unaligned rows too.
        """
        a = _to_device(torch.randn(*shape))[:, ::2]
        assert not a.is_contiguous()
        out = torch.abs(a)
        torch.testing.assert_close(out.cpu(), torch.abs(a.cpu()), rtol=1e-5, atol=1e-5)

    @pytest.mark.anyplatform
    def test_empty_tensor(self):
        a = _to_device(torch.empty(0, 4))
        out = torch.abs(a)
        assert out.shape == (0, 4)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_bool_raises_exactly_like_cpu(self):
        """bool is not a valid ``abs`` dtype anywhere.

        MPSGraph does have a bool element type, so the kernel could compute an
        identity. It declines instead and takes the CPU round-trip, which raises
        the same ``NotImplementedError`` as the reference -- so a bool operand
        gets an error, not a silently-wrong answer.
        """
        a = _to_device(torch.tensor([True, False, True]))
        with pytest.raises(NotImplementedError):
            torch.abs(a)


class TestScheduleAndPlanReuse:
    """The marshalling contract every MPS kernel shares (mps_common.mm).

    Two properties live below the individual kernels, so no single-op case can
    show them and a regression in either shows up as a wrong number rather than
    as an ordering assertion: the compiled graph is cached per operand signature
    and reused, and the run is scheduled rather than awaited.
    """

    @pytest.mark.anyplatform
    def test_interleaved_signatures_never_share_a_plan(self):
        """Each signature gets its own compiled graph, asked for round-robin.

        A cache key that named only the op would let the first graph built serve
        every later case -- an f32 graph reading bf16 storage, or a reshape sized
        for one shape reading a longer buffer. Rotating through the signatures
        instead of grouping them is what puts a different shape or dtype on a
        warm cache on every step.
        """
        cases = [
            ((128,), torch.float32),
            ((128,), torch.float16),
            ((128,), torch.int32),
            ((32, 128), torch.float32),
            ((32, 128), torch.float16),
            ((4, 5), torch.float32),  # 20-byte rows, against 128-byte ones
            ((512, 512), torch.float32),
        ]
        hosts = []
        for index, (shape, dtype) in enumerate(cases):
            generator = torch.Generator().manual_seed(index)
            if dtype.is_floating_point:
                host = torch.randn(*shape, generator=generator) * 3.0
            else:
                host = torch.randint(-50, 50, shape, generator=generator)
            hosts.append(host.to(dtype))
        wanted = [torch.abs(host) for host in hosts]
        on_device = [_to_device(host) for host in hosts]

        for round_ in range(3):
            for offset in range(len(cases)):
                index = (offset + round_) % len(cases)
                out = torch.abs(on_device[index])
                assert out.dtype == wanted[index].dtype
                assert out.shape == wanted[index].shape
                torch.testing.assert_close(
                    out.cpu(), wanted[index], rtol=1e-5, atol=1e-5
                )

    @pytest.mark.anyplatform
    def test_results_survive_block_reuse_and_immediate_readback(self):
        """Nothing waits for the GPU, so both ends of the call are checked here.

        An operand's block goes back to the pool as soon as the caller drops it,
        while the graph reading it has only been scheduled; and the value the
        caller reads back is produced by work the runtime has not awaited. Both
        rest on the sentinel command buffer the runtime commits behind the graph
        (see mps_common.h), so a call that returned before its work was visible
        to StreamQuery would show up here as a wrong result.
        """
        held = []
        for _ in range(64):
            src = torch.randn(1 << 13, device=DEVICE) * 3.0
            wanted = torch.abs(src.cpu())
            out = torch.abs(src)
            del src  # recycled while the graph above may still be reading it
            for _ in range(4):  # blocks the allocator is free to hand back out
                torch.abs(torch.randn(1 << 13, device=DEVICE))
            held.append((out.cpu(), wanted))  # read back without a synchronize

        for index, (got, wanted) in enumerate(held):
            torch.testing.assert_close(got, wanted, rtol=1e-5, atol=1e-5)
            assert got.shape == wanted.shape, f"round {index}"


class TestAbsDispatch:
    """``abs`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_abs_to_mps(self):
        assert routed_backend_or_none("abs") == "mps"

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """The dispatch log is the proof the kernel ran, not a fallback.

        A numeric-only check passes whether MPSGraph ran or ``abs`` reached
        ``cpu_fallback``, because both compute the same answer. The log line is
        emitted by the C++ dispatcher only on the ``mps`` slot, so it is what
        distinguishes the two. Run in a subprocess (as ``test_abs_dispatch.py``
        does) because ``LogEnabled`` latches the environment on first use.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "a = torch.randn(4, 4, device='flagos:0'); torch.abs(a)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos dispatch] abs -> mps" in result.stderr
