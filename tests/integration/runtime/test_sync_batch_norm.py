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

"""Integration tests for stream-capture query and SyncBatchNorm on FlagOS."""

import pytest
import torch_fl  # noqa: F401

# isort: split
import torch
import torch.nn as nn


@pytest.mark.skipif(
    not torch.flagos.is_available(), reason="requires a FlagOS accelerator"
)
def test_stream_capture_query_returns_false_outside_capture():
    assert torch.flagos.is_current_stream_capturing() is False
    assert torch.cuda.is_current_stream_capturing() is False
    assert torch.cuda.graphs.is_current_stream_capturing() is False


@pytest.mark.skipif(
    not torch.flagos.is_available(), reason="requires a FlagOS accelerator"
)
@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 2e-2, 2e-2),
    ],
)
def test_sync_batch_norm_aten_primitives_match_cpu_batch_norm(dtype, atol, rtol):
    """Verify the six PrivateUse1 SyncBatchNorm ATen ops against CPU BatchNorm2d."""
    torch.manual_seed(20261125)
    c = 8
    # Simulate two unequal shards: rank0 has N=3, rank1 has N=5 (global N=8).
    x_global_cpu = torch.randn(8, c, 4, 4, dtype=torch.float32)
    dy_global_cpu = torch.randn_like(x_global_cpu)

    bn_cpu = nn.BatchNorm2d(c, eps=1e-5, momentum=0.1).float()
    nn.init.uniform_(bn_cpu.weight, 0.5, 1.5)
    nn.init.uniform_(bn_cpu.bias, -0.5, 0.5)
    bn_cpu.running_mean.copy_(torch.randn(c) * 0.1)
    bn_cpu.running_var.copy_(torch.rand(c) + 0.5)

    rm_init = bn_cpu.running_mean.clone()
    rv_init = bn_cpu.running_var.clone()

    x_ref = x_global_cpu.to(dtype).float().detach().requires_grad_(True)
    dy_ref = dy_global_cpu.to(dtype).float()
    y_ref = bn_cpu(x_ref)
    y_ref.backward(dy_ref)

    # Execute the exact two-shard SyncBatchNorm operator sequence on flagos:0.
    dev = torch.device("flagos:0")
    x0 = x_global_cpu[:3].to(device=dev, dtype=dtype).detach().requires_grad_(True)
    assert torch.flagos.is_current_stream_capturing() is False
    assert torch.cuda.is_current_stream_capturing() is False
    assert torch.cuda.graphs.is_current_stream_capturing() is False
    x1 = x_global_cpu[3:].to(device=dev, dtype=dtype).detach().requires_grad_(True)
    dy0 = dy_global_cpu[:3].to(device=dev, dtype=dtype)
    dy1 = dy_global_cpu[3:].to(device=dev, dtype=dtype)
    w = bn_cpu.weight.detach().to(device=dev)
    b = bn_cpu.bias.detach().to(device=dev)
    rm = rm_init.to(device=dev)
    rv = rv_init.to(device=dev)

    m0, v0 = torch.batch_norm_stats(x0, 1e-5)
    m1, v1 = torch.batch_norm_stats(x1, 1e-5)
    mean_all = torch.stack([m0, m1], dim=0)
    invstd_all = torch.stack([v0, v1], dim=0)
    counts = torch.tensor(
        [x0.numel() // c, x1.numel() // c], device=dev, dtype=rm.dtype
    )

    g_mean, g_invstd = torch.batch_norm_gather_stats_with_counts(
        x0, mean_all, invstd_all, rm, rv, 0.1, 1e-5, counts
    )
    y0 = torch.batch_norm_elemt(x0, w, b, g_mean, g_invstd, 1e-5)
    y1 = torch.batch_norm_elemt(x1, w, b, g_mean, g_invstd, 1e-5)
    y_got = torch.cat([y0, y1], dim=0).float().cpu()

    torch.testing.assert_close(y_got, y_ref.detach(), atol=atol, rtol=rtol)
    torch.testing.assert_close(rm.cpu(), bn_cpu.running_mean, atol=atol, rtol=rtol)
    torch.testing.assert_close(rv.cpu(), bn_cpu.running_var, atol=atol, rtol=rtol)

    s_dy0, s_dy_xmu0, gw0, gb0 = torch.batch_norm_backward_reduce(
        dy0, x0, g_mean, g_invstd, w, True, True, True
    )
    s_dy1, s_dy_xmu1, gw1, gb1 = torch.batch_norm_backward_reduce(
        dy1, x1, g_mean, g_invstd, w, True, True, True
    )
    s_dy = s_dy0 + s_dy1
    s_dy_xmu = s_dy_xmu0 + s_dy_xmu1
    count_i32 = counts.to(torch.int32)
    dx0 = torch.batch_norm_backward_elemt(
        dy0, x0, g_mean, g_invstd, w, s_dy, s_dy_xmu, count_i32
    )
    dx1 = torch.batch_norm_backward_elemt(
        dy1, x1, g_mean, g_invstd, w, s_dy, s_dy_xmu, count_i32
    )
    dx_got = torch.cat([dx0, dx1], dim=0).float().cpu()
    gw_got = (gw0 + gw1).float().cpu()
    gb_got = (gb0 + gb1).float().cpu()

    torch.testing.assert_close(dx_got, x_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(gw_got, bn_cpu.weight.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(gb_got, bn_cpu.bias.grad, atol=atol, rtol=rtol)
