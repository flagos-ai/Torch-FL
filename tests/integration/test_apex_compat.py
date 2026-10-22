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

"""Integration tests for optional Apex multi-tensor compatibility on real hardware."""

import pytest

torch = pytest.importorskip("torch")
torch_fl = pytest.importorskip("torch_fl")
apex = pytest.importorskip("apex")
amp_C = pytest.importorskip("amp_C")

from apex.multi_tensor_apply import multi_tensor_applier  # noqa: E402
from apex.optimizers import FusedAdam, FusedLAMB, FusedSGD  # noqa: E402


@pytest.mark.anyplatform
def test_fused_adam_step_changes_parameters():
    """FusedAdam completes a step and changes flagos parameters."""
    torch.manual_seed(42)
    param = torch.nn.Parameter(torch.randn(8, 8, device="flagos:0"))
    opt = FusedAdam([param], lr=0.1)
    before = param.detach().clone()
    opt.zero_grad()
    (param * param).sum().backward()
    opt.step()
    assert not torch.allclose(before, param.detach())
    assert torch.isfinite(param.detach()).all()


@pytest.mark.anyplatform
def test_fused_sgd_step_changes_parameters():
    """FusedSGD completes a step without CUDA device check failure."""
    torch.manual_seed(43)
    param = torch.nn.Parameter(torch.randn(8, 8, device="flagos:0"))
    opt = FusedSGD([param], lr=0.1, momentum=0.9)
    before = param.detach().clone()
    opt.zero_grad()
    (param * param).sum().backward()
    opt.step()
    assert not torch.allclose(before, param.detach())
    assert torch.isfinite(param.detach()).all()


@pytest.mark.anyplatform
def test_fused_lamb_step_changes_parameters():
    """FusedLAMB completes a step without CUDA device check failure."""
    torch.manual_seed(44)
    param = torch.nn.Parameter(torch.randn(8, 8, device="flagos:0"))
    opt = FusedLAMB([param], lr=0.1)
    before = param.detach().clone()
    opt.zero_grad()
    (param * param).sum().backward()
    opt.step()
    assert not torch.allclose(before, param.detach())
    assert torch.isfinite(param.detach()).all()


@pytest.mark.anyplatform
def test_fused_adam_cpu_parity():
    """FusedAdam multi-step numerical parity against CPU AdamW."""
    torch.manual_seed(100)
    w = torch.randn(16, 16)
    pf = torch.nn.Parameter(w.clone().to("flagos:0"))
    pc = torch.nn.Parameter(w.clone())
    of = FusedAdam([pf], lr=1e-2, adam_w_mode=True, weight_decay=0.0)
    oc = torch.optim.AdamW([pc], lr=1e-2, weight_decay=0.0)
    for _ in range(3):
        for p_, o_ in ((pf, of), (pc, oc)):
            o_.zero_grad()
            (p_ * p_).sum().backward()
            o_.step()
    diff = (pf.detach().cpu() - pc.detach()).abs().max().item()
    assert diff < 1e-6


@pytest.mark.anyplatform
def test_optimizer_state_device_and_type():
    """FusedAdam state tensors remain on flagos device."""
    torch.manual_seed(101)
    param = torch.nn.Parameter(torch.randn(4, 4, device="flagos:0"))
    opt = FusedAdam([param], lr=0.01)
    opt.zero_grad()
    (param * param).sum().backward()
    opt.step()
    state = opt.state[param]
    assert state["exp_avg"].device.type == "flagos"
    assert state["exp_avg_sq"].device.type == "flagos"
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32


@pytest.mark.anyplatform
def test_l2norm_tuple_output_reuse():
    """multi_tensor_l2norm returns flagos tensors that can be reused as inputs."""
    buf = torch.zeros(1, dtype=torch.int, device="flagos:0")
    ts = [
        torch.arange(1.0, 5.0, device="flagos:0"),
        torch.arange(1.0, 3.0, device="flagos:0"),
    ]
    norm, per_tensor = multi_tensor_applier(amp_C.multi_tensor_l2norm, buf, [ts], True)
    assert norm.device.type == "flagos"
    assert per_tensor.device.type == "flagos"
    ref = torch.cat([t.cpu().flatten() for t in ts]).norm().item()
    assert abs(norm.item() - ref) < 1e-5
    reused = (norm * 2).item()
    assert abs(reused - 2 * ref) < 1e-5


@pytest.mark.anyplatform
def test_cpu_cuda_invocation_pass_through():
    """Apex calls on CPU/CUDA tensors without flagos tensors remain unchanged."""
    buf = torch.zeros(1, dtype=torch.int, device="cuda:0")
    ts = [torch.arange(1.0, 5.0, device="cuda:0")]
    norm, per_tensor = multi_tensor_applier(amp_C.multi_tensor_l2norm, buf, [ts], True)
    assert norm.device.type == "cuda"
    assert per_tensor.device.type == "cuda"
