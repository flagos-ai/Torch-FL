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

"""mul runs FlagGems' kernel rather than the host fallback.

`aten::mul.Tensor` used to be held on the CUDA route. The hold was for a guard
in flag_gems' ``mul_broadcast_func`` that compares ``device.type`` against
flag_gems' own ``_DEVICE_NAME`` and redispaches into a fallback when they differ:
with ``_DEVICE_NAME == "cuda"`` a flagos tensor failed that comparison, so
routing the op to FlagGems meant routing it into the fallback.

On this stack the constant reads ``"flagos"`` -- the same string a flagos tensor
reports -- so the guard does not fire and the kernel runs. That is what these
cases pin, because the hold is one constant away from being needed again:

* if ``_DEVICE_NAME`` stops matching the device name, the first case fails;
* if the route is lost from the conf, the second case fails.

The first case is skipped rather than failed when the two names disagree, since
that is a statement about the installed flag_gems and not about this repository;
the second case is not skipped on a CUDA platform with a flagos device.
"""

import pytest
import torch

from torch_fl._platform import detect_platform

if detect_platform() != "cuda":
    pytest.skip("CUDA FlagGems route regression", allow_module_level=True)

DEVICE = "flagos:0"

pytest.importorskip("flag_gems", reason="flag_gems is not installed")


def _flaggems_device_name():
    """The device name flag_gems' guards compare against."""
    import sys

    for mod_name in ("flag_gems.ops.mul", "flag_gems.utils", "flag_gems"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "_DEVICE_NAME"):
            return mod._DEVICE_NAME
    return None


def _device_available():
    try:
        return torch.accelerator.is_available()
    except Exception:
        return False


pytestmark = [
    pytest.mark.flaggems,
    pytest.mark.skipif(not _device_available(), reason="no flagos device"),
]


def test_flaggems_mul_guard_matches_the_device_name():
    """The guard's premise: flag_gems' _DEVICE_NAME equals what flagos reports.

    When this holds, ``mul_broadcast_func`` takes its kernel branch on a flagos
    tensor. When it does not, the op redispatches into the fallback and holding
    ``mul.Tensor`` off the FlagGems route is correct again -- see the comment on
    ``flaggems_recursive_fallback`` in scripts/codegen/codegen_ops.py.
    """
    name = _flaggems_device_name()
    if name is None:
        pytest.skip("installed flag_gems has no module-level _DEVICE_NAME")
    actual = torch.empty(1, device=DEVICE).device.type
    if name != actual:
        pytest.skip(
            f"flag_gems _DEVICE_NAME={name!r} does not match device type "
            f"{actual!r}; mul.Tensor must be held off the FlagGems route"
        )
    assert name == actual


def test_mul_does_not_cross_the_host_boundary():
    """The routed op must compute on device.

    A cpu_fallback moves both operands and the result across the bus; the
    FlagGems kernel performs no copies. Same discriminator the addmm case uses,
    and the reason this is asserted as a count of crossings rather than of
    kernel time -- the latter needs a timing harness, the former is a fact about
    which implementation ran.
    """
    a = torch.randn(640, 1024, dtype=torch.bfloat16, device=DEVICE)
    b = torch.randn(640, 1024, dtype=torch.bfloat16, device=DEVICE)
    for _ in range(3):  # warm any lazy init out of the measured window
        torch.mul(a, b)
    torch.cuda.synchronize()

    crossings = 0
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.mul(a, b)
        torch.cuda.synchronize()
    for evt in prof.events():
        if "Memcpy" in evt.name and ("DtoH" in evt.name or "HtoD" in evt.name):
            crossings += 1
    assert crossings == 0, (
        f"torch.mul crossed the host boundary {crossings} times; "
        "aten::mul.Tensor is on the cpu_fallback rather than the FlagGems route"
    )


@pytest.mark.parametrize(
    "shape_a,shape_b",
    [
        ((640, 1024), (640, 1024)),  # same shape, the model's case
        ((640, 1024), (1024,)),  # broadcast row bias
        ((640, 1024), (1,)),  # scalar-ish broadcast
    ],
)
def test_mul_matches_cpu(shape_a, shape_b):
    """Whatever the route, the values must be ATen's."""
    torch.manual_seed(0)
    a = torch.randn(*shape_a, dtype=torch.bfloat16)
    b = torch.randn(*shape_b, dtype=torch.bfloat16)
    want = torch.mul(a, b)
    got = torch.mul(a.to(DEVICE), b.to(DEVICE)).cpu()
    assert got.shape == want.shape
    scale = float(want.abs().max()) + 1e-9
    diff = float((got.float() - want.float()).abs().max())
    assert diff / scale < 2e-2, f"{shape_a} x {shape_b}: rel {diff / scale:.3e}"


def test_mul_inplace_with_a_scalar_stays_correct():
    """`Tensor.mul_(float)` dispatches to aten::mul_.Tensor, not mul.Tensor.

    It is held in flaggems_runtime_broken for an independent reason (its
    fallback branch has no kernel to land on), so this case exists to show the
    mul.Tensor change did not drag it into the FlagGems route by accident.
    """
    a = torch.randn(64, 128, dtype=torch.bfloat16)
    want = a * 0.5
    got = a.to(DEVICE).clone()
    got.mul_(0.5)
    got = got.cpu()
    scale = float(want.abs().max()) + 1e-9
    diff = float((got.float() - want.float()).abs().max())
    assert diff / scale < 2e-2
