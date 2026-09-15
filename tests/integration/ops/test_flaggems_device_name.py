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

"""FlagGems device identity on the generic NVIDIA path.

FlagGems decides per op between running its own Triton kernel and handing the
call back to ATen by comparing the device type of its inputs against the device
string its vendor backend declares::

    device = _select_device(a, b)
    if device.type != _DEVICE_NAME:
        return torch.ops.aten.mul.Tensor.redispatch(_FALLBACK_KEYSET, a, b)

Its nvidia descriptor names that device ``cuda``, while torch_fl registers the
accelerator as ``flagos``. The two never compare equal, so a module that makes
that comparison took the fallback branch for every flagos input rather than only
for the ones it means to exclude, and its Triton kernel never ran. Eleven of the
``flaggems`` routes in ``backends_cuda.conf`` land on such a module.

The fallback is also what made CUDA CI fail: it redispatches to the ``Tensor``
overload, which cannot take the Python scalar a wrapped number is now handed
over as, so ``mask * 1.3333333333333333`` raised "Expected a value of type
'Tensor' for argument 'other' but instead found type 'float'".

The invariant these tests pin is the one the fix establishes: on a box whose
FlagGems resolved to the nvidia vendor, FlagGems' device name equals the name
torch_fl registered, and its entry points accept a flagos tensor. Every test
carries ``main_ops`` because CI's CUDA operator jobs select on it, and the
alignment applies whenever FlagGems is importable -- the ``backends_cuda.conf``
routing does not wait for ``FLAGOS_USE_FLAGGEMS`` -- so no backend marker is
used. Vendors other than nvidia skip internally, which keeps the file selectable
in the other platforms' runs.

Usage:
    pytest tests/integration/ops/test_flaggems_device_name.py -v
"""

import importlib

import pytest

import torch
import torch_fl  # noqa: F401

DEVICE = "flagos:0"
SEED = 1234


def _nvidia_device_detector():
    """FlagGems' DeviceDetector, or a skip when FlagGems/nvidia is not in play."""
    flag_gems = pytest.importorskip("flag_gems")
    try:
        from flag_gems.runtime.backend.device_finder import DeviceDetector
    except ImportError:
        from flag_gems.runtime.backend.device import DeviceDetector

    detector = DeviceDetector()
    if detector.vendor_name != "nvidia":
        pytest.skip(f"FlagGems resolved the {detector.vendor_name} vendor, not nvidia")
    return flag_gems, detector


def _registered_device_name():
    """The device name torch_fl registered with PyTorch, after device init."""
    torch_fl.flagos.init()
    return torch._C._get_privateuse1_backend_name()


def _flag_gems_mul():
    """FlagGems' ``mul`` entry point.

    ``flag_gems.ops`` re-exports each op *callable* under its submodule's own
    name, so this is normally the function; fall back to the module attribute if
    a FlagGems build stops shadowing the module.
    """
    entry = importlib.import_module("flag_gems.ops.mul")
    return entry if callable(entry) else entry.mul


class TestFlaggemsDeviceName:
    """The name FlagGems uses must be the one torch_fl registered."""

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_detector_name_matches_registered_backend(self):
        flag_gems, detector = _nvidia_device_detector()
        registered = _registered_device_name()
        assert detector.name == registered
        assert flag_gems.device == registered

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_op_module_copies_are_realigned(self):
        """The modules that capture the name at import time must be fixed too.

        ``DeviceDetector`` is a singleton, so correcting it covers every later
        reader, but the op modules do ``device = device.name`` at module scope
        (``ops/mul.py`` uses the ``_DEVICE_NAME`` spelling) and importing any part
        of ``flag_gems.runtime`` eagerly imports them. Those already hold the old
        literal by the time the alignment can run.
        """
        _nvidia_device_detector()
        registered = _registered_device_name()

        # Imported by path: flag_gems.ops re-exports each op *callable* under the
        # submodule's own name, shadowing the module.
        cumsum_mod = importlib.import_module("flag_gems.ops.cumsum")
        mul_mod = importlib.import_module("flag_gems.ops.mul")

        assert cumsum_mod.device == registered
        assert mul_mod._DEVICE_NAME == registered

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_flag_gems_mul_accepts_a_python_scalar(self):
        """The guard's fallback cannot take the scalar a wrapped number becomes.

        Called on the entry point rather than through ``x * 1.333`` because the
        routed call only reaches the guard with a Python scalar on a build that
        carries ``TensorToPython``'s wrapped-number handling; this spelling
        reproduces the CUDA CI failure on every build. Before the alignment it
        raised ``RuntimeError: aten::mul() Expected a value of type 'Tensor' for
        argument 'other' but instead found type 'float'.``
        """
        _nvidia_device_detector()
        torch.manual_seed(SEED)
        value = torch.randn((64, 64), device=DEVICE)
        reference = value.to("cpu") * 1.3333333333333333
        result = _flag_gems_mul()(value, 1.3333333333333333)
        assert torch.allclose(result.to("cpu"), reference)

    @pytest.mark.anyplatform
    @pytest.mark.main_ops
    def test_python_float_multiply_matches_cpu(self):
        """The same case on the routed path, as a user writes it.

        Python-float multiplication always dispatches to ``aten.mul.Tensor``
        with a Python ``float`` as ``other`` -- never to ``mul.Scalar`` -- so it
        is the routed overload, and the one the CUDA CI failure came in on.
        """
        _nvidia_device_detector()
        torch.manual_seed(SEED)
        value = torch.randn((64, 64), device=DEVICE)
        reference = value.to("cpu") * 1.3333333333333333
        assert torch.allclose((value * 1.3333333333333333).to("cpu"), reference)
