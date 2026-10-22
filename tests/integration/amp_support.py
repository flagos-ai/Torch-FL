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

"""Shared capabilities for the cross-backend AMP contract.

Loaded as a pytest plugin, so torch must not be imported at module scope: the
integration conftest imports torch_fl after plugin loading and importing torch
first breaks the device asset preload order.
"""

from dataclasses import dataclass

import pytest

from platform_support import detect_platform


#: Device used by every AMP contract test.
AMP_DEVICE = "flagos:0"

#: Backends whose device AMP kernels are their own: the autocast policy lists
#: dispatch straight to vendor operator libraries (aclnn, topsaten, mudnn).
AMP_NATIVE_DEVICE_PLATFORMS = frozenset({"ascend", "gcu", "musa"})

#: CUDA-ABI backends whose device AMP kernels are the generated CUDA boxing
#: kernels. Every op this contract exercises is routed `cuda` by their confs --
#: `convolution`/`convolution_overrideable`, `_amp_foreach_non_finite_check_and_unscale`,
#: `_amp_update_scale` and the `_foreach_mul`/`_foreach_div` family behind
#: GradScaler all read `= cuda` in backends_cuda.conf. Boxing became an explicit
#: opt-in in #520, so on a default (FlagGems-primary) wheel there is no kernel
#: behind that route and the device cases belong on the skipping side.
AMP_BOXING_DEVICE_PLATFORMS = frozenset({"cuda", "dcu", "metax", "ppu"})


def _boxing_built() -> bool:
    """True when this wheel carries the generated CUDA boxing kernels.

    Read from the build record (``KERNELS`` in ``torch_fl/_build_config.py``)
    rather than inferred from the accelerator name: since #520 the two are not
    interchangeable, because boxing is an opt-in even on a ``cuda`` wheel.
    """
    from torch_fl import _env

    return "boxing" in _env.build_kernels()


@dataclass(frozen=True)
class AmpCapabilities:
    """Observable AMP features expected from the active backend."""

    platform: str
    device: bool
    convolution: bool
    grad_scaler: bool


def capabilities_for_platform(platform: str) -> AmpCapabilities:
    """Describe the AMP features each backend actually provides.

    The autocast policy lists are shared by every backend, so what differs is
    which device routes exist behind them. Convolution and GradScaler are
    tracked separately because a backend can provide one without the other:
    GCU needed `convolution_overrideable` added before the lower-precision
    convolution policy could run at all, while MUSA reaches the unscale and
    scale-update ops through the fallback rather than native routes.

    The CUDA-ABI backends reach those routes through the generated CUDA boxing
    kernels, so a wheel built without them -- the default since #520 -- answers
    False for all three, and the cases that need a device kernel skip rather
    than assert routing the wheel cannot perform. Reading the build record keeps
    the answer tied to the wheel instead of to the accelerator name.
    """
    device = platform in AMP_NATIVE_DEVICE_PLATFORMS or (
        platform in AMP_BOXING_DEVICE_PLATFORMS and _boxing_built()
    )
    return AmpCapabilities(
        platform=platform,
        device=device,
        convolution=device,
        grad_scaler=device,
    )


@pytest.fixture(scope="session")
def amp_capabilities():
    """AMP capabilities for the active hardware/backend."""
    return capabilities_for_platform(detect_platform())


@pytest.fixture(scope="session")
def amp_device():
    """Device string shared by the AMP contract."""
    return AMP_DEVICE


def require(capabilities: AmpCapabilities, feature: str) -> None:
    """Skip the calling test when the backend does not provide `feature`."""
    if not getattr(capabilities, feature):
        pytest.skip(f"{capabilities.platform} has no AMP {feature} routes")
