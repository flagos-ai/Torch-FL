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

#: Backends with device AMP kernels behind the shared AutocastPrivateUse1
#: policy lists, either natively or through CUDA boxing. A backend outside this
#: set still runs the device-independent autocast API and policy-state cases.
AMP_DEVICE_PLATFORMS = frozenset(
    {"ascend", "cuda", "dcu", "gcu", "metax", "musa", "ppu"}
)


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
    """
    device = platform in AMP_DEVICE_PLATFORMS
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
