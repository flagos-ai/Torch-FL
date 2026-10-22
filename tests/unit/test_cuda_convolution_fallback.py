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

"""The CUDA-ABI convolution fallback is registered on non-boxing builds.

``convolution_overrideable`` is a CompositeExplicitAutograd op whose fallback
raises. A CPU fallback cannot move that call because it redispatches the same
op. CUDA and DCU therefore compile a wrapper that crosses to CPU before calling
``aten::convolution``. MetaX uses its boxing build, so only the three
non-boxing CUDA-ABI platforms register the forward fallback. PPU routes forward
through FlagGems and needs this wrapper only for backward.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "csrc" / "aten" / "register.cc"
CONFIGS = ROOT / "torch_fl" / "configs"


def _convolution_routes(platform: str) -> set[str]:
    text = (CONFIGS / f"backends_{platform}.conf").read_text(encoding="utf-8")
    routes = set()
    for line in text.splitlines():
        if line.startswith(("convolution_overrideable =", "convolution_overrideable.")):
            routes.add(line.split("=", 1)[1].strip().split()[0])
    return routes


def test_fallback_is_limited_by_platform_and_route():
    source = REGISTER.read_text(encoding="utf-8")
    registrations = source[source.index("#elif defined(FLAGOS_ACCEL_CUDA)") :]
    assert registrations.startswith(
        "#elif defined(FLAGOS_ACCEL_CUDA) || defined(FLAGOS_ACCEL_DCU)\n"
        "  // The native convolution needs boxing, which these two builds do not have.\n"
        '  m.impl("convolution_overrideable", '
        "FlagosWrapperConvolutionOverrideable);"
    )
    assert "#elif defined(FLAGOS_ACCEL_PPU)" in registrations
    assert "defined(FLAGOS_ACCEL_METAX)" not in registrations
    assert "flaggems" not in _convolution_routes("cuda")
    assert "flaggems" not in _convolution_routes("dcu")
    assert "flaggems" in _convolution_routes("ppu")


def test_fallback_forward_does_not_shadow_a_flag_gems_route():
    assert "flaggems" not in _convolution_routes("cuda")
    assert "flaggems" not in _convolution_routes("dcu")
    assert _convolution_routes("ppu") == {"flaggems"}
