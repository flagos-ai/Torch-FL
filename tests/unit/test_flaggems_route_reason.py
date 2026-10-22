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

"""Check shared routing expectations against a wheel's registration lists."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/integration/flaggems"))

import flaggems_caps  # noqa: E402

_expected_route = flaggems_caps._expected_route


@pytest.fixture(autouse=True)
def _registration_source(monkeypatch):
    monkeypatch.setenv("TORCH_FL_SOURCE_ROOT", str(ROOT))


def test_flag_gems_only_cuda_wheel_falls_back_for_an_unregistered_native_route():
    """No generated CUDA kernel means an unregistered `cuda` route is a fallback.

    On the H100 FlagGems-only wheel, `log` is `cuda` in the conf. The generated
    FlagGems registration has no aten::log entry, and the measured call prints
    `[flagos cpu_fallback] aten::log`, so a dispatch to CUDA would be a false
    assertion.
    """
    assert _expected_route("cuda", "log", "cuda", frozenset({"flaggems"})) == (
        None,
        "conf_route_unregistered",
    )


def test_flag_gems_only_cuda_wheel_keeps_registered_native_routes():
    """The generated FlagGems list claims a few `cuda` routes on PrivateUse1."""
    assert _expected_route(
        "cuda", "convolution_overrideable", "cuda", frozenset({"flaggems"})
    ) == ("cuda", "routed")


def test_cuda_boxing_wheel_claims_the_generated_native_routes():
    assert _expected_route("cuda", "log", "cuda", frozenset({"boxing"})) == (
        "cuda",
        "routed",
    )


def test_missing_registration_source_is_a_failure(monkeypatch):
    """An unreadable source cannot pass for a complete native registration."""
    monkeypatch.delenv("TORCH_FL_SOURCE_ROOT")
    with pytest.raises(RuntimeError, match="TORCH_FL_SOURCE_ROOT"):
        _expected_route("cuda", "log", "cuda", frozenset({"flaggems"}))


@pytest.mark.parametrize("route", [None, "none", "flaggems", "tileops"])
def test_non_native_routes_remain_conf_driven(route):
    """Missing conf rows and FlagGems rows do not use the registration rule."""
    expected = {
        None: ("flagos_python", "unlisted"),
        "none": (None, "conf_route_none"),
        "flaggems": ("flagos_python", "routed"),
        "tileops": ("tileops", "routed"),
    }
    assert (
        _expected_route("cuda", "log", route, frozenset({"flaggems"}))
        == expected[route]
    )


@pytest.mark.parametrize(
    ("accel", "op", "raw"),
    [
        ("cuda", "amin", "cuda"),
        ("cuda", "embedding", "cuda"),
        ("cuda", "index_add_", "cuda"),
        ("cuda", "log", "cuda"),
    ],
)
def test_flag_gems_only_h100_routes_measured_on_the_host_fall_back(accel, op, raw):
    """Pin the four native routes the H100 FlagGems wheel leaves to the CPU."""
    assert _expected_route(accel, op, raw, frozenset({"flaggems"})) == (
        None,
        "conf_route_unregistered",
    )
