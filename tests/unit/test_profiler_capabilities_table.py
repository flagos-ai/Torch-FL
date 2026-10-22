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

"""Check profiler capability rows before they reach hardware runners."""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_INTEGRATION = _REPO / "tests" / "integration"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("platform_support", _INTEGRATION / "platform_support.py")
profiler_support = _load(
    "profiler_support_under_test", _INTEGRATION / "profiler_support.py"
)
ALL_PLATFORMS = (
    "ascend",
    "cuda",
    "dcu",
    "gcu",
    "iluvatar",
    "kunlunxin",
    "metax",
    "mps",
    "musa",
    "ppu",
)


@pytest.fixture(autouse=True)
def _fake_build_record(monkeypatch):
    env = type("Env", (), {"build_kernels": staticmethod(lambda: frozenset())})()
    monkeypatch.setattr(
        profiler_support,
        "_boxing_built",
        lambda: "boxing" in env.build_kernels(),
    )
    return env


_BOOL_ROWS = (
    "device",
    "kernel",
    "demangled",
    "runtime",
    "cbid",
    "memcpy",
    "memset",
    "flow",
    "linkage",
    "metadata",
)


@pytest.mark.parametrize("platform", ALL_PLATFORMS)
def test_every_platform_has_boolean_capabilities(platform):
    caps = profiler_support.capabilities_for_platform(platform)
    assert caps.platform == platform
    for row in _BOOL_ROWS:
        assert isinstance(getattr(caps, row), bool), row


@pytest.mark.parametrize("platform", ALL_PLATFORMS)
def test_device_rows_match_the_measured_tracers(platform):
    caps = profiler_support.capabilities_for_platform(platform)
    assert caps.device == (platform != "kunlunxin")
    for row in ("kernel", "runtime", "flow", "metadata"):
        assert getattr(caps, row) == caps.device
    assert caps.linkage == (caps.device and platform != "gcu")
    assert caps.demangled == (caps.device and platform != "gcu")
    if platform == "kunlunxin":
        assert not caps.memcpy and not caps.memset
    elif platform not in {"cuda", "metax"}:
        assert not caps.memset


@pytest.mark.parametrize("platform", ("cuda", "metax"))
def test_memset_follows_the_boxing_build_record(platform, _fake_build_record):
    assert not profiler_support.capabilities_for_platform(platform).memset
    _fake_build_record.build_kernels = staticmethod(lambda: frozenset({"boxing"}))
    assert profiler_support.capabilities_for_platform(platform).memset
