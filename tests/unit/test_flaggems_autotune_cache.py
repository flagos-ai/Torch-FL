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

"""Unit coverage for the Triton autotune-cache knob installed on DCU.

``_enable_flaggems_autotune_cache`` reads the environment and writes a knob on
the ``triton.knobs`` singleton, so the risk it carries is not the 0.8 s it saves
-- that is measured on hardware and recorded in
``docs/reference/operator-support.md`` -- but the two ways it could be wrong on
a host that is not the one it was measured on:

* it could fire where it was not asked to, and turn on a caching mode for a
  vendor whose FlagGems stack was never measured with it, and
* it could overwrite a setting the user made deliberately.

Both are decisions taken before any device is touched, so they are pinned here;
the saving itself cannot be asserted off a DCU.
"""

import importlib
import os

import pytest

from torch_fl import flagos

triton_knobs = None
try:
    triton_knobs = importlib.import_module("triton.knobs")
except Exception:  # noqa: BLE001 - any import failure means "not installed here"
    triton_knobs = None

pytestmark = pytest.mark.skipif(
    triton_knobs is None, reason="triton is not installed in this environment"
)

ENV_VAR = "TRITON_CACHE_AUTOTUNING"


@pytest.fixture
def as_dcu(monkeypatch):
    """Make the patch believe it is on the configuration it was measured on."""
    monkeypatch.setattr("torch_fl._build_accelerator", lambda: "dcu")
    monkeypatch.setattr("torch_fl._conf_routes_to_flaggems", lambda: True)
    monkeypatch.delenv(ENV_VAR, raising=False)
    yield
    # Clearing the override, not restoring its value: ``env_base.__set__`` calls
    # ``knobs.setenv``, so writing the knob back would leave the variable set for
    # whatever runs next in this process. Handing it the ``Env`` sentinel pops the
    # override instead, and the value falls back to the environment, which
    # monkeypatch is about to put back.
    triton_knobs.autotuning.cache = triton_knobs.env


def test_noop_off_the_measured_configuration(monkeypatch):
    """Anywhere but a FlagGems-routed DCU build, nothing is touched. Forced
    rather than assumed: this suite also runs on the DCU host the number was
    measured on, where the real detector answers "dcu"."""
    monkeypatch.setattr("torch_fl._build_accelerator", lambda: "cuda")
    monkeypatch.setattr("torch_fl._conf_routes_to_flaggems", lambda: True)
    monkeypatch.delenv(ENV_VAR, raising=False)
    previous = triton_knobs.autotuning.cache

    flagos._enable_flaggems_autotune_cache()

    assert ENV_VAR not in os.environ
    assert triton_knobs.autotuning.cache is previous


def test_cuda_route_alone_is_not_enough(monkeypatch, as_dcu):
    """The conf has to route to FlagGems as well: a DCU build on the CUDA route
    has no FlagGems autotuner to save anything on."""
    monkeypatch.setattr("torch_fl._conf_routes_to_flaggems", lambda: False)
    previous = triton_knobs.autotuning.cache

    flagos._enable_flaggems_autotune_cache()

    assert ENV_VAR not in os.environ
    assert triton_knobs.autotuning.cache is previous


def test_enables_both_halves(as_dcu):
    """The env var is for readers that have not imported triton yet, and the
    knob is for the ones that have -- triton.knobs reads the variable once, when
    its knob objects are built, which is before this runs."""
    flagos._enable_flaggems_autotune_cache()

    assert os.environ[ENV_VAR] == "1"
    assert triton_knobs.autotuning.cache is True


def test_user_setting_wins(as_dcu, monkeypatch):
    """A deliberate 0 -- a user who wants the benchmark back, or who is chasing
    an autotune that settled on a config they disagree with -- is left alone.

    Both halves, because the two are written by the same assignment: ``env_bool``
    is a data descriptor whose ``__set__`` calls ``knobs.setenv``, so a knock-on
    write of the variable is exactly how this went wrong the first time.
    """
    monkeypatch.setenv(ENV_VAR, "0")

    flagos._enable_flaggems_autotune_cache()

    assert os.environ[ENV_VAR] == "0"
    assert triton_knobs.autotuning.cache is False


def test_unparsable_setting_is_left_alone(as_dcu, monkeypatch):
    """Triton's own parser decides, not a hand-rolled truthiness test: anything
    that does not read as "on" there does not get turned on here."""
    monkeypatch.setenv(ENV_VAR, "off")

    flagos._enable_flaggems_autotune_cache()

    assert os.environ[ENV_VAR] == "off"
    assert triton_knobs.autotuning.cache is False


def test_failure_is_not_fatal(as_dcu, monkeypatch):
    """Best-effort, like the patches around it: this runs inside device init."""

    def boom(name, *args, **kwargs):
        raise RuntimeError("no triton here")

    monkeypatch.setattr(flagos.importlib, "import_module", boom)

    flagos._enable_flaggems_autotune_cache()  # must not raise
    assert triton_knobs.autotuning.cache is False
