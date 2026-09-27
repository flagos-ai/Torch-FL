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

"""Unit coverage for the MUSA tune-space repair installed at device init.

``_patch_flaggems_tune_space`` wraps FlagGems' ``LibTuner.run`` so that a tune
space the device refuses is replaced by one that fits. The space in question is
the mthreads ``mm`` entry -- two 64x64x64 tiles at ``num_stages`` 5 and 4 -- which
asks 262144 B and 196608 B of static shared memory against this device's 196608 B
on float64, which is why ``mm`` fails on every float64 shape whose M, K and N all
exceed 32.

Four separate things can be wrong here, and none of them needs a MUSA device to
check, so all of them are pinned off hardware:

* the shared-memory model, which is arithmetic over the tile and the operand
  element size and is the only thing a repair is computed from;
* the boundary -- the limit is exclusive, because an ask that lands exactly on it
  compiles and then fails at launch, so ``<=`` would keep handing out a config
  that cannot run;
* when the repair fires, which is only after a refusal: prefitting was measured on
  S5000 to cost 5.9x to 6.2x on float64 shapes with a narrow N, where the declared
  space runs unaltered (the numbers are in the module under test);
* the wrapper's bookkeeping, which has to hand the declared space back and drop
  FlagGems' cached ``configs_hash``/``kernel_hash`` afterwards, or the next
  launcher is named after a tune space that no longer applies.

That the repair is *enough* on this hardware -- float64 ``mm`` exact across the
tile boundary, and every repaired space inside the limit -- is a measured claim,
and it is asserted in ``tests/integration/ops/test_musa_flaggems.py`` instead.
"""

import importlib
import types

import pytest
import torch

from torch_fl import flagos

triton = None
try:
    triton = importlib.import_module("triton")
except Exception:  # noqa: BLE001 - any import failure means "not installed here"
    triton = None

pytestmark = pytest.mark.skipif(
    triton is None, reason="triton is not installed in this environment"
)

# ``shared_memory_per_multiprocessor`` on an MTT S5000, which is the number
# triton's mthreads backend derives its static shared-memory check from.
LIMIT_S5000 = 196608

# Every row measured on the MTT S5000 by inverting the relation with a stage
# count large enough to trip the compiler's own check: ``OutOfResources`` reports
# the exact ask (``Required: 262144, Hardware limit: 196608``), and the arms that
# do fit run against a CPU reference. The two fitting rows are in the table for
# the same reason as the failing ones: together with the failing pairs they make
# the ``(num_stages - 1)`` factor and the element-size scaling separately
# falsifiable.
MEASURED = [
    ("float32", 64, 64, 64, 8, 229376),
    ("float32", 64, 64, 64, 5, 131072),
    ("float64", 64, 64, 64, 5, 262144),
    ("float64", 64, 64, 64, 4, 196608),
    ("float64", 64, 64, 64, 3, 131072),
    ("float64", 32, 64, 64, 5, 196608),
    ("float64", 32, 64, 64, 4, 147456),
]


def _dot_config(block_m, block_n, block_k, num_stages):
    """The shape of config the mthreads ``mm`` entry declares."""
    return triton.Config(
        {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
        num_warps=4,
        num_stages=num_stages,
    )


def _mm_space():
    """The declared mthreads ``mm`` space, in the order the conf carries it."""
    return [_dot_config(64, 64, 64, 5), _dot_config(64, 64, 64, 4)]


def _operand(dtype):
    """A one-tensor argument tuple of the given dtype: enough to size a launch."""
    return (torch.empty(2, dtype=dtype),)


@pytest.mark.parametrize("dtype,bm,bn,bk,stages,ask", MEASURED)
def test_the_ask_matches_the_measured_table(dtype, bm, bn, bk, stages, ask):
    element_size = torch.empty(0, dtype=getattr(torch, dtype)).element_size()

    assert flagos._pipeline_shared_memory(stages, (bm, bn, bk), element_size) == ask


def test_the_limit_is_an_exclusive_bound():
    """A fitting space has to stay strictly under the limit.

    float64 64x64x64 at 4 stages asks exactly 196608 B. That config compiles and
    is then rejected at launch with a bare ``RuntimeError: Triton Error [MUSA]:
    invalid argument`` -- not with ``OutOfResources``, which is why the
    measurement and not the message is what says the boundary is exclusive.
    """
    tile = (64, 64, 64)
    assert flagos._pipeline_shared_memory(4, tile, 8) == LIMIT_S5000
    for stages in (4, 5):
        config = _dot_config(64, 64, 64, stages)
        assert flagos._fitting_stages(config, tile, 8, LIMIT_S5000) == 3


def test_float32_keeps_the_declared_depth():
    """The element size is the whole difference: the same space on float32 asks
    half as much and every config in it already fits, so the fit has to leave the
    declared stages -- and the space itself -- exactly as they are."""
    tile = (64, 64, 64)
    for stages in (5, 4):
        config = _dot_config(64, 64, 64, stages)
        assert flagos._fitting_stages(config, tile, 4, LIMIT_S5000) == stages
    assert (
        flagos._fitted_tune_space(_mm_space(), _operand(torch.float32), {}, LIMIT_S5000)
        is None
    )


def test_collapsed_configs_are_deduped():
    """Both declared configurations cut back to the same 3-stage tile, and a
    duplicate would only cost a second benchmark."""
    space = flagos._fitted_tune_space(
        _mm_space(), _operand(torch.float64), {}, LIMIT_S5000
    )

    assert [(config.num_stages, config.kwargs) for config in space] == [
        (3, {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64})
    ]


def test_the_declared_space_is_not_mutated():
    """The fit builds new ``triton.Config`` objects rather than editing the
    declared ones in place: FlagGems keeps the space on the tuner for the life of
    the process, and a float64 launch must not narrow it for the float32 one."""
    declared = _mm_space()

    flagos._fitted_tune_space(declared, _operand(torch.float64), {}, LIMIT_S5000)

    assert [config.num_stages for config in declared] == [5, 4]
    assert [config.kwargs["BLOCK_K"] for config in declared] == [64, 64]


def test_the_widest_operand_decides():
    """The per-launch dtype is what the fit is sized by, so a mixed launch is
    sized by the wider operand rather than the first one."""
    f32 = torch.empty(2, dtype=torch.float32)
    f64 = torch.empty(2, dtype=torch.float64)

    assert flagos._operand_element_size((f32,), {}) == 4
    assert flagos._operand_element_size((f32, f64), {}) == 8
    assert flagos._operand_element_size((f32,), {"other": f64}) == 8
    assert flagos._operand_element_size((), {}) is None
    assert flagos._operand_element_size((1, 2), {"dtype": torch.float64}) is None


def test_a_space_without_a_dot_tile_is_left_alone():
    """Reductions, pointwise blocks and anything keyed differently are not
    pipelined matmuls, and the model above does not describe them."""
    space = [
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": "auto"}, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=5),
    ]

    assert [flagos._dot_tile(config) for config in space] == [None, None, None]
    assert (
        flagos._fitted_tune_space(space, _operand(torch.float64), {}, LIMIT_S5000)
        is None
    )


class _Refused(Exception):
    """Stands in for the compiler's ``OutOfResources``, which a tuner cannot
    absorb: it is a ``TritonError``, hence an ``Exception`` and not a
    ``RuntimeError``, so FlagGems' ``bench`` does not catch it and the whole
    tuning pass ends on the first config that cannot be built."""


class _StubTuner:
    """A stand-in for ``LibTuner`` carrying only what the wrapper touches.

    ``run`` records the space it was handed on every attempt -- the wrapper
    restores ``configs`` in a ``finally``, so the attempts are the one place a
    fitted list is observable -- and refuses the first ``refusals`` launches the
    way the device does, before writing the ``configs_hash`` FlagGems' cached
    property would have left behind.
    """

    def __init__(self, refusals=0):
        self.configs = []
        self.refusals = refusals
        self.attempts = []

    def run(self, *args, **kwargs):
        self.attempts.append(list(self.configs))
        if len(self.attempts) <= self.refusals:
            raise _Refused(f"launch {len(self.attempts)} refused")
        self.__dict__["configs_hash"] = "derived from the list above"
        return "ran"


@pytest.fixture
def as_musa(monkeypatch):
    """Force the configuration the patch is gated on rather than assuming it:
    this suite also runs on the MTT S5000, where the real detector agrees, and on
    hosts where it answers something else."""
    monkeypatch.setattr("torch_fl._build_accelerator", lambda: "musa")
    monkeypatch.setattr(flagos, "_musa_shared_memory_limit", lambda: LIMIT_S5000)


@pytest.fixture
def stub_libentry(monkeypatch):
    """Stand in for ``flag_gems.utils.libentry``.

    Intercepted at the ``importlib`` call rather than by installing a module in
    ``sys.modules``, so a real FlagGems install cannot be half-replaced: names
    other than the one the patch asks for still resolve to the real module.

    The stub is *not* patched here -- that is ``fitted_tuner``, which takes this
    fixture -- because the gate has to be checked against an untouched one.
    """
    module = types.SimpleNamespace(
        LibTuner=type(
            "LibTuner",
            (),
            {"__init__": _StubTuner.__init__, "run": _StubTuner.run},
        )
    )
    real_import = importlib.import_module

    def fake(name, *args, **kwargs):
        if name == "flag_gems.utils.libentry":
            return module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(flagos.importlib, "import_module", fake)
    return module


@pytest.fixture
def fitted_tuner(as_musa, stub_libentry):
    """The stub with the real patch installed on it, as device init leaves it.

    Installed per test rather than once per session because the marker lives on
    the stub class, and a class built per test has no marker -- sharing one
    instance across tests would also make the patch's own once-only guard the
    thing under test instead of the wrapper.
    """
    flagos._patch_flaggems_tune_space()
    assert getattr(stub_libentry.LibTuner, flagos._TUNE_SPACE_PATCHED, False)
    return stub_libentry


def test_off_the_measured_backend_nothing_is_patched(monkeypatch, stub_libentry):
    """Anywhere but a MUSA build the tuner is untouched, so no other platform's
    ``tl.dot`` space is rewritten by this."""
    monkeypatch.setattr("torch_fl._build_accelerator", lambda: "cuda")
    before = stub_libentry.LibTuner.run

    flagos._patch_flaggems_tune_space()

    assert stub_libentry.LibTuner.run is before
    assert not getattr(stub_libentry.LibTuner, flagos._TUNE_SPACE_PATCHED, False)


def test_a_launch_the_device_takes_is_never_refitted(monkeypatch, fitted_tuner):
    """The float64 narrow-N case, and every other space the device accepts: the
    declared configs reach the tuner, the fit is not even asked, and nothing needs
    restoring.

    This is the whole reason the repair is failure-driven. The declared space is
    the tuned one -- FlagTree's AABS narrows its tile per shape -- and a fit that
    ran before the first launch was measured on S5000 to cost 5.9x at
    ``2048x2048x16`` and 6.2x at ``4096x4096x16``, where the declared space is
    what the tuner benchmarks and what it should keep benchmarking.
    """
    tuner = fitted_tuner.LibTuner()
    declared = _mm_space()
    tuner.configs = declared
    asked = []

    def spy(*call):
        asked.append(call)

    monkeypatch.setattr(flagos, "_fitted_tune_space", spy)

    assert tuner.run(torch.empty(2, dtype=torch.float64)) == "ran"

    assert [
        [config.num_stages for config in attempt] for attempt in tuner.attempts
    ] == [[5, 4]]
    assert tuner.configs is declared
    assert asked == []


def test_a_refused_launch_is_retried_with_a_fitting_space(fitted_tuner):
    """The failing case end to end: the declared 5/4 pair is refused, the retry is
    handed the 3-stage tile, and the declared pair is back in place afterwards.

    The hash assertion is why the restore is not cosmetic: FlagGems caches
    ``configs_hash`` and ``kernel_hash`` from whichever list is current, so a
    value left behind by the fitted launch names a launcher nothing declares. The
    stub writes one on the launch that succeeds, which is what makes its absence
    after the call an assertion about the wrapper rather than about the stub.
    """
    tuner = fitted_tuner.LibTuner(refusals=1)
    declared = _mm_space()
    tuner.configs = declared

    assert tuner.run(torch.empty(2, dtype=torch.float64)) == "ran"

    assert [
        [config.num_stages for config in attempt] for attempt in tuner.attempts
    ] == [[5, 4], [3]]
    assert tuner.configs is declared
    assert "configs_hash" not in tuner.__dict__


def test_a_refusal_with_nothing_to_fit_is_the_callers_exception(fitted_tuner):
    """A space that already fits has nothing to repair, so the refusal is raised
    as it arrived: this patch does not turn an unrelated failure into a silent
    second attempt."""
    tuner = fitted_tuner.LibTuner(refusals=1)
    tuner.configs = _mm_space()

    with pytest.raises(_Refused):
        tuner.run(torch.empty(2, dtype=torch.float32))

    assert [
        [config.num_stages for config in attempt] for attempt in tuner.attempts
    ] == [[5, 4]]


def test_a_failed_repair_reports_the_original_refusal(fitted_tuner):
    """The retry is a repair, not a redefinition: when the repaired space does not
    take either, the caller sees the device's refusal of the declared space rather
    than whatever its replacement ran into."""
    tuner = fitted_tuner.LibTuner(refusals=2)
    tuner.configs = _mm_space()

    with pytest.raises(_Refused) as caught:
        tuner.run(torch.empty(2, dtype=torch.float64))

    assert str(caught.value) == "launch 1 refused"
    assert [
        [config.num_stages for config in attempt] for attempt in tuner.attempts
    ] == [[5, 4], [3]]
    assert "configs_hash" not in tuner.__dict__


def test_a_single_config_space_is_still_repaired(fitted_tuner):
    """A one-config space takes ``LibTuner.run``'s ``else`` branch and never calls
    ``prune_configs``, so a patch on the FlagTree AABS hook would not reach it --
    and a one-config space is what a repaired float64 space collapses to, so the
    wrapper has to repair on its own second attempt as well."""
    tuner = fitted_tuner.LibTuner(refusals=1)
    tuner.configs = [_dot_config(64, 64, 64, 4)]

    tuner.run(torch.empty(2, dtype=torch.float64))

    assert [
        [config.num_stages for config in attempt] for attempt in tuner.attempts
    ] == [[4], [3]]


def test_the_patch_is_installed_once(fitted_tuner):
    """Device init can run more than once, and the marker is what keeps the
    wrapper from being stacked on itself -- a doubled wrapper would restore the
    fitted list as if it were the declared one."""
    first = fitted_tuner.LibTuner.run

    flagos._patch_flaggems_tune_space()

    assert fitted_tuner.LibTuner.run is first


def test_failure_is_not_fatal(monkeypatch, as_musa):
    """Best-effort, like the patches around it: this runs inside device init."""

    def boom(name, *args, **kwargs):
        raise RuntimeError("no FlagGems here")

    monkeypatch.setattr(flagos.importlib, "import_module", boom)

    flagos._patch_flaggems_tune_space()  # must not raise
