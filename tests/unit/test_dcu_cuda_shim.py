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

"""Unit coverage for the DCU ``torch.cuda`` compatibility shim.

Dropping DTK's ``libtorch_python.so`` leaves the official wheel's
``torch.cuda.is_available()`` False, which disables triton-hcu and breaks seeding
even though the DTK kernels are registered and computing.  These tests drive
``patch_torch_cuda_for_dcu()`` with a stubbed flagos runtime and stubbed HIP
driver, so they need neither a DCU device nor DTK.
"""

import ctypes
import types

import pytest
import torch
from dcu_module_loader import load_module

_dcu_compat = load_module(
    "torch_fl/accelerator/dcu/_dcu_compat.py", "_flagos_test_dcu_compat"
)

_PATCHED_ATTRS = (
    "get_device_properties",
    "get_device_name",
    "get_device_capability",
    "is_available",
    "_lazy_init",
    "device_count",
    "_initialized",
    "set_device",
    "current_device",
    "synchronize",
    "_exchange_device",
    "_maybe_exchange_device",
    "current_stream",
    "default_stream",
    "manual_seed",
    "manual_seed_all",
    "default_generators",
    "Event",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "max_memory_allocated",
    "max_memory_reserved",
    "empty_cache",
    "reset_peak_memory_stats",
    "reset_max_memory_allocated",
)

_SENTINEL = object()


class _FakeEvent:
    """Stand-in for ``torch.flagos.Event``, which needs a live device."""

    def __init__(self, enable_timing=False, **kwargs):
        self.enable_timing = enable_timing


class _FakeGenerator:
    def __init__(self, index):
        self.index = index
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self

    def get_state(self):
        return torch.tensor([self.index, 0], dtype=torch.int64).view(torch.uint8)


class _FakeFlagosAllocator:
    """The flagos caching allocator's query surface, as torch.cuda callers see it.

    Values are distinguishable so a test can tell which one an accessor read, and
    the flat names are flagos's own -- ``peak_allocated_bytes``, not torch's
    nested ``allocated_bytes.all.peak``.
    """

    def __init__(self):
        self.allocated = 5 * 1024**3
        self.reserved = 6 * 1024**3
        self.peak_allocated = 7 * 1024**3
        self.peak_reserved = 8 * 1024**3
        self.calls = []

    def memory_stats(self, index):
        self.calls.append(("memory_stats", index))
        return {
            "allocated_bytes": self.allocated,
            "reserved_bytes": self.reserved,
            "peak_allocated_bytes": self.peak_allocated,
            "peak_reserved_bytes": self.peak_reserved,
        }

    def memory_allocated(self, index):
        self.calls.append(("memory_allocated", index))
        return self.allocated

    def memory_reserved(self, index):
        self.calls.append(("memory_reserved", index))
        return self.reserved

    def empty_cache(self):
        self.calls.append(("empty_cache", None))

    def reset_peak_memory_stats(self, index):
        self.calls.append(("reset_peak_memory_stats", index))


def _sdpa_state():
    return (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )


def _restore_sdpa_state(state):
    # Tolerant on purpose: monkeypatch undoes a delattr only after this fixture
    # has torn down, so a test that removed one of the setters still gets here.
    for setter, enabled in zip(
        ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"), state
    ):
        setter = getattr(torch.backends.cuda, setter, None)
        if setter is not None:
            setter(enabled)


@pytest.fixture
def shim(monkeypatch):
    """Reset shim state and restore everything ``torch.cuda`` it overwrites."""
    saved = {name: getattr(torch.cuda, name, _SENTINEL) for name in _PATCHED_ATTRS}
    saved_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", _SENTINEL)
    saved_queued = list(getattr(torch.cuda, "_queued_calls", []))
    saved_sdpa = _sdpa_state()
    # The SDPA switch is opt-out by environment variable, so an ambient value
    # would decide these tests instead of the fixture.
    monkeypatch.delenv("FLAGOS_DCU_SDPA_FLASH", raising=False)

    monkeypatch.setattr(_dcu_compat, "_cuda_patched", False)
    monkeypatch.setattr(_dcu_compat, "_props_cache", {})
    # Pre-populate the generator cache so nothing tries to build a real
    # torch.Generator(device="cuda"), which needs a device and libcaffe2_nvrtc.
    generators = {i: _FakeGenerator(i) for i in range(4)}
    monkeypatch.setattr(_dcu_compat, "_cuda_generators", generators)

    calls = []
    allocator = _FakeFlagosAllocator()
    flagos = types.SimpleNamespace(
        device_count=lambda: 4,
        current_device=lambda: calls[-1][1] if calls else 0,
        set_device=lambda idx: calls.append(("set_device", idx)),
        synchronize=lambda device=None: calls.append(("synchronize", device)),
        Event=_FakeEvent,
        memory_stats=allocator.memory_stats,
        memory_allocated=allocator.memory_allocated,
        memory_reserved=allocator.memory_reserved,
        empty_cache=allocator.empty_cache,
        reset_peak_memory_stats=allocator.reset_peak_memory_stats,
    )
    monkeypatch.setattr(torch, "flagos", flagos, raising=False)
    # The stock +cpu wheel answers False here; that is the case the shim exists for.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_dcu_compat, "_load_hip", lambda: None)

    try:
        yield types.SimpleNamespace(
            calls=calls, generators=generators, flagos=flagos, allocator=allocator
        )
    finally:
        _restore_sdpa_state(saved_sdpa)
        for name, value in saved.items():
            if value is _SENTINEL:
                if hasattr(torch.cuda, name):
                    delattr(torch.cuda, name)
            else:
                setattr(torch.cuda, name, value)
        if saved_raw_stream is _SENTINEL:
            if hasattr(torch._C, "_cuda_getCurrentRawStream"):
                del torch._C._cuda_getCurrentRawStream
        else:
            torch._C._cuda_getCurrentRawStream = saved_raw_stream
        if hasattr(torch.cuda, "_queued_calls"):
            torch.cuda._queued_calls[:] = saved_queued


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


def test_patch_declines_when_a_real_dtk_torch_is_in_front(shim, monkeypatch):
    """A native torch.cuda is strictly better and must not be shadowed."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _dcu_compat.patch_torch_cuda_for_dcu() is False
    assert _dcu_compat._cuda_patched is False


def test_patch_declines_without_a_reachable_device(shim, monkeypatch):
    monkeypatch.setattr(shim.flagos, "device_count", lambda: 0)
    assert _dcu_compat.patch_torch_cuda_for_dcu() is False
    assert torch.cuda.is_available() is False


def test_patch_declines_when_flagos_runtime_raises(shim, monkeypatch):
    def boom():
        raise RuntimeError("no flagos runtime")

    monkeypatch.setattr(shim.flagos, "device_count", boom)
    assert _dcu_compat.patch_torch_cuda_for_dcu() is False


def test_patch_is_idempotent(shim):
    assert _dcu_compat.patch_torch_cuda_for_dcu() is True
    first = torch.cuda.get_device_properties
    assert _dcu_compat.patch_torch_cuda_for_dcu() is True
    assert torch.cuda.get_device_properties is first


# ---------------------------------------------------------------------------
# Availability and device routing
# ---------------------------------------------------------------------------


def test_availability_gate_for_triton_hcu(shim):
    """triton's hcu backend gates on is_available() and torch.version.hip."""
    assert _dcu_compat.patch_torch_cuda_for_dcu() is True
    assert torch.cuda.is_available() is True
    assert torch.cuda.device_count() == 4
    assert torch.cuda._lazy_init() is None


def test_set_device_reaches_the_flagos_runtime(shim):
    _dcu_compat.patch_torch_cuda_for_dcu()
    torch.cuda.set_device(3)
    assert ("set_device", 3) in shim.calls
    assert torch.cuda.current_device() == 3


@pytest.mark.parametrize(
    "device,expected",
    [
        (2, 2),
        ("cuda:2", 2),
        ("cuda", 0),
        (torch.device("cuda", 2), 2),
        (torch.device("cuda"), 0),
    ],
)
def test_device_argument_forms(shim, device, expected):
    _dcu_compat.patch_torch_cuda_for_dcu()
    torch.cuda.set_device(device)
    assert shim.calls[-1] == ("set_device", expected)


def test_exchange_device_returns_the_previous_index(shim):
    _dcu_compat.patch_torch_cuda_for_dcu()
    torch.cuda.set_device(1)
    assert torch.cuda._exchange_device(2) == 1
    assert torch.cuda.current_device() == 2
    # Inductor's device guard passes -1 to mean "no switch".
    assert torch.cuda._exchange_device(-1) == -1
    assert torch.cuda.current_device() == 2


def test_synchronize_forwards_to_flagos(shim):
    _dcu_compat.patch_torch_cuda_for_dcu()
    torch.cuda.synchronize()
    assert ("synchronize", None) in shim.calls


def test_streams_agree_on_the_default_handle(shim):
    _dcu_compat.patch_torch_cuda_for_dcu()
    assert torch.cuda.current_stream(1).cuda_stream == 0
    assert torch.cuda.default_stream().cuda_stream == 0
    assert torch._C._cuda_getCurrentRawStream(1) == 0
    assert torch.cuda.current_stream(1).device_index == 1


def test_event_is_constructible_for_the_triton_autotuner(shim):
    """triton.testing.do_bench times candidates with Event(enable_timing=True).

    The +cpu wheel's torch.cuda.Event is a dummy base class that raises "Tried to
    instantiate dummy base class Event" on construction, which failed every
    FlagGems op that autotunes -- add.Tensor and mm among them.
    """
    _dcu_compat.patch_torch_cuda_for_dcu()
    assert torch.cuda.Event is shim.flagos.Event
    assert torch.cuda.Event(enable_timing=True).enable_timing is True


def test_event_is_left_alone_when_flagos_has_none(shim, monkeypatch):
    """A flagos module without Event must not install ``None`` over torch.cuda's."""
    monkeypatch.delattr(shim.flagos, "Event")
    before = torch.cuda.Event
    _dcu_compat.patch_torch_cuda_for_dcu()
    assert torch.cuda.Event is before


# ---------------------------------------------------------------------------
# Device properties
# ---------------------------------------------------------------------------


def test_properties_fall_back_to_cuda_shaped_defaults_without_a_driver(shim):
    _dcu_compat.patch_torch_cuda_for_dcu()
    props = torch.cuda.get_device_properties(0)
    assert props.warp_size == 64
    assert torch.cuda.get_device_capability(0) == (props.major, props.minor)
    assert torch.cuda.get_device_name(0) == props.name
    assert props.name


def test_properties_expose_gcn_arch_name_for_inductor(shim, monkeypatch):
    """codecache reads gcnArchName whenever torch.version.cuda is None."""
    monkeypatch.setattr(_dcu_compat, "_load_hip", lambda: object())
    monkeypatch.setattr(_dcu_compat, "_hip_device_name", lambda hip, i: f"DCU {i}")
    monkeypatch.setattr(_dcu_compat, "_hip_gcn_arch_name", lambda hip, i: "gfx936")
    monkeypatch.setattr(_dcu_compat, "_hip_total_memory", lambda hip, i: 68702699520)
    monkeypatch.setattr(
        _dcu_compat, "_hip_attribute", lambda hip, attr, i, default=0: default or 7
    )
    _dcu_compat.patch_torch_cuda_for_dcu()
    props = torch.cuda.get_device_properties(1)
    assert props.gcnArchName == "gfx936"
    assert props.name == "DCU 1"
    assert props.total_memory == 68702699520


def test_properties_are_cached_per_device(shim, monkeypatch):
    queries = []
    real = _dcu_compat._query_device_properties
    monkeypatch.setattr(
        _dcu_compat,
        "_query_device_properties",
        lambda idx: queries.append(idx) or real(idx),
    )
    _dcu_compat.patch_torch_cuda_for_dcu()
    torch.cuda.get_device_properties(2)
    torch.cuda.get_device_properties(2)
    torch.cuda.get_device_properties(3)
    assert queries == [2, 3]


def test_hip_attribute_returns_the_default_on_driver_error(monkeypatch):
    class Hip:
        def hipDeviceGetAttribute(self, out, attr, index):
            return 1  # hipErrorInvalidValue

    assert _dcu_compat._hip_attribute(Hip(), 27, 0, default=64) == 64


def test_hip_attribute_reads_the_out_parameter(monkeypatch):
    class Hip:
        def hipDeviceGetAttribute(self, out, attr, index):
            ctypes.cast(out, ctypes.POINTER(ctypes.c_int))[0] = 80
            return 0

    assert _dcu_compat._hip_attribute(Hip(), 32, 0, default=0) == 80


def test_gcn_arch_is_scraped_from_the_properties_blob(monkeypatch):
    """The GcnArchName attribute id errors on DTK, so the blob is scanned."""

    class Hip:
        def hipGetDeviceProperties(self, out, index):
            blob = ctypes.cast(out, ctypes.POINTER(ctypes.c_char * 2048))[0]
            blob[64:96] = b"gfx936:sramecc+:xnack-".ljust(32, b"\x00")
            return 0

    assert _dcu_compat._hip_gcn_arch_name(Hip(), 0) == "gfx936:sramecc+:xnack-"


# ---------------------------------------------------------------------------
# Generators / seeding
# ---------------------------------------------------------------------------


def test_default_generators_are_bounded(shim):
    _dcu_compat.patch_torch_cuda_for_dcu()
    gens = torch.cuda.default_generators
    assert len(gens) == 4
    assert gens[0] is shim.generators[0]
    assert gens[-1] is shim.generators[3]
    with pytest.raises(IndexError):
        gens[4]
    with pytest.raises(IndexError):
        gens[-5]


def test_default_generators_iterate_without_running_away(shim):
    """Unbounded __getitem__ would make ``for g in gens`` loop forever."""
    _dcu_compat.patch_torch_cuda_for_dcu()
    assert len(list(torch.cuda.default_generators)) == 4
    assert len(tuple(torch.cuda.default_generators)) == 4
    assert len(torch.cuda.default_generators[1:3]) == 2


def test_manual_seed_all_seeds_every_device(shim):
    _dcu_compat.patch_torch_cuda_for_dcu()
    torch.cuda.manual_seed_all(1234)
    assert [g.seed for g in shim.generators.values()] == [1234] * 4


def test_manual_seed_targets_the_current_device(shim):
    _dcu_compat.patch_torch_cuda_for_dcu()
    torch.cuda.set_device(2)
    torch.cuda.manual_seed(99)
    assert shim.generators[2].seed == 99
    assert shim.generators[0].seed is None


def test_torch_manual_seed_no_longer_raises(shim):
    """With is_available()=True, torch.manual_seed() walks the CUDA generators;
    an empty collection there raised IndexError before the shim existed."""
    _dcu_compat.patch_torch_cuda_for_dcu()
    torch.manual_seed(7)
    assert [g.seed for g in shim.generators.values()] == [7] * 4


# ---------------------------------------------------------------------------
# Memory queries
# ---------------------------------------------------------------------------


def test_memory_queries_route_to_the_flagos_allocator(shim):
    """Every torch.cuda.memory_* name must land on flagos, never on torch._C.

    The shim has to set torch.cuda._initialized = True for triton; torch's own
    memory_* wrappers gate on is_initialized() and fall through to
    torch._C._cuda_memoryStats once it is true. That half of libtorch is not in
    the +cpu wheel, so routing is the difference between a number and an
    AttributeError -- and diffusers calls empty_device_cache() at the end of
    every _load_pretrained_model, before any of the model work.
    """
    _dcu_compat.patch_torch_cuda_for_dcu()
    assert torch.cuda.memory_allocated() == shim.allocator.allocated
    assert torch.cuda.memory_reserved() == shim.allocator.reserved
    assert torch.cuda.memory_allocated("cuda:3") == shim.allocator.allocated
    assert (
        torch.cuda.memory_reserved(torch.device("cuda", 1)) == shim.allocator.reserved
    )
    assert ("memory_allocated", 3) in shim.allocator.calls
    assert ("memory_reserved", 1) in shim.allocator.calls


def test_memory_stats_carry_the_names_torch_callers_look_up(shim):
    """flagos reports flat names; torch.cuda's schema is nested."""
    _dcu_compat.patch_torch_cuda_for_dcu()
    stats = torch.cuda.memory_stats(1)
    assert stats["allocated_bytes.all.current"] == shim.allocator.allocated
    assert stats["reserved_bytes.all.current"] == shim.allocator.reserved
    assert stats["allocated_bytes.all.peak"] == shim.allocator.peak_allocated
    assert stats["reserved_bytes.all.peak"] == shim.allocator.peak_reserved
    # The flat originals stay readable for callers that speak flagos's schema.
    assert stats["peak_allocated_bytes"] == shim.allocator.peak_allocated


def test_peak_accessors_read_the_peak_keys(shim):
    _dcu_compat.patch_torch_cuda_for_dcu()
    assert torch.cuda.max_memory_allocated() == shim.allocator.peak_allocated
    assert torch.cuda.max_memory_reserved() == shim.allocator.peak_reserved


def test_empty_cache_and_peak_reset_reach_the_allocator(shim):
    _dcu_compat.patch_torch_cuda_for_dcu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(2)
    torch.cuda.reset_max_memory_allocated(2)
    assert ("empty_cache", None) in shim.allocator.calls
    assert ("reset_peak_memory_stats", 2) in shim.allocator.calls


def test_memory_shim_is_skipped_without_a_flagos_allocator(shim, monkeypatch):
    """A flagos runtime with no allocator surface must not half-install it."""
    monkeypatch.delattr(shim.flagos, "memory_stats")
    before_allocated = torch.cuda.memory_allocated
    before_empty = torch.cuda.empty_cache
    _dcu_compat.patch_torch_cuda_for_dcu()
    assert torch.cuda.memory_allocated is before_allocated
    assert torch.cuda.empty_cache is before_empty


# ---------------------------------------------------------------------------
# SDPA backend selection
# ---------------------------------------------------------------------------


def test_sdpa_selects_the_one_backend_this_stack_can_execute(shim):
    """The decoupled wheel can only run the math decomposition.

    DTK's CUTLASS flash adapter is unreachable: its support probe
    (at::native::can_use_flash_attention) resolves out of the *CPU* libtorch_cpu
    and answers False, so pytorch_flash::mha_fwd skips the adapter and falls into
    its un-compiled aotriton branch -- RuntimeError, not a clean decline.
    Memory-efficient attention is not compiled at all. Leaving the fused flags
    on lets aten's composite pick a leaf that then raises, which is what killed
    every attention call in the Qwen-Image transformer.
    """
    _dcu_compat.patch_torch_cuda_for_dcu()
    assert torch.backends.cuda.flash_sdp_enabled() is False
    assert torch.backends.cuda.mem_efficient_sdp_enabled() is False
    assert torch.backends.cuda.math_sdp_enabled() is True


def test_sdpa_keeps_the_fused_backends_when_flagged(shim, monkeypatch):
    """FLAGOS_DCU_SDPA_FLASH=1 is the escape hatch for a stack where they work."""
    monkeypatch.setenv("FLAGOS_DCU_SDPA_FLASH", "1")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    _dcu_compat.patch_torch_cuda_for_dcu()
    assert torch.backends.cuda.flash_sdp_enabled() is True
    assert torch.backends.cuda.mem_efficient_sdp_enabled() is True


def test_sdpa_switch_does_not_take_the_whole_shim_down(shim, monkeypatch):
    """A torch build without the backend flags must still get the rest."""
    monkeypatch.delattr(torch.backends.cuda, "enable_flash_sdp")
    assert _dcu_compat.patch_torch_cuda_for_dcu() is True
    assert torch.cuda.is_available() is True


# install_dcu_rng_bridge() is covered by tests/unit/test_dcu_rng_bridge.py, which
# imports the real torch_fl package (the bridge needs torch_fl.flagos).
