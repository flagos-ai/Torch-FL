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

"""Unit tests for the compile platform profile and its consumers.

These run on any platform, including plain CPU: everything under test is the
*selection* logic that decides which Triton target name, hardware module and
codegen classes a build uses. The profile is patched rather than inferred from
the running build, so the Ascend path is exercised on a CUDA box and vice versa.

What is deliberately not covered here: whether the Ascend toolchain actually
compiles a kernel. That needs a real 910 and lives in
tests/integration/test_compile.py.
"""

import functools

import pytest

import torch

from torch_fl.compile import platform_profile as pp


@pytest.fixture
def as_ascend(monkeypatch):
    """Force every profile consumer onto the Ascend (non-CUDA) profile."""
    for module in ("platform_profile", "device_interface", "inductor_codegen"):
        monkeypatch.setattr(
            f"torch_fl.compile.{module}.platform_profile",
            lambda: pp._ASCEND_PROFILE,
            raising=False,
        )
    return pp._ASCEND_PROFILE


@pytest.fixture
def as_cuda(monkeypatch):
    for module in ("platform_profile", "device_interface", "inductor_codegen"):
        monkeypatch.setattr(
            f"torch_fl.compile.{module}.platform_profile",
            lambda: pp._CUDA_PROFILE,
            raising=False,
        )
    monkeypatch.setattr("torch_fl._build_config.ACCELERATOR", "cuda")
    return pp._CUDA_PROFILE


@pytest.fixture
def as_musa(monkeypatch):
    """Force every profile consumer onto the MUSA profile.

    Not CUDA-like, but not Ascend either: the third shape, and the one a single
    `is_cuda_like` boolean cannot express. ACCELERATOR is patched alongside the
    profile because the generated-code snippets branch on it directly, so a run
    on any other platform's CI would otherwise take the wrong branch.
    """
    for module in ("platform_profile", "device_interface", "inductor_codegen"):
        monkeypatch.setattr(
            f"torch_fl.compile.{module}.platform_profile",
            lambda: pp._MUSA_PROFILE,
            raising=False,
        )
    monkeypatch.setattr("torch_fl._build_config.ACCELERATOR", "musa")
    return pp._MUSA_PROFILE


@pytest.fixture
def as_kunlunxin(monkeypatch):
    """Force every profile consumer onto the Kunlunxin profile.

    The interesting one for the device registry: it is not CUDA-like, yet its
    reported Triton device type *is* "cuda" (FlagTree's XPU backend presents the
    target that way), so it is the only accelerator that lands on a name the
    stock CudaInterface also claims.
    """
    for module in ("platform_profile", "device_interface", "inductor_codegen"):
        monkeypatch.setattr(
            f"torch_fl.compile.{module}.platform_profile",
            lambda: pp._KUNLUNXIN_PROFILE,
            raising=False,
        )
    monkeypatch.setattr("torch_fl._build_config.ACCELERATOR", "kunlunxin")
    return pp._KUNLUNXIN_PROFILE


def _cuda_platform_without_cuda_torch(monkeypatch) -> None:
    """Pin the exact shape of the default FlagGems-primary CUDA wheel.

    The recurring bug is a site gated on the *platform* (``is_cuda_like``) when
    the question it is really asking is where the *torch* stands -- the torch
    that wheel links is the CPU-only pip wheel, with no CUDA bindings at all.

    Tests simulate that by deleting ``torch._C._cuda_getDeviceCount``, which is
    the probe ``_torch_has_cuda_kernels()`` reads. That is only reliable when the
    probe is *not* stubbed: on this machine it answers False (CPU torch) but on a
    CUDA CI runner it would answer True. Pin both halves so the test means the
    same thing wherever it runs.
    """
    monkeypatch.delattr(torch._C, "_cuda_getDeviceCount", raising=False)
    monkeypatch.setattr(
        "torch_fl.compile.device_interface._torch_has_cuda_kernels", lambda: False
    )


# --- profile selection ------------------------------------------------------


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    "accelerator,expected",
    [
        ("ascend", pp._ASCEND_PROFILE),
        ("metax", pp._METAX_PROFILE),
        ("dcu", pp._DCU_PROFILE),
        ("musa", pp._MUSA_PROFILE),
        ("gcu", pp._GCU_PROFILE),
        ("cuda", pp._CUDA_PROFILE),
        ("ppu", pp._CUDA_PROFILE),
        ("", pp._CUDA_PROFILE),
    ],
)
def test_profile_selected_by_accelerator(monkeypatch, accelerator, expected):
    """ACCELERATOR picks the profile; unknown values fall back to CUDA."""
    monkeypatch.setattr("torch_fl._build_config.ACCELERATOR", accelerator)
    assert pp.platform_profile() == expected


@pytest.mark.anyplatform
def test_triton_target_and_package_key_differ_on_ascend():
    """The two names are not interchangeable, which is why they are separate.

    Measured on a real 910: the driver reports GPUTarget(backend='npu', ...) and
    AscendBackend.supports_target accepts only "npu", but the package registers
    that backend under the key "ascend". Collapsing the two breaks either the
    toolchain probe or the compile.
    """
    profile = pp._ASCEND_PROFILE
    assert profile.triton_device_type == "npu"
    assert profile.triton_backend_key == "ascend"
    assert profile.triton_device_type != profile.triton_backend_key


@pytest.mark.anyplatform
def test_only_ascend_is_not_cuda_like():
    """is_cuda_like gates every torch.cuda / CUDA-codegen route."""
    assert not pp._ASCEND_PROFILE.is_cuda_like
    assert pp._CUDA_PROFILE.is_cuda_like
    assert pp._METAX_PROFILE.is_cuda_like
    assert pp._DCU_PROFILE.is_cuda_like


@pytest.mark.anyplatform
def test_dcu_compiles_for_hip_on_a_cuda_like_runtime():
    """DCU is the one build that is CUDA-like but must not compile for cuda.

    DTK wraps HIP behind a CUDA ABI, so boxing, streams and device queries all go
    through torch.cuda -- hence is_cuda_like. The Triton side is not CUDA at all:
    FlagTree's HCU backend accepts only ``target.backend == "hip"`` and registers
    itself under the key ``hcu``. Falling back to _CUDA_PROFILE here reports
    ``cuda`` onward and inductor dies with "0 compatible backends for target
    (cuda)"; measured on gfx936, that took 9 of the compile cases in
    tests/integration/test_compile.py down with it.
    """
    profile = pp._DCU_PROFILE
    assert profile.triton_device_type == "hip"
    assert profile.triton_backend_key == "hcu"
    assert profile.is_cuda_like is True
    # Not the CUDA profile: the whole point is that the compiler target differs
    # even though the runtime is shared.
    assert profile.triton_device_type != pp._CUDA_PROFILE.triton_device_type


# --- device interface -------------------------------------------------------


@pytest.mark.anyplatform
def test_hardware_module_avoids_torch_cuda_on_ascend(as_ascend):
    """Ascend has no CUDA runtime, so no hardware query may reach torch.cuda."""
    from torch_fl.compile import device_interface as di

    assert di._hardware_module() is torch.flagos


@pytest.mark.anyplatform
def test_hardware_module_is_torch_cuda_on_cuda_like(as_cuda):
    from torch_fl.compile import device_interface as di

    assert di._hardware_module() is torch.cuda


@pytest.mark.anyplatform
def test_raw_stream_comes_from_acl_registry_on_ascend(as_ascend, monkeypatch):
    """The launch-path stream must be torch_fl's ACL stream, not CUDA's.

    A kernel launched on rt stream 0 is not ordered against the aclnn ops that
    produced its inputs; that silently corrupted results once already (see
    scripts/vendor/patch_triton_ascend.py).
    """
    from torch_fl.compile import device_interface as di

    seen = []
    monkeypatch.setattr(
        "torch_fl.accelerator.ascend.acl_stream.current_acl_raw_stream",
        lambda idx: seen.append(idx) or 0x1234,
    )

    assert di._raw_stream(3) == 0x1234
    assert seen == [3]


@pytest.mark.anyplatform
def test_bf16_support_read_from_flagos_autocast_list(as_ascend, monkeypatch):
    """Ascend reports what torch_fl advertises for autocast, not a hard-coded yes."""
    from torch_fl.compile import device_interface as di

    monkeypatch.setattr(torch.flagos, "get_amp_supported_dtype", lambda: [torch.half])
    assert di.FlagOSDeviceInterface.is_bf16_supported() is False

    monkeypatch.setattr(
        torch.flagos, "get_amp_supported_dtype", lambda: [torch.half, torch.bfloat16]
    )
    assert di.FlagOSDeviceInterface.is_bf16_supported() is True


@pytest.mark.anyplatform
def test_compute_capability_is_soc_name_on_ascend(as_ascend, monkeypatch):
    """`cc` reaches Triton as GPUTarget.arch, which Ascend expects to be a SoC."""
    from torch_fl.compile import device_interface as di

    monkeypatch.setattr(di, "_ascend_arch", lambda: "Ascend910_9382")
    assert di.FlagOSDeviceInterface.get_compute_capability() == "Ascend910_9382"


@pytest.mark.anyplatform
def test_triton_capability_has_no_compute_floor_on_ascend(as_ascend):
    """The CUDA sm_70 floor is meaningless on Ascend; the toolchain decides."""
    from torch_fl.compile import device_interface as di

    assert di.FlagOSDeviceInterface.is_triton_capable() is True


@pytest.mark.anyplatform
def test_exchange_device_uses_flagos_state_on_ascend(as_ascend, monkeypatch):
    from torch_fl.compile import device_interface as di

    monkeypatch.setattr(torch.flagos, "current_device", lambda: 2)
    moved = []
    monkeypatch.setattr(torch.flagos, "set_device", moved.append)

    assert di._exchange_device(5) == 2
    assert moved == [5]
    # A negative index means "no device"; nothing should move.
    moved.clear()
    assert di._exchange_device(-1) == -1
    assert moved == []


@pytest.mark.anyplatform
def test_exchange_device_uses_flagos_state_without_cuda_kernels(as_cuda, monkeypatch):
    """The CUDA-like/CPU-torch wheel must switch the flagos device, not torch.cuda.

    `FlagOSDeviceInterface.exchange_device` is this helper, and the runtime
    launcher calls it from `DeviceGuard.__enter__` before every kernel launch.
    On the default FlagGems-primary wheel `torch.cuda._exchange_device` raises
    "PyTorch was compiled without CUDA support", so the platform gate has to be
    paired with the torch probe.
    """
    from torch_fl.compile import device_interface as di

    _cuda_platform_without_cuda_torch(monkeypatch)
    monkeypatch.setattr(torch.flagos, "current_device", lambda: 2)
    moved = []
    monkeypatch.setattr(torch.flagos, "set_device", moved.append)

    assert di._exchange_device(5) == 2
    assert moved == [5]


@pytest.mark.anyplatform
def test_compute_capability_comes_from_the_driver_without_cuda_kernels(
    as_cuda, monkeypatch
):
    """No `torch.cuda.get_device_capability` on this wheel; the driver has the arch."""
    from torch_fl.compile import device_interface as di

    class _Target:
        arch = "sm_80"

    _cuda_platform_without_cuda_torch(monkeypatch)
    monkeypatch.setattr(di, "_triton_current_target", lambda: _Target())

    assert di.FlagOSDeviceInterface.get_compute_capability() == 80


# --- device properties on a CUDA-like build with no CUDA torch --------------


class _StubProps:
    """The subset of a device-properties record the layout fields read."""

    def __init__(self, **overrides):
        self.multi_processor_count = 108
        self.major = 8
        self.warp_size = 32
        for name, value in overrides.items():
            setattr(self, name, value)


def _stub_flagos_properties(monkeypatch, props):
    """Make `FlagOSDeviceInterface.get_device_properties` answer `props`."""
    from torch_fl.compile import device_interface as di

    monkeypatch.setattr(
        di.FlagOSDeviceInterface, "get_device_properties", lambda device=None: props
    )


@pytest.mark.anyplatform
def test_device_properties_avoid_torch_cuda_on_a_cpu_torch(as_cuda, monkeypatch):
    """A CUDA-like platform over a CPU-only torch must not query torch.cuda.

    The default FlagGems-primary wheel is exactly that shape. Delegating to
    upstream's `DeviceProperties.create` sends the lookup into
    `torch.cuda.get_device_properties`, which on a torch with no CUDA bindings
    calls `_get_device_properties` after a `_lazy_init` that never defines it
    (torch_fl sets `torch.cuda._initialized`, so the guard that would have
    raised "Torch not compiled with CUDA enabled" is skipped) -- inductor then
    dies with `NameError: name '_get_device_properties' is not defined`, which
    is how `test_flex_attention_with_grads` failed on the A100.
    """
    from torch._inductor.runtime.hints import DeviceProperties

    from torch_fl.compile import device_interface as di

    _cuda_platform_without_cuda_torch(monkeypatch)
    _stub_flagos_properties(monkeypatch, _StubProps())
    monkeypatch.setattr(
        di.FlagOSDeviceInterface,
        "get_compute_capability",
        staticmethod(lambda device=None: 80),
    )
    monkeypatch.setattr(di, "_triton_current_target", lambda: None)

    def _torch_cuda_must_not_be_reached(device):
        raise AssertionError("DeviceProperties.create delegated to torch.cuda")

    monkeypatch.setattr(DeviceProperties, "create", _torch_cuda_must_not_be_reached)
    # `_patch_device_properties` keys off the attribute marker, and the stub
    # above replaced the classmethod wholesale.
    monkeypatch.delattr(DeviceProperties.create, "_flagos_patched", raising=False)

    di._patch_device_properties()

    record = DeviceProperties.create(torch.device("flagos", 0))
    assert record.type == "cuda"
    assert record.index == 0
    assert record.multi_processor_count == 108
    assert record.cc == 80
    assert record.warp_size == 32


@pytest.mark.anyplatform
def test_device_properties_keep_the_torch_cuda_path_on_a_cuda_torch(
    as_cuda, monkeypatch
):
    """A CUDA-enabled torch keeps upstream's lookup, as the vendor wheels do."""
    from torch._inductor.runtime.hints import DeviceProperties

    from torch_fl.compile import device_interface as di

    monkeypatch.setattr(torch._C, "_cuda_getDeviceCount", lambda: 1, raising=False)
    _stub_flagos_properties(monkeypatch, _StubProps())

    delegated = []

    def _record(device):
        delegated.append(device)
        raise AssertionError("stop here; the delegation is what is under test")

    monkeypatch.setattr(DeviceProperties, "create", _record)
    monkeypatch.delattr(DeviceProperties.create, "_flagos_patched", raising=False)

    di._patch_device_properties()

    with pytest.raises(AssertionError, match="the delegation is what is under test"):
        DeviceProperties.create(torch.device("flagos", 0))
    # Translated to the cuda device, which is where torch.cuda looks up the
    # properties of the same physical GPU.
    assert [d.type for d in delegated] == ["cuda"]


@pytest.mark.anyplatform
def test_triton_driver_props_reads_arch_and_warp_size(monkeypatch):
    """The layout fields fall back to the Triton driver, which is authoritative."""

    class _Target:
        arch = "sm_80"
        warp_size = 32

    from torch_fl.compile import device_interface as di

    monkeypatch.setattr(di, "_triton_current_target", lambda: _Target())

    layout = di._triton_driver_props(_StubProps(warp_size=None))
    assert layout["cc"] == 80
    assert layout["warp_size"] == 32
    assert layout["multi_processor_count"] == 108


# --- device registry --------------------------------------------------------


def _capture_registrations(di, monkeypatch):
    """Record what `register_flagos_device_interface` registers.

    Only the registry decision matters to the tests below; the other patch steps
    rewrite global inductor state, so no-op them and intercept the one
    registration call that is left.
    """
    for name in (
        "_register_gpu_type",
        "_patch_device_properties",
        "_patch_inductor_benchmark_device",
        "_repair_stock_cuda_interface",
        "_patch_joint_graph_pattern_init",
        "_patch_benchmarker",
        "_prime_has_triton",
    ):
        monkeypatch.setattr(di, name, lambda: None, raising=False)
    # Skip the has_triton override tail; it does not register anything.
    monkeypatch.setattr(di, "is_native_accelerator", lambda: False)

    registered = {}

    import torch._dynamo.device_interface as dyn_di

    monkeypatch.setattr(
        dyn_di,
        "register_interface_for_device",
        lambda device, interface: registered.__setitem__(device, interface),
    )
    return registered


@pytest.mark.anyplatform
def test_native_accelerator_claims_the_reported_cuda_name(as_kunlunxin, monkeypatch):
    """The reported device name must map to the flagos interface.

    FlagTree's XPU backend reports the Kunlunxin target as "cuda", so inductor's
    *runtime* launcher (`triton_heuristics.get_device_interface`, which looks up
    ``device_props.type``) lands on the stock `CudaInterface` -- whose
    ``exchange_device`` is ``torch.cuda._exchange_device``, raising "PyTorch was
    compiled without CUDA support" on a build with no CUDA runtime. The reported
    name has to be claimed for `FlagOSDeviceInterface` instead.
    """
    from torch_fl.compile import device_interface as di

    registered = _capture_registrations(di, monkeypatch)
    di.register_flagos_device_interface()

    assert registered[di.DEVICE_TYPE] is di.FlagOSDeviceInterface
    assert registered["cuda"] is di.FlagOSDeviceInterface


@pytest.mark.anyplatform
def test_cuda_like_build_leaves_the_cuda_name_alone(as_cuda, monkeypatch):
    """On a CUDA-*enabled* torch the stock CudaInterface is the correct one.

    "CUDA build" means a torch that carries the CUDA bindings, not merely the
    CUDA platform: that is what makes the members CudaInterface binds off
    ``torch.cuda`` answerable. The platform-with-CPU-torch shape -- the default
    FlagGems-primary wheel -- is the opposite case, covered by
    `test_cuda_wheel_without_cuda_kernels_claims_the_reported_cuda_name`.
    """
    from torch_fl.compile import device_interface as di

    monkeypatch.setattr(di, "_torch_has_cuda_kernels", lambda: True)
    registered = _capture_registrations(di, monkeypatch)
    di.register_flagos_device_interface()

    assert registered[di.DEVICE_TYPE] is di.FlagOSDeviceInterface
    assert "cuda" not in registered


@pytest.mark.anyplatform
def test_cuda_wheel_without_cuda_kernels_claims_the_reported_cuda_name(
    as_cuda, monkeypatch
):
    """The default FlagGems-primary CUDA wheel has no CUDA bindings in its torch.

    It reports the Triton target as "cuda" (NVIDIA's backend) exactly as a CUDA
    build does, so the launcher lands on the stock `CudaInterface` -- but that
    class was bound from a `torch.cuda` that cannot answer, and `_make_launchers`
    calls `exchange_device` then `synchronize(current_device())` before every
    kernel launch. On this build both must resolve to `FlagOSDeviceInterface`,
    which reads the same physical device through torch.flagos.
    """
    from torch_fl.compile import device_interface as di

    _cuda_platform_without_cuda_torch(monkeypatch)
    registered = _capture_registrations(di, monkeypatch)
    di.register_flagos_device_interface()

    assert registered[di.DEVICE_TYPE] is di.FlagOSDeviceInterface
    assert registered["cuda"] is di.FlagOSDeviceInterface


@pytest.mark.anyplatform
def test_stock_cuda_interface_leaves_exchange_device_on_cuda(as_cuda, monkeypatch):
    """On a CUDA build the stock `exchange_device` is correct and stays untouched.

    The repair splits: `exchange_device` dispatches to a real CUDA runtime, so the
    stock binding must be left alone. `get_raw_stream` is bound at
    `torch._dynamo.device_interface` import time from

        get_cuda_stream = _cuda_getCurrentRawStream if torch.cuda._is_compiled() else None

    which the flagos CPU wheel (plus torch_fl's own shim, installed later) misses,
    so it is restored on every build -- gating it behind `is_cuda_like` is exactly
    the regression from run 38354801090, where the CUDA lane died with
    "'NoneType' object is not callable" at `get_raw_stream`.

    "CUDA build" here means a torch that carries the CUDA bindings: the default
    FlagGems-primary wheel is CUDA-*like* without them, and is covered by
    `test_stock_cuda_interface_rebinds_exchange_device_without_cuda_kernels`.
    """
    import torch

    from torch._dynamo import device_interface as dyn_di

    from torch_fl.compile import device_interface as di

    monkeypatch.setattr(torch._C, "_cuda_getDeviceCount", lambda: 1, raising=False)

    before_exchange = dyn_di.CudaInterface.exchange_device
    before_maybe = dyn_di.CudaInterface.maybe_exchange_device
    before_raw = dyn_di.CudaInterface.get_raw_stream
    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)

    di._repair_stock_cuda_interface()

    assert dyn_di.CudaInterface.exchange_device is before_exchange
    assert dyn_di.CudaInterface.maybe_exchange_device is before_maybe
    if getter is not None and before_raw is None:
        assert dyn_di.CudaInterface.get_raw_stream is getter
    else:
        assert dyn_di.CudaInterface.get_raw_stream is before_raw


@pytest.mark.anyplatform
def test_stock_cuda_interface_rebinds_exchange_device_without_cuda_kernels(
    as_cuda, monkeypatch
):
    """A CUDA-like platform over a CPU-only torch needs the same repair as Kunlunxin.

    `_patch_device_properties` reports the flagos device as "cuda" there too, so
    the runtime launcher resolves the stock `CudaInterface` -- whose
    `exchange_device` is `torch.cuda._exchange_device` and raises "PyTorch was
    compiled without CUDA support" from `DeviceGuard.__enter__` before every
    launch. `test_flex_attention_with_grads` on the A100 reached exactly that
    call after codegen succeeded.
    """
    from torch._dynamo import device_interface as dyn_di

    from torch_fl.compile import device_interface as di

    _cuda_platform_without_cuda_torch(monkeypatch)

    saved = {
        name: getattr(dyn_di.CudaInterface, name)
        for name in ("exchange_device", "maybe_exchange_device", "get_raw_stream")
    }
    try:
        di._repair_stock_cuda_interface()
        assert dyn_di.CudaInterface.exchange_device is di._exchange_device
        assert dyn_di.CudaInterface.maybe_exchange_device is di._exchange_device
    finally:
        for name, value in saved.items():
            setattr(dyn_di.CudaInterface, name, value)


@pytest.mark.anyplatform
def test_stock_cuda_interface_exchange_device_rebound_on_native(as_kunlunxin):
    """The runtime launcher resolves the stock class, so its entry point must work."""
    from torch._dynamo import device_interface as dyn_di

    from torch_fl.compile import device_interface as di

    saved = {
        name: getattr(dyn_di.CudaInterface, name)
        for name in ("exchange_device", "maybe_exchange_device", "get_raw_stream")
    }
    try:
        di._repair_stock_cuda_interface()
        assert dyn_di.CudaInterface.exchange_device is di._exchange_device
        assert dyn_di.CudaInterface.maybe_exchange_device is di._exchange_device
    finally:
        for name, value in saved.items():
            setattr(dyn_di.CudaInterface, name, value)


# --- autotune benchmarking --------------------------------------------------


@pytest.mark.anyplatform
def test_benchmarker_patch_translates_triton_device_name(as_ascend, monkeypatch):
    """`bench()` passes the *Triton* backend name, which is not a torch device.

    CachingAutotuner.bench calls benchmarker.benchmark(device=device_props.type),
    and on Ascend that string is "npu" -- torch.device("npu") raises. The patch
    rewrites it to flagos, which is the device the tensors are actually on.
    """
    from torch._inductor.runtime import benchmarking

    from torch_fl.compile import device_interface as di

    calls = []
    monkeypatch.setattr(
        benchmarking.benchmarker,
        "benchmark",
        lambda **kwargs: calls.append(kwargs) or 1.0,
        raising=False,
    )
    # raising=False: the marker only exists once _patch_benchmarker has run on a
    # non-CUDA-like build, and on a CUDA-like one the real import-time call
    # returns before setting it. Without this the test fails on exactly the
    # platform where the patch is a no-op.
    monkeypatch.setattr(
        benchmarking.benchmarker, "_flagos_patched", False, raising=False
    )

    di._patch_benchmarker()
    benchmarking.benchmarker.benchmark(fn=lambda: None, device="npu")
    benchmarking.benchmarker.benchmark(fn=lambda: None, device="cpu")

    assert [call["device"] for call in calls] == ["flagos", "cpu"]


@pytest.mark.anyplatform
def test_class_level_benchmark_device_rewrite_avoids_cuda_on_ascend(
    as_ascend, monkeypatch
):
    """The Benchmarker *class* rewrite must not name cuda on a CUDA-less build.

    That patch exists for MetaX, where mapping the triton name to "cuda" is
    correct because flagos and cuda are the same GPU. Ascend has no CUDA runtime,
    so the same substitution would hand a device that cannot be benchmarked on;
    the replacement has to be flagos.
    """
    from torch._inductor.runtime import benchmarking

    from torch_fl.compile import device_interface as di

    monkeypatch.setattr(di, "_triton_backend", lambda: ("npu", "ascend"))

    calls = []
    original = benchmarking.Benchmarker.benchmark

    def record(self, fn, fn_args=None, fn_kwargs=None, device=None, **kwargs):
        calls.append(device)
        return 0.0

    monkeypatch.setattr(benchmarking.Benchmarker, "benchmark", record)

    di._patch_inductor_benchmark_device()
    try:
        benchmarking.Benchmarker().benchmark(lambda: None, device="npu")
    finally:
        benchmarking.Benchmarker.benchmark = original

    assert calls == ["flagos"]


@pytest.mark.anyplatform
def test_benchmarker_patch_drops_cuda_event_timing(as_ascend, monkeypatch):
    """InductorBenchmarker times with torch.cuda.Event and reads L2_cache_size.

    Neither exists on a CPU torch wheel (Event is a dummy base class that raises
    on construction), yet it is what gets selected: the choice is made at import
    time from `torch.cuda.is_available()`, which torch_fl aliases to the flagos
    device count. Fall back to TritonBenchmarker, which times through triton's
    own do_bench and therefore through torch.flagos events.
    """
    from torch._inductor.runtime import benchmarking

    from torch_fl.compile import device_interface as di

    obj = benchmarking.InductorBenchmarker()
    monkeypatch.setattr(benchmarking, "benchmarker", obj)

    di._patch_benchmarker()

    assert obj.benchmark_gpu.__func__ is benchmarking.TritonBenchmarker.benchmark_gpu


@pytest.mark.anyplatform
def test_benchmarker_patch_is_idempotent(as_ascend, monkeypatch):
    from torch._inductor.runtime import benchmarking

    from torch_fl.compile import device_interface as di

    obj = benchmarking.TritonBenchmarker()
    monkeypatch.setattr(benchmarking, "benchmarker", obj)

    di._patch_benchmarker()
    once = obj.benchmark
    di._patch_benchmarker()

    assert obj.benchmark is once


@pytest.mark.anyplatform
def test_benchmarker_untouched_on_cuda_like(as_cuda, monkeypatch):
    from torch._inductor.runtime import benchmarking

    from torch_fl.compile import device_interface as di

    # A CUDA-enabled torch, which is what "CUDA-like and nothing to fix" means:
    # InductorBenchmarker can allocate its own L2-flush buffer there.
    monkeypatch.setattr(torch._C, "_cuda_getDeviceCount", lambda: 1, raising=False)
    obj = benchmarking.TritonBenchmarker()
    monkeypatch.setattr(benchmarking, "benchmarker", obj)

    di._patch_benchmarker()
    assert not getattr(obj, "_flagos_patched", False)


@pytest.mark.anyplatform
def test_benchmarker_patch_reaches_the_default_cuda_wheel(as_cuda, monkeypatch):
    """A CUDA-like build without CUDA kernels needs the same fallback as Ascend.

    That is the default FlagGems-primary wheel: the singleton is still an
    `InductorBenchmarker` (selected from `torch.cuda.is_available()`, which
    _cuda_compat.py answers True), and its `benchmark_gpu` flushes the L2 cache
    with `torch.empty(..., device="cuda")` -- a kernel this wheel does not have.
    """
    from torch._inductor.runtime import benchmarking

    from torch_fl.compile import device_interface as di

    _cuda_platform_without_cuda_torch(monkeypatch)
    obj = benchmarking.InductorBenchmarker()
    monkeypatch.setattr(benchmarking, "benchmarker", obj)

    di._patch_benchmarker()

    assert obj.benchmark_gpu.__func__ is benchmarking.TritonBenchmarker.benchmark_gpu


# --- joint_graph pattern priming --------------------------------------------


class _RecordedCacheInfo:
    def __init__(self, currsize):
        self.currsize = currsize


class _RecordingLazyInit:
    """Stand-in for `joint_graph.lazy_init` that records the probe it saw."""

    def __init__(self, currsize=0):
        self.calls = 0
        self.probe = None
        self._currsize = currsize

    def cache_info(self):
        return _RecordedCacheInfo(self._currsize)

    def __call__(self):
        self.calls += 1
        self.probe = torch.cuda.is_available()


def _stand_in_for_joint_graph_lazy_init(monkeypatch, currsize=0):
    joint_graph = pytest.importorskip("torch._inductor.fx_passes.joint_graph")
    recorder = _RecordingLazyInit(currsize)
    monkeypatch.setattr(joint_graph, "lazy_init", recorder)
    return recorder


@pytest.mark.anyplatform
def test_joint_graph_patterns_keep_the_cuda_trace_on_a_cuda_torch(as_cuda, monkeypatch):
    """A CUDA-enabled torch traces the replacement patterns on cuda, as upstream."""
    from torch_fl.compile import device_interface as di

    monkeypatch.setattr(torch._C, "_cuda_getDeviceCount", lambda: 1, raising=False)
    recorder = _stand_in_for_joint_graph_lazy_init(monkeypatch)

    di._patch_joint_graph_pattern_init()

    assert recorder.calls == 0


@pytest.mark.anyplatform
def test_joint_graph_patterns_primed_off_cuda_without_cuda_kernels(
    as_cuda, monkeypatch
):
    """The CUDA platform on the default wheel traces them on cpu instead.

    Both conditions hold at once there, which is why the platform profile cannot
    decide it: the platform is CUDA (the flagos device is the same physical GPU,
    so `is_cuda_like`), and the torch is the CPU-only pip wheel, so `_pad_mm_init`
    dies with `NotImplementedError: Could not run 'aten::empty.memory_format' with
    arguments from the 'CUDA' backend`. The probe answers False for the one call
    that picks the trace device, then goes back to reporting the shim's device.
    """
    from torch_fl.compile import device_interface as di

    _cuda_platform_without_cuda_torch(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    recorder = _stand_in_for_joint_graph_lazy_init(monkeypatch)

    di._patch_joint_graph_pattern_init()

    assert recorder.calls == 1
    assert recorder.probe is False
    assert torch.cuda.is_available() is True


@pytest.mark.anyplatform
def test_joint_graph_pattern_priming_is_one_shot(as_cuda, monkeypatch):
    """`lazy_init` is `functools.cache`d; an already-primed table is left alone."""
    from torch_fl.compile import device_interface as di

    _cuda_platform_without_cuda_torch(monkeypatch)
    recorder = _stand_in_for_joint_graph_lazy_init(monkeypatch, currsize=1)

    di._patch_joint_graph_pattern_init()

    assert recorder.calls == 0


# --- has_triton -------------------------------------------------------------


@pytest.mark.anyplatform
def test_has_triton_priming_restores_the_device_table(as_ascend, monkeypatch):
    """Priming must not leave the borrowed cuda/xpu rows pointing at flagos."""
    import torch._dynamo.device_interface as dyn_di
    from torch.utils import _triton

    from torch_fl.compile import device_interface as di

    # A fresh cache over the same function, so the priming path is really taken
    # (it returns early when something is already memoized) without discarding
    # the answer the rest of the process is using. `__wrapped__` is only there
    # while has_triton is still the functools.cache'd original: on a native
    # accelerator host `import torch_fl` has already swapped in the plain
    # has_native_triton override, and reaching through it raised AttributeError
    # before the test could assert anything.
    original = getattr(_triton.has_triton, "__wrapped__", _triton.has_triton)
    monkeypatch.setattr(_triton, "has_triton", functools.cache(original))
    dyn_di.get_interface_for_device("cuda")
    before = dict(dyn_di.device_interfaces)

    di._prime_has_triton()

    assert dyn_di.device_interfaces == before
    assert _triton.has_triton.cache_info().currsize == 1


@pytest.mark.anyplatform
def test_has_triton_priming_tolerates_the_native_override(as_musa, monkeypatch):
    """Priming must not assume has_triton is still the functools.cache'd original.

    On a native accelerator the tail of register_flagos_device_interface()
    replaces `_triton.has_triton` with a plain function. That function has no
    `cache_info`, so reading it raises AttributeError -- and because the compile
    backend re-registers before every compile_fx, this is what every
    torch.compile after the first hits. It surfaced as
    `BackendCompilerFailed: 'function' object has no attribute 'cache_info'`,
    i.e. torch.compile broken outright on MUSA and GCU.
    """
    from torch.utils import _triton

    from torch_fl.compile import device_interface as di

    def has_native_triton() -> bool:
        return True

    monkeypatch.setattr(_triton, "has_triton", has_native_triton)

    di._prime_has_triton()

    assert _triton.has_triton is has_native_triton


# --- codegen registration ---------------------------------------------------


@pytest.mark.anyplatform
def test_ascend_codegen_classes_avoid_cuda_wrappers(as_ascend):
    """Ascend takes plain TritonScheduling and no C++ wrapper.

    CppWrapperGpu emits CUDA-runtime C++ and CUDACombinedScheduling delegates to
    CUTLASS/ROCm/CuteDSL template paths, none of which exist here. Registering
    None is an explicit "unsupported", not a silent wrong answer.
    """
    from torch._inductor.codegen.triton import TritonScheduling
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen

    from torch_fl.compile import inductor_codegen as ic

    scheduling, wrapper, cpp_wrapper, _ = ic._codegen_classes()

    assert scheduling is TritonScheduling
    assert wrapper is PythonWrapperCodegen
    assert cpp_wrapper is None


@pytest.mark.anyplatform
def test_cuda_codegen_classes_keep_cpp_wrapper(as_cuda):
    from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu

    from torch_fl.compile import inductor_codegen as ic

    _, _, cpp_wrapper, _ = ic._codegen_classes()
    assert cpp_wrapper is CppWrapperGpu


@pytest.mark.anyplatform
def test_ascend_device_op_overrides_emit_no_cuda(as_ascend):
    """Generated snippets must name flagos/ACL APIs, never CUDA ones."""
    from torch_fl.compile import inductor_codegen as ic

    overrides = ic._device_op_overrides()
    assert isinstance(overrides, ic.FlagOSAscendDeviceOpOverrides)

    snippets = [
        overrides.set_device(0),
        overrides.device_guard(0),
        overrides.synchronize(),
        overrides.import_get_raw_stream_as("get_raw_stream"),
    ]
    for snippet in snippets:
        assert "cuda" not in snippet.lower(), snippet
    assert "current_acl_raw_stream" in snippets[-1]


@pytest.mark.anyplatform
def test_musa_device_op_overrides_emit_the_musa_raw_stream(as_musa):
    """A non-CUDA build that is not Ascend must not get the ACL snippets.

    MUSA and GCU are `is_cuda_like=False` like Ascend, but their raw-stream
    getters are branches of FlagOSDeviceOpOverrides, so selecting on that boolean
    emitted `current_acl_raw_stream` into generated MUSA code -- which dies at
    call time on `libflagos.so: undefined symbol: GetCurrentStream`.
    """
    from torch_fl.compile import inductor_codegen as ic

    overrides = ic._device_op_overrides()
    assert isinstance(overrides, ic.FlagOSDeviceOpOverrides)
    assert not isinstance(overrides, ic.FlagOSAscendDeviceOpOverrides)

    snippet = overrides.import_get_raw_stream_as("get_raw_stream")
    assert "current_acl_raw_stream" not in snippet
    assert "get_musa_current_raw_stream as get_raw_stream" in snippet


@pytest.mark.anyplatform
def test_ascend_device_op_overrides_reject_cpp_wrapper(as_ascend):
    """The C++ wrapper members must fail loudly rather than emit CANN-invalid C++."""
    from torch_fl.compile import inductor_codegen as ic

    overrides = ic._device_op_overrides()
    with pytest.raises(NotImplementedError):
        overrides.cpp_device_guard()


@pytest.mark.anyplatform
def test_cuda_device_op_overrides_use_flagos_device_state(as_cuda):
    """CUDA-like builds keep the CUDA snippets, but move the *flagos* device.

    flagos.set_device also moves the CUDA current device, so going through
    torch.flagos keeps the two in sync; emitting torch.cuda.set_device would
    desync them.
    """
    from torch_fl.compile import inductor_codegen as ic

    overrides = ic._device_op_overrides()
    assert isinstance(overrides, ic.FlagOSDeviceOpOverrides)
    assert overrides.set_device(1) == "torch.flagos.set_device(1)"
    assert overrides.device_guard(1) == "torch.flagos.device(1)"
    assert "_cuda_getCurrentRawStream" in overrides.import_get_raw_stream_as("s")


@pytest.mark.anyplatform
def test_publish_on_device_module_is_a_noop_without_cpp_wrapper(as_ascend, monkeypatch):
    """Inductor's PrivateUse1 hook needs all four classes; Ascend has three.

    Publishing a partial set would make init_backend_registration fail on the
    missing name, so this path stays quiet and register_flagos_codegen is the
    only registration route on Ascend.

    The names are deleted first because a CUDA-like build publishes them at
    import time (torch_fl/__init__.py), and that write is a plain setattr that
    monkeypatch cannot roll back on its own. Deleting them makes the assertion
    test what this function does rather than what the process already did.
    """
    from torch_fl.compile import inductor_codegen as ic

    for name in (
        "Scheduling",
        "PythonWrapperCodegen",
        "CppWrapperCodegen",
        "WrapperFxCodegen",
    ):
        monkeypatch.delattr(torch.flagos, name, raising=False)

    ic.publish_codegen_on_device_module()
    assert not hasattr(torch.flagos, "Scheduling")


@pytest.mark.anyplatform
def test_register_codegen_is_idempotent(as_ascend):
    """Called before every compile_fx, so a second call must not re-register."""
    from torch._inductor.codegen.common import (
        device_op_overrides_dict,
        get_scheduling_for_device,
    )

    from torch_fl.compile import inductor_codegen as ic

    ic.register_flagos_codegen()
    scheduling = get_scheduling_for_device(ic.DEVICE_TYPE)
    overrides = device_op_overrides_dict[ic.DEVICE_TYPE]

    ic.register_flagos_codegen()

    assert get_scheduling_for_device(ic.DEVICE_TYPE) is scheduling
    assert device_op_overrides_dict[ic.DEVICE_TYPE] is overrides


@pytest.mark.anyplatform
def test_ub_overflow_is_recognized_as_a_resource_limit():
    """The vendor's "ub overflow" wording must parse into required/available.

    This is what lets inductor drop an oversized autotune config instead of
    failing the compile, so the parse is the whole mechanism.
    """
    from torch_fl.compile import triton_resource_limits as trl

    exc = RuntimeError(
        "error: ub overflow, requires 4194816 bits while 1572864 bits available!"
    )
    assert trl._resource_limit(exc) == ("ub", 4194816, 1572864)


@pytest.mark.anyplatform
def test_unrelated_compiler_errors_keep_their_type():
    """Only capacity errors are translated; everything else must propagate as is."""
    from torch_fl.compile import triton_resource_limits as trl

    assert (
        trl._resource_limit(RuntimeError("Failed to run BiShengHIR pipeline")) is None
    )
    assert trl._resource_limit(ValueError("unsupported operation")) is None


@pytest.mark.anyplatform
def test_resource_limit_patch_skipped_on_cuda(as_cuda):
    """CUDA-like builds get Triton's own OutOfResources and need no translation."""
    triton = pytest.importorskip("triton")

    from torch_fl.compile import triton_resource_limits as trl

    before = triton.compile
    trl.patch_triton_resource_limit_errors()

    assert triton.compile is before


@pytest.mark.anyplatform
def test_libdevice_patch_skipped_on_cuda(as_cuda):
    """NVIDIA's Triton backend fills its own module map; leave it alone."""
    triton = pytest.importorskip("triton")

    from torch_fl.compile import triton_libdevice as tld

    backend = triton.backends.backends.get("nvidia")
    if backend is None:
        pytest.skip("no nvidia triton backend in this env")

    before = backend.compiler.get_module_map
    tld.patch_triton_libdevice_module_map()

    assert backend.compiler.get_module_map is before


@pytest.mark.anyplatform
def test_byte_load_workaround_skipped_on_cuda(as_cuda, monkeypatch):
    """The masked byte-load miscompile is Ascend's; CUDA signatures must not move."""
    triton = pytest.importorskip("triton")

    monkeypatch.setattr(
        "torch_fl.compile.triton_byte_loads.platform_profile",
        lambda: pp._CUDA_PROFILE,
    )

    from torch._inductor.codegen import triton_utils

    from torch_fl.compile import triton_byte_loads as tbl

    before_signature = triton_utils.signature_of
    before_compile = triton.compile
    tbl.patch_triton_byte_load_workarounds()

    assert triton_utils.signature_of is before_signature
    assert triton.compile is before_compile


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_repair_stock_cuda_interface_repoints_a_stale_device_properties(
    as_cuda, monkeypatch
):
    """A CudaInterface bound before torch_fl's patch must be re-pointed.

    `DeviceInterface` snapshots `torch.cuda.get_device_properties` at import
    time, so importing `torch._dynamo.device_interface` before torch_fl freezes
    the stock CPU-wheel getter -- the one that calls `_lazy_init()` and then an
    undefined `_get_device_properties`. `_patch_device_properties` rewrites a
    flagos device to `cuda`, so `DeviceProperties.create` resolves exactly that
    class, and flex_attention's lowering dies with the NameError (measured on
    metax, flagos-ai/Torch-FL). The repair is in the unconditional block, so it
    runs for a CUDA-like build too.
    """
    import torch._dynamo.device_interface as dyn_di

    from torch_fl.compile import device_interface as di

    def stale_cuda_getter(device=None):
        raise NotImplementedError("stock torch.cuda.get_device_properties")

    def vendor_getter(device=None):
        return "vendor-props"

    monkeypatch.setattr(torch.cuda, "get_device_properties", vendor_getter)
    monkeypatch.setattr(
        dyn_di.CudaInterface, "get_device_properties", staticmethod(stale_cuda_getter)
    )

    di._repair_stock_cuda_interface()

    assert dyn_di.CudaInterface.get_device_properties is vendor_getter


@pytest.mark.anyplatform
def test_repair_stock_cuda_interface_leaves_a_live_device_properties(
    as_cuda, monkeypatch
):
    """An already-correct binding is left alone.

    On a torch whose cuda module is untouched the class attribute and the module
    attribute are the same object, so the repair is a no-op and the CUDA-like
    hardware proxy is unchanged.
    """
    import torch._dynamo.device_interface as dyn_di

    from torch_fl.compile import device_interface as di

    def getter(device=None):
        return "props"

    monkeypatch.setattr(torch.cuda, "get_device_properties", getter)
    monkeypatch.setattr(
        dyn_di.CudaInterface, "get_device_properties", staticmethod(getter)
    )

    di._repair_stock_cuda_interface()

    assert dyn_di.CudaInterface.get_device_properties is getter


# --- import-time compile-thread pin -----------------------------------------

_COMPILE_THREADS_ENV = "TORCHINDUCTOR_COMPILE_THREADS"


def _run_threads_pin(monkeypatch, *, backend, env=None, threads=32):
    """Point the pin at a vendor backend and a fresh inductor config, then run it."""
    from torch._inductor import config as inductor_config

    from torch_fl.compile import inductor_backend

    if env is None:
        monkeypatch.delenv(_COMPILE_THREADS_ENV, raising=False)
    else:
        monkeypatch.setenv(_COMPILE_THREADS_ENV, env)
    monkeypatch.setattr(
        "torch_fl.compile.flagtree_shim.flagtree_backend", lambda: backend
    )
    monkeypatch.setattr(inductor_config, "compile_threads", threads)

    applied = inductor_backend.pin_compile_threads()
    return applied, inductor_config.compile_threads


@pytest.mark.anyplatform
def test_pin_compile_threads_on_musa(as_musa, monkeypatch):
    """The pin must reach `backend="inductor"`, not only `backend="flagos"`.

    transformers' `CompiledFlexAttention` (and much ecosystem code) compiles with
    the stock backend name, so `_patch_vendor_flagtree_compile_workers` -- which
    only runs inside `flagos_compile_backend` -- never sees it. Ordered
    compilation then spawns `python -m torch._inductor.compile_worker`, which
    imports torch+triton but not torch_fl, and on MUSA the FlagTree/`mthreads`
    driver finds no active backend there ("Could not find an active GPU
    backend"). Pinning the *config* value at import is what makes the default
    backend serial too.
    """
    applied, threads = _run_threads_pin(monkeypatch, backend="mthreads")
    assert applied is True
    assert threads == 1


@pytest.mark.anyplatform
def test_pin_compile_threads_on_cpu_torch_cuda_platform(as_cuda, monkeypatch):
    """The default FlagGems-primary CUDA wheel must pin too.

    That wheel links the CPU-only pip torch, so `torch.cuda.is_available()` is
    torch_fl's own shim. The compile worker imports torch+triton but never
    torch_fl, so the nvidia Triton driver there finds no active backend and a
    *cold* Triton cache dies in `set_driver_to_gpu()` -- measured on A100 as
    `test_flex_attention_with_grads`, which passes once the cache is warm. The
    load-bearing input is the torch, not the platform, so the pin keys on
    `_torch_has_cuda_kernels()`.
    """
    monkeypatch.setattr(
        "torch_fl.compile.device_interface._torch_has_cuda_kernels", lambda: False
    )
    applied, threads = _run_threads_pin(monkeypatch, backend="cuda")
    assert applied is True
    assert threads == 1


@pytest.mark.anyplatform
def test_pin_compile_threads_leaves_a_cuda_kernel_build_alone(as_cuda, monkeypatch):
    """A cuda platform whose torch has CUDA kernels keeps upstream's parallelism.

    Its worker links the nvidia Triton backend and finds its driver without
    torch_fl, so there is nothing to work around and serializing would only cost
    compile time.
    """
    monkeypatch.setattr(
        "torch_fl.compile.device_interface._torch_has_cuda_kernels", lambda: True
    )
    applied, threads = _run_threads_pin(monkeypatch, backend="cuda")
    assert applied is False
    assert threads == 32


@pytest.mark.anyplatform
def test_pin_compile_threads_respects_the_env_override(as_musa, monkeypatch):
    """`TORCHINDUCTOR_COMPILE_THREADS` stays authoritative."""
    applied, threads = _run_threads_pin(
        monkeypatch, backend="mthreads", env="8", threads=8
    )
    assert applied is False
    assert threads == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
