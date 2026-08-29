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
    scripts/patch_triton_ascend.py).
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
    monkeypatch.setattr(benchmarking.benchmarker, "_flagos_patched", False)

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

    obj = benchmarking.TritonBenchmarker()
    monkeypatch.setattr(benchmarking, "benchmarker", obj)

    di._patch_benchmarker()
    assert not getattr(obj, "_flagos_patched", False)


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
def test_publish_on_device_module_is_a_noop_without_cpp_wrapper(as_ascend):
    """Inductor's PrivateUse1 hook needs all four classes; Ascend has three.

    Publishing a partial set would make init_backend_registration fail on the
    missing name, so this path stays quiet and register_flagos_codegen is the
    only registration route on Ascend.
    """
    from torch_fl.compile import inductor_codegen as ic

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
