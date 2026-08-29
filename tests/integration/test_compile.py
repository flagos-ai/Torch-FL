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

"""
Integration tests for torch.compile on flagos device.

Tests basic compilation, fusion gains, and FlagTree detection.
"""

import importlib.util
import os
import sys

import pytest
import torch_fl
import torch

from torch_fl._build_config import ACCELERATOR


# Skip all tests if torch.compile not available (torch < 2.0)
try:
    import torch._dynamo

    HAS_COMPILE = True
except ImportError:
    HAS_COMPILE = False

# Inductor needs a Triton to generate kernels with. The GCU CI image ships
# neither stock Triton nor Enflame's triton_gcu, so the compile tests have to
# skip there rather than fail with TritonMissing.
HAS_TRITON = importlib.util.find_spec("triton") is not None

pytestmark = pytest.mark.skipif(
    not HAS_COMPILE, reason="torch.compile not available (torch < 2.0)"
)

# flagos tensors report either name depending on how the device was spelled.
FLAGOS_DEVICE_TYPES = ("privateuseone", "flagos")


def assert_on_flagos(tensor, what="output"):
    """The graph is compiled *on* flagos, so results must come back on flagos.

    A cuda round trip would both cost a copy per call and produce stream-less
    autograd nodes (see torch_fl/compile/inductor_backend.py), so this is a
    load-bearing assertion, not a smoke check. Native MUSA kernels are
    asynchronous, so drain the shared default stream before comparing results.
    """
    assert tensor.device.type in FLAGOS_DEVICE_TYPES, (
        f"{what} landed on {tensor.device}, expected flagos"
    )
    if tensor.device.type in FLAGOS_DEVICE_TYPES and torch_fl.flagos.is_available():
        torch_fl.flagos.synchronize()


@pytest.fixture
def device():
    """Flagos device for testing."""
    if torch_fl.flagos.device_count() == 0:
        pytest.skip("No flagos devices available")
    return "flagos:0"


class SimpleModel(torch.nn.Module):
    """Simple model with fusible ops."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        x = x * 2.0
        x = x + 1.0
        return x


class MatMulModel(torch.nn.Module):
    """Model with matrix multiplications."""

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        c = torch.mm(a, b)
        d = torch.mm(c, b.t())
        return d + 1.0


def test_compile_backend_registered():
    """Test that 'flagos' backend is registered with dynamo."""
    import torch._dynamo

    # Check backend is in registry
    backends = torch._dynamo.list_backends()
    assert "flagos" in backends, (
        f"'flagos' backend not registered. Available: {backends}"
    )


def test_basic_compile(device):
    """Test basic torch.compile with flagos backend."""
    if not HAS_TRITON:
        pytest.skip("Triton required for compilation")

    model = SimpleModel().to(device)
    x = torch.randn(32, 128, device=device)

    # Compile with flagos backend
    compiled_model = torch.compile(model, backend="flagos")

    # Run compiled model
    output = compiled_model(x)

    # Verify output shape and device. The graph is compiled *on* flagos (no
    # cuda round trip), so the result must come back on flagos.
    assert output.shape == (32, 128)
    assert_on_flagos(output)

    # Compare with eager mode
    eager_output = model(x)
    torch.testing.assert_close(output, eager_output, rtol=1e-4, atol=1e-4)


def test_compile_vs_eager_correctness(device):
    """Test numerical correctness of compiled vs eager execution."""
    if not HAS_TRITON:
        pytest.skip("Triton required for compilation")

    model = MatMulModel().to(device)
    a = torch.randn(64, 64, device=device)
    b = torch.randn(64, 64, device=device)

    # Eager mode
    eager_output = model(a, b)

    # Compiled mode
    compiled_model = torch.compile(model, backend="flagos")
    compiled_output = compiled_model(a, b)

    assert_on_flagos(compiled_output)

    # Should be numerically identical (or very close)
    torch.testing.assert_close(compiled_output, eager_output, rtol=1e-4, atol=1e-4)


def test_compile_with_max_autotune(device):
    """Test torch.compile with max-autotune mode."""
    if not HAS_TRITON:
        pytest.skip("Triton required for compilation")

    model = SimpleModel().to(device)
    x = torch.randn(32, 128, device=device)

    # Compile with max-autotune (aggressive fusion)
    compiled_model = torch.compile(model, backend="flagos", mode="max-autotune")

    output = compiled_model(x)
    eager_output = model(x)

    assert_on_flagos(output)
    torch.testing.assert_close(output, eager_output, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    torch_fl._build_accelerator() != "metax",
    reason="MetaX compatibility event regression",
)
def test_metax_inductor_event_uses_real_stream(device):
    """Inductor timing events must not record on the raw-stream-only shim."""
    assert torch.cuda.Event is torch.flagos.Event

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = torch.ones(4096, device=device) + 1
    end.record()
    end.synchronize()

    assert_on_flagos(output)
    assert start.elapsed_time(end) >= 0


metax_only = pytest.mark.skipif(
    torch_fl._build_accelerator() != "metax",
    reason="MetaX build required",
)


@metax_only
def test_metax_stream_shim_supports_event_ordering(device):
    """The stream shim must carry the event API, not just a raw handle.

    Inductor's cudagraph manager runs
    ``torch.cuda.Stream().wait_stream(torch.cuda.current_stream())``, and
    ``wait_stream`` is ``self.wait_event(stream.record_event())`` -- so a shim
    without ``record_event`` made torch.compile die with AttributeError before it
    could run a single kernel (issue #157).
    """
    current = torch.cuda.current_stream()

    # The launcher contract the shim exists for in the first place.
    assert current.cuda_stream == 0
    assert current.device == torch.device("cuda", torch_fl.flagos.current_device())

    # The exact call from the traceback.
    torch.cuda.Stream().wait_stream(current)

    event = current.record_event()
    assert isinstance(event, torch.cuda.Event)
    event.synchronize()
    assert event.query()

    # An explicitly supplied event must be the one returned, as on a real Stream.
    provided = torch.cuda.Event()
    assert current.record_event(provided) is provided

    current.wait_event(event)
    assert isinstance(current.query(), bool)
    current.synchronize()

    output = torch.ones(1024, device=device) + 1
    assert_on_flagos(output)


@metax_only
def test_metax_stream_context_selects_capture_stream(device):
    """``get_raw_stream`` must follow the current stream, not always report 0.

    Generated Inductor code launches every kernel on ``get_raw_stream(idx)``, and
    ``torch.cuda.graph()`` makes its capture stream current before capturing. A
    hardcoded 0 would send those launches to the default stream, silently leaving
    them out of the captured graph rather than failing.
    """
    default_raw = torch._C._cuda_getCurrentRawStream(torch_fl.flagos.current_device())
    assert default_raw == 0

    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        assert torch.cuda.current_stream().cuda_stream == side.cuda_stream
        assert torch._C._cuda_getCurrentRawStream(side.device_index) == side.cuda_stream
        output = torch.ones(1024, device=device) + 1

    # The context manager has to put the default stream back.
    assert torch._C._cuda_getCurrentRawStream(torch_fl.flagos.current_device()) == 0
    assert_on_flagos(output)


def test_inductor_benchmark_accepts_triton_backend_name():
    """The Triton backend name must not reach torch.device() as a device type.

    DeviceProperties reports the *Triton* backend name (``maca`` on MetaX) so
    triton picks the right backend, but inductor forwards that same string to its
    benchmarker as a torch device. ``maca`` is not one, so autotuning died with
    "Expected one of cpu, cuda, ... device string: maca". The vendor MetaX torch
    patches this in-tree; the official CPU wheel we ship against does not, so
    torch_fl maps it back to cuda (same physical GPU).
    """
    from torch._inductor.runtime import benchmarking

    from torch_fl.compile.device_interface import (
        _triton_backend,
        register_flagos_device_interface,
    )

    register_flagos_device_interface()
    device_type, _ = _triton_backend()

    if device_type == "cuda":
        pytest.skip("triton backend name is already a valid torch device type")

    # The name inductor would hand over is normally not a torch device at all
    # (``maca``). Two builds are exceptions. On native MUSA it parses because
    # torch_fl's FLAGOS_ALIAS_CUDA mode deliberately aliases the ``musa``
    # spelling onto flagos. On Hygon DCU it parses as a genuine, distinct device
    # type: DAS PyTorch is a HIP build, so ``hip`` is a device type of its own
    # (measured: ``torch.device("hip").type == "hip"``, while privateuse1 is
    # ``flagos``). Either way the name is not the device the tensors live on, so
    # what matters is that benchmark() translates it -- asserted below.
    try:
        aliased_to = torch.device(device_type)
    except RuntimeError as exc:
        assert "device type at start of device string" in str(exc)
    else:
        assert aliased_to.type in (
            torch._C._get_privateuse1_backend_name(),
            device_type,
        )

    # Either way, benchmark() must translate the name rather than pass it through.
    assert benchmarking.Benchmarker.benchmark._flagos_patched

    recorded = {}

    def fake_gpu_benchmark(self, _callable, **kwargs):
        recorded["ran"] = True
        return 0.0

    original = benchmarking.Benchmarker.benchmark_gpu
    benchmarking.Benchmarker.benchmark_gpu = fake_gpu_benchmark
    try:
        benchmarking.Benchmarker().benchmark(lambda: None, device=device_type)
    finally:
        benchmarking.Benchmarker.benchmark_gpu = original

    assert recorded.get("ran"), "benchmark() did not reach the GPU implementation"


def test_compile_multiple_inputs(device):
    """Test compilation with multiple input tensors."""
    if not HAS_TRITON:
        pytest.skip("Triton required for compilation")

    model = MatMulModel().to(device)
    a = torch.randn(32, 32, device=device)
    b = torch.randn(32, 32, device=device)

    compiled_model = torch.compile(model, backend="flagos")

    output = compiled_model(a, b)
    eager_output = model(a, b)

    assert_on_flagos(output)
    torch.testing.assert_close(output, eager_output, rtol=1e-4, atol=1e-4)


def test_compile_backward(device):
    """Test that compiled model supports backward pass."""
    if not HAS_TRITON:
        pytest.skip("Triton required for compilation")

    model = SimpleModel().to(device)
    x = torch.randn(32, 128, device=device, requires_grad=True)

    compiled_model = torch.compile(model, backend="flagos")

    # Forward + backward
    output = compiled_model(x)
    loss = output.sum()
    loss.backward()

    # Check gradients exist. The gradient staying on flagos is what proves the
    # backward graph was never rewritten to cuda -- that rewrite produced
    # stream-less autograd nodes and tripped engine.cpp's stream assertion.
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert_on_flagos(x.grad, "gradient")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_compile_dtypes(device, dtype):
    """Test compilation with different dtypes."""
    if not HAS_TRITON:
        pytest.skip("Triton required for compilation")

    model = SimpleModel().to(device).to(dtype)
    x = torch.randn(32, 128, device=device, dtype=dtype)

    compiled_model = torch.compile(model, backend="flagos")
    output = compiled_model(x)

    assert output.dtype == dtype
    assert_on_flagos(output)

    eager_output = model(x)
    # Float16 has lower precision
    rtol = 1e-2 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(output, eager_output, rtol=rtol, atol=rtol)


def test_compile_recompile(device):
    """Test that recompiling doesn't break."""
    if not HAS_TRITON:
        pytest.skip("Triton required for compilation")

    model = SimpleModel().to(device)
    x = torch.randn(32, 128, device=device)

    # First compilation
    compiled_model = torch.compile(model, backend="flagos")
    output1 = compiled_model(x)

    # Reset dynamo cache and recompile
    torch._dynamo.reset()
    compiled_model2 = torch.compile(model, backend="flagos")
    output2 = compiled_model2(x)

    torch.testing.assert_close(output1, output2, rtol=1e-6, atol=1e-6)


def test_fake_tensor_detach(device):
    """detach must not re-dispatch to itself under FakeTensorMode.

    The generated CUDA kernel used to call ``at::detach(self)``, which is
    registered on PrivateUse1 too and so dispatched straight back into itself.
    In eager, ``DeviceBoxingGuard``'s device rewrite masked the recursion; under
    FakeTensorMode it cannot, because the Python dispatch key sits above the
    backend key -- rewriting metadata does not change where dispatch goes. The
    kernel now calls ``at::native::detach``. Dynamo traces every ``nn.Linear``
    through detach, so a regression here is a stack-overflow crash, not a
    failure.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.randn(32, 128, device=device)
        d = x.detach()
        assert d.shape == x.shape
        assert d.device.type == x.device.type
        assert not d.requires_grad


def test_fake_tensor_linear(device):
    """F.linear under FakeTensorMode -- the shape dynamo actually traces.

    nn.Linear goes through detach internally; this is the end-to-end form of
    test_fake_tensor_detach and the exact call that used to segfault at trace
    time. Parameters are built inside the mode rather than by moving a module
    into it (nn.Module._apply cannot swap real params for fake ones).
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        weight = torch.nn.Parameter(torch.randn(64, 128, device=device))
        bias = torch.nn.Parameter(torch.randn(64, device=device))
        x = torch.randn(32, 128, device=device)
        out = torch.nn.functional.linear(x, weight, bias)
        assert out.shape == (32, 64)
        assert out.device.type == x.device.type


class NormalizedModel(torch.nn.Module):
    """Matmul + normalization, the reduction shape a transformer block hits.

    Exercises a different codegen path from SimpleModel: layer_norm lowers to a
    Triton reduction kernel, where `dynamic_scale_rblock` and the autotuner's
    per-config benchmarking come into play.
    """

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(64)

    def forward(self, a, b):
        return self.norm(torch.mm(a, b))


# --- Ascend ----------------------------------------------------------------
#
# Ascend is the one platform here that is not a CUDA-shaped GPU behind a shim:
# it has no CUDA runtime, so its device properties, raw streams and generated
# device snippets all come from the ACL runtime (see
# torch_fl/compile/platform_profile.py). These tests assert that the *Ascend*
# profile is what a flagos compile actually uses, rather than the CUDA one
# silently answering, and are marked so a CUDA box does not report them as
# passing.


ascend_only = pytest.mark.skipif(
    ACCELERATOR != "ascend",
    reason="Ascend build required",
)


@pytest.mark.ascend
@ascend_only
def test_ascend_profile_selected():
    """A flagos build on Ascend must not present itself as CUDA-like."""
    from torch_fl.compile.platform_profile import platform_profile

    profile = platform_profile()
    assert not profile.is_cuda_like
    assert profile.triton_device_type == "npu"
    assert profile.triton_backend_key == "ascend"


@pytest.mark.ascend
@ascend_only
def test_ascend_triton_backend_present():
    """The Ascend toolchain has to be installed for any of this to compile.

    Fails rather than skips: the alternative is a green suite that proves only
    that the compile path was never taken. See docs/vendors/ascend/installation.md
    for the triton-ascend + scripts/patch_triton_ascend.py setup.
    """
    import triton.backends

    from torch_fl.compile.platform_profile import platform_profile

    assert platform_profile().triton_backend_key in triton.backends.backends


@pytest.mark.ascend
@ascend_only
def test_ascend_reports_soc_arch_to_triton():
    """`cc` reaches Triton as GPUTarget.arch, and Ascend wants a SoC name there.

    A number (the CUDA compute-capability shape) makes the Ascend backend reject
    the target, so this pins the string form -- e.g. "Ascend910_9382".
    """
    from torch_fl.compile.device_interface import FlagOSDeviceInterface

    arch = FlagOSDeviceInterface.get_compute_capability()
    assert isinstance(arch, str)
    assert arch.lower().startswith("ascend")


@pytest.mark.ascend
@ascend_only
def test_ascend_raw_stream_matches_torch_fl_stream():
    """The launch stream must be the one torch_fl's aclnn ops run on.

    rt stream 0 is not ordered against the ops producing a kernel's inputs, which
    corrupts results silently instead of failing (the nan-loss regression in
    scripts/patch_triton_ascend.py). Equality with torch_fl's registry is the
    whole property.
    """
    from torch_fl.accelerator.ascend.acl_stream import current_acl_raw_stream
    from torch_fl.compile.device_interface import FlagOSDeviceInterface

    idx = torch_fl.flagos.current_device()
    raw = FlagOSDeviceInterface.get_raw_stream(idx)

    assert raw != 0
    assert raw == current_acl_raw_stream(idx)


@pytest.mark.ascend
@ascend_only
def test_ascend_registration_uses_ascend_codegen():
    """Registration must install the ACL snippets, not the CUDA ones."""
    from torch._inductor.codegen.common import (
        device_op_overrides_dict,
        get_scheduling_for_device,
    )
    from torch._inductor.codegen.triton import TritonScheduling

    from torch_fl.compile.device_interface import register_flagos_device_interface
    from torch_fl.compile.inductor_codegen import (
        DEVICE_TYPE,
        FlagOSAscendDeviceOpOverrides,
        register_flagos_codegen,
    )

    register_flagos_device_interface()
    register_flagos_codegen()

    assert get_scheduling_for_device(DEVICE_TYPE) is TritonScheduling
    assert isinstance(
        device_op_overrides_dict[DEVICE_TYPE], FlagOSAscendDeviceOpOverrides
    )


@pytest.mark.ascend
@ascend_only
def test_ascend_compile_fused_elementwise(device):
    """One Triton kernel for the whole pointwise chain, matching eager."""
    model = SimpleModel().to(device)
    x = torch.randn(32, 128, device=device)

    eager_output = model(x)
    compiled_model = torch.compile(model, backend="flagos")
    output = compiled_model(x)

    assert_on_flagos(output)
    torch.testing.assert_close(output, eager_output, rtol=1e-3, atol=1e-3)


@pytest.mark.ascend
@ascend_only
def test_ascend_compile_matmul_and_normalization(device):
    """Reduction codegen: matmul feeding a LayerNorm."""
    model = NormalizedModel().to(device)
    a = torch.randn(64, 64, device=device)
    b = torch.randn(64, 64, device=device)

    eager_output = model(a, b)
    compiled_model = torch.compile(model, backend="flagos")
    output = compiled_model(a, b)

    assert_on_flagos(output)
    torch.testing.assert_close(output, eager_output, rtol=1e-3, atol=1e-3)


@pytest.mark.ascend
@ascend_only
def test_ascend_compile_backward_stays_on_flagos(device):
    """AOT autograd's backward must stay on flagos, gradients included.

    A cuda-rewritten graph yields stream-less autograd nodes and trips
    engine.cpp's stream assertion (see torch_fl/compile/inductor_backend.py), so
    the gradient's device is the load-bearing assertion here.
    """
    model = SimpleModel().to(device)
    x = torch.randn(32, 128, device=device, requires_grad=True)

    compiled_model = torch.compile(model, backend="flagos")
    compiled_model(x).sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert_on_flagos(x.grad, "gradient")

    reference = SimpleModel().to(device)
    reference.load_state_dict(model.state_dict())
    y = x.detach().clone().requires_grad_(True)
    reference(y).sum().backward()

    torch.testing.assert_close(x.grad, y.grad, rtol=1e-3, atol=1e-3)


def test_flagtree_requires_flagtree_install():
    """FLAGOS_USE_FLAGTREE=1 must not silently no-op on a stock-Triton env.

    FlagTree replaces triton at install time, so the flag can only assert; it
    cannot switch anything on. Whichever Triton this env has, exactly one of the
    two branches below is the correct behaviour.
    """
    from torch_fl.compile.flagtree_shim import is_flagtree_active, require_flagtree

    if is_flagtree_active():
        require_flagtree()  # must not raise
    else:
        with pytest.raises(RuntimeError, match="not FlagTree"):
            require_flagtree()


def test_torch_backends_entry_point_is_registered():
    """A bare `import torch` must be able to initialize flagos.

    Inductor's parallel compile workers are fresh interpreters that import torch
    and triton but never torch_fl, so without this entry point they see the stock
    `torch.cuda.is_available()` and Triton's driver probe finds no active GPU
    backend -- every cold compile then dies in set_driver_to_gpu(). Asserting on
    the installed metadata rather than on import side effects, because this test
    process has already imported torch_fl the normal way.
    """
    from importlib.metadata import entry_points

    registered = {ep.name: ep.value for ep in entry_points(group="torch.backends")}
    assert registered.get("torch_fl") == "torch_fl._autoload:init", (
        "torch_fl must register a torch.backends entry point; "
        f"found {registered}. Reinstall the package if this is a stale env."
    )


def test_torch_backends_entry_point_init_is_idempotent():
    """Calling the hook in a process that already has torch_fl is a no-op."""
    from torch_fl._autoload import init

    init()
    init()
    assert torch.flagos.is_available()


def test_ppu_flagtree_defaults_to_serial_compile(monkeypatch):
    """PPU FlagTree must avoid CUDA initialization in forked compile workers."""
    from torch_fl.compile import inductor_backend

    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.setattr(
        "torch_fl.compile.flagtree_shim.flagtree_backend", lambda: "ppu"
    )

    patches = {}
    inductor_backend._patch_ppu_flagtree_compile_workers(patches)

    assert patches["compile_threads"] == 1


def test_ppu_flagtree_preserves_explicit_compile_threads(monkeypatch):
    """An explicit per-compile or environment setting remains authoritative."""
    from torch_fl.compile import inductor_backend

    monkeypatch.setattr(
        "torch_fl.compile.flagtree_shim.flagtree_backend", lambda: "ppu"
    )

    patches = {"compile_threads": 4}
    inductor_backend._patch_ppu_flagtree_compile_workers(patches)
    assert patches["compile_threads"] == 4

    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "8")
    patches = {}
    inductor_backend._patch_ppu_flagtree_compile_workers(patches)
    assert "compile_threads" not in patches


def test_non_ppu_flagtree_keeps_default_compile_threads(monkeypatch):
    """Other FlagTree backends keep Inductor's asynchronous compilation."""
    from torch_fl.compile import inductor_backend

    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.setattr(
        "torch_fl.compile.flagtree_shim.flagtree_backend", lambda: "hcu"
    )
    # The serialization decision is per accelerator as well as per FlagTree
    # backend, so pin it: on a GCU build the vendor driver serializes regardless
    # of the FlagTree backend name, which is a different rule than the one under
    # test here.
    monkeypatch.setattr("torch_fl._build_config.ACCELERATOR", "cuda")

    patches = {}
    inductor_backend._patch_ppu_flagtree_compile_workers(patches)

    assert "compile_threads" not in patches


def test_musa_flagtree_defaults_to_serial_compile(monkeypatch):
    """MThreads FlagTree must not initialize MUSA in a forked worker."""
    from torch_fl.compile import inductor_backend

    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.setattr(
        "torch_fl.compile.flagtree_shim.flagtree_backend", lambda: "mthreads"
    )
    monkeypatch.setattr("torch_fl._build_config.ACCELERATOR", "musa")

    patches = {}
    inductor_backend._patch_vendor_flagtree_compile_workers(patches)

    assert patches["compile_threads"] == 1


def test_musa_flagtree_preserves_explicit_compile_threads(monkeypatch):
    """MUSA callers can override vendor serialization when they need to."""
    from torch_fl.compile import inductor_backend

    monkeypatch.setattr(
        "torch_fl.compile.flagtree_shim.flagtree_backend", lambda: "mthreads"
    )
    monkeypatch.setattr("torch_fl._build_config.ACCELERATOR", "musa")

    patches = {"compile_threads": 4}
    inductor_backend._patch_vendor_flagtree_compile_workers(patches)
    assert patches["compile_threads"] == 4


def test_flagtree_is_never_importable_as_flagtree():
    """Guard the packaging trap: the wheel is 'flagtree', the module is 'triton'.

    A future contributor reaching for `import flagtree` would write code that can
    only ever raise, which is exactly the bug this test pins down.
    """
    import importlib.util

    assert importlib.util.find_spec("flagtree") is None


@pytest.mark.skipif(
    os.environ.get("FLAGOS_USE_FLAGTREE", "0") != "1",
    reason="FlagTree compilation not requested (set FLAGOS_USE_FLAGTREE=1)",
)
def test_flagtree_compiles_correct_results(device):
    """Compile the basic model through the active vendor FlagTree runtime."""
    from torch_fl.compile.flagtree_shim import is_flagtree_active

    if not is_flagtree_active():
        pytest.skip("active triton is stock Triton, not FlagTree")

    model = SimpleModel().to(device)
    x = torch.randn(32, 128, device=device)

    eager_output = model(x)
    compiled_model = torch.compile(model, backend="flagos")
    output = compiled_model(x)

    assert_on_flagos(output)
    torch.testing.assert_close(output, eager_output, rtol=1e-4, atol=1e-4)


@pytest.mark.musa
def test_musa_flagtree_binds_to_torch_fl_runtime():
    """FlagTree must reach MUSA through torch_fl, never through torch_musa.

    The torch_musa plugin cannot coexist with torch_fl: its ``__init__`` claims
    the process-global PrivateUse1 hooks torch_fl must own. The vendor MThreads
    driver nevertheless reads its device/stream/capability from ``torch_musa``,
    so torch_fl rebinds those lookups onto its own runtime. This asserts the
    binding rather than the absence of a crash -- a driver reading from some
    other runtime would still compile, just not against the device that owns the
    tensors.
    """
    from torch_fl._build_config import ACCELERATOR
    from torch_fl.compile import flagtree_shim

    if ACCELERATOR != "musa":
        pytest.skip("MUSA build required")
    if (
        not flagtree_shim.is_flagtree_active()
        or flagtree_shim.flagtree_backend() != "mthreads"
    ):
        pytest.skip("MThreads FlagTree runtime required")

    assert flagtree_shim.bind_flagtree_musa_driver()

    # The driver's own target resolution now runs entirely through torch_fl.
    target = flagtree_shim.flagtree_musa_driver_target()
    assert target is not None
    backend, capability, warp_size = target
    major, minor = flagtree_shim.get_musa_device_capability()
    assert backend == "musa"
    assert capability == major * 10 + minor
    assert warp_size == (32 if major > 2 else 128)

    # The compiled kernels launch on the same stream the mudnn kernels use, so
    # this handle must be torch_fl's, and it must be a real stream.
    raw_stream = flagtree_shim.get_musa_current_raw_stream()
    assert raw_stream == torch_fl.flagos.current_stream().musa_stream
    assert raw_stream != 0

    # Every lookup the driver now performs resolves inside torch_fl. Asserted on
    # the module of the bound callables rather than on ``torch_musa`` being
    # absent from sys.modules: FlagGems discovery legitimately publishes a small
    # compatibility surface under that name (_install_musa_flaggems_compat), and
    # this test must hold whether or not FlagGems is enabled.
    from triton.backends import backends as triton_backends

    driver_cls = triton_backends["mthreads"].driver
    for attr in (
        "is_active",
        "_get_device_capability",
        "_get_current_stream",
        "_get_current_device",
        "_set_current_device",
    ):
        func = getattr(driver_cls, attr)
        assert func.__module__.startswith("torch_fl"), (attr, func)


@pytest.mark.musa
def test_musa_flagtree_compiles_forward_backward(device):
    """The MThreads FlagTree path must preserve native MUSA autograd."""
    from torch_fl._build_config import ACCELERATOR
    from torch_fl.compile.flagtree_shim import flagtree_backend, is_flagtree_active

    if ACCELERATOR != "musa":
        pytest.skip("MUSA build required")
    if not is_flagtree_active() or flagtree_backend() != "mthreads":
        pytest.skip("MThreads FlagTree runtime required")

    model = SimpleModel().to(device)
    x = torch.randn(32, 128, device=device, requires_grad=True)
    eager = model(x)
    compiled = torch.compile(model, backend="flagos")(x)
    assert_on_flagos(compiled)
    torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-4)
    compiled.sum().backward()
    assert x.grad is not None
    assert_on_flagos(x.grad, "gradient")
    # A full compile+launch cycle must not have imported the real plugin. Only
    # torch_fl's own shim may hold this name (see the binding test above).
    plugin = sys.modules.get("torch_musa")
    assert plugin is None or plugin.__spec__.origin == "torch_fl_shim"


def _skip_unless_gcu(*, needs_triton: bool = True):
    """Gate the GCU tests on the build and, by default, on triton_gcu.

    Compiling anything on GCU needs Enflame's triton_gcu plugin, which the CI
    image does not install (see the FLAGGEMS_KERNEL=0 note in set_env_gcu.sh).
    Checking the accelerator alone would turn a missing vendor Triton stack into
    a test failure rather than a skip. ``needs_triton=False`` is for the checks
    that only read torch_fl's own state.
    """
    from torch_fl._build_config import ACCELERATOR

    if ACCELERATOR != "gcu":
        pytest.skip("GCU build required")
    if needs_triton:
        from torch_fl.accelerator.gcu._gcu_compat import is_triton_gcu_available

        if not is_triton_gcu_available():
            pytest.skip("triton_gcu required (vendor Triton stack not installed)")


@pytest.mark.gcu
def test_gcu_raw_stream_is_the_eager_stream():
    """Compiled GCU kernels must launch on the stream eager work already uses."""
    _skip_unless_gcu()
    from torch_fl.compile.flagtree_shim import get_gcu_current_raw_stream

    raw_stream = get_gcu_current_raw_stream()
    assert raw_stream == torch_fl.flagos.current_stream().gcu_stream
    assert raw_stream != 0


@pytest.mark.gcu
def test_gcu_device_properties_use_vendor_warp_size():
    """Inductor must size kernels by the GCU warp, not CUDA's 32.

    ``_DeviceProperties`` carries no ``warp_size`` on GCU, and Inductor's default
    is CUDA's. The authority is the Triton driver, since it compiles the kernel.
    """
    _skip_unless_gcu()
    from torch._inductor.runtime.hints import DeviceProperties
    from triton.runtime import driver

    props = DeviceProperties.create(torch.device("flagos", 0))
    assert props.type == "gcu"
    assert props.warp_size == driver.active.get_current_target().warp_size


@pytest.mark.gcu
def test_gcu_cache_key_survives_pickling():
    """Inductor's FX graph cache must not be bypassed on GCU.

    Two things break this key. ``CacheBase.get_system`` reads the CUDA device
    getter torch_fl aliases onto flagos and, with ``torch.version.cuda`` unset,
    takes the HIP branch into ``gcnArchName``. And ``torch.device`` is replaced
    by a function-local class that pickle cannot resolve by name, which
    downgrades every compile to BypassFxGraphCache.

    Both are torch_fl-side, so this holds without the vendor Triton stack.
    """
    _skip_unless_gcu(needs_triton=False)
    import pickle

    from torch._inductor.codecache import CacheBase

    system = CacheBase.get_system()
    assert system["device"]["name"]
    assert "hash" in system

    assert pickle.loads(pickle.dumps(torch.device)) is torch.device
    assert isinstance(torch.device("flagos:0"), torch.device)


@pytest.mark.gcu
def test_gcu_compiles_dim0_reduction(device):
    """A reduction over the non-contiguous axis must match eager.

    triton_gcu miscompiles a persistent reduction at XBLOCK=1 -- silently, with
    wrong values rather than a failure. It is Inductor's first config for every
    persistent reduction, and dim-0 reductions are mostly a backward-pass shape,
    so unguarded this surfaces only as bad gradients.
    """
    _skip_unless_gcu()

    def fn(a, b):
        return (a * b).sum(dim=0)

    x = torch.randn(16, 32, device=device)
    y = torch.randn(16, 32, device=device)
    compiled = torch.compile(fn, backend="flagos")(x, y)
    assert_on_flagos(compiled)
    torch.testing.assert_close(compiled, fn(x, y), rtol=1e-4, atol=1e-4)


@pytest.mark.gcu
def test_gcu_compiles_transposed_load(device):
    """A transposed load must compile despite the vendor's tt.trans bug.

    2D tiling makes Inductor emit a transposed load, which Triton lowers to
    ``tt.trans``; triton_gcu's layout inference rejects its own inferred type
    there. One tile per kernel keeps the indexing linear.
    """
    _skip_unless_gcu()

    def fn(a, b):
        return a.t().contiguous() + b

    x = torch.randn(128, 128, device=device)
    y = torch.randn(128, 128, device=device)
    compiled = torch.compile(fn, backend="flagos")(x, y)
    assert_on_flagos(compiled)
    torch.testing.assert_close(compiled, fn(x, y), rtol=1e-4, atol=1e-4)


@pytest.mark.gcu
def test_gcu_compiles_forward_backward(device):
    """The GCU path must preserve native autograd, gradients included."""
    _skip_unless_gcu()

    model = SimpleModel().to(device)
    x = torch.randn(32, 128, device=device, requires_grad=True)
    eager = model(x)
    eager.sum().backward()
    eager_grads = [p.grad.clone() for p in model.parameters()]
    for param in model.parameters():
        param.grad = None

    compiled = torch.compile(model, backend="flagos")(x)
    assert_on_flagos(compiled)
    torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-4)

    compiled.sum().backward()
    assert x.grad is not None
    assert_on_flagos(x.grad, "gradient")
    for expected, param in zip(eager_grads, model.parameters()):
        torch.testing.assert_close(param.grad, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.gcu
def test_gcu_defaults_to_serial_compile(monkeypatch):
    """A tops pointer only resolves against the current device, so no forking.

    This is a configuration check on torch_fl's side; triton_gcu need not be
    installed for the rule to hold.
    """
    from torch_fl.compile import inductor_backend

    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.setattr("torch_fl._build_config.ACCELERATOR", "gcu")

    patches = {}
    inductor_backend._patch_vendor_flagtree_compile_workers(patches)
    assert patches["compile_threads"] == 1

    # An explicit request still wins.
    patches = {"compile_threads": 4}
    inductor_backend._patch_vendor_flagtree_compile_workers(patches)
    assert patches["compile_threads"] == 4

    _skip_unless_gcu(needs_triton=False)


# The generic FlagTree test above also covers the MThreads runtime when the
# vendor environment is active; the MUSA-specific test adds backward coverage.


# End-to-end fallback remains useful on CPU-only hosts.


# Keep the fallback test below independent of vendor availability.
@pytest.mark.ascend
@ascend_only
def test_ascend_masked_byte_load_requires_linearize(device):
    """Reduction of the miscompile behind wrong `relu` gradients on Ascend.

    A masked 2-D strided load of an 8-bit dtype returns only the first two elements
    of each row, repeated, with no error raised. `enable_linearize=True` -- which
    torch_fl/compile/triton_byte_loads.py injects into every inductor compile --
    makes the same kernel read correct data. Written against triton directly
    because the defect is in the vendor codegen, not in any one fused graph.

    The option is passed explicitly here: a bare `@triton.jit` launch compiles
    through `JITFunction.compile`, bound from `triton.compiler.compile` when the
    binder is built, so it does not see the patch on the `triton.compile` alias
    that inductor calls. End-to-end coverage of the patched path is
    test_ascend_compile_backward_stays_on_flagos, whose relu backward is what
    surfaced this in the first place.
    """
    pytest.importorskip("triton")
    from compile_support import masked_byte_row_load

    # Distinct values per element, so a repeated-element read cannot pass by luck.
    expected = (torch.arange(32 * 128) % 100).to(torch.int8).view(32, 128)
    source = expected.to(device)

    def run(**options):
        out = torch.zeros(32, 128, device=device)
        masked_byte_row_load[(4,)](source, out, 128, XBLOCK=32, R0_BLOCK=32, **options)
        return out.cpu()

    assert not torch.equal(run(), expected.to(torch.float32)), (
        "masked byte load is correct without enable_linearize: the vendor bug this "
        "workaround exists for is fixed, so drop the workaround"
    )
    torch.testing.assert_close(run(enable_linearize=True), expected.to(torch.float32))


@pytest.mark.ascend
@ascend_only
def test_ascend_bool_tensors_declared_as_byte_pointers():
    """Bool tensor args must reach Triton as ``*i8``, not ``*i1``.

    ``enable_linearize`` does not fix the ``*i1`` pointer type, so the signature
    rewrite is the half of the workaround that makes boolean masks -- and therefore
    `relu` backward -- read correct data.
    """
    pytest.importorskip("triton")
    from torch._inductor.codegen import triton_utils
    from torch._inductor.codegen.common import TensorArg
    from torch._inductor.virtualized import V

    from torch_fl.compile.triton_byte_loads import patch_triton_byte_load_workarounds

    patch_triton_byte_load_workarounds()

    class _Graph:
        """`signature_of` asks the graph whether an arg was an unspec scalar."""

        def is_unspec_arg(self, name):
            return False

    with V.set_graph_handler(_Graph()):
        bool_arg = TensorArg(name="in_ptr0", buffer="buf0", dtype=torch.bool)
        assert triton_utils.signature_of(bool_arg, size_dtype=None) == "*i8"

        # Only bool changes; every other dtype keeps its own pointer type.
        float_arg = TensorArg(name="in_ptr1", buffer="buf1", dtype=torch.float32)
        assert triton_utils.signature_of(float_arg, size_dtype=None) == "*fp32"


@pytest.mark.ascend
@ascend_only
def test_ascend_oversized_config_does_not_fail_compile(device):
    """A tile that overflows UB must cost one autotune config, not the compile.

    Inductor precompiles several block sizes and expects the large ones to be
    rejected, but triton-ascend reports "ub overflow" as a generic
    MLIRCompilationError, which upstream does not treat as a resource limit. The
    model below is the one that first hit this: its `relu` backward is a persistent
    reduction whose XBLOCK=128 and 64 candidates both exceed the 192 KB budget.
    """
    pytest.importorskip("triton")

    model = SimpleModel().to(device)
    x = torch.randn(32, 128, device=device, requires_grad=True)

    reference = SimpleModel().to(device)
    reference.load_state_dict(model.state_dict())
    y = x.detach().clone().requires_grad_(True)

    torch.compile(model, backend="flagos")(x).sum().backward()
    reference(y).sum().backward()

    assert x.grad is not None
    assert_on_flagos(x.grad, "gradient")
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-3, atol=1e-3)


@pytest.mark.ascend
@ascend_only
def test_ascend_defaults_to_serial_compile():
    """Ascend compiles in the parent process unless told otherwise.

    A kernel built in a compile worker segfaults when the parent launches it, so
    the default has to be serial; an explicit setting must still win.
    """
    from torch_fl.compile.inductor_backend import _patch_ascend_compile_workers
    from torch_fl.compile.platform_profile import platform_profile

    if platform_profile().is_cuda_like:
        pytest.skip("serial compile default applies to non-CUDA-like builds")

    saved = os.environ.pop("TORCHINDUCTOR_COMPILE_THREADS", None)
    try:
        patches = {}
        _patch_ascend_compile_workers(patches)
        assert patches["compile_threads"] == 1

        explicit = {"compile_threads": 8}
        _patch_ascend_compile_workers(explicit)
        assert explicit["compile_threads"] == 8

        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "4"
        from_env = {}
        _patch_ascend_compile_workers(from_env)
        assert "compile_threads" not in from_env
    finally:
        os.environ.pop("TORCHINDUCTOR_COMPILE_THREADS", None)
        if saved is not None:
            os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = saved


def test_compile_fallback_eager():
    """Test fallback to eager mode when compilation fails."""
    # Set fallback env var
    os.environ["FLAGOS_COMPILE_FALLBACK_EAGER"] = "1"

    try:

        class ProblematicModel(torch.nn.Module):
            def forward(self, x):
                return x

        model = ProblematicModel()
        x = torch.randn(10, 10)
        compiled_model = torch.compile(model, backend="flagos")
        output = compiled_model(x)
        assert output.shape == (10, 10)
    finally:
        os.environ.pop("FLAGOS_COMPILE_FALLBACK_EAGER", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
