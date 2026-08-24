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

import os

import pytest
import torch_fl
import torch


# Skip all tests if torch.compile not available (torch < 2.0)
try:
    import torch._dynamo

    HAS_COMPILE = True
except ImportError:
    HAS_COMPILE = False

pytestmark = pytest.mark.skipif(
    not HAS_COMPILE, reason="torch.compile not available (torch < 2.0)"
)

# flagos tensors report either name depending on how the device was spelled.
FLAGOS_DEVICE_TYPES = ("privateuseone", "flagos")


def assert_on_flagos(tensor, what="output"):
    """The graph is compiled *on* flagos, so results must come back on flagos.

    A cuda round trip would both cost a copy per call and produce stream-less
    autograd nodes (see torch_fl/compile/inductor_backend.py), so this is a
    load-bearing assertion, not a smoke check.
    """
    assert tensor.device.type in FLAGOS_DEVICE_TYPES, (
        f"{what} landed on {tensor.device}, expected flagos"
    )


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

    # The name inductor would hand over is not a torch device on its own ...
    with pytest.raises(RuntimeError, match="device type at start of device string"):
        torch.device(device_type)

    # ... so benchmark() must translate it rather than pass it through.
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

    patches = {}
    inductor_backend._patch_ppu_flagtree_compile_workers(patches)

    assert "compile_threads" not in patches


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
    """Compiling through FlagTree must match eager.

    Requires a FlagTree-built env; see docs/architecture/torch-compile-integration.md.
    """
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


def test_compile_fallback_eager():
    """Test fallback to eager mode when compilation fails."""
    # Set fallback env var
    os.environ["FLAGOS_COMPILE_FALLBACK_EAGER"] = "1"

    try:
        # Create a model that might cause compilation issues
        class ProblematicModel(torch.nn.Module):
            def forward(self, x):
                # Some operation that might not compile cleanly
                return x

        model = ProblematicModel()
        x = torch.randn(10, 10)

        # Should not raise, falls back to eager
        compiled_model = torch.compile(model, backend="flagos")
        output = compiled_model(x)

        assert output.shape == (10, 10)
    finally:
        os.environ.pop("FLAGOS_COMPILE_FALLBACK_EAGER", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
