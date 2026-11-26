# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Real TOPS submissions through the installed source-generated host bundle."""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch_fl

pytestmark = [
    pytest.mark.enflame_native_host,
    pytest.mark.flaggems,
    pytest.mark.main_ops,
]


def launcher():
    return importlib.import_module(torch_fl.native_host._hosts.package + "._launcher")


def metadata(**kwargs):
    from triton.runtime import driver

    arch = driver.active.get_current_target().arch.split("--")[-1]
    return SimpleNamespace(arch=arch, num_warps=1, shared=0, **kwargs)


def test_installed_backend_contract_matches_its_manifest():
    from torch_fl.native_host.backend import COMPILED_FIELDS

    record = torch_fl.native_host._hosts.manifest["backend"]
    assert record["vendor"] == "enflame" and record["compiler_backend"] == "gcu"
    assert launcher().backend_contract() == {k: record[k] for k in COMPILED_FIELDS}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(17,), (3, 5, 7), (19, 31)])
def test_native_pointwise_binds_live_inputs_and_outputs(dtype, shape):
    import flag_gems

    cpu = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(dtype) / 128
    value = cpu.to("flagos")
    before = torch_fl.native_host.audit()["host_artifacts"]
    submissions = sum(launcher().enqueue_stats().values())
    with torch.no_grad():
        first = flag_gems.cos(value)
        second = flag_gems.cos(value + 0.25)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(first.cpu(), cpu.cos())
    torch.testing.assert_close(second.cpu(), (cpu + 0.25).cos())
    assert first.data_ptr() != second.data_ptr()
    assert sum(launcher().enqueue_stats().values()) >= submissions + 2
    assert torch_fl.native_host.audit()["host_artifacts"] == before


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_matmul_preserves_noncontiguous_inputs(dtype):
    import flag_gems

    left = (torch.arange(17 * 31).reshape(31, 17) % 13 - 6).to(dtype) / 8
    right = (torch.arange(31 * 13).reshape(31, 13) % 7 - 3).to(dtype) / 8
    a, b = left.to("flagos").t(), right.to("flagos")
    assert a.stride() == (1, 17)
    expected = left.t() @ right
    retained = []
    with torch.no_grad():
        for _ in range(3):
            retained.append(flag_gems.mm(a, b))
    torch_fl.flagos.synchronize()
    for actual in retained:
        torch.testing.assert_close(actual.cpu(), expected)
    assert len({value.data_ptr() for value in retained}) == 3


@pytest.mark.parametrize(
    "kind,value",
    [
        ("i8", -128),
        ("u16", 65535),
        ("i64", -(2**63)),
        ("u64", 2**64 - 1),
        ("fp32", -1.5),
        ("bf16", 1.5),
    ],
)
def test_zero_grid_packs_scalar_widths_without_submitting(kind, value):
    before = launcher().enqueue_stats()
    call = launcher().Launcher(SimpleNamespace(signature={0: kind}), metadata())
    call(0, 1, 1, 0, 0, None, None, None, None, value)
    assert launcher().enqueue_stats() == before


def test_unsupported_scratch_is_rejected_before_submission():
    with pytest.raises(ValueError, match="scratch"):
        launcher().Launcher(
            SimpleNamespace(signature={}), metadata(global_scratch_size=1)
        )


def test_grid_overflow_is_rejected_before_submission():
    call = launcher().Launcher(SimpleNamespace(signature={}), metadata())
    with pytest.raises(ValueError, match="grid"):
        call(65536, 1, 1, 0, 0, None, None, None, None)


def test_prebuilt_driver_refuses_runtime_launcher_compilation():
    backend = importlib.import_module("triton.backends.enflame.backend")
    for function in (backend.build_gcu_ext, backend.compile_module_from_src):
        with pytest.raises(RuntimeError, match="compilation is forbidden"):
            function()


@pytest.mark.parametrize("padding", ["zero", "nan"])
def test_software_descriptor_abi_is_packed_without_submission(padding):
    from triton.tools.tensor_descriptor import TensorDescriptor

    value = torch.empty(3, 16, device="flagos")
    descriptor = TensorDescriptor(value, [3, 16], [16, 1], [1, 16], padding=padding)
    call = launcher().Launcher(
        SimpleNamespace(signature={0: "tensordesc<fp32[2]>"}), metadata()
    )
    call(0, 1, 1, 0, 0, None, None, None, None, descriptor)


def test_warm_silu_dispatch_uses_the_compiled_vendor_host():
    import flag_gems

    cpu = torch.arange(129, dtype=torch.float32) / 128
    value = cpu.to("flagos")
    with torch.no_grad():
        flag_gems.silu(value)
        before = torch_fl.native_host.audit()["host_artifacts"]
        result = flag_gems.silu(value)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(result.cpu(), torch.nn.functional.silu(cpu))
    assert torch_fl.native_host.audit()["host_artifacts"] == before
    assert any(
        "silu" in name for name in torch_fl.native_host.audit()["launched_kernels"]
    )


def test_current_nondefault_stream_and_driver_reconstruction():
    import flag_gems

    from torch_fl.accelerator.enflame.tops_stream import TopsStream

    backend = importlib.import_module("triton.backends.enflame.backend")
    previous = launcher().current_stream(0)
    stream = TopsStream(0)
    assert stream.handle != previous
    with torch_fl.flagos.stream(stream), torch.no_grad():
        value = torch.arange(129, dtype=torch.float32, device="flagos") / 128
        result = flag_gems.cos(value)
        driver = backend.GCUDriver()
        assert driver.get_current_stream(0) == launcher().current_stream(0)
        assert driver.get_current_stream(0) == stream.handle
    stream.synchronize()
    assert launcher().current_stream(0) == previous
    torch.testing.assert_close(result.cpu(), (torch.arange(129) / 128).cos())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_weak_scalar_multiply_preserves_opmath_precision(dtype):
    cpu = torch.arange(-1025, 1026, dtype=torch.float32).to(dtype)
    value = cpu.to("flagos")
    scale = 128**-0.5
    expected = (cpu.float() * scale).to(dtype)
    with torch.no_grad():
        actual = value * scale
        reflected = scale * value
        explicit = torch.ops.aten.mul.Scalar(value, scale)
    torch_fl.flagos.synchronize()
    for result in (actual, reflected, explicit):
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dim,keepdim", [(None, False), (None, True), (0, False), (-1, True)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vendor_argmax_returns_device_int64_indices(dtype, dim, keepdim):
    cpu = torch.tensor([[1, 7, 7, 2], [9, 0, 3, 9], [2, 4, 5, 1]], dtype=dtype)
    value = cpu.to("flagos").t()
    result = torch.argmax(value, dim=dim, keepdim=keepdim)
    torch_fl.flagos.synchronize()
    assert result.dtype == torch.int64 and result.device.type == "flagos"
    torch.testing.assert_close(result.cpu(), cpu.t().argmax(dim=dim, keepdim=keepdim))


def test_argmax_out_preserves_noncontiguous_int64_storage():
    cpu = torch.tensor([[1, 9, 9], [8, 2, 3], [0, 6, 2]], dtype=torch.bfloat16)
    value = cpu.to("flagos")
    storage = torch.full((7,), -9, dtype=torch.int64, device="cpu").to("flagos")
    output = storage[1::2]
    result = torch.argmax(value, dim=1, out=output)
    torch_fl.flagos.synchronize()
    assert result.data_ptr() == output.data_ptr()
    expected = torch.full((7,), -9, dtype=torch.int64)
    expected[1::2] = cpu.argmax(1)
    torch.testing.assert_close(storage.cpu(), expected)


def test_int64_strided_copy_keeps_both_words_on_device():
    cpu = (torch.arange(-21, 21, dtype=torch.int64).reshape(6, 7) * 2**35 + 17).t()
    source = cpu.contiguous().to("flagos").t().contiguous().t()
    output = torch.empty_strided(
        source.shape, (1, 11), dtype=torch.int64, device="flagos"
    )
    output.copy_(source)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(output.cpu(), cpu)


@pytest.mark.parametrize(
    "op",
    [torch.add, torch.sub, torch.eq, torch.ne, torch.lt, torch.le, torch.gt, torch.ge],
)
def test_vendor_word_composition_preserves_int64_carries_and_comparisons(op):
    cpu = torch.tensor(
        [
            -(2**63),
            -(2**35) - 1,
            -(2**31),
            -1,
            0,
            2**31 - 1,
            2**32 - 1,
            2**35,
            2**63 - 1,
        ],
        dtype=torch.int64,
    )
    other = cpu.flip(0).reshape(-1, 1)
    result = op(cpu.to("flagos"), other.to("flagos"))
    scalar = op(cpu.to("flagos"), 1)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(result.cpu(), op(cpu, other), rtol=0, atol=0)
    torch.testing.assert_close(scalar.cpu(), op(cpu, 1), rtol=0, atol=0)


@pytest.mark.parametrize(
    "start,end,step", [(0, 32, 1), (-31, 17, 3), (12, -15, -2), (0, 0, 1)]
)
def test_vendor_int64_arange_and_cast(start, end, step):
    expected = torch.arange(start, end, step, dtype=torch.int64)
    result = torch.arange(start, end, step, dtype=torch.int64, device="flagos")
    floating = result.float()
    roundtrip = floating.int().long()
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(floating.cpu(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(roundtrip.cpu(), expected, rtol=0, atol=0)


def test_embedding_checks_int64_bounds_without_truncation():
    weight = torch.arange(42, dtype=torch.float32).reshape(7, 6)
    indices = torch.tensor([[0, 6], [3, 2]], dtype=torch.int64)
    actual = torch.nn.functional.embedding(indices.to("flagos"), weight.to("flagos"))
    torch.testing.assert_close(
        actual.cpu(), torch.nn.functional.embedding(indices, weight)
    )
    for invalid in (-1, 7, 2**32 + 1):
        with pytest.raises(IndexError, match="index out of range"):
            torch.nn.functional.embedding(
                torch.tensor([invalid]).to("flagos"), weight.to("flagos")
            )


def test_argmax_scalar_nan_and_empty_contracts():
    for cpu, dim in [
        (torch.tensor(3.0), None),
        (torch.tensor(3.0), 0),
        (torch.tensor([1.0, float("nan"), 7.0, float("nan")]), 0),
    ]:
        actual = torch.argmax(cpu.to("flagos"), dim=dim)
        torch.testing.assert_close(actual.cpu(), torch.argmax(cpu, dim=dim))
    with pytest.raises((RuntimeError, IndexError), match="empty"):
        torch.argmax(torch.empty(0, device="flagos"))


def test_fp32_scalar_add_uses_vendor_scalar_opmath():
    cpu = torch.arange(-1025, 1026, dtype=torch.float32) / 128
    value = cpu.to("flagos")
    for scalar, alpha in [(1e-6, 1), (0.3, 2), (-2.0, 0.5)]:
        actual = torch.ops.aten.add.Scalar(value, scalar, alpha=alpha)
        torch.testing.assert_close(
            actual.cpu(), torch.add(cpu, scalar, alpha=alpha), rtol=0, atol=0
        )


def test_fp32_softmax_backward_preserves_vendor_reduction_order():
    rng = torch.Generator().manual_seed(42)
    grad = torch.randn((2, 3, 128), generator=rng)
    output = torch.randn((2, 3, 128), generator=rng).softmax(-1)
    chunks = (grad * output).unflatten(-1, (8, 16)).unbind(-2)
    reduction = sum(chunks[1:], chunks[0])
    for width in (8, 4, 2, 1):
        reduction = reduction[..., :width] + reduction[..., width:]
    expected = (grad - reduction) * output

    with torch_fl.native_host.execution_scope(0):
        actual = torch.ops.aten._softmax_backward_data.default(
            grad.to("flagos"), output.to("flagos"), -1, torch.float32
        )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_fp32_log_softmax_backward_preserves_small_probabilities():
    rng = torch.Generator().manual_seed(42)
    output = torch.randn((4, 320), generator=rng).log_softmax(-1)
    output[:, 1] = -99
    grad = torch.zeros_like(output)
    grad[:, 11] = -0.25
    expected = torch.ops.aten._log_softmax_backward_data.default(
        grad, output, -1, torch.float32
    )
    with torch_fl.native_host.execution_scope(0):
        actual = torch.ops.aten._log_softmax_backward_data.default(
            grad.to("flagos"), output.to("flagos"), -1, torch.float32
        )
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-7, atol=1e-44)


def test_mixed_producer_chain_is_ordered_without_per_operator_drains():
    import flag_gems

    cpu = torch.arange(4096, dtype=torch.float32).reshape(32, 128) / 4096
    value = cpu.to("flagos")
    with torch.no_grad():
        for _ in range(40):
            # Real Triton -> topsaten -> Triton -> topsaten dependencies.
            mean = flag_gems.mean_dim(value * value, dim=[-1], keepdim=True)
            normalized = torch.rsqrt(mean + 1e-6)
            actual = torch.sin(flag_gems.mul(value, normalized))
    torch_fl.flagos.synchronize()
    expected = torch.sin(cpu * torch.rsqrt(cpu.pow(2).mean(-1, keepdim=True) + 1e-6))
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_vendor_rsub_uses_reflected_operand_order(dtype):
    value = torch.tensor([1, -2, 0, 9], dtype=dtype)
    other = torch.tensor([5, 2, -8, 1], dtype=dtype)
    x, y = value.to("flagos"), other.to("flagos")
    rsub = torch.ops.aten.rsub
    before = torch_fl.native_host.enqueue_origins()["python"]
    with torch_fl.native_host.execution_scope(0):
        scalar = rsub.Scalar(x, 3, 2)
        tensor = rsub.Tensor(x, y, alpha=2)
    torch.testing.assert_close(scalar.cpu(), rsub.Scalar(value, 3, 2), rtol=0, atol=0)
    torch.testing.assert_close(
        tensor.cpu(), rsub.Tensor(value, other, alpha=2), rtol=0, atol=0
    )
    assert torch_fl.native_host.enqueue_origins()["python"] == before


@pytest.mark.parametrize("layout", ["dense", "transpose", "scalar", "empty"])
def test_fp32_square_composition_preserves_values_and_layouts(layout):
    cpu = torch.tensor(
        [-0.0, 0.0, -1, 1, -3, 3, 1e-30, 1e20, float("inf"), float("nan")]
    )
    if layout == "transpose":
        cpu = cpu.repeat(3, 1).t()
    elif layout == "scalar":
        cpu = cpu[2]
    elif layout == "empty":
        cpu = cpu[:0]
    value = cpu.to("flagos")
    for exponent in (2, 2.0):
        actual = torch.pow(value, exponent).cpu()
        expected = cpu.pow(exponent)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(torch.signbit(actual), torch.signbit(expected))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
@pytest.mark.parametrize("layout", ["dense", "transpose", "slice", "expand"])
def test_compiled_pointwise_schedule_matches_published_host(monkeypatch, dtype, layout):
    hosts = torch_fl.native_host._hosts
    original = hosts.original_ops["cos"].__globals__["cos_func"]
    compiled = hosts.resolve(original)
    value = torch.arange(21, dtype=dtype, device="flagos").reshape(3, 7)
    if layout == "transpose":
        value = value.t()
    elif layout == "slice":
        value = value[:, ::2]
    elif layout == "expand":
        value = value[:1].expand(3, 7)
    ndim, args, kwargs, enable = original.prepare_args(value)
    reference = original.instantiate(ndim)
    calls = []

    class Capture:
        def __init__(self, variant):
            self.variant = variant

        def __getitem__(self, grid):
            return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)

        def run(self, *args, grid, **kwargs):
            packed = tuple(
                ("pointer", v.data_ptr()) if hasattr(v, "data_ptr") else v for v in args
            )
            calls.append((self.variant, grid, packed, kwargs))

    name = f"{original._scalar_fn.__name__}_kernel_rank_{ndim}"
    for variant, suffix in enumerate(("", "_stride_constexpr")):
        monkeypatch.setitem(reference.__globals__, name + suffix, Capture(variant))
    reference(*args, enable_i64=enable, **kwargs)
    hosts.pointwise_kernel(compiled, ndim)
    native = hosts.runtime.EnflamePointwiseOverload(
        compiled, ndim, (Capture(0), Capture(1))
    )
    native(*args, enable_i64=enable, **kwargs)
    assert len(calls) == 2 and calls[0] == calls[1]
    output = kwargs["out0"].unwrap()
    cpp = compiled.instantiate(ndim).native
    plan = cpp.describe_enflame(
        (value,),
        [output],
        list(kwargs["out0"].shape),
        original.use_fast_path([output, value]),
    )
    packed = tuple(
        ("pointer", item.data_ptr()) if hasattr(item, "data_ptr") else item
        for item in plan["arguments"]
    )
    assert (plan["variant"], plan["grid"], packed, plan["options"]) == calls[0]
