# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Installed native routes needed by the unmodified eager Qwen3 workload."""

from pathlib import Path
import os
import subprocess
import sys
import textwrap

import pytest
import torch
import torch_fl

pytestmark = [pytest.mark.metax, pytest.mark.native_source_host, pytest.mark.flaggems]


@pytest.fixture(autouse=True)
def native_contract():
    assert torch_fl._build_config.NATIVE_HOSTS
    assert "boxing" not in torch_fl._build_config.KERNELS
    assert (
        Path(torch_fl.backend_config_path()).name == "backends_metax_native_host.conf"
    )
    import flag_gems

    assert flag_gems.runtime.device.name == "flagos"


@pytest.mark.parametrize("operation", ["add", "mul", "mean", "cat", "slice", "mul_"])
@torch.no_grad()
def test_qwen_native_route_matches_cpu(operation):
    generator = torch.Generator().manual_seed(845)
    reference = torch.randn(1, 60, 1024, generator=generator)
    if operation in ("add", "mul", "cat"):
        reference = reference.bfloat16()
    value = reference.to("flagos:0")
    if operation == "add":
        actual, expected = value + value, reference + reference
    elif operation == "mul":
        actual, expected = value * value, reference * reference
    elif operation == "mean":
        actual, expected = (
            value.mean(-1, keepdim=True),
            reference.mean(-1, keepdim=True),
        )
    elif operation == "cat":
        actual, expected = (
            torch.cat([value, value], dim=1),
            torch.cat([reference, reference], dim=1),
        )
    elif operation == "slice":
        actual, expected = value[:, 1::2, :], reference[:, 1::2, :]
        assert actual.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        assert actual.stride() == expected.stride()
    else:
        address = value.data_ptr()
        actual, expected = value.mul_(value), reference.mul_(reference)
        assert actual.data_ptr() == address
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("tokens", [1, 60])
@pytest.mark.parametrize("out_variant", [False, True])
@torch.no_grad()
def test_qwen_rope_small_k_bmm(tokens, out_variant):
    generator = torch.Generator().manual_seed(832)
    left = torch.randn(1, 64, 1, generator=generator) * 0.25
    right = torch.randn(1, 1, tokens, generator=generator) * 0.25
    a, b = left.to("flagos:0"), right.to("flagos:0")
    if out_variant:
        storage = torch.empty(1, 64, tokens * 2, device="flagos:0")
        output = storage[..., ::2]
        actual = torch.bmm(a, b, out=output)
        assert actual.data_ptr() == output.data_ptr()
        assert actual.stride() == output.stride()
    else:
        actual = torch.bmm(a, b)
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(
        actual.cpu(), torch.bmm(left, right), atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize("allow_tf32", [False, True])
def test_native_rope_outer_product_honors_the_callers_tf32_context(allow_tf32):
    environment = os.environ.copy()
    environment.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import torch
                import torch_fl

                torch.backends.cuda.matmul.allow_tf32 = {allow_tf32!r}
                bits = torch.tensor([
                    0x00000000, 0x80000000, 0x3f800fff, 0x3f801000,
                    0x3f803000, 0xbf801000, 0xbf803000, 0x00000001,
                    0x7f800000, 0xff800000, 0x7fc00000,
                ], dtype=torch.int64).to(torch.int32)
                left = bits.view(torch.float32).reshape(1, -1, 1)
                right = torch.tensor([0.031249, 60.0, 61.0]).reshape(1, 1, -1)

                def rounded(value):
                    raw = value.view(torch.int32)
                    special = (raw & 0x7fffffff) >= 0x7f800000
                    low_bit = (raw >> 13) & 1
                    quantized = ((raw + 4095 + low_bit) & -8192).view(torch.float32)
                    return torch.where(special, value, quantized)

                expected = (rounded(left) * rounded(right)
                            if {allow_tf32!r} else left * right)
                if {allow_tf32!r}:
                    assert not torch.equal(expected[:, 2:7], (left * right)[:, 2:7])
                a, b = left.to('flagos:0'), right.to('flagos:0')
                actual = torch.bmm(a, b)
                storage = torch.empty(1, left.size(1), 6, device='flagos:0')
                output = storage[..., ::2]
                returned = torch.bmm(a, b, out=output)
                assert returned.data_ptr() == output.data_ptr()
                assert returned.stride() == output.stride()
                torch_fl.flagos.synchronize()
                for value in (actual.cpu(), returned.cpu()):
                    torch.testing.assert_close(value, expected, atol=0, rtol=0,
                                               equal_nan=True)
                    assert torch.equal(torch.signbit(value[:, :2]),
                                       torch.signbit(expected[:, :2]))
                print('TF32 context and strided out passed')
                """
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TF32 context and strided out passed" in result.stdout


@torch.no_grad()
def test_missing_numerical_route_still_rejects_cpu_fallback():
    value = torch.ones(4, device="flagos:0")
    with pytest.raises(
        RuntimeError, match="native FlagGems hosts do not permit CPU fallback"
    ):
        torch.ops.aten.igamma.default(value, value)


@torch.no_grad()
def test_float64_matmul_rejects_the_explicit_cpu_escape():
    value = torch.ones(2, 2, dtype=torch.float64).to("flagos:0")
    with pytest.raises(
        RuntimeError, match="CPU fallback: aten::mm .*host matmul escape"
    ):
        torch.mm(value, value)


@pytest.mark.parametrize("cast", [False, True])
@torch.no_grad()
def test_strided_device_copy_stays_on_the_published_gpu_source(cast):
    reference = torch.arange(17 * 38, dtype=torch.float32).reshape(17, 38) / 64
    source = reference.to("flagos:0")[:, ::2]
    dtype = torch.bfloat16 if cast else torch.float32
    storage = torch.empty(17, 38, dtype=dtype, device="flagos:0")
    output = storage[:, ::2]
    result = output.copy_(source)
    assert result.data_ptr() == output.data_ptr()
    assert result.stride() == output.stride()
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(
        result.cpu(), reference[:, ::2].to(dtype), atol=0, rtol=0
    )


@torch.no_grad()
def test_unsupported_device_copy_rejects_its_explicit_cpu_escape():
    source = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64).to("flagos:0")
    output = torch.empty(2, dtype=torch.float32, device="flagos:0")
    with pytest.raises(RuntimeError, match="CPU fallback: device copy/cast"):
        output.copy_(source)


@torch.no_grad()
def test_unsupported_device_gather_rejects_its_explicit_cpu_escape():
    source = torch.tensor([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=torch.complex64)
    value = source.to("flagos:0").t()
    with pytest.raises(RuntimeError, match="CPU fallback: device contiguous/clone"):
        value.contiguous()


@pytest.mark.parametrize("shape", [(60, 61), (1, 62)])
@torch.no_grad()
def test_causal_mask_bool_to_bf16_copy_and_inplace_multiply_stay_on_device(shape):
    reference = torch.arange(shape[1]).remainder(3).ne(0).reshape(1, -1)
    source = reference.to("flagos:0")
    output = torch.empty(shape, dtype=torch.bfloat16, device="flagos:0")
    output.copy_(source)
    value = source.to(torch.bfloat16)
    mask = torch.full(shape, -16.0, dtype=torch.bfloat16, device="flagos:0")
    mask.mul_(source)
    torch_fl.flagos.synchronize()
    expected = reference.to(torch.bfloat16)
    torch.testing.assert_close(output.cpu(), expected.expand(shape), atol=0, rtol=0)
    torch.testing.assert_close(value.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(
        mask.cpu(), -16.0 * expected.expand(shape), atol=0, rtol=0
    )


def _exact_bf16(shape):
    count = 1
    for extent in shape:
        count *= extent
    return (
        ((torch.arange(count).remainder(13) - 6).float() / 8).reshape(shape).bfloat16()
    )


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("rows", [1, 60])
@pytest.mark.parametrize("strided", [False, True])
@torch.no_grad()
def test_native_bf16_sdk_matrix_layouts(rank, rows, strided):
    assert "vendor" in torch_fl._build_config.KERNELS
    batch = (2,) if rank == 3 else ()
    left, right = _exact_bf16((*batch, rows, 64)), _exact_bf16((*batch, 64, 32))
    if strided:
        a = left.transpose(-1, -2).contiguous().to("flagos:0").transpose(-1, -2)
        b = right.transpose(-1, -2).contiguous().to("flagos:0").transpose(-1, -2)
    else:
        a, b = left.to("flagos:0"), right.to("flagos:0")
    function = torch.mm if rank == 2 else torch.bmm
    name = "mm" if rank == 2 else "bmm"
    before = dict(torch.ops.flagos.native_metax_sdk_hits())
    actual = function(a, b)
    torch_fl.flagos.synchronize()
    after = dict(torch.ops.flagos.native_metax_sdk_hits())
    assert after[name] == before[name] + 1
    torch.testing.assert_close(actual.cpu(), function(left, right), atol=0, rtol=0)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("contracted", [0, 64])
@torch.no_grad()
def test_native_bf16_sdk_empty_matrices(rank, contracted):
    batch = (2,) if rank == 3 else ()
    rows = 3 if contracted == 0 else 0
    left, right = (
        _exact_bf16((*batch, rows, contracted)),
        _exact_bf16((*batch, contracted, 5)),
    )
    function = torch.mm if rank == 2 else torch.bmm
    actual = function(left.to("flagos:0"), right.to("flagos:0"))
    torch_fl.flagos.synchronize()
    torch.testing.assert_close(actual.cpu(), function(left, right), atol=0, rtol=0)


def test_native_bf16_sdk_audit_preserves_the_cpu_core_and_declared_dtypes():
    from torch_fl.native_host import vendor

    report = vendor.audit()
    assert report["vendor"] == "metax"
    report = report.get("sdk_precision", report)
    assert {row["op"]: row["dtypes"] for row in report["policy"]["operators"]} == {
        "mm": ["BFloat16"],
        "bmm": ["BFloat16"],
    }
    assert set(report["numerical_libraries"]) == {"mcblas", "mcblas_lt", "mcruntime"}
    assert torch.__version__ == "2.10.0+cpu"


@pytest.mark.parametrize("allow_reduction", [False, True])
@torch.no_grad()
def test_native_bf16_sdk_preserves_the_callers_reduction_setting(allow_reduction):
    settings = torch.backends.cuda.matmul
    original = settings.allow_bf16_reduced_precision_reduction
    try:
        settings.allow_bf16_reduced_precision_reduction = allow_reduction
        left = torch.full((4, 64), 0.125, dtype=torch.bfloat16, device="flagos:0")
        right = torch.full((64, 3), 0.25, dtype=torch.bfloat16, device="flagos:0")
        actual = torch.mm(left, right)
        torch_fl.flagos.synchronize()
        assert settings.allow_bf16_reduced_precision_reduction == allow_reduction
        torch.testing.assert_close(
            actual.cpu(), torch.full((4, 3), 2.0, dtype=torch.bfloat16), atol=0, rtol=0
        )
    finally:
        settings.allow_bf16_reduced_precision_reduction = original


@torch.no_grad()
def test_native_bf16_sdk_rejects_an_unenforceable_split_k_setting():
    settings = torch.backends.cuda.matmul
    original = settings.allow_bf16_reduced_precision_reduction
    try:
        settings.allow_bf16_reduced_precision_reduction = (False, False)
        left = torch.ones((2, 2), dtype=torch.bfloat16, device="flagos:0")
        with pytest.raises(RuntimeError, match="split-K prohibition"):
            torch.mm(left, left)
    finally:
        settings.allow_bf16_reduced_precision_reduction = original


def _vendor_mean_reference(source):
    # Independent transcription of the published MACA ReduceConfigMaca and
    # continuous-reduce host/kernel contract, evaluated only for test evidence.
    rows, columns = source.shape
    vector = 4 if columns > 128 else 1
    extent = columns // vector
    width = min(extent, 16 if columns <= 256 else 64)
    height = min(1 << (min(rows, 512).bit_length() - 1), 512 // width)
    width = min(extent, 512 // height)
    accumulators = torch.zeros(rows, width, dtype=torch.float32)
    for start in range(0, columns, width * vector):
        values = source[:, start : start + width * vector].reshape(rows, width, vector)
        for component in range(vector):
            accumulators = accumulators + values[:, :, component]
    while accumulators.size(1) > 64:
        half = accumulators.size(1) // 2
        accumulators = accumulators[:, :half] + accumulators[:, half:]
    while accumulators.size(1) > 1:
        accumulators = accumulators[:, ::2] + accumulators[:, 1::2]
    return accumulators / columns


@pytest.mark.parametrize("columns", [128, 1024])
@pytest.mark.parametrize("rows", [1, 8, 16, 60, 480, 960])
@torch.no_grad()
def test_native_fp32_mean_preserves_the_vendor_reduction_tree(rows, columns):
    reference = torch.randn(
        rows, columns, generator=torch.Generator().manual_seed(17002)
    )
    runtime = torch_fl.native_host._hosts.runtime
    before = runtime.metax_mean_calls()
    actual = reference.to("flagos:0").mean(-1, keepdim=True)
    torch_fl.flagos.synchronize()
    assert runtime.metax_mean_calls() == before + 1
    torch.testing.assert_close(
        actual.cpu(), _vendor_mean_reference(reference), atol=0, rtol=0
    )


@torch.no_grad()
def test_native_fp32_mean_unmeasured_layout_keeps_the_original_route():
    source = torch.randn(4, 256, generator=torch.Generator().manual_seed(17003))
    value = source.to("flagos:0")[:, ::2]
    runtime = torch_fl.native_host._hosts.runtime
    before = runtime.metax_mean_calls()
    actual = value.mean(-1)
    torch_fl.flagos.synchronize()
    assert runtime.metax_mean_calls() == before
    torch.testing.assert_close(
        actual.cpu(), source[:, ::2].mean(-1), atol=1e-3, rtol=1e-3
    )


@torch.no_grad()
def test_native_fp32_mean_rank_one_preserves_scalar_shape():
    source = torch.randn(1, 128, generator=torch.Generator().manual_seed(17004))
    actual = source.flatten().to("flagos:0").mean(-1)
    torch_fl.flagos.synchronize()
    assert actual.shape == torch.Size([])
    torch.testing.assert_close(
        actual.cpu(), _vendor_mean_reference(source).reshape(()), atol=0, rtol=0
    )
