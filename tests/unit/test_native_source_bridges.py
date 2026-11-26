# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate generated source bridges against real Torch schemas, without a GPU."""

import importlib.util
import json
from pathlib import Path
import re

import pytest

torchgen = pytest.importorskip("torchgen")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "source_bridge_codegen", ROOT / "scripts/codegen/codegen_ops.py"
)
codegen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codegen)


@pytest.fixture(scope="module")
def functions():
    directory = Path(torchgen.__file__).parent / "packaged/ATen/native"
    parsed = codegen.parse_native_yaml(
        str(directory / "native_functions.yaml"), str(directory / "tags.yaml")
    )
    return {str(f.func.name): f for f in parsed.native_functions}


def test_copy_bridge_borrows_handles_and_calls_source_once(functions):
    header, source = codegen.gen_flaggems_source_bridges(
        functions, {"copy_": ("flag_gems.copy_", "inplace", [])}
    )
    assert (
        "const at::Tensor & self, const at::Tensor & src, bool non_blocking" in header
    )
    assert "CopyInplaceSourceHost" in header
    assert source.count('CallPythonOp_Generic("flag_gems.copy_"') == 1
    assert "{self, src, non_blocking}" in source
    assert "[&]() { return CallPythonOp_Generic" in source
    assert ".copy_(" not in source and "redispatch" not in source
    assert "UnboxToFlagos" not in source and "device_boxing.h" not in source
    assert "REGISTER_IMPL_TO_DISPATCHER" not in source


def test_source_bridges_do_not_duplicate_dispatcher_registration(functions):
    header, source = codegen.gen_flaggems_source_bridges(
        functions,
        {
            "add.Tensor": ("flag_gems.add", "functional_pure", []),
            "_reshape_alias": ("flag_gems._reshape_alias", "functional_pure", []),
        },
    )
    assert "AddTensor" not in header + source
    assert "PrivReshapeAliasSourceHost" in header
    assert "at::IntArrayRef size, at::IntArrayRef stride" in header
    assert '"aten::_reshape_alias"' in source
    assert "REGISTER_IMPL_TO_DISPATCHER" not in source


def test_committed_bridges_match_their_catalog_and_faithful_schemas(functions):
    directory = ROOT / "csrc/aten/generated"
    source = (directory / "flaggems_source_bridges.cc").read_text()
    discovered = {}
    decoder = json.JSONDecoder()
    for payload in re.findall(r"^static TypedSourceSlot \w+\((.*)\);$", source, re.M):
        fields = []
        for _ in range(4):
            value, end = decoder.raw_decode(payload)
            fields.append(value)
            payload = payload[end:].lstrip(", ")
        op, host, _, catalog = fields
        op = op.removeprefix("aten::")
        record = json.loads(catalog)
        keywords = {a["keyword"] for a in record["arguments"] if a["keyword"]}
        kwargs = [
            (str(a.type), a.name)
            for a in functions[op].func.arguments.flat_all
            if a.name in keywords
        ]
        discovered[op] = (host, codegen.detect_category(functions[op]), kwargs)
    assert "copy_" in discovered
    header, generated = codegen.gen_flaggems_source_bridges(functions, discovered)
    assert header == (directory / "flaggems_source_bridges.h").read_text()
    assert generated == source


def test_out_adapters_preserve_storage_and_use_configured_functional_routes(functions):
    source = codegen.gen_flaggems_out_adapters(functions)
    assert source.count("resize_output(") == 10
    assert source.count("assert_no_internal_overlap(") == 13
    assert "assert_no_overlap(out, input)" in source
    for call in (
        "add_tensor_dispatcher(self, other, alpha)",
        "sub_tensor_dispatcher(self, other, alpha)",
        "mul_tensor_dispatcher(self, other)",
        "mean_dim_dispatcher(self, dim, keepdim, dtype.value_or(out.scalar_type()))",
        "cat_dispatcher(tensors, dim)",
    ):
        assert call in source
    assert "USE_DCU" in source
    assert "Backend::kFlagGems, MeanOutIluvatar" in source
    assert "Backend::kFlagGems, NllLossMeanForwardOutNvidia" in source
    assert "Backend::kFlagGems, NllLossMeanBackwardOutNvidia" in source
    assert "Backend::kFlagGems, LogSoftmaxBackwardOutNvidia" in source
    assert "Backend::kFlagGems, SoftmaxBackwardOutNvidia" in source
    assert "Backend::kFlagGems, SumIntlistOutNvidia" in source
    assert "Backend::kFlagGems, MulOutFlagGems" in source
    assert "DeviceBoxingGuard" not in source and "redispatch" not in source
    committed = (ROOT / "csrc/aten/generated/flaggems_python_kernels.cc").read_text()
    assert committed.endswith(source)


def test_hcu_floating_square_keeps_the_configured_mul_dispatcher(functions):
    func = functions["pow.Tensor_Scalar"]
    with codegen.local.parametrize(
        use_const_ref_for_mutable_tensors=func.use_const_ref_for_mutable_tensors,
        use_ilistref_for_tensor_lists=not codegen.should_use_arrayref(func),
    ):
        _, result, args = codegen.unified_sig(func)
    fn_type, _ = codegen.schema_to_cpp_name("pow.Tensor_Scalar")
    generated = codegen.gen_flaggems_python_kernel(
        "pow.Tensor_Scalar",
        fn_type,
        result,
        args,
        "flag_gems.pow_tensor_scalar",
        "functional_pure",
        func,
        typed=True,
    )
    assert "MulTensorFnSource.Available()" in generated
    assert "c10::isFloatingType(self.scalar_type())" in generated
    assert "exponent.toDouble() == 2.0" in generated
    assert "return at::mul(self, self);" in generated
    assert "DeviceBoxingGuard" not in generated
    assert 'CallPythonOp_Generic("flag_gems.pow_tensor_scalar"' in generated
    assert (
        generated
        in (ROOT / "csrc/aten/generated/flaggems_python_kernels.cc").read_text()
    )


@pytest.mark.parametrize("op", codegen.FLAGGEMS_HCU_METADATA_ADAPTERS)
def test_hcu_metadata_adapters_reuse_existing_aten_primitives(functions, op):
    func = functions[op]
    with codegen.local.parametrize(
        use_const_ref_for_mutable_tensors=func.use_const_ref_for_mutable_tensors,
        use_ilistref_for_tensor_lists=not codegen.should_use_arrayref(func),
    ):
        _, result, args = codegen.unified_sig(func)
    fn_type, _ = codegen.schema_to_cpp_name(op)
    host = {
        "_unsafe_view": "_unsafe_view",
        "transpose.int": "transpose",
        "unsqueeze": "unsqueeze",
        "slice.Tensor": "slice",
    }[op]
    source = codegen.gen_flaggems_python_kernel(
        op,
        fn_type,
        result,
        args,
        "flag_gems." + host,
        "functional_pure",
        func,
        typed=True,
    )
    target = codegen.FLAGGEMS_HCU_METADATA_ADAPTERS[op]
    assert f"return at::native::flagos::{target}(" in source
    assert "#if defined(USE_DCU)" in source
    assert "DeviceBoxingGuard" not in source
    assert (
        source in (ROOT / "csrc/aten/generated/flaggems_python_kernels.cc").read_text()
    )
