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
