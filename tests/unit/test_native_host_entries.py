# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Schema binding and admission checks need no Torch or accelerator import."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entries = load("torch_fl/native_host/compiler/entries.py", "typed_host_entries")
codegen = load("scripts/codegen/native_source_entries.py", "typed_host_codegen")


class Type:
    def __init__(self, kind, element=None, elements=()):
        self._kind, self._element, self._elements = kind, element, elements

    def kind(self):
        return self._kind

    def getElementType(self):
        return self._element

    def elements(self):
        return self._elements


TENSOR = Type("TensorType")
INTEGER = Type("IntType")


def parameter(name, *, keyword=False, default=None, has_default=False):
    return dict(name=name, keyword=keyword, default=default, has_default=has_default)


def record(arguments, outputs=1):
    return {
        "op": "aten::test.Tensor",
        "host": "flag_gems.test",
        "abi": "test-abi",
        "arguments": [{"type": t, "keyword": k} for t, k in arguments],
        "outputs": outputs,
        "result_type": "at::Tensor" if outputs == 1 else "std::vector<at::Tensor>",
    }


def group(parameters, types, result=TENSOR):
    args = [SimpleNamespace(name=p["name"]) for p in parameters]
    return ("Host0", parameters, args, {}, types, result)


def test_positional_names_and_keyword_defaults_bind_separately():
    public = [
        parameter("A"),
        parameter("axis", keyword=True),
        parameter("out", keyword=True, has_default=True),
    ]
    bound = entries.bind_parameters(
        record([("at::Tensor", None), ("int64_t", "axis")]), public
    )
    assert bound == {
        "A": ("at::Tensor", "a0"),
        "axis": ("int64_t", "a1"),
        "out": ("std::nullopt_t", "std::nullopt"),
    }


@pytest.mark.parametrize(
    "arguments,public,reason",
    [
        ([("int64_t", None), ("int64_t", "axis")], [parameter("axis")], "duplicate"),
        ([], [parameter("out", keyword=True)], "missing"),
        ([("int64_t", "unknown")], [], "unconsumed"),
    ],
)
def test_argument_mismatches_decline_before_host_execution(arguments, public, reason):
    with pytest.raises(NotImplementedError, match=reason):
        entries.bind_parameters(record(arguments), public)


@pytest.mark.parametrize(
    "kind,predicate,conversion",
    [
        ("BoolType", "isBoolean()", "toBool()"),
        ("IntType", "isIntegral(false)", "toLong()"),
        ("FloatType", "isFloatingPoint()", "toDouble()"),
    ],
)
def test_scalar_admission_preserves_python_scalar_types(kind, predicate, conversion):
    check, value = entries.convert("at::Scalar", Type(kind), "alpha")
    assert check == "alpha." + predicate
    assert value == "alpha." + conversion


def test_optional_admission_checks_presence_before_dereference():
    check, value = entries.convert("std::optional<at::Scalar>", INTEGER, "alpha")
    assert check.startswith("(alpha.has_value() &&")
    assert value == "(*alpha).toLong()"
    check, value = entries.convert(
        "std::optional<at::Scalar>", Type("OptionalType", INTEGER), "alpha"
    )
    assert check.startswith("(!alpha.has_value() ||")
    assert "std::make_optional" in value and ": std::nullopt" in value


def test_none_and_dtype_have_explicit_native_representations():
    optional = Type("OptionalType", INTEGER)
    assert entries.convert("std::nullopt_t", optional, "std::nullopt") == (
        "true",
        "std::nullopt",
    )
    assert entries.convert("at::ScalarType", INTEGER, "dtype") == (
        "true",
        "static_cast<int64_t>(dtype)",
    )


def test_shape_lists_avoid_an_ivalue_conversion():
    assert entries.convert("at::IntArrayRef", Type("ListType", INTEGER), "shape") == (
        "true",
        "HostIntList(shape)",
    )
    check, value = entries.convert(
        "at::OptionalIntArrayRef",
        Type("OptionalType", Type("ListType", INTEGER)),
        "shape",
    )
    assert "has_value" in check and "HostIntList" in value
    assert "IValue" not in check + value


def test_wrapped_number_admission_does_not_read_tensor_data():
    check, value = entries.convert("at::Tensor", Type("FloatType"), "scalar")
    assert "TypedWrappedNumber" in check
    assert "isFloatingType" in check
    assert ".item()" not in check
    assert value == "scalar.item().toDouble()"
    check, _ = entries.convert("at::Tensor", TENSOR, "scalar")
    assert "defined()" in check and "!TypedWrappedNumber" in check


def test_multi_output_callback_keeps_order_and_owns_return_handles():
    code, records, rejected = entries.emit_entries(
        [record([("at::Tensor", None)], outputs=2)],
        {
            "test": [
                group(
                    [parameter("value")],
                    {"value": TENSOR},
                    Type("TupleType", elements=[TENSOR, TENSOR]),
                )
            ]
        },
    )
    assert not rejected
    assert records[0]["op"] == "aten::test.Tensor"
    assert "output={std::get<0>(result), std::get<1>(result)}" in code[0]
    assert "std::vector<at::Tensor>& output, const at::Tensor& a0" in code[0]
    assert "IValue" not in code[0]
    assert code[0].index("if(") < code[0].index("Host0::Forward")


def test_return_mismatch_and_missing_host_are_reported():
    code, records, rejected = entries.emit_entries(
        [record([("at::Tensor", None)], outputs=2)],
        {"test": [group([parameter("value")], {"value": TENSOR})]},
    )
    assert not code and not records
    assert "Tensor tuple" in rejected[0]["reason"]
    assert entries.emit_entries([record([])], {})[2] == [
        {"op": "aten::test.Tensor", "reason": "no lowered host variant"}
    ]


def test_partial_variant_binding_remains_observable():
    public = [parameter("value")]
    code, records, rejected = entries.emit_entries(
        [record([("at::Tensor", None)])],
        {
            "test": [
                group(public, {"value": TENSOR}),
                (
                    "Unsupported",
                    public,
                    [SimpleNamespace(name="value")],
                    {},
                    {"value": Type("GeneratorType")},
                    TENSOR,
                ),
            ]
        },
    )
    assert len(code) == len(records) == len(rejected) == 1
    assert records[0]["variants"] == ["Host0"]
    assert rejected[0]["namespace"] == "Unsupported"


def test_codegen_defers_generic_packing_until_fallback():
    declaration, call = codegen.wrap_call(
        "test.Tensor",
        "flag_gems.test",
        "TestSlot",
        'CallPythonOp_Generic("flag_gems.test", {x, axis})',
        [("const at::Tensor &", "x", None), ("int64_t", "axis", "axis")],
    )
    assert "static TypedSourceSlot TestSlot" in declaration
    assert "aten::test.Tensor" in declaration
    assert ".Call<at::Tensor>([&]() { return CallPythonOp_Generic" in call
    assert "static_cast<const at::Tensor&>(x)" in call
    assert "static_cast<const int64_t&>(axis)" in call


def test_abi_identity_depends_on_overload_host_types_and_keywords():
    def declaration(
        op="test.Tensor", host="flag_gems.test", t="at::Tensor", keyword=None, outputs=1
    ):
        return codegen.wrap_call(
            op, host, "Slot", "fallback()", [(t, "x", keyword)], outputs
        )[0]

    baseline = declaration()
    assert baseline == declaration()
    assert (
        len(
            {
                baseline,
                declaration(op="test.Scalar"),
                declaration(host="flag_gems.other"),
                declaration(t="at::Scalar"),
                declaration(keyword="out"),
                declaration(outputs=2),
            }
        )
        == 6
    )


def test_factory_preparation_only_runs_for_an_available_native_entry():
    _, call = codegen.wrap_call(
        "ones",
        "flag_gems.ones",
        "Slot",
        "fallback()",
        [],
        preparation="const auto device = resolve_device();",
    )
    assert call.index("if (!Slot.Available())") < call.index("resolve_device()")
