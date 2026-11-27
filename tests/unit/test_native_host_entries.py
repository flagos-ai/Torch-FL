# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Schema binding and admission checks need no Torch or accelerator import."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
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


def test_published_source_metadata_guard_preserves_unrestricted_selections(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    source = tmp_path / "guard.cc"
    source.write_text(
        r"""
#include <optional>
#include <vector>
#include <cstdint>
#include <cassert>
namespace c10 {
struct Device {
  int index=0; bool cpu=false;
  bool is_privateuseone() const {return !cpu;}
  bool is_cpu() const {return cpu;}
  bool operator!=(const Device& other) const {return index!=other.index || cpu!=other.cpu;}
};
template<class T> struct List:std::vector<T> {using std::vector<T>::vector;};
}
namespace at {
enum class ScalarType {Float, BFloat16, Half};
struct Tensor {
  struct Impl {bool wrapped=false; bool is_wrapped_number() const {return wrapped;}} impl;
  ScalarType type=ScalarType::Float; int64_t rank=4, inner=1; c10::Device dev{};
  bool defined() const {return true;}
  bool is_complex() const {return false;}
  c10::Device device() const {return dev;}
  ScalarType scalar_type() const {return type;}
  int64_t dim() const {return rank;}
  int64_t numel() const {return 16;}
  int64_t size(int64_t) const {return 2;}
  int64_t stride(int64_t) const {return inner;}
  const Impl* unsafeGetTensorImpl() const {return &impl;}
};
template<class T> struct ArrayRef:std::vector<T> {using std::vector<T>::vector;};
struct ITensorListRef:std::vector<Tensor> {using std::vector<Tensor>::vector;};
}
"""
        + entries.HEADER[
            entries.HEADER.index("class PublishedSourceGuard") : entries.HEADER.index(
                "inline bool TypedWrappedNumber"
            )
        ]
        + r"""
int main() {
 const uint64_t mask=3;
 at::Tensor first,second;
 c10::List<at::Tensor> inputs{first,second};
 assert(PublishedSourceAcceptsRestricted(mask,true,2,false,inputs,int64_t(-1)));
 inputs[1].type=at::ScalarType::BFloat16;
 assert(!PublishedSourceAcceptsRestricted(mask,true,2,false,inputs));
 assert(PublishedSourceAccepts(mask,inputs));
 inputs[1]=second; inputs[1].rank=2;
 assert(!PublishedSourceAcceptsRestricted(mask,true,2,false,inputs));
 assert(PublishedSourceAccepts(mask,inputs));
 inputs[1]=second; inputs[1].dev.index=1;
 assert(!PublishedSourceAcceptsRestricted(mask,true,2,false,inputs));
 assert(!PublishedSourceAccepts(mask,inputs));
 inputs[1]=second; inputs[1].dev.cpu=true;
 assert(!PublishedSourceAcceptsRestricted(mask,true,2,false,inputs));
 inputs.resize(1);
 assert(!PublishedSourceAcceptsRestricted(mask,true,2,false,inputs,int64_t(7)));
 assert(PublishedSourceAccepts(mask,inputs));
 inputs.clear(); assert(!PublishedSourceAccepts(mask,inputs));
 assert(!PublishedSourceAcceptsRestricted(mask,true,2,false,inputs));
 inputs={first,second};
 assert(PublishedSourceAcceptsRestricted(mask,true,2,true,inputs));
 inputs[1].inner=2;
 assert(!PublishedSourceAcceptsRestricted(mask,true,2,true,inputs));
 assert(PublishedSourceAccepts(mask,inputs));
 inputs[1]=second; std::optional<at::Tensor> absent;
 assert(PublishedSourceAcceptsRestricted(mask,true,2,false,inputs,absent));
 at::Tensor scalar;
 scalar.dev.cpu=true; scalar.rank=0; scalar.impl.wrapped=true;
 assert(!PublishedSourceAccepts(mask,first,scalar));
 assert(PublishedSourceAccepts<true>(mask,first,scalar));
 assert(!PublishedSourceAccepts<true>(mask,scalar));
 assert(!PublishedSourceAcceptsRestricted<true>(mask,true,2,false,first,scalar));
 assert(PublishedSourceAcceptsRestricted<true>(mask,true,2,false,inputs,scalar));
 scalar.impl.wrapped=false;
 assert(!PublishedSourceAccepts<true>(mask,first,scalar));
}
"""
    )
    executable = tmp_path / "guard"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-Werror", str(source), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    subprocess.run([str(executable)], check=True, timeout=10)


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
    assert value == "TypedWrappedScalar(scalar).toDouble()"
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


def test_rank_specialization_checks_tensor_arguments_before_calling_host():
    arguments = [("at::Tensor", None), ("int64_t", None), ("at::Tensor", None)]
    public = [parameter("destination"), parameter("dim"), parameter("source")]
    kinds = {"destination": TENSOR, "dim": INTEGER, "source": TENSOR}
    code, records, rejected = entries.emit_entries(
        [record(arguments)],
        {"test": [group(public, kinds)]},
        tensor_ranks={"test": {"destination": 4, "source": 4}},
    )
    assert not rejected and len(records) == 1
    for guard in ("a0.dim()==4", "a2.dim()==4"):
        assert code[0].index(guard) < code[0].index("Host0::Forward")
    assert "a1.dim()" not in code[0]
    assert "return false;" in code[0]


@pytest.mark.parametrize("restricted", [False, True])
def test_published_selection_emits_its_guard_before_the_host(restricted):
    selection = {
        "id": "published_test",
        "op": "aten::test.Tensor",
        "dtypes": ["Float", "BFloat16"],
    }
    if restricted:
        selection.update(
            same_tensor_metadata=True, min_tensors=2, unit_stride_axis=True
        )
    code, records, rejected = entries.emit_entries(
        [record([("at::TensorList", None)])],
        {
            "published_test": [
                group([parameter("values")], {"values": Type("ListType", TENSOR)})
            ]
        },
        [selection],
    )
    assert not rejected
    body = code[-1]
    guard = (
        "PublishedSourceAcceptsRestricted(" if restricted else "PublishedSourceAccepts("
    )
    assert guard in body
    assert (", true, 2, true, a0)" in body) is restricted
    assert body.index(guard) < body.index("Host0::Forward")
    assert "++SourceVariantHits_published_test" in body
    assert records[0]["source_variants"][0]["id"] == "published_test"


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
