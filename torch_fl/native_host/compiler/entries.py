# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Bind generated C++ call schemas to source-derived typed host variants."""

import json
import math
import re


def normalize(value):
    return re.sub(r"\s+", "", value).replace("::std::", "std::")


def optional_element(source):
    if source.startswith("std::optional<"):
        return source[len("std::optional<") : -1]
    if source in ("at::OptionalIntArrayRef", "c10::OptionalArrayRef<int64_t>"):
        return "at::IntArrayRef"
    return None


def convert(source, target, value):
    """Return a side-effect-free admission predicate and a typed conversion."""
    source = normalize(source)
    kind = target.kind()
    element = optional_element(source)
    if kind == "OptionalType":
        if source == "std::nullopt_t":
            return "true", "std::nullopt"
        if element:
            check, expression = convert(element, target.getElementType(), f"(*{value})")
            return (
                f"(!{value}.has_value() || ({check}))",
                f"({value}.has_value() ? std::make_optional({expression}) : std::nullopt)",
            )
        return convert(source, target.getElementType(), value)
    if element:
        check, expression = convert(element, target, f"(*{value})")
        return f"({value}.has_value() && ({check}))", expression
    if kind == "TensorType" and source == "at::Tensor":
        return f"({value}.defined() && !TypedWrappedNumber({value}))", value
    if kind in ("BoolType", "IntType", "FloatType"):
        native, method, test = {
            "BoolType": ("bool", "toBool", "isBoolean()"),
            "IntType": ("int64_t", "toLong", "isIntegral(false)"),
            "FloatType": ("double", "toDouble", "isFloatingPoint()"),
        }[kind]
        if source == "at::Scalar":
            return f"{value}.{test}", f"{value}.{method}()"
        if source == "at::Tensor":
            dtype = {
                "BoolType": f"{value}.scalar_type()==at::kBool",
                "IntType": f"c10::isIntegralType({value}.scalar_type(), false)",
                "FloatType": f"c10::isFloatingType({value}.scalar_type())",
            }[kind]
            return (
                f"(TypedWrappedNumber({value}) && {dtype})",
                f"{value}.item().{method}()",
            )
        if source == native:
            return "true", value
        if kind == "IntType" and source in (
            "at::ScalarType",
            "c10::ScalarType",
            "at::Layout",
            "c10::Layout",
            "at::MemoryFormat",
            "c10::MemoryFormat",
        ):
            return "true", f"static_cast<int64_t>({value})"
        raise NotImplementedError(f"{source} to {kind}")
    if kind == "StringType" and source in (
        "c10::string_view",
        "std::string_view",
        "std::string",
    ):
        return "true", f"std::string({value})"
    if kind == "DeviceObjType" and source in ("at::Device", "c10::Device"):
        return "true", value
    if kind == "ListType":
        item = target.getElementType().kind()
        if item == "IntType" and source in (
            "at::IntArrayRef",
            "c10::ArrayRef<int64_t>",
        ):
            return "true", f"HostIntList({value})"
        if item == "TensorType" and source in (
            "at::TensorList",
            "at::ITensorListRef",
            "c10::ArrayRef<at::Tensor>",
        ):
            return "true", f"TypedTensorList({value})"
        native = {
            "IntType": "int64_t",
            "TensorType": "at::Tensor",
            "BoolType": "bool",
            "FloatType": "double",
        }.get(item)
        if native and source == f"c10::List<{native}>":
            return "true", value
        if item == "OptionalType" and source == "c10::List<std::optional<at::Tensor>>":
            return "true", value
        raise NotImplementedError(f"{source} to {target}")
    raise NotImplementedError(f"{source} to {target}")


def default_argument(value):
    if value is None:
        return "std::nullopt_t", "std::nullopt"
    if type(value) is bool:
        return "bool", "true" if value else "false"
    if type(value) is int:
        return "int64_t", f"int64_t({value})"
    if type(value) is float:
        if math.isnan(value):
            return "double", "std::numeric_limits<double>::quiet_NaN()"
        if math.isinf(value):
            return "double", (
                "-" if value < 0 else ""
            ) + "std::numeric_limits<double>::infinity()"
        return "double", repr(value)
    if type(value) is str:
        return "std::string", json.dumps(value)
    raise NotImplementedError("unsupported host default")


def bind_parameters(record, public):
    arguments = record["arguments"]
    positional = [i for i, a in enumerate(arguments) if a["keyword"] is None]
    keywords = {
        a["keyword"]: i for i, a in enumerate(arguments) if a["keyword"] is not None
    }
    position = 0
    used = set()
    bound = {}
    for parameter in public:
        name = parameter["name"]
        index = None
        if not parameter["keyword"] and position < len(positional):
            index = positional[position]
            position += 1
            if name in keywords:
                raise NotImplementedError("duplicate host parameter: " + name)
        elif name in keywords:
            index = keywords[name]
        if index is not None:
            used.add(index)
            bound[name] = (arguments[index]["type"], "a" + str(index))
        elif parameter["has_default"]:
            bound[name] = default_argument(parameter["default"])
        else:
            raise NotImplementedError("missing host parameter: " + name)
    if len(used) != len(arguments):
        raise NotImplementedError("unconsumed native call arguments")
    return bound


def emit_entries(catalog, groups, selections=()):
    """Generate schema-specific callbacks; unsupported conversions stay visible."""
    declarations, records, rejected = [], [], []
    declarations.extend(
        f"std::atomic<int64_t> SourceVariantHits_{row['id']}{{0}};"
        for row in selections
    )
    admitted_selections = set()
    for record in catalog:
        name = record["host"].removeprefix("flag_gems.")
        group = groups.get(name, [])
        alternatives = [row for row in selections if row["op"] == record["op"]]
        candidates = [
            (entry, row) for row in alternatives for entry in groups.get(row["id"], [])
        ] + [(entry, None) for entry in group]
        symbol = "TypedEntry" + str(len(records))
        lines = [
            "bool "
            + symbol
            + "("
            + record["result_type"]
            + "& output"
            + "".join(
                f", const {a['type']}& a{i}" for i, a in enumerate(record["arguments"])
            )
            + ") {"
        ]
        accepted = []
        selected = []
        for entry, selection in candidates:
            namespace, public, graph_args, types, jit_types, result_type = entry
            try:
                bound = bind_parameters(record, public)
                checks, values = [], {}
                if selection is not None:
                    mask = " | ".join(
                        "(uint64_t{1} << static_cast<unsigned>(at::ScalarType::"
                        + dtype
                        + "))"
                        for dtype in selection["dtypes"]
                    )
                    checks.append(
                        "PublishedSourceAccepts("
                        + mask
                        + ", "
                        + ", ".join(f"a{i}" for i in range(len(record["arguments"])))
                        + ")"
                    )
                for argument in graph_args:
                    source, expression = bound[argument.name]
                    check, expression = convert(
                        source, jit_types[argument.name], expression
                    )
                    checks.append(check)
                    values[argument.name] = expression
                invocation = (
                    namespace
                    + "::Forward("
                    + ", ".join(values[a.name] for a in graph_args)
                    + ")"
                )
                if record["outputs"] == 1:
                    if result_type.kind() != "TensorType":
                        raise NotImplementedError(
                            "native call requires one Tensor result"
                        )
                    result = "output=" + invocation + ";"
                else:
                    if (
                        result_type.kind() != "TupleType"
                        or len(result_type.elements()) != record["outputs"]
                        or any(t.kind() != "TensorType" for t in result_type.elements())
                    ):
                        raise NotImplementedError(
                            "native call requires a fixed Tensor tuple"
                        )
                    result = (
                        "auto result="
                        + invocation
                        + ";output={"
                        + ", ".join(
                            f"std::get<{i}>(result)" for i in range(record["outputs"])
                        )
                        + "};"
                    )
                lines.append(
                    "if("
                    + " && ".join(checks or ["true"])
                    + ") {"
                    + result
                    + (
                        f"++SourceVariantHits_{selection['id']};"
                        if selection is not None
                        else ""
                    )
                    + "return true;}"
                )
                accepted.append(namespace)
                if selection is not None:
                    admitted_selections.add(selection["id"])
                    selected.append({**selection, "namespace": namespace})
            except NotImplementedError as error:
                rejected.append(
                    {"op": record["op"], "namespace": namespace, "reason": str(error)}
                )
        if not accepted:
            if not group:
                rejected.append(
                    {"op": record["op"], "reason": "no lowered host variant"}
                )
            continue
        lines.append("return false; }")
        declarations.append("\n".join(lines))
        entry = {**record, "symbol": symbol, "variants": accepted}
        if selected:
            entry["source_variants"] = selected
        records.append(entry)
    missing = {row["id"] for row in selections} - admitted_selections
    if missing:
        raise ValueError(
            f"published source selections did not lower: {sorted(missing)}"
        )
    return declarations, records, rejected


HEADER = r"""
#include <atomic>
class PublishedSourceGuard {
 uint64_t mask_;
 std::optional<c10::Device> device_;
 public:
 explicit PublishedSourceGuard(uint64_t mask):mask_(mask){}
 bool Check(const at::Tensor& value) {
  if(!value.defined() || !value.device().is_privateuseone() ||
     !(mask_ & (uint64_t{1} << static_cast<unsigned>(value.scalar_type()))))return false;
  if(device_ && *device_!=value.device())return false;
  device_=value.device();return true;
 }
 template<class T> bool Check(const T&){return true;}
 template<class T> bool Check(const std::optional<T>& value){return !value || Check(*value);}
 template<class T> bool Check(const c10::List<T>& values){
  for(const auto& value:values)if(!Check(static_cast<T>(value)))return false;
  return true;
 }
 template<class T> bool Check(at::ArrayRef<T> values){
  for(const auto& value:values)if(!Check(value))return false;
  return true;
 }
 bool Check(const at::ITensorListRef& values){
  for(const auto& value:values)if(!Check(value))return false;
  return true;
 }
 bool HasTensor() const{return device_.has_value();}
};
template<class... Args> bool PublishedSourceAccepts(uint64_t mask,const Args&... args){
 PublishedSourceGuard guard(mask);
 return (guard.Check(args) && ...) && guard.HasTensor();
}
inline bool TypedWrappedNumber(const at::Tensor& value) {
 return value.defined() && value.device().is_cpu() && value.unsafeGetTensorImpl()->is_wrapped_number();
}
template <class T> c10::List<at::Tensor> TypedTensorList(const T& values) {
 c10::List<at::Tensor> result;
 result.reserve(values.size());
 for(const auto& value:values) result.push_back(value);
 return result;
}
"""
