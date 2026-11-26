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
                f"TypedWrappedScalar({value}).{method}()",
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


TSINGMICRO_EMBEDDING_GUARD = r"""
inline bool TsingMicroEmbeddingBFloat16Accepts(
    const at::Tensor& gradient, const at::Tensor& indices,
    int64_t weights, int64_t padding, bool scale_by_frequency) {
  constexpr int64_t extent = 1024;
  constexpr int64_t tokens = 128;
  return !scale_by_frequency
      && weights > 0 && weights <= std::numeric_limits<int32_t>::max() / extent
      && padding >= -1 && padding < weights
      && gradient.defined() && indices.defined()
      && gradient.device().is_privateuseone() && gradient.device() == indices.device()
      && gradient.scalar_type() == at::kBFloat16 && indices.scalar_type() == at::kLong
      && gradient.layout() == at::kStrided && indices.layout() == at::kStrided
      && gradient.is_contiguous() && indices.is_contiguous()
      && !gradient.is_neg() && !indices.is_neg()
      && gradient.dim() >= 2 && gradient.size(-1) == extent
      && gradient.numel() == tokens * extent
      && gradient.sizes().slice(0, gradient.dim() - 1).equals(indices.sizes())
      && reinterpret_cast<uintptr_t>(gradient.const_data_ptr()) % 16 == 0
      && reinterpret_cast<uintptr_t>(indices.const_data_ptr()) % 16 == 0;
}
"""


TSINGMICRO_MASKED_DIVIDE_GUARD = r"""
inline bool TsingMicroMaskedDivideAccepts(const at::Tensor& input, const at::Scalar& divisor) {
  constexpr auto max_elements = std::numeric_limits<int32_t>::max();
  const bool supported = input.defined() && input.device().is_privateuseone()
      && input.scalar_type() == at::kFloat && input.layout() == at::kStrided
      && (input.dim() == 3 || input.dim() == 4)
      && input.size(-1) > 1 && input.stride(-1) == 0
      && input.numel() > 0 && input.numel() <= max_elements
      && input.storage().nbytes() / sizeof(float) <= max_elements
      && !input.is_neg() && !input.has_names()
      && reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 16 == 0
      && divisor.isIntegral(false);
  if (!supported) return false;
  const auto reduction = divisor.toLong();
  if (input.dim() == 3) return input.stride(1) == 1 && reduction == 1024;
  return input.size(-1) == 128 && reduction == 128 && input.stride(-2) == 1
      && input.stride(-3) == input.size(-2);
}
"""


TSINGMICRO_ARGMAX_GUARD = r"""
inline bool TsingMicroArgmaxAccepts(const at::Tensor& input,
                                   std::optional<int64_t> dim, bool keepdim) {
  return dim && (*dim == -1 || *dim == 1) && !keepdim
      && input.defined() && input.device().is_privateuseone()
      && input.scalar_type() == at::kFloat && input.layout() == at::kStrided
      && input.dim() == 2 && input.size(0) == 1 && input.size(1) == 151936
      && input.is_contiguous() && !input.is_neg() && !input.has_names()
      && reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 16 == 0;
}
"""


TSINGMICRO_NLL_MEAN_GUARDS = r"""
inline bool TsingMicroNllMeanForwardAccepts(
    const at::Tensor& input, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction, int64_t ignore_index) {
  return !weight && reduction == 1 && ignore_index == -100
      && input.defined() && target.defined()
      && input.device().is_privateuseone() && target.device() == input.device()
      && input.scalar_type() == at::kFloat && target.scalar_type() == at::kLong
      && input.layout() == at::kStrided && target.layout() == at::kStrided
      && input.dim() == 2 && input.size(0) == 128 && input.size(1) == 151936
      && target.dim() == 1 && target.size(0) == input.size(0)
      && input.is_contiguous() && target.is_contiguous()
      && !input.is_neg() && !target.is_neg() && !input.has_names() && !target.has_names()
      && reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 16 == 0
      && reinterpret_cast<uintptr_t>(target.const_data_ptr()) % 16 == 8;
}

inline bool TsingMicroNllMeanScalarAccepts(const at::Tensor& value, c10::Device device) {
  return value.defined() && value.device() == device && value.scalar_type() == at::kFloat
      && value.layout() == at::kStrided && value.dim() == 0 && !value.is_neg();
}

inline bool TsingMicroNllMeanBackwardAccepts(
    const at::Tensor& gradient, const at::Tensor& input, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction, int64_t ignore_index,
    const at::Tensor& total_weight) {
  return TsingMicroNllMeanForwardAccepts(input, target, weight, reduction, ignore_index)
      && TsingMicroNllMeanScalarAccepts(gradient, input.device())
      && TsingMicroNllMeanScalarAccepts(total_weight, input.device())
      && reinterpret_cast<uintptr_t>(gradient.const_data_ptr()) % 16 == 0
      && reinterpret_cast<uintptr_t>(total_weight.const_data_ptr()) % 16 == 4;
}
"""


TSINGMICRO_LONG_ARITHMETIC_GUARD = r"""
inline bool TsingMicroLongArithmeticAccepts(
    const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  if (!self.defined() || !other.defined() || !self.device().is_privateuseone()
      || self.scalar_type() != at::kLong || other.scalar_type() != at::kLong
      || self.layout() != at::kStrided || other.layout() != at::kStrided
      || self.dim() < 1 || self.dim() > 2 || self.numel() > 192
      || !self.is_contiguous() || !other.is_contiguous()
      || self.is_neg() || other.is_neg() || self.has_names() || other.has_names()
      || !alpha.isIntegral(false) || alpha.toLong() != 1) {
    return false;
  }
  return TypedWrappedNumber(other)
      || (other.device() == self.device() && other.sizes().equals(self.sizes()));
}
"""


TSINGMICRO_DENSE_UNARY_GUARD = r"""
inline bool TsingMicroDenseUnaryAccepts(
    const at::Tensor& tensor, at::ScalarType dtype, int64_t width) {
  return tensor.defined() && tensor.device().is_privateuseone()
      && tensor.scalar_type() == dtype && tensor.layout() == at::kStrided
      && tensor.dim() == 3 && tensor.size(0) == 1
      && (tensor.size(1) == 1 || tensor.size(1) == 128)
      && tensor.size(2) == width && tensor.is_contiguous()
      && !tensor.is_neg() && !tensor.is_conj() && !tensor.has_names();
}
"""


TSINGMICRO_ORDERED_SUM_GUARD = r"""
inline bool TsingMicroOrderedSumAccepts(
    const at::Tensor& input, at::OptionalIntArrayRef dims, bool keepdim,
    std::optional<at::ScalarType> dtype) {
  if (!input.defined() || !input.is_privateuseone()
      || input.layout() != at::kStrided || !input.is_non_overlapping_and_dense()
      || input.storage_offset() != 0 || input.is_neg() || input.is_conj() || input.has_names()
      || !dims || !keepdim || (dtype && *dtype != input.scalar_type())) {
    return false;
  }
  const auto rank = input.dim();
  if (input.scalar_type() == at::kBFloat16) {
    if ((rank != 3 && rank != 4) || dims->size() != static_cast<size_t>(rank - 1)
        || (*dims)[0] != 0 || (*dims)[1] != 1
        || input.size(0) != 1 || input.size(1) != 128 || input.stride(-1) != 1) {
      return false;
    }
    if (rank == 3) {
      return input.size(2) == 1024 && input.is_contiguous();
    }
    return (*dims)[2] == 2 && (input.size(2) == 8 || input.size(2) == 16)
        && input.size(3) == 128;
  }
  if (input.scalar_type() != at::kFloat || dims->size() != 1) {
    return false;
  }
  const auto dim = (*dims)[0];
  if ((rank != 3 && rank != 4) || (dim != -1 && dim != rank - 1)
      || input.size(0) != 1 || input.size(1) != 128 || input.stride(-1) != 1) {
    return false;
  }
  if (rank == 3) {
    return input.size(2) == 1024;
  }
  return (input.size(2) == 8 || input.size(2) == 16) && input.size(3) == 128;
}
"""


TSINGMICRO_BF16_ADDCDIV_GUARD = r"""
inline bool TsingMicroBFloat16AddcdivAccepts(
    const at::Tensor& self, const at::Tensor& first, const at::Tensor& second,
    const at::Scalar& value, const at::Tensor& out) {
  if (!self.defined() || !out.is_same(self) || !self.is_privateuseone() ||
      self.scalar_type() != at::kBFloat16 || self.layout() != at::kStrided ||
      !self.is_contiguous() || self.numel() <= 0 || self.storage_offset() != 0 ||
      self.is_neg() || self.is_conj() || self.has_names() || !value.isFloatingPoint()) {
    return false;
  }
  const auto eligible = [&](const at::Tensor& other) {
    return other.defined() && other.layout() == at::kStrided &&
        other.scalar_type() == self.scalar_type() && other.device() == self.device() &&
        other.sizes().equals(self.sizes()) && other.is_contiguous() &&
        other.storage_offset() == 0 && !other.is_neg() && !other.is_conj() &&
        !other.has_names() && (!other.is_alias_of(self) || other.is_same(self));
  };
  return eligible(first) && eligible(second);
}
"""


KUNLUNXIN_BF16_LERP_GUARD = r"""
inline bool KunlunxinBFloat16LerpAccepts(
    const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight) {
  if (!weight.isFloatingPoint() || !self.defined() || !end.defined() ||
      !self.is_privateuseone() || self.scalar_type() != at::kBFloat16 ||
      end.scalar_type() != self.scalar_type() || self.device() != end.device() ||
      !self.sizes().equals(end.sizes()) || !self.is_contiguous() || !end.is_contiguous() ||
      self.numel() == 0 || self.is_alias_of(end) || self.is_neg() || end.is_neg() ||
      self.is_conj() || end.is_conj() || self.has_names() || end.has_names()) {
    return false;
  }
  const auto scalar = weight.toDouble();
  return scalar > 0.0 && scalar < 0.5 &&
      static_cast<float>(scalar) > 0.0f && static_cast<float>(scalar) < 0.5f;
}
"""


_ADAPTED_ENTRY_GUARDS = {
    "kunlunxin_bf16_lerp_round_even": (
        KUNLUNXIN_BF16_LERP_GUARD,
        "KunlunxinBFloat16LerpAccepts(a0, a1, a2)",
    ),
    "tsingmicro_addcdiv_bf16_scale_first": (
        TSINGMICRO_BF16_ADDCDIV_GUARD,
        "TsingMicroBFloat16AddcdivAccepts(a0, a1, a2, a3, a4)",
    ),
    "tsingmicro_sum_ordered": (
        TSINGMICRO_ORDERED_SUM_GUARD,
        "TsingMicroOrderedSumAccepts(a0, a1, a2, a3)",
    ),
    "tsingmicro_embedding_bf16_input_order": (
        TSINGMICRO_EMBEDDING_GUARD,
        "TsingMicroEmbeddingBFloat16Accepts(a0, a1, a2, a3, a4)",
    ),
    "tsingmicro_float_scalar_divide_masked": (
        TSINGMICRO_MASKED_DIVIDE_GUARD,
        "TsingMicroMaskedDivideAccepts(a0, a1)",
    ),
    "tsingmicro_argmax_first_nan": (
        TSINGMICRO_ARGMAX_GUARD,
        "TsingMicroArgmaxAccepts(a0, a1, a2)",
    ),
    "tsingmicro_sin_reduced_argument": (
        TSINGMICRO_DENSE_UNARY_GUARD,
        "TsingMicroDenseUnaryAccepts(a0, at::kFloat, 128)",
    ),
    "tsingmicro_silu_stable_positive": (
        TSINGMICRO_DENSE_UNARY_GUARD,
        "TsingMicroDenseUnaryAccepts(a0, at::kBFloat16, 3072)",
    ),
    "tsingmicro_silu_backward_bf16_stages": (
        TSINGMICRO_DENSE_UNARY_GUARD,
        "!c10::InferenceMode::is_enabled() && "
        "TsingMicroDenseUnaryAccepts(a0, at::kBFloat16, 3072) && "
        "TsingMicroDenseUnaryAccepts(a1, at::kBFloat16, 3072) && "
        "a0.size(1) == 128 && a0.sizes() == a1.sizes() && "
        "a0.device() == a1.device() && "
        "a0.storage_offset() == 0 && a1.storage_offset() == 0",
    ),
    "tsingmicro_nll_mean_forward_integer_offsets": (
        TSINGMICRO_NLL_MEAN_GUARDS,
        "TsingMicroNllMeanForwardAccepts(a0, a1, a2, a3, a4)",
    ),
    "tsingmicro_nll_mean_backward_integer_offsets": (
        TSINGMICRO_NLL_MEAN_GUARDS,
        "TsingMicroNllMeanBackwardAccepts(a0, a1, a2, a3, a4, a5, a6)",
    ),
    "tsingmicro_long_add_scalar_lanes": (
        TSINGMICRO_LONG_ARITHMETIC_GUARD,
        "TsingMicroLongArithmeticAccepts(a0, a1, a2)",
    ),
    "tsingmicro_long_sub_scalar_lanes": (
        TSINGMICRO_LONG_ARITHMETIC_GUARD,
        "TsingMicroLongArithmeticAccepts(a0, a1, a2)",
    ),
}


def emit_entries(catalog, groups, selections=(), *, tensor_ranks=None):
    """Generate schema-specific callbacks; unsupported conversions stay visible."""
    tensor_ranks = tensor_ranks or {}
    declarations, records, rejected = [], [], []
    scoped_guards = {
        row["adaptation"]: _ADAPTED_ENTRY_GUARDS[row["adaptation"]]
        for row in selections
        if row.get("adaptation") in _ADAPTED_ENTRY_GUARDS
    }
    declarations.extend(
        dict.fromkeys(declaration for declaration, _ in scoped_guards.values())
    )
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
                adaptation = (
                    selection.get("adaptation") if selection is not None else None
                )
                if adaptation in scoped_guards:
                    checks.append(scoped_guards[adaptation][1])
                elif selection is not None:
                    mask = " | ".join(
                        "(uint64_t{1} << static_cast<unsigned>(at::ScalarType::"
                        + dtype
                        + "))"
                        for dtype in selection["dtypes"]
                    )
                    restricted = (
                        selection.get("same_tensor_metadata", False)
                        or (selection.get("min_tensors", 1) != 1)
                        or selection.get("unit_stride_axis", False)
                    )
                    guard = (
                        "PublishedSourceAcceptsRestricted"
                        if restricted
                        else "PublishedSourceAccepts"
                    )
                    if selection.get("allow_wrapped_numbers", False):
                        guard += "<true>"
                    options = (
                        ", "
                        + str(selection.get("same_tensor_metadata", False)).lower()
                        + ", "
                        + str(selection.get("min_tensors", 1))
                        + ", "
                        + str(selection.get("unit_stride_axis", False)).lower()
                        if restricted
                        else ""
                    )
                    checks.append(
                        guard
                        + "("
                        + mask
                        + options
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
                for parameter, rank in tensor_ranks.get(name, {}).items():
                    tensor = values[parameter]
                    checks.append(f"{tensor}.dim()=={rank}")
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
#include <ATen/ops/_local_scalar_dense_native.h>
#include <c10/core/InferenceMode.h>
class PublishedSourceGuard {
 uint64_t mask_;
 std::optional<c10::Device> device_;
 bool same_metadata_, unit_stride_, wrapped_numbers_;
 int64_t minimum_, count_ = 0;
 std::optional<at::ScalarType> dtype_;
 std::optional<int64_t> rank_;
 public:
 explicit PublishedSourceGuard(uint64_t mask,bool same_metadata=false,int64_t minimum=1,
                               bool unit_stride=false,bool wrapped_numbers=false)
   :mask_(mask),same_metadata_(same_metadata),unit_stride_(unit_stride),
    wrapped_numbers_(wrapped_numbers),minimum_(minimum){}
 bool Check(const at::Tensor& value) {
  if(wrapped_numbers_ && value.defined() && value.device().is_cpu() &&
     value.dim()==0 && value.unsafeGetTensorImpl()->is_wrapped_number() &&
     !value.is_complex())return true;
  if(!value.defined() || !value.device().is_privateuseone() ||
     !(mask_ & (uint64_t{1} << static_cast<unsigned>(value.scalar_type()))))return false;
  if(device_ && *device_!=value.device())return false;
  if(same_metadata_ && ((dtype_ && *dtype_!=value.scalar_type()) ||
                       (rank_ && *rank_!=value.dim())))return false;
  if(unit_stride_ && value.numel()>1) {
   bool found=false;
   for(int64_t dim=0;dim<value.dim();++dim)
    if(value.size(dim)>1 && value.stride(dim)==1)found=true;
   if(!found)return false;
  }
  dtype_=value.scalar_type();rank_=value.dim();++count_;
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
 bool HasTensor() const{return device_.has_value() && count_>=minimum_;}
};
template<bool WrappedNumbers=false,class... Args> bool PublishedSourceAccepts(uint64_t mask,const Args&... args){
 PublishedSourceGuard guard(mask,false,1,false,WrappedNumbers);
 return (guard.Check(args) && ...) && guard.HasTensor();
}
template<bool WrappedNumbers=false,class... Args> bool PublishedSourceAcceptsRestricted(
 uint64_t mask,bool same_metadata,int64_t minimum,bool unit_stride,const Args&... args){
 PublishedSourceGuard guard(mask,same_metadata,minimum,unit_stride,WrappedNumbers);
 return (guard.Check(args) && ...) && guard.HasTensor();
}
inline bool TypedWrappedNumber(const at::Tensor& value) {
 return value.defined() && value.device().is_cpu() && value.unsafeGetTensorImpl()->is_wrapped_number();
}
inline at::Scalar TypedWrappedScalar(const at::Tensor& value) {
 if(TypedWrappedNumber(value) && value.layout()==at::kStrided && value.numel()==1 &&
    !value.is_quantized() && !value.is_conj() && !value.is_neg())
  return at::native::_local_scalar_dense_cpu(value);
 return value.item();
}
template <class T> c10::List<at::Tensor> TypedTensorList(const T& values) {
 c10::List<at::Tensor> result;
 result.reserve(values.size());
 for(const auto& value:values) result.push_back(value);
 return result;
}
"""
