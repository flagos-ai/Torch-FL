# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Use the installed Torch schemas and generated headers for typed host calls."""

from pathlib import Path
import re
import torch
from torchgen.api.types import DispatcherSignature
from torchgen.model import FunctionSchema

_HEADER_ROOT = Path(torch.__file__).parent / "include/ATen/ops"
_headers = {}


def native_call(node, names, int_array_views=()):
    text = node.schema()
    if not text.startswith("aten::"):
        raise NotImplementedError("not an ATen schema")
    schema = FunctionSchema.parse(text.removeprefix("aten::"))
    base = schema.name.name.base
    identifier = schema.name.unambiguous_name()
    if base not in _headers:
        path = _HEADER_ROOT / (base + "_ops.h")
        _headers[base] = path.read_text() if path.exists() else ""
    if not re.search(
        r"\bstruct\s+TORCH_API\s+" + re.escape(identifier) + r"\s*\{", _headers[base]
    ):
        raise NotImplementedError("no typed ATen declaration")
    signature = DispatcherSignature.from_schema(schema)
    inputs = list(node.inputs())
    bindings = signature.arguments()
    if len(inputs) != len(bindings):
        raise NotImplementedError("ATen signature arity")

    def convert(target, value, kind):
        target = target.removeprefix("const ").removesuffix(" &").strip()
        optional = re.fullmatch(r"(?:::)?std::optional<(.+)>", target)
        if optional:
            inner = optional.group(1)
            if str(kind) == "NoneType":
                return "std::optional<" + inner + ">{}"
            if kind.kind() == "OptionalType":
                return (
                    "("
                    + value
                    + " ? std::optional<"
                    + inner
                    + ">("
                    + convert(inner, "(*" + value + ")", kind.getElementType())
                    + ") : std::nullopt)"
                )
            return "std::optional<" + inner + ">(" + convert(inner, value, kind) + ")"
        if target in (
            "at::OptionalIntArrayRef",
            "at::OptionalSymIntArrayRef",
            "c10::OptionalSymIntArrayRef",
        ):
            if str(kind) == "NoneType":
                return target + "{}"
            inner = (
                "at::IntArrayRef"
                if target == "at::OptionalIntArrayRef"
                else "c10::SymIntArrayRef"
            )
            if kind.kind() == "OptionalType":
                return (
                    "("
                    + value
                    + " ? "
                    + target
                    + "("
                    + convert(inner, "(*" + value + ")", kind.getElementType())
                    + ") : "
                    + target
                    + "{})"
                )
            return target + "(" + convert(inner, value, kind) + ")"
        if target in ("at::ScalarType", "at::Layout", "at::MemoryFormat"):
            return "static_cast<" + target + ">(" + value + ")"
        if target in ("at::IntArrayRef", "at::TensorList", "at::ITensorListRef"):
            if kind.kind() != "ListType":
                raise NotImplementedError("array reference input")
            data = value if value in int_array_views else value + ".vec()"
            return target + "(" + data + ")"
        if target == "c10::SymIntArrayRef":
            if kind.kind() != "ListType":
                raise NotImplementedError("symbolic array input")
            data = value if value in int_array_views else value + ".vec()"
            return "c10::fromIntArrayRefSlow(" + data + ")"
        if target in ("c10::SymInt", "at::Scalar"):
            return target + "(" + value + ")"
        if target in (
            "at::Tensor",
            "at::Device",
            "c10::Device",
            "int64_t",
            "double",
            "bool",
            "c10::string_view",
            "std::string",
        ):
            return value
        raise NotImplementedError("ATen C++ type " + target)

    arguments = [
        convert(binding.type, name, value.type())
        for binding, name, value in zip(bindings, names, inputs)
    ]
    result = "at::_ops::" + identifier + "::call(" + ", ".join(arguments) + ")"
    if len(schema.returns) == 1 and str(schema.returns[0].type) in (
        "ScalarType",
        "Layout",
        "MemoryFormat",
    ):
        result = "int64_t(" + result + ")"
    return result
