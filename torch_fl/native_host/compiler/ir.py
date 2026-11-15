# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Lower typed host control flow to C++; numerical calls remain opaque FG slots."""

import json
import math
from pathlib import Path


def emit(
    graph, function_name="Forward", binding=True, grid_slots=None, slot_tables=False
):
    types = {
        "Tensor": "at::Tensor",
        "int": "int64_t",
        "bool": "bool",
        "float": "double",
        "str": "std::string",
        "NoneType": "std::nullptr_t",
        "int[]": "std::vector<int64_t>",
        "List[int]": "std::vector<int64_t>",
        "Any[]": "c10::List<c10::IValue>",
        "List[Any]": "c10::List<c10::IValue>",
        "Dict(str, Any)": "c10::impl::GenericDict",
        "Dict[str, Any]": "c10::impl::GenericDict",
        "Any": "c10::IValue",
    }

    def name(v):
        return "v" + str(v.unique())

    types["Dict[str, int]"] = "c10::Dict<std::string,int64_t>"
    if function_name.startswith("Grid"):
        types["Dict[str, int]"] = "const HostGridMeta&"
    types.update(
        {
            "Device": "c10::Device",
            "Tensor[]": "std::vector<at::Tensor>",
            "List[Tensor]": "std::vector<at::Tensor>",
            "List[List[int]]": "std::vector<std::vector<int64_t>>",
            "int[][]": "std::vector<std::vector<int64_t>>",
        }
    )
    if not function_name.startswith("Grid"):
        types = {k: v.replace("std::vector<", "c10::List<") for k, v in types.items()}
    types["NoneType"] = "std::nullopt_t"

    def cpp_type(t):
        label = str(t)
        if label in types:
            return types[label]
        if t.kind() == "OptionalType":
            return "std::optional<" + cpp_type(t.getElementType()) + ">"
        if t.kind() == "ListType":
            return (
                ("std::vector<" if function_name.startswith("Grid") else "c10::List<")
                + cpp_type(t.getElementType())
                + ">"
            )
        if t.kind() == "TupleType":
            return "std::tuple<" + ", ".join(cpp_type(e) for e in t.elements()) + ">"
        raise NotImplementedError("type " + label)

    aliases = graph.alias_db()

    def readonly_ints(v):
        return (
            not function_name.startswith("Grid")
            and str(v.type()) in ("List[int]", "int[]")
            and not aliases.has_writers(v)
        )

    def kind(v):
        return "HostIntList" if readonly_ints(v) else cpp_type(v.type())

    def default_value(t):
        if str(t) == "NoneType":
            return "std::nullopt"
        if str(t) == "Device":
            return "c10::Device(c10::DeviceType::CPU)"
        if t.kind() == "TupleType":
            return (
                "std::make_tuple("
                + ", ".join(default_value(e) for e in t.elements())
                + ")"
            )
        if str(t) in ("Any[]", "List[Any]"):
            return "c10::List<c10::IValue>(c10::AnyType::get())"
        return cpp_type(t) + "{}"

    def empty(v):
        return (
            " = HostIntList{}" if readonly_ints(v) else " = " + default_value(v.type())
        )

    def convert(v, expr):
        label = str(v.type())
        methods = {
            "Tensor": "toTensor",
            "int": "toInt",
            "bool": "toBool",
            "float": "toDouble",
            "str": "toStringRef",
            "Device": "toDevice",
            "int[]": "toIntList",
            "List[int]": "toIntList",
            "Tensor[]": "toTensorList",
            "List[Tensor]": "toTensorList",
        }
        if function_name.startswith("Grid"):
            methods.update({"List[int]": "toIntVector", "int[]": "toIntVector"})
        if label == "Any":
            return expr
        if label == "NoneType":
            return "(" + expr + ", std::nullopt)"
        return (
            "(" + expr + ")." + methods[label] + "()"
            if label in methods
            else "(" + expr + ").to<" + kind(v) + ">()"
        )

    packed_ops = {
        "flagos_script_host::launch",
        "flagos_source_native::launch",
        "flagos_source_native::pointwise_apply",
        "flagos_source_native::foreign_value",
        "flagos_source_native::foreign_bool",
        "flagos_source_native::foreign_tensor",
    }
    packed_inputs = set()
    packed_pointwise_launches = set()
    pointwise_projections = {}

    def project(scope):
        nodes = list(scope.nodes())
        for index, node in enumerate(nodes):
            for block in node.blocks():
                project(block)
            if node.kind() != "flagos_source_native::pointwise_apply":
                continue
            value = list(node.outputs())[0]
            uses = list(value.uses())
            if len(uses) != 1 or uses[0].user.kind() != "aten::__getitem__":
                continue
            for later in nodes[index + 1 :]:
                if later is uses[0].user:
                    pointwise_projections[value.unique()] = node
                    break
                if later.kind() != "prim::Constant":
                    break

    project(graph)

    def scan(scope):
        for node in scope.nodes():
            for block in node.blocks():
                scan(block)
            if node.kind() == "flagos_source_native::pointwise_launch":
                values = list(node.inputs())
                if all(
                    values[index].node().kind() == "prim::ListConstruct"
                    and len(list(values[index].uses())) == 1
                    for index in (2, 3)
                ):
                    packed_inputs.update(values[index].unique() for index in (2, 3))
                    packed_pointwise_launches.add(node)
                continue
            if node.kind() not in packed_ops:
                continue
            candidates = []
            for index, kind_ in (
                (1, "prim::ListConstruct"),
                (2, "prim::DictConstruct"),
            ):
                value = list(node.inputs())[index]
                if value.node().kind() != kind_:
                    break
                if not all(
                    use.user.kind() in packed_ops and use.offset == index
                    for use in value.uses()
                ):
                    break
                candidates.append(value.unique())
            if len(candidates) == 2:
                packed_inputs.update(candidates)

    scan(graph)

    def packed_call(node, slot, grid=None, function=None, index=None):
        values = list(node.inputs())
        args, kwargs = values[1:3]
        if args.unique() not in packed_inputs or kwargs.unique() not in packed_inputs:
            return None
        parameters = (
            "{" + ", ".join("Box(" + name(v) + ")" for v in args.node().inputs()) + "}"
        )
        pairs = list(kwargs.node().inputs())
        keywords = (
            "{"
            + ", ".join(
                "{" + name(pairs[i]) + ", Box(" + name(pairs[i + 1]) + ")}"
                for i in range(0, len(pairs), 2)
            )
            + "}"
        )
        if node.kind() == "flagos_source_native::pointwise_apply":
            return (
                (
                    "ApplyPointwiseAtPacked("
                    + slot
                    + ", "
                    + parameters
                    + ", "
                    + keywords
                    + ", "
                    + index
                    + ")"
                )
                if index is not None
                else "ApplyPointwisePacked("
                + slot
                + ", "
                + parameters
                + ", "
                + keywords
                + ")"
            )
        if node.kind().startswith("flagos_source_native::foreign_"):
            return "ForeignPacked(" + slot + ", " + parameters + ", " + keywords + ")"
        return (
            "LaunchPacked("
            + slot
            + ", "
            + parameters
            + ", "
            + keywords
            + ", "
            + grid
            + ", "
            + function
            + ")"
        )

    lines = []

    def write(s, depth):
        lines.append("  " * depth + s)

    def literal(value):
        if isinstance(value, tuple):
            return "std::make_tuple(" + ", ".join(literal(v) for v in value) + ")"
        if value is None:
            return "std::nullopt"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return "std::string(" + json.dumps(value) + ")"
        if type(value).__name__ == "device":
            return "c10::Device(" + json.dumps(str(value)) + ")"
        if isinstance(value, float) and not math.isfinite(value):
            if math.isnan(value):
                return "std::numeric_limits<double>::quiet_NaN()"
            prefix = "-" if value < 0 else ""
            return prefix + "std::numeric_limits<double>::infinity()"
        return str(value)

    def lower(scope, depth, targets=()):
        for node in scope.nodes():
            op = node.kind()
            ins = list(node.inputs())
            outs = list(node.outputs())
            a = [name(v) for v in ins]
            if len(outs) == 1 and (
                outs[0].unique() in packed_inputs
                or outs[0].unique() in pointwise_projections
            ):
                continue
            if op == "prim::Constant":
                if outs[0].type().kind() == "FunctionType":
                    continue
                qualifier = "static const " if str(outs[0].type()) == "str" else ""
                write(
                    f"{qualifier}{kind(outs[0])} {name(outs[0])} = {literal(outs[0].toIValue())};",
                    depth,
                )
                continue
            if op == "prim::Uninitialized":
                write(f"{kind(outs[0])} {name(outs[0])}{empty(outs[0])};", depth)
                continue
            if op == "prim::If":
                for out in outs:
                    write(f"{kind(out)} {name(out)}{empty(out)};", depth)
                write(f"if ({a[0]}) {{", depth)
                block_values = list(node.blocks())
                lower(block_values[0], depth + 1, [name(v) for v in outs])
                write("} else {", depth)
                lower(block_values[1], depth + 1, [name(v) for v in outs])
                write("}", depth)
                continue
            if op == "prim::Loop":
                body = list(node.blocks())[0]
                args = list(body.inputs())
                for out, initial in zip(outs, a[2:]):
                    write(f"{kind(out)} {name(out)} = {initial};", depth)
                cond = "condition_" + str(args[0].unique())
                write(f"bool {cond} = {a[1]};", depth)
                write(
                    f"for(int64_t {name(args[0])}=0; {name(args[0])} < {a[0]} && {cond}; ++{name(args[0])}) {{",
                    depth,
                )
                for arg, out in zip(args[1:], outs):
                    write(f"{kind(arg)} {name(arg)} = {name(out)};", depth + 1)
                lower(body, depth + 1, [cond] + [name(v) for v in outs])
                write("}", depth)
                continue
            if op == "prim::RaiseException":
                write(f"throw std::runtime_error({a[0]});", depth)
                continue
            if op == "prim::TupleUnpack":
                for i, out in enumerate(outs):
                    write(f"{kind(out)} {name(out)} = std::get<{i}>({a[0]});", depth)
                continue
            if op == "prim::ListUnpack":
                write(
                    f'TORCH_CHECK({a[0]}.size() == {len(outs)}, "list unpack arity mismatch");',
                    depth,
                )
                for i, out in enumerate(outs):
                    write(
                        f"{kind(out)} {name(out)} = Get({a[0]}, int64_t({i}));", depth
                    )
                continue
            if op == "prim::TupleConstruct":
                expr = "std::make_tuple(" + ", ".join(a) + ")"
            elif op == "prim::TupleIndex":
                index = ins[1].toIValue()
                count = len(ins[0].type().elements())
                if index is not None:
                    index = index + count if index < 0 else index
                    if not 0 <= index < count:
                        raise NotImplementedError("tuple index out of range")
                    expr = f"std::get<{index}>({a[0]})"
                else:
                    cases = " ".join(
                        f"case {i}: return std::get<{i}>({a[0]});" for i in range(count)
                    )
                    expr = (
                        f"([&]() -> {kind(outs[0])} {{ int64_t index={a[1]}; "
                        f"if(index<0)index+={count}; switch(index) {{ {cases} "
                        'default: throw std::out_of_range("tuple index out of range");'
                        " } })()"
                    )
            elif op == "prim::ListConstruct":
                if str(outs[0].type()) in ("Any[]", "List[Any]"):
                    expr = "AnyList({" + ", ".join("Box(" + v + ")" for v in a) + "})"
                else:
                    expr = kind(outs[0]) + "{" + ", ".join(a) + "}"
            elif op == "prim::unchecked_cast":
                expr = (
                    convert(outs[0], a[0])
                    if str(ins[0].type()) == "Any"
                    else (
                        f"*{a[0]}" if ins[0].type().kind() == "OptionalType" else a[0]
                    )
                )
            elif op == "prim::isinstance":
                checks = {
                    "Tensor": "isTensor",
                    "int": "isInt",
                    "float": "isDouble",
                    "bool": "isBool",
                    "str": "isString",
                }

                def check(t, expr=a[0]):
                    if str(t) == "NoneType":
                        return expr + ".isNone()"
                    if t.kind() == "OptionalType":
                        return (
                            "("
                            + expr
                            + ".isNone() || "
                            + check(t.getElementType(), expr)
                            + ")"
                        )
                    if t.kind() == "TupleType":
                        members = t.elements()
                        return (
                            "("
                            + expr
                            + ".isTuple() && "
                            + expr
                            + ".toTupleRef().elements().size()=="
                            + str(len(members))
                            + " && "
                            + " && ".join(
                                check(
                                    v, expr + ".toTupleRef().elements()[" + str(i) + "]"
                                )
                                for i, v in enumerate(members)
                            )
                            + ")"
                        )
                    if str(t) == "Device":
                        return expr + ".isDevice()"
                    method = (
                        checks[str(t)]
                        if t.kind() != "ListType"
                        else {"int": "isIntList", "Tensor": "isTensorList"}[
                            str(t.getElementType())
                        ]
                    )
                    return expr + "." + method + "()"

                expr = "(" + " || ".join(check(t) for t in node.tys("types")) + ")"
            elif op == "prim::DictConstruct":
                expr = kind(outs[0]) + "(c10::StringType::get(), c10::AnyType::get())"
                write(f"{kind(outs[0])} {name(outs[0])} = {expr};", depth)
                for i in range(0, len(a), 2):
                    write(
                        f"{name(outs[0])}.insert(Box({a[i]}), Box({a[i + 1]}));", depth
                    )
                continue
            elif op == "prim::dtype":
                expr = f"int64_t({a[0]}.scalar_type())"
            elif op == "prim::layout":
                expr = f"int64_t({a[0]}.layout())"
            elif op == "prim::device":
                expr = f"{a[0]}.device()"
            elif op == "prim::type":
                expr = f"DeviceName({a[0]})"
            elif op in ("prim::is_cuda", "prim::is_cpu"):
                expr = f"{a[0]}.{op.split('::')[1]}()"
            elif op == "prim::is_quantized":
                expr = f"{a[0]}.is_quantized()"
            elif op in ("prim::min", "prim::max"):
                operation = op.split("::")[1]
                expr = (
                    "std::" + operation + "(" + ", ".join(a) + ")"
                    if len(a) == 2
                    else "*std::"
                    + operation
                    + "_element("
                    + a[0]
                    + ".begin(), "
                    + a[0]
                    + ".end())"
                )
            elif op in ["aten::dim", "aten::numel"]:
                expr = f"{a[0]}.{op.split('::')[1]}()"
            elif op == "aten::size":
                expr = (
                    (
                        f"HostIntList({a[0]}.sizes())"
                        if readonly_ints(outs[0])
                        else f"{a[0]}.sizes().vec()"
                    )
                    if len(a) == 1
                    else f"{a[0]}.size({a[1]})"
                )
            elif op == "aten::stride":
                expr = (
                    (
                        f"HostIntList({a[0]}.strides())"
                        if readonly_ints(outs[0])
                        else f"{a[0]}.strides().vec()"
                    )
                    if len(a) == 1
                    else f"{a[0]}.stride({a[1]})"
                )
            elif op in [
                "aten::storage_offset",
                "aten::is_complex",
                "aten::is_conj",
                "aten::is_neg",
                "aten::_is_zerotensor",
            ]:
                expr = f"{a[0]}.{op.split('::')[1]}()"
            elif op == "aten::__not__":
                expr = f"(!{a[0]})"
            elif op in ("aten::__and__", "aten::__or__", "aten::__xor__"):
                sign = {
                    "aten::__and__": "&",
                    "aten::__or__": "|",
                    "aten::__xor__": "^",
                }[op]
                expr = f"({a[0]} {sign} {a[1]})"
            elif op in ("aten::__is__", "aten::__isnot__"):
                if (
                    str(ins[1].type()) == "NoneType"
                    and ins[0].type().kind() == "OptionalType"
                ):
                    expr = (
                        f"{a[0]}.has_value()"
                        if op.endswith("isnot__")
                        else f"!{a[0]}.has_value()"
                    )
                elif ins[0].type().kind() == ins[1].type().kind() == "TensorType":
                    expr = f"{a[0]}.is_same({a[1]})"
                    if op.endswith("isnot__"):
                        expr = "!" + expr
                elif ins[0].type().kind() in ("ListType", "TupleType", "DictType"):
                    raise NotImplementedError(
                        "container identity requires object semantics"
                    )
                elif ins[0].type().kind() == "AnyType":
                    expr = f"Box({a[0]}).isSameIdentity(Box({a[1]}))"
                    if op.endswith("isnot__"):
                        expr = "!" + expr
                else:
                    expr = (
                        f"({a[0]} "
                        + ("!=" if op.endswith("isnot__") else "==")
                        + f" {a[1]})"
                    )
            elif op == "aten::list":
                expr = a[0] + ".copy()"
            elif op == "aten::_set_item" and ins[0].type().kind() == "ListType":
                expr = "SetItem(" + ", ".join(a) + ")"
            elif op == "aten::pop" and ins[0].type().kind() == "ListType":
                expr = "Pop(" + ", ".join(a) + ")"
            elif op == "aten::extend" and ins[0].type().kind() == "ListType":
                expr = a[0] + ".append(" + a[1] + ")"
            elif op == "aten::slice" and ins[0].type().kind() == "ListType":
                expr = "Slice(" + ", ".join(a) + ")"
            elif op == "aten::__contains__" and ins[0].type().kind() == "ListType":
                expr = "Contains(" + ", ".join(a) + ")"
            elif op == "aten::sum" and ins[0].type().kind() == "ListType":
                expr = "Sum(" + a[0] + ")"
            elif op == "aten::__range_length":
                expr = "RangeLength(" + ", ".join(a) + ")"
            elif op == "aten::__derive_index":
                expr = "(" + a[1] + " + " + a[0] + " * " + a[2] + ")"
            elif op in ("aten::ceil", "aten::sqrt"):
                expr = "std::" + op.split("::")[1] + "(" + a[0] + ")"
            elif op == "aten::element_size":
                expr = a[0] + ".element_size()"
            elif op == "aten::len":
                expr = f"int64_t({a[0]}.size())"
            elif op == "aten::Bool":
                expr = (
                    f"at::is_nonzero({a[0]})"
                    if ins[0].type().kind() == "TensorType"
                    else f"Truth({a[0]})"
                )
            elif op == "aten::append":
                write(f"{a[0]}.push_back({a[1]});", depth)
                if outs:
                    write(f"auto& {name(outs[0])} = {a[0]};", depth)
                continue
            elif op == "aten::split" and str(ins[0].type()) == "str":
                expr = "Split(" + ", ".join(a) + ")"
            elif op == "aten::is_contiguous":
                format_ = f"static_cast<at::MemoryFormat>({a[1]})" if len(a) > 1 else ""
                expr = f"{a[0]}.is_contiguous({format_})"
            elif op == "aten::tensor":
                expr = "TensorValue(" + ", ".join(a) + ")"
            elif op == "aten::__getitem__":
                producer = pointwise_projections.get(ins[0].unique())
                if producer is not None:
                    inputs = list(producer.inputs())
                    parts = [name(v) for v in inputs]
                    slot = (
                        f"pointwise_slots.at({parts[0]})" if slot_tables else parts[0]
                    )
                    expr = (
                        packed_call(producer, slot, index=a[1])
                        or "ApplyPointwiseAt("
                        + slot
                        + ", "
                        + parts[1]
                        + ", "
                        + parts[2]
                        + ", "
                        + a[1]
                        + ")"
                    )
                else:
                    expr = f"Get({a[0]}, {a[1]})"
            elif op == "aten::contiguous":
                expr = f"{a[0]}.contiguous(static_cast<at::MemoryFormat>({a[1]}))"
            elif op == "aten::resize_":
                expr = f"{a[0]}.resize_({a[1]}.vec())"
            elif op == "aten::format":
                expr = (
                    "Format("
                    + a[0]
                    + ", {"
                    + ", ".join("Box(" + v + ")" for v in a[1:])
                    + "})"
                )
            elif op in ("aten::__lshift__", "aten::__rshift__"):
                expr = (
                    "("
                    + a[0]
                    + (" << " if op == "aten::__lshift__" else " >> ")
                    + a[1]
                    + ")"
                )
            elif (
                op in ("aten::abs", "prim::abs")
                and ins[0].type().kind() != "TensorType"
            ):
                expr = "std::abs(" + a[0] + ")"
            elif op == "aten::div" and ins[0].type().kind() != "TensorType":
                expr = "(double(" + a[0] + ") / " + a[1] + ")"
            elif op == "aten::Int" and ins[0].type().kind() != "TensorType":
                expr = "int64_t(" + a[0] + ")"
            elif op == "aten::Float" and ins[0].type().kind() != "TensorType":
                expr = "double(" + a[0] + ")"
            elif op == "aten::neg" and ins[0].type().kind() != "TensorType":
                expr = f"(-{a[0]})"
            elif op == "aten::remainder" and all(
                v.type().kind() != "TensorType" for v in ins
            ):
                expr = f"Remainder({a[0]}, {a[1]})"
            elif op == "aten::floordiv" and all(
                v.type().kind() != "TensorType" for v in ins
            ):
                expr = f"FloorDiv({a[0]}, {a[1]})"
            elif op == "aten::mul" and ins[0].type().kind() == "ListType":
                expr = "Repeat(" + a[0] + ", " + a[1] + ")"
            elif op in {
                "aten::ge",
                "aten::lt",
                "aten::eq",
                "aten::ne",
                "aten::mul",
                "aten::gt",
                "aten::le",
                "aten::add",
                "aten::sub",
            } and all(v.type().kind() != "TensorType" for v in ins):
                sign = {
                    "ge": ">=",
                    "lt": "<",
                    "eq": "==",
                    "ne": "!=",
                    "mul": "*",
                    "gt": ">",
                    "le": "<=",
                    "add": "+",
                    "sub": "-",
                }[op.split("::")[1]]
                expr = f"({a[0]} {sign} {a[1]})"
            elif op == "flagos_script_host::launch" and grid_slots is not None:
                slot = grid_slots[int(ins[0].toIValue())]
                grid_id = int(ins[3].toIValue())
                slot = (
                    "kernel_slots.at(" + str(slot) + ")"
                    if slot_tables
                    else "int64_t(" + str(slot) + ")"
                )
                function = (
                    "int64_t(reinterpret_cast<uintptr_t>(&Grid" + str(grid_id) + "))"
                )
                expr = (
                    packed_call(node, slot, a[4], function)
                    or "Launch("
                    + slot
                    + ", "
                    + a[1]
                    + ", "
                    + a[2]
                    + ", "
                    + a[4]
                    + ", "
                    + function
                    + ")"
                )
            elif op == "flagos_source_native::launch":
                slot = f"kernel_slots.at({a[0]})" if slot_tables else a[0]
                expr = (
                    packed_call(node, slot, a[3], "0")
                    or "Launch(" + ", ".join([slot] + a[1:]) + ", 0)"
                )
            elif op == "flagos_script_host::deterministic":
                expr = "at::globalContext().deterministicAlgorithms()"
            elif op.startswith("flagos_script_host::global_"):
                expr = convert(outs[0], "module_globals.at(" + a[0] + ")")
            elif op == "flagos_script_host::data_ptr":
                expr = (
                    "int64_t(reinterpret_cast<uintptr_t>("
                    + a[0]
                    + ".const_data_ptr()))"
                )
            elif op == "flagos_script_host::descriptor":
                expr = "MakeDescriptor(" + ", ".join(a) + ")"
            elif op == "aten::is_floating_point":
                expr = a[0] + ".is_floating_point()"
            elif op == "flagos_source_native::tuple_value":
                expr = "TupleValue(Box(" + a[0] + "))"
            elif op == "flagos_source_native::broadcast_shapes":
                expr = "BroadcastShapes(" + a[0] + ")"
            elif op == "flagos_source_native::is_alias":
                expr = f"{a[0]}.is_alias_of({a[1]})"
            elif op == "flagos_source_native::dtype_str":
                expr = "dtype_names.at(" + a[0] + ")"
            elif op == "flagos_source_native::pointwise_apply":
                slot = f"pointwise_slots.at({a[0]})" if slot_tables else a[0]
                expr = (
                    packed_call(node, slot)
                    or "ApplyPointwise(" + ", ".join([slot] + a[1:]) + ")"
                )
            elif (
                op == "flagos_source_native::pointwise_launch"
                and node in packed_pointwise_launches
            ):
                slot = f"pointwise_slots.at({a[0]})" if slot_tables else a[0]
                parameters = (
                    "{"
                    + ", ".join("Box(" + name(v) + ")" for v in ins[2].node().inputs())
                    + "}"
                )
                outputs = "{" + ", ".join(name(v) for v in ins[3].node().inputs()) + "}"
                expr = (
                    "LaunchPointwisePacked("
                    + slot
                    + ", "
                    + a[1]
                    + ", "
                    + parameters
                    + ", "
                    + outputs
                    + ")"
                )
            elif op.startswith("flagos_"):
                if slot_tables and op.startswith("flagos_source_native::foreign_"):
                    a[0] = "foreign_slots.at(" + a[0] + ")"
                if slot_tables and op == "flagos_source_native::pointwise_launch":
                    a[0] = "pointwise_slots.at(" + a[0] + ")"
                if op.startswith("flagos_source_native::foreign_"):
                    expr = packed_call(node, a[0]) or "Foreign(" + ", ".join(a) + ")"
                else:
                    expr = (
                        "Invoke("
                        + json.dumps(op)
                        + ", {"
                        + ", ".join("Box(" + v + ")" for v in a)
                        + "})"
                    )
                if outs:
                    expr = convert(outs[0], expr)
            elif op.startswith("aten::"):
                from .aten_calls import native_call

                try:
                    expr = native_call(
                        node, a, {name(v) for v in ins if readonly_ints(v)}
                    )
                except (NotImplementedError, AssertionError, ValueError):
                    import torch

                    schema = torch._C.parse_schema(node.schema())
                    expr = (
                        "Invoke("
                        + json.dumps(op)
                        + ", {"
                        + ", ".join("Box(" + v + ")" for v in a)
                        + "}, "
                        + json.dumps(schema.overload_name)
                        + ")"
                    )
                    if outs:
                        expr = convert(outs[0], expr)
            else:
                raise NotImplementedError(op)
            if len(outs) == 1:
                if outs[0].type().kind() == "ListType" and not function_name.startswith(
                    "Grid"
                ):
                    expr = kind(outs[0]) + "(" + expr + ")"
                write(f"{kind(outs[0])} {name(outs[0])} = {expr};", depth)
            elif not outs:
                write(expr + ";", depth)
            else:
                raise NotImplementedError(op + " multiple outputs")
        for target, value in zip(targets, scope.outputs()):
            write(target + " = " + name(value) + ";", depth)

    inputs = list(graph.inputs())
    if inputs[0].type().kind() == "ClassType":
        inputs = inputs[1:]
    write(
        kind(list(graph.outputs())[0])
        + " "
        + function_name
        + "("
        + ", ".join(kind(v) + " " + name(v) for v in inputs)
        + ") {",
        0,
    )
    lower(graph, 1)
    write("return " + name(list(graph.outputs())[0]) + ";", 1)
    write("}", 0)
    code = "\n".join(lines)
    if not binding:
        return code
    return (
        HEADER
        + "\n"
        + code
        + """
PYBIND11_MODULE(source_host_ir, module) {
  module.def("bind_globals", [](py::list values) {
    module_globals.clear();
    for(auto value:values) {
      if(PyBool_Check(value.ptr()))module_globals.emplace_back(py::cast<bool>(value));
      else if(PyLong_Check(value.ptr()))module_globals.emplace_back(py::cast<int64_t>(value));
      else if(PyFloat_Check(value.ptr()))module_globals.emplace_back(py::cast<double>(value));
      else if(PyUnicode_Check(value.ptr()))module_globals.emplace_back(py::cast<std::string>(value));
      else throw std::invalid_argument("unsupported initialized host global");
    }
  });
  module.def("bind_foreign", [](py::capsule capsule) {
    foreign_api=reinterpret_cast<flagos::native_host::Foreign>(PyCapsule_GetPointer(capsule.ptr(),"flagos.native_host.foreign.v1"));
    if(!foreign_api)throw py::error_already_set();
  });
  module.def("bind_pointwise_launch", [](py::capsule capsule) {
    pointwise_launch_api=reinterpret_cast<flagos::native_host::PointwiseLaunch>(PyCapsule_GetPointer(capsule.ptr(),"flagos.native_host.pointwise_launch.v1"));
    if(!pointwise_launch_api)throw py::error_already_set();
  });
  module.def("bind_pointwise_at", [](py::capsule capsule) {
    pointwise_at_api=reinterpret_cast<flagos::native_host::PointwiseAt>(PyCapsule_GetPointer(capsule.ptr(),"flagos.native_host.pointwise_at.v1"));
    if(!pointwise_at_api)throw py::error_already_set();
  });
  module.def("bind_api", [](py::capsule capsule,py::capsule pointwise) {
    launch_api = reinterpret_cast<LaunchApi>(PyCapsule_GetPointer(capsule.ptr(), "flagos.native_host.launch.v2"));
    if(!launch_api) throw py::error_already_set();
    pointwise_api=reinterpret_cast<PointwiseApi>(PyCapsule_GetPointer(pointwise.ptr(),"flagos.native_host.pointwise.v2"));
    if(!pointwise_api) throw py::error_already_set();
  });
  dtype_names.resize(static_cast<size_t>(at::ScalarType::NumOptions));
  for(size_t i=0;i<dtype_names.size();++i) if(i!=static_cast<size_t>(at::ScalarType::Undefined)) {
    auto* dtype=torch::getTHPDtype(static_cast<at::ScalarType>(i));
    if(dtype) dtype_names[i]=py::str(reinterpret_cast<PyObject*>(dtype));
  }
  module.def("forward", &Forward, py::call_guard<py::gil_scoped_release>());
}
"""
    )


HEADER = (Path(__file__).resolve().parents[1] / "csrc/host_runtime.h").read_text()
