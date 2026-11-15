# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Account for every registry row without importing Torch or loading a device."""

import ast
from collections import Counter
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import textwrap


def annotation(t, name: str, scalar: str) -> str:
    kind = t.kind()
    if name == "dtype":
        return "Optional[torch.dtype]" if kind == "OptionalType" else "torch.dtype"
    if kind == "OptionalType":
        return "Optional[" + annotation(t.getElementType(), name, scalar) + "]"
    if kind == "ListType":
        return "List[" + annotation(t.getElementType(), name, scalar) + "]"
    return {
        "TensorType": "torch.Tensor",
        "IntType": "int",
        "SymIntType": "int",
        "FloatType": "float",
        "BoolType": "bool",
        "StringType": "str",
        "DeviceObjType": "torch.device",
        "NumberType": scalar,
    }[kind]


def symbol(function) -> str:
    return (
        getattr(function, "__module__", type(function).__module__)
        + "."
        + getattr(function, "__qualname__", type(function).__qualname__)
    )


def return_annotation(returns) -> str:
    if not returns:
        return "None"
    fields = [annotation(value.type, "", "float") for value in returns]
    return fields[0] if len(fields) == 1 else "Tuple[" + ", ".join(fields) + "]"


def discover_variants(registry, original_ops: dict, names: set, schema_lookup) -> tuple:
    """Return deduplicated compiler inputs and one record for each registry row.

    Selection describes the host bundle, not device support. Schema aliases keep
    separate rows even when they share the same generated host signature.
    """
    seen = set()
    variants = []
    entries = []
    for index, (op, registered, *_) in enumerate(registry):
        name = getattr(registered, "__name__", None)
        function = original_ops.get(name)
        entry = {
            "registry_index": index,
            "operator": "aten::" + op,
            "name": name,
            "registered_host": symbol(registered),
            "resolved_host": symbol(function) if function is not None else None,
            "status": "excluded",
            "reason": None,
            "signatures": [],
        }
        entries.append(entry)
        if name not in names:
            entry.update(status="not_selected", reason="not_in_host_bundle")
            continue
        if function is None:
            entry["reason"] = "missing_host"
            continue
        base, _, overload = op.partition(".")
        try:
            schemas = schema_lookup("aten::" + base)
        except RuntimeError:
            entry["reason"] = "schema_lookup_failed"
            continue
        schema = next((s for s in schemas if s.overload_name == overload), None)
        if schema is None:
            entry["reason"] = "missing_schema"
            continue
        entry["schema"] = str(schema)
        try:
            result_annotation = return_annotation(schema.returns)
        except KeyError:
            entry["reason"] = "unsupported_return"
            continue
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            entry["reason"] = "signature_unavailable"
            continue
        parameters = list(signature.parameters.values())
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters):
            entry["reason"] = "variadic_parameters"
            continue
        try:
            source = inspect.getsource(function)
            syntax = ast.parse(textwrap.dedent(source))
        except (OSError, TypeError, SyntaxError, IndentationError):
            entry["reason"] = "source_unavailable"
            continue
        entry["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        flexible = sorted(
            {
                call.args[0].id
                for call in ast.walk(syntax)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "isinstance"
                and call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id in signature.parameters
            }
        )
        for scalar, scalarized, number in itertools.product(
            ("bool", "int", "float"), [None] + flexible, ("bool", "int", "float")
        ):
            fields = []
            public = []
            position = 0
            reason = None
            required_none = False
            for p in parameters:
                if p.kind == p.KEYWORD_ONLY:
                    argument = next(
                        (a for a in schema.arguments if a.name == p.name), None
                    )
                else:
                    argument = (
                        schema.arguments[position]
                        if position < len(schema.arguments)
                        else None
                    )
                    position += 1
                if argument is None and p.name == "out" and p.default is None:
                    kind = "Optional[torch.Tensor]"
                elif argument is None:
                    reason = "unmapped_parameter"
                    break
                else:
                    try:
                        kind = annotation(argument.type, p.name, number)
                    except KeyError:
                        reason = "unsupported_parameter_type"
                        break
                default = p.default
                if p.name == scalarized and kind == "torch.Tensor":
                    kind = scalar
                if default is None and not kind.startswith("Optional["):
                    # The schema may require a value even when the Python
                    # helper has a None default it cannot execute with. Native
                    # callers pass every parameter; defaults remain in public
                    # for admission of the legacy Python-style call surface.
                    required_none = True
                if default is not p.empty and not (
                    default is None or type(default) in (bool, int, float, str)
                ):
                    reason = "unsupported_default"
                    break
                text = p.name + ": " + kind
                if default is not p.empty:
                    text += "=" + repr(default)
                if p.kind == p.KEYWORD_ONLY and not any(item == "*" for item in fields):
                    fields.append("*")
                fields.append(text)
                public.append(
                    {
                        "name": p.name,
                        "type": kind,
                        "keyword": p.kind == p.KEYWORD_ONLY,
                        "has_default": default is not p.empty,
                        "default": None if default is p.empty else default,
                    }
                )
            if reason:
                entry["reason"] = reason
                continue
            if required_none:
                fields = []
                for parameter in public:
                    if parameter["keyword"] and "*" not in fields:
                        fields.append("*")
                    fields.append(parameter["name"] + ": " + parameter["type"])
            header = (
                "def f(" + ", ".join(fields) + ") -> " + result_annotation + ": pass"
            )
            if header not in entry["signatures"]:
                entry["signatures"].append(header)
            key = (name, header)
            if key not in seen:
                seen.add(key)
                variants.append((name, function, header, public))
        if entry["signatures"]:
            entry.update(status="pending", reason=None)
    return variants, entries


def coverage_report(
    entries: list,
    compiled: list,
    fallback: list,
    *,
    cohort: dict,
    complete: bool = False,
    artifact_built: bool = False,
) -> dict:
    """Reconcile signature outcomes with registry rows; never infer execution."""
    expected = {(row["name"], s) for row in entries for s in row["signatures"]}
    outcomes = {}
    for status, records in (("lowered", compiled), ("compiler_fallback", fallback)):
        for record in records:
            key = (record["name"], record["signature"])
            if key not in expected or key in outcomes:
                raise ValueError(
                    f"unexpected or duplicate host signature result: {key}"
                )
            outcomes[key] = status
    if complete and expected != outcomes.keys():
        raise ValueError("missing host signature results in completed coverage report")
    if artifact_built and not complete:
        raise ValueError("a built artifact requires complete lowering results")
    rows = []
    for entry in entries:
        row = dict(entry)
        counts = Counter(
            outcomes.get((row["name"], s), "pending") for s in row["signatures"]
        )
        row["signature_outcomes"] = dict(sorted(counts.items()))
        if counts:
            if counts["pending"]:
                row["status"] = "pending"
            elif counts["lowered"] and counts["compiler_fallback"]:
                row.update(status="partial", reason="compile_error")
            elif counts["lowered"]:
                row["status"] = "lowered"
            else:
                row.update(status="compiler_fallback", reason="compile_error")
        rows.append(row)
    summary = {
        "registry_entries": len(rows),
        "unique_operators": len({row["operator"] for row in rows}),
        "entries_by_status": dict(sorted(Counter(r["status"] for r in rows).items())),
        "entries_by_reason": dict(
            sorted(Counter(r["reason"] for r in rows if r["reason"]).items())
        ),
        "unique_signatures": len(expected),
        "lowered_signatures": len(compiled),
        "fallback_signatures": len(fallback),
        "pending_signatures": len(expected) - len(outcomes),
    }
    return {
        "schema_version": 1,
        "scope": "registry_host_lowering",
        "lowering_complete": complete,
        "native_artifact_built": artifact_built,
        "cohort": cohort,
        "summary": summary,
        "entries": rows,
    }


def write_coverage(
    directory: Path, entries: list, compiled: list, fallback: list, **kwargs
) -> dict:
    """Persist build diagnostics, including an empty or entirely excluded bundle."""
    report = coverage_report(entries, compiled, fallback, **kwargs)
    (directory / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / "diagnostics.json").write_text(
        json.dumps({"compiled": compiled, "fallback": fallback}, indent=2) + "\n"
    )
    return report
