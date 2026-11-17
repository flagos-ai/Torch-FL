# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate typed call slots from the existing FlagGems argument mapping."""

import hashlib
import json
import re


def value_type(cpp_type: str) -> str:
    return re.sub(r"\s+", " ", cpp_type.replace("const ", "").replace("&", "")).strip()


def wrap_call(op, host, symbol, call, arguments, outputs=1, preparation=""):
    """Return a static ABI slot and a lazy-fallback expression for one call.

    arguments are (C++ type, expression, Python keyword or None). C++ types come
    from torchgen's actual faithful signature, after existing call adaptation.
    """
    result_type = "at::Tensor" if outputs == 1 else "std::vector<at::Tensor>"
    record = {
        "op": "aten::" + op,
        "host": host,
        "result_type": result_type,
        "outputs": outputs,
        "arguments": [
            {"type": value_type(t), "keyword": keyword} for t, _, keyword in arguments
        ],
    }
    record["abi"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()
    declaration = (
        f"static TypedSourceSlot {symbol}("
        + ", ".join(json.dumps(record[k]) for k in ("op", "host", "abi"))
        + ", "
        + json.dumps(json.dumps(record, sort_keys=True))
        + ");\n"
    )
    parameters = "".join(
        ", static_cast<const " + value_type(t) + "&>(" + expr + ")"
        for t, expr, _ in arguments
    )
    expression = f"{symbol}.Call<{result_type}>([&]() {{ return {call}; }}{parameters})"
    if preparation:
        expression = (
            f"([&]() {{ if (!{symbol}.Available()) return {call}; "
            + preparation
            + " return "
            + expression
            + "; }())"
        )
    return declaration, expression
