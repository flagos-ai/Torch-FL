#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extract authoritative name map from existing csrc/aten/*.h headers.

Parses:
  - csrc/aten/xxx.h: `using XxxFn = ...` typedef + `DECLARE_DISPATCHER(XxxFn, xxx_dispatcher)`
  - csrc/aten/xxx.cc: `ADD_IMPL_TO_DISPATCHER(XxxFn, xxx_dispatcher, "op.name")`

Builds: op_name → {fn_type, dispatcher_name, signature_raw}

This is the single source of truth for symbol naming — codegen will use this
to ensure generated code matches existing ABI exactly.
"""

import re
from pathlib import Path
from typing import Dict, Optional


def parse_typedef_and_dispatcher(header_path: Path) -> Optional[tuple]:
    """
    Parse a single header for:
      using FooFn = RetType (*)(Args...);
      DECLARE_DISPATCHER(FooFn, foo_dispatcher)

    Returns: (fn_type_name, dispatcher_name, signature_raw) or None
    """
    text = header_path.read_text()

    # Match: using XxxFn = ...;
    typedef_match = re.search(
        r"using\s+(\w+Fn)\s*=\s*(.+?)\s*\(\*\)\s*\(([^)]*(?:\([^)]*\)[^)]*)*)\)\s*;",
        text,
        re.DOTALL,
    )
    if not typedef_match:
        return None

    fn_type = typedef_match.group(1)
    ret_type = typedef_match.group(2).strip()
    args_raw = typedef_match.group(3).strip()

    # Match: DECLARE_DISPATCHER(XxxFn, xxx_dispatcher)
    decl_match = re.search(
        rf"DECLARE_DISPATCHER\s*\(\s*{re.escape(fn_type)}\s*,\s*(\w+)\s*\)", text
    )
    if not decl_match:
        return None

    dispatcher_name = decl_match.group(1)

    # Reconstruct full signature
    signature = f"{ret_type} (*)({args_raw})"

    return (fn_type, dispatcher_name, signature)


def parse_add_impl_op_name(
    cc_path: Path, fn_type: str, dispatcher_name: str
) -> Optional[str]:
    """
    Parse csrc/aten/xxx.cc for:
      ADD_IMPL_TO_DISPATCHER(FooFn, foo_dispatcher, "op.name")

    Returns: "op.name" or None
    """
    if not cc_path.exists():
        return None

    text = cc_path.read_text()

    # Match: ADD_IMPL_TO_DISPATCHER(FooFn, foo_dispatcher, "op.name")
    pattern = rf'ADD_IMPL_TO_DISPATCHER\s*\(\s*{re.escape(fn_type)}\s*,\s*{re.escape(dispatcher_name)}\s*,\s*"([^"]+)"\s*\)'
    match = re.search(pattern, text)

    if match:
        return match.group(1)

    return None


def extract_all_name_maps(aten_dir: Path) -> Dict[str, dict]:
    """
    Scan all csrc/aten/*.h headers and build name map.

    Returns: {
        "op.name": {
            "fn_type": "AbsFn",
            "dispatcher_name": "abs_dispatcher",
            "signature": "at::Tensor (*)(const at::Tensor&)",
            "header_file": "abs.h",
            "cc_file": "abs.cc"
        },
        ...
    }
    """
    name_map = {}

    # Scan all .h files in csrc/aten/ (exclude subdirs)
    for header in sorted(aten_dir.glob("*.h")):
        result = parse_typedef_and_dispatcher(header)
        if not result:
            continue

        fn_type, dispatcher_name, signature = result

        # Try to find corresponding .cc and extract op_name
        cc_path = header.with_suffix(".cc")
        op_name = parse_add_impl_op_name(cc_path, fn_type, dispatcher_name)

        if not op_name:
            # Some dispatchers handle multiple ops (mm handles both "mm" and "mm.out")
            # For now, just record what we have
            continue

        name_map[op_name] = {
            "fn_type": fn_type,
            "dispatcher_name": dispatcher_name,
            "signature": signature,
            "header_file": header.name,
            "cc_file": cc_path.name if cc_path.exists() else None,
        }

    return name_map


def main():
    repo_root = Path(__file__).parent.parent.parent
    aten_dir = repo_root / "csrc/aten"

    print("Extracting name map from csrc/aten/*.h headers...")
    name_map = extract_all_name_maps(aten_dir)

    print(f"\nExtracted {len(name_map)} op → symbol mappings:\n")

    # Print in sorted order
    for op_name in sorted(name_map.keys()):
        info = name_map[op_name]
        print(f"  {op_name:30s} → {info['fn_type']:25s} {info['dispatcher_name']}")

    print(f"\n✅ Name map extracted: {len(name_map)} ops")

    # Save to JSON for codegen to consume
    import json

    out_path = repo_root / "csrc/aten/generated/name_map.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(name_map, indent=2))
    print(f"   Saved to: {out_path}")


if __name__ == "__main__":
    main()
