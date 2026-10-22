#!/usr/bin/env python3
"""
Track and manage MUSA FlagGems ops that fail CI tests and need fallback to mudnn.

Usage:
  # After CI run, record failing ops
  python3 scripts/tools/record_musa_flaggems_failures.py add add.Tensor floor_divide embedding

  # Regenerate configs with failures moved to NATIVE_TRITON_GAPS
  python3 scripts/codegen/gen_vendor_confs.py

  # View current failures
  python3 scripts/tools/record_musa_flaggems_failures.py list
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
GEN_CONFS = REPO_ROOT / "scripts" / "codegen" / "gen_vendor_confs.py"


def get_current_gaps():
    """Parse NATIVE_TRITON_GAPS['musa'] from gen_vendor_confs.py."""
    content = GEN_CONFS.read_text()

    # Find NATIVE_TRITON_GAPS section
    start = content.find("NATIVE_TRITON_GAPS = {")
    if start == -1:
        return set()

    # Find musa section
    musa_start = content.find('"musa":', start)
    if musa_start == -1:
        return set()

    # Find the closing brace for musa's set
    brace_start = content.find("{", musa_start)
    brace_end = content.find("}", brace_start)

    if brace_start == -1 or brace_end == -1:
        return set()

    # Extract ops between braces
    ops_text = content[brace_start + 1 : brace_end]
    ops = set()
    for line in ops_text.split("\n"):
        line = line.strip()
        if line.startswith('"') and line.endswith(('",', '"')):
            op = line.strip('",').strip()
            if op:
                ops.add(op)

    return ops


def update_gaps(new_gaps):
    """Update NATIVE_TRITON_GAPS['musa'] in gen_vendor_confs.py."""
    content = GEN_CONFS.read_text()

    # Find NATIVE_TRITON_GAPS
    start_marker = "NATIVE_TRITON_GAPS = {"
    start = content.find(start_marker)
    if start == -1:
        print("Error: NATIVE_TRITON_GAPS not found")
        return False

    # Find musa section or insert point
    musa_marker = '"musa":'
    musa_start = content.find(musa_marker, start)

    # Generate new musa section
    if new_gaps:
        gaps_lines = ['    "musa": {']
        for op in sorted(new_gaps):
            gaps_lines.append(f'        "{op}",')
        gaps_lines.append("    },")
        new_musa_section = "\n".join(gaps_lines)
    else:
        new_musa_section = '    "musa": set(),'

    if musa_start == -1:
        # musa not in NATIVE_TRITON_GAPS yet, add it after ascend
        ascend_end = content.find("},", start + len(start_marker))
        if ascend_end == -1:
            print("Error: Could not find insertion point")
            return False
        insert_point = ascend_end + 2
        new_content = (
            content[:insert_point] + "\n" + new_musa_section + content[insert_point:]
        )
    else:
        # Replace existing musa section
        musa_set_start = content.find("{", musa_start)
        musa_set_end = content.find("},", musa_set_start)
        if musa_set_start == -1 or musa_set_end == -1:
            print("Error: Could not parse musa section")
            return False

        # Keep the '"musa":' prefix
        new_content = (
            content[: musa_start + len(musa_marker)]
            + "\n"
            + new_musa_section[len('    "musa":') :]
            + content[musa_set_end + 2 :]
        )

    GEN_CONFS.write_text(new_content)
    return True


def cmd_add(ops):
    """Add ops to NATIVE_TRITON_GAPS['musa']."""
    current = get_current_gaps()
    new_gaps = current | set(ops)

    if new_gaps == current:
        print(f"No new ops to add (already have {len(current)} gaps)")
        return

    if update_gaps(new_gaps):
        print(f"Added {len(new_gaps - current)} ops to NATIVE_TRITON_GAPS['musa']")
        print(f"Total gaps: {len(new_gaps)}")
        print("\nNext steps:")
        print("  1. python3 scripts/codegen/gen_vendor_confs.py")
        print("  2. Rebuild and test")
    else:
        print("Failed to update gen_vendor_confs.py")
        sys.exit(1)


def cmd_remove(ops):
    """Remove ops from NATIVE_TRITON_GAPS['musa']."""
    current = get_current_gaps()
    new_gaps = current - set(ops)

    if new_gaps == current:
        print(f"Ops not found in gaps (still have {len(current)} gaps)")
        return

    if update_gaps(new_gaps):
        print(f"Removed {len(current - new_gaps)} ops from NATIVE_TRITON_GAPS['musa']")
        print(f"Total gaps: {len(new_gaps)}")
    else:
        print("Failed to update gen_vendor_confs.py")
        sys.exit(1)


def cmd_list():
    """List current NATIVE_TRITON_GAPS['musa']."""
    gaps = get_current_gaps()
    if not gaps:
        print("NATIVE_TRITON_GAPS['musa'] is empty")
        print("All 482 FlagGems ops are enabled on MUSA")
    else:
        print(f"NATIVE_TRITON_GAPS['musa']: {len(gaps)} ops")
        for op in sorted(gaps):
            print(f"  {op}")
        print(f"\nFlagGems ops enabled: {482 - len(gaps)}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "add":
        if len(sys.argv) < 3:
            print("Usage: record_musa_flaggems_failures.py add OP1 [OP2 ...]")
            sys.exit(1)
        cmd_add(sys.argv[2:])

    elif cmd == "remove":
        if len(sys.argv) < 3:
            print("Usage: record_musa_flaggems_failures.py remove OP1 [OP2 ...]")
            sys.exit(1)
        cmd_remove(sys.argv[2:])

    elif cmd == "list":
        cmd_list()

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
